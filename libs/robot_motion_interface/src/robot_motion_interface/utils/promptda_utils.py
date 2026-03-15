"""
PromptDA inference wrapper for RealSense input.
Encapsulates model loading, preprocessing, and inference.

Available checkpoints (all use encoder=vits, place in models/promptda/):
  PromptDA-s-transparent.ckpt  -- vits, fine-tuned for transparent objects (default)
  PromptDA-s.ckpt              -- vits, general
  PromptDA-l.ckpt              -- vitl, general (larger, more accurate)

Benchmark (synthetic):
    python -m robot_motion_interface.utils.promptda_utils \
        --encoder vits \
        --ckpt models/promptda/PromptDA-s-transparent.ckpt

Run on recorded realsense data:
    python -m robot_motion_interface.utils.promptda_utils \
        --ckpt models/promptda/PromptDA-s-transparent.ckpt \
        --frames_dir models/data_examples/realsense/rs_record_20260314_223830
"""

import cv2
import numpy as np
import torch

from promptda.promptda import PromptDA

class PromptDAInference:
    """
    Wraps PromptDA model loading, RealSense preprocessing, and inference.

    Usage:
        pda = PromptDAInference(ckpt_path="/path/to/model.ckpt", encoder="vits", device="cuda")
        metric_depth = pda.infer(bgr_uint8, depth_uint16)  # → [H, W] float32 metres on GPU
    """

    def __init__(self,
                 ckpt_path: str,
                 encoder: str = "vits",
                 device: str = "cuda",
                 max_size: int = 1008,
                 multiple_of: int = 14,
                 depth_scale: float = 0.001):
        """
        Args:
            ckpt_path:    path to PromptDA .ckpt checkpoint file
            encoder:      "vits" or "vitl"
            device:       torch device string
            max_size:     longest edge of color image is capped here
            multiple_of:  ViT patch size requirement (DINOv2 = 14)
            depth_scale:  metres per raw depth unit (RealSense D-series default: 0.001)
        """

        self.device = torch.device(device)
        self.max_size = max_size
        self.multiple_of = multiple_of
        self.depth_scale = depth_scale

        self.model = PromptDA.from_pretrained(
            ckpt_path, model_kwargs={"encoder": encoder}
        ).to(self.device).eval()

    def _preprocess_color(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 HxWx3 → [1, 3, H', W'] float32 on device."""
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            h = int(h * scale) // self.multiple_of * self.multiple_of
            w = int(w * scale) // self.multiple_of * self.multiple_of
        else:
            h = h // self.multiple_of * self.multiple_of
            w = w // self.multiple_of * self.multiple_of
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _preprocess_depth(self, depth_u16: np.ndarray) -> torch.Tensor:
        """uint16 HxW → [1, 1, H, W] float32 metres on device."""
        depth = depth_u16.astype(np.float32) * self.depth_scale
        return torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def infer(self, bgr: np.ndarray, depth_u16: np.ndarray) -> torch.Tensor:
        """
        Run PromptDA inference on a RealSense frame pair.

        Args:
            bgr:       HxWx3 uint8 BGR color frame
            depth_u16: HxW uint16 depth frame (aligned to color, raw units)

        Returns:
            [H', W'] float32 tensor of metric depth in metres, on self.device
            H'/W' = input H/W rounded down to nearest multiple of 14
        """
        image_tensor = self._preprocess_color(bgr)
        depth_tensor = self._preprocess_depth(depth_u16)
        return self.model.predict(image_tensor, depth_tensor)


# ── Benchmark entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    # Repo root: promptda_utils.py is 5 levels deep inside the repo
    # libs/robot_motion_interface/src/robot_motion_interface/utils/promptda_utils.py
    _REPO_ROOT = Path(__file__).resolve().parents[5]

    parser = argparse.ArgumentParser(description="PromptDA benchmark / folder inference")
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="Checkpoint path (default: models/promptda/PromptDA-s-transparent.ckpt)")
    parser.add_argument("--encoder",    type=str, default="vits", choices=["vits", "vitl"])
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--width",      type=int, default=640)
    parser.add_argument("--height",     type=int, default=480)
    parser.add_argument("--warmup",     type=int, default=1)
    parser.add_argument("--iters",      type=int, default=20,
                        help="Timed iterations (synthetic mode only)")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Root of a realsense_record directory containing "
                             "color/ and depth/ sub-folders. Activates folder mode.")
    parser.add_argument("--out_dir",    type=str, default=None,
                        help="Output directory for folder mode (default: <frames_dir>_promptda)")
    parser.add_argument("--video_fps",  type=float, default=30.0,
                        help="FPS of the output video (folder mode, default: 30)")
    args = parser.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return _REPO_ROOT / path if not path.is_absolute() else path

    ckpt_path = str(_resolve(args.ckpt) if args.ckpt else
                    _REPO_ROOT / "models" / "promptda" / "PromptDA-s-transparent.ckpt")

    # ── Folder mode setup ───────────────────────────────────────────────────
    record_dir:  Path       = Path()
    depth_dir:   Path       = Path()
    color_paths: list[Path] = []
    out_dir:     Path       = Path()

    if args.frames_dir is not None:
        record_dir = _resolve(args.frames_dir)
        color_dir  = record_dir / "color"
        depth_dir  = record_dir / "depth"
        if not color_dir.is_dir() or not depth_dir.is_dir():
            raise FileNotFoundError(
                f"Expected color/ and depth/ inside {record_dir}"
            )
        color_paths = sorted(p for p in color_dir.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not color_paths:
            raise FileNotFoundError(f"No color frames in {color_dir}")
        out_dir = _resolve(args.out_dir) if args.out_dir else \
                  record_dir / "promptda"
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "folder"
    else:
        mode = "synthetic"

    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        gpu_info = (f"{torch.cuda.get_device_name(0)}  "
                    f"({_p.total_memory // (1024**2)} MiB)  "
                    f"sm_{_p.major}{_p.minor}  CUDA {torch.version.cuda}")
    else:
        gpu_info = "N/A (CPU only)"

    print("=" * 60)
    print("  PromptDA Inference")
    print("=" * 60)
    print(f"  GPU        : {gpu_info}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  encoder    : {args.encoder}")
    print(f"  device     : {args.device}")
    if mode == "folder":
        print(f"  frames_dir : {record_dir}  ({len(color_paths)} frames)")
        print(f"  out_dir    : {out_dir}")
    else:
        print(f"  frame size : {args.width}x{args.height}")
        print(f"  warm-up    : {args.warmup}  |  timed iters: {args.iters}")
    print("=" * 60)

    t_load = time.time()
    pda = PromptDAInference(ckpt_path=ckpt_path, encoder=args.encoder, device=args.device)
    print(f"Model loaded in {time.time() - t_load:.2f}s\n")

    # ── Warmup ──────────────────────────────────────────────────────────────
    if mode == "folder":
        warmup_bgr   = cv2.imread(str(color_paths[0]))
        stem         = color_paths[0].stem
        depth_p      = depth_dir / (stem + ".png")
        warmup_depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
        if warmup_bgr is None or warmup_depth is None:
            raise FileNotFoundError(f"Cannot read warmup pair for {stem}")
    else:
        rng          = np.random.default_rng(42)
        warmup_bgr   = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
        warmup_depth = rng.integers(0, 10000, (args.height, args.width), dtype=np.uint16)

    print(f"Running {args.warmup} warm-up frame(s) ...")
    for _ in range(args.warmup):
        pda.infer(warmup_bgr, warmup_depth)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    # ── Inference loop ──────────────────────────────────────────────────────
    latencies: list[float] = []

    if mode == "folder":
        for i, color_p in enumerate(color_paths):
            stem    = color_p.stem
            depth_p = depth_dir / (stem + ".png")
            if not depth_p.exists():
                print(f"  [skip] no depth for {stem}")
                continue

            bgr       = cv2.imread(str(color_p))
            depth_u16 = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
            if bgr is None or depth_u16 is None:
                print(f"  [skip] cannot read pair for {stem}")
                continue

            # Replicate cv_node filter chain: decimation(2x) + hole_filling.
            # Decimation: area-average 2x2 → halves resolution and reduces noise.
            dh, dw = depth_u16.shape
            depth_u16 = cv2.resize(depth_u16, (dw // 2, dh // 2),
                                   interpolation=cv2.INTER_AREA)
            # Hole-filling: replace zeros with nearest non-zero neighbor (dilate trick).
            mask_zero = depth_u16 == 0
            if mask_zero.any():
                filled = cv2.dilate(depth_u16, np.ones((5, 5), np.uint8))
                depth_u16[mask_zero] = filled[mask_zero]

            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            metric_depth = pda.infer(bgr, depth_u16)  # [H', W'] float32
            if args.device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)

            # Visualise metric depth as inferno colormap (16-bit PNG + RGB vis)
            depth_np = metric_depth.squeeze().cpu().numpy()      # [H', W']
            d_min, d_max = depth_np.min(), depth_np.max()
            d_norm = ((depth_np - d_min) / max(d_max - d_min, 1e-6) * 65535).astype(np.uint16)
            cv2.imwrite(str(out_dir / (stem + "_depth16.png")), d_norm)

            d_8 = ((depth_np - d_min) / max(d_max - d_min, 1e-6) * 255).astype(np.uint8)
            vis = cv2.applyColorMap(d_8, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(out_dir / (stem + ".jpg")), vis)

            print(f"  frame {i+1:>4d}/{len(color_paths)}: {ms:>7.2f} ms  "
                  f"depth [{d_min:.3f}, {d_max:.3f}] m")

    else:
        bgr       = warmup_bgr
        depth_u16 = warmup_depth
        for i in range(args.iters):
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pda.infer(bgr, depth_u16)
            if args.device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
            print(f"  iter {i+1:>3d}: {ms:>7.2f} ms")

    mean_ms = sum(latencies) / len(latencies)
    min_ms  = min(latencies)
    max_ms  = max(latencies)
    std_ms  = (sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5

    print()
    print("=" * 60)
    print(f"  frames     : {len(latencies)}")
    print(f"  mean  : {mean_ms:>7.2f} ms   ({1000/mean_ms:>5.1f} Hz)")
    print(f"  min   : {min_ms:>7.2f} ms")
    print(f"  max   : {max_ms:>7.2f} ms")
    print(f"  std   : {std_ms:>7.2f} ms")
    print("=" * 60)

    if mode == "folder":
        video_path = out_dir.parent / (out_dir.name + ".mp4")
        vis_paths  = sorted(p for p in out_dir.iterdir()
                            if p.suffix.lower() == ".jpg")
        if vis_paths:
            sample = cv2.imread(str(vis_paths[0]))
            h_v, w_v = sample.shape[:2]
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                args.video_fps, (w_v, h_v)
            )
            for p in vis_paths:
                f = cv2.imread(str(p))
                if f is not None:
                    writer.write(f)
            writer.release()
            print(f"\n  depth frames → {out_dir}")
            print(f"  video saved  → {video_path}  ({args.video_fps:.1f} fps)")
