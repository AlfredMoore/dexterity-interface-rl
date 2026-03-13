"""
PromptDA inference wrapper for RealSense input.
Encapsulates model loading, preprocessing, and inference.

Available checkpoints (all use encoder=vits, place in models/promptDA/):
  PromptDA-s-transparent.ckpt  -- vits, fine-tuned for transparent objects (default)
  PromptDA-s.ckpt              -- vits, general
  PromptDA-l.ckpt              -- vitl, general (larger, more accurate)

Benchmark:
  python -m robot_motion_interface.utils.promptda_utils
  python -m robot_motion_interface.utils.promptda_utils --encoder vitl --ckpt models/promptDA/PromptDA-l.ckpt
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
    _MODELS    = _REPO_ROOT / "models" / "promptDA"

    parser = argparse.ArgumentParser(description="Benchmark PromptDA throughput")
    parser.add_argument("--ckpt",    type=str, default=None,
                        help="Checkpoint path (default: models/promptDA/PromptDA-s-transparent.ckpt)")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitl"])
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--width",   type=int, default=640)
    parser.add_argument("--height",  type=int, default=480)
    parser.add_argument("--warmup",  type=int, default=1)
    parser.add_argument("--iters",   type=int, default=20)
    args = parser.parse_args()

    ckpt_path = args.ckpt if args.ckpt is not None else str(_MODELS / "PromptDA-s-transparent.ckpt")
    if not Path(ckpt_path).is_absolute():
        ckpt_path = str(_REPO_ROOT / ckpt_path)

    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        gpu_info = (f"{torch.cuda.get_device_name(0)}  "
                    f"({_p.total_memory // (1024**2)} MiB)  "
                    f"sm_{_p.major}{_p.minor}  CUDA {torch.version.cuda}")
    else:
        gpu_info = "N/A (CPU only)"

    print("=" * 60)
    print("  PromptDA Inference Frequency Benchmark")
    print("=" * 60)
    print(f"  GPU        : {gpu_info}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  encoder    : {args.encoder}")
    print(f"  device     : {args.device}")
    print(f"  frame size : {args.width}x{args.height}")
    print(f"  warm-up    : {args.warmup}  |  timed iters: {args.iters}")
    print("=" * 60)

    t_load = time.time()
    pda = PromptDAInference(ckpt_path=ckpt_path, encoder=args.encoder, device=args.device)
    print(f"Model loaded in {time.time() - t_load:.2f}s\n")

    rng       = np.random.default_rng(42)
    bgr       = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    depth_u16 = rng.integers(0, 10000, (args.height, args.width), dtype=np.uint16)

    print(f"Running {args.warmup} warm-up frame(s) ...")
    for _ in range(args.warmup):
        pda.infer(bgr, depth_u16)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    latencies = []
    for i in range(args.iters):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pda.infer(bgr, depth_u16)
        if args.device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        print(f"  iter {i+1:>3d}: {latencies[-1]:>7.2f} ms")

    mean_ms = sum(latencies) / len(latencies)
    min_ms  = min(latencies)
    max_ms  = max(latencies)
    std_ms  = (sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5

    print()
    print("=" * 60)
    print(f"  mean  : {mean_ms:>7.2f} ms   ({1000/mean_ms:>5.1f} Hz)")
    print(f"  min   : {min_ms:>7.2f} ms")
    print(f"  max   : {max_ms:>7.2f} ms")
    print(f"  std   : {std_ms:>7.2f} ms")
    print("=" * 60)
