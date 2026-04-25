"""
Depth Anything 3 (DA3) inference wrapper for RealSense color input.

Supports monocular metric depth estimation from RGB images only — no depth prompt needed.

Available models (HuggingFace or local directory):
  depth-anything/DA3METRIC-LARGE          -- monocular metric depth  (Apache 2.0, recommended)
  depth-anything/DA3MONO-LARGE            -- monocular relative depth (Apache 2.0)
  depth-anything/DA3NESTED-GIANT-LARGE-1.1 -- any-view, already metric, supports intrinsics (CC BY-NC 4.0)

Metric depth:
  DA3METRIC-LARGE:          metric_depth = focal * net_output / 300.  (requires --focal)
  DA3NESTED-GIANT-LARGE-*:  output already in metres; pass intrinsics for better accuracy.

Benchmark (synthetic):
    python -m robot_motion_interface.utils.da3_utils \
        --model depth-anything/DA3METRIC-LARGE

Run on recorded realsense data (METRIC-LARGE):
    # Mean 28.91 ms ( 34.6 Hz)
    python -m robot_motion_interface.utils.da3_utils \
        --model depth-anything/DA3METRIC-LARGE \
        --frames_dir models/data_examples/realsense/rs_record_distant_h2g2h \
        --focal 612.5

Run with NESTED model + intrinsics:
    # Mean 90ms (11.0hz)
    python -m robot_motion_interface.utils.da3_utils \
        --model depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
        --frames_dir models/data_examples/realsense/rs_record_distant_h2g2h \
        --fx 612.5 --fy 612.5 --cx 320.0 --cy 240.0

Intrinsics are read from rl_policy_node_config.yaml → realsense.color_intrinsics (fx, fy, cx, cy).
Models are downloaded to models/da3/ (HuggingFace cache format).
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# DA3 source priority:
# 1) DA3_SRC env var
# 2) dep/Depth-Anything-3-HAND/src
# 3) dep/Depth-Anything-3/src
_UTILS_FILE = Path(__file__).resolve()
_REPO_ROOT = _UTILS_FILE.parents[5]
_da3_src_env = os.environ.get("DA3_SRC", "").strip()
if _da3_src_env:
    _DA3_SRC = Path(_da3_src_env).expanduser().resolve()
else:
    _da3_candidates = [
        _REPO_ROOT / "dep" / "Depth-Anything-3-HAND" / "src",
        _REPO_ROOT / "dep" / "Depth-Anything-3" / "src",
    ]
    _DA3_SRC = next((p for p in _da3_candidates if p.exists()), _da3_candidates[0])

if not _DA3_SRC.exists():
    raise FileNotFoundError(
        f"Cannot find DA3 source at {_DA3_SRC}. "
        "Set DA3_SRC to a valid Depth-Anything-3 src directory."
    )

if str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

# moviepy.editor was removed in moviepy 2.x; mock it so DA3's gs.py import doesn't fail
import types as _types
if "moviepy.editor" not in sys.modules:
    sys.modules["moviepy.editor"] = _types.ModuleType("moviepy.editor")

from depth_anything_3.api import DepthAnything3  # noqa: E402

DEFAULT_DA3_MODEL = "depth-anything/DA3-BASE"


def _normalize_da3_model(model: str | None) -> str:
    if model is None:
        return DEFAULT_DA3_MODEL
    text = str(model).strip()
    if text.lower() in {"", "none", "null", "~"}:
        return DEFAULT_DA3_MODEL
    return text


class DA3Inference:
    """
    Wraps DA3 model loading and monocular inference.

    Usage:
        da3 = DA3Inference(model="depth-anything/DA3METRIC-LARGE", focal=612.5)
        metric_depth = da3.infer(bgr_uint8)  # → [H, W] float32 metres, numpy
    """

    def __init__(self,
                 model: str = DEFAULT_DA3_MODEL,
                 focal: float | None = None,
                 fx: float | None = None,
                 fy: float | None = None,
                 cx: float | None = None,
                 cy: float | None = None,
                 device: str = "cuda",
                 process_res: int = 504):
        """
        Args:
            model:       HuggingFace repo name or local directory path.
            focal:       Camera focal length in pixels (average of fx, fy).
                         Used for DA3METRIC-LARGE metric scaling: depth = focal * net / 300.
            fx, fy, cx, cy: Full camera intrinsics. When provided, passed to DA3NESTED-*
                         for pose-conditioned depth (better accuracy). Ignored by other models.
            device:      torch device string.
            process_res: Internal processing resolution (default 504).
        """
        model = _normalize_da3_model(model)
        self.device = torch.device(device)
        self.focal = focal
        self.process_res = process_res
        self._model_name = model.lower()

        # Build (1, 3, 3) intrinsics matrix if full intrinsics provided
        self._intrinsics: np.ndarray | None = None
        self._intrinsics_torch: torch.Tensor | None = None
        if fx is not None and fy is not None and cx is not None and cy is not None:
            self._intrinsics = np.array([[fx, 0.0, cx],
                                         [0.0, fy, cy],
                                         [0.0, 0.0, 1.0]], dtype=np.float32)[None]  # (1, 3, 3)
            self._intrinsics_torch = torch.from_numpy(self._intrinsics).to(
                device=self.device, dtype=torch.float32
            )

        cache_dir = _REPO_ROOT / "models" / "da3"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = DepthAnything3.from_pretrained(model, cache_dir=str(cache_dir))
        self.model = self.model.to(device=self.device).eval()

    @torch.inference_mode()
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """
        Run monocular depth inference on a single BGR color frame.

        Args:
            bgr: HxWx3 uint8 BGR color frame (as returned by cv2.imread).

        Returns:
            [H', W'] float32 numpy array of depth values.
            - DA3METRIC-LARGE:        metric metres (if focal is provided),
                                      otherwise raw network output.
            - DA3MONO-LARGE:          relative depth (arbitrary scale).
            - DA3NESTED-GIANT-LARGE:  metric metres (scale is built-in).
            H'/W' may differ from input due to internal resize.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        prediction = self.model.inference(
            [pil_image],
            intrinsics=self._intrinsics,
            process_res=self.process_res,
        )

        depth = prediction.depth[0]  # [H', W'] float32

        # Apply metric scaling for DA3METRIC-LARGE
        if "metric" in self._model_name and self.focal is not None:
            depth = self.focal * depth / 300.0

        return depth

    @torch.inference_mode()
    def infer_torch_batched(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Run batched RGB depth inference on GPU with torch tensors.

        Args:
            rgb: Tensor with shape (N, H, W, 3) or (N, 3, H, W), RGB order.

        Returns:
            Depth tensor with shape (N, H', W') on the same device as input.
        """
        if not isinstance(rgb, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(rgb)}")
        if rgb.dim() != 4:
            raise ValueError(
                "Expected RGB tensor with shape (N,H,W,3) or (N,3,H,W), "
                f"got {tuple(rgb.shape)}"
            )
        if not hasattr(self.model, "inference_torch"):
            raise RuntimeError(
                "Loaded DA3 model does not provide inference_torch. "
                "Ensure Depth-Anything-3-HAND is selected."
            )

        intrinsics_t: torch.Tensor | None = None
        if self._intrinsics_torch is not None:
            if self._intrinsics_torch.device != rgb.device:
                self._intrinsics_torch = self._intrinsics_torch.to(
                    device=rgb.device, dtype=torch.float32
                )
            if self._intrinsics_torch.shape[0] == 1:
                intrinsics_t = self._intrinsics_torch.expand(rgb.shape[0], -1, -1)
            elif self._intrinsics_torch.shape[0] == rgb.shape[0]:
                intrinsics_t = self._intrinsics_torch
            else:
                intrinsics_t = self._intrinsics_torch.repeat(rgb.shape[0], 1, 1)

        outputs = self.model.inference_torch(
            image=rgb,
            intrinsics=intrinsics_t,
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
        )
        depth = outputs["depth"]
        if "metric" in self._model_name and self.focal is not None:
            depth = self.focal * depth / 300.0
        return depth


# ── Benchmark / folder-mode entry point ───────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="DA3 benchmark / folder inference")
    parser.add_argument("--model",      type=str,
                        default=DEFAULT_DA3_MODEL,
                        help="HuggingFace repo name or local model directory")
    parser.add_argument("--focal",      type=float, default=None,
                        help="Camera focal length in pixels (average of fx, fy). "
                             "Required for DA3METRIC-LARGE to output metric depth.")
    parser.add_argument("--fx",         type=float, default=None, help="Camera intrinsic fx (pixels)")
    parser.add_argument("--fy",         type=float, default=None, help="Camera intrinsic fy (pixels)")
    parser.add_argument("--cx",         type=float, default=None, help="Camera intrinsic cx (pixels)")
    parser.add_argument("--cy",         type=float, default=None, help="Camera intrinsic cy (pixels)")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--width",      type=int, default=640)
    parser.add_argument("--height",     type=int, default=480)
    parser.add_argument("--warmup",     type=int, default=1)
    parser.add_argument("--iters",      type=int, default=10,
                        help="Timed iterations (synthetic mode only)")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Root of a realsense_record directory containing "
                             "color/ sub-folder. Activates folder mode.")
    parser.add_argument("--out_dir",    type=str, default=None,
                        help="Output directory for folder mode (default: <frames_dir>/da3)")
    parser.add_argument("--video_fps",  type=float, default=30.0,
                        help="FPS of the output video (folder mode, default: 30)")
    args = parser.parse_args()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return _REPO_ROOT / path if not path.is_absolute() else path

    # ── Folder mode setup ────────────────────────────────────────────────────
    color_paths: list[Path] = []
    record_dir = Path()
    out_dir = Path()

    if args.frames_dir is not None:
        record_dir = _resolve(args.frames_dir)
        color_dir  = record_dir / "color"
        if not color_dir.is_dir():
            raise FileNotFoundError(f"Expected color/ inside {record_dir}")
        color_paths = sorted(p for p in color_dir.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not color_paths:
            raise FileNotFoundError(f"No color frames in {color_dir}")
        out_dir = _resolve(args.out_dir) if args.out_dir else record_dir / "da3"
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
    print("  Depth Anything 3 Inference")
    print("=" * 60)
    print(f"  GPU        : {gpu_info}")
    print(f"  model      : {args.model}")
    print(f"  focal      : {args.focal}")
    print(f"  device     : {args.device}")
    if mode == "folder":
        print(f"  frames_dir : {record_dir}  ({len(color_paths)} frames)")
        print(f"  out_dir    : {out_dir}")
    else:
        print(f"  frame size : {args.width}x{args.height}")
        print(f"  warm-up    : {args.warmup}  |  timed iters: {args.iters}")
    print("=" * 60)

    t_load = time.time()
    da3 = DA3Inference(
        model=args.model,
        focal=args.focal,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        device=args.device,
        process_res=args.process_res,
    )
    print(f"Model loaded in {time.time() - t_load:.2f}s\n")

    # ── Warmup ───────────────────────────────────────────────────────────────
    if mode == "folder":
        warmup_bgr = cv2.imread(str(color_paths[0]))
        if warmup_bgr is None:
            raise FileNotFoundError(f"Cannot read {color_paths[0]}")
    else:
        rng = np.random.default_rng(42)
        warmup_bgr = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)

    print(f"Running {args.warmup} warm-up frame(s) ...")
    for _ in range(args.warmup):
        da3.infer(warmup_bgr)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    # ── Inference loop ───────────────────────────────────────────────────────
    latencies: list[float] = []

    if mode == "folder":
        for i, color_p in enumerate(color_paths):
            bgr = cv2.imread(str(color_p))
            if bgr is None:
                print(f"  [skip] cannot read {color_p.name}")
                continue

            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            depth = da3.infer(bgr)
            if args.device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)

            # Save 16-bit PNG + inferno colormap
            stem = color_p.stem
            d_min, d_max = depth.min(), depth.max()
            d_norm = ((depth - d_min) / max(d_max - d_min, 1e-6) * 65535).astype(np.uint16)
            cv2.imwrite(str(out_dir / (stem + "_depth16.png")), d_norm)

            d_8 = ((depth - d_min) / max(d_max - d_min, 1e-6) * 255).astype(np.uint8)
            vis = cv2.applyColorMap(d_8, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(out_dir / (stem + ".jpg")), vis)

            print(f"  frame {i+1:>4d}/{len(color_paths)}: {ms:>7.2f} ms  "
                  f"depth [{d_min:.3f}, {d_max:.3f}]")

    else:
        bgr = warmup_bgr
        for i in range(args.iters):
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            da3.infer(bgr)
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
        import subprocess
        video_path = out_dir.parent / (out_dir.name + ".mp4")
        vis_paths  = sorted(p for p in out_dir.iterdir()
                            if p.suffix.lower() == ".jpg")
        if vis_paths:
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(args.video_fps),
                "-pattern_type", "glob", "-i", str(out_dir / "*.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(video_path),
            ], check=True)
            print(f"\n  depth frames → {out_dir}")
            print(f"  video saved  → {video_path}  ({args.video_fps:.1f} fps)")
