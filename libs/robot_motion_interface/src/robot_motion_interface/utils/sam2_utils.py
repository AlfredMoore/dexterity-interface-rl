"""
SAM 2 inference wrapper for per-frame image segmentation.
Encapsulates model loading, BGR preprocessing, and prompt-based inference.

Available SAM 2.1 variants (checkpoints in models/sam2/):
  Tiny   -- sam2.1_hiera_tiny.pt      cfg: configs/sam2.1/sam2.1_hiera_t.yaml
  Small  -- sam2.1_hiera_small.pt     cfg: configs/sam2.1/sam2.1_hiera_s.yaml
  Base+  -- sam2.1_hiera_base_plus.pt cfg: configs/sam2.1/sam2.1_hiera_b+.yaml
  Large  -- sam2.1_hiera_large.pt     cfg: configs/sam2.1/sam2.1_hiera_l.yaml

Benchmark (defaults to small if no args given):
  python -m robot_motion_interface.utils.sam2_utils
  python -m robot_motion_interface.utils.sam2_utils --ckpt models/sam2/sam2.1_hiera_tiny.pt --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml
  python -m robot_motion_interface.utils.sam2_utils --ckpt models/sam2/sam2.1_hiera_large.pt --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Inference:
    """
    Wraps SAM 2 model loading and per-frame segmentation for a live camera stream.

    Accepts BGR frames (as returned by RealSense / cv2), converts to RGB internally,
    and supports point, box, and mask prompts.

    The image encoder can be compiled via torch.compile for a significant
    per-frame speedup at the cost of a one-time warm-up on the first call.

    Usage:
        sam = SAM2Inference(
            ckpt_path="models/sam2/sam2.1_hiera_small.pt",
            model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
            device="cuda",
            compile_image_encoder=True,
        )

        masks, scores, logits = sam.infer(
            bgr_uint8,
            point_coords=np.array([[cx, cy]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
        )
        # masks:  (N_masks, H, W)  bool
        # scores: (N_masks,)       float32 IoU confidence
        # logits: (N_masks, H/4, W/4) float32  -- pass back as mask_input for refinement
    """

    def __init__(
        self,
        ckpt_path: str,
        model_cfg: str,
        device: str = "cuda",
        compile_image_encoder: bool = True,
        multimask_output: bool = True,
    ):
        """
        Args:
            ckpt_path:             Path to the SAM 2 .pt checkpoint file.
            model_cfg:             Hydra config name resolved relative to the sam2 package,
                                   e.g. "configs/sam2.1/sam2.1_hiera_s.yaml".
            device:                torch device string ("cuda" or "cpu").
            compile_image_encoder: Apply torch.compile to the image encoder for faster
                                   per-frame inference. First call triggers a warm-up.
                                   Requires CUDA; automatically disabled on CPU.
            multimask_output:      If True, SAM 2 returns 3 ranked mask candidates;
                                   if False, returns 1 mask (more suitable for iterative
                                   refinement with mask_input).
        """
        self.device = torch.device(device)
        self.multimask_output = multimask_output

        # Disable compilation on CPU -- torch.compile gains are CUDA-specific
        _compile = compile_image_encoder and self.device.type == "cuda"

        hydra_overrides: list[str] = []
        if _compile:
            hydra_overrides.append("++model.compile_image_encoder=True")

        sam_model = build_sam2(
            config_file=model_cfg,
            ckpt_path=ckpt_path,
            device=str(self.device),
            hydra_overrides_extra=hydra_overrides,
        )
        self.predictor = SAM2ImagePredictor(sam_model)

    @torch.inference_mode()
    def infer(
        self,
        bgr: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment a single BGR frame with optional prompts.

        At least one prompt (point_coords/box/mask_input) must be provided;
        if all are None the call will raise inside SAM 2.

        Args:
            bgr:          (H, W, 3) uint8 BGR frame from RealSense / cv2.
            point_coords: (N, 2) float32 prompt points in pixel (x, y) coordinates.
            point_labels: (N,)   int32  1 = foreground, 0 = background click.
            box:          (4,)   float32 bounding box [x_min, y_min, x_max, y_max].
            mask_input:   (1, H/4, W/4) float32 low-res mask logits from a previous
                          call's `logits` output (enables iterative refinement).

        Returns:
            masks:  (N_masks, H, W) bool   -- binary segmentation masks.
            scores: (N_masks,)      float32 -- IoU confidence scores, highest first.
            logits: (N_masks, H/4, W/4) float32 -- raw logits for iterative refinement.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            self.predictor.set_image(rgb)
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=self.multimask_output,
                return_logits=True,  # always return logits for potential refinement
            )

        return masks, scores, logits


# ── Benchmark entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    # Repo root: sam2_utils.py is 5 levels deep inside the repo
    # libs/robot_motion_interface/src/robot_motion_interface/utils/sam2_utils.py
    _REPO_ROOT = Path(__file__).resolve().parents[5]
    _MODELS    = _REPO_ROOT / "models" / "sam2"

    parser = argparse.ArgumentParser(description="Benchmark SAM2 throughput")
    parser.add_argument("--ckpt",      type=str, default=None,
                        help="Checkpoint path (default: models/sam2/sam2.1_hiera_small.pt)")
    parser.add_argument("--model_cfg", type=str, default=None,
                        help="Hydra config (default: configs/sam2.1/sam2.1_hiera_s.yaml)")
    parser.add_argument("--device",    type=str, default="cuda")
    parser.add_argument("--compile",   action="store_true",
                        help="Enable torch.compile on the image encoder")
    parser.add_argument("--width",     type=int, default=640)
    parser.add_argument("--height",    type=int, default=480)
    parser.add_argument("--warmup",    type=int, default=1)
    parser.add_argument("--iters",     type=int, default=20)
    args = parser.parse_args()

    # Resolve relative paths against repo root
    ckpt_path = args.ckpt if args.ckpt is not None else str(_MODELS / "sam2.1_hiera_small.pt")
    model_cfg = args.model_cfg if args.model_cfg is not None else "configs/sam2.1/sam2.1_hiera_s.yaml"
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
    print("  SAM2 Inference Frequency Benchmark")
    print("=" * 60)
    print(f"  GPU        : {gpu_info}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  model_cfg  : {model_cfg}")
    print(f"  device     : {args.device}  compile={args.compile}")
    print(f"  frame size : {args.width}x{args.height}")
    print(f"  warm-up    : {args.warmup}  |  timed iters: {args.iters}")
    print("=" * 60)

    t_load = time.time()
    sam = SAM2Inference(
        ckpt_path=ckpt_path,
        model_cfg=model_cfg,
        device=args.device,
        compile_image_encoder=args.compile,
    )
    print(f"Model loaded in {time.time() - t_load:.2f}s\n")

    rng   = np.random.default_rng(42)
    frame = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    pt_coords = np.array([[args.width // 2, args.height // 2]], dtype=np.float32)
    pt_labels = np.array([1], dtype=np.int32)

    print(f"Running {args.warmup} warm-up frame(s) ...")
    for _ in range(args.warmup):
        sam.infer(frame, point_coords=pt_coords, point_labels=pt_labels)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    latencies = []
    for i in range(args.iters):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sam.infer(frame, point_coords=pt_coords, point_labels=pt_labels)
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
