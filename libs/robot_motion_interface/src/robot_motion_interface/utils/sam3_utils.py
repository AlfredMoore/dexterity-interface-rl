"""
SAM 3 / EfficientSAM3 inference wrapper for per-frame semantic segmentation.

Three modes are supported (controlled by the `mode` argument):

  "sam3"      -- Original SAM3 (ViT-H vision + CLIP text, 848 M params).
                 Requires: dep/sam3-HAND installed.
                 Checkpoint: models/sam3/sam3.pt

                 Benchmark:
                   python -m robot_motion_interface.utils.sam3_utils \\
                     --mode sam3 --ckpt models/sam3/sam3.pt

  "litetext"  -- SAM3-LiteText: keeps ViT-H vision encoder but replaces
                 the 353 M-param CLIP text encoder with a MobileCLIP variant
                 (42-124 M params, 88 % smaller).  Accuracy/mask quality is
                 essentially identical to "sam3"; only text-side latency drops.
                 Requires: dep/efficientsam3 installed (must be on sys.path).

                 Available checkpoints (place in models/sam3/):
                   SAM3-LiteText-S0-16  →  efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt
                   SAM3-LiteText-S1-16  →  efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt
                   SAM3-LiteText-L-16   →  efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt

                 Benchmark commands:
                   # S0-16 (42 M text params — lightest)
                   python -m robot_motion_interface.utils.sam3_utils \\
                     --mode litetext \\
                     --ckpt models/sam3/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt \\
                     --text_encoder_type MobileCLIP-S0 \\
                     --text_encoder_context_length 16

                   # S1-16 (64 M text params)
                   python -m robot_motion_interface.utils.sam3_utils \\
                     --mode litetext \\
                     --ckpt models/sam3/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \\
                     --text_encoder_type MobileCLIP-S1 \\
                     --text_encoder_context_length 16

                   # L-16 (124 M text params — most accurate text)
                   python -m robot_motion_interface.utils.sam3_utils \\
                     --mode litetext \\
                     --ckpt models/sam3/efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt \\
                     --text_encoder_type MobileCLIP2-L \\
                     --text_encoder_context_length 16

  "efficient" -- EfficientSAM3: replaces both vision encoder (0.68-22 M params)
                 and optionally the text encoder with lightweight student models.
                 Fastest option; some accuracy trade-off vs "sam3".
                 Requires: dep/efficientsam3 installed.
                 Checkpoint examples (place in models/sam3/, fine-tuned variants recommended):
                   efficient_sam3_tinyvit_m_ft.pt      (backbone_type=tinyvit,      model_name=11m)
                   efficient_sam3_tinyvit_l_ft.pt      (backbone_type=tinyvit,      model_name=21m)
                   efficient_sam3_efficientvit_s.pt    (backbone_type=efficientvit, model_name=b0)
                   efficient_sam3_efficientvit_m_ft.pt (backbone_type=efficientvit, model_name=b1)
                   efficient_sam3_repvit_m_ft.pt       (backbone_type=repvit,       model_name=m1.1)

                 Benchmark:
                   python -m robot_motion_interface.utils.sam3_utils \\
                     --mode efficient \\
                     --ckpt models/sam3/efficient_sam3_tinyvit_m_ft.pt \\
                     --backbone_type tinyvit --model_name 11m
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, Literal, Optional


# Object IDs assigned to each semantic concept (fixed at init, used downstream)
CONCEPT_LEFT_ARM  = 1
CONCEPT_RIGHT_ARM = 2
CONCEPT_CUP       = 3

# Supported mode strings
_MODES = ("sam3", "litetext", "efficient")


def _build_model(
    mode: str,
    ckpt_path: str,
    device: str,
    compile: bool,
    backbone_type: str,
    model_name: str,
    text_encoder_type: Optional[str],
    text_encoder_context_length: int,
):
    """Construct and return the raw SAM3/EfficientSAM3 model object."""
    _compile = compile and torch.device(device).type == "cuda"

    if mode == "sam3":
        from sam3.model_builder import build_sam3_image_model
        return build_sam3_image_model(
            checkpoint_path=ckpt_path,
            device=device,
            eval_mode=True,
            load_from_HF=False,
            compile=_compile,
        )

    if mode == "litetext":
        # SAM3-LiteText: ViT-H vision + MobileCLIP text encoder.
        # build_sam3_image_model from efficientsam3 accepts text_encoder_type.
        from sam3.model_builder import build_sam3_image_model
        return build_sam3_image_model(
            checkpoint_path=ckpt_path,
            device=device,
            eval_mode=True,
            load_from_HF=False,
            compile=_compile,
            text_encoder_type=text_encoder_type,
            text_encoder_context_length=text_encoder_context_length,
        )

    if mode == "efficient":
        from sam3.model_builder import build_efficientsam3_image_model
        return build_efficientsam3_image_model(
            checkpoint_path=ckpt_path,
            device=device,
            eval_mode=True,
            load_from_HF=False,
            compile=_compile,
            backbone_type=backbone_type,
            model_name=model_name,
            text_encoder_type=text_encoder_type,
            text_encoder_context_length=text_encoder_context_length,
        )

    raise ValueError(f"Unknown mode '{mode}'. Choose from {_MODES}.")


class SAM3Inference:
    """
    Wraps SAM 3 / EfficientSAM3 image model for per-frame semantic segmentation.

    Concepts (text prompts + their object IDs) are registered once at init.
    Each call to infer() runs detection for all registered concepts on a
    single BGR frame and returns per-concept mask/box/score results.

    Usage:
        # Original SAM3 (default)
        sam3 = SAM3Inference(ckpt_path="models/sam3/sam3.pt")

        # SAM3-LiteText (MobileCLIP-S0 text encoder, context length 16)
        sam3 = SAM3Inference(
            ckpt_path="models/sam3/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt",
            mode="litetext",
            text_encoder_type="MobileCLIP-S0",
            text_encoder_context_length=16,
        )

        # EfficientSAM3 (TinyViT-11M vision encoder, fine-tuned)
        sam3 = SAM3Inference(
            ckpt_path="models/sam3/efficient_sam3_tinyvit_m_ft.pt",
            mode="efficient",
            backbone_type="tinyvit",
            model_name="11m",
        )

        results = sam3.infer(bgr_uint8)
        # results: dict[obj_id -> {"concept", "masks", "boxes", "scores"}]
        #   masks:  (N, 1, H, W) bool   -- N detections for this concept
        #   boxes:  (N, 4)       float32 -- pixel-space [x0, y0, x1, y1]
        #   scores: (N,)         float32 -- confidence scores
    """

    # Default concept map for this project: text_prompt -> object_id
    DEFAULT_CONCEPT_MAP: Dict[str, int] = {
        "left robot arm":  CONCEPT_LEFT_ARM,
        "right robot arm": CONCEPT_RIGHT_ARM,
        "cup":             CONCEPT_CUP,
    }

    def __init__(
        self,
        ckpt_path: str,
        mode: str = "sam3",
        device: str = "cuda",
        compile: bool = True,
        concept_map: Optional[Dict[str, int]] = None,
        confidence_threshold: float = 0.5,
        # ── EfficientSAM3 "efficient" mode ──────────────────────────────
        backbone_type: str = "tinyvit",
        model_name: str = "11m",
        # ── EfficientSAM3 "litetext" / "efficient" modes ─────────────────
        text_encoder_type: Optional[str] = None,
        text_encoder_context_length: int = 16,
    ):
        """
        Args:
            ckpt_path:                  Path to the checkpoint (.pt / .pth) file.
            mode:                       One of "sam3" | "litetext" | "efficient".
            device:                     torch device string ("cuda" or "cpu").
            compile:                    Apply torch.compile for faster per-frame
                                        inference (CUDA only; first call triggers warm-up).
            concept_map:                Optional override for {text_prompt: object_id}.
                                        Defaults to DEFAULT_CONCEPT_MAP.
            confidence_threshold:       Detection score threshold; lower → more recalls.

            backbone_type:              [mode="efficient" only]
                                        Vision backbone: "tinyvit" | "efficientvit" | "repvit".
            model_name:                 [mode="efficient" only]
                                        Backbone variant.
                                        tinyvit    → "5m" | "11m" | "21m"
                                        efficientvit → "b0" | "b1" | "b2"
                                        repvit     → "m0.9" | "m1.1" | "m2.3"

            text_encoder_type:          [mode="litetext" or "efficient" only]
                                        MobileCLIP text encoder variant:
                                        "MobileCLIP-S0" | "MobileCLIP-S1" | "MobileCLIP2-L"
                                        None = keep original SAM3 CLIP text encoder.
            text_encoder_context_length:[mode="litetext" only] Token context length: 16 | 32 | 77.
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got '{mode}'")

        self.device = device
        self.concept_map = concept_map if concept_map is not None else self.DEFAULT_CONCEPT_MAP

        from sam3.model.sam3_image_processor import Sam3Processor

        model = _build_model(
            mode=mode,
            ckpt_path=ckpt_path,
            device=device,
            compile=compile,
            backbone_type=backbone_type,
            model_name=model_name,
            text_encoder_type=text_encoder_type,
            text_encoder_context_length=text_encoder_context_length,
        )
        self.processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    @torch.inference_mode()
    def infer(self, bgr: np.ndarray) -> Dict[int, dict]:
        """
        Run SAM 3 semantic segmentation on a single BGR frame.

        Vision encoding is performed once; each registered concept then runs its
        own text-conditioned grounding forward pass over the shared visual features.

        Args:
            bgr: (H, W, 3) uint8 BGR frame from RealSense / cv2.

        Returns:
            dict keyed by object_id (int), each value is a dict with:
                "concept": str                       -- the text prompt used
                "masks":   (N, 1, H, W) bool tensor  -- per-detection binary masks (None if no detections)
                "boxes":   (N, 4)       float32 tensor -- pixel-space [x0, y0, x1, y1] (None if no detections)
                "scores":  (N,)         float32 tensor -- confidence scores (None if no detections)
        """
        # BGR numpy -> PIL RGB (Sam3Processor expects PIL Image)
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # Encode visual features once for all concepts
        base_state = self.processor.set_image(pil_img)

        results: Dict[int, dict] = {}
        for concept, obj_id in self.concept_map.items():
            # Shallow-copy the state so that each concept's text features
            # do not bleed into the next concept's grounding pass.
            state = dict(base_state)
            state["backbone_out"] = dict(base_state["backbone_out"])

            state = self.processor.set_text_prompt(prompt=concept, state=state)

            masks  = state.get("masks")   # (N, 1, H, W) bool tensor or absent
            boxes  = state.get("boxes")   # (N, 4) float tensor or absent
            scores = state.get("scores")  # (N,) float tensor or absent

            results[obj_id] = {
                "concept": concept,
                "masks":   masks  if masks  is not None and masks.numel()  > 0 else None,
                "boxes":   boxes  if boxes  is not None and boxes.numel()  > 0 else None,
                "scores":  scores if scores is not None and scores.numel() > 0 else None,
            }

        return results


# ── Benchmark entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    # Default checkpoint filenames per mode (relative to models/)
    _DEFAULT_CKPT = {
        "sam3":      "sam3.pt",
        "litetext":  "efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt",
        "efficient": "efficient_sam3_tinyvit_m_ft.pt",
    }

    parser = argparse.ArgumentParser(description="Benchmark SAM3/EfficientSAM3 throughput")
    parser.add_argument("--ckpt",    type=str, default=None,
                        help="Path to checkpoint (default: auto-detect from models/)")
    parser.add_argument("--mode",    type=str, default="sam3",
                        choices=list(_MODES),
                        help="Model mode: sam3 | litetext | efficient  (default: sam3)")
    parser.add_argument("--device",  type=str, default="cuda",
                        help="torch device  (default: cuda)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (adds warm-up overhead)")
    parser.add_argument("--width",   type=int, default=640,
                        help="Synthetic frame width  (default: 640)")
    parser.add_argument("--height",  type=int, default=480,
                        help="Synthetic frame height  (default: 480)")
    parser.add_argument("--warmup",  type=int, default=1,
                        help="Number of warm-up frames before timing  (default: 1)")
    parser.add_argument("--iters",   type=int, default=20,
                        help="Number of timed iterations  (default: 20)")
    # EfficientSAM3-specific
    parser.add_argument("--backbone_type", type=str, default="tinyvit",
                        help="[mode=efficient] backbone: tinyvit | efficientvit | repvit")
    parser.add_argument("--model_name",    type=str, default="11m",
                        help="[mode=efficient] model variant e.g. 11m / b1 / m1.1")
    parser.add_argument("--text_encoder_type", type=str, default=None,
                        help="[mode=litetext/efficient] MobileCLIP-S0 | MobileCLIP-S1 | MobileCLIP2-L")
    parser.add_argument("--text_encoder_context_length", type=int, default=16,
                        help="[mode=litetext] token context length: 16 | 32 | 77  (default: 16)")
    args = parser.parse_args()

    # --- Resolve checkpoint path ---
    if args.ckpt is not None:
        ckpt_path = args.ckpt
    else:
        _repo_root = Path(__file__).resolve().parents[5]
        ckpt_path  = str(_repo_root / "models" / "sam3" / _DEFAULT_CKPT[args.mode])

    # --- GPU info ---
    if torch.cuda.is_available():
        _props = torch.cuda.get_device_properties(0)
        gpu_info = (
            f"{torch.cuda.get_device_name(0)}  "
            f"({_props.total_memory // (1024**2)} MiB)  "
            f"sm_{_props.major}{_props.minor}  "
            f"CUDA {torch.version.cuda}"
        )
    else:
        gpu_info = "N/A (CPU only)"

    print("=" * 60)
    print("  SAM3 / EfficientSAM3 Inference Frequency Benchmark")
    print("=" * 60)
    print(f"  GPU              : {gpu_info}")
    print(f"  mode             : {args.mode}")
    print(f"  checkpoint       : {ckpt_path}")
    print(f"  device           : {args.device}")
    print(f"  compile          : {args.compile}")
    if args.mode == "efficient":
        print(f"  backbone_type    : {args.backbone_type}")
        print(f"  model_name       : {args.model_name}")
    if args.mode in ("litetext", "efficient") and args.text_encoder_type:
        print(f"  text_encoder     : {args.text_encoder_type}  ctx={args.text_encoder_context_length}")
    print(f"  frame size       : {args.width}x{args.height}")
    print(f"  warm-up          : {args.warmup}  |  timed iters: {args.iters}")
    print("=" * 60)

    # --- Build model ---
    t_load = time.time()
    sam3 = SAM3Inference(
        ckpt_path=ckpt_path,
        mode=args.mode,
        device=args.device,
        compile=args.compile,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
        text_encoder_context_length=args.text_encoder_context_length,
    )
    print(f"Model loaded in {time.time() - t_load:.2f}s\n")

    # Shared synthetic BGR frame (constant across iterations to isolate GPU time)
    rng   = np.random.default_rng(42)
    frame = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)

    # --- Warm-up (not timed) ---
    print(f"Running {args.warmup} warm-up frame(s) ...")
    for _ in range(args.warmup):
        sam3.infer(frame)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    # --- Timed benchmark ---
    latencies = []
    for i in range(args.iters):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = sam3.infer(frame)
        if args.device == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

        det_summary = "  ".join(
            f"{results[oid]['concept'].split()[0]}="
            f"{'det' if results[oid]['masks'] is not None else 'none'}"
            for oid in (CONCEPT_LEFT_ARM, CONCEPT_RIGHT_ARM, CONCEPT_CUP)
        )
        print(f"  iter {i+1:>3d}: {latencies[-1]*1000:>7.2f} ms  [{det_summary}]")

    latencies_ms = [l * 1000 for l in latencies]
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms  = min(latencies_ms)
    max_ms  = max(latencies_ms)
    var     = sum((x - mean_ms) ** 2 for x in latencies_ms) / len(latencies_ms)
    std_ms  = var ** 0.5

    print()
    print("=" * 60)
    print(f"  mean  : {mean_ms:>7.2f} ms   ({1000/mean_ms:>5.1f} Hz)")
    print(f"  min   : {min_ms:>7.2f} ms")
    print(f"  max   : {max_ms:>7.2f} ms")
    print(f"  std   : {std_ms:>7.2f} ms")
    print("=" * 60)
