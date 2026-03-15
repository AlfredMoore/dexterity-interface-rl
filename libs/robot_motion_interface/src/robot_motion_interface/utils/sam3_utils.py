"""
SAM 3 / EfficientSAM3 inference wrapper for per-frame semantic segmentation.

Three modes are supported (controlled by the `mode` argument):

  "sam3"      -- Original SAM3 (ViT-H vision + CLIP text, 848 M params).
                 Requires: dep/sam3-HAND installed.
                 Checkpoint: models/sam3/sam3.pt

                 Benchmark:
                   python -m robot_motion_interface.utils.sam3_utils \
                    --frames_dir models/data_examples/hand_setup_frames \
                    --mode sam3 --ckpt models/sam3/sam3.pt \
                    --compile

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
                   python -m robot_motion_interface.utils.sam3_utils \
                     --mode litetext \
                     --ckpt models/sam3/efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt \
                     --text_encoder_type MobileCLIP-S0 \
                     --text_encoder_context_length 16

                   # S1-16 (64 M text params)
                   python -m robot_motion_interface.utils.sam3_utils \
                     --mode litetext \
                     --ckpt models/sam3/efficient_sam3_image_encoder_mobileclip_s1_ctx16.pt \
                     --text_encoder_type MobileCLIP-S1 \
                     --text_encoder_context_length 16

                   # L-16 (124 M text params — most accurate text)
                   python -m robot_motion_interface.utils.sam3_utils \
                     --mode litetext \
                     --ckpt models/sam3/efficient_sam3_image_encoder_mobileclip2_l_ctx16.pt \
                     --text_encoder_type MobileCLIP2-L \
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
                   python -m robot_motion_interface.utils.sam3_utils \
                     --mode efficient \
                     --ckpt models/sam3/efficient_sam3_tinyvit_m_ft.pt \
                     --backbone_type tinyvit --model_name 11m
"""

import time
import cv2
import numpy as np
import torch
from torchvision.ops import nms as tv_nms
from PIL import Image
from typing import Dict, Literal, Optional
from sam3.model.sam3_image_processor import Sam3Processor


# Object IDs assigned to each semantic concept (fixed at init, used downstream)
CONCEPT_ARM  = 1   # single prompt, up to 2 instances (both arms)
CONCEPT_HAND = 2   # single prompt, up to 2 instances (both hands)
CONCEPT_OBJ  = 3

# Supported mode strings
_MODES = ("sam3", "litetext", "efficient")

# Default concept map used by the benchmark (callers may pass their own)
DEFAULT_CONCEPT_MAP: Dict[str, int] = {
    "robot arm":  CONCEPT_ARM,
    # "robot hand": CONCEPT_HAND,
    "cup":        CONCEPT_OBJ,
}

# Default max instances per concept ID for the benchmark
DEFAULT_MAX_INSTANCES: Dict[int, int] = {
    CONCEPT_ARM:  2,
    CONCEPT_HAND: 2,
    CONCEPT_OBJ:  1,
}


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

    def __init__(
        self,
        ckpt_path: str,
        concept_map: Dict[str, int],
        mode: str = "sam3",
        device: str = "cuda",
        compile: bool = True,
        confidence_threshold: float = 0.5,
        max_instances_per_concept: Optional[Dict[int, int]] = None,
        nms_iou_threshold: float = 0.5,
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
            concept_map:                Required. Maps text prompt → integer object_id.
                                        E.g. {"robot arm": 1, "cup": 3}.
                                        All detection results are keyed by object_id.
            mode:                       One of "sam3" | "litetext" | "efficient".
            device:                     torch device string ("cuda" or "cpu").
            compile:                    Apply torch.compile for faster per-frame
                                        inference (CUDA only; first call triggers warm-up).
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
        self.concept_map = concept_map
        # max_instances: fallback to 1 per concept if not specified
        self.max_instances: Dict[int, int] = max_instances_per_concept or {}
        self.nms_iou_threshold = nms_iou_threshold

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

            if boxes is not None and boxes.numel() > 0:
                keep = tv_nms(boxes.float(), scores.float(), self.nms_iou_threshold)
                boxes  = boxes[keep]
                scores = scores[keep]
                masks  = masks[keep] if masks is not None else None

                top_k = self.max_instances.get(obj_id, 1)
                boxes  = boxes[:top_k]
                scores = scores[:top_k]
                masks  = masks[:top_k] if masks is not None else None

            results[obj_id] = {
                "concept": concept,
                "masks":   masks  if masks  is not None and masks.numel()  > 0 else None,
                "boxes":   boxes  if boxes  is not None and boxes.numel()  > 0 else None,
                "scores":  scores if scores is not None and scores.numel() > 0 else None,
            }

        return results


    def merge_masks(self, results: Dict[int, dict], obj_ids: Optional[list] = None) -> Dict[int, Optional[torch.Tensor]]:
        """
        Merge all per-instance masks for each concept into a single binary mask.

        Args:
            results:  Output of SAM3Inference.infer() — dict[obj_id -> {masks, ...}].
            obj_ids:  List of obj_ids to process. None = all obj_ids in results.

        Returns:
            dict[obj_id -> merged_mask] where merged_mask is:
                (1, H, W) bool tensor  — union of all instance masks for that concept.
                None                   — if the concept had no detections.
        """
        ids = obj_ids if obj_ids is not None else list(results.keys())
        merged: Dict[int, Optional[torch.Tensor]] = {}
        for oid in ids:
            res = results.get(oid)
            if res is None or res["masks"] is None:
                merged[oid] = None
                continue
            # masks: (N, 1, H, W) bool — reduce across instance dim
            merged[oid] = res["masks"].any(dim=0, keepdim=False)  # (1, H, W) bool
        return merged


# ── Benchmark entry point ──────────────────────────────────────────────────────

# Generic colour palette (BGR) — cycles across (obj_id, instance_idx) pairs
_VIS_PALETTE = [
    (0,   80, 255),   # orange-red
    (255,  80,   0),  # blue
    (0,  200,   0),   # green
    (200,   0, 200),  # purple
    (0,  200, 200),   # yellow
    (200, 200,   0),  # cyan
]


def _draw_results(frame: np.ndarray, results: Dict[int, dict]) -> np.ndarray:
    """Draw masks + boxes + labels from sam3.infer() onto a BGR frame copy."""
    vis = frame.copy()
    overlay = vis.copy()
    h, w = vis.shape[:2]

    color_idx = 0
    for _, res in results.items():
        label = res["concept"]
        n = len(res["masks"]) if res["masks"] is not None else 0

        for idx in range(n):
            color = _VIS_PALETTE[color_idx % len(_VIS_PALETTE)]
            color_idx += 1

            # Mask overlay
            mask = res["masks"][idx, 0].cpu().numpy()   # (H, W) bool
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            overlay[mask_resized] = color

            # Bounding box + label
            if res["boxes"] is not None:
                box   = res["boxes"][idx].cpu().numpy().astype(int)
                score = float(res["scores"][idx].cpu()) if res["scores"] is not None else 0.0
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{label}#{idx} {score:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Blend mask overlay
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    return vis


if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    # Default checkpoint filenames per mode (relative to models/sam3/)
    _DEFAULT_CKPT = {
        "sam3":      "sam3.pt",
        "litetext":  "efficient_sam3_image_encoder_mobileclip_s0_ctx16.pt",
        "efficient": "efficient_sam3_tinyvit_m_ft.pt",
    }

    _REPO_ROOT = Path(__file__).resolve().parents[5]

    parser = argparse.ArgumentParser(description="Benchmark SAM3/EfficientSAM3 throughput")
    parser.add_argument("--ckpt",    type=str, default=None,
                        help="Path to checkpoint (default: auto-detect from models/sam3/)")
    parser.add_argument("--mode",    type=str, default="sam3", choices=list(_MODES))
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--width",   type=int, default=640)
    parser.add_argument("--height",  type=int, default=480)
    parser.add_argument("--warmup",  type=int, default=1)
    parser.add_argument("--iters",   type=int, default=20,
                        help="Timed iterations (single-image / synthetic mode only)")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Folder of frames to run sequentially (jpg/png, sorted by name).")
    parser.add_argument("--out_dir",    type=str, default=None,
                        help="Save annotated frames here (only with --frames_dir).")
    parser.add_argument("--video_fps",  type=float, default=30.0,
                        help="Playback fps of the output video (default: 30).")
    # EfficientSAM3-specific
    parser.add_argument("--backbone_type",              type=str, default="tinyvit")
    parser.add_argument("--model_name",                 type=str, default="11m")
    parser.add_argument("--text_encoder_type",          type=str, default=None)
    parser.add_argument("--text_encoder_context_length",type=int, default=16)
    args = parser.parse_args()

    # ── Resolve checkpoint path ────────────────────────────────────────────────
    ckpt_path = args.ckpt or str(_REPO_ROOT / "models" / "sam3" / _DEFAULT_CKPT[args.mode])

    def _resolve(p: str) -> Path:
        path = Path(p)
        return _REPO_ROOT / path if not path.is_absolute() else path

    # ── Frame list ─────────────────────────────────────────────────────────────
    if args.frames_dir is not None:
        frames_dir  = _resolve(args.frames_dir)
        frame_paths = sorted(p for p in frames_dir.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not frame_paths:
            raise FileNotFoundError(f"No jpg/png frames in: {frames_dir}")
        out_dir = _resolve(args.out_dir) if args.out_dir else \
                  frames_dir.parent / (frames_dir.name + "_sam3")
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "folder"
    else:
        rng         = np.random.default_rng(42)
        _synth      = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
        frame_paths = [None] * args.iters   # sentinel: use synthetic frame
        mode        = "synthetic"

    # ── GPU info ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        _props = torch.cuda.get_device_properties(0)
        gpu_info = (f"{torch.cuda.get_device_name(0)}  "
                    f"({_props.total_memory // (1024**2)} MiB)  "
                    f"sm_{_props.major}{_props.minor}  CUDA {torch.version.cuda}")
    else:
        gpu_info = "N/A (CPU only)"

    print("=" * 60)
    print("  SAM3 / EfficientSAM3 Inference Frequency Benchmark")
    print("=" * 60)
    print(f"  GPU              : {gpu_info}")
    print(f"  mode             : {args.mode}")
    print(f"  checkpoint       : {ckpt_path}")
    print(f"  device           : {args.device}  compile={args.compile}")
    if args.mode == "efficient":
        print(f"  backbone         : {args.backbone_type}/{args.model_name}")
    if args.mode in ("litetext", "efficient") and args.text_encoder_type:
        print(f"  text_encoder     : {args.text_encoder_type}  ctx={args.text_encoder_context_length}")
    print(f"  frame size       : {args.width}x{args.height}")
    if mode == "folder":
        print(f"  frames_dir       : {args.frames_dir}  ({len(frame_paths)} frames)")
        print(f"  out_dir          : {out_dir}")
    print(f"  warm-up          : {args.warmup}")
    print("=" * 60)

    # ── Build model ────────────────────────────────────────────────────────────
    t_load = time.time()
    sam3 = SAM3Inference(
        ckpt_path=ckpt_path,
        concept_map=DEFAULT_CONCEPT_MAP,
        mode=args.mode,
        device=args.device,
        compile=args.compile,
        max_instances_per_concept=DEFAULT_MAX_INSTANCES,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
        text_encoder_context_length=args.text_encoder_context_length,
    )
    print(f"Model loaded in {time.time() - t_load:.2f}s\n")

    # ── Warm-up ────────────────────────────────────────────────────────────────
    warmup_frame = (cv2.imread(str(frame_paths[0])) if mode == "folder"
                    else _synth)
    warmup_frame = cv2.resize(warmup_frame, (args.width, args.height))
    print(f"Running {args.warmup} warm-up frame(s) ...")
    for _ in range(args.warmup):
        sam3.infer(warmup_frame)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    # ── Timed inference ────────────────────────────────────────────────────────
    latencies = []
    for i, fpath in enumerate(frame_paths):
        frame = (cv2.imread(str(fpath)) if fpath is not None else _synth)
        frame = cv2.resize(frame, (args.width, args.height))

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = sam3.infer(frame)
        if args.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        det_summary = "  ".join(
            f"{res['concept']}="
            f"{len(res['masks']) if res['masks'] is not None else 0}det"
            for res in results.values()
        )
        label = "frame" if mode == "folder" else "iter"
        print(f"  {label} {i+1:>4d}/{len(frame_paths)}: {ms:>7.2f} ms  [{det_summary}]")

        if mode == "folder":
            merged = sam3.merge_masks(results, obj_ids=[CONCEPT_ARM])
            arm_mask = merged[CONCEPT_ARM]  # (1, H, W) bool or None
            vis = _draw_results(frame, results)
            if arm_mask is not None:
                # Draw merged arm mask as a filled green overlay
                arm_np = arm_mask[0].cpu().numpy()  # (H, W) bool
                arm_np = cv2.resize(arm_np.astype(np.uint8),
                                    (frame.shape[1], frame.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                overlay = vis.copy()
                overlay[arm_np] = (0, 255, 0)
                cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
                cv2.putText(vis, "arm(merged)", (8, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / fpath.name), vis)

    latencies_ms = latencies
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms  = min(latencies_ms)
    max_ms  = max(latencies_ms)
    std_ms  = (sum((x - mean_ms) ** 2 for x in latencies_ms) / len(latencies_ms)) ** 0.5

    print()
    print("=" * 60)
    print(f"  frames     : {len(latencies_ms)}")
    print(f"  mean  : {mean_ms:>7.2f} ms   ({1000/mean_ms:>5.1f} Hz)")
    print(f"  min   : {min_ms:>7.2f} ms")
    print(f"  max   : {max_ms:>7.2f} ms")
    print(f"  std   : {std_ms:>7.2f} ms")
    print("=" * 60)

    if mode == "folder":
        video_path = out_dir.parent / (out_dir.name + ".mp4")
        video_fps  = args.video_fps
        sample_bgr = cv2.imread(str(sorted(out_dir.iterdir())[0]))
        h_v, w_v   = sample_bgr.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w_v, h_v)
        )
        for p in sorted(out_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                f = cv2.imread(str(p))
                if f is not None:
                    writer.write(f)
        writer.release()
        print(f"\n  annotated frames → {out_dir}")
        print(f"  video saved      → {video_path}  ({video_fps:.1f} fps)")
