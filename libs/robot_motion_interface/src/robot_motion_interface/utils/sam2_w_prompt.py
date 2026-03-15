"""
SAM3-init + SAM2-track pipeline in a single class.

SAM3 runs twice (warmup + init_prompt) to provide text-prompted bounding boxes
and masks as initial prompts for SAM2.  After init_prompt(), SAM3 is offloaded
to CPU and VRAM is freed.  All subsequent sam2_infer() calls use only SAM2,
self-tracking via prev_logits (dense mask embeddings).

Flow:
    tracker = SAM2WithPrompt(...)
    tracker.warmup(first_frame)          # SAM3 + SAM2 compile warmup
    tracker.init_prompt(init_frame)      # SAM3 infer → init state, SAM3 offloaded
    while True:
        results = tracker.sam2_infer(frame)  # SAM2 only, ~30 Hz
        for obj in results:
            obj["mask"]  # (H, W) bool
            obj["box"]   # (4,)   float32 xyxy (last known from SAM3)
            obj["score"] # float
            obj["concept"], obj["instance_idx"]

Benchmark:
    # sam2.1_hiera_small fps: 51.2hz / 59.2hz(--compile)
    python -m robot_motion_interface.utils.sam2_w_prompt \
        --sam3_ckpt models/sam3/sam3.pt \
        --sam2_ckpt models/sam2/sam2.1_hiera_small.pt \
        --frames_dir models/data_examples/hand_setup_frames \
        --compile
    
    # sam2.1_hiera_b+(better segmentation) fps: 41.9hz / 42.0hz(--compile)
    python -m robot_motion_interface.utils.sam2_w_prompt \
        --sam3_ckpt models/sam3/sam3.pt \
        --sam2_ckpt models/sam2/sam2.1_hiera_base_plus.pt \
        --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
        --frames_dir models/data_examples/hand_setup_frames \
        --compile
    
    # with hands 
    # sam2.1_hiera_small.pt fps:40.4hz / 41.8hz(--compile)
    python -m robot_motion_interface.utils.sam2_w_prompt \
        --sam3_ckpt models/sam3/sam3.pt \
        --sam2_ckpt models/sam2/sam2.1_hiera_small.pt \
        --frames_dir models/data_examples/hand_setup_frames \
        --hand --compile
    
    # sam2.1_hiera_b+ fps: 33.6hz / 37.1hz(--compile)
    python -m robot_motion_interface.utils.sam2_w_prompt \
        --sam3_ckpt models/sam3/sam3.pt \
        --sam2_ckpt models/sam2/sam2.1_hiera_base_plus.pt \
        --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
        --frames_dir models/data_examples/hand_setup_frames \
        --hand --compile
        
    # real data
    python -m robot_motion_interface.utils.sam2_w_prompt \
        --sam3_ckpt models/sam3/sam3.pt \
        --sam2_ckpt models/sam2/sam2.1_hiera_base_plus.pt \
        --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
        --frames_dir models/data_examples/realsense/rs_record_20260314_223830/color \
        --video_fps 60 \
        --compile
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional

from robot_motion_interface.utils.sam3_utils import SAM3Inference, CONCEPT_ARM as _SAM3_CONCEPT_ARM
from robot_motion_interface.utils.sam2_utils import SAM2Inference


# Text prompts sent to SAM3. Change these strings to adjust detection vocabulary.
TEXT_ARM  = "robot arm"
TEXT_HAND = "robot hand"
TEXT_OBJ  = "box"

CONCEPT_ARM  = 1   # single prompt, up to 2 instances (both arms)
CONCEPT_HAND = 2   # single prompt, up to 2 instances (both hands)
CONCEPT_OBJ  = 3

# Default concept map: text prompt → SAM3 object ID.

DEFAULT_CONCEPT_MAP: Dict[str, int] = {
    TEXT_ARM:  CONCEPT_ARM,
    # TEXT_HAND: CONCEPT_HAND,
    TEXT_OBJ:  CONCEPT_OBJ,
}


# Default max instances per concept ID (bimanual = 2, objects = 1).
DEFAULT_MAX_INSTANCES: Dict[int, int] = {
    CONCEPT_ARM:  2,
    CONCEPT_HAND: 2,
    CONCEPT_OBJ:  1,
}

def bgr2rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV / RealSense) to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb2bgr(frame: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR (for cv2.imwrite / OpenCV display)."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _sam3_mask_to_logits(mask: torch.Tensor, h4: int, w4: int) -> np.ndarray:
    """
    Convert a SAM3 binary mask to SAM2 mask_input (logit) format.

    Args:
        mask: (1, H, W) bool tensor (one instance from SAM3 output).
        h4:   Target height  = original H // 4.
        w4:   Target width   = original W // 4.

    Returns:
        (1, h4, w4) float32 numpy array.  Foreground pixels → +20, background → -20.
    """
    mask_t = mask.float().unsqueeze(0)                        # (1, 1, H, W)
    logits = F.interpolate(mask_t, (h4, w4),
                           mode='bilinear', align_corners=False).squeeze(0)  # (1, h4, w4)
    logits = logits * 40.0 - 20.0                            # scale to strong logit range
    return logits.cpu().numpy().astype(np.float32)


class SAM2WithPrompt:
    """
    SAM3-initialised SAM2 tracker.

    Uses the original SAM3 (ViT-H + CLIP, full accuracy) for init_prompt(),
    then offloads it to CPU and tracks with SAM2 only.

    Usage:
        tracker = SAM2WithPrompt(
            sam2_ckpt="models/sam2/sam2.1_hiera_small.pt",
            sam2_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
            sam3_ckpt="models/sam3/sam3.pt",
        )
        tracker.warmup(first_bgr_frame)
        tracker.init_prompt(init_bgr_frame)
        for bgr in camera_stream:
            results = tracker.sam2_infer(bgr)
    """

    def __init__(
        self,
        sam2_ckpt: str,
        sam2_cfg: str,
        sam3_ckpt: str,
        concept_map: dict,
        max_instances_per_concept: dict,
        device: str = "cuda",
        compile: bool = True,
        merge_arm_mask: bool = True,
    ):
        """
        Args:
            sam2_ckpt:                  Path to SAM2 checkpoint.
            sam2_cfg:                   SAM2 Hydra config name, e.g. "configs/sam2.1/sam2.1_hiera_s.yaml".
            sam3_ckpt:                  Path to SAM3 checkpoint (sam3.pt).
            use_hand:                   Include TEXT_HAND in detection. Ignored when concept_map is
                                        provided explicitly.
            concept_map:                {text_prompt: object_id}. Overrides use_hand when provided.
            max_instances_per_concept:  {object_id: max_count}. Defaults to 1 per concept if not set.
                                        E.g. {CONCEPT_ARM: 2, CONCEPT_HAND: 2} for bimanual tracking.
            device:                     torch device string ("cuda" or "cpu").
            compile:                    Apply torch.compile to SAM2 (CUDA only).
        """
        self.device = device
        self._merge_arm_mask = merge_arm_mask
        self._objects: list[dict] = []

        self.sam3 = SAM3Inference(
            ckpt_path=sam3_ckpt,
            concept_map=concept_map,
            mode="sam3",
            device=device,
            compile=False,
            max_instances_per_concept=max_instances_per_concept,
        )
        self.sam2 = SAM2Inference(
            ckpt_path=sam2_ckpt,
            model_cfg=sam2_cfg,
            device=device,
            compile_image_encoder=compile,
            multimask_output=False,
        )

    # ── Warmup ─────────────────────────────────────────────────────────────

    def warmup(self, frame: np.ndarray, n: int = 3) -> None:
        """
        Warm up both models so that torch.compile completes its tracing.

        SAM3 is warmed up by running full infer() passes.
        SAM2 is warmed up with set_image() + predict() using a dummy center box.
        SAM3 is NOT released here — init_prompt() still needs it.

        Args:
            frame: (H, W, 3) uint8 BGR frame.
            n:     Number of warmup iterations (default 3).
        """
        h, w = frame.shape[:2]
        dummy_box = np.array([w // 4, h // 4, 3 * w // 4, 3 * h // 4], dtype=np.float32)
        rgb = bgr2rgb(frame)

        print(f"  Warming up SAM2 ({n} iters) ...")
        for _ in range(n):
            with torch.inference_mode():
                with torch.autocast(self.sam2.device.type, dtype=torch.bfloat16):
                    self.sam2.predictor.set_image(rgb)
                    self.sam2.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=dummy_box,
                        mask_input=None,
                        multimask_output=False,
                        return_logits=True,
                    )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("  Warmup done.")

    # ── Initialisation ──────────────────────────────────────────────────────

    def init_prompt(self, frame: np.ndarray) -> list[dict]:
        """
        Run SAM3 once to get initial bounding boxes and masks for all concepts.
        SAM3 is offloaded to CPU after this call to free VRAM.

        Args:
            frame: (H, W, 3) uint8 BGR frame.

        Returns:
            List of object dicts (same format as sam2_infer).
            Masks are not yet filled (prev_logits = None at this stage).
        """
        h, w = frame.shape[:2]
        h4, w4 = h // 4, w // 4

        self._objects.clear()
        sam3_results = self.sam3.infer(frame)  # SAM3 expects BGR

        for obj_id, result in sam3_results.items():
            concept = result["concept"]
            boxes   = result["boxes"]   # (N, 4) tensor or None
            masks   = result["masks"]   # (N, 1, H, W) bool tensor or None
            scores  = result["scores"]  # (N,)   tensor or None

            if boxes is None:
                print(f"  [init_prompt] no detections for '{concept}'")
                continue

            # ── Merged arm path ─────────────────────────────────────────────
            # When merge_arm_mask is enabled, collapse all arm instances into
            # one SAM2 object using the union mask and its bounding box.
            if self._merge_arm_mask and obj_id == _SAM3_CONCEPT_ARM and masks is not None:
                merged = self.sam3.merge_masks(sam3_results, obj_ids=[_SAM3_CONCEPT_ARM])
                merged_mask = merged[_SAM3_CONCEPT_ARM]  # (1, H, W) bool or None
                if merged_mask is not None:
                    ys, xs = torch.where(merged_mask[0])
                    merged_box = np.array(
                        [xs.min().item(), ys.min().item(),
                         xs.max().item(), ys.max().item()], dtype=np.float32
                    )
                    sam3_logits = _sam3_mask_to_logits(merged_mask, h4, w4)
                    mean_score  = float(scores.mean().cpu())
                    self._objects.append({
                        "concept":      concept,
                        "obj_id":       obj_id,
                        "instance_idx": 0,
                        "box":          merged_box,
                        "sam3_logits":  sam3_logits,
                        "_track_box":   merged_box.copy(),
                        "prev_logits":  None,
                        "mask":         None,
                        "score":        mean_score,
                    })
                    print(f"  [init_prompt] {concept}(merged)  box={merged_box.astype(int)}  "
                          f"instances={len(boxes)}  score={mean_score:.3f}")
                continue

            # ── Per-instance path (all other concepts) ───────────────────────
            boxes_np  = boxes.cpu().numpy().astype(np.float32)
            scores_np = scores.cpu().numpy().astype(np.float32)

            instances = []
            for i, (box, score) in enumerate(zip(boxes_np, scores_np)):
                sam3_logits = None
                if masks is not None and i < len(masks):
                    sam3_logits = _sam3_mask_to_logits(masks[i], h4, w4)
                instances.append((box, score, sam3_logits))

            if self.sam3.max_instances.get(obj_id, 1) > 1:
                # Sort by box x-center: instance_idx 0 = left, 1 = right
                instances.sort(key=lambda t: (t[0][0] + t[0][2]) / 2)

            for i, (box, score, sam3_logits) in enumerate(instances):
                self._objects.append({
                    "concept":      concept,
                    "obj_id":       obj_id,
                    "instance_idx": i,
                    "box":          box,
                    "sam3_logits":  sam3_logits,
                    "_track_box":   box.copy(),
                    "prev_logits":  None,
                    "mask":         None,
                    "score":        float(score),
                })
                side = f" ({'left' if i == 0 else 'right'})" if self.sam3.max_instances.get(obj_id, 1) > 1 else ""
                print(f"  [init_prompt] {concept}#{i}{side}  box={box.astype(int)}  score={score:.3f}")

        # Offload SAM3 to CPU — VRAM freed
        self.sam3.processor.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  SAM3 offloaded to CPU.")

        return list(self._objects)

    # ── Per-frame SAM2 tracking ─────────────────────────────────────────────

    @torch.inference_mode()
    def sam2_infer(self, frame: np.ndarray) -> list[dict]:
        """
        Track all objects in one RGB frame using SAM2 only.

        First call after init_prompt(): SAM2 is prompted with box (from SAM3).
        All subsequent calls: only prev_logits are used (self-tracking).

        Args:
            frame: (H, W, 3) uint8 BGR frame.

        Returns:
            List of object dicts, one per tracked object:
                concept      str
                instance_idx int
                box          (4,) float32  xyxy (last known from SAM3 init)
                mask         (H, W) bool
                score        float
        """
        if not self._objects:
            return []

        with torch.autocast(self.sam2.device.type, dtype=torch.bfloat16):
            self.sam2.predictor.set_image(bgr2rgb(frame))  # SAM2 expects RGB

            for obj in self._objects:
                # Box prompt: SAM3 box on first frame, then derived from prev mask.
                # Providing a box every frame gives SAM2 a spatial anchor and
                # prevents the mask from drifting when using mask_input only.
                box_arg        = obj["_track_box"]   # None on very first call
                mask_input_arg = obj["prev_logits"]  # None on very first call

                if box_arg is None and mask_input_arg is None:
                    continue

                masks, scores, logits = self.sam2.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_arg,
                    mask_input=mask_input_arg,
                    multimask_output=False,
                    return_logits=True,
                )
                obj["mask"]        = masks[0] > 0.0  # binarize full-res logits
                obj["score"]       = float(scores[0])
                obj["prev_logits"] = logits[[0]]     # (1, h4, w4) float32

                # Update tracking box for next frame from current mask
                if obj["mask"].any():
                    ys, xs = np.where(obj["mask"])
                    obj["_track_box"] = np.array(
                        [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32
                    )
                # If mask is empty, keep previous _track_box so SAM2 still has an anchor

        return list(self._objects)


# ── Visualisation ───────────────────────────────────────────────────────────

_PALETTE = [
    (  0,  80, 255),
    (255,  80,   0),
    (  0, 200,   0),
    (200,   0, 200),
    (  0, 200, 200),
]


def draw_results(frame: np.ndarray, objects: list[dict]) -> np.ndarray:
    """Overlay masks and bounding boxes on a frame.

    Box is derived from the current SAM2 mask, not the SAM3 init box.
    Skips objects whose mask is None or empty.
    """
    vis = frame.copy()
    for i, obj in enumerate(objects):
        mask = obj.get("mask")
        if mask is None or not mask.any():
            continue
        color = _PALETTE[i % len(_PALETTE)]

        # Mask overlay
        colored = np.zeros_like(vis)
        colored[:, :] = color
        vis = np.where(mask[:, :, None],
                       (vis * 0.55 + colored * 0.45).astype(np.uint8), vis)

        # Bounding box derived from current mask
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{obj['concept']}#{obj['instance_idx']} {obj['score']:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


# ── Benchmark entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parents[5]

    parser = argparse.ArgumentParser(
        description="Benchmark SAM3-init + SAM2-track pipeline"
    )
    parser.add_argument("--sam3_ckpt", default="models/sam3/sam3.pt")
    parser.add_argument("--sam2_ckpt",
                        default="models/sam2/sam2.1_hiera_small.pt")
    parser.add_argument("--sam2_cfg",
                        default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--compile",   action="store_true")
    parser.add_argument("--warmup",    type=int, default=3)
    parser.add_argument("--frames_dir",default=None)
    parser.add_argument("--out_dir",   default=None)
    parser.add_argument("--video_fps", type=float, default=30.0)
    parser.add_argument("--width",     type=int,   default=640)
    parser.add_argument("--height",    type=int,   default=480)
    args = parser.parse_args()
    

    def _resolve(p: str) -> str:
        path = Path(p)
        return str(_REPO_ROOT / path) if not path.is_absolute() else p

    # ── Frame list ──────────────────────────────────────────────────────────
    if args.frames_dir is not None:
        frames_dir  = Path(_resolve(args.frames_dir))
        frame_paths = sorted(p for p in frames_dir.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not frame_paths:
            raise FileNotFoundError(f"No jpg/png frames in: {frames_dir}")
        out_dir = Path(_resolve(args.out_dir)) if args.out_dir else \
                  frames_dir.parent / (frames_dir.name + "_sam2wp")
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "folder"
    else:
        frame_paths = [_REPO_ROOT / "models" / "data_examples" / "image.jpg"]
        mode        = "single"

    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        gpu_info = (f"{torch.cuda.get_device_name(0)}  "
                    f"({_p.total_memory // (1024**2)} MiB)  "
                    f"sm_{_p.major}{_p.minor}  CUDA {torch.version.cuda}")
    else:
        gpu_info = "N/A (CPU only)"

    print("=" * 65)
    print("  SAM3-init + SAM2-track Benchmark")
    print("=" * 65)
    print(f"  GPU        : {gpu_info}")
    print(f"  SAM3 ckpt  : {_resolve(args.sam3_ckpt)}")
    print(f"  SAM2 ckpt  : {_resolve(args.sam2_ckpt)}")
    print(f"  compile    : {args.compile}")
    if mode == "folder":
        print(f"  frames_dir : {args.frames_dir}  ({len(frame_paths)} frames)")
    print("=" * 65)

    # ── Build ────────────────────────────────────────────────────────────────
    t_load = time.time()
    tracker = SAM2WithPrompt(
        sam2_ckpt=_resolve(args.sam2_ckpt),
        sam2_cfg=args.sam2_cfg,
        sam3_ckpt=_resolve(args.sam3_ckpt),
        concept_map=DEFAULT_CONCEPT_MAP,
        max_instances_per_concept=DEFAULT_MAX_INSTANCES,
        device=args.device,
        compile=args.compile,
    )
    print(f"Models loaded in {time.time() - t_load:.2f}s\n")

    # ── Warmup ───────────────────────────────────────────────────────────────
    warmup_frame = cv2.imread(str(frame_paths[0]))
    warmup_frame = cv2.resize(warmup_frame, (args.width, args.height))
    print(f"Running warmup ({args.warmup} iters) ...")
    tracker.warmup(warmup_frame, n=args.warmup)
    print()

    # ── Init prompt ──────────────────────────────────────────────────────────
    init_frame = cv2.imread(str(frame_paths[0]))
    init_frame = cv2.resize(init_frame, (args.width, args.height))
    print("Running SAM3 init_prompt ...")
    t_init = time.time()
    init_objects = tracker.init_prompt(init_frame)
    print(f"init_prompt done in {(time.time() - t_init)*1000:.1f} ms")
    if not init_objects:
        raise RuntimeError(
            "SAM3 found no objects. Check that --frames_dir points to frames "
            f"containing the target concepts ({TEXT_ARM}, {TEXT_OBJ})."
        )
    print(f"Tracking {len(init_objects)} object(s): "
          f"{[(o['concept'], o['instance_idx']) for o in init_objects]}\n")

    # ── SAM2 tracking loop ───────────────────────────────────────────────────
    n_iters   = len(frame_paths) if mode == "folder" else 50
    latencies: list[float] = []

    for i in range(n_iters):
        fpath = frame_paths[i % len(frame_paths)]
        frame = cv2.imread(str(fpath))
        if frame is None:
            print(f"  [skip] cannot read {fpath.name}")
            continue
        frame = cv2.resize(frame, (args.width, args.height))

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = tracker.sam2_infer(frame)
        if args.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        obj_str = "  ".join(
            f"{o['concept']}#{o['instance_idx']} {o['score']:.2f}" for o in results
        ) or "none"
        label = "frame" if mode == "folder" else "iter"
        print(f"  {label} {i+1:>4d}/{n_iters}: {ms:>7.2f} ms   [{obj_str}]")

        if mode == "folder":
            vis = draw_results(frame, results)
            cv2.imwrite(str(out_dir / fpath.name), vis)

    mean_ms = sum(latencies) / len(latencies)
    min_ms  = min(latencies)
    max_ms  = max(latencies)
    std_ms  = (sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5

    print()
    print("=" * 65)
    print(f"  frames (SAM2 only) : {len(latencies)}")
    print(f"  mean  : {mean_ms:>7.2f} ms   ({1000/mean_ms:>5.1f} Hz)")
    print(f"  min   : {min_ms:>7.2f} ms")
    print(f"  max   : {max_ms:>7.2f} ms")
    print(f"  std   : {std_ms:>7.2f} ms")
    print("=" * 65)

    if mode == "folder":
        print(f"\n  annotated frames → {out_dir}")
        video_path = out_dir.parent / (out_dir.name + ".mp4")
        sample_bgr = cv2.imread(str(sorted(out_dir.iterdir())[0]))
        h_v, w_v   = sample_bgr.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (w_v, h_v)
        )
        for p in sorted(out_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                f = cv2.imread(str(p))
                if f is not None:
                    writer.write(f)
        writer.release()
        print(f"  video saved       → {video_path}  ({args.video_fps:.1f} fps)")
    else:
        _OUT_PATH = _REPO_ROOT / "models" / "data_examples" / "image-sam2wp.jpg"
        vis = draw_results(init_frame, init_objects)
        cv2.imwrite(str(_OUT_PATH), vis)
        print(f"\n  annotated image (SAM3 init) saved → {_OUT_PATH}")
