"""
Ultralytics YOLO26 wrapper for bbox-based depth masking.

Detects a target class (default: "bottle") on the RGB frame, then zeros out
every depth pixel outside the detected bbox. Used as a lightweight semantic
prior for sim2real depth pipelines — only the bottle's depth survives, the
background and (most of) the manipulator hand are masked out.

Model weights are stored under `models/ultralytics/` relative to the repo
root and auto-downloaded by ultralytics on first use.

Benchmark (synthetic frames):
    python -m robot_motion_interface.utils.ultralytics_utils

Run on a recorded RealSense session:
    python -m robot_motion_interface.utils.ultralytics_utils \
        --frames_dir models/data_examples/realsense/rs_record_distant_20260316_053733 \
        --variant n
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Repo root: this file lives at libs/robot_motion_interface/src/robot_motion_interface/utils/
_REPO_ROOT = Path(__file__).resolve().parents[5]
_WEIGHTS_DIR = _REPO_ROOT / "models" / "ultralytics"

# YOLO26 weight filenames per size variant. Ultralytics downloads from GitHub
# releases automatically when YOLO(<absolute_path>) is called with a missing file.
_VARIANTS = {
    "n": "yolo26n.pt",   # Nano    fastest
    "s": "yolo26s.pt",   # Small
    "m": "yolo26m.pt",   # Medium
    "l": "yolo26l.pt",   # Large
    "x": "yolo26x.pt",   # XLarge  most accurate
}

_SEG_VARIANTS = {
    "n": "yolo26n-seg.pt",   # Nano    fastest
    "s": "yolo26s-seg.pt",   # Small
    "m": "yolo26m-seg.pt",   # Medium
    "l": "yolo26l-seg.pt",   # Large
    "x": "yolo26x-seg.pt",   # XLarge  most accurate
}


class YOLOBboxDepthMasker:
    """
    Single-step pipeline:  (bgr, depth) ──► masked_depth, bbox

    Outside-bbox depth pixels are replaced with `fill_value`. If no detection
    passes `conf_threshold`, the whole depth frame is zeroed and bbox=None.
    """

    def __init__(
        self,
        variant: str = "n",
        target_class: str = "bottle",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        padding_ratio: float = 0.10,
        fill_value: float = 0.0,
    ):
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {list(_VARIANTS)}, got {variant!r}")

        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.padding_ratio = float(padding_ratio)
        self.fill_value = float(fill_value)

        # Make sure weights dir exists; ultralytics downloads here on miss.
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = _WEIGHTS_DIR / _VARIANTS[variant]
        self.model = YOLO(str(weights_path)).to(self.device)

        # Resolve target class name to its integer id used by this model.
        name_to_id = {n.lower(): i for i, n in self.model.names.items()}
        key = target_class.lower()
        if key not in name_to_id:
            sample = list(self.model.names.values())[:10]
            raise ValueError(f"target_class {target_class!r} not in model classes; first 10: {sample}")
        self.target_class_id = name_to_id[key]
        self.target_class_name = key

    def predict_bbox(
        self, bgr: np.ndarray, shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Args:
            bgr:   (H, W, 3) uint8 BGR colour frame.
            shape: target image shape as (height, width), normally depth.shape[:2].

        Returns:
            (x1, y1, x2, y2) integer pixel coords, padded and clipped;
            None when no detection passes the confidence threshold.
        """
        h, w = int(shape[0]), int(shape[1])

        results = self.model.track(
            bgr,
            conf=self.conf_threshold,
            classes=[self.target_class_id],
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return None

        confs = boxes.conf.detach().cpu().numpy()
        x1, y1, x2, y2 = boxes.xyxy[int(np.argmax(confs))].detach().cpu().numpy()

        pad_x = (x2 - x1) * self.padding_ratio
        pad_y = (y2 - y1) * self.padding_ratio
        x1 = int(max(0, np.floor(x1 - pad_x)))
        y1 = int(max(0, np.floor(y1 - pad_y)))
        x2 = int(min(w, np.ceil(x2 + pad_x)))
        y2 = int(min(h, np.ceil(y2 + pad_y)))
        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def mask_depth_with_bbox(
        self, depth: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Return depth with only the bbox region kept."""
        masked = np.full_like(depth, self.fill_value)
        if bbox is None:
            return masked

        x1, y1, x2, y2 = bbox
        masked[y1:y2, x1:x2] = depth[y1:y2, x1:x2]
        return masked

    def mask_depth(
        self, bgr: np.ndarray, depth: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Args:
            bgr:   (H, W, 3) uint8  BGR colour frame
            depth: (H, W) or (H, W, 1)  aligned depth frame (any numeric dtype)

        Returns:
            masked_depth: same shape & dtype as `depth`; pixels outside the bbox
                          replaced with `fill_value`. All-zeros when no detection.
            bbox:         (x1, y1, x2, y2) integer pixel coords, padded and clipped;
                          None when no detection passes the confidence threshold.
        """
        bbox = self.predict_bbox(bgr, depth.shape[:2])
        return self.mask_depth_with_bbox(depth, bbox), bbox


class YOLOSegDepthMasker:
    """
    Single-step pipeline:  (bgr, depth) -> masked_depth, bbox

    Uses a YOLO26 segmentation model and keeps only depth pixels inside the
    top-1 target-class instance mask. If no detection passes `conf_threshold`,
    the whole depth frame is filled with `fill_value` and bbox=None.
    """

    def __init__(
        self,
        variant: str = "n",
        target_class: str = "bottle",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        mask_threshold: float = 0.5,
        mask_dilation: int = 0,
        fill_value: float = 0.0,
    ):
        if variant not in _SEG_VARIANTS:
            raise ValueError(f"variant must be one of {list(_SEG_VARIANTS)}, got {variant!r}")
        if mask_dilation < 0:
            raise ValueError(f"mask_dilation must be >= 0, got {mask_dilation}")

        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.mask_threshold = float(mask_threshold)
        self.mask_dilation = int(mask_dilation)
        self.fill_value = float(fill_value)

        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = _WEIGHTS_DIR / _SEG_VARIANTS[variant]
        self.model = YOLO(str(weights_path)).to(self.device)

        name_to_id = {n.lower(): i for i, n in self.model.names.items()}
        key = target_class.lower()
        if key not in name_to_id:
            sample = list(self.model.names.values())[:10]
            raise ValueError(f"target_class {target_class!r} not in model classes; first 10: {sample}")
        self.target_class_id = name_to_id[key]
        self.target_class_name = key

    def predict_mask(self, bgr: np.ndarray, shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Returns a boolean instance mask and tight bbox for the top target-class detection.

        Args:
            bgr:   (H, W, 3) uint8 BGR colour frame.
            shape: target mask shape as (height, width), normally depth.shape[:2].
        """
        h, w = int(shape[0]), int(shape[1])
        results = self.model.track(
            bgr,
            conf=self.conf_threshold,
            classes=[self.target_class_id],
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        if not results:
            return None, None

        result = results[0]
        boxes = result.boxes
        masks = result.masks
        if boxes is None or masks is None or len(boxes) == 0:
            return None, None

        confs = boxes.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(confs))
        mask = masks.data[best_idx].detach().cpu().numpy() > self.mask_threshold
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        if self.mask_dilation > 0:
            kernel = np.ones((self.mask_dilation, self.mask_dilation), dtype=np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        ys, xs = np.where(mask)
        if xs.size == 0:
            return None, None

        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        return mask, bbox

    def mask_depth_with_mask(self, depth: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Return depth with only the boolean segmentation mask region kept."""
        masked = np.full_like(depth, self.fill_value)
        if mask is None:
            return masked

        if depth.ndim == 3:
            masked[mask, :] = depth[mask, :]
        else:
            masked[mask] = depth[mask]
        return masked

    def mask_depth(
        self, bgr: np.ndarray, depth: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Args:
            bgr:   (H, W, 3) uint8 BGR colour frame.
            depth: (H, W) or (H, W, 1) aligned depth frame (any numeric dtype).

        Returns:
            masked_depth: same shape & dtype as `depth`; pixels outside the
                          segmentation mask are replaced with `fill_value`.
            bbox:         tight bbox around the selected segmentation mask;
                          None when no valid target-class mask is found.
        """
        mask, bbox = self.predict_mask(bgr, depth.shape[:2])
        return self.mask_depth_with_mask(depth, mask), bbox


# ────────────────────────────────────────────────────────────────────────────
# Test entry: run on a recorded RealSense session (color/ + depth/), write
# a sibling folder with [bbox overlay | masked depth colormap] images.
#
"""
python -m robot_motion_interface.utils.ultralytics_utils \
    --frames_dir /workspace/data/rs_record_20260513_183410 \
    --out_dir /workspace/data/rs_record_20260513_183410_yolo_seg_masked_gray \
    --masker seg \
    --vis_mode gray \
    --variant n \
    --target_class bottle \
    --conf 0.05 \
    --mask_threshold 0.5 \
    --mask_dilation 3 \
    --depth_scale 0.001 \
    --clip 0.1 1.1
"""
#
# Output (default): <frames_dir>_yolo_masked/000000.jpg, 000001.jpg, ...
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="YOLO26 depth-mask test on a color/+depth/ folder")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Folder containing color/ and depth/ subfolders")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output folder. Defaults depend on --masker/--vis_mode.")
    parser.add_argument("--masker", choices=["bbox", "seg"], default="bbox",
                        help="Depth mask source: bbox uses YOLO detection boxes, seg uses YOLO segmentation masks.")
    parser.add_argument("--vis_mode", choices=["panel", "gray"], default="panel",
                        help="panel writes [RGB overlay | depth colormap]; gray writes normalized masked depth only.")
    parser.add_argument("--variant", choices=list(_VARIANTS), default="n")
    parser.add_argument("--target_class", type=str, default="bottle")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--padding_ratio", type=float, default=0.10,
                        help="BBox masker padding ratio; ignored by --masker seg.")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="Segmentation mask threshold; used only by --masker seg.")
    parser.add_argument("--mask_dilation", type=int, default=0,
                        help="Segmentation mask dilation kernel size; used only by --masker seg.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--depth_scale", type=float, default=0.001,
                        help="Metres per raw depth unit (RealSense D-series default: 0.001)")
    parser.add_argument("--clip", type=float, nargs=2, default=[0.1, 1.1],
                        metavar=("NEAR", "FAR"),
                        help="Fixed [near, far] (metres) for depth normalisation")
    args = parser.parse_args()

    # Resolve paths (relative paths are relative to repo root).
    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_absolute():
        frames_dir = _REPO_ROOT / frames_dir
    color_dir, depth_dir = frames_dir / "color", frames_dir / "depth"
    if not color_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(f"Expected color/ and depth/ inside {frames_dir}")
    color_paths = sorted(p for p in color_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not color_paths:
        raise FileNotFoundError(f"No color frames in {color_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.masker == "bbox" and args.vis_mode == "panel":
        out_dir = frames_dir / "yolo_masked"
    else:
        out_dir = frames_dir.parent / f"{frames_dir.name}_yolo_{args.masker}_masked_{args.vis_mode}"
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("YOLO26 depth-mask folder test")
    print(f"  frames_dir   : {frames_dir}  ({len(color_paths)} frames)")
    print(f"  out_dir      : {out_dir}")
    print(f"  masker       : {args.masker}")
    print(f"  vis_mode     : {args.vis_mode}")
    print(f"  variant      : yolo26{args.variant}{'-seg' if args.masker == 'seg' else ''}")
    print(f"  target_class : {args.target_class!r}  conf>={args.conf}")
    if args.masker == "bbox":
        print(f"  padding      : {args.padding_ratio*100:.0f}%")
    else:
        print(f"  mask         : threshold={args.mask_threshold}  dilation={args.mask_dilation}")

    if args.masker == "bbox":
        masker = YOLOBboxDepthMasker(
            variant=args.variant,
            target_class=args.target_class,
            device=args.device,
            conf_threshold=args.conf,
            padding_ratio=args.padding_ratio,
        )
    else:
        masker = YOLOSegDepthMasker(
            variant=args.variant,
            target_class=args.target_class,
            device=args.device,
            conf_threshold=args.conf,
            mask_threshold=args.mask_threshold,
            mask_dilation=args.mask_dilation,
        )

    latencies: list[float] = []
    n_hit = n_miss = 0
    near, far = float(args.clip[0]), float(args.clip[1])
    for color_p in color_paths:
        depth_p = depth_dir / (color_p.stem + ".png")
        bgr = cv2.imread(str(color_p))
        depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None:
            print(f"  [skip] cannot read pair {color_p.stem}")
            continue

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        masked_depth, bbox = masker.mask_depth(bgr, depth)
        if args.device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if bbox is None:
            n_miss += 1
        else:
            n_hit += 1

        # Masked depth -> metres -> fixed [near, far] clip & normalize.
        d_m = masked_depth.astype(np.float32) * args.depth_scale
        invalid = d_m <= 0
        norm = np.clip((d_m - near) / (far - near + 1e-6), 0, 1)
        depth_u8 = (norm * 255).astype(np.uint8)
        depth_u8[invalid] = 0
        if depth_u8.ndim == 3 and depth_u8.shape[-1] == 1:
            depth_u8 = depth_u8[..., 0]

        if args.vis_mode == "gray":
            cv2.imwrite(str(out_dir / f"{color_p.stem}.png"), depth_u8)
            continue

        # Panel mode: left BGR overlay + right masked depth colormap.
        overlay = bgr.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, args.target_class, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        depth_color[depth_u8 == 0] = 0
        if depth_color.shape[:2] != overlay.shape[:2]:
            depth_color = cv2.resize(depth_color, (overlay.shape[1], overlay.shape[0]))

        stacked = np.hstack([overlay, depth_color])
        cv2.putText(stacked, f"{color_p.stem}  {args.masker}={'hit' if bbox else 'MISS'}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"{color_p.stem}.jpg"), stacked, [cv2.IMWRITE_JPEG_QUALITY, 90])

    total = n_hit + n_miss
    if total:
        print(f"\n  hits: {n_hit}  misses: {n_miss}  ({100.0 * n_hit / total:.1f}% detection rate)")
    if latencies:
        arr = np.asarray(latencies)
        print(f"  latency mean={arr.mean():.2f} ms  p50={np.percentile(arr,50):.2f}  p95={np.percentile(arr,95):.2f}  max={arr.max():.2f}")
    print(f"  written      : {out_dir}")
