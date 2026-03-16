"""
Ultralytics YOLO-World + ByteTrack wrapper for open-vocabulary bbox detection
and persistent object tracking.

Designed to feed SAM2 with per-object bbox prompts:
    tracker = UltralyticsTracker(classes=["robot arm", "cup"])
    detections = tracker.infer(bgr_frame)
    for det in detections:
        boxes, scores, logits = sam2.infer(bgr_frame, box=det.box)

Available YOLO-World v2 variants (auto-downloaded by ultralytics on first use):
    yolov8s-worldv2.pt  -- Small,  ~15ms  (~67 Hz)  [default]
    yolov8m-worldv2.pt  -- Medium, ~22ms  (~45 Hz)
    yolov8l-worldv2.pt  -- Large,  ~35ms  (~29 Hz)
    yolov8x-worldv2.pt  -- XLarge, ~60ms  (~17 Hz)

Benchmark:
    python -m robot_motion_interface.utils.ultralytics_utils
    python -m robot_motion_interface.utils.ultralytics_utils --variant m

Run on folder:
    python -m robot_motion_interface.utils.ultralytics_utils \
        --frames_dir models/data_examples/hand_setup_frames \
        --classes "robot arm. cup."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    """Single tracked object detection."""
    track_id: int
    label: str
    box: np.ndarray   # (4,) float32 xyxy
    score: float


class UltralyticsTracker:
    """
    YOLO-World open-vocabulary detector + ByteTrack persistent tracker.

    Text classes are set once; each infer() call runs detection + tracking
    on a single BGR frame and returns a list of Detection objects with
    stable track IDs across frames.

    Usage:
        tracker = UltralyticsTracker(
            model_path="yolov8s-worldv2.pt",
            classes=["robot arm", "cup"],
            device="cuda",
        )
        detections = tracker.infer(bgr_frame)
        for det in detections:
            print(det.track_id, det.label, det.box, det.score)
    """

    def __init__(
        self,
        model_path: str = "yolov8s-worldv2.pt",
        classes: Optional[List[str]] = None,
        device: str = "cuda",
        conf: float = 0.3,
        iou: float = 0.5,
    ) -> None:
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to(device)
        self._classes = classes or ["robot arm", "cup"]
        self.model.set_classes(self._classes)
        self.conf = conf
        self.iou  = iou

    def set_classes(self, classes: List[str]) -> None:
        """Update detection classes (re-encodes text embeddings)."""
        self._classes = classes
        self.model.set_classes(classes)

    def infer(self, bgr: np.ndarray) -> List[Detection]:
        """
        Run detection + tracking on a single BGR frame.

        Args:
            bgr: (H, W, 3) uint8 BGR frame.

        Returns:
            List of Detection, one per tracked object.
            Empty list if nothing detected or tracker has no IDs yet.
        """
        results = self.model.track(
            bgr,
            persist=True,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
        )
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return []

        boxes      = r.boxes.xyxy.cpu().numpy().astype(np.float32)   # (N, 4)
        track_ids  = r.boxes.id.int().cpu().numpy()                  # (N,)
        cls_ids    = r.boxes.cls.int().cpu().numpy()                  # (N,)
        scores     = r.boxes.conf.cpu().numpy().astype(np.float32)   # (N,)

        detections = []
        for box, tid, cid, score in zip(boxes, track_ids, cls_ids, scores):
            label = self._classes[int(cid)] if int(cid) < len(self._classes) else "unknown"
            detections.append(Detection(
                track_id=int(tid),
                label=label,
                box=box,
                score=float(score),
            ))
        return detections

    def reset(self) -> None:
        """Reset tracker state (clears all track IDs)."""
        self.model.predictor = None


def _parse_classes(text: str) -> List[str]:
    """'robot arm. cup.' → ['robot arm', 'cup']"""
    return [c.strip().rstrip(".") for c in text.split(".") if c.strip()]


def _draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    vis = frame.copy()
    palette = [
        (0,  80, 255),
        (255, 80,   0),
        (0,  200,   0),
        (200,   0, 200),
        (0,  200, 200),
    ]
    for det in detections:
        color = palette[det.track_id % len(palette)]
        x1, y1, x2, y2 = det.box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{det.label}#{det.track_id} {det.score:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


# ── Benchmark entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time
    import torch

    _REPO_ROOT = Path(__file__).resolve().parents[5]

    _VARIANTS = {
        "s": "yolov8s-worldv2.pt",
        "m": "yolov8m-worldv2.pt",
        "l": "yolov8l-worldv2.pt",
        "x": "yolov8x-worldv2.pt",
    }

    parser = argparse.ArgumentParser(description="Benchmark YOLO-World + ByteTrack throughput")
    parser.add_argument("--variant",    type=str, default="s", choices=list(_VARIANTS),
                        help="Model variant: s (default) / m / l / x")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path (e.g. local .pt file)")
    parser.add_argument("--classes",    type=str, default="robot arm. cup.",
                        help="Dot-separated class list")
    parser.add_argument("--conf",       type=float, default=0.3)
    parser.add_argument("--iou",        type=float, default=0.5)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--width",      type=int, default=640)
    parser.add_argument("--height",     type=int, default=480)
    parser.add_argument("--warmup",     type=int, default=3)
    parser.add_argument("--iters",      type=int, default=50,
                        help="Timed iterations (single-image mode only)")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Folder of frames to run sequentially (jpg/png, sorted by name).")
    parser.add_argument("--out_dir",    type=str, default=None,
                        help="Save annotated frames here (only with --frames_dir).")
    parser.add_argument("--video_fps",  type=float, default=30.0,
                        help="Playback fps of the output video (default: 30).")
    args = parser.parse_args()

    model_path = args.model_path or _VARIANTS[args.variant]
    classes    = _parse_classes(args.classes)

    # ── Frame list ─────────────────────────────────────────────────────────────
    if args.frames_dir is not None:
        frames_dir  = Path(args.frames_dir)
        frame_paths = sorted(p for p in frames_dir.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not frame_paths:
            raise FileNotFoundError(f"No jpg/png frames in: {frames_dir}")
        out_dir = Path(args.out_dir) if args.out_dir else \
                  frames_dir.parent / (frames_dir.name + "_annotated")
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "folder"
    else:
        _IMG_PATH   = _REPO_ROOT / "models" / "data_examples" / "image.jpg"
        _OUT_PATH   = _REPO_ROOT / "models" / "data_examples" / "image-ultralytics.jpg"
        frame_paths = [_IMG_PATH]
        mode        = "single"

    # ── GPU info ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        gpu_info = (f"{torch.cuda.get_device_name(0)}  "
                    f"({_p.total_memory // (1024**2)} MiB)  "
                    f"sm_{_p.major}{_p.minor}  CUDA {torch.version.cuda}")
    else:
        gpu_info = "N/A (CPU only)"

    print("=" * 60)
    print("  YOLO-World v2 + ByteTrack Benchmark")
    print("=" * 60)
    print(f"  GPU        : {gpu_info}")
    print(f"  model      : {model_path}")
    print(f"  classes    : {classes}")
    print(f"  conf={args.conf}  iou={args.iou}  device={args.device}")
    print(f"  frame size : {args.width}x{args.height}")
    if mode == "folder":
        print(f"  frames_dir : {args.frames_dir}  ({len(frame_paths)} frames)")
        print(f"  out_dir    : {out_dir}")
    print(f"  warm-up    : {args.warmup}")
    print("=" * 60)

    t_load = time.time()
    tracker = UltralyticsTracker(
        model_path=model_path,
        classes=classes,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )
    print(f"Model loaded in {time.time() - t_load:.2f}s\n")

    # ── Warm-up ────────────────────────────────────────────────────────────────
    warmup_frame = cv2.imread(str(frame_paths[0]))
    warmup_frame = cv2.resize(warmup_frame, (args.width, args.height))
    print(f"Running {args.warmup} warm-up frame(s) ...")
    for _ in range(args.warmup):
        tracker.infer(warmup_frame)
    tracker.reset()   # clear track IDs accumulated during warm-up
    print("Warm-up done.\n")

    # ── Timed inference ────────────────────────────────────────────────────────
    n_iters   = len(frame_paths) if mode == "folder" else args.iters
    latencies: list[float] = []
    last_dets: List[Detection] = []

    for i in range(n_iters):
        fpath = frame_paths[i % len(frame_paths)]
        frame = cv2.imread(str(fpath))
        if frame is None:
            print(f"  [skip] cannot read {fpath.name}")
            continue
        frame = cv2.resize(frame, (args.width, args.height))

        t0 = time.perf_counter()
        dets = tracker.infer(frame)
        ms   = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        last_dets = dets

        label = "frame" if mode == "folder" else "iter"
        det_str = "  ".join(f"{d.label}#{d.track_id}" for d in dets) or "none"
        print(f"  {label} {i+1:>4d}/{n_iters}: {ms:>7.2f} ms   [{det_str}]")

        if mode == "folder":
            vis = _draw_detections(frame, dets)
            cv2.imwrite(str(out_dir / fpath.name), vis)

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

    if mode == "single":
        vis = _draw_detections(warmup_frame, last_dets)
        cv2.imwrite(str(_OUT_PATH), vis)
        print(f"\n  annotated image saved → {_OUT_PATH}")
    else:
        print(f"\n  annotated frames saved → {out_dir}")
        import subprocess
        video_path = out_dir.parent / (out_dir.name + ".mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(args.video_fps),
            "-pattern_type", "glob", "-i", str(out_dir / "*.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path),
        ], check=True)
        print(f"  video saved            → {video_path}  ({args.video_fps:.1f} fps)")
