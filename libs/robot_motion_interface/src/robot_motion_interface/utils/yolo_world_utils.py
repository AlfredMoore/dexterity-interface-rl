"""
YOLO-World inference wrapper for open-vocabulary object detection.

YOLO-World uses a CLIP-based text encoder + YOLO backbone for real-time
open-vocabulary detection. Text embeddings are pre-encoded once via
reparameterize(), then each frame only runs the image backbone — much
faster than Florence-2 (generative) or Grounding DINO (two-stage).

Available model variants (download checkpoint to models/yolo_world/):
    yolo_world_v2_s  -- Small,   ~26ms  (~38 Hz)
    yolo_world_v2_m  -- Medium,  ~32ms  (~31 Hz)
    yolo_world_v2_l  -- Large,   ~40ms  (~25 Hz)  [default]
    yolo_world_v2_x  -- XLarge,  ~55ms  (~18 Hz)

Checkpoints: https://github.com/AILab-CVC/YOLO-World/blob/master/docs/model_zoo.md

Benchmark:
    python -m robot_motion_interface.utils.yolo_world_utils
    python -m robot_motion_interface.utils.yolo_world_utils --variant s

Run on folder:
    python -m robot_motion_interface.utils.yolo_world_utils \
        --frames_dir models/data_examples/hand_setup_frames \
        --classes "robot arm. water cup." \
        --variant l
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

# YOLO-World uses mmdet / mmengine — add the repo to sys.path so that
# configs and custom modules are discoverable without pip install.
_YOLO_WORLD_ROOT = Path(__file__).resolve().parents[5] / "dep" / "YOLO-World"
if str(_YOLO_WORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLO_WORLD_ROOT))

from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

# ── Model variant → (config filename, default ckpt filename) ──────────────────
_VARIANTS = {
    "s": (
        "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
        "yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth",
    ),
    "m": (
        "yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
        "yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth",
    ),
    "l": (
        "yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
        "yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth",
    ),
    "x": (
        "yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
        "yolo_world_v2_x_obj365v1_goldg_pretrain-d7ec5e20.pth",
    ),
}
_DEFAULT_VARIANT = "l"


def _parse_classes(text: str) -> List[List[str]]:
    """Convert 'cup. robot arm.' → [['cup'], ['robot arm'], [' ']]."""
    classes = [c.strip().rstrip(".") for c in text.split(".") if c.strip()]
    return [[c] for c in classes] + [[" "]]  # background token required


class YOLOWorldInference:
    """
    Real-time YOLO-World detector.

    Usage:
        detector = YOLOWorldInference(
            config_path="dep/YOLO-World/configs/pretrain/yolo_world_v2_l_...",
            ckpt_path="models/yolo_world/yolo_world_v2_l_....pth",
            classes="robot arm. cup.",
        )
        boxes, labels, scores = detector.infer(bgr_frame)
        # boxes  : np.ndarray (N, 4)  xyxy float32
        # labels : list[str]          class name per detection
        # scores : np.ndarray (N,)    confidence float32
    """

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        classes: str = "object.",
        device: str = "cuda",
        score_thr: float = 0.3,
        max_dets: int = 100,
    ) -> None:
        self.device = device
        self.score_thr = score_thr
        self.max_dets = max_dets

        cfg = Config.fromfile(config_path)
        cfg.load_from = ckpt_path
        self.model = init_detector(cfg, checkpoint=ckpt_path, device=device)
        self.model.eval()

        # Build mmdet test pipeline (image loading → augmentation → collate)
        pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        self.pipeline = Compose(pipeline_cfg)

        # Pre-encode text embeddings once — subsequent infer() skips text encoder
        self.set_classes(classes)

    def set_classes(self, classes: str) -> None:
        """Update detection classes and re-encode text embeddings."""
        self._texts = _parse_classes(classes)
        self._class_names = [t[0] for t in self._texts[:-1]]  # exclude background
        self.model.reparameterize(self._texts)

    @torch.inference_mode()
    def infer(self, bgr: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Run detection on a single BGR frame.

        Args:
            bgr: np.ndarray of shape (H, W, 3), dtype uint8, BGR channel order.

        Returns:
            boxes  : (N, 4) float32 xyxy pixel coordinates
            labels : list[str] of length N
            scores : (N,) float32 confidence values
        """
        # mmdet pipeline expects RGB
        rgb = bgr[:, :, ::-1]
        data_info = {"img": rgb, "img_id": 0, "texts": self._texts}
        data_info = self.pipeline(data_info)
        data_batch = {
            "inputs": data_info["inputs"].unsqueeze(0),
            "data_samples": [data_info["data_samples"]],
        }

        output = self.model.test_step(data_batch)[0]
        pred = output.pred_instances

        # Score threshold
        keep = pred.scores.float() > self.score_thr
        pred = pred[keep]

        # Top-k
        if len(pred.scores) > self.max_dets:
            idx = pred.scores.float().topk(self.max_dets)[1]
            pred = pred[idx]

        pred = pred.cpu().numpy()
        if len(pred["bboxes"]) == 0:
            return np.zeros((0, 4), dtype=np.float32), [], np.zeros((0,), dtype=np.float32)

        boxes  = pred["bboxes"].astype(np.float32)
        scores = pred["scores"].astype(np.float32)
        labels = [self._class_names[int(lbl)] for lbl in pred["labels"]]
        return boxes, labels, scores


# ── Benchmark entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    _REPO_ROOT  = Path(__file__).resolve().parents[5]
    _CONFIG_DIR = _REPO_ROOT / "dep" / "YOLO-World" / "configs" / "pretrain"
    _CKPT_DIR   = _REPO_ROOT / "models" / "yolo_world"

    parser = argparse.ArgumentParser(description="Benchmark YOLO-World throughput")
    parser.add_argument("--variant",    type=str, default=_DEFAULT_VARIANT, choices=list(_VARIANTS),
                        help="Model variant: s / m / l (default) / x")
    parser.add_argument("--config",     type=str, default=None,
                        help="Override config path (default: inferred from --variant)")
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="Override checkpoint path (default: inferred from --variant)")
    parser.add_argument("--classes",    type=str, default="robot arm. cup.",
                        help="Dot-separated class list, e.g. 'cup. robot arm.'")
    parser.add_argument("--score_thr",  type=float, default=0.3)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--width",      type=int, default=640)
    parser.add_argument("--height",     type=int, default=480)
    parser.add_argument("--warmup",     type=int, default=3)
    parser.add_argument("--iters",      type=int, default=50,
                        help="Number of timed iterations (single-image mode only)")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Folder of frames to run sequentially (jpg/png, sorted by name).")
    parser.add_argument("--out_dir",    type=str, default=None,
                        help="Save annotated frames here (only with --frames_dir).")
    parser.add_argument("--video_fps",  type=float, default=30.0,
                        help="Playback fps of the output video (default: 30).")
    args = parser.parse_args()

    cfg_name, ckpt_name = _VARIANTS[args.variant]
    config_path = args.config or str(_CONFIG_DIR / cfg_name)
    ckpt_path   = args.ckpt   or str(_CKPT_DIR   / ckpt_name)

    # ── Frame list ─────────────────────────────────────────────────────────────
    if args.frames_dir is not None:
        frames_dir  = Path(args.frames_dir)
        frame_paths = sorted(p for p in frames_dir.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not frame_paths:
            raise FileNotFoundError(f"No jpg/png frames found in: {frames_dir}")
        out_dir = Path(args.out_dir) if args.out_dir else \
                  frames_dir.parent / (frames_dir.name + "_annotated")
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "folder"
    else:
        _IMG_PATH   = _REPO_ROOT / "models" / "data_examples" / "image.jpg"
        _OUT_PATH   = _REPO_ROOT / "models" / "data_examples" / "image-yolo-world.jpg"
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
    print("  YOLO-World Inference Frequency Benchmark")
    print("=" * 60)
    print(f"  GPU        : {gpu_info}")
    print(f"  variant    : yolo_world_v2_{args.variant}")
    print(f"  config     : {config_path}")
    print(f"  ckpt       : {ckpt_path}")
    print(f"  classes    : {args.classes}")
    print(f"  score_thr  : {args.score_thr}  |  device: {args.device}")
    print(f"  frame size : {args.width}x{args.height}")
    if mode == "folder":
        print(f"  frames_dir : {args.frames_dir}  ({len(frame_paths)} frames)")
        print(f"  out_dir    : {out_dir}")
    print(f"  warm-up    : {args.warmup}")
    print("=" * 60)

    t_load = time.time()
    detector = YOLOWorldInference(
        config_path=config_path,
        ckpt_path=ckpt_path,
        classes=args.classes,
        device=args.device,
        score_thr=args.score_thr,
    )
    print(f"Model loaded in {time.time() - t_load:.2f}s\n")

    # ── Warm-up ────────────────────────────────────────────────────────────────
    warmup_frame = cv2.imread(str(frame_paths[0]))
    warmup_frame = cv2.resize(warmup_frame, (args.width, args.height))
    print(f"Running {args.warmup} warm-up frame(s) ...")
    for _ in range(args.warmup):
        detector.infer(warmup_frame)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    # ── Timed inference ────────────────────────────────────────────────────────
    n_iters = len(frame_paths) if mode == "folder" else args.iters
    latencies: list[float] = []
    boxes, labels, scores = np.zeros((0, 4), dtype=np.float32), [], np.zeros(0)

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
        boxes, labels, scores = detector.infer(frame)
        if args.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  {'frame' if mode == 'folder' else 'iter'} "
              f"{i+1:>4d}/{n_iters}: {ms:>7.2f} ms   detections={len(boxes)}")

        if mode == "folder":
            vis = frame.copy()
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{label} {score:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
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
        vis = warmup_frame.copy()
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{label} {score:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(_OUT_PATH), vis)
        print(f"\n  annotated image saved → {_OUT_PATH}")
    else:
        print(f"\n  annotated frames saved → {out_dir}")

        video_path = out_dir.parent / (out_dir.name + ".mp4")
        video_fps  = args.video_fps
        sample_bgr = cv2.imread(str(sorted(out_dir.iterdir())[0]))
        h_v, w_v   = sample_bgr.shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (w_v, h_v),
        )
        for p in sorted(out_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                f = cv2.imread(str(p))
                if f is not None:
                    writer.write(f)
        writer.release()
        print(f"  video saved            → {video_path}  ({video_fps:.1f} fps)")
