"""
Florence-2 inference wrapper for open-vocabulary object detection.
Encapsulates model loading, BGR preprocessing, and text-prompted detection.

The model lives in dep/Florence-2-base/ (already downloaded, no network needed).
Load with trust_remote_code=True because it contains local custom modelling code.

Supported task prompts:
    <OD>                        -- generic detection, no text input required
    <OPEN_VOCABULARY_DETECTION> -- text-driven detection, requires text_input
                                    e.g. "left robot arm. right robot arm. cup."
    <CAPTION_TO_PHRASE_GROUNDING> -- phrase grounding from a caption sentence

Note on num_beams:
    The model default is num_beams=3 (beam search).
    For real-time use set num_beams=1 (greedy) to cut latency by ~2-3x with
    minimal accuracy loss on detection tasks.

Benchmark:
    python -m robot_motion_interface.utils.florence2_utils
    python -m robot_motion_interface.utils.florence2_utils --task OD --num_beams 1
    python -m robot_motion_interface.utils.florence2_utils --task OD --num_beams 3

Run:
    python -m robot_motion_interface.utils.florence2_utils \
        --task PHRASE_GROUND \
        --text_input "sofa. book." \
        --model_dir dep/Florence-2-base-ft \
        --device cuda \
        --num_beams 1 \
        --max_new_tokens 256 \
        --compile \
        --warmup 5 \
        --iters 200
        
    python -m robot_motion_interface.utils.florence2_utils \
        --frames_dir models/data_examples/hand_setup_frames \
        --task PHRASE_GROUND \
        --text_input "robot arm. water cup." \
        --model_dir dep/Florence-2-base \
        --num_beams 1 \
        --max_new_tokens 256
        

Test:
    # baseline
    python -m robot_motion_interface.utils.florence2_utils

    # less max_new_tokens
    python -m robot_motion_interface.utils.florence2_utils --max_new_tokens 256

    # torch.compile
    python -m robot_motion_interface.utils.florence2_utils --compile --warmup 5

    # combined
    python -m robot_motion_interface.utils.florence2_utils --max_new_tokens 256 --compile --warmup 5

"""



import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image as _PILImage

from transformers import AutoProcessor, AutoModelForCausalLM


# Task prompt tokens
TASK_OD             = "<OD>"
TASK_OVD            = "<OPEN_VOCABULARY_DETECTION>"
TASK_PHRASE_GROUND  = "<CAPTION_TO_PHRASE_GROUNDING>"


class Florence2Inference:
    """
    Wraps Florence-2 model loading and text-prompted bounding box detection.

    Accepts BGR frames (as returned by RealSense / cv2), converts to RGB internally,
    runs the Florence-2 generative decode, and returns pixel-space bounding boxes.

    Usage:
        f2 = Florence2Inference(
            model_dir="dep/Florence-2-base",
            device="cuda",
            num_beams=1,   # greedy for real-time; 3 for higher accuracy
        )

        boxes, labels = f2.infer(
            bgr_uint8,
            task=TASK_OVD,
            text_input="left robot arm. right robot arm. cup.",
        )
        # boxes:  (N, 4) float32  [x1, y1, x2, y2] in pixel coordinates
        # labels: list[str] of length N
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        num_beams: int = 1,
        max_new_tokens: int = 1024,
        compile: bool = False,
    ):
        """
        Args:
            model_dir:      Path to the Florence-2 model directory
                            (contains config.json, model.safetensors, etc.).
                            Use dep/Florence-2-base — already downloaded, no network needed.
            device:         torch device string ("cuda" or "cpu").
            num_beams:      Beam search width. 1 = greedy (fastest, ~2-3x faster than 3).
                            3 = matches the paper default (higher accuracy, slower).
            max_new_tokens: Upper bound on generated tokens per forward pass.
                            Reduce to 256 for detection-only tasks (typical output is
                            ~50-100 tokens for a few objects).
            compile:        Apply torch.compile(mode="reduce-overhead") to the model.
                            Adds ~30s one-time compilation cost; saves ~10-20% per call.
        """
        self.device = torch.device(device)
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self._dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
        if compile:
            print("Compiling the model with torch.compile(mode='reduce-overhead') ...")
            self.model = torch.compile(self.model, fullgraph=True, mode='reduce-overhead')

    @torch.inference_mode()
    def infer(
        self,
        bgr: np.ndarray,
        task: str = TASK_OVD,
        text_input: Optional[str] = None,
    ) -> Tuple[np.ndarray, list]:
        """
        Run Florence-2 detection on a single BGR frame.

        Args:
            bgr:        (H, W, 3) uint8 BGR frame from RealSense / cv2.
            task:       Task prompt token. Use TASK_OVD with text_input for
                        text-driven detection, or TASK_OD for fully automatic detection.
            text_input: Text prompt appended after the task token.
                        Required for TASK_OVD and TASK_PHRASE_GROUND.
                        Format for detection: "object one. object two. object three."

        Returns:
            boxes:  (N, 4) float32 bounding boxes [x1, y1, x2, y2] in pixel coords.
                    Returns empty array (0, 4) if nothing is detected.
            labels: list[str] of length N, one label per detected box.
        """
        h, w = bgr.shape[:2]
        pil_img = _PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        prompt = task if text_input is None else task + text_input
        inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt")

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device, self._dtype),
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            early_stopping=False,
            do_sample=False,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=task, image_size=(w, h)
        )
        result = list(parsed.values())[0] if parsed else {}

        boxes_raw = result.get("bboxes", [])
        # <OD> uses "labels"; <OPEN_VOCABULARY_DETECTION> uses "bboxes_labels"
        labels = result.get("bboxes_labels", result.get("labels", []))

        if not boxes_raw:
            return np.zeros((0, 4), dtype=np.float32), []

        boxes = np.array(boxes_raw, dtype=np.float32)   # (N, 4) xyxy
        return boxes, list(labels)


# ── Benchmark entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    # florence2_utils.py is 5 levels deep:
    # libs/robot_motion_interface/src/robot_motion_interface/utils/florence2_utils.py
    _REPO_ROOT  = Path(__file__).resolve().parents[5]
    _MODEL_DIR  = _REPO_ROOT / "dep" / "Florence-2-base"

    _TASK_MAP = {
        "OD":  TASK_OD,
        "OVD": TASK_OVD,
        "PHRASE_GROUND": TASK_PHRASE_GROUND,
    }

    parser = argparse.ArgumentParser(description="Benchmark Florence-2 throughput")
    parser.add_argument("--model_dir",  type=str,  default=None,
                        help=f"Florence-2 model dir (default: dep/Florence-2-base)")
    parser.add_argument("--task",       type=str,  default="OVD",
                        choices=["OD", "OVD", "PHRASE_GROUND"],
                        help="Task prompt: OD (no text) or OVD (text-driven, default)")
    parser.add_argument("--text_input", type=str,
                        default="left robot arm. right robot arm. cup.",
                        help="Text prompt for OVD task")
    parser.add_argument("--device",     type=str,  default="cuda")
    parser.add_argument("--num_beams",      type=int,  default=1,
                        help="Beam search width (1=greedy/fastest, 3=paper default)")
    parser.add_argument("--max_new_tokens", type=int,  default=1024,
                        help="Max tokens to generate (try 256 for detection-only tasks)")
    parser.add_argument("--compile",        action="store_true",
                        help="Apply torch.compile(mode='reduce-overhead') (~30s warmup)")
    parser.add_argument("--width",          type=int,  default=640)
    parser.add_argument("--height",         type=int,  default=480)
    parser.add_argument("--warmup",         type=int,  default=2)
    parser.add_argument("--frames_dir",     type=str,  default=None,
                        help="Folder of frames to run sequentially (jpg/png, sorted by name).")
    parser.add_argument("--out_dir",        type=str,  default=None,
                        help="Save annotated frames here (only with --frames_dir).")
    parser.add_argument("--video_fps",      type=float, default=30.0,
                        help="Playback fps of the output video (default: 30).")
    args = parser.parse_args()

    model_dir  = args.model_dir if args.model_dir is not None else str(_MODEL_DIR)
    task_token = _TASK_MAP[args.task]
    text_input = args.text_input if args.task in ("OVD", "PHRASE_GROUND") else None

    # ── Load frame list ────────────────────────────────────────────────────────
    if args.frames_dir is not None:
        frames_dir = Path(args.frames_dir)
        frame_paths = sorted(
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not frame_paths:
            raise FileNotFoundError(f"No jpg/png frames found in: {frames_dir}")
        out_dir = Path(args.out_dir) if args.out_dir else frames_dir.parent / (frames_dir.name + "_annotated")
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "folder"
    else:
        _IMG_PATH = _REPO_ROOT / "models" / "data_examples" / "image.jpg"
        _OUT_PATH = _REPO_ROOT / "models" / "data_examples" / "image-florence-2.jpg"
        frame_paths = [_IMG_PATH]
        mode = "single"

    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        gpu_info = (f"{torch.cuda.get_device_name(0)}  "
                    f"({_p.total_memory // (1024**2)} MiB)  "
                    f"sm_{_p.major}{_p.minor}  CUDA {torch.version.cuda}")
    else:
        gpu_info = "N/A (CPU only)"

    print("=" * 60)
    print("  Florence-2 Inference Frequency Benchmark")
    print("=" * 60)
    print(f"  GPU        : {gpu_info}")
    print(f"  model_dir  : {model_dir}")
    print(f"  task       : {task_token}")
    if text_input is not None:
        print(f"  text_input : {text_input}")
    print(f"  device     : {args.device}  num_beams={args.num_beams}  max_new_tokens={args.max_new_tokens}  compile={args.compile}")
    print(f"  frame size : {args.width}x{args.height}")
    if mode == "folder":
        print(f"  frames_dir : {args.frames_dir}  ({len(frame_paths)} frames)")
        print(f"  out_dir    : {out_dir}")
    print(f"  warm-up    : {args.warmup}")
    print("=" * 60)

    t_load = time.time()
    f2 = Florence2Inference(
        model_dir=model_dir,
        device=args.device,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        compile=args.compile,
    )
    print(f"Model loaded in {time.time() - t_load:.2f}s\n")

    # ── Warm-up ────────────────────────────────────────────────────────────────
    warmup_frame = cv2.imread(str(frame_paths[0]))
    warmup_frame = cv2.resize(warmup_frame, (args.width, args.height))
    print(f"Running {args.warmup} warm-up frame(s) ...")
    for _ in range(args.warmup):
        f2.infer(warmup_frame, task=task_token, text_input=text_input)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up done.\n")

    # ── Timed inference over all frames ────────────────────────────────────────
    latencies = []
    for i, fpath in enumerate(frame_paths):
        frame = cv2.imread(str(fpath))
        if frame is None:
            print(f"  [skip] cannot read {fpath.name}")
            continue
        frame = cv2.resize(frame, (args.width, args.height))

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        boxes, labels = f2.infer(frame, task=task_token, text_input=text_input)
        if args.device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  frame {i+1:>4d}/{len(frame_paths)}: {ms:>7.2f} ms   detections={len(boxes)}")

        if mode == "folder":
            vis = frame.copy()
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)
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
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(_OUT_PATH), vis)
        print(f"\n  annotated image saved → {_OUT_PATH}")
    else:
        print(f"\n  annotated frames saved → {out_dir}")

        # Write video with same name as out_dir
        import subprocess
        video_path = out_dir.parent / (out_dir.name + ".mp4")
        video_fps  = args.video_fps
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(video_fps),
            "-pattern_type", "glob", "-i", str(out_dir / "*.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path),
        ], check=True)
        print(f"  video saved            → {video_path}  ({video_fps:.1f} fps)")
