"""
Video frame sampler.

Usage:
    python video2img.py input.mov --start 5 --end 20 --fps 30 --out frames/
    python video2img.py input.mp4 --start 0 --end 10 --out frames/
"""

import argparse
import os
import cv2


def sample_video(video_path: str, out_dir: str, start: float, end: float, fps: float) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    print(f"Video: {src_fps:.2f} fps  |  duration: {duration:.2f}s  |  frames: {total_frames}")

    if end < 0:
        end = duration
    if start < 0 or start >= end:
        raise ValueError(f"Invalid range [{start}, {end})")

    os.makedirs(out_dir, exist_ok=True)

    interval_sec = 1.0 / fps
    next_sample_time = start
    saved = 0

    start_frame = int(start * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time > end:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if current_time >= next_sample_time:
            out_path = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            next_sample_time += interval_sec

    cap.release()
    print(f"Saved {saved} frames → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample frames from a video segment.")
    parser.add_argument("video", help="Path to input video (.mp4, .mov, etc.)")
    parser.add_argument("--start", type=float, default=0.0,          help="Start time in seconds (default: 0)")
    parser.add_argument("--end",   type=float, default=-1.0,         help="End time in seconds (default: end of video)")
    parser.add_argument("--fps",   type=float, default=60.0,         help="Sampling frequency in Hz (default: 60)")
    parser.add_argument("--out",   type=str,   default="sampled_frames", help="Output folder (default: sampled_frames/)")
    args = parser.parse_args()

    sample_video(args.video, args.out, args.start, args.end, args.fps)
