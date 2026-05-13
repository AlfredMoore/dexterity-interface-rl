"""
RealSense RGB-D recorder.

Records color (JPEG) and depth (16-bit PNG) frames to a timestamped output
directory, using the same pipeline configuration as cv_node.py.

Usage:
    python -m robot_motion_interface.utils.realsense_record
    python -m robot_motion_interface.utils.realsense_record --output /tmp/my_recording
    python -m robot_motion_interface.utils.realsense_record --output /tmp/my_recording --no-preview

Output layout:
    <output_dir>/
        color/
            000000.jpg
            000001.jpg
            ...
        depth/
            000000.png   (uint16, raw sensor units — multiply by depth_scale for metres)
            000001.png
            ...
        metadata.json   (intrinsics, depth_scale, per-frame timestamps)

Press 'q' (in the preview window) or Ctrl-C to stop.
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


# ---------------------------------------------------------------------------
# Resolve paths (same pattern as cv_node.py / realsense_test.py)
# ---------------------------------------------------------------------------

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface package")
RMI_ROOT = Path(spec.origin).parent.parent.parent   # libs/robot_motion_interface/
DEFAULT_CONFIG_PATH = RMI_ROOT / "config" / "realsense_config.yaml"
REPO_ROOT = RMI_ROOT.parent.parent   # root of the whole repository
DATA_ROOT = REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_sensor_settings(profile: rs.pipeline_profile, sensor_settings: dict) -> None:
    """Apply exposure / gain settings — identical logic to cv_node._apply_sensor_settings."""
    if not sensor_settings:
        return
    try:
        sensors = profile.get_device().query_sensors()
    except Exception:
        return

    auto_exposure = sensor_settings.get("auto_exposure", False)
    exposure      = sensor_settings.get("exposure", 350)
    gain          = sensor_settings.get("gain", 16)

    for sensor in sensors:
        if auto_exposure is not None and sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
        if auto_exposure is False:
            if exposure is not None and sensor.supports(rs.option.exposure):
                sensor.set_option(rs.option.exposure, float(exposure))
            if gain is not None and sensor.supports(rs.option.gain):
                sensor.set_option(rs.option.gain, float(gain))


def load_depth_clip(rs_cfg: dict) -> tuple[float, float]:
    clip = rs_cfg.get("clip")
    if not isinstance(clip, (list, tuple)) or len(clip) != 2:
        raise ValueError(f"realsense.clip must be [min_m, max_m], got: {clip}")

    clip_min, clip_max = float(clip[0]), float(clip[1])
    if clip_min >= clip_max:
        raise ValueError(f"realsense.clip min must be < max, got: {clip}")
    return clip_min, clip_max


def depth_to_colormap(
    depth_u16: np.ndarray,
    depth_scale: float,
    depth_clip: tuple[float, float],
) -> np.ndarray:
    """Convert raw uint16 depth to a false-colour BGR image for preview."""
    depth_m = depth_u16.astype(np.float32) * depth_scale
    clip_min, clip_max = depth_clip
    depth_m = np.clip(depth_m, clip_min, clip_max)
    norm = (depth_m - clip_min) / (clip_max - clip_min)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RealSense RGB-D recorder")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory. Defaults to data/rs_record_<timestamp>.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.resolve()),
        help="Path to realsense_config.yaml",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV preview window (useful when DISPLAY is unavailable).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Maximum recording duration in seconds (default: 10).",
    )
    args = parser.parse_args()

    # Output directory
    if args.output is None:
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = DATA_ROOT / f"rs_record_{run_ts}"
    else:
        out_dir = Path(args.output)

    color_dir = out_dir / "color"
    depth_dir = out_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config = load_config(args.config)
    rs_cfg   = config["realsense"]
    fps      = rs_cfg["rs_fps"]
    c_intr   = rs_cfg["color_intrinsics"]
    d_intr   = rs_cfg["depth_intrinsics"]
    sens_set = rs_cfg.get("sensor_settings", {})
    depth_clip = load_depth_clip(rs_cfg)

    show_preview = not args.no_preview and bool(os.environ.get("DISPLAY"))

    # Filters: decimation(2) + hole_filling(2), applied before align
    decimation_filter = rs.decimation_filter()
    decimation_filter.set_option(rs.option.filter_magnitude, 2)
    hole_filling_filter = rs.hole_filling_filter()
    hole_filling_filter.set_option(rs.option.holes_fill, 2)

    # Pipeline
    pipeline  = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.color,
        c_intr["width"], c_intr["height"],
        rs.format.bgr8, fps,
    )
    rs_config.enable_stream(
        rs.stream.depth,
        d_intr["width"], d_intr["height"],
        rs.format.z16, fps,
    )
    profile = pipeline.start(rs_config)
    align   = rs.align(rs.stream.color)   # align depth -> color frame

    apply_sensor_settings(profile, sens_set)

    try:
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    except Exception:
        depth_scale = 0.001

    device = profile.get_device()
    print(
        f"RealSense recording started\n"
        f"  device : {device.get_info(rs.camera_info.name)}\n"
        f"  serial : {device.get_info(rs.camera_info.serial_number)}\n"
        f"  color  : {c_intr['width']}x{c_intr['height']}@{fps}fps\n"
        f"  depth  : {d_intr['width']}x{d_intr['height']}@{fps}fps\n"
        f"  depth_scale : {depth_scale}\n"
        f"  depth_clip : [{depth_clip[0]}, {depth_clip[1]}] m\n"
        f"  output : {out_dir}\n"
        f"  preview: {'on' if show_preview else 'off'}\n"
        f"  duration: {args.duration}s max\n"
        f"Press Ctrl-C (or 'q' in preview) to stop."
    )

    input("\n##### Press Enter to start recording... #####\n")

    timestamps = []   # list of {"frame": int, "t_capture": float}
    frame_idx  = 0
    t_start    = time.time()

    try:
        while time.time() - t_start < args.duration:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                print("[warn] wait_for_frames timed out, retrying...")
                continue

            frames = decimation_filter.process(frames).as_frameset()
            frames = hole_filling_filter.process(frames).as_frameset()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            t_capture = time.time()
            color = np.asanyarray(color_frame.get_data())   # HxWx3 uint8 BGR
            depth = np.asanyarray(depth_frame.get_data())   # HxW  uint16

            # Save color as JPEG (lossless quality=95)
            cv2.imwrite(str(color_dir / f"{frame_idx:06d}.jpg"), color,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            # Save depth as 16-bit PNG (lossless, preserves raw units)
            cv2.imwrite(str(depth_dir / f"{frame_idx:06d}.png"), depth)

            timestamps.append({"frame": frame_idx, "t_capture": t_capture})

            if frame_idx % 30 == 0:
                print(f"[frame {frame_idx:6d}]  t={t_capture:.3f}")

            if show_preview:
                depth_vis = depth_to_colormap(depth, depth_scale, depth_clip)
                preview   = np.concatenate([color, depth_vis], axis=1)
                cv2.imshow("Recording: RGB | Depth  (q to stop)", preview)
                if cv2.waitKey(1) == ord("q"):
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pipeline.stop()
        if show_preview:
            cv2.destroyAllWindows()

        # Write metadata
        metadata = {
            "depth_scale": depth_scale,
            "color_intrinsics": c_intr,
            "depth_intrinsics": d_intr,
            "T_color_depth": rs_cfg.get("T_color_depth"),
            "depth_clip": [float(depth_clip[0]), float(depth_clip[1])],
            "fps": fps,
            "total_frames": frame_idx,
            "frames": timestamps,
        }
        meta_path = out_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nRecording complete: {frame_idx} frames saved to {out_dir}")
        print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
