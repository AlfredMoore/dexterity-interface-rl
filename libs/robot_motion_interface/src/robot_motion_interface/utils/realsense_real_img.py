"""
Capture RealSense color images one frame at a time.

Usage:
    python -m robot_motion_interface.utils.realsense_real_img /path/to/output_dir
    python -m robot_motion_interface.utils.realsense_real_img my_run

Controls:
    Enter   capture and save the current color frame
    q       quit and write metadata.json

Output layout:
    <output_dir>/
        000000.jpg
        000001.jpg
        ...
        metadata.json
"""

import argparse
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


# ---------------------------------------------------------------------------
# Resolve paths (same pattern as realsense_record.py)
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
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a dict: {config_path}")
    return config


def apply_sensor_settings(profile: rs.pipeline_profile, sensor_settings: dict) -> None:
    """Apply exposure / gain settings, matching realsense_record.py."""
    if not sensor_settings:
        return
    try:
        sensors = profile.get_device().query_sensors()
    except Exception:
        return

    auto_exposure = sensor_settings.get("auto_exposure", False)
    exposure = sensor_settings.get("exposure", 350)
    gain = sensor_settings.get("gain", 16)

    for sensor in sensors:
        if auto_exposure is not None and sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
        if auto_exposure is False:
            if exposure is not None and sensor.supports(rs.option.exposure):
                sensor.set_option(rs.option.exposure, float(exposure))
            if gain is not None and sensor.supports(rs.option.gain):
                sensor.set_option(rs.option.gain, float(gain))


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (DATA_ROOT / path).resolve()


def read_input_loop(commands: Queue) -> None:
    while True:
        try:
            cmd = input("capture> ").strip().lower()
        except EOFError:
            commands.put("q")
            return
        commands.put(cmd)
        if cmd == "q":
            return


def save_color_image(
    output_dir: Path,
    frame_idx: int,
    color: np.ndarray,
    t_capture: float,
) -> dict:
    image_name = f"{frame_idx:06d}.jpg"
    image_path = output_dir / image_name
    ok = cv2.imwrite(str(image_path), color, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to write image: {image_path}")
    print(f"[saved] {image_path}")
    return {
        "frame": frame_idx,
        "file": image_name,
        "t_capture": t_capture,
    }


def write_metadata(
    output_dir: Path,
    config_path: str,
    rs_cfg: dict,
    fps: int,
    depth_scale: float,
    device_name: str,
    serial_number: str,
    frames: list[dict],
) -> Path:
    metadata = {
        "config_path": config_path,
        "device": device_name,
        "serial": serial_number,
        "depth_scale": depth_scale,
        "color_intrinsics": rs_cfg["color_intrinsics"],
        "depth_intrinsics": rs_cfg["depth_intrinsics"],
        "T_color_depth": rs_cfg.get("T_color_depth"),
        "fps": fps,
        "total_frames": len(frames),
        "frames": frames,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Capture RealSense color images on Enter")
    parser.add_argument(
        "output_dir",
        help="Required output directory. Relative paths are resolved under repo data/.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.resolve()),
        help="Path to realsense_config.yaml",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)
    ensure_output_dir(output_dir)

    config_path = str(Path(args.config).expanduser().resolve())
    config = load_config(config_path)
    rs_cfg = config["realsense"]
    fps = rs_cfg["rs_fps"]
    c_intr = rs_cfg["color_intrinsics"]
    d_intr = rs_cfg["depth_intrinsics"]
    sens_set = rs_cfg.get("sensor_settings", {})

    pipeline = rs.pipeline()
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
    apply_sensor_settings(profile, sens_set)

    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    device = profile.get_device()
    device_name = device.get_info(rs.camera_info.name)
    serial_number = device.get_info(rs.camera_info.serial_number)

    print(
        "RealSense single-image capture started\n"
        f"  device : {device_name}\n"
        f"  serial : {serial_number}\n"
        f"  color  : {c_intr['width']}x{c_intr['height']}@{fps}fps\n"
        f"  output : {output_dir}\n"
        "Live image is shown in the OpenCV window.\n"
        "Press Enter in this terminal to capture, input 'q' to quit."
    )

    frames = []
    frame_idx = 0
    meta_path = output_dir / "metadata.json"
    commands = Queue()
    input_thread = threading.Thread(target=read_input_loop, args=(commands,), daemon=True)
    input_thread.start()

    try:
        cv2.namedWindow("RealSense color", cv2.WINDOW_AUTOSIZE)
        latest_color = None
        latest_capture_t = None

        while True:
            should_quit = False
            while True:
                try:
                    cmd = commands.get_nowait()
                except Empty:
                    break

                if cmd == "q":
                    should_quit = True
                    break
                if cmd != "":
                    print("Input must be empty Enter to capture, or 'q' to quit.")
                    continue
                if latest_color is None or latest_capture_t is None:
                    print("[warn] no color frame available yet; cannot save")
                    continue

                frames.append(save_color_image(output_dir, frame_idx, latest_color, latest_capture_t))
                frame_idx += 1

            if should_quit:
                break

            try:
                rs_frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                print("[warn] wait_for_frames timed out, retrying...")
                continue

            color_frame = rs_frames.get_color_frame()
            if not color_frame:
                continue

            latest_capture_t = time.time()
            latest_color = np.asanyarray(color_frame.get_data()).copy()
            cv2.imshow("RealSense color", latest_color)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        meta_path = write_metadata(
            output_dir,
            config_path,
            rs_cfg,
            fps,
            depth_scale,
            device_name,
            serial_number,
            frames,
        )

    print(f"\nCapture complete: {len(frames)} images saved to {output_dir}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
