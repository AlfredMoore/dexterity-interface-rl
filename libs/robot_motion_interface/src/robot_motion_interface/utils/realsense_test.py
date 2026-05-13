"""
python -m robot_motion_interface.utils.realsense_test
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import torch
import numpy as np
import threading
import pyrealsense2 as rs
import cv2
from pathlib import Path
import importlib.util
import yaml
import os
import time
from typing import Dict

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT     = Path(spec.origin).parent.parent.parent   # libs/robot_motion_interface/
PROJECT_ROOT = RMI_ROOT.parent.parent                   # dexterity-interface-rl/

DEFAULT_CONFIG_PATH = RMI_ROOT / "config" / "realsense_config.yaml"


def _require_keys(config: dict, config_path: str) -> None:
    required_keys = (
        ("realsense",),
        ("realsense", "rs_fps"),
        ("realsense", "sensor_settings"),
        ("realsense", "sensor_settings", "auto_exposure"),
        ("realsense", "sensor_settings", "exposure"),
        ("realsense", "sensor_settings", "gain"),
        ("realsense", "color_intrinsics"),
        ("realsense", "color_intrinsics", "width"),
        ("realsense", "color_intrinsics", "height"),
        ("realsense", "depth_intrinsics"),
        ("realsense", "depth_intrinsics", "width"),
        ("realsense", "depth_intrinsics", "height"),
    )
    missing = []
    for key_path in required_keys:
        node = config
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                missing.append(".".join(key_path))
                break
            node = node[key]
    if missing:
        raise KeyError(f"Missing required key(s) in {config_path}: {', '.join(missing)}")


config_path: str = str(DEFAULT_CONFIG_PATH.resolve())
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at: {config_path}")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
if not isinstance(config, dict):
    raise ValueError(f"Config root must be a dict: {config_path}")
_require_keys(config, config_path)

_rs_config = config['realsense']
_rs_fps = _rs_config['rs_fps']
_sensor_settings = _rs_config['sensor_settings']
_c_intrinsics = _rs_config['color_intrinsics']
_d_intrinsics = _rs_config['depth_intrinsics']

print(
    "RealSense config loaded and validated:\n"
    f"  path={config_path}\n"
    f"  rs_fps={_rs_fps}\n"
    f"  color={_c_intrinsics['width']}x{_c_intrinsics['height']}\n"
    f"  depth={_d_intrinsics['width']}x{_d_intrinsics['height']}"
)

# RealSense
rs_pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(
    rs.stream.color,
    _c_intrinsics['width'],
    _c_intrinsics['height'],
    rs.format.bgr8,
    _rs_fps,
)
rs_config.enable_stream(
    rs.stream.depth,
    _d_intrinsics['width'],
    _d_intrinsics['height'],
    rs.format.z16,
    _rs_fps,
)
rs_profile = rs_pipeline.start(rs_config)
rs_align = rs.align(rs.stream.color)

decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 2)  # 640x480 → 320x240

hole_filling = rs.hole_filling_filter()
hole_filling.set_option(rs.option.holes_fill, 2)      # fill with far neighbour

def reset_camera():
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        print(f"restarting: {dev.get_info(rs.camera_info.name)}")
        dev.hardware_reset()

def _apply_sensor_settings(profile: rs.pipeline_profile) -> None:
    if not _sensor_settings:
        return

    try:
        device = profile.get_device()
        sensors = device.query_sensors()
    except Exception:
        return

    auto_exposure = _sensor_settings.get("auto_exposure", False)
    exposure = _sensor_settings.get("exposure", 350)
    gain = _sensor_settings.get("gain", 16)

    for sensor in sensors:
        if auto_exposure is not None and sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)

        # Manual settings only take effect when auto-exposure is disabled
        if auto_exposure is False:
            if exposure is not None and sensor.supports(rs.option.exposure):
                sensor.set_option(rs.option.exposure, float(exposure))
            if gain is not None and sensor.supports(rs.option.gain):
                sensor.set_option(rs.option.gain, float(gain))

        sensor.get_stream_profiles()

def sensor_profiles(rs_profile: rs.pipeline_profile):
    try:
        device = rs_profile.get_device()
        sensors = device.query_sensors()
    except Exception:
        return
    
    for sensor in sensors:
        print(f"\n[Sensor]: {sensor.get_info(rs.camera_info.name)}")
        
        profiles = sensor.get_stream_profiles()

        supported_configs = set()
        for p in profiles:
            if p.is_video_stream_profile():
                v_p = p.as_video_stream_profile()

                supported_configs.add((v_p.width(), v_p.height(), v_p.fps(), v_p.format().name))
    
        for w, h, fps, fmt in sorted(list(supported_configs)):
            print(f"  res: {w:4d}x{h:4d} | FPS: {fps:3d} | fmt: {fmt}")

sensor_profiles(rs_profile)

_apply_sensor_settings(rs_profile)
print("Sensor settings applied")

try:
    _depth_scale = float(rs_profile.get_device().first_depth_sensor().get_depth_scale())
except Exception:
    _depth_scale = 0.001

rs_device = rs_profile.get_device()
print(
    f"RealSense initialized:\n"
    f"  device={rs_device.get_info(rs.camera_info.name)}\n"
    f"  serial={rs_device.get_info(rs.camera_info.serial_number)}\n"
    f"  color={_c_intrinsics['width']}x{_c_intrinsics['height']}@{_rs_fps}fps\n"
    f"  depth={_d_intrinsics['width']}x{_d_intrinsics['height']}@{_rs_fps}fps\n"
    f"  depth_scale={_depth_scale}\n"
)


def _depth_to_colormap(depth_u16: np.ndarray) -> np.ndarray:
    depth_m = depth_u16.astype(np.float32) * _depth_scale
    valid = depth_m > 0
    lo, hi = (np.percentile(depth_m[valid], (2, 98)) if valid.any() else (0.0, 1.0))
    norm = np.clip((depth_m - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


try:
    while True:
        try:
            frames = rs_pipeline.wait_for_frames(timeout_ms=2000)
            if not frames:
                time.sleep(0.001)
                continue
        except RuntimeError:
            reset_camera()
            print("[warn] wait_for_frames timed out, resetting camera and retrying...")
            time.sleep(1)
            continue

        frames = decimation.process(frames).as_frameset()
        frames = hole_filling.process(frames).as_frameset()

        aligned_frames = rs_align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        color = np.array(color_frame.get_data())
        depth = np.array(depth_frame.get_data())
        depth_vis = _depth_to_colormap(depth)
        h, w = color.shape[:2]
        if depth_vis.shape[:2] != (h, w): depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('preview', np.concatenate([color, depth_vis], axis=1))
        if cv2.waitKey(1) == ord('q'):
            break
        time.sleep(0.001)
finally:
    rs_pipeline.stop()
    cv2.destroyAllWindows()

        
