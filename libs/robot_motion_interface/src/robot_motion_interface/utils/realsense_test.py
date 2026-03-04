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

DEFAULT_CONFIG_PATH = RMI_ROOT / "config" / "rl_policy_node_config.yaml"


config_path: str = str(DEFAULT_CONFIG_PATH.resolve())
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at: {config_path}")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

_infer_rate = config['infer_rate']
_rs_config = config['realsense']
_rs_fps = _rs_config['rs_fps']
_sensor_settings = _rs_config['sensor_settings']
_c_intrinsics = _rs_config['color_intrinsics']
_d_intrinsics = _rs_config['depth_intrinsics']

# RealSense
rs_pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(
    rs.stream.color, 
    _c_intrinsics['width'], 
    _c_intrinsics['height'], 
    rs.format.bgr8, 
    _rs_fps
)
# rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, _rs_fps)
rs_profile = rs_pipeline.start(rs_config)
# rs_align = rs.align(rs.stream.color)  # align depth -> color frame

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

rs_device = rs_profile.get_device()
print(
    f"RealSense initialized:\n"
    f"  device={rs_device.get_info(rs.camera_info.name)}\n"
    f"  serial={rs_device.get_info(rs.camera_info.serial_number)}\n"
    f"  color=640x480@{_rs_fps}fps\n"
        )

try:
    while True:
        frames =rs_pipeline.poll_for_frames()
        if not frames:
            continue
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        cv2.imshow('preview', img)
        if cv2.waitKey(1) == ord('q'):
            break
        time.sleep(0.001)
finally:
    rs_pipeline.stop()
    cv2.destroyAllWindows()

        