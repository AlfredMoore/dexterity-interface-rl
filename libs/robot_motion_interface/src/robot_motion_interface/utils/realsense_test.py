import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
# from vision_msgs.msg import Detection3D
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

_rs_fps = config['rs_fps']
_infer_rate = config['infer_rate']

# RealSense
rs_pipeline = rs.pipeline()
rs_config = rs.config()
# self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self._rs_fps)
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, _rs_fps)
rs_profile = rs_pipeline.start(rs_config)
# self.rs_align = rs.align(rs.stream.color)  # align depth -> color frame

rs_device = rs_profile.get_device()
print(
    f"RealSense initialized: "
    f"  device={rs_device.get_info(rs.camera_info.name)}  "
    f"  serial={rs_device.get_info(rs.camera_info.serial_number)}  "
    f"  color=640x480@{_rs_fps}fps"
        )

try:
    while True:
        frames = rs.composite_frame(rs.frame())
        if not rs_pipeline.poll_for_frames(frames):
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

        