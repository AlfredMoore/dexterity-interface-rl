import threading
import time

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3D
import torch
import numpy as np
import pyrealsense2 as rs
import cv2
import yaml
from pathlib import Path
import importlib.util
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter

# --- QoS Config: low latency (Best Effort) ---
from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
JS_QOS = HIGH_PERF_QOS
BBOX_QOS = HIGH_PERF_QOS
T_JS_QOS = HIGH_RELIA_QOS

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT     = Path(spec.origin).parent.parent.parent   # libs/robot_motion_interface/
PROJECT_ROOT = RMI_ROOT.parent.parent                   # dexterity-interface-rl/

DEFAULT_CONFIG_PATH = RMI_ROOT / "config" / "rl_policy_node_config.yaml"

class CVPerceptionNode(Node):
    def __init__(self):
        super().__init__('cv_perception_node')

        self.declare_parameter('config_path', str(DEFAULT_CONFIG_PATH.resolve()))
        config_path: str = self.get_parameter('config_path').value
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.get_logger().info(f"Loaded config from: {config_path}")

        # RealSense ##############################################
        # Depth settings are commented
        _infer_rate = config['infer_rate']
        _rs_config = config['realsense']
        _rs_fps = _rs_config['rs_fps']
        _sensor_settings = _rs_config['sensor_settings']
        _c_intrinsics = _rs_config['color_intrinsics']
        _d_intrinsics = _rs_config['depth_intrinsics']

        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color, 
            _c_intrinsics['width'], 
            _c_intrinsics['height'], 
            rs.format.bgr8, 
            _rs_fps
        )
        # self.rs_config.enable_stream(
        #     rs.stream.depth, 
        #     _d_intrinsics['width'], 
        #     _d_intrinsics['height'], 
        #     rs.format.z16, 
        #     _rs_fps
        # )
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        # self.rs_align = rs.align(rs.stream.color)  # align depth -> color frame
        self._apply_sensor_settings(_sensor_settings)
        # try:
        #     depth_sensor = self.rs_profile.get_device().first_depth_sensor()
        #     self._depth_scale = float(depth_sensor.get_depth_scale())
        # except Exception:
        #     self._depth_scale = 1.0

        rs_device = self.rs_profile.get_device()
        self.get_logger().info(
            f"RealSense initialized: "
            f"  device={rs_device.get_info(rs.camera_info.name)}  "
            f"  serial={rs_device.get_info(rs.camera_info.serial_number)}  "
            f"  color={_c_intrinsics['width']}x{_c_intrinsics['height']}@{_rs_fps}fps"
            # f"  depth={_d_intrinsics['width']}x{_d_intrinsics['height']}@{_rs_fps}fps"
        )
        # Image capture thread
        self._latest_color: np.ndarray | None = None
        self._img_lock = threading.Lock()
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="rs_capture", daemon=True
        )
        self._capture_thread.start()

        # CV Model ##############################################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: cv model is not ready
        # self.cv_model = torch.jit.load(
        #     config["cv_model_path"], map_location=self.device
        # ).eval()

        self.object_detection_pub = self.create_publisher(Detection3D, '/object_detection', BBOX_QOS)
        # Inference timer
        self.create_timer(1.0 / _infer_rate, self._infer_callback)
        self.get_logger().info(
            f"CVPerceptionNode ready: "
            f"  device={self.device}  "
            f"  infer={_infer_rate}Hz  "
            f"  topic=/object_detection"
        )

    # Helper functions
    def _apply_sensor_settings(self, sensor_settings) -> None:
        if not sensor_settings:
            return
        try:
            device = self.rs_profile.get_device()
            sensors = device.query_sensors()
        except Exception:
            return

        auto_exposure = sensor_settings.get("auto_exposure", False)
        exposure = sensor_settings.get("exposure", 350)
        gain = sensor_settings.get("gain", 16)

        for sensor in sensors:
            if auto_exposure is not None and sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)

            # Manual settings only take effect when auto-exposure is disabled
            if auto_exposure is False:
                if exposure is not None and sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, float(exposure))
                if gain is not None and sensor.supports(rs.option.gain):
                    sensor.set_option(rs.option.gain, float(gain))

    # ── Capture thread ─────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        while self._capture_running:
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue

            # if self.rs_align is not None:
            #     try:
            #         frames = self.rs_align.process(frames)
            #     except Exception:
            #         continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            # depth_frame = frames.get_depth_frame()
            # if not depth_frame:
            #     continue

            color = np.asanyarray(color_frame.get_data())   # rgb8
            # depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            # depth = depth_raw * self._depth_scale  # meters

            with self._img_lock:
                self._latest_color = color
            time.sleep(0.001)
            

    # ── Inference timer callback ───────────────────────────────────────────────

    def _infer_callback(self) -> None:
        with self._img_lock:
            color = self._latest_color
        if color is None:
            return

        # For debugging
        cv2.imshow('preview', color)
        cv2.waitKey(1)
        return  # TODO: cv model is not ready

        try:
            img_tensor = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float() / 255.0
            with torch.inference_mode():
                # TODO: we need object pose and size
                pose_tensor = self.cv_model(img_tensor.unsqueeze(0))
                pose_data = pose_tensor.squeeze().cpu().numpy()

            msg = Detection3D()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_optical_frame"
            msg.bbox.center.position.x, msg.bbox.center.position.y, msg.bbox.center.position.z = pose_data[:3]
            msg.bbox.size.x = float(pose_data[7])  # width
            msg.bbox.size.y = float(pose_data[8])  # height
            msg.bbox.size.z = ...  # TODO: fixed depth size for simplicity
            self.object_detection_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"CV Error: {e}")

    def __del__(self):
        self._capture_running = False
        self.rs_pipeline.stop()


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CVPerceptionNode())
    rclpy.shutdown()
