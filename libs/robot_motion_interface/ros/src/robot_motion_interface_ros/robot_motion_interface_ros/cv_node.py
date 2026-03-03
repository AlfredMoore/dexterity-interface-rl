import threading
import time

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3D
import torch
import numpy as np
import pyrealsense2 as rs
import yaml
from pathlib import Path
import importlib.util
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter

# --- QoS Config: low latency (Best Effort) ---

HIGH_PERF_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

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

        self._rs_fps = config['rs_fps']
        self._infer_rate = config['infer_rate']

        # RealSense
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        # self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self._rs_fps)
        self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self._rs_fps)
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        # self.rs_align = rs.align(rs.stream.color)  # align depth -> color frame

        rs_device = self.rs_profile.get_device()
        self.get_logger().info(
            f"RealSense initialized: "
            f"  device={rs_device.get_info(rs.camera_info.name)}  "
            f"  serial={rs_device.get_info(rs.camera_info.serial_number)}  "
            f"  color=640x480@{self._rs_fps}fps"
        )

        # Model (GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_model = torch.jit.load(
            config["cv_model_path"], map_location=self.device
        ).eval()

        # Publisher
        self.object_detection_pub = self.create_publisher(Detection3D, '/object_detection', HIGH_PERF_QOS)

        # Shared latest frame (capture thread writes, infer timer reads)
        self._latest_img: np.ndarray | None = None
        self._img_lock = threading.Lock()

        # Capture thread — runs at camera fps, always holds the newest frame
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="rs_capture", daemon=True
        )
        self._capture_thread.start()

        # Inference timer — decoupled rate
        self.create_timer(1.0 / self._infer_rate, self._infer_callback)

        self.get_logger().info(
            f"CVPerceptionNode ready: "
            f"  device={self.device}  "
            f"  capture={self._rs_fps}Hz  infer={self._infer_rate}Hz  "
            f"  topic=/object_detection"
        )

    # ── Capture thread ─────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        while self._capture_running:
            frames = rs.composite_frame(rs.frame())
            if not self.rs_pipeline.poll_for_frames(frames):
                continue
            color_frame = frames.get_color_frame()
            # depth_frame = frames.get_depth_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
            with self._img_lock:
                self._latest_img = img
            time.sleep(0.001)
            

    # ── Inference timer callback ───────────────────────────────────────────────

    def _infer_callback(self) -> None:
        with self._img_lock:
            img = self._latest_img
        if img is None:
            return

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
