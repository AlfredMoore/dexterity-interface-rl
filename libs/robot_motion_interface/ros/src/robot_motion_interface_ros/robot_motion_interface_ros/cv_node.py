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
from robot_motion_interface.utils.promptda_utils import PromptDAInference
JS_QOS = HIGH_PERF_QOS
BBOX_QOS = HIGH_PERF_QOS
T_JS_QOS = HIGH_RELIA_QOS

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT     = Path(spec.origin).parent.parent.parent   # libs/robot_motion_interface/
PROJECT_ROOT = RMI_ROOT.parent.parent                   # dexterity-interface-rl/
MODEL_ROOT = PROJECT_ROOT / "models"

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
        self.rs_config.enable_stream(
            rs.stream.depth,
            _d_intrinsics['width'],
            _d_intrinsics['height'],
            rs.format.z16,
            _rs_fps
        )
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        self.rs_align = rs.align(rs.stream.color)  # align depth -> color frame
        self._apply_sensor_settings(_sensor_settings)
        try:
            depth_sensor = self.rs_profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
        except Exception:
            self._depth_scale = 0.001
        self._depth_filters = self.build_rs_depth_filters()

        rs_device = self.rs_profile.get_device()
        self.get_logger().info(
            f"RealSense initialized: "
            f"  device={rs_device.get_info(rs.camera_info.name)}  "
            f"  serial={rs_device.get_info(rs.camera_info.serial_number)}  "
            f"  color={_c_intrinsics['width']}x{_c_intrinsics['height']}@{_rs_fps}fps  "
            f"  depth={_d_intrinsics['width']}x{_d_intrinsics['height']}@{_rs_fps}fps  "
            f"  depth_scale={self._depth_scale}"
        )
        # Image capture thread — stores (color_bgr, depth_u16) atomically
        self._latest_color: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._img_lock = threading.Lock()
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="rs_capture", daemon=True
        )
        self._capture_thread.start()

        # CV Model ##############################################
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pda_ckpt = str((MODEL_ROOT / config["promptda_ckpt"]).resolve())
        pda_encoder = config.get("promptda_encoder", "vits")
        self.get_logger().info(f"Loading PromptDA ({pda_encoder}) from {pda_ckpt} ...")
        self.pda = PromptDAInference(
            ckpt_path=pda_ckpt,
            encoder=pda_encoder,
            device=self.device,
            depth_scale=self._depth_scale,
        )
        self.get_logger().info("PromptDA loaded.")

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

            try:
                frames = self.rs_align.process(frames)
            except Exception:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Apply decimation + hole_filling filters
            depth_frame = self.apply_depth_filters(depth_frame, self._depth_filters)

            color = np.asanyarray(color_frame.get_data())        # HxWx3 uint8 BGR
            depth = np.asanyarray(depth_frame.get_data())        # HxW uint16 (raw units)

            with self._img_lock:
                self._latest_color = color
                self._latest_depth = depth
            time.sleep(0.001)
            

    # ── Inference timer callback ───────────────────────────────────────────────

    def _infer_callback(self) -> None:
        with self._img_lock:
            color = self._latest_color
            depth = self._latest_depth
        if color is None or depth is None:
            return

        try:
            metric_depth = self.pda.infer(color, depth)
            # metric_depth: [H', W'] float32 tensor in metres, on GPU

            # TODO: use metric_depth for object pose estimation and publish Detection3D

            # ── Preview ───────────────────────────────────────────────────────
            orig_h, orig_w = color.shape[:2]

            def _to_colormap(depth_f32: np.ndarray) -> np.ndarray:
                """float32 depth → BGR uint8 colormap, resized to (orig_w, orig_h)."""
                d = cv2.resize(depth_f32, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                lo, hi = np.percentile(d[d > 0], (2, 98)) if np.any(d > 0) else (0, 1)
                u8 = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
                return cv2.applyColorMap((u8 * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

            def _label(img: np.ndarray, text: str) -> np.ndarray:
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 0), 3, cv2.LINE_AA)   # black outline
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 1, cv2.LINE_AA)  # white text
                return img

            color_vis  = _label(color.copy(), "RGB")
            raw_vis    = _label(_to_colormap(depth.astype(np.float32) * self._depth_scale), "Raw Depth")
            metric_vis = _label(_to_colormap(metric_depth.squeeze().cpu().numpy()), "Metric Depth")
            preview = np.concatenate([color_vis, raw_vis, metric_vis], axis=1)
            cv2.imshow('RGB | Raw Depth | Metric Depth', preview)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f"CV Error: {e}")

    def __del__(self):
        self._capture_running = False
        self.rs_pipeline.stop()

    
    def build_rs_depth_filters(self):
        """
        Build the recommended pyrealsense2 post-processing filter chain for PromptDA.

        Applied filters (in order):
        1. decimation  - halve resolution (640→320); PromptDA only needs low-res prompt depth.
        2. hole_filling - fill zero/invalid pixels so min/max normalisation is not skewed.

        Excluded:
        - spatial_filter:  extra CPU cost for marginal benefit
        - temporal_filter: introduces 1-2 frame latency

        Returns:
            list of rs2 filter objects; apply sequentially to a depth frame.
        """

        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)  # 640x480 → 320x240

        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, 2)      # fill with far neighbour

        return [decimation, hole_filling]


    def apply_depth_filters(self, depth_frame, filters: list):
        """Apply a list of rs2 filter objects to a depth frame."""
        for f in filters:
            depth_frame = f.process(depth_frame)
        return depth_frame


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CVPerceptionNode())
    rclpy.shutdown()
