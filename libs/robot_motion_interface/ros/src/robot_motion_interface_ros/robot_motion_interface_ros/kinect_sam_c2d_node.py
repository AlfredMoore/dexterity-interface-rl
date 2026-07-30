"""Azure Kinect native RGB/depth -> SAM3 -> native masked depth."""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import torch
import yaml
from pyk4a import (
    CalibrationType,
    ColorResolution,
    Config,
    DepthMode,
    FPS,
    ImageFormat,
    PyK4A,
)
from pyk4a.errors import K4ATimeoutException
from pyk4a.transformation import color_image_to_depth_camera
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float32

from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
from sam3_trt import Sam3TRT


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent

# Feature consumed by policy_state_estimator.py; order and units match the sim extero group.
STATE_ESTIMATOR_TOPIC = "/state_estimator/extero"
_EXTERO_NAMES = [
    "bottle_x", "bottle_y", "bottle_z",
    "cap_x", "cap_y", "cap_z",
    "body_r", "body_h", "cap_r", "cap_h",
]
# HARDCODING JAR GEOMETRY:
_TABLETOP_Z = 0.9369  # table top in the calibrated world frame (sim: 0.9144 + 0.045/2)
_JARS: dict[str, dict[str, float]] = {
    #                cap_r   cap_h   bot_r   bot_h
    "green_vita":   {"cap_r": 0.030, "cap_h": 0.020, "bot_r": 0.042, "bot_h": 0.165},
    "blue_peanut":  {"cap_r": 0.036, "cap_h": 0.020, "bot_r": 0.036, "bot_h": 0.100},
    "printed":      {"cap_r": 0.040, "cap_h": 0.030, "bot_r": 0.042, "bot_h": 0.120},
    "brown_peanut": {"cap_r": 0.048, "cap_h": 0.020, "bot_r": 0.050, "bot_h": 0.150},
    "black_peanut": {"cap_r": 0.033, "cap_h": 0.020, "bot_r": 0.043, "bot_h": 0.180},
}

def hardcode_pred(pred: list[float], jar_name: str) -> list[float]:
    """
    green_vita, blue_peanut, printed, brown_peanut, black_peanut
    """
    jar = _JARS[jar_name]
    bot_h, cap_h = jar["bot_h"], jar["cap_h"]
    pred[2] = _TABLETOP_Z + bot_h / 2.0          # bottle_z
    pred[5] = _TABLETOP_Z + bot_h + cap_h / 2.0  # cap_z
    pred[6:10] = [jar["bot_r"], bot_h, jar["cap_r"], cap_h]
    return pred


WORKSPACE_ROOT = Path("/workspace")

_QOS = {"best_effort": HIGH_PERF_QOS, "reliable": HIGH_RELIA_QOS}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _workspace_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else WORKSPACE_ROOT / path


class StateEstimator:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        input_hw: tuple[int, int] = (288, 320),
        near: float = 0.1,
        far: float = 1.1,
    ) -> None:
        self.device = torch.device(device)
        self.input_hw = input_hw
        self.near = near
        self.far = far
        self.net = torch.jit.load(str(model_path), map_location=self.device).eval()

    @torch.inference_mode()
    def infer(self, masked_depth: np.ndarray) -> torch.Tensor:
        height, width = self.input_hw
        depth = cv2.resize(masked_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depth = torch.from_numpy(depth).to(self.device, dtype=torch.float32)
        depth = depth.unsqueeze(0).unsqueeze(0).mul_(0.001)
        depth.clamp_(self.near, self.far).sub_(self.near).div_(self.far - self.near)

        _, pred = self.net(depth)
        return pred


class KinectSamC2DNode(Node):
    def __init__(self) -> None:
        super().__init__("kinect_sam_c2d_node")
        self._load_kinect_config()
        self._load_sam_config()

        self.sam = Sam3TRT(
            str(self.engine_path), str(self.processor_path), self.sam_device
        )
        self.estimator = StateEstimator(
            RMI_ROOT / "runtime" / "state_estimator" / "exported" / "depth_predictor.pt",
            device=self.sam_device,
        )
        self._setup_camera()
        self._setup_publishers()

        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="kinect_sam_c2d"
        )
        self._capture_thread.start()
        self.get_logger().info(
            f"KinectSamC2DNode ready: prompt={self.prompt!r}, "
            f"mask->{self.mask_topic}, masked_depth->{self.masked_depth_topic}"
        )

    def _load_kinect_config(self) -> None:
        kinect = _load_yaml(RMI_ROOT / "config" / "kinect_node.yaml")["kinect"]
        preset = kinect["c2d"]
        pub = kinect["publish"]

        self.device_id = int(kinect["device_id"])
        self.capture_timeout_ms = int(kinect["capture_timeout_ms"])
        self.color_resolution = str(preset["color_resolution"])
        self.color_format = str(preset["color_format"])
        self.depth_mode = str(preset["depth_mode"])
        self.camera_fps = str(preset["camera_fps"])
        self.synchronized_images_only = bool(preset["synchronized_images_only"])
        self.color_frame_id = str(preset["color_frame_id"])
        self.depth_frame_id = str(preset["depth_frame_id"])

        self.color_topic = str(pub["color_topic"])
        self.color_info_topic = str(pub["color_info_topic"])
        self.depth_topic = str(pub["depth_topic"])
        self.depth_info_topic = str(pub["depth_info_topic"])
        self.kinect_qos = _QOS[str(pub["qos"])]

    def _load_sam_config(self) -> None:
        sam = _load_yaml(RMI_ROOT / "config" / "sam_node.yaml")["sam3"]
        model, infer, ros = sam["model"], sam["inference"], sam["ros"]

        self.engine_path = _workspace_path(model["engine_path"])
        self.processor_path = _workspace_path(model["processor_path"])
        self.sam_device = str(model["device"])
        self.prompt = str(infer["prompt"])
        self.mask_threshold = float(infer["mask_threshold"])
        self.presence_threshold = float(infer["presence_threshold"])
        self.mask_topic = str(ros["mask_topic"])
        self.depth_mask_topic = str(ros["depth_mask_topic"])
        self.masked_depth_topic = str(ros["masked_depth_topic"])
        self.presence_topic = str(ros["presence_topic"])
        self.sam_qos = _QOS[str(ros["qos"])]

    def _setup_camera(self) -> None:
        self.camera = PyK4A(
            Config(
                color_resolution=getattr(ColorResolution, self.color_resolution),
                color_format=getattr(ImageFormat, self.color_format),
                depth_mode=getattr(DepthMode, self.depth_mode),
                camera_fps=getattr(FPS, self.camera_fps),
                synchronized_images_only=self.synchronized_images_only,
            ),
            device_id=self.device_id,
        )
        self.camera.start()
        self.calib = self.camera.calibration
        self._color_K = np.asarray(self.calib.get_camera_matrix(CalibrationType.COLOR), dtype=float)
        self._color_D = np.asarray(self.calib.get_distortion_coefficients(CalibrationType.COLOR), dtype=float)
        self._depth_K = np.asarray(self.calib.get_camera_matrix(CalibrationType.DEPTH), dtype=float)
        self._depth_D = np.asarray(self.calib.get_distortion_coefficients(CalibrationType.DEPTH), dtype=float)
        depth_size = {
            DepthMode.NFOV_2X2BINNED: (320, 288),
            DepthMode.NFOV_UNBINNED: (640, 576),
            DepthMode.WFOV_2X2BINNED: (512, 512),
            DepthMode.WFOV_UNBINNED: (1024, 1024),
        }[getattr(DepthMode, self.depth_mode)]
        self._depth_map1, self._depth_map2 = cv2.initUndistortRectifyMap(
            self._depth_K,
            self._depth_D,
            None,
            self._depth_K,
            depth_size,
            cv2.CV_32FC1,
        )
        self._color_info: CameraInfo | None = None
        self._depth_info: CameraInfo | None = None

    def _setup_publishers(self) -> None:
        self.color_pub = self.create_publisher(Image, self.color_topic, self.kinect_qos)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, self.kinect_qos)
        self.color_info_pub = self.create_publisher(CameraInfo, self.color_info_topic, self.kinect_qos)
        self.depth_info_pub = self.create_publisher(CameraInfo, self.depth_info_topic, self.kinect_qos)
        self.mask_pub = self.create_publisher(Image, self.mask_topic, self.sam_qos)
        self.depth_mask_pub = self.create_publisher(Image, self.depth_mask_topic, self.sam_qos)
        self.masked_depth_pub = self.create_publisher(Image, self.masked_depth_topic, self.sam_qos)
        self.presence_pub = self.create_publisher(Float32, self.presence_topic, self.sam_qos)
        self.extero_pub = self.create_publisher(
            JointState, STATE_ESTIMATOR_TOPIC, self.sam_qos
        )

    def _latest_capture(self):
        capture = self.camera.get_capture(timeout=self.capture_timeout_ms)
        while True:
            try:
                capture = self.camera.get_capture(timeout=0)
            except K4ATimeoutException:
                return capture

    def _capture_loop(self) -> None:
        while self._running and rclpy.ok():
            try:
                self._process(self._latest_capture())
            except K4ATimeoutException:
                continue
            except Exception as exc:
                self.get_logger().error(f"Kinect/SAM pipeline error: {exc}")
                self._running = False
                rclpy.shutdown()

    def _process(self, capture) -> None:
        process_start = time.perf_counter()

        color_bgra, depth = capture.color, capture.depth
        if color_bgra is None or depth is None:
            return

        stamp = self.get_clock().now().to_msg()

        bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)

        mask, presence = self.sam.infer(bgr, self.prompt, self.mask_threshold)
        if presence < self.presence_threshold:
            self.get_logger().warning(
                f"SAM3 low presence: {presence:.3f} < {self.presence_threshold:.3f}",
                throttle_duration_sec=1.0,
            )
            mask.zero_()

        mask = np.ascontiguousarray(mask.cpu().numpy(), dtype=np.uint8)

        mask_bgra = np.repeat(mask[:, :, None], 4, axis=2)

        transformed = color_image_to_depth_camera(
            mask_bgra, depth, self.calib, self.camera.thread_safe
        )

        masked_depth_raw = np.where(
            transformed[:, :, 0] >= 128, depth, 0
        ).astype(np.uint16)
        masked_depth = cv2.remap(
            masked_depth_raw,
            self._depth_map1,
            self._depth_map2,
            cv2.INTER_NEAREST,
        )
        depth_mask = (masked_depth != 0).astype(np.uint8) * 255

        if self._color_info is None:
            self._color_info = self._camera_info(bgr, self._color_K, self._color_D, self.color_frame_id)
            self._depth_info = self._camera_info(depth, self._depth_K, self._depth_D, self.depth_frame_id)

        # State estimate: masked depth -> [bottle_pos(3), cap_pos(3), jar_geom(4)], metres,
        pred = self.estimator.infer(masked_depth)[0].tolist()
        # HARDCODE STARTS HERE
        pred = hardcode_pred(pred, jar_name="green_vita")
        
        extero = JointState()
        extero.header.stamp = stamp
        extero.header.frame_id = self.depth_frame_id
        extero.name = _EXTERO_NAMES
        extero.position = pred
        self.extero_pub.publish(extero)

        self.masked_depth_pub.publish(self._image(masked_depth, stamp, "16UC1", self.depth_frame_id))
        self.depth_mask_pub.publish(self._image(depth_mask, stamp, "mono8", self.depth_frame_id))
        self.presence_pub.publish(Float32(data=float(presence)))
        self.mask_pub.publish(self._image(mask, stamp, "mono8", self.color_frame_id))
        self.color_pub.publish(self._image(bgr, stamp, "bgr8", self.color_frame_id))
        self.depth_pub.publish(self._image(depth, stamp, "16UC1", self.depth_frame_id))
        self._color_info.header.stamp = stamp
        self._depth_info.header.stamp = stamp
        self.color_info_pub.publish(self._color_info)
        self.depth_info_pub.publish(self._depth_info)

        total_ms = (time.perf_counter() - process_start) * 1000
        print(
            f"latency={total_ms:7.2f}ms  body xy=({pred[0]:+.3f},{pred[1]:+.3f})  "
            f"cap xy=({pred[3]:+.3f},{pred[4]:+.3f})  "
            f"cap rh=({pred[8]:.3f},{pred[9]:.3f})\033[K",
            end="\r", flush=True,
        )

    @staticmethod
    def _image(arr: np.ndarray, stamp, encoding: str, frame_id: str) -> Image:
        arr = np.ascontiguousarray(arr)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = arr.shape[:2]
        msg.encoding = encoding
        msg.step = arr.strides[0]
        msg.data = arr.tobytes()
        return msg

    @staticmethod
    def _camera_info(image: np.ndarray, K: np.ndarray, D: np.ndarray, frame_id: str) -> CameraInfo:
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.height, msg.width = image.shape[:2]
        msg.distortion_model = "rational_polynomial"
        msg.d = D.reshape(-1).tolist()
        msg.k = K.reshape(-1).tolist()
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def destroy_node(self) -> bool:
        print()  # keep the last \r-refreshed status line instead of letting ^C overwrite it
        self._running = False
        self._capture_thread.join(timeout=self.capture_timeout_ms / 1000.0 + 1.0)
        self.camera.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KinectSamC2DNode()
    executor = SingleThreadedExecutor()
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
