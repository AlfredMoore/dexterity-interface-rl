"""ROS 2 node: SAM3 TensorRT segmentation from the Kinect color stream."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
from sam3_trt import Sam3TRT


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent
WORKSPACE_ROOT = Path("/workspace")
DEFAULT_CONFIG_PATH = RMI_ROOT / "config" / "sam_node.yaml"

_QOS_BY_NAME = {
    "best_effort": HIGH_PERF_QOS,
    "reliable": HIGH_RELIA_QOS,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


def _workspace_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve(strict=False)


class Sam3Node(Node):
    def __init__(self) -> None:
        super().__init__("sam3_node")

        self.declare_parameter("sam_node_cfg_path", str(DEFAULT_CONFIG_PATH))
        cfg_path = Path(self.get_parameter("sam_node_cfg_path").value)
        cfg = _load_yaml(cfg_path)["sam3"]

        model_cfg = cfg["model"]
        infer_cfg = cfg["inference"]
        ros_cfg = cfg["ros"]

        self.prompt = str(infer_cfg["prompt"])
        self.mask_threshold = float(infer_cfg["mask_threshold"])
        self.presence_threshold = float(infer_cfg["presence_threshold"])

        self.input_topic = str(ros_cfg["input_topic"])
        self.mask_topic = str(ros_cfg["mask_topic"])
        self.presence_topic = str(ros_cfg["presence_topic"])
        qos_name = str(ros_cfg["qos"])
        if qos_name not in _QOS_BY_NAME:
            raise ValueError(
                f"Unknown qos {qos_name!r}; expected one of {list(_QOS_BY_NAME)}"
            )
        qos = _QOS_BY_NAME[qos_name]

        engine_path = _workspace_path(model_cfg["engine_path"])
        processor_path = _workspace_path(model_cfg["processor_path"])
        device = str(model_cfg["device"])

        self.sam = Sam3TRT(
            engine_path=str(engine_path),
            processor_path=str(processor_path),
            device=device,
        )

        self.mask_pub = self.create_publisher(Image, self.mask_topic, qos)
        self.presence_pub = self.create_publisher(
            Float32, self.presence_topic, qos
        )
        self.image_sub = self.create_subscription(
            Image, self.input_topic, self._image_callback, qos
        )

        self.get_logger().info(
            f"Sam3Node ready: {self.input_topic} -> {self.mask_topic}, "
            f"prompt={self.prompt!r}, device={device}, qos={qos_name}"
        )

    def _image_callback(self, msg: Image) -> None:
        bgr = self._image_to_bgr(msg)
        mask, presence = self.sam.infer(
            bgr,
            prompt=self.prompt,
            threshold=self.mask_threshold,
        )

        if presence < self.presence_threshold:
            self.get_logger().warning(
                f"SAM3 low presence for {self.prompt!r}: "
                f"{presence:.3f} < {self.presence_threshold:.3f}; "
                "publishing an empty mask",
                throttle_duration_sec=1.0,
            )
            mask.zero_()

        # Sam3TRT leaves the mask on CUDA; ROS Image requires host bytes.
        mask_np = np.ascontiguousarray(mask.cpu().numpy(), dtype=np.uint8)
        self.mask_pub.publish(self._mask_message(mask_np, msg))
        self.presence_pub.publish(Float32(data=float(presence)))

    @staticmethod
    def _image_to_bgr(msg: Image) -> np.ndarray:
        if msg.encoding not in ("bgr8", "rgb8"):
            raise ValueError(
                f"Unsupported encoding {msg.encoding!r}; expected 'bgr8' or 'rgb8'"
            )

        row_width = msg.width * 3
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = rows[:, :row_width].reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(image)

    @staticmethod
    def _mask_message(mask: np.ndarray, source: Image) -> Image:
        msg = Image()
        msg.header.stamp = source.header.stamp
        msg.header.frame_id = source.header.frame_id
        msg.height, msg.width = mask.shape
        msg.encoding = "mono8"
        msg.is_bigendian = False
        msg.step = msg.width
        msg.data = mask.tobytes()
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Sam3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
