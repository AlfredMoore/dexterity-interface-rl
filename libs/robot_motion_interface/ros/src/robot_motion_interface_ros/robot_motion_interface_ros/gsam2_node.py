"""
ROS 2 node: live Grounded-SAM-2 segmentation-mask depth visualization from a RealSense camera.

Pipeline:
    wait_for_frames
        -> depth-domain decimation_filter
        -> depth-domain hole_filling_filter
        -> align filtered depth to the 320x240 color frame
        -> GroundedSAM2Masker.mask_depth(color, aligned_depth)
        -> render masked depth as INFERNO colormap, normalized over [near, far]

GSAM2 settings are read from
`libs/robot_motion_interface/config/gsam2_config.yaml`.
RealSense settings are read from
`libs/robot_motion_interface/config/realsense_config.yaml`.

Runs inside the handrl-policy docker. Press 'q' in the preview window or
Ctrl-C in the terminal to stop.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import yaml
import time
from rclpy.node import Node

from robot_motion_interface.utils.groundsam2_utils import GroundedSAM2Masker


WORKSPACE_ROOT = Path("/workspace")
REALSENSE_CONFIG_PATH = "libs/robot_motion_interface/config/realsense_config.yaml"
GSAM2_CONFIG_PATH = "libs/robot_motion_interface/config/gsam2_config.yaml"


def _resolve_workspace_path(value: object) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("Path value must not be empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve(strict=False)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = _resolve_workspace_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


def _optional_workspace_path(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return str(_resolve_workspace_path(text))


class GSam2Node(Node):
    def __init__(self):
        super().__init__("gsam2_node_depth")

        # ROS parameters for config locations. Model/runtime defaults live in YAML.
        self.declare_parameter("realsense_config_path", REALSENSE_CONFIG_PATH)
        self.declare_parameter("gsam2_config_path", GSAM2_CONFIG_PATH)
        realsense_config_path = _resolve_workspace_path(self.get_parameter("realsense_config_path").value)
        gsam2_config_path = _resolve_workspace_path(self.get_parameter("gsam2_config_path").value)

        # RealSense config.
        rs_full_cfg = _load_yaml(realsense_config_path)
        rs_cfg = rs_full_cfg["realsense"]
        fps = int(rs_cfg["rs_fps"])
        c_intr = rs_cfg["color_intrinsics"]
        d_intr = rs_cfg["depth_intrinsics"]
        sens_set = rs_cfg.get("sensor_settings", {})
        clip = rs_cfg.get("clip", [0.1, 1.1])

        # Grounded-SAM2 config.
        gsam2_full_cfg = _load_yaml(gsam2_config_path)
        gsam2_cfg = gsam2_full_cfg.get("gsam2", {})
        preview_cfg = gsam2_full_cfg.get("preview", {})

        self.declare_parameter("prompt", str(gsam2_cfg.get("prompt", "bottle.")))
        self.declare_parameter("sam2_variant", str(gsam2_cfg.get("sam2_variant", "s")))
        self.declare_parameter("sam2_ckpt_path", str(gsam2_cfg.get("sam2_ckpt_path", "")))
        self.declare_parameter("sam2_cache_dir", str(gsam2_cfg.get("sam2_cache_dir", "")))
        self.declare_parameter("grounding_model_id", str(gsam2_cfg.get("grounding_model_id", "IDEA-Research/grounding-dino-tiny")))
        self.declare_parameter("grounding_cache_dir", str(gsam2_cfg.get("grounding_cache_dir", "")))
        self.declare_parameter("device", str(gsam2_cfg.get("device", "cuda")))
        self.declare_parameter("detection_interval", int(gsam2_cfg.get("detection_interval", 20)))
        self.declare_parameter("box_threshold", float(gsam2_cfg.get("box_threshold", 0.25)))
        self.declare_parameter("text_threshold", float(gsam2_cfg.get("text_threshold", 0.25)))
        self.declare_parameter("fill_value", float(gsam2_cfg.get("fill_value", 0.0)))
        self.declare_parameter("mask_dilation", int(gsam2_cfg.get("mask_dilation", 0)))
        self.declare_parameter("clip_near", float(preview_cfg.get("clip_near", clip[0])))
        self.declare_parameter("clip_far", float(preview_cfg.get("clip_far", clip[1])))

        prompt = str(self.get_parameter("prompt").value)
        sam2_variant = str(self.get_parameter("sam2_variant").value)
        sam2_ckpt_path = _optional_workspace_path(self.get_parameter("sam2_ckpt_path").value)
        sam2_cache_dir = _optional_workspace_path(self.get_parameter("sam2_cache_dir").value)
        grounding_model_id = str(self.get_parameter("grounding_model_id").value)
        grounding_cache_dir = _optional_workspace_path(self.get_parameter("grounding_cache_dir").value)
        device = str(self.get_parameter("device").value)
        detection_interval = int(self.get_parameter("detection_interval").value)
        box_threshold = float(self.get_parameter("box_threshold").value)
        text_threshold = float(self.get_parameter("text_threshold").value)
        fill_value = float(self.get_parameter("fill_value").value)
        mask_dilation = int(self.get_parameter("mask_dilation").value)
        self.clip_near = float(self.get_parameter("clip_near").value)
        self.clip_far = float(self.get_parameter("clip_far").value)

        # Depth filters run before alignment, in the native depth stream domain.
        # Color is already streamed at 320x240, so aligned depth lands directly on
        # the inference image grid without an extra resize step.
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 2)

        # Pipeline + align (depth -> color frame).
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, c_intr["width"], c_intr["height"], rs.format.bgr8, fps)
        rs_config.enable_stream(rs.stream.depth, d_intr["width"], d_intr["height"], rs.format.z16, fps)
        profile = self.pipeline.start(rs_config)
        self.align = rs.align(rs.stream.color)

        # Exposure / gain settings.
        if sens_set:
            auto_exposure = sens_set.get("auto_exposure", False)
            exposure = sens_set.get("exposure", 350)
            gain = sens_set.get("gain", 16)
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
                if not auto_exposure:
                    if exposure is not None and sensor.supports(rs.option.exposure):
                        sensor.set_option(rs.option.exposure, float(exposure))
                    if gain is not None and sensor.supports(rs.option.gain):
                        sensor.set_option(rs.option.gain, float(gain))

        try:
            self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        except Exception:
            self.depth_scale = 0.001

        # Grounded-SAM2 masker.
        self.masker = GroundedSAM2Masker(
            prompt=prompt,
            sam2_variant=sam2_variant,
            sam2_ckpt_path=sam2_ckpt_path,
            sam2_cache_dir=sam2_cache_dir,
            grounding_model_id=grounding_model_id,
            grounding_cache_dir=grounding_cache_dir,
            device=device,
            detection_interval=detection_interval,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            fill_value=fill_value,
            mask_dilation=mask_dilation,
        )

        self.get_logger().info(
            f"GSam2Node ready: rs={c_intr['width']}x{c_intr['height']}@{fps}fps  "
            f"sam2={sam2_variant} prompt={prompt!r} interval={detection_interval}  "
            f"box_thr={box_threshold} text_thr={text_threshold} dilation={mask_dilation}  "
            f"clip=[{self.clip_near}, {self.clip_far}]m"
        )

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gsam2_capture")
        self._thread.start()

    def _loop(self):
        win = "gsam2 masked depth  (q = quit)"
        while self._running and rclpy.ok():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                self.get_logger().warn("wait_for_frames timed out, retrying...")
                continue
            t = time.perf_counter()
            # Depth-domain filtering first, then align filtered depth to color.
            frames = self.decimation.process(frames).as_frameset()
            frames = self.hole_filling.process(frames).as_frameset()
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            t = time.perf_counter()
            masked_depth, mask = self.masker.mask_depth(color, depth)
            
            d_m = masked_depth.astype(np.float32) * self.depth_scale
            self.get_logger().info(f"Frame processed in {(time.perf_counter() - t)*1000:.1f} ms")
            
            invalid = d_m <= 0
            norm = np.clip(
                (d_m - self.clip_near) / (self.clip_far - self.clip_near + 1e-6),
                0,
                1,
            )
            depth_u8 = (norm * 255).astype(np.uint8)
            depth_u8[invalid] = 0
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
            depth_color[invalid] = 0

            if mask is not None:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(depth_color, contours, -1, (0, 255, 0), 1)

            cv2.imshow(win, depth_color)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                self._running = False
                rclpy.try_shutdown()
                return

    def destroy_node(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = GSam2Node()
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
