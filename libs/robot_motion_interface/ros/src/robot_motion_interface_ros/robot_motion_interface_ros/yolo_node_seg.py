"""
ROS 2 node: live YOLO segmentation-mask depth visualization from a RealSense camera.

Pipeline (matches realsense_record.py order — filter THEN align):
    wait_for_frames
        → decimation_filter
        → hole_filling_filter
        → align(depth → color frame)
        → YOLOSegDepthMasker.mask_depth(color, depth)
        → render masked depth as INFERNO colormap, normalised over [near, far]

RealSense settings (resolution / fps / intrinsics / exposure) are read from
`libs/robot_motion_interface/config/realsense_config.yaml`.

Pressure test on RTX 4090:
    n: 620MB VRAM, 6% utl
    s: 680MB VRAM, 8% utl
    m: 820MB VRAM, 13% utl
    l: 840MB VRAM, 16% utl
    x: 1.1GB VRAM, 23% utl

Runs inside the handrl-policy docker. Press 'q' in the preview window or
Ctrl-C in the terminal to stop.
"""

import os
import threading

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import yaml
from rclpy.node import Node

from robot_motion_interface.utils.ultralytics_utils import YOLOSegDepthMasker


REALSENSE_CONFIG_PATH = "/workspace/libs/robot_motion_interface/config/realsense_config.yaml"


class YoloNodeSeg(Node):
    def __init__(self):
        super().__init__("yolo_node_seg_depth")

        # ── ROS parameters ──────────────────────────────────────────────────
        self.declare_parameter("variant", "s")
        self.declare_parameter("target_class", "bottle")
        self.declare_parameter("conf", 0.05)
        self.declare_parameter("mask_threshold", 0.5)
        self.declare_parameter("mask_dilation", 0)
        self.declare_parameter("clip_near", 0.1)
        self.declare_parameter("clip_far",  1.1)

        config_path   = REALSENSE_CONFIG_PATH
        variant       = str(self.get_parameter("variant").value)
        target_class  = str(self.get_parameter("target_class").value)
        conf          = float(self.get_parameter("conf").value)
        mask_threshold = float(self.get_parameter("mask_threshold").value)
        mask_dilation = int(self.get_parameter("mask_dilation").value)
        self.clip_near = float(self.get_parameter("clip_near").value)
        self.clip_far  = float(self.get_parameter("clip_far").value)

        # ── RealSense config (mirrors realsense_record.py) ──────────────────
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        rs_cfg   = cfg["realsense"]
        fps      = int(rs_cfg["rs_fps"])
        c_intr   = rs_cfg["color_intrinsics"]
        d_intr   = rs_cfg["depth_intrinsics"]
        sens_set = rs_cfg.get("sensor_settings", {})

        # Filters: decimation(2) → hole_filling(2). Both run BEFORE align.
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 2)

        # Pipeline + align (depth → color frame).
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, c_intr["width"], c_intr["height"], rs.format.bgr8, fps)
        rs_config.enable_stream(rs.stream.depth, d_intr["width"], d_intr["height"], rs.format.z16, fps)
        profile = self.pipeline.start(rs_config)
        self.align = rs.align(rs.stream.color)

        # Exposure / gain / emitter / laser_power.
        if sens_set:
            auto_exposure   = sens_set.get("auto_exposure", False)
            exposure        = sens_set.get("exposure", 350)
            gain            = sens_set.get("gain", 16)
            emitter_enabled = sens_set.get("emitter_enabled", None)
            laser_power     = sens_set.get("laser_power", None)
            for sensor in profile.get_device().query_sensors():
                sensor_name = sensor.get_info(rs.camera_info.name)
                if sensor.supports(rs.option.enable_auto_exposure):
                    sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
                if not auto_exposure:
                    if exposure is not None and sensor.supports(rs.option.exposure):
                        sensor.set_option(rs.option.exposure, float(exposure))
                    if gain is not None and sensor.supports(rs.option.gain):
                        sensor.set_option(rs.option.gain, float(gain))
                # Emitter / laser power live on the stereo sensor only;
                # supports() filters color sensor automatically. laser_power
                # is clamped to the sensor's reported range so the same yaml
                # value works across D435 (max=360) / D405 (max=100) etc.
                if emitter_enabled is not None and sensor.supports(rs.option.emitter_enabled):
                    sensor.set_option(rs.option.emitter_enabled, float(emitter_enabled))
                    self.get_logger().info(
                        f"[{sensor_name}] emitter_enabled -> "
                        f"{sensor.get_option(rs.option.emitter_enabled)}"
                    )
                if laser_power is not None and sensor.supports(rs.option.laser_power):
                    lp_range = sensor.get_option_range(rs.option.laser_power)
                    clamped = max(lp_range.min, min(float(laser_power), lp_range.max))
                    sensor.set_option(rs.option.laser_power, clamped)
                    self.get_logger().info(
                        f"[{sensor_name}] laser_power -> "
                        f"{sensor.get_option(rs.option.laser_power)} mW "
                        f"(yaml={laser_power}, range {lp_range.min}..{lp_range.max})"
                    )

        try:
            self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        except Exception:
            self.depth_scale = 0.001

        # ── YOLO masker ─────────────────────────────────────────────────────
        self.masker = YOLOSegDepthMasker(
            variant=variant,
            target_class=target_class,
            conf_threshold=conf,
            mask_threshold=mask_threshold,
            mask_dilation=mask_dilation,
        )

        self.get_logger().info(
            f"YoloNodeSeg ready: rs={c_intr['width']}x{c_intr['height']}@{fps}fps  "
            f"yolo26{variant}-seg  target={target_class!r}  "
            f"mask_thr={mask_threshold}  dilation={mask_dilation}  "
            f"clip=[{self.clip_near}, {self.clip_far}]m"
        )

        # ── Background capture+display loop ─────────────────────────────────
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="yolo_capture")
        self._thread.start()

    def _loop(self):
        win = "yolo seg masked depth  (q = quit)"
        while self._running and rclpy.ok():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                self.get_logger().warn("wait_for_frames timed out, retrying...")
                continue

            # Filter chain — order must match realsense_record.py:
            #   decimation → hole_filling → align
            frames = self.decimation.process(frames).as_frameset()
            frames = self.hole_filling.process(frames).as_frameset()
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())   # HxWx3 uint8 BGR
            depth = np.asanyarray(depth_frame.get_data())   # HxW   uint16

            # YOLO segmentation + mask outside-instance depth pixels to 0.
            masked_depth, _bbox = self.masker.mask_depth(color, depth)

            # Render: masked depth → metres → fixed [near, far] clip → colormap.
            d_m = masked_depth.astype(np.float32) * self.depth_scale
            invalid = d_m <= 0   # mask-out / no detection
            norm = np.clip((d_m - self.clip_near) / (self.clip_far - self.clip_near + 1e-6), 0, 1)
            depth_u8 = (norm * 255).astype(np.uint8)
            depth_u8[invalid] = 0
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
            depth_color[invalid] = 0   # keep masked-out region cleanly black

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
    node = YoloNodeSeg()
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
