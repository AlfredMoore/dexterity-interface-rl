"""cam_node: RealSense capture + DA3 depth inference, exposing the depth
tensor to other processes via CUDA IPC.

Architecture:
  - A daemon Python thread pulls color frames from RealSense at rs_fps and
    publishes the latest BGR->RGB ndarray under vision_lock (same pattern as
    the old aux_policy_v2 capture loop).
  - A ROS timer at da3_cfg.rate runs DA3 inference on the latest color frame
    and copy_'s the result into a persistent CUDA buffer self.depth_buf,
    seeded by the warmup pass so its shape/dtype match every subsequent infer.
  - The CUDA IPC handle of self.depth_buf is encoded once at init and served
    via a std_srvs/Trigger service. aux_policy calls it once during its own
    init, unpickles, and reads the buffer directly with no lock — accepting
    that occasional torn reads are cheap compared to lock/IPC overhead.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import threading
import time
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import torch
import torch.multiprocessing  # noqa: F401  # registers CUDA IPC reducers in ForkingPickler
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from robot_motion_interface.utils.da3_compile_utils import DA3Inference


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = yaml.unsafe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def _encode_cuda_ipc(tensor: torch.Tensor) -> str:
    """Serialize a CUDA tensor's IPC handle to a base64 ASCII string.

    Uses ForkingPickler so torch.multiprocessing.reductions.reduce_tensor is
    invoked, which produces a payload carrying a cudaIpcMemHandle instead of
    the tensor data itself.
    """
    buf = io.BytesIO()
    ForkingPickler(buf).dump(tensor)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class DepthNode(Node):
    def __init__(self) -> None:
        super().__init__("depth_node")

        self._declare_parameters()
        self._load_configs()
        self._init_device()

        self.vision_lock = threading.Lock()
        self.latest_color_rgb: np.ndarray | None = None

        self._setup_realsense()
        self._setup_da3_and_buffer()
        self._encoded_handle = _encode_cuda_ipc(self.depth_buf)

        self._start_capture_thread()
        self._init_da3_timer()
        self._init_handle_service()

        self.get_logger().info(
            f"DepthNode ready: rs={self.rs_width}x{self.rs_height}@{int(self.capture_hz)}Hz, "
            f"da3_rate={self.da3_hz}Hz, depth_buf={tuple(self.depth_buf.shape)} "
            f"{self.depth_buf.dtype}, service={self.handle_service_name}"
        )

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        # self.declare_parameter(
        #     "policy_node_cfg_path",
        #     str((RMI_ROOT / "config" / "rl_policy_node_config.yaml").resolve()),
        # )
        self.declare_parameter(
            "da3_cfg_path",
            str((RMI_ROOT / "config" / "da3_compile_config.yaml").resolve()),
        )
        self.declare_parameter(
            "env_cfg_path",
            str((RMI_ROOT / "runtime" / "env.yaml").resolve()),
        )

        # self.policy_node_cfg_path = Path(self.get_parameter("policy_node_cfg_path").value)
        self.da3_cfg_path = Path(self.get_parameter("da3_cfg_path").value)
        self.env_cfg_path = Path(self.get_parameter("env_cfg_path").value)

    def _load_configs(self) -> None:
        # self.policy_node_cfg = _load_yaml(self.policy_node_cfg_path)
        self.da3_full_cfg = _load_yaml(self.da3_cfg_path)
        self.env_cfg = _load_yaml(self.env_cfg_path)

        self.da3_cfg = self.da3_full_cfg["da3_cfg"]
        env_process_res = int(self.env_cfg["da3_process_res"])
        da3_process_res = int(self.da3_cfg["process_res"])
        if env_process_res != da3_process_res:
            raise ValueError(
                "process_res mismatch: env_cfg.da3_process_res="
                f"{env_process_res} vs da3_cfg.process_res={da3_process_res}"
            )

        self.depth_clip_min = float(self.da3_cfg["depth_clip_min"])
        self.depth_clip_max = float(self.da3_cfg["depth_clip_max"])

        ipc_cfg = self.da3_full_cfg["ipc"]
        self.handle_service_name = str(ipc_cfg["handle_service_name"])

        realsense_cfg = self.da3_cfg["realsense"]
        self.realsense_cfg = realsense_cfg
        color_intrinsics = realsense_cfg["color_intrinsics"]
        self.rs_width = int(color_intrinsics["width"])
        self.rs_height = int(color_intrinsics["height"])
        self.capture_hz = float(realsense_cfg["rs_fps"])
        self.da3_hz = float(self.da3_cfg["rate"])

    def _init_device(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available.")
        self.device = torch.device("cuda")

    # ------------------------------------------------------------------
    # RealSense
    # ------------------------------------------------------------------
    def _setup_realsense(self) -> None:
        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.color, self.rs_width, self.rs_height, rs.format.bgr8, int(self.capture_hz)
        )
        rs_profile = self.rs_pipeline.start(rs_config)
        self._apply_sensor_settings(rs_profile, self.realsense_cfg.get("sensor_settings", {}))
        self.get_logger().info(
            f"RealSense capture started: {self.rs_width}x{self.rs_height}@{int(self.capture_hz)}Hz"
        )
        time.sleep(0.5)

    def _apply_sensor_settings(self, rs_profile, sensor_settings: dict[str, Any]) -> None:
        if not sensor_settings:
            return
        sensors = rs_profile.get_device().query_sensors()
        auto_exposure = sensor_settings.get("auto_exposure", False)
        exposure = sensor_settings.get("exposure")
        gain = sensor_settings.get("gain")
        for sensor in sensors:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
            if not auto_exposure:
                if exposure is not None and sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, float(exposure))
                if gain is not None and sensor.supports(rs.option.gain):
                    sensor.set_option(rs.option.gain, float(gain))

    def _start_capture_thread(self) -> None:
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._cam_capture_loop, daemon=True, name="cam_rs_capture"
        )
        self._capture_thread.start()

    def _cam_capture_loop(self) -> None:
        period = 1.0 / max(self.capture_hz, 1e-3)
        while self._capture_running:
            loop_start = time.perf_counter()
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                color_rgb = cv2.cvtColor(
                    np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB
                )
                with self.vision_lock:
                    self.latest_color_rgb = color_rgb
            except Exception as exc:
                self.get_logger().warn(f"RealSense capture error: {exc}")
                self._capture_running = False
                rclpy.shutdown()
                return

            elapsed = time.perf_counter() - loop_start
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    # ------------------------------------------------------------------
    # DA3 + persistent depth buffer
    # ------------------------------------------------------------------
    def _setup_da3_and_buffer(self) -> None:
        self.get_logger().info("DA3 compilation started.")
        self.da3 = DA3Inference.from_dict(self.da3_cfg)

        # Warmup with a zero RGB tensor; the resulting depth tensor doubles as
        # the seed for the persistent IPC buffer (caller wanted to reuse the
        # warmup output as the storage tensor).
        warmup_rgb = torch.zeros(
            (1, self.rs_height, self.rs_width, 3), dtype=torch.uint8, device=self.device
        )
        warmup_depth = self.da3.infer_no_chunk(warmup_rgb)
        warmup_depth = self._normalize_depth_shape(warmup_depth)
        warmup_depth = warmup_depth.to(dtype=torch.float32).clamp(
            min=self.depth_clip_min, max=self.depth_clip_max
        ).to(torch.float16)

        # Persistent buffer reused for every subsequent infer via copy_, so its
        # CUDA IPC handle remains valid for the lifetime of the process.
        self.depth_buf = warmup_depth.clone().contiguous()
        torch.cuda.synchronize(self.device)
        self.get_logger().info(
            f"DA3 warmup done: depth_buf shape={tuple(self.depth_buf.shape)} dtype={self.depth_buf.dtype}"
        )

    def _normalize_depth_shape(self, depth_t: torch.Tensor) -> torch.Tensor:
        if depth_t.dim() == 2:
            return depth_t.unsqueeze(0)
        if depth_t.dim() == 3 and depth_t.shape[0] == 1:
            return depth_t
        raise ValueError(f"Unexpected DA3 depth shape: {tuple(depth_t.shape)}")

    def _init_da3_timer(self) -> None:
        self.da3_timer = self.create_timer(1.0 / max(self.da3_hz, 1e-3), self._da3_step)

    def _da3_step(self) -> None:
        loop_start = time.perf_counter()
        with self.vision_lock:
            color_rgb = self.latest_color_rgb.copy() if self.latest_color_rgb is not None else None
        if color_rgb is None:
            return  # no frame yet; capture thread will fill it shortly

        try:
            color_rgb_t = torch.from_numpy(color_rgb).to(self.device, non_blocking=True).unsqueeze(0)
            depth_t = self._normalize_depth_shape(self.da3.infer_no_chunk(color_rgb_t))
            # In-place into the IPC-shared buffer; consumers read this same
            # GPU memory. Auto-casts to float16 because depth_buf is float16.
            self.depth_buf.copy_(
                depth_t.to(dtype=torch.float32).clamp(
                    min=self.depth_clip_min, 
                    max=self.depth_clip_max), non_blocking=True)
        except Exception as exc:
            self.get_logger().error(f"DA3 inference failed: {exc}")
            rclpy.shutdown()
            return

        elapsed = time.perf_counter() - loop_start
        period = 1.0 / max(self.da3_hz, 1e-3)
        if elapsed > period:
            self.get_logger().warn(
                f"[SLOW_DA3] total={elapsed:.4f}s, target_period={period:.4f}s"
            )

    # ------------------------------------------------------------------
    # IPC handle service
    # ------------------------------------------------------------------
    def _init_handle_service(self) -> None:
        self.handle_srv = self.create_service(
            Trigger, self.handle_service_name, self._on_get_handle
        )

    def _on_get_handle(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = True
        response.message = self._encoded_handle
        self.get_logger().info("Served depth IPC handle to a client.")
        return response

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def destroy_node(self) -> bool:
        self._capture_running = False
        if hasattr(self, "da3_timer") and self.da3_timer is not None:
            self.da3_timer.cancel()
        if getattr(self, "_capture_thread", None) is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if hasattr(self, "rs_pipeline") and self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except Exception as exc:
                self.get_logger().error(f"Error stopping RealSense pipeline: {exc}")
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CamNode()
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
