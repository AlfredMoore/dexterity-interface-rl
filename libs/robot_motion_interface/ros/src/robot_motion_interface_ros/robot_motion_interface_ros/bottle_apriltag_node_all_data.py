"""ROS 2 node: bottle_apriltag_node + full cotrain rolling buffer.

Single process owns RealSense for both
  1) AprilTag pose publishing (so apriltag_policy keeps running), and
  2) full-modality cotrain data collection (no YOLO / no depth_feat / no IPC).

Relative to bottle_apriltag_node:
  * enables the depth stream alongside color (decimation + hole_filling + align)
  * CPU-downsamples 640x480 -> 320x240
        - color via INTER_AREA   (clean low-res RGB)
        - depth via INTER_NEAREST (no averaging across object edges)
  * subscribes to fk_topic and assembles the 24-d FK vector each frame
  * keeps six 300-frame rolling buffers; Enter dumps them to disk as a
    timestamped .npz + meta.json under /workspace/data/cotrain/
  * on detect miss / PnP fail / filter reject: REPUBLISHES the last-good
    pose so apriltag_policy doesn't stutter, but records apriltag_valid=0
    in the buffer so offline training never sees stale poses as ground
    truth (the data row's body/cap fields are zeros, valid flag = 0).

Relative to depth_feat_node_all_data:
  * no YOLO, no DepthFeatureExtractor, no CUDA IPC
  * masked-depth is NOT computed here — reconstruct offline from raw_depth
    + a YOLO/GSAM2 mask pass over the saved rgb_lores stream (we have all
    the inputs and don't pay the YOLO + film-net cost during collection).

Architecture: capture thread owns wait_for_frames + detection + publish +
buffer append. Main thread runs the ROS executor solely for the FK
subscription callback (which writes into fk_pose_dict under fk_lock).

Configs (paths workspace-relative):
  * april_tag_node_config.yaml  — tag dict / size / offsets / bias / filters / topics
  * realsense_config.yaml       — sensor settings (depth scale checked 0.001)
  * runtime/rs_config.yaml      — T_world_cam
  * depth_feat_config.yaml      — fk_dim + link selection (reused so the FK
                                  vector layout matches depth_feat_node_all_data)
  * fk_config.yaml              — fk_topic + ordered link_names
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS


WORKSPACE_ROOT = Path("/workspace")
APRILTAG_CONFIG_PATH    = "libs/robot_motion_interface/config/april_tag_node_config.yaml"
REALSENSE_CONFIG_PATH   = "libs/robot_motion_interface/config/realsense_config.yaml"
EXTRINSICS_CONFIG_PATH  = "libs/robot_motion_interface/runtime/rs_config.yaml"
DEPTH_FEAT_CONFIG_PATH  = "libs/robot_motion_interface/config/depth_feat_config.yaml"
FK_CONFIG_PATH          = "libs/robot_motion_interface/config/fk_config.yaml"

# Same conventions as depth_feat_node_all_data: 300 frames @ capture rate
# (~10s @ 30Hz). Bump if you want longer trajectories per sample (cost is
# ~1.3MB / frame on disk uncompressed).
_ROLLING_BUFFER_SIZE = 300
_OUT_DIR = WORKSPACE_ROOT / "data" / "cotrain"

# CPU-downsampled "policy resolution" — matches the training collect pipeline
# (collect_policy_realsense.py + train_depth_predictor.py). Used for both the
# 320x240 depth and 320x240 rgb that feed offline state-estimation finetune.
_POLICY_WIDTH = 320
_POLICY_HEIGHT = 240


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


class BottleAprilTagNodeAllData(Node):
    def __init__(self) -> None:
        super().__init__("bottle_apriltag_node_all_data")

        self._declare_parameters()
        self._load_configs()

        # Capture-thread-local state — single writer/reader, no lock needed.
        self._prev_tag_z_world: np.ndarray | None = None
        # Last-good cache: republished on detect miss so apriltag_policy
        # doesn't stutter. NOT used as a buffer record source — when
        # detection fails, the buffer row is zeros + valid=0.
        self._last_body_world: np.ndarray | None = None
        self._last_cap_world:  np.ndarray | None = None
        self._diag_counter: int = 0

        # FK subscription: written in executor thread, read in capture thread.
        self.fk_lock = threading.Lock()

        # Rolling buffers: written in capture thread, snapshotted in stdin
        # thread inside _handle_save_trigger. Layout mirrors
        # depth_feat_node_all_data so offline pipelines can ingest both
        # collectors uniformly.
        self._rolling_lock = threading.Lock()
        self._rgb_hires_buf: deque = deque(maxlen=_ROLLING_BUFFER_SIZE)  # (480,640,3) uint8
        self._rgb_lores_buf: deque = deque(maxlen=_ROLLING_BUFFER_SIZE)  # (240,320,3) uint8
        self._depth_buf:     deque = deque(maxlen=_ROLLING_BUFFER_SIZE)  # (240,320)   uint16
        self._fk_buf:        deque = deque(maxlen=_ROLLING_BUFFER_SIZE)  # (24,)       float32
        self._apriltag_buf:  deque = deque(maxlen=_ROLLING_BUFFER_SIZE)  # (7,)        float32 [valid,b.xyz,c.xyz]
        self._timestamp_buf: deque = deque(maxlen=_ROLLING_BUFFER_SIZE)  # ()          float64 unix-s
        self._save_counter: int = 0

        # Capture-thread + stdin-thread control flags. Daemon threads, no
        # graceful join needed beyond a `_running = False`.
        self._capture_running: bool = True
        self._capture_thread:  threading.Thread | None = None
        self._stdin_running:   bool = True
        self._stdin_thread:    threading.Thread | None = None

        self._setup_fk_state()
        self._setup_realsense()
        self._setup_apriltag()
        self._init_publishers()
        self._init_fk_subscription()

        self.get_logger().info(
            f"BottleAprilTagNodeAllData ready:\n"
            f"  rs          = {self.rs_color_width}x{self.rs_color_height}@{int(self.capture_hz)}Hz "
            f"-> ({_POLICY_WIDTH}x{_POLICY_HEIGHT} lores)\n"
            f"  tag         = {self.apriltag_dict_name}  id={self.target_tag_id}  size={self.marker_size_m*1000:.1f}mm\n"
            f"  offsets     = cap-{self.cap_dz*1000:.1f}mm  body-{self.body_dz*1000:.1f}mm (along tag +z)\n"
            f"  pose_bias   = ({self.pose_bias[0]*1000:+.1f}, {self.pose_bias[1]*1000:+.1f}, {self.pose_bias[2]*1000:+.1f}) mm\n"
            f"  ws_top_z    = {self.workstation_top_z:+.5f} m\n"
            f"  geom        = {self.bottle_geom.tolist()}\n"
            f"  filters     = reject_z_below_horizon={self.reject_z_below_horizon}  reject_sudden_flip={self.reject_sudden_flip}\n"
            f"  fk_topic    = {self.fk_topic}  ({self.fk_n_links} links, fk_dim={self.fk_dim})\n"
            f"  pub topics  = {self.poses_topic}  {self.geom_topic}  {self.vis_topic}\n"
            f"  cotrain     = rolling buffer {_ROLLING_BUFFER_SIZE} frames "
            f"(~{_ROLLING_BUFFER_SIZE / max(self.capture_hz, 1e-3):.1f}s @ {int(self.capture_hz)}Hz), "
            f"out_dir={_OUT_DIR}\n"
            f"  detect miss = republish last-good pose; buffer row apriltag_valid=0"
        )

        self._start_capture_thread()
        self._start_stdin_listener()
        self.get_logger().info(
            "[cotrain] press ENTER to save the last "
            f"{_ROLLING_BUFFER_SIZE} frames to "
            f"{_OUT_DIR}/sample_<timestamp>/data.npz; Ctrl-C to exit "
            f"(no auto-save on exit)."
        )

    # ------------------------------------------------------------------
    # init: parameters / configs
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("apriltag_config_path",   APRILTAG_CONFIG_PATH)
        self.declare_parameter("realsense_config_path",  REALSENSE_CONFIG_PATH)
        self.declare_parameter("extrinsics_config_path", EXTRINSICS_CONFIG_PATH)
        self.declare_parameter("depth_feat_config_path", DEPTH_FEAT_CONFIG_PATH)
        self.declare_parameter("fk_config_path",         FK_CONFIG_PATH)
        self.apriltag_config_path = _resolve_workspace_path(
            self.get_parameter("apriltag_config_path").value
        )
        self.realsense_config_path = _resolve_workspace_path(
            self.get_parameter("realsense_config_path").value
        )
        self.extrinsics_config_path = _resolve_workspace_path(
            self.get_parameter("extrinsics_config_path").value
        )
        self.depth_feat_config_path = _resolve_workspace_path(
            self.get_parameter("depth_feat_config_path").value
        )
        self.fk_config_path = _resolve_workspace_path(
            self.get_parameter("fk_config_path").value
        )

    def _load_configs(self) -> None:
        cfg = _load_yaml(self.apriltag_config_path)
        rs_cfg_full = _load_yaml(self.realsense_config_path)
        ext_cfg = _load_yaml(self.extrinsics_config_path)
        df_cfg = _load_yaml(self.depth_feat_config_path)
        fk_cfg = _load_yaml(self.fk_config_path)

        self._rs_cfg = rs_cfg_full["realsense"]
        self._df_cfg = df_cfg
        self._fk_cfg = fk_cfg

        # ---- AprilTag ----
        at_cfg = cfg["apriltag"]
        self.apriltag_dict_name = str(at_cfg["dict"])
        self.target_tag_id      = int(at_cfg["tag_id"])
        self.marker_size_m      = float(at_cfg["marker_size_m"])

        # ---- bottle_geom -> cap_dz / body_dz ----
        bg = cfg["bottle_geom"]
        if len(bg) != 4:
            raise ValueError(
                f"bottle_geom must be a 4-element list [body_radius, body_height, "
                f"cap_radius, cap_height]; got {bg}"
            )
        self.bottle_geom = np.array(bg, dtype=np.float32)
        body_h = float(self.bottle_geom[1])
        cap_h  = float(self.bottle_geom[3])
        # Tag glued flat on cap top -> walk along tag +z to reach each centre.
        self.cap_dz  = cap_h / 2.0
        self.body_dz = cap_h + body_h / 2.0

        # ---- Geometry biases ----
        geom_cfg = cfg.get("geometry", {})
        self.pose_bias = np.array(
            [
                float(geom_cfg.get("pose_bias_x", 0.0)),
                float(geom_cfg.get("pose_bias_y", 0.0)),
                float(geom_cfg.get("pose_bias_z", 0.0)),
            ],
            dtype=np.float64,
        )
        self.workstation_top_z = float(geom_cfg.get("workstation_top_z", 0.0))

        # ---- Filters ----
        filt_cfg = cfg.get("filters", {})
        self.reject_z_below_horizon = bool(filt_cfg.get("reject_z_below_horizon", True))
        self.reject_sudden_flip     = bool(filt_cfg.get("reject_sudden_flip",     True))

        # ---- Output topics ----
        out_cfg = cfg["output"]
        self.world_frame_id = str(out_cfg.get("world_frame_id", "world"))
        self.publish_poses  = bool(out_cfg.get("publish_poses", True))
        self.poses_topic    = str(out_cfg.get("poses_topic", "/bottle_apriltag/poses"))
        self.publish_geom   = bool(out_cfg.get("publish_geom", True))
        self.geom_topic     = str(out_cfg.get("geom_topic",  "/bottle_apriltag/geom"))
        self.enable_vis     = bool(out_cfg.get("enable_vis", True))
        self.vis_topic      = str(out_cfg.get("vis_topic",   "/bottle_apriltag/vis"))

        # ---- Extrinsics ----
        if "T_world_cam" not in ext_cfg:
            raise KeyError(
                f"{self.extrinsics_config_path} has no T_world_cam — "
                f"re-run realsense_artag_cali.py to regenerate."
            )
        self._T_world_cam = np.array(ext_cfg["T_world_cam"], dtype=np.float64)
        if self._T_world_cam.shape != (4, 4):
            raise ValueError(
                f"T_world_cam shape {self._T_world_cam.shape}, expected (4, 4)."
            )

        # ---- RealSense rate ----
        self.capture_hz = float(self._rs_cfg["rs_fps"])

    # ------------------------------------------------------------------
    # FK state assembly (24-d vector; layout copied from depth_feat_node_all_data
    # so offline state-estimation finetune can mix samples from either collector)
    # ------------------------------------------------------------------
    def _setup_fk_state(self) -> None:
        feat_cfg = self._df_cfg["depth_feature"]
        fk_sel = self._df_cfg["fk"]

        self.fk_dim: int = int(feat_cfg["fk_dim"])
        self.fk_link_names: list[str] = list(self._fk_cfg["link_names"])
        self.fk_topic: str = str(self._fk_cfg["fk_topic"])
        self.fk_n_links = len(self.fk_link_names)

        self.left_hand_base_names  = list(fk_sel["left_hand_base_links"])
        self.right_hand_base_names = list(fk_sel["right_hand_base_links"])
        self.left_fingertip_names  = list(fk_sel["left_fingertip_links"])
        self.right_fingertip_names = list(fk_sel["right_fingertip_links"])

        derived_fk_dim = 3 * (
            len(self.left_hand_base_names) + len(self.right_hand_base_names)
            + len(self.left_fingertip_names) + len(self.right_fingertip_names)
        )
        if derived_fk_dim != self.fk_dim:
            raise ValueError(
                f"fk_dim mismatch: depth_feature.fk_dim={self.fk_dim} vs "
                f"link-derived={derived_fk_dim}"
            )

        self.fk_pose_dict: dict[str, np.ndarray] = {
            name + "_pos": np.zeros(3, dtype=np.float32) for name in self.fk_link_names
        }
        self.has_fk: bool = False

    def _init_fk_subscription(self) -> None:
        self.create_subscription(
            PoseArray, self.fk_topic, self._sub_fk_cb, HIGH_PERF_QOS
        )

    def _sub_fk_cb(self, msg: PoseArray) -> None:
        if len(msg.poses) != self.fk_n_links:
            self.get_logger().error(
                f"{self.fk_topic} pose count mismatch: {len(msg.poses)} vs "
                f"expected {self.fk_n_links} (fk_cfg.link_names)"
            )
            rclpy.shutdown()
            return
        with self.fk_lock:
            for name, p in zip(self.fk_link_names, msg.poses):
                pos = self.fk_pose_dict[name + "_pos"]
                pos[0] = p.position.x
                pos[1] = p.position.y
                pos[2] = p.position.z
            self.has_fk = True

    def _assemble_fk_vector(self) -> np.ndarray | None:
        """Build a fresh (fk_dim,) float32 ndarray from the latest FK dict.

        Returns None until the first FK message arrives. A fresh ndarray is
        returned each call so it can be stored directly into the rolling
        buffer without aliasing the next-frame assembly.
        """
        if not self.has_fk:
            return None
        out = np.zeros(self.fk_dim, dtype=np.float32)
        with self.fk_lock:
            offset = 0
            for name in self.left_hand_base_names:
                out[offset:offset + 3] = self.fk_pose_dict[name + "_pos"]
                offset += 3
            for name in self.right_hand_base_names:
                out[offset:offset + 3] = self.fk_pose_dict[name + "_pos"]
                offset += 3
            for name in self.left_fingertip_names:
                out[offset:offset + 3] = self.fk_pose_dict[name + "_pos"]
                offset += 3
            for name in self.right_fingertip_names:
                out[offset:offset + 3] = self.fk_pose_dict[name + "_pos"]
                offset += 3
        return out

    # ------------------------------------------------------------------
    # RealSense
    # ------------------------------------------------------------------
    def _setup_realsense(self) -> None:
        # AprilTag PnP needs the full 640x480; the cotrain seg-finetune
        # dataset also wants hires color. Depth is decimated+hole-filled and
        # then aligned to color, yielding a 640x480 depth frame in the color
        # optical frame. CPU resize comes after for the policy-res view.
        self.rs_color_width  = 640
        self.rs_color_height = 480
        rs_cfg = self._rs_cfg
        sens_set = rs_cfg.get("sensor_settings", {})

        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 2)

        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.color, self.rs_color_width, self.rs_color_height,
            rs.format.bgr8, int(self.capture_hz),
        )
        rs_config.enable_stream(
            rs.stream.depth, self.rs_color_width, self.rs_color_height,
            rs.format.z16, int(self.capture_hz),
        )
        rs_profile = self.rs_pipeline.start(rs_config)
        self.align = rs.align(rs.stream.color)

        self._apply_sensor_settings(rs_profile, sens_set)

        # Depth scale sanity check — training (collect_policy_realsense.py +
        # train_depth_predictor.py) assumes raw_value * 1e-3 = metres. Refusing
        # to start on a different scale prevents silently mismatched units in
        # the offline finetune pipelines.
        try:
            self.depth_scale = float(
                rs_profile.get_device().first_depth_sensor().get_depth_scale()
            )
        except Exception:
            self.depth_scale = 0.001
        if abs(self.depth_scale - 0.001) > 1e-6:
            raise RuntimeError(
                f"RealSense depth_scale={self.depth_scale} != 0.001 m/unit; "
                f"training assumed mm-to-metres conversion."
            )

        # Live intrinsics for AprilTag PnP — taken from the device so they
        # match the actual capture (hires) stream, not the yaml's stored values.
        color_profile = rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self._K = np.array(
            [[intr.fx, 0.0,     intr.ppx],
             [0.0,     intr.fy, intr.ppy],
             [0.0,     0.0,     1.0]],
            dtype=np.float64,
        )
        self._dist = np.array(intr.coeffs, dtype=np.float64).reshape(-1)

        self.get_logger().info(
            f"RealSense capture started: {self.rs_color_width}x{self.rs_color_height}@{int(self.capture_hz)}Hz "
            f"(depth_scale={self.depth_scale})"
        )
        time.sleep(0.5)

    def _apply_sensor_settings(self, rs_profile, sensor_settings: dict[str, Any]) -> None:
        if not sensor_settings:
            return
        auto_exposure   = sensor_settings.get("auto_exposure", False)
        exposure        = sensor_settings.get("exposure")
        gain            = sensor_settings.get("gain")
        emitter_enabled = sensor_settings.get("emitter_enabled", None)
        laser_power     = sensor_settings.get("laser_power", None)
        for sensor in rs_profile.get_device().query_sensors():
            sensor_name = sensor.get_info(rs.camera_info.name)
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
            if not auto_exposure:
                if exposure is not None and sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, float(exposure))
                if gain is not None and sensor.supports(rs.option.gain):
                    sensor.set_option(rs.option.gain, float(gain))
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
                    f"{sensor.get_option(rs.option.laser_power)} mW"
                )

    # ------------------------------------------------------------------
    # AprilTag detector
    # ------------------------------------------------------------------
    def _setup_apriltag(self) -> None:
        dict_id = getattr(cv2.aruco, self.apriltag_dict_name, None)
        if dict_id is None:
            raise ValueError(
                f"OpenCV build doesn't ship dict {self.apriltag_dict_name}."
            )
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(int(dict_id))

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.01
        self._detector_params = params
        self._use_new_api = hasattr(cv2.aruco, "ArucoDetector")
        self._detector = (
            cv2.aruco.ArucoDetector(self._aruco_dict, params) if self._use_new_api else None
        )

        half = self.marker_size_m / 2.0
        self._obj_pts = np.array(
            [[-half,  half, 0.0],
             [ half,  half, 0.0],
             [ half, -half, 0.0],
             [-half, -half, 0.0]],
            dtype=np.float32,
        )
        self._axis_len = self.marker_size_m * 1.5

    def _detect_markers(self, gray: np.ndarray):
        if self._use_new_api and self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._detector_params
            )
        return corners, ids

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------
    def _init_publishers(self) -> None:
        self.poses_pub = (
            self.create_publisher(PoseArray, self.poses_topic, HIGH_RELIA_QOS)
            if self.publish_poses else None
        )
        self.geom_pub = (
            self.create_publisher(Float32MultiArray, self.geom_topic, HIGH_RELIA_QOS)
            if self.publish_geom else None
        )
        self.vis_pub = (
            self.create_publisher(Image, self.vis_topic, HIGH_RELIA_QOS)
            if self.enable_vis else None
        )

    # ------------------------------------------------------------------
    # Capture thread — owns wait_for_frames + detection + publish + collect
    # ------------------------------------------------------------------
    def _start_capture_thread(self) -> None:
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="apriltag_all_data_rs"
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Pull aligned (color, depth) frames at rs_fps, detect, publish, collect.

        Frame-rate driven: blocks on wait_for_frames at native rate, no sleeps.
        Heavy work (apriltag detect + buffer ops) all live here so we never
        race with the ROS executor thread.
        """
        frame_period = 1.0 / max(self.capture_hz, 1e-3)
        while self._capture_running and rclpy.ok():
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError as exc:
                self.get_logger().warn(f"RealSense wait_for_frames timed out: {exc}")
                continue
            except Exception as exc:
                self.get_logger().error(f"RealSense capture error: {exc}")
                self._capture_running = False
                rclpy.shutdown()
                return

            # Measure CPU work after the SDK has handed us a frame — the
            # blocking wait above is normal idle, not "slow code".
            proc_start = time.perf_counter()

            # SDK filter chain — same order as depth_feat_node_all_data so
            # the saved depth has identical characteristics.
            frames = self.decimation.process(frames).as_frameset()
            frames = self.hole_filling.process(frames).as_frameset()
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Owned copies (the SDK reuses its buffers next frame).
            color_hires = np.asanyarray(color_frame.get_data()).copy()  # (480,640,3) uint8
            depth_hires = np.asanyarray(depth_frame.get_data()).copy()  # (480,640)   uint16

            # CPU resize 640x480 -> 320x240.
            #   color: INTER_AREA   (clean low-res RGB, no aliasing)
            #   depth: INTER_NEAREST (no averaging across object edges —
            #     mean would invent mid-distance pixels at boundaries).
            color_lores = cv2.resize(
                color_hires, (_POLICY_WIDTH, _POLICY_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            depth_lores = cv2.resize(
                depth_hires, (_POLICY_WIDTH, _POLICY_HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )

            self._process_and_collect(color_hires, color_lores, depth_lores)

            elapsed = time.perf_counter() - proc_start
            if elapsed > frame_period:
                self.get_logger().warn(
                    f"[SLOW_PROC] processing={elapsed*1000:.1f}ms > "
                    f"frame_period={frame_period*1000:.1f}ms; falling behind capture rate."
                )

    def _process_and_collect(
        self,
        color_hires: np.ndarray,
        color_lores: np.ndarray,
        depth_lores: np.ndarray,
    ) -> None:
        """Detect tag on hires color, publish, append one frame to all buffers.

        Detection outcome decoupled from publish + buffer:
          * detected & filters pass: publish fresh poses, update last-good
              cache, buffer apriltag_valid=1 + body/cap.
          * miss / PnP fail / filter reject: republish last-good poses (if
              ever cached), buffer apriltag_valid=0 + zero positions.
          * geom + vis published every tick regardless.

        FK is always recorded (zeros until first FK message lands).
        """
        gray = cv2.cvtColor(color_hires, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect_markers(gray)
        vis = color_hires.copy()

        # ---- Detection branch ----
        body_world: np.ndarray | None = None
        cap_world:  np.ndarray | None = None
        rvec_for_axes = None
        tvec_for_axes = None

        chosen_idx: int | None = None
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            id_list = ids.flatten().tolist()
            if self.target_tag_id in id_list:
                chosen_idx = id_list.index(self.target_tag_id)

        if chosen_idx is not None:
            img_pts = corners[chosen_idx].reshape(-1, 2).astype(np.float32)
            flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_pts, img_pts, self._K, self._dist, flags=flag
            )
            if ok:
                R_cam_tag, _ = cv2.Rodrigues(rvec)
                T_cam_tag = np.eye(4, dtype=np.float64)
                T_cam_tag[:3, :3] = R_cam_tag
                T_cam_tag[:3, 3]  = tvec.reshape(3)
                T_world_tag = self._T_world_cam @ T_cam_tag

                tag_world   = T_world_tag[:3, 3] + self.pose_bias
                tag_z_world = T_world_tag[:3, 2]
                norm = float(np.linalg.norm(tag_z_world))
                if norm >= 1e-9:
                    tag_z_world = tag_z_world / norm

                    mirror = (self.reject_z_below_horizon and tag_z_world[2] < 0.0)
                    flipped = (
                        self.reject_sudden_flip
                        and self._prev_tag_z_world is not None
                        and float(np.dot(tag_z_world, self._prev_tag_z_world)) < 0.0
                    )

                    if not mirror and not flipped:
                        cap_w  = tag_world - self.cap_dz  * tag_z_world
                        body_w = tag_world - self.body_dz * tag_z_world
                        cap_w[2]  += self.workstation_top_z
                        body_w[2] += self.workstation_top_z
                        body_world = body_w.astype(np.float32)
                        cap_world  = cap_w.astype(np.float32)
                        self._prev_tag_z_world = tag_z_world
                        rvec_for_axes = rvec
                        tvec_for_axes = tvec

                        # Diagnostic log throttled to ~1 Hz @ 30 fps.
                        self._diag_counter += 1
                        if self._diag_counter >= 30:
                            self._diag_counter = 0
                            self.get_logger().info(
                                f"[tag@world] pos=({tag_world[0]:+.4f}, "
                                f"{tag_world[1]:+.4f}, {tag_world[2]:+.4f}) m  "
                                f"z=({tag_z_world[0]:+.3f}, {tag_z_world[1]:+.3f}, "
                                f"{tag_z_world[2]:+.3f})"
                            )

        # ---- Publish branch ----
        # Fresh detection: cache it and publish. Detect miss: republish the
        # last-good cache so apriltag_policy doesn't stutter.
        if body_world is not None and cap_world is not None:
            self._last_body_world = body_world
            self._last_cap_world  = cap_world
            self._publish_poses(body_world, cap_world)
        elif self._last_body_world is not None and self._last_cap_world is not None:
            self._publish_poses(self._last_body_world, self._last_cap_world)
            self.get_logger().warn(
                "AprilTag detect miss / filter reject — republished last-good pose."
            )
        else:
            # Never had a detection yet — nothing to republish.
            self.get_logger().warn(
                f"AprilTag id={self.target_tag_id} not detected and no cache yet."
            )

        if rvec_for_axes is not None and tvec_for_axes is not None:
            cv2.drawFrameAxes(
                vis, self._K, self._dist, rvec_for_axes, tvec_for_axes,
                self._axis_len, 3,
            )
        self._publish_geom()
        self._publish_vis(vis)

        # ---- Buffer append ----
        # On miss / filter-reject, body/cap fields are NaN (not zero) so any
        # offline code that forgets to filter by valid==1 gets NaN-propagated
        # losses immediately instead of silently training on (0,0,0). The
        # leading valid flag stays a real number (0 or 1) so a `row[:, 0] == 1`
        # mask still works for filtering.
        apriltag_row = np.full(7, np.nan, dtype=np.float32)
        apriltag_row[0] = 0.0
        if body_world is not None and cap_world is not None:
            apriltag_row[0]   = 1.0
            apriltag_row[1:4] = body_world
            apriltag_row[4:7] = cap_world

        fk_np = self._assemble_fk_vector()
        if fk_np is None:
            fk_np = np.zeros(self.fk_dim, dtype=np.float32)

        with self._rolling_lock:
            self._rgb_hires_buf.append(color_hires)
            self._rgb_lores_buf.append(color_lores)
            self._depth_buf.append(depth_lores)
            self._fk_buf.append(fk_np)
            self._apriltag_buf.append(apriltag_row)
            self._timestamp_buf.append(time.time())

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------
    def _publish_poses(self, body_world: np.ndarray, cap_world: np.ndarray) -> None:
        if self.poses_pub is None:
            return
        body = Pose()
        body.position.x = float(body_world[0])
        body.position.y = float(body_world[1])
        body.position.z = float(body_world[2])
        body.orientation.w = 1.0
        cap = Pose()
        cap.position.x = float(cap_world[0])
        cap.position.y = float(cap_world[1])
        cap.position.z = float(cap_world[2])
        cap.orientation.w = 1.0
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame_id
        msg.poses = [body, cap]
        self.poses_pub.publish(msg)

    def _publish_geom(self) -> None:
        if self.geom_pub is None:
            return
        msg = Float32MultiArray()
        msg.data = self.bottle_geom.tolist()
        self.geom_pub.publish(msg)

    def _publish_vis(self, color_bgr: np.ndarray) -> None:
        if self.vis_pub is None:
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = color_bgr.shape[0]
        msg.width  = color_bgr.shape[1]
        msg.encoding = "bgr8"
        msg.step = color_bgr.shape[1] * 3
        msg.data = np.ascontiguousarray(color_bgr).tobytes()
        self.vis_pub.publish(msg)

    # ------------------------------------------------------------------
    # Stdin save trigger
    # ------------------------------------------------------------------
    def _start_stdin_listener(self) -> None:
        self._stdin_thread = threading.Thread(
            target=self._stdin_listener_loop, daemon=True, name="cotrain_stdin"
        )
        self._stdin_thread.start()

    def _stdin_listener_loop(self) -> None:
        while self._stdin_running and rclpy.ok():
            try:
                line = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt):
                return
            if not line:
                return
            if not self._stdin_running:
                return
            self._handle_save_trigger()

    def _handle_save_trigger(self) -> None:
        """Snapshot the six rolling deques into a .npz + meta.json. Buffers
        keep rolling — pressing Enter again captures the next 300-frame window
        (with overlap if pressed quickly)."""
        with self._rolling_lock:
            n_buf = len(self._rgb_hires_buf)
            if n_buf == 0:
                print("[cotrain] buffer empty — no frame received yet, skipping save.")
                return
            rgb_hires_snap = list(self._rgb_hires_buf)
            rgb_lores_snap = list(self._rgb_lores_buf)
            depth_snap     = list(self._depth_buf)
            fk_snap        = list(self._fk_buf)
            apriltag_snap  = list(self._apriltag_buf)
            ts_snap        = list(self._timestamp_buf)

        rgb_hires_arr = np.stack(rgb_hires_snap, axis=0)      # (N,480,640,3) uint8
        rgb_lores_arr = np.stack(rgb_lores_snap, axis=0)      # (N,240,320,3) uint8
        depth_arr     = np.stack(depth_snap,     axis=0)      # (N,240,320)   uint16
        fk_arr        = np.stack(fk_snap,        axis=0)      # (N, fk_dim)   float32
        apriltag_arr  = np.stack(apriltag_snap,  axis=0)      # (N, 7)        float32
        ts_arr        = np.asarray(ts_snap, dtype=np.float64) # (N,)          float64

        n_apriltag_valid = int(apriltag_arr[:, 0].sum())

        try:
            _OUT_DIR.mkdir(parents=True, exist_ok=True)
            sample_name = datetime.now().strftime("sample_%Y%m%d_%H%M%S_%f")[:-3]
            sample_dir = _OUT_DIR / sample_name
            sample_dir.mkdir(parents=True, exist_ok=False)

            npz_path = sample_dir / "data.npz"
            np.savez(
                str(npz_path),
                rgb_hires = rgb_hires_arr,
                rgb_lores = rgb_lores_arr,
                depth     = depth_arr,
                fk        = fk_arr,
                apriltag  = apriltag_arr,
                timestamp = ts_arr,
            )

            meta_path = sample_dir / "meta.json"
            meta = {
                "source":               "bottle_apriltag_node_all_data",
                "n_frames":             n_buf,
                "n_apriltag_valid":     n_apriltag_valid,
                "capture_resolution":   [self.rs_color_width, self.rs_color_height],
                "policy_resolution":    [_POLICY_WIDTH, _POLICY_HEIGHT],
                "capture_hz":           self.capture_hz,
                "fk_dim":               int(self.fk_dim),
                "apriltag_dict":        self.apriltag_dict_name,
                "apriltag_id":          int(self.target_tag_id),
                "marker_size_m":        float(self.marker_size_m),
                "cap_dz":               float(self.cap_dz),
                "body_dz":              float(self.body_dz),
                "bottle_geom":          self.bottle_geom.tolist(),
                "pose_bias":            self.pose_bias.tolist(),
                "workstation_top_z":    float(self.workstation_top_z),
                "T_world_cam":          self._T_world_cam.tolist(),
                "fk_topic":             self.fk_topic,
                "fk_link_names":        list(self.fk_link_names),
                "apriltag_row_layout":  "[valid, body_x, body_y, body_z, cap_x, cap_y, cap_z]; "
                                        "valid=0 rows have NaN positions (NOT inherited from "
                                        "last-good — NaN propagates so forgetting to filter "
                                        "by valid==1 crashes the loss instead of silently "
                                        "training on bad GT).",
                "saved_at":             datetime.now().isoformat(),
            }
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            self._save_counter += 1
            print(
                f"[cotrain] saved sample #{self._save_counter} -> {sample_dir.name}\n"
                f"  npz:       {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)\n"
                f"  frames:    {n_buf}\n"
                f"  apriltag:  {n_apriltag_valid}/{n_buf} valid "
                f"({100 * n_apriltag_valid / max(n_buf,1):.1f}%)\n"
                f"  buffers continue rolling — press ENTER again for next sample."
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[cotrain] save failed: {exc}")

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def destroy_node(self) -> bool:
        self._capture_running = False
        self._stdin_running = False
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
    node = BottleAprilTagNodeAllData()
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
