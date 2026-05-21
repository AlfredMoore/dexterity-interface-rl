"""ROS 2 node: AprilTag-based bottle pose publisher.

Single-threaded frame-driven loop:
    while ok:
        wait_for_frames
        detect AprilTag
        solvePnP -> T_cam_tag
        T_world_tag = T_world_cam @ T_cam_tag
        filters (mirror / sudden-flip)
        cap_world  = tag_world - cap_dz  * tag_z_in_world
        body_world = tag_world - body_dz * tag_z_in_world
        publish (PoseArray, Float32MultiArray geom, Image viz)

No ROS timer, no capture thread, no IPC, no CUDA — every published message
comes from the exact RealSense frame the detector just looked at. Lower
latency and less CPU than the timer-based depth_feat_node design, because
AprilTag detection is fast and stable enough that frame-rate publishing is
the right cadence.

Filters (configurable in april_tag_node_config.yaml::filters):
  * `tag_z_world.z < 0`           → reject PnP mirror solution
  * `dot(tag_z, prev_tag_z) < 0`  → reject sudden axis flip

On a missed / rejected detection the node skips the pose+geom publish for
that tick (warns to ROS log) but still publishes the viz image so the
operator can see what the camera sees.

Configs:
  * april_tag_node_config.yaml  — tag dict, marker size, offsets, geom
                                  override, topic names, filter toggles.
  * realsense_config.yaml       — sensor settings (exposure/gain/laser etc).
                                  Color resolution is hardcoded to 640x480
                                  for PnP accuracy.
  * runtime/rs_config.yaml      — T_world_cam (from realsense_artag_cali.py).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from robot_motion_interface.utils.qos import HIGH_RELIA_QOS


# ── Path resolution (mirrors gsam2_node / depth_feat_node) ──────────────────
WORKSPACE_ROOT = Path("/workspace")
APRILTAG_CONFIG_PATH    = "libs/robot_motion_interface/config/april_tag_node_config.yaml"
REALSENSE_CONFIG_PATH   = "libs/robot_motion_interface/config/realsense_config.yaml"
EXTRINSICS_CONFIG_PATH  = "libs/robot_motion_interface/runtime/rs_config.yaml"


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


class BottleAprilTagNode(Node):
    def __init__(self) -> None:
        super().__init__("bottle_apriltag_node")

        self._declare_parameters()
        self._load_configs()

        # Filter state — last accepted tag-z in world frame (unit vector).
        self._prev_tag_z_world: np.ndarray | None = None
        # Loop control flag — flipped by run() / destroy_node().
        self._running = True
        # Diagnostic frame counter — throttles the per-frame "tag in world"
        # log so we don't spam stdout at full capture rate.
        self._diag_counter: int = 0

        self._setup_realsense()
        self._setup_apriltag()
        self._init_publishers()

        self.get_logger().info(
            f"BottleAprilTagNode ready:\n"
            f"  color     = {self.rs_color_width}x{self.rs_color_height}@{int(self.capture_hz)}Hz (frame-driven)\n"
            f"  tag       = {self.apriltag_dict_name}  id={self.target_tag_id}  size={self.marker_size_m*1000:.1f}mm\n"
            f"  offsets   = cap-{self.cap_dz*1000:.1f}mm  body-{self.body_dz*1000:.1f}mm (along tag +z)\n"
            f"  pose_bias = ({self.pose_bias[0]*1000:+.1f}, {self.pose_bias[1]*1000:+.1f}, {self.pose_bias[2]*1000:+.1f}) mm (world frame)\n"
            f"  ws_top_z  = {self.workstation_top_z:+.5f} m (frame lift, added to cap/body z)\n"
            f"  geom      = {self.bottle_geom.tolist()}\n"
            f"  filters   = reject_z_below_horizon={self.reject_z_below_horizon}  reject_sudden_flip={self.reject_sudden_flip}\n"
            f"  topics    = {self.poses_topic}  {self.geom_topic}  {self.vis_topic}"
        )

    # ------------------------------------------------------------------
    # init: parameters / configs
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter("apriltag_config_path",   APRILTAG_CONFIG_PATH)
        self.declare_parameter("realsense_config_path",  REALSENSE_CONFIG_PATH)
        self.declare_parameter("extrinsics_config_path", EXTRINSICS_CONFIG_PATH)
        self.apriltag_config_path = _resolve_workspace_path(
            self.get_parameter("apriltag_config_path").value
        )
        self.realsense_config_path = _resolve_workspace_path(
            self.get_parameter("realsense_config_path").value
        )
        self.extrinsics_config_path = _resolve_workspace_path(
            self.get_parameter("extrinsics_config_path").value
        )

    def _load_configs(self) -> None:
        cfg = _load_yaml(self.apriltag_config_path)
        rs_cfg_full = _load_yaml(self.realsense_config_path)
        ext_cfg = _load_yaml(self.extrinsics_config_path)

        self._rs_cfg = rs_cfg_full["realsense"]

        # ---- AprilTag ----
        apriltag_cfg = cfg["apriltag"]
        self.apriltag_dict_name = str(apriltag_cfg["dict"])
        self.target_tag_id      = int(apriltag_cfg["tag_id"])
        self.marker_size_m      = float(apriltag_cfg["marker_size_m"])

        # ---- Static bottle geom override ----
        # Loaded first because the geometry offsets (cap_dz / body_dz) are
        # derived from it below.
        bg = cfg["bottle_geom"]
        if len(bg) != 4:
            raise ValueError(
                f"bottle_geom must be a 4-element list [body_radius, body_height, "
                f"cap_radius, cap_height]; got {bg}"
            )
        self.bottle_geom = np.array(bg, dtype=np.float32)

        # ---- Geometry offsets (derived from bottle_geom) ----
        # bottle_geom layout: [body_radius, body_height, cap_radius, cap_height]
        # Tag is glued flat on the cap top, so along the tag's +z axis:
        #   cap_dz  = cap_height / 2                  (tag center → cap center)
        #   body_dz = cap_height + body_height / 2    (tag center → body center)
        body_h = float(self.bottle_geom[1])
        cap_h  = float(self.bottle_geom[3])
        self.cap_dz  = cap_h / 2.0
        self.body_dz = cap_h + body_h / 2.0

        geom_cfg = cfg.get("geometry", {})
        # World-frame additive bias on tag center — compensates for cali tag
        # mounting offset / z=0 plane not exactly on the tabletop. Same shift
        # applies to cap and body (they're both derived from the bias-shifted
        # tag_world). All three components default to 0 if absent from yaml.
        self.pose_bias = np.array(
            [
                float(geom_cfg.get("pose_bias_x", 0.0)),
                float(geom_cfg.get("pose_bias_y", 0.0)),
                float(geom_cfg.get("pose_bias_z", 0.0)),
            ],
            dtype=np.float64,
        )
        # Frame conversion constant: lift apriltag world (z=0 at tabletop)
        # into the policy obs frame (z=0 at robot_base + workstation_top/2).
        # Added to both cap_world.z and body_world.z after the tag_z offset
        # subtraction. Default 0 keeps backwards compatibility.
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

        # ---- RealSense capture rate (for logs / SLOW warns) ----
        self.capture_hz = float(self._rs_cfg["rs_fps"])

    # ------------------------------------------------------------------
    # RealSense
    # ------------------------------------------------------------------
    def _setup_realsense(self) -> None:
        # AprilTag PnP needs the higher pixel count; deployment 320x240 is
        # too coarse for stable corner sub-pixel refinement. Hardcoded 640x480
        # here regardless of what realsense_config.yaml says.
        self.rs_color_width  = 640
        self.rs_color_height = 480
        sens_set = self._rs_cfg.get("sensor_settings", {})

        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.color,
            self.rs_color_width,
            self.rs_color_height,
            rs.format.bgr8,
            int(self.capture_hz),
        )
        rs_profile = self.rs_pipeline.start(rs_config)

        self._apply_sensor_settings(rs_profile, sens_set)

        # Live intrinsics (device-reported — matches the actual stream).
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
            f"RealSense capture started: {self.rs_color_width}x{self.rs_color_height}@{int(self.capture_hz)}Hz"
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

    def _pose_from_corners(self, img_pts: np.ndarray):
        img_pts = img_pts.reshape(-1, 2).astype(np.float32)
        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(self._obj_pts, img_pts, self._K, self._dist, flags=flag)
        return ok, rvec, tvec

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
    # Frame-driven main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Block on RealSense frames; process + publish each one immediately.

        No timer / executor — this is the entire control flow. Returns when
        rclpy shuts down or `self._running` is flipped (e.g. by destroy_node).
        """
        # frame_period is what we use to decide "did we miss the next frame".
        # The loop is self-throttled by wait_for_frames (blocks until the next
        # RealSense frame arrives ~ every 1/rs_fps seconds) — no explicit sleep
        # needed; adding one would only push us into the next frame queue.
        frame_period = 1.0 / max(self.capture_hz, 1e-3)
        while self._running and rclpy.ok():
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError as exc:
                self.get_logger().warn(f"RealSense wait_for_frames timed out: {exc}")
                continue
            except Exception as exc:
                self.get_logger().error(f"RealSense capture error: {exc}")
                return

            # Start measuring *after* a frame is available — anything before
            # this is just SDK blocking for the next capture, not slow code.
            proc_start = time.perf_counter()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_bgr = np.asanyarray(color_frame.get_data())   # zero-copy view
            self._process_frame(color_bgr)

            elapsed = time.perf_counter() - proc_start
            if elapsed > frame_period:
                # Processing one frame took longer than the inter-frame gap —
                # we're starting to fall behind 30 fps. Warn only here; the
                # blocking wait_for_frames time is normal and not a problem.
                self.get_logger().warn(
                    f"[SLOW_PROC] processing={elapsed*1000:.1f}ms > "
                    f"frame_period={frame_period*1000:.1f}ms; falling behind capture rate."
                )

    def _process_frame(self, color_bgr: np.ndarray) -> None:
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect_markers(gray)
        vis = color_bgr.copy()

        chosen_idx: int | None = None
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            id_list = ids.flatten().tolist()
            if self.target_tag_id in id_list:
                chosen_idx = id_list.index(self.target_tag_id)

        if chosen_idx is None:
            self.get_logger().warn(
                f"AprilTag id={self.target_tag_id} not detected — no PoseArray this tick."
            )
            self._publish_vis(vis)
            return

        ok, rvec, tvec = self._pose_from_corners(corners[chosen_idx])
        if not ok:
            self.get_logger().warn("solvePnP failed — no PoseArray this tick.")
            self._publish_vis(vis)
            return

        # cam -> tag in world frame.
        R_cam_tag, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4, dtype=np.float64)
        T_cam_tag[:3, :3] = R_cam_tag
        T_cam_tag[:3, 3] = tvec.reshape(3)
        T_world_tag = self._T_world_cam @ T_cam_tag

        # Apply world-frame bias here (additive on translation only). Cap and
        # body both inherit the same shift since they're derived from
        # tag_world. tag_z (rotation column) is unaffected — filters still see
        # the physical orientation, not the post-bias direction.
        tag_world  = T_world_tag[:3, 3] + self.pose_bias
        tag_z_world = T_world_tag[:3, 2]
        norm = np.linalg.norm(tag_z_world)
        if norm < 1e-9:
            self.get_logger().warn("Degenerate tag rotation — skipping.")
            self._publish_vis(vis)
            return
        tag_z_world = tag_z_world / norm

        # Diagnostic log — print tag-in-world before any filter / offset is
        # applied, so we can sanity-check the PnP + extrinsics chain
        # independently from cap/body derivation. Throttled to ~1 Hz @ 30 fps.
        self._diag_counter += 1
        if self._diag_counter >= 30:
            self._diag_counter = 0
            self.get_logger().info(
                f"[tag@world] pos=({tag_world[0]:+.4f}, {tag_world[1]:+.4f}, {tag_world[2]:+.4f}) m  "
                f"z_axis=({tag_z_world[0]:+.3f}, {tag_z_world[1]:+.3f}, {tag_z_world[2]:+.3f})  "
                f"cam_dist={float(np.linalg.norm(T_cam_tag[:3, 3])):.3f} m"
            )

        # Filter 1: PnP mirror solution (tag z pointing down through world).
        if self.reject_z_below_horizon and tag_z_world[2] < 0.0:
            self.get_logger().warn(
                f"tag_z_world.z = {tag_z_world[2]:+.3f} < 0 — likely PnP mirror, skipping."
            )
            self._publish_vis(vis)
            return

        # Filter 2: sudden axis flip vs last accepted detection.
        if self.reject_sudden_flip and self._prev_tag_z_world is not None:
            cos_angle = float(np.dot(tag_z_world, self._prev_tag_z_world))
            if cos_angle < 0.0:
                self.get_logger().warn(
                    f"tag z flipped vs last frame (dot={cos_angle:+.3f}) — skipping."
                )
                self._publish_vis(vis)
                return

        # Both checks passed — derive landmarks along the bottle axis.
        cap_world  = tag_world - self.cap_dz  * tag_z_world
        body_world = tag_world - self.body_dz * tag_z_world
        # Lift into policy obs frame (z=0 at robot_base + workstation_top/2).
        # Applied identically to both so cap-body relative geometry is intact.
        cap_world[2]  += self.workstation_top_z
        body_world[2] += self.workstation_top_z

        self._prev_tag_z_world = tag_z_world

        # Overlay axes on the color frame.
        cv2.drawFrameAxes(vis, self._K, self._dist, rvec, tvec, self._axis_len, 3)

        self._publish_poses(body_world.astype(np.float32), cap_world.astype(np.float32))
        self._publish_geom()
        self._publish_vis(vis)

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
    # shutdown
    # ------------------------------------------------------------------
    def destroy_node(self) -> bool:
        self._running = False
        if hasattr(self, "rs_pipeline") and self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except Exception as exc:
                self.get_logger().error(f"Error stopping RealSense pipeline: {exc}")
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BottleAprilTagNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
