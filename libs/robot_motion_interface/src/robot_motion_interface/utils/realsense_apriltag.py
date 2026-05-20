"""
RealSense AprilTag live visualisation in the WORLD frame.

Slim companion to realsense_artag_cali.py — same detection + solvePnP path,
but stripped of the world-frame redefinition, multi-convention quaternion
output, and YAML save logic. Use this to sanity-check a freshly printed +
mounted tag *before* plugging it into a downstream compensation pipeline:

  - detect AprilTag (or any ArUco dict via --aruco-dict)
  - estimate T_cam_tag via solvePnP (IPPE_SQUARE)
  - load T_world_cam from the calibration yaml (default
    runtime/rs_config.yaml, produced by realsense_artag_cali.py)
  - report T_world_tag = T_world_cam @ T_cam_tag — i.e. the tag's pose
    in the same world frame the rest of the stack (FK, policy) uses
  - overlay tag axes, ID, world position, FPS on the live color stream
  - print throttled per-frame stats to stdout

Run:
  python -m robot_motion_interface.utils.realsense_apriltag \
      --marker-size 0.030          # measured edge in meters (use calipers)

Optional:
  --aruco-dict DICT_APRILTAG_36h11   (default; same picker as cali script)
  --target-tag-id 0                  (default: show every detected tag)
  --extrinsics-config <path>         (default: <RMI>/runtime/rs_config.yaml)
  --print-every-n 30                 (stdout throttle in frames)

Keys (click the OpenCV window first for keyboard focus):
  q       quit
  Ctrl-C  also quits cleanly
"""

import argparse
import importlib.util
import os
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

# ── Tunable defaults ────────────────────────────────────────────────────────
DEFAULT_ARUCO_DICT_NAME = "DICT_APRILTAG_36h11"
AXIS_THICKNESS = 3
DEFAULT_PRINT_EVERY_N = 30

# Same dict list as the calibration script so users can try a different
# family/dict without having to remember which strings OpenCV ships with.
_DICT_CANDIDATES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]


# ── Path resolution (same pattern as realsense_artag_cali.py) ───────────────
spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent
DEFAULT_CONFIG_PATH = RMI_ROOT / "config" / "realsense_config.yaml"
DEFAULT_EXTRINSICS_PATH = RMI_ROOT / "runtime" / "rs_config.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live AprilTag visualisation from a RealSense color stream."
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        required=True,
        help="AprilTag physical edge length in METERS (measure with calipers, e.g. 0.030).",
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default=DEFAULT_ARUCO_DICT_NAME,
        choices=_DICT_CANDIDATES,
        help=f"OpenCV ArUco dictionary (default: {DEFAULT_ARUCO_DICT_NAME}).",
    )
    parser.add_argument(
        "--target-tag-id",
        type=int,
        default=None,
        help="Optional ID filter. Default: visualise every detected tag.",
    )
    parser.add_argument(
        "--print-every-n",
        type=int,
        default=DEFAULT_PRINT_EVERY_N,
        help=f"stdout throttle in frames (default: {DEFAULT_PRINT_EVERY_N}).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.resolve()),
        help="Path to realsense_config.yaml.",
    )
    parser.add_argument(
        "--extrinsics-config",
        default=str(DEFAULT_EXTRINSICS_PATH.resolve()),
        help=(
            "Path to the extrinsics yaml saved by realsense_artag_cali.py "
            "(must contain a top-level T_world_cam 4x4 matrix). Used to map "
            "the detected tag pose from camera frame into world frame."
        ),
    )
    return parser.parse_args()


def _require_keys(config: dict, config_path: str) -> None:
    required_keys = (
        ("realsense",),
        ("realsense", "rs_fps"),
        ("realsense", "sensor_settings"),
        ("realsense", "color_intrinsics"),
        ("realsense", "color_intrinsics", "width"),
        ("realsense", "color_intrinsics", "height"),
        ("realsense", "depth_intrinsics"),
        ("realsense", "depth_intrinsics", "width"),
        ("realsense", "depth_intrinsics", "height"),
    )
    missing = []
    for key_path in required_keys:
        node = config
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                missing.append(".".join(key_path))
                break
            node = node[key]
    if missing:
        raise KeyError(f"Missing required key(s) in {config_path}: {', '.join(missing)}")


ARGS = _parse_args()

if ARGS.marker_size <= 0.0:
    raise ValueError(f"--marker-size must be > 0, got {ARGS.marker_size}")

ARUCO_DICT_NAME = ARGS.aruco_dict
ARUCO_DICT = getattr(cv2.aruco, ARUCO_DICT_NAME, None)
if ARUCO_DICT is None:
    raise ValueError(f"Aruco dictionary not available in this OpenCV build: {ARUCO_DICT_NAME}")

MARKER_SIZE_M = float(ARGS.marker_size)
TARGET_TAG_ID = ARGS.target_tag_id  # may be None
AXIS_LENGTH_M = MARKER_SIZE_M * 1.5
PRINT_EVERY_N = max(1, int(ARGS.print_every_n))

# ── Config ─────────────────────────────────────────────────────────────────
config_path = str(Path(ARGS.config).expanduser().resolve())
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at: {config_path}")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
if not isinstance(config, dict):
    raise ValueError(f"Config root must be a dict: {config_path}")
_require_keys(config, config_path)

_rs_config = config["realsense"]
_rs_fps = _rs_config["rs_fps"]
_sensor_settings = _rs_config.get("sensor_settings", {})
_c_intrinsics = _rs_config["color_intrinsics"]
_c_intrinsics["width"] = 640    # hardcoded hear
_c_intrinsics["height"] = 480
_d_intrinsics = _rs_config["depth_intrinsics"]

print(
    "RealSense config loaded:\n"
    f"  path={config_path}\n"
    f"  rs_fps={_rs_fps}\n"
    f"  color={_c_intrinsics['width']}x{_c_intrinsics['height']}"
)

# ── Camera extrinsics (T_world_cam) ─────────────────────────────────────────
# Loaded once at startup from the yaml saved by realsense_artag_cali.py.
# Convention (matches cali script): T_world_cam transforms a point from the
# camera frame into the world frame, i.e. world_pt = T_world_cam @ cam_pt.
# We chain T_world_tag = T_world_cam @ T_cam_tag per frame.
extrinsics_path = str(Path(ARGS.extrinsics_config).expanduser().resolve())
if not os.path.exists(extrinsics_path):
    raise FileNotFoundError(
        f"Extrinsics yaml not found at {extrinsics_path}. "
        f"Run realsense_artag_cali.py first to produce it (autosaves to "
        f"<RMI>/runtime/rs_config_*.yaml) and either rename to rs_config.yaml "
        f"or pass --extrinsics-config <path>."
    )
with open(extrinsics_path, "r") as f:
    _ext_cfg = yaml.safe_load(f)
if "T_world_cam" not in _ext_cfg:
    raise KeyError(
        f"{extrinsics_path} has no top-level 'T_world_cam' field. Was it "
        f"saved by realsense_artag_cali.py? Re-run calibration to regenerate."
    )
T_WORLD_CAM = np.array(_ext_cfg["T_world_cam"], dtype=np.float64)
if T_WORLD_CAM.shape != (4, 4):
    raise ValueError(
        f"T_world_cam in {extrinsics_path} has shape {T_WORLD_CAM.shape}, "
        f"expected (4, 4)."
    )
print(
    f"Extrinsics loaded:\n"
    f"  path={extrinsics_path}\n"
    f"  T_world_cam translation = {T_WORLD_CAM[:3, 3].tolist()}"
)

# ── RealSense init ──────────────────────────────────────────────────────────
rs_pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(
    rs.stream.color,
    _c_intrinsics["width"],
    _c_intrinsics["height"],
    rs.format.bgr8,
    _rs_fps,
)
# Depth stream isn't strictly needed for PnP — we run AprilTag detection on
# the color image only — but keep it enabled so this script behaves identically
# to the rest of the pipeline (realsense_test, depth_feat_node, etc.) and any
# future debug code can read aligned depth without re-touching the config.
rs_config.enable_stream(
    rs.stream.depth,
    _d_intrinsics["width"],
    _d_intrinsics["height"],
    rs.format.z16,
    _rs_fps,
)
rs_profile = rs_pipeline.start(rs_config)
rs_align = rs.align(rs.stream.color)


def _apply_sensor_settings(profile: rs.pipeline_profile) -> None:
    """Apply exposure/gain + emitter/laser_power from realsense_config.yaml.

    Mirrors the other nodes so the visualiser sees the same image the depth
    node + YOLO node see (consistent IR illumination, exposure, etc.).
    """
    if not _sensor_settings:
        return
    try:
        sensors = profile.get_device().query_sensors()
    except Exception:
        return

    auto_exposure   = _sensor_settings.get("auto_exposure", False)
    exposure        = _sensor_settings.get("exposure", 350)
    gain            = _sensor_settings.get("gain", 16)
    emitter_enabled = _sensor_settings.get("emitter_enabled", None)
    laser_power     = _sensor_settings.get("laser_power", None)

    for sensor in sensors:
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
            print(f"  [{sensor_name}] emitter_enabled -> {sensor.get_option(rs.option.emitter_enabled)}")
        if laser_power is not None and sensor.supports(rs.option.laser_power):
            lp_range = sensor.get_option_range(rs.option.laser_power)
            clamped = max(lp_range.min, min(float(laser_power), lp_range.max))
            sensor.set_option(rs.option.laser_power, clamped)
            print(
                f"  [{sensor_name}] laser_power -> {sensor.get_option(rs.option.laser_power)} mW "
                f"(yaml={laser_power}, range {lp_range.min}..{lp_range.max})"
            )


_apply_sensor_settings(rs_profile)
print("Sensor settings applied")


# ── Intrinsics (live from device) ───────────────────────────────────────────
_color_profile = rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
_color_intr = _color_profile.get_intrinsics()
K = np.array(
    [
        [_color_intr.fx, 0.0,           _color_intr.ppx],
        [0.0,           _color_intr.fy, _color_intr.ppy],
        [0.0,           0.0,            1.0],
    ],
    dtype=np.float64,
)
dist = np.array(_color_intr.coeffs, dtype=np.float64).reshape(-1)
print(f"K (live) =\n{K}")
print(f"dist coeffs = {dist.tolist()}")


# ── ArUco / AprilTag detector ───────────────────────────────────────────────
detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector_params.cornerRefinementWinSize = 5
detector_params.cornerRefinementMaxIterations = 30
detector_params.cornerRefinementMinAccuracy = 0.01
_USE_NEW_API = hasattr(cv2.aruco, "ArucoDetector")

aruco_dict = cv2.aruco.getPredefinedDictionary(int(ARUCO_DICT))
if _USE_NEW_API:
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
else:
    detector = None  # legacy fallback uses module function

print(
    f"Detector ready: dict={ARUCO_DICT_NAME}  marker_size={MARKER_SIZE_M:.4f} m  "
    f"target_id={'ANY' if TARGET_TAG_ID is None else TARGET_TAG_ID}  "
    f"new_api={_USE_NEW_API}"
)


def _detect_markers(gray: np.ndarray):
    if _USE_NEW_API and detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
    return corners, ids


# Marker-frame 3D corner coords (centered at origin, +Z out of page).
# Order matches detectMarkers: top-left, top-right, bottom-right, bottom-left.
_HALF = MARKER_SIZE_M / 2.0
_OBJ_PTS = np.array(
    [
        [-_HALF,  _HALF, 0.0],
        [ _HALF,  _HALF, 0.0],
        [ _HALF, -_HALF, 0.0],
        [-_HALF, -_HALF, 0.0],
    ],
    dtype=np.float32,
)


def _pose_from_corners(img_pts: np.ndarray):
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(_OBJ_PTS, img_pts, K, dist, flags=flag)
    return ok, rvec, tvec


def _draw_labeled_axes(
    img: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    length: float,
    thickness: int,
    label_prefix: str = "",
) -> None:
    """Project + draw XYZ axes (RGB = +X, +Y, +Z) with tip labels."""
    axis_pts = np.array(
        [
            [0.0,    0.0,    0.0],
            [length, 0.0,    0.0],
            [0.0,    length, 0.0],
            [0.0,    0.0,    length],
        ],
        dtype=np.float32,
    )
    img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1, 2).astype(int)
    o, x, y, z = (tuple(p) for p in img_pts)

    cv2.line(img, o, x, (0, 0, 255), thickness)   # +X = red (BGR)
    cv2.line(img, o, y, (0, 255, 0), thickness)   # +Y = green
    cv2.line(img, o, z, (255, 0, 0), thickness)   # +Z = blue

    def _put(text: str, pos: tuple, fill: tuple) -> None:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, fill,      1, cv2.LINE_AA)

    _put(f"{label_prefix}x", x, (0, 0, 255))
    _put(f"{label_prefix}y", y, (0, 255, 0))
    _put(f"{label_prefix}z", z, (255, 0, 0))


def _euler_deg_from_R(R: np.ndarray) -> np.ndarray:
    """ZYX intrinsic Euler angles (deg) — gimbal-safe-ish, for stdout only."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z], dtype=np.float64))


# ── Main loop ───────────────────────────────────────────────────────────────
_should_quit = False


def _on_sigint(signum, _frame):
    global _should_quit
    _should_quit = True
    print("\n[signal] SIGINT received; exiting cleanly.")


signal.signal(signal.SIGINT, _on_sigint)

cv2.namedWindow("apriltag_viz", cv2.WINDOW_AUTOSIZE)

print(
    "Starting loop. Press 'q' to quit. Click on the OpenCV window first to give "
    "it keyboard focus. Ctrl-C also exits cleanly."
)

loop_idx = 0
last_t = time.time()
fps_ema = 0.0

try:
    while True:
        try:
            frames = rs_pipeline.wait_for_frames(timeout_ms=2000)
            if not frames:
                time.sleep(0.001)
                continue
        except RuntimeError:
            print("[warn] wait_for_frames timed out, retrying...")
            time.sleep(0.5)
            continue

        aligned_frames = rs_align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids = _detect_markers(gray)
        vis = color.copy()

        n_drawn = 0
        # HUD info from the first plotted tag: (tag_id, world_t, cam_dist)
        first_pose: tuple | None = None

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            id_list = ids.flatten().tolist()

            for idx, tag_id in enumerate(id_list):
                # Optional ID filter
                if TARGET_TAG_ID is not None and tag_id != TARGET_TAG_ID:
                    continue

                ok, rvec, tvec = _pose_from_corners(corners[idx])
                if not ok:
                    continue

                # Build T_cam_tag (cam <- tag) then chain into world frame.
                R_cam_tag, _ = cv2.Rodrigues(rvec)
                T_cam_tag = np.eye(4, dtype=np.float64)
                T_cam_tag[:3, :3] = R_cam_tag
                T_cam_tag[:3, 3] = tvec.reshape(3)
                T_world_tag = T_WORLD_CAM @ T_cam_tag
                world_t = T_world_tag[:3, 3]
                world_R = T_world_tag[:3, :3]

                _draw_labeled_axes(
                    vis, rvec, tvec, AXIS_LENGTH_M, AXIS_THICKNESS,
                    label_prefix=f"{tag_id}_",
                )
                n_drawn += 1
                cam_dist = float(np.linalg.norm(T_cam_tag[:3, 3]))
                if first_pose is None:
                    first_pose = (int(tag_id), world_t.copy(), cam_dist)

                if loop_idx % PRINT_EVERY_N == 0:
                    cam_t = T_cam_tag[:3, 3]
                    world_rpy = _euler_deg_from_R(world_R)
                    print(
                        f"[{loop_idx:6d}] tag={int(tag_id):3d}  "
                        f"world_t=({world_t[0]:+.3f},{world_t[1]:+.3f},{world_t[2]:+.3f}) m  "
                        f"world_rpy=({world_rpy[0]:+.1f},{world_rpy[1]:+.1f},{world_rpy[2]:+.1f}) deg  "
                        f"cam_t=({cam_t[0]:+.3f},{cam_t[1]:+.3f},{cam_t[2]:+.3f}) m  "
                        f"cam_dist={cam_dist:.3f} m"
                    )

            if TARGET_TAG_ID is not None and TARGET_TAG_ID not in id_list:
                if loop_idx % PRINT_EVERY_N == 0:
                    print(
                        f"[{loop_idx:6d}] target_tag_id={TARGET_TAG_ID} not in detected ids={id_list}"
                    )
        else:
            if loop_idx % PRINT_EVERY_N == 0:
                print(f"[{loop_idx:6d}] no tags detected")

        # FPS
        now = time.time()
        dt = max(now - last_t, 1e-6)
        last_t = now
        fps_inst = 1.0 / dt
        fps_ema = fps_inst if fps_ema == 0 else (0.9 * fps_ema + 0.1 * fps_inst)

        # HUD — show world-frame position (what compensation cares about)
        # plus camera-to-tag distance (useful while moving the bottle).
        if first_pose is not None:
            tag_id, world_t, cam_dist = first_pose
            hud = (
                f"tag={tag_id}  "
                f"world=({world_t[0]:+.3f},{world_t[1]:+.3f},{world_t[2]:+.3f})m  "
                f"cam_d={cam_dist:.3f}m  n={n_drawn}  fps={fps_ema:.1f}"
            )
        else:
            hud = f"no marker  fps={fps_ema:.1f}"
        cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        legend = f"dict={ARUCO_DICT_NAME}  size={MARKER_SIZE_M*1000:.1f}mm"
        cv2.putText(vis, legend, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, legend, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("apriltag_viz", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or _should_quit:
            break

        loop_idx += 1
finally:
    try:
        rs_pipeline.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
