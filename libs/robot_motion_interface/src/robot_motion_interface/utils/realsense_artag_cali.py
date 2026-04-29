"""
RealSense extrinsic calibration via ArUco AR tag.

Per loop:
  - detect ArUco/AprilTag marker(s) in the color stream (auto-scans dicts on first hit)
  - estimate T_cam_tag (tag pose in camera frame) via solvePnP / IPPE_SQUARE
  - redefine the world frame as the tag frame rotated about its own +Z by
    WORLD_TAG_Z_ROT_DEG (default 180), then compute camera pose in this world
  - emit camera pose as quaternions in three IsaacSim camera conventions
    (ros, opengl, world) for direct copy-paste into IsaacSim camera configs
  - print throttled pose info and overlay tag axes on the live image

Run:
  python -m robot_motion_interface.utils.realsense_artag_cali

Keys (click the OpenCV window first to give it keyboard focus):
  q      quit (with autosave if a pose was captured)
  s      manual save now
  Ctrl-C quit cleanly with autosave (works without window focus)

Saved YAML mirrors the IsaacSim `realsense:` block layout, populated with values
read live from the device profile (color/depth intrinsics, distortion model,
depth_scale, depth→color extrinsics) plus the computed `pose` and per-convention
quaternions.
"""

import argparse
import importlib.util
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

# ── Tunable constants / defaults ────────────────────────────────────────────
DEFAULT_ARUCO_DICT_NAME = "DICT_4X4_50"
DEFAULT_AUTO_DETECT_DICT = False
AXIS_THICKNESS = 3                        # line thickness for drawn axes
DEFAULT_PRINT_EVERY_N = 30                # print every N loops to throttle stdout
DEFAULT_WORLD_TAG_Z_ROT_DEG = 180.0

# Default depth clip range (m) and output render size copied from existing IsaacSim configs.
# These are user-defined and not present in the RealSense profile, so we keep sane defaults.
DEFAULT_DEPTH_CLIP   = [0.2, 2.0]
DEFAULT_OUTPUT_SCALE = 0.5  # output_w/h = scale * native_w/h

# IsaacSim/IsaacLab camera-frame conventions used downstream.
#   ros:    +X right, +Y down, +Z forward (== OpenCV optical / what solvePnP returns)
#   opengl: +X right, +Y up,   +Z backward (USD/OpenGL camera)
#   world:  +X forward, +Y left, +Z up    (robot/world style; "look-along-X")
# Rotation matrices applied on the right of R_world_cam_ros to convert the camera's
# *local* frame from ros into the target convention. See _convert_R_to_convention.
_M_ROS    = np.eye(3, dtype=np.float64)
_M_OPENGL = np.diag([1.0, -1.0, -1.0])
_M_WORLD  = np.array([[0.0, -1.0,  0.0],
                      [0.0,  0.0, -1.0],
                      [1.0,  0.0,  0.0]], dtype=np.float64)

# OpenCV's solvePnP places the camera in the "ros" optical convention.
SOURCE_CONVENTION = "ros"

# All ArUco/AprilTag dicts shipped with OpenCV (skipped silently if absent in the build)
_DICT_CANDIDATES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RealSense extrinsic calibration with ArUco. "
            "Use explicit marker size/dictionary/target tag to avoid pose jumps."
        )
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        required=True,
        help="Physical ArUco side length in meters (required, e.g. 0.1).",
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default=DEFAULT_ARUCO_DICT_NAME,
        choices=_DICT_CANDIDATES,
        help=f"Aruco dictionary to use when auto-detect is off (default: {DEFAULT_ARUCO_DICT_NAME}).",
    )
    parser.add_argument(
        "--target-tag-id",
        type=int,
        required=True,
        help="Target marker ID to track/save. Required to avoid 'first detected ID' drift.",
    )
    parser.add_argument(
        "--auto-detect-dict",
        action="store_true",
        default=DEFAULT_AUTO_DETECT_DICT,
        help="Scan all dicts and lock first hit. Disabled by default due to mis-detect risk.",
    )
    parser.add_argument(
        "--print-every-n",
        type=int,
        default=DEFAULT_PRINT_EVERY_N,
        help=f"Print throttle interval in frames (default: {DEFAULT_PRINT_EVERY_N}).",
    )
    parser.add_argument(
        "--world-tag-z-rot-deg",
        type=float,
        default=DEFAULT_WORLD_TAG_Z_ROT_DEG,
        help=f"World frame = tag frame rotated about +Z by this angle (default: {DEFAULT_WORLD_TAG_Z_ROT_DEG}).",
    )
    return parser.parse_args()


ARGS = _parse_args()

if ARGS.marker_size <= 0.0:
    raise ValueError(f"--marker-size must be > 0, got {ARGS.marker_size}")

ARUCO_DICT_NAME = ARGS.aruco_dict
ARUCO_DICT = getattr(cv2.aruco, ARUCO_DICT_NAME, None)
if ARUCO_DICT is None:
    raise ValueError(f"Aruco dictionary not available in this OpenCV build: {ARUCO_DICT_NAME}")

AUTO_DETECT_DICT = bool(ARGS.auto_detect_dict)
if AUTO_DETECT_DICT:
    print("[WARN] --auto-detect-dict enabled. This may lock a wrong dictionary in cluttered scenes.")

MARKER_SIZE_M = float(ARGS.marker_size)
TARGET_TAG_ID = int(ARGS.target_tag_id)
AXIS_LENGTH_M = MARKER_SIZE_M * 1.5
PRINT_EVERY_N = max(1, int(ARGS.print_every_n))

# World frame is the AR-tag frame rotated about its own +Z by this angle, then taken as origin.
WORLD_TAG_Z_ROT_DEG = float(ARGS.world_tag_z_rot_deg)

# ── Config + paths (mirrors realsense_test.py) ──────────────────────────────
spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent
DEFAULT_CONFIG_PATH = RMI_ROOT / "config" / "rl_policy_node_config.yaml"

config_path = str(DEFAULT_CONFIG_PATH.resolve())
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at: {config_path}")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

_rs_config = config["realsense"]
_rs_fps = _rs_config["rs_fps"]
_sensor_settings = _rs_config["sensor_settings"]
_c_intrinsics = _rs_config["color_intrinsics"]
_d_intrinsics = _rs_config["depth_intrinsics"]

# ── RealSense init (mirrors realsense_test.py) ──────────────────────────────
rs_pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(
    rs.stream.color,
    _c_intrinsics["width"],
    _c_intrinsics["height"],
    rs.format.bgr8,
    _rs_fps,
)
rs_config.enable_stream(
    rs.stream.depth,
    _d_intrinsics["width"],
    _d_intrinsics["height"],
    rs.format.z16,
    _rs_fps,
)
rs_profile = rs_pipeline.start(rs_config)
rs_align = rs.align(rs.stream.color)


def reset_camera():
    ctx = rs.context()
    for dev in ctx.query_devices():
        print(f"restarting: {dev.get_info(rs.camera_info.name)}")
        dev.hardware_reset()


def _apply_sensor_settings(profile: rs.pipeline_profile) -> None:
    if not _sensor_settings:
        return
    try:
        device = profile.get_device()
        sensors = device.query_sensors()
    except Exception:
        return

    auto_exposure = _sensor_settings.get("auto_exposure", False)
    exposure = _sensor_settings.get("exposure", 350)
    gain = _sensor_settings.get("gain", 16)

    for sensor in sensors:
        if auto_exposure is not None and sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
        if auto_exposure is False:
            if exposure is not None and sensor.supports(rs.option.exposure):
                sensor.set_option(rs.option.exposure, float(exposure))
            if gain is not None and sensor.supports(rs.option.gain):
                sensor.set_option(rs.option.gain, float(gain))


_apply_sensor_settings(rs_profile)
print("Sensor settings applied")

rs_device = rs_profile.get_device()
print(
    "RealSense initialized:\n"
    f"  device={rs_device.get_info(rs.camera_info.name)}\n"
    f"  serial={rs_device.get_info(rs.camera_info.serial_number)}\n"
    f"  color={_c_intrinsics['width']}x{_c_intrinsics['height']}@{_rs_fps}fps\n"
)

# ── Read intrinsics + extrinsics directly from RealSense profiles ───────────
def _rs_intrinsics_dict(intr) -> dict:
    """Convert pyrealsense2 intrinsics into an IsaacSim-compatible dict."""
    model_name = str(intr.model).split(".")[-1]   # e.g. "brown_conrady"
    return {
        "width":      int(intr.width),
        "height":     int(intr.height),
        "fx":         float(intr.fx),
        "fy":         float(intr.fy),
        "cx":         float(intr.ppx),
        "cy":         float(intr.ppy),
        "distortion": [float(c) for c in intr.coeffs],
        "model":      f"distortion.{model_name}",
    }


def _rs_extrinsics_to_T(ext) -> list[list[float]]:
    """rs.extrinsics has rotation (col-major 9 floats) and translation (3). Build 4x4 row-major list."""
    R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3, order="F")  # column-major
    t = np.array(ext.translation, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T.tolist()


# pyrealsense2 stream profiles for both color and depth (post-pipeline-start so device-active values).
_color_profile = rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
_depth_profile = rs_profile.get_stream(rs.stream.depth).as_video_stream_profile()
_rs_color_intr = _rs_intrinsics_dict(_color_profile.get_intrinsics())
_rs_depth_intr = _rs_intrinsics_dict(_depth_profile.get_intrinsics())
# T mapping points from depth optical frame → color optical frame (matches IsaacSim layout).
_T_color_depth = _rs_extrinsics_to_T(_depth_profile.get_extrinsics_to(_color_profile))
try:
    _depth_scale = float(rs_profile.get_device().first_depth_sensor().get_depth_scale())
except Exception:
    _depth_scale = 0.001

K = np.array(
    [
        [_rs_color_intr["fx"], 0.0,                  _rs_color_intr["cx"]],
        [0.0,                  _rs_color_intr["fy"], _rs_color_intr["cy"]],
        [0.0,                  0.0,                  1.0],
    ],
    dtype=np.float64,
)
dist = np.array(_rs_color_intr["distortion"], dtype=np.float64).reshape(-1)

print("Camera matrix K (live from RealSense) =")
print(K)
print(f"color distortion model = {_rs_color_intr['model']}, coeffs = {_rs_color_intr['distortion']}")
print(f"depth distortion model = {_rs_depth_intr['model']}, depth_scale = {_depth_scale}")


def _diff_intrinsics(yaml_d: dict, device_d: dict, label: str) -> None:
    """Print yaml-vs-device intrinsics side by side; flag any mismatch with a marker."""
    keys = ["width", "height", "fx", "fy", "cx", "cy", "distortion", "model"]
    print(f"\n[compare] {label}: yaml (config) vs device (live)")
    print(f"  {'key':<10} | {'yaml':<28} | {'device':<28} | match")
    print(f"  {'-'*10}-+-{'-'*28}-+-{'-'*28}-+------")
    for k in keys:
        yv = yaml_d.get(k)
        dv = device_d.get(k)
        # Float-tolerant compare for scalar numeric keys
        if isinstance(yv, (int, float)) and isinstance(dv, (int, float)):
            same = abs(float(yv) - float(dv)) < 1e-3
        elif isinstance(yv, list) and isinstance(dv, list):
            if len(yv) != len(dv):
                same = False
            else:
                same = all(abs(float(a) - float(b)) < 1e-3 for a, b in zip(yv, dv))
        else:
            same = (yv == dv)
        flag = "OK" if same else "DIFF"
        print(f"  {k:<10} | {str(yv):<28} | {str(dv):<28} | {flag}")


_diff_intrinsics(_c_intrinsics, _rs_color_intr, "color_intrinsics")
_diff_intrinsics(_d_intrinsics, _rs_depth_intr, "depth_intrinsics")
print()

# ── ArUco detector (OpenCV >= 4.7 API) ──────────────────────────────────────
detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector_params.cornerRefinementWinSize = 5
detector_params.cornerRefinementMaxIterations = 30
detector_params.cornerRefinementMinAccuracy = 0.01
_USE_NEW_API = hasattr(cv2.aruco, "ArucoDetector")


def _make_detector(dict_id: int):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    if _USE_NEW_API:
        return d, cv2.aruco.ArucoDetector(d, detector_params)
    return d, None


def _detector_dict_pairs() -> list[tuple[str, int, object, object]]:
    """Return list of (name, id, aruco_dict, detector) for each available dict.

    In auto mode this is all known dicts; otherwise just the configured one.
    """
    out = []
    if AUTO_DETECT_DICT:
        for name in _DICT_CANDIDATES:
            did = getattr(cv2.aruco, name, None)
            if did is None:
                continue
            d, det = _make_detector(int(did))
            out.append((name, int(did), d, det))
    else:
        d, det = _make_detector(int(ARUCO_DICT))
        out.append((f"id={ARUCO_DICT}", int(ARUCO_DICT), d, det))
    return out


_DICT_PAIRS = _detector_dict_pairs()
_LOCKED_PAIR_IDX: int | None = None  # set after first successful detection in auto mode

# Active detector kept for legacy path (unused once locked in auto mode)
aruco_dict, detector = _DICT_PAIRS[0][2], _DICT_PAIRS[0][3]

print(
    f"ArUco detector ready (new_api={_USE_NEW_API}, "
    f"auto_detect_dict={AUTO_DETECT_DICT}, candidates={len(_DICT_PAIRS)}, "
    f"marker={MARKER_SIZE_M} m)"
)
print(
    f"OpenCV={cv2.__version__}  "
    f"drawFrameAxes={'yes' if hasattr(cv2, 'drawFrameAxes') else 'no'}  "
    f"aruco.drawAxis={'yes' if hasattr(cv2.aruco, 'drawAxis') else 'no'}"
)

# Marker-frame object points: marker centered at origin, +Z out of page,
# +X right, +Y up. Order matches detectMarkers corner order:
# top-left, top-right, bottom-right, bottom-left.
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


def _detect_with_pair(gray: np.ndarray, pair) -> tuple[object, object]:
    _, _, d, det = pair
    if _USE_NEW_API and det is not None:
        corners, ids, _ = det.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=detector_params)
    return corners, ids


def _detect_markers(gray: np.ndarray):
    """Detect using locked dict if available; otherwise scan all candidates and lock the first hit."""
    global _LOCKED_PAIR_IDX
    if _LOCKED_PAIR_IDX is not None:
        return _detect_with_pair(gray, _DICT_PAIRS[_LOCKED_PAIR_IDX])

    for idx, pair in enumerate(_DICT_PAIRS):
        corners, ids = _detect_with_pair(gray, pair)
        if ids is not None and len(ids) > 0:
            _LOCKED_PAIR_IDX = idx
            print(f"[auto-detect] locked dict={pair[0]} (id={pair[1]}); ids found={ids.flatten().tolist()}")
            return corners, ids
    return (), None


def _pose_from_corners(img_pts: np.ndarray):
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(_OBJ_PTS, img_pts, K, dist, flags=flag)
    return ok, rvec, tvec


def _draw_axes(img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, length: float, thickness: int) -> bool:
    """Draw XYZ axes (RGB) on `img`. Tries cv2.drawFrameAxes, falls back to manual projection.

    Returns True if drawn, False otherwise (used for debug print).
    """
    # Try modern OpenCV API
    if hasattr(cv2, "drawFrameAxes"):
        try:
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, float(length), int(thickness))
            return True
        except cv2.error:
            pass
    # Try legacy aruco API
    if hasattr(cv2.aruco, "drawAxis"):
        try:
            cv2.aruco.drawAxis(img, K, dist, rvec, tvec, float(length))
            return True
        except cv2.error:
            pass
    # Manual fallback: project 4 endpoints and draw colored lines
    axis_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],
            [0.0, length, 0.0],
            [0.0, 0.0, length],
        ],
        dtype=np.float32,
    )
    img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1, 2).astype(int)
    o, x, y, z = img_pts
    cv2.line(img, tuple(o), tuple(x), (0, 0, 255), thickness)   # X = red (BGR)
    cv2.line(img, tuple(o), tuple(y), (0, 255, 0), thickness)   # Y = green
    cv2.line(img, tuple(o), tuple(z), (255, 0, 0), thickness)   # Z = blue
    return True


def _draw_labeled_axes(
    img: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    length: float,
    thickness: int,
    label_prefix: str,
) -> None:
    """Project + draw XYZ axes with text labels at the tips.

    Uses the manual projection path (not drawFrameAxes) so we keep full control
    over color and can attach per-axis labels. Color scheme is RGB-for-XYZ.
    """
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

    cv2.line(img, o, x, (0, 0, 255), thickness)   # X = red (BGR)
    cv2.line(img, o, y, (0, 255, 0), thickness)   # Y = green
    cv2.line(img, o, z, (255, 0, 0), thickness)   # Z = blue

    # Tip labels with black outline + colored fill for legibility.
    def _put(text: str, pos: tuple, fill: tuple) -> None:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, fill,      1, cv2.LINE_AA)

    _put(f"{label_prefix}x", x, (0, 0, 255))
    _put(f"{label_prefix}y", y, (0, 255, 0))
    _put(f"{label_prefix}z", z, (255, 0, 0))


def _world_axes_in_cam(T_cam_tag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (rvec, tvec) of the *world* frame as seen by the camera.

    World := tag rotated about its own +Z by WORLD_TAG_Z_ROT_DEG; same origin.
    Returned rvec/tvec are suitable for cv2.projectPoints / drawFrameAxes.
    """
    R_tag_world = _rot_z(WORLD_TAG_Z_ROT_DEG)
    T_tag_world = np.eye(4, dtype=np.float64)
    T_tag_world[:3, :3] = R_tag_world
    T_cam_world = T_cam_tag @ T_tag_world          # pose of world in camera frame
    rvec_w, _ = cv2.Rodrigues(T_cam_world[:3, :3])
    tvec_w = T_cam_world[:3, 3].reshape(3, 1)
    return rvec_w, tvec_w


def _make_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _euler_deg_from_R(R: np.ndarray) -> np.ndarray:
    """ZYX intrinsic Euler angles in degrees (yaw-pitch-roll), gimbal-safe-ish."""
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


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _world_redefine_T(T_cam_tag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Redefine world frame as the AR-tag frame rotated about its own +Z by WORLD_TAG_Z_ROT_DEG.

    Returns (T_world_cam, T_cam_world) with the camera still in OpenCV/ROS optical convention.
    """
    R_tag_world = _rot_z(WORLD_TAG_Z_ROT_DEG)             # world expressed in tag frame (rotation only)
    T_tag_world = np.eye(4, dtype=np.float64)
    T_tag_world[:3, :3] = R_tag_world                      # translation 0: world origin == tag origin
    T_world_tag = _invert_T(T_tag_world)                   # = T_tag_world for 180°, but kept general
    T_tag_cam = _invert_T(T_cam_tag)
    T_world_cam = T_world_tag @ T_tag_cam
    T_cam_world = _invert_T(T_world_cam)
    return T_world_cam, T_cam_world


def _convert_R_to_convention(R_world_cam_ros: np.ndarray, convention: str) -> np.ndarray:
    """Re-express R_world_cam by changing the camera's *local* axis convention.

    The world frame stays the same; only the camera's body-frame axes change.
    """
    if convention == "ros":
        M = _M_ROS
    elif convention == "opengl":
        M = _M_OPENGL
    elif convention == "world":
        M = _M_WORLD
    else:
        raise ValueError(f"Unknown convention: {convention}")
    return R_world_cam_ros @ M


def _rot_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """Robust rotation → quaternion (w, x, y, z), matching IsaacLab convention."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0.0:
        q = -q       # normalize hemisphere (w >= 0) for stable output
    return [float(v) for v in q]


def _camera_pose_in_world(T_cam_tag: np.ndarray) -> dict:
    """Build the full camera pose payload (pos + quats in all 3 conventions)."""
    T_world_cam, T_cam_world = _world_redefine_T(T_cam_tag)
    R_ros = T_world_cam[:3, :3]
    pos   = T_world_cam[:3, 3]
    return {
        "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
        "quat_wxyz_by_convention": {
            "ros":    _rot_to_quat_wxyz(_convert_R_to_convention(R_ros, "ros")),
            "opengl": _rot_to_quat_wxyz(_convert_R_to_convention(R_ros, "opengl")),
            "world":  _rot_to_quat_wxyz(_convert_R_to_convention(R_ros, "world")),
        },
        "T_world_cam": T_world_cam.tolist(),
        "T_cam_world": T_cam_world.tolist(),
    }


def _save_extrinsics(T_cam_tag: np.ndarray, T_tag_cam: np.ndarray, tag_id: int,
                      reason: str = "manual") -> str:
    """Dump the calibration to YAML mirroring the IsaacSim realsense block layout.

    Includes everything readable from the live RealSense profile (intrinsics,
    extrinsics, depth scale, distortion model) plus the computed camera pose
    in our redefined world frame, in all three IsaacSim camera conventions.
    """
    save_dir = RMI_ROOT / "runtime"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"rs_config_{ts}.yaml"

    cam_pose = _camera_pose_in_world(T_cam_tag)
    quats = cam_pose["quat_wxyz_by_convention"]

    # Pick a "primary" rot to fill the standard pose.rot field. We default to "world"
    # because the existing HandEnv.yaml realsense pose uses convention=world.
    primary_conv = "world"

    pos_xyz = [float(v) for v in cam_pose["pos"]]
    rot_ros = [float(v) for v in quats["ros"]]
    rot_opengl = [float(v) for v in quats["opengl"]]
    rot_world = [float(v) for v in quats["world"]]
    rot_primary = {
        "ros": rot_ros,
        "opengl": rot_opengl,
        "world": rot_world,
    }[primary_conv]

    realsense_block = {
        "rs_fps": int(_rs_fps),
        "width":  int(_rs_color_intr["width"]),
        "height": int(_rs_color_intr["height"]),
        "output_width":  int(round(_rs_color_intr["width"]  * DEFAULT_OUTPUT_SCALE)),
        "output_height": int(round(_rs_color_intr["height"] * DEFAULT_OUTPUT_SCALE)),
        "clip": list(DEFAULT_DEPTH_CLIP),
        "depth_scale": float(_depth_scale),
        "pose": {
            "pos": list(pos_xyz),
            "rot": list(rot_primary),            # quaternion (w, x, y, z)
            "convention": primary_conv,
        },
        "pose_all_conventions": {
            "source_convention": SOURCE_CONVENTION,  # OpenCV solvePnP natural output
            "ros": {
                "pos": list(pos_xyz),
                "rot": list(rot_ros),
                "convention": "ros",
            },
            "opengl": {
                "pos": list(pos_xyz),
                "rot": list(rot_opengl),
                "convention": "opengl",
            },
            "world": {
                "pos": list(pos_xyz),
                "rot": list(rot_world),
                "convention": "world",
            },
        },
        "color_intrinsics": _rs_color_intr,
        "depth_intrinsics": {
            **_rs_depth_intr,
            "T_color_depth": _T_color_depth,
        },
    }

    payload = {
        "_meta": {
            "saved_at":        ts,
            "save_reason":     reason,
            "tag_id":          int(tag_id),
            "marker_size_m":   float(MARKER_SIZE_M),
            "aruco_dict_id":   int(_DICT_PAIRS[_LOCKED_PAIR_IDX][1]) if _LOCKED_PAIR_IDX is not None else int(ARUCO_DICT),
            "aruco_dict_name": _DICT_PAIRS[_LOCKED_PAIR_IDX][0]      if _LOCKED_PAIR_IDX is not None else "unknown",
            "world_tag_z_rot_deg": float(WORLD_TAG_Z_ROT_DEG),
            "device":          rs_device.get_info(rs.camera_info.name),
            "serial":          rs_device.get_info(rs.camera_info.serial_number),
        },
        "T_cam_tag": T_cam_tag.tolist(),
        "T_tag_cam": T_tag_cam.tolist(),
        "T_world_cam": cam_pose["T_world_cam"],
        "T_cam_world": cam_pose["T_cam_world"],
        "realsense":  realsense_block,
    }

    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return str(out_path)


# ── Main loop ───────────────────────────────────────────────────────────────
loop_idx = 0
last_t = time.time()
fps_ema = 0.0
last_T_cam_tag: np.ndarray | None = None
last_T_tag_cam: np.ndarray | None = None
last_tag_id: int | None = None

_should_quit = False


def _on_sigint(signum, _frame):
    global _should_quit
    _should_quit = True
    print("\n[signal] SIGINT received; will save (if pose available) and exit cleanly.")


signal.signal(signal.SIGINT, _on_sigint)

# Force window creation early so it can take focus reliably.
cv2.namedWindow("artag_cali", cv2.WINDOW_AUTOSIZE)

print(
    "Starting loop. Press 'q' to quit, 's' to save extrinsics. "
    "Click on the OpenCV window first to give it keyboard focus. "
    "Ctrl-C also works and will autosave on exit."
)

try:
    while True:
        try:
            frames = rs_pipeline.wait_for_frames(timeout_ms=2000)
            if not frames:
                time.sleep(0.001)
                continue
        except RuntimeError:
            print("[warn] wait_for_frames timed out, resetting camera and retrying...")
            reset_camera()
            time.sleep(1)
            continue

        aligned_frames = rs_align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        corners, ids = _detect_markers(gray)
        vis = color.copy()

        active_idx = -1
        active_T_cam_tag = None
        active_rvec = None
        active_tvec = None
        active_id = -1

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            id_list = ids.flatten().tolist()
            if TARGET_TAG_ID not in id_list:
                if loop_idx % PRINT_EVERY_N == 0:
                    print(
                        f"[{loop_idx:6d}] target_tag_id={TARGET_TAG_ID} not found in detected ids={id_list}"
                    )
            else:
                active_idx = id_list.index(TARGET_TAG_ID)
                active_id = int(id_list[active_idx])

                ok, rvec, tvec = _pose_from_corners(corners[active_idx])
                if ok:
                    active_rvec = rvec
                    active_tvec = tvec
                    active_T_cam_tag = _make_T(rvec, tvec)
                    T_tag_cam = _invert_T(active_T_cam_tag)
                    last_T_cam_tag = active_T_cam_tag
                    last_T_tag_cam = T_tag_cam
                    last_tag_id = active_id

                    # Tag frame: short, labeled "T" (T_x / T_y / T_z)
                    _draw_labeled_axes(
                        vis, rvec, tvec, AXIS_LENGTH_M, AXIS_THICKNESS, label_prefix="T_"
                    )
                    # World frame (tag rotated +Z by WORLD_TAG_Z_ROT_DEG): longer arms,
                    # labeled "W" so it visually overlays at the same origin but reaches further.
                    rvec_w, tvec_w = _world_axes_in_cam(active_T_cam_tag)
                    _draw_labeled_axes(
                        vis, rvec_w, tvec_w, AXIS_LENGTH_M * 2.0, AXIS_THICKNESS + 1, label_prefix="W_"
                    )

                    if loop_idx % PRINT_EVERY_N == 0:
                        t_ct = active_T_cam_tag[:3, 3]
                        eul_ct = _euler_deg_from_R(active_T_cam_tag[:3, :3])
                        t_tc = T_tag_cam[:3, 3]
                        eul_tc = _euler_deg_from_R(T_tag_cam[:3, :3])
                        print(
                            f"[{loop_idx:6d}] tag={active_id} | "
                            f"t_cam_tag=({t_ct[0]:+.3f},{t_ct[1]:+.3f},{t_ct[2]:+.3f}) m  "
                            f"rpy_deg=({eul_ct[0]:+.1f},{eul_ct[1]:+.1f},{eul_ct[2]:+.1f}) | "
                            f"t_tag_cam=({t_tc[0]:+.3f},{t_tc[1]:+.3f},{t_tc[2]:+.3f}) m  "
                            f"rpy_deg=({eul_tc[0]:+.1f},{eul_tc[1]:+.1f},{eul_tc[2]:+.1f})"
                        )

        # FPS
        now = time.time()
        dt = max(now - last_t, 1e-6)
        last_t = now
        fps_inst = 1.0 / dt
        fps_ema = fps_inst if fps_ema == 0 else (0.9 * fps_ema + 0.1 * fps_inst)

        # HUD
        if active_T_cam_tag is not None:
            d = float(np.linalg.norm(active_T_cam_tag[:3, 3]))
            hud = f"tag={active_id}  dist={d:.3f}m  fps={fps_ema:.1f}"
        else:
            hud = f"no marker detected  fps={fps_ema:.1f}"
        cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Frame legend (always shown so the user can read T_ vs W_ axis labels)
        legend = f"axes: T_=tag (short)  W_=world=tag@Rz({WORLD_TAG_Z_ROT_DEG:+.0f} deg) (long)"
        cv2.putText(vis, legend, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, legend, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("artag_cali", vis)
        # Keep waitKey short so the loop stays at 60 fps with the RealSense stream.
        # If 'q'/'s' don't catch reliably (window-focus issues), Ctrl-C still works
        # via the SIGINT handler and triggers the same autosave path.
        key = cv2.waitKey(1) & 0xFF
        if key not in (0, 255):
            print(f"[debug] key={key} ({chr(key) if 32 <= key < 127 else '?'})")
        if key == ord("q") or _should_quit:
            break
        if key == ord("s"):
            if last_T_cam_tag is not None and last_T_tag_cam is not None and last_tag_id is not None:
                out_path = _save_extrinsics(
                    last_T_cam_tag, last_T_tag_cam, last_tag_id, reason="manual_keypress"
                )
                print(f"[saved] extrinsics -> {out_path}")
            else:
                print("[warn] no pose available yet; cannot save")

        loop_idx += 1
finally:
    # Autosave on any exit path (q, Ctrl-C, exception) when we have a pose.
    if last_T_cam_tag is not None and last_T_tag_cam is not None and last_tag_id is not None:
        try:
            out_path = _save_extrinsics(
                last_T_cam_tag, last_T_tag_cam, last_tag_id, reason="exit_autosave"
            )
            print(f"[autosaved] extrinsics -> {out_path}")
        except Exception as exc:
            print(f"[warn] autosave failed: {exc}")
    else:
        print("[exit] no pose was ever captured; nothing saved.")

    try:
        rs_pipeline.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
