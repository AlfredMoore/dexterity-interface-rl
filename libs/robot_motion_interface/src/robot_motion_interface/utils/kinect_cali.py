"""Azure Kinect color/depth extrinsic calibration with an AprilTag.

The tag pose is solved in the native color optical frame. The depth-camera
pose is then obtained from the device calibration:

    T_world_depth = T_world_color @ T_color_depth

where ``T_A_B`` maps points from frame B into frame A. Press ``s`` to save the
latest averaged pose and ``q`` to quit.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
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


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent
DEFAULT_CONFIG = RMI_ROOT / "config" / "kinect_node.yaml"
DEFAULT_OUTPUT = RMI_ROOT / "runtime" / "kinect_cali.yaml"

_ROS_TO_OPENGL = np.diag([1.0, -1.0, -1.0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Azure Kinect AprilTag extrinsic calibration")
    parser.add_argument("--marker-size", type=float, required=True, help="Tag edge length in metres")
    parser.add_argument("--tag-id", type=int, default=0, help="AprilTag ID to detect (default: 0)")
    parser.add_argument("--aruco-dict", default="DICT_APRILTAG_36h11", help="OpenCV AprilTag family (default: DICT_APRILTAG_36h11)")
    parser.add_argument("--samples", type=int, default=30, help="Number of recent poses to average")
    parser.add_argument("--world-tag-z-rot-deg", type=float, default=180.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_kinect_config(path: Path) -> dict:
    with open(path.expanduser(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["kinect"]


def _invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def _make_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(3)
    return T


def _mean_transform(samples: deque[np.ndarray]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    R_sum = np.sum([sample[:3, :3] for sample in samples], axis=0)
    u, _, vt = np.linalg.svd(R_sum)
    R = u @ vt
    if np.linalg.det(R) < 0.0:
        u[:, -1] *= -1.0
        R = u @ vt
    T[:3, :3] = R
    T[:3, 3] = np.median([sample[:3, 3] for sample in samples], axis=0)
    return T


def _quat_wxyz(R: np.ndarray) -> list[float]:
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[:] = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[:] = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[:] = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[:] = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    if q[0] < 0.0:
        q = -q
    return q.tolist()


def _recover_color_depth(calibration) -> np.ndarray:
    """Return T_color_depth with translation converted from mm to metres."""

    def convert(point: tuple[float, float, float]) -> np.ndarray:
        return np.asarray(
            calibration._convert_3d_to_3d(
                point, CalibrationType.DEPTH, CalibrationType.COLOR
            ),
            dtype=np.float64,
        )

    origin = convert((0.0, 0.0, 0.0))
    basis = np.column_stack(
        [
            (convert((1000.0, 0.0, 0.0)) - origin) / 1000.0,
            (convert((0.0, 1000.0, 0.0)) - origin) / 1000.0,
            (convert((0.0, 0.0, 1000.0)) - origin) / 1000.0,
        ]
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = basis
    T[:3, 3] = origin / 1000.0
    return T


def _world_color(T_color_tag: np.ndarray, z_rotation_deg: float) -> np.ndarray:
    angle = np.deg2rad(z_rotation_deg)
    c, s = np.cos(angle), np.sin(angle)
    T_tag_world = np.eye(4, dtype=np.float64)
    T_tag_world[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return _invert(T_tag_world) @ _invert(T_color_tag)


def _camera_block(
    name: str,
    shape: tuple[int, int],
    K: np.ndarray,
    D: np.ndarray,
    T_world_camera: np.ndarray,
) -> dict:
    height, width = shape
    R_ros = T_world_camera[:3, :3]
    position = T_world_camera[:3, 3].tolist()
    return {
        "width": int(width),
        "height": int(height),
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        f"T_world_{name}": T_world_camera.tolist(),
        f"T_{name}_world": _invert(T_world_camera).tolist(),
        "pose_ros_optical": {
            "pos": position,
            "quat_wxyz": _quat_wxyz(R_ros),
        },
        "pose_opengl": {
            "pos": position,
            "quat_wxyz": _quat_wxyz(R_ros @ _ROS_TO_OPENGL),
        },
    }


def _save(
    path: Path,
    args: argparse.Namespace,
    cfg: dict,
    T_color_tag: np.ndarray,
    T_color_depth: np.ndarray,
    color_shape: tuple[int, int],
    depth_shape: tuple[int, int],
    color_K: np.ndarray,
    color_D: np.ndarray,
    depth_K: np.ndarray,
    depth_D: np.ndarray,
    sample_count: int,
) -> None:
    T_world_color = _world_color(T_color_tag, args.world_tag_z_rot_deg)
    T_world_depth = T_world_color @ T_color_depth
    payload = {
        "_meta": {
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "device_id": int(cfg["device_id"]),
            "tag_id": int(args.tag_id),
            "marker_size_m": float(args.marker_size),
            "aruco_dict": args.aruco_dict,
            "samples": int(sample_count),
            "world_tag_z_rot_deg": float(args.world_tag_z_rot_deg),
            "transform_convention": "T_A_B maps p_B into frame A; translation is metres",
        },
        "tag": {
            "T_color_tag": T_color_tag.tolist(),
            "T_tag_color": _invert(T_color_tag).tolist(),
        },
        "device_extrinsics": {
            "T_color_depth": T_color_depth.tolist(),
            "T_depth_color": _invert(T_color_depth).tolist(),
        },
        "color": _camera_block("color", color_shape, color_K, color_D, T_world_color),
        "depth": _camera_block("depth", depth_shape, depth_K, depth_D, T_world_depth),
    }
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[saved] {path.resolve()}")


def main() -> None:
    args = _parse_args()
    cfg = _load_kinect_config(args.config)
    preset = cfg[cfg["align"]]
    camera = PyK4A(
        Config(
            color_resolution=getattr(ColorResolution, preset["color_resolution"]),
            color_format=getattr(ImageFormat, preset["color_format"]),
            depth_mode=getattr(DepthMode, preset["depth_mode"]),
            camera_fps=getattr(FPS, preset["camera_fps"]),
            synchronized_images_only=bool(preset["synchronized_images_only"]),
        ),
        device_id=int(cfg["device_id"]),
    )
    camera.start()

    calibration = camera.calibration
    color_K = np.asarray(calibration.get_camera_matrix(CalibrationType.COLOR), dtype=np.float64)
    color_D = np.asarray(
        calibration.get_distortion_coefficients(CalibrationType.COLOR), dtype=np.float64
    )
    depth_K = np.asarray(calibration.get_camera_matrix(CalibrationType.DEPTH), dtype=np.float64)
    depth_D = np.asarray(
        calibration.get_distortion_coefficients(CalibrationType.DEPTH), dtype=np.float64
    )
    T_color_depth = _recover_color_depth(calibration)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    half = args.marker_size * 0.5
    object_points = np.asarray(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )
    poses: deque[np.ndarray] = deque(maxlen=args.samples)
    tag_family = args.aruco_dict.removeprefix("DICT_")
    tag_seen = False

    print("Kinect calibration ready: s=save, q=quit")
    print(f"target tag: family={tag_family}, id={args.tag_id}, edge={args.marker_size:.4f} m")
    print(f"depth->color baseline: {np.linalg.norm(T_color_depth[:3, 3]) * 1000.0:.1f} mm")
    try:
        while True:
            capture = camera.get_capture(timeout=int(cfg["capture_timeout_ms"]))
            if capture.color is None or capture.depth is None:
                continue
            bgr = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
                ids_flat = ids.reshape(-1)
                matches = np.flatnonzero(ids_flat == args.tag_id)
                if matches.size:
                    if not tag_seen:
                        print(f"[detected] family={tag_family}, id={args.tag_id}")
                        tag_seen = True

                    image_points = corners[int(matches[0])].reshape(4, 2).astype(np.float32)
                    ok, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        color_K,
                        color_D,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )
                    if ok:
                        T_color_tag = _make_T(rvec, tvec)
                        poses.append(T_color_tag)

                        T_color_world = _invert(_world_color(T_color_tag, args.world_tag_z_rot_deg))
                        world_rvec = cv2.Rodrigues(T_color_world[:3, :3])[0]
                        world_tvec = T_color_world[:3, 3].reshape(3, 1)
                        cv2.drawFrameAxes(
                            bgr, color_K, color_D, world_rvec, world_tvec, args.marker_size * 2.0, 6
                        )
                        cv2.drawFrameAxes(
                            bgr, color_K, color_D, rvec, tvec, args.marker_size * 1.2, 2
                        )

            text = f"{tag_family} id={args.tag_id}  samples={len(poses)}/{poses.maxlen}"
            cv2.putText(bgr, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(bgr, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            legend = f"short=tag  long=world Rz({args.world_tag_z_rot_deg:+.0f} deg)"
            cv2.putText(bgr, legend, (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(bgr, legend, (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("kinect_cali", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if not poses:
                    print("[wait] target tag has not been detected")
                    continue
                _save(
                    args.output,
                    args,
                    cfg,
                    _mean_transform(poses),
                    T_color_depth,
                    bgr.shape[:2],
                    capture.depth.shape[:2],
                    color_K,
                    color_D,
                    depth_K,
                    depth_D,
                    len(poses),
                )
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
