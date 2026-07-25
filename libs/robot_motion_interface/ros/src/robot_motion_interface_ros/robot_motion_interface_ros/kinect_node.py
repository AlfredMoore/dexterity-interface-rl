"""kinect_node: Azure Kinect (pyk4a) capture, republished as RGB + depth.

Mirrors depth_node's RealSense structure -- RMI_ROOT config resolution, a daemon
capture thread, a SingleThreadedExecutor in main -- but does no inference. It
publishes color, depth, and a CameraInfo per stream, in one of two alignment
modes selected by `align` in the config:

  d2c: depth registered INTO the color frame (capture.transformed_depth). Color
       and depth share one frame/resolution; a color mask applies directly.
       depth_info carries COLOR intrinsics. Edges have FoV corner holes.

  c2d: native depth kept dense (no FoV corner holes), via one of two methods
       (config c2d.method):
         transformed_color -- Azure built-in; color resampled into the depth
           frame, so color+depth share the depth frame/intrinsics and a mask
           applies directly.
         native_reproject -- keep native full-res color (best for SAM) + native
           depth in separate frames, bridged by a static depth->color TF so a
           downstream estimator can reproject depth points into the color mask.

Images are hand-built (no cv_bridge dependency), matching depth_feat_node.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

from pyk4a import (
    PyK4A,
    Config,
    FPS,
    ColorResolution,
    DepthMode,
    ImageFormat,
    CalibrationType,
)

from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def _mat_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (x, y, z, w) quaternion (Shepperd's method)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return float(x), float(y), float(z), float(w)


_QOS_BY_NAME = {"best_effort": HIGH_PERF_QOS, "reliable": HIGH_RELIA_QOS}
_BGRA_TO = {"bgr8": cv2.COLOR_BGRA2BGR, "rgb8": cv2.COLOR_BGRA2RGB}


class KinectNode(Node):
    def __init__(self) -> None:
        super().__init__("kinect_node")

        self._declare_parameters()
        self._load_config()
        self._setup_camera()
        self._setup_publishers()
        self._setup_vis()
        if self.align == "c2d" and self.c2d_method == "native_reproject" and self.publish_extrinsic_tf:
            self._publish_static_tf()
        self._start_capture_thread()

        detail = self.align if self.align == "d2c" else f"c2d/{self.c2d_method}"
        self.get_logger().info(
            f"KinectNode ready: align={detail}, color->{self.color_topic} "
            f"({self.color_encoding}), depth->{self.depth_topic} ({self.depth_encoding}), "
            f"qos={self.qos_name}"
        )

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "kinect_node_cfg_path",
            str((RMI_ROOT / "config" / "kinect_node.yaml").resolve()),
        )
        self.cfg_path = Path(self.get_parameter("kinect_node_cfg_path").value)

    def _load_config(self) -> None:
        self.cfg = _load_yaml(self.cfg_path)["kinect"]

        self.device_id = int(self.cfg["device_id"])

        self.align = str(self.cfg["align"])
        if self.align not in ("d2c", "c2d"):
            raise ValueError(f"Unknown align {self.align!r}; expected 'd2c' or 'c2d'")

        pub = self.cfg["publish"]
        self.color_topic = str(pub["color_topic"])
        self.color_info_topic = str(pub["color_info_topic"])
        self.depth_topic = str(pub["depth_topic"])
        self.depth_info_topic = str(pub["depth_info_topic"])
        self.color_encoding = str(pub["color_encoding"])
        self.depth_encoding = str(pub["depth_encoding"])
        self.qos_name = str(pub["qos"])

        if self.color_encoding not in _BGRA_TO:
            raise ValueError(
                f"Unsupported color_encoding {self.color_encoding!r}; expected one of {list(_BGRA_TO)}"
            )
        if self.depth_encoding != "16UC1":
            raise ValueError(
                f"Unsupported depth_encoding {self.depth_encoding!r}; only '16UC1' is emitted"
            )
        if self.qos_name not in _QOS_BY_NAME:
            raise ValueError(
                f"Unknown qos {self.qos_name!r}; expected one of {list(_QOS_BY_NAME)}"
            )
        self._bgra_to = _BGRA_TO[self.color_encoding]
        self.qos = _QOS_BY_NAME[self.qos_name]

        vis = self.cfg["debug_vis"]
        self.vis_enabled = bool(vis["enabled"])
        self.vis_rate = float(vis["rate"])
        self.vis_height = int(vis["height"])
        self.vis_max_width = int(vis["max_width"])
        self.vis_clip_min = float(vis["depth_clip_min_mm"])
        self.vis_clip_max = float(vis["depth_clip_max_mm"])
        self.vis_window = str(vis["window_name"])

        # The selected align block carries both the camera modes and the frame /
        # method settings. d2c shares one frame (depth lives in the color frame);
        # c2d keeps color and depth in separate frames bridged by a static TF.
        self.c2d_method = ""
        self.publish_extrinsic_tf = False
        if self.align == "d2c":
            preset = self.cfg["d2c"]
            frame_id = str(preset["frame_id"])
            self.color_frame_id = frame_id
            self.depth_frame_id = frame_id
        else:
            preset = self.cfg["c2d"]
            self.c2d_method = str(preset["method"])
            if self.c2d_method not in ("transformed_color", "native_reproject"):
                raise ValueError(
                    f"Unknown c2d.method {self.c2d_method!r}; expected "
                    "'transformed_color' or 'native_reproject'"
                )
            self.color_frame_id = str(preset["color_frame_id"])
            self.depth_frame_id = str(preset["depth_frame_id"])
            self.publish_extrinsic_tf = bool(preset["publish_extrinsic_tf"])

        # Camera modes come from the selected preset (the best depth mode differs
        # by alignment: WFOV_2X2BINNED for d2c, NFOV_UNBINNED for c2d).
        self.color_resolution = str(preset["color_resolution"])
        self.color_format = str(preset["color_format"])
        self.depth_mode = str(preset["depth_mode"])
        self.camera_fps = str(preset["camera_fps"])
        self.synchronized_images_only = bool(preset["synchronized_images_only"])

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

        # Read intrinsics live from the device calibration (per-resolution, so
        # this must run after start). CameraInfo messages are built lazily on the
        # first frame, once the image shapes are known.
        calib = self.camera.calibration
        self._color_K = np.asarray(calib.get_camera_matrix(CalibrationType.COLOR), dtype=float)
        self._color_D = np.asarray(
            calib.get_distortion_coefficients(CalibrationType.COLOR), dtype=float
        )
        self._color_info: CameraInfo | None = None
        self._depth_info: CameraInfo | None = None

        if self.align == "c2d":
            self._depth_K = np.asarray(calib.get_camera_matrix(CalibrationType.DEPTH), dtype=float)
            self._depth_D = np.asarray(
                calib.get_distortion_coefficients(CalibrationType.DEPTH), dtype=float
            )
            # Extrinsic is only needed to bridge the two native frames.
            if self.c2d_method == "native_reproject":
                self._extrinsic_R, self._extrinsic_t = self._recover_depth_color_extrinsic(calib)
                # Sanity check: the Kinect's depth->color baseline is ~32 mm, so
                # |t| far from that means the recovery picked up the wrong transform.
                self.get_logger().info(
                    f"depth->color extrinsic: t={np.round(self._extrinsic_t, 4).tolist()} m "
                    f"(|t|={np.linalg.norm(self._extrinsic_t) * 1000:.1f} mm)"
                )

        detail = self.align if self.align == "d2c" else f"c2d/{self.c2d_method}"
        self.get_logger().info(
            f"Azure Kinect started (device_id={self.device_id}, align={detail})"
        )

    @staticmethod
    def _recover_depth_color_extrinsic(calib) -> tuple[np.ndarray, np.ndarray]:
        """depth->color extrinsic (R, t) with t in metres. This is the fixed 3D
        rigid pose between the two lenses, NOT the d2c/c2d image-alignment mode.
        convert_3d_to_3d is per-point, so probe the origin + unit basis
        (1000 mm = 1 m) to lift the full transform: X_color = R @ X_depth + t."""
        def conv(p):
            # pyk4a 1.5.0 ships no public wrapper for the 3d->3d transform (unlike
            # convert_3d_to_2d), so the underscore-prefixed method is the entry point.
            return np.asarray(
                calib._convert_3d_to_3d(p, CalibrationType.DEPTH, CalibrationType.COLOR),
                dtype=float,
            )

        o = conv((0.0, 0.0, 0.0))
        ex = conv((1000.0, 0.0, 0.0))
        ey = conv((0.0, 1000.0, 0.0))
        ez = conv((0.0, 0.0, 1000.0))
        R = np.column_stack([(ex - o) / 1000.0, (ey - o) / 1000.0, (ez - o) / 1000.0])
        t = o / 1000.0  # mm -> m
        return R, t

    def _setup_publishers(self) -> None:
        self.color_pub = self.create_publisher(Image, self.color_topic, self.qos)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, self.qos)
        self.color_info_pub = self.create_publisher(CameraInfo, self.color_info_topic, self.qos)
        self.depth_info_pub = self.create_publisher(CameraInfo, self.depth_info_topic, self.qos)
        if self.align == "c2d" and self.c2d_method == "native_reproject" and self.publish_extrinsic_tf:
            self._static_tf_broadcaster = StaticTransformBroadcaster(self)

    def _publish_static_tf(self) -> None:
        """Static TF: parent = color frame, child = depth frame, holding the pose
        of the depth camera in the color frame (the depth->color extrinsic)."""
        qx, qy, qz, qw = _mat_to_quat(self._extrinsic_R)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.color_frame_id
        tf.child_frame_id = self.depth_frame_id
        tf.transform.translation.x = float(self._extrinsic_t[0])
        tf.transform.translation.y = float(self._extrinsic_t[1])
        tf.transform.translation.z = float(self._extrinsic_t[2])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._static_tf_broadcaster.sendTransform(tf)
        self.get_logger().info(
            f"Published static TF {self.color_frame_id} -> {self.depth_frame_id}"
        )

    # ------------------------------------------------------------------
    # cv2 preview
    # ------------------------------------------------------------------
    def _setup_vis(self) -> None:
        """Preview window. The capture thread only stashes the latest frames;
        the imshow itself runs in a ROS timer so it stays on the executor's
        (main) thread, matching depth_node's capture-thread + timer split."""
        self._vis_lock = threading.Lock()
        self._vis_color: np.ndarray | None = None
        self._vis_depth: np.ndarray | None = None
        self._vis_window_sized = False
        if not self.vis_enabled:
            return
        cv2.namedWindow(self.vis_window, cv2.WINDOW_NORMAL)
        self._vis_timer = self.create_timer(1.0 / max(self.vis_rate, 1e-3), self._vis_step)

    def _stash_vis(self, color: np.ndarray, depth: np.ndarray) -> None:
        if not self.vis_enabled:
            return
        with self._vis_lock:
            # color is a fresh cvtColor output; depth is backed by the pyk4a
            # capture buffer, so it must be copied before the capture is released.
            self._vis_color = color
            self._vis_depth = depth.copy()

    def _vis_step(self) -> None:
        with self._vis_lock:
            color, depth = self._vis_color, self._vis_depth
            self._vis_color = self._vis_depth = None
        if color is None or depth is None:
            return  # no new frame since the last tick
        panel = np.hstack(
            [
                self._fit_height(self._color_to_bgr(color), self.vis_height),
                self._fit_height(self._depth_to_bgr(depth), self.vis_height),
            ]
        )
        if not self._vis_window_sized:
            # WINDOW_NORMAL keeps its (small) default size and just scales the
            # image into it, so size it to the panel once, capped to max_width.
            h, w = panel.shape[:2]
            if w > self.vis_max_width:
                h = int(round(h * self.vis_max_width / w))
                w = self.vis_max_width
            cv2.resizeWindow(self.vis_window, w, h)
            self._vis_window_sized = True
        cv2.imshow(self.vis_window, panel)
        cv2.waitKey(1)

    def _color_to_bgr(self, color: np.ndarray) -> np.ndarray:
        # imshow expects BGR; the published buffer is bgr8 or rgb8 per config.
        return color if self.color_encoding == "bgr8" else cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

    def _depth_to_bgr(self, depth: np.ndarray) -> np.ndarray:
        """Colormap depth over a FIXED millimetre range, so brightness is
        comparable across frames (rviz2's per-frame auto-normalise is what makes
        the depth view flicker). Invalid pixels (0) are forced to black."""
        invalid = depth == 0
        span = max(self.vis_clip_max - self.vis_clip_min, 1e-6)
        norm = (np.clip(depth.astype(np.float32), self.vis_clip_min, self.vis_clip_max)
                - self.vis_clip_min) / span
        # Inverted so near reads warm, matching dep/k4a-install/test_kinect.py.
        vis = cv2.applyColorMap((255 - norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[invalid] = 0
        return vis

    @staticmethod
    def _fit_height(img: np.ndarray, height: int) -> np.ndarray:
        """Scale to a common height so the two panels stack regardless of mode
        (c2d/native_reproject has 1280x720 color next to 640x576 depth)."""
        if img.shape[0] == height:
            return img
        width = int(round(img.shape[1] * height / img.shape[0]))
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def _start_capture_thread(self) -> None:
        self._capture_running = True
        self._device_failed = False
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="kinect_capture"
        )
        self._capture_thread.start()

    # ------------------------------------------------------------------
    # capture + publish
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        while self._capture_running and rclpy.ok():
            try:
                capture = self.camera.get_capture()
            except Exception as exc:
                self.get_logger().error(f"Kinect capture error: {exc}")
                self._device_failed = True
                self._capture_running = False
                rclpy.shutdown()
                return

            try:
                if self.align == "d2c":
                    self._publish_d2c(capture)
                elif self.c2d_method == "transformed_color":
                    self._publish_c2d_transformed(capture)
                else:
                    self._publish_c2d_native(capture)
            except Exception:
                # Ctrl-C invalidates the context while a publish is in flight;
                # that is a clean stop. Anything else is real and must surface.
                if rclpy.ok():
                    raise
                return

    def _publish_d2c(self, capture) -> None:
        # transformed_depth registers depth into the color frame: same
        # resolution/frame as color, depth_info carries the color intrinsics.
        depth = capture.transformed_depth
        if capture.color is None or depth is None:
            return
        stamp = self.get_clock().now().to_msg()
        color = cv2.cvtColor(capture.color, self._bgra_to)
        if self._color_info is None:
            self._color_info = self._make_camera_info(
                color.shape[1], color.shape[0], self._color_K, self._color_D, self.color_frame_id
            )
            self._depth_info = self._color_info  # depth is in the color frame

        self.color_pub.publish(self._image_msg(color, stamp, self.color_encoding, self.color_frame_id, 3))
        self.depth_pub.publish(self._image_msg(depth, stamp, self.depth_encoding, self.depth_frame_id, 2))
        self._publish_infos(stamp)
        self._stash_vis(color, depth)

    def _publish_c2d_transformed(self, capture) -> None:
        # Azure built-in C2D: color resampled INTO the depth frame. Color and
        # depth share the depth frame + depth intrinsics, so a color mask applies
        # directly. `color` here is exactly what a downstream SAM would segment.
        depth = capture.depth
        transformed_color = capture.transformed_color
        if transformed_color is None or depth is None:
            return
        stamp = self.get_clock().now().to_msg()
        color = cv2.cvtColor(transformed_color, self._bgra_to)
        if self._color_info is None:
            self._color_info = self._make_camera_info(
                color.shape[1], color.shape[0], self._depth_K, self._depth_D, self.depth_frame_id
            )
            self._depth_info = self._color_info  # same depth grid / intrinsics / frame

        self.color_pub.publish(self._image_msg(color, stamp, self.color_encoding, self.depth_frame_id, 3))
        self.depth_pub.publish(self._image_msg(depth, stamp, self.depth_encoding, self.depth_frame_id, 2))
        self._publish_infos(stamp)
        self._stash_vis(color, depth)

    def _publish_c2d_native(self, capture) -> None:
        # Native full-res color (color frame) + native depth (depth frame), in
        # separate frames bridged by the static TF. depth_info carries DEPTH
        # intrinsics; a downstream estimator reprojects depth into the color mask.
        depth = capture.depth
        if capture.color is None or depth is None:
            return
        stamp = self.get_clock().now().to_msg()
        color = cv2.cvtColor(capture.color, self._bgra_to)
        if self._color_info is None:
            self._color_info = self._make_camera_info(
                color.shape[1], color.shape[0], self._color_K, self._color_D, self.color_frame_id
            )
            self._depth_info = self._make_camera_info(
                depth.shape[1], depth.shape[0], self._depth_K, self._depth_D, self.depth_frame_id
            )

        self.color_pub.publish(self._image_msg(color, stamp, self.color_encoding, self.color_frame_id, 3))
        self.depth_pub.publish(self._image_msg(depth, stamp, self.depth_encoding, self.depth_frame_id, 2))
        self._publish_infos(stamp)
        self._stash_vis(color, depth)

    def _publish_infos(self, stamp) -> None:
        assert self._color_info is not None and self._depth_info is not None
        self._color_info.header.stamp = stamp
        self._depth_info.header.stamp = stamp
        self.color_info_pub.publish(self._color_info)
        self.depth_info_pub.publish(self._depth_info)

    @staticmethod
    def _image_msg(arr: np.ndarray, stamp, encoding: str, frame_id: str, bytes_per_px: int) -> Image:
        arr = np.ascontiguousarray(arr)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = arr.shape[0]
        msg.width = arr.shape[1]
        msg.encoding = encoding
        msg.step = arr.shape[1] * bytes_per_px
        msg.data = arr.tobytes()
        return msg

    @staticmethod
    def _make_camera_info(
        width: int, height: int, K: np.ndarray, D: np.ndarray, frame_id: str
    ) -> CameraInfo:
        """CameraInfo from a factory calibration. Azure Kinect uses an
        8-coefficient Brown-Conrady model, which ROS names 'rational_polynomial'
        with D = [k1, k2, p1, p2, k3, k4, k5, k6]."""
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        info = CameraInfo()
        info.header.frame_id = frame_id
        info.width = width
        info.height = height
        info.distortion_model = "rational_polynomial"
        info.d = D.reshape(-1).tolist()
        info.k = K.reshape(-1).tolist()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def destroy_node(self) -> bool:
        self._capture_running = False
        if getattr(self, "_capture_thread", None) is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self.vis_enabled:
            cv2.destroyAllWindows()
        camera = getattr(self, "camera", None)
        if camera is not None and not getattr(self, "_device_failed", False):
            try:
                camera.stop()
            except Exception as exc:
                self.get_logger().error(f"Error stopping Kinect: {exc}")
        elif getattr(self, "_device_failed", False):
            # The USB device is gone; stop() would block libk4a/libusb on a dead
            # handle and wedge shutdown (Ctrl-C included). Skip it -- the OS
            # reclaims the handle when the process exits.
            self.get_logger().warn("Kinect device lost; skipping camera.stop() to avoid a hung shutdown")
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KinectNode()
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
