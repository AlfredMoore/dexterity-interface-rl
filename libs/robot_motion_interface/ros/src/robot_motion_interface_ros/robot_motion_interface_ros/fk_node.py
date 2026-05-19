from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from robot_motion_interface.utils.qos import HIGH_PERF_QOS


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent
LIBS_ROOT = RMI_ROOT.parent


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


class FKNode(Node):
    """Bimanual forward-kinematics node.

    Subscribes /joint_states (real driver order) and publishes a single
    PoseArray containing all frames listed in fk_config.link_names, in that
    exact order. Per-tick FK is computed once into a {name: SE3} dict, then
    packed into the PoseArray by iterating `link_names`.
    """

    def __init__(self) -> None:
        super().__init__("fk_node")

        self._declare_parameters()
        self._load_configs()
        self._build_pinocchio()
        self._init_state()
        self._init_pub_sub()
        self._init_timer()

        self.get_logger().info(
            "FKNode ready: nq=%d, n_links=%d, fk_rate=%.1fHz, topic=%s"
            % (self.pin_model.nq, len(self.link_names), self.fk_hz, self.fk_topic)
        )

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "fk_cfg_path",
            str((LIBS_ROOT / "robot_motion_interface" / "config" / "fk_config.yaml").resolve()),
        )
        self.fk_cfg_path = Path(self.get_parameter("fk_cfg_path").value)

    def _load_configs(self) -> None:
        self.fk_cfg = _load_yaml(self.fk_cfg_path)

        self.fk_hz = float(self.fk_cfg["fk_rate"])
        if self.fk_hz <= 0.0:
            raise ValueError(f"fk_rate must be > 0, got {self.fk_hz}")

        self.world_frame_id = str(self.fk_cfg["world_frame_id"])
        self.fk_topic = str(self.fk_cfg["fk_topic"])

        urdf_path_raw = Path(str(self.fk_cfg["urdf_path"]))
        self.urdf_path = urdf_path_raw if urdf_path_raw.is_absolute() else (LIBS_ROOT / urdf_path_raw).resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        link_names = list(self.fk_cfg["link_names"])
        if len(link_names) == 0:
            raise ValueError("fk_config.link_names is empty")
        if len(set(link_names)) != len(link_names):
            raise ValueError("fk_config.link_names contains duplicates")
        self.link_names = link_names

    def _build_pinocchio(self) -> None:
        self.pin_model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.pin_data = self.pin_model.createData()

        # Resolve every link name to a frame id; fail loudly if any is missing
        # (pinocchio returns model.nframes when not found).
        self.frame_ids: list[int] = []
        for name in self.link_names:
            fid = self.pin_model.getFrameId(name)
            if fid >= self.pin_model.nframes:
                raise ValueError(f"URDF has no frame named '{name}'")
            self.frame_ids.append(fid)

    def _init_state(self) -> None:
        self.joint_lock = threading.Lock()
        self.action_num = self.pin_model.nq
        self.latest_q = np.zeros((self.action_num,), dtype=np.float64)
        self.has_joint_state = False
        self._q_snapshot = np.zeros((self.action_num,), dtype=np.float64)

    def _init_pub_sub(self) -> None:
        self.fk_pub = self.create_publisher(PoseArray, self.fk_topic, HIGH_PERF_QOS)

        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            HIGH_PERF_QOS,
        )

    def _init_timer(self) -> None:
        self.fk_timer = self.create_timer(1.0 / self.fk_hz, self._fk_step)

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState) -> None:
        # self.get_logger().info("Received joint state update")
        if len(msg.position) != self.action_num:
            # Fail loudly: URDF nq must match the driver's joint count, or FK
            # mapping is wrong everywhere downstream. log+shutdown was too
            # quiet; raise gives a clean stack trace at the boundary.
            raise RuntimeError(
                f"/joint_states DoF mismatch: msg.position has "
                f"{len(msg.position)} entries, pinocchio nq={self.action_num}. "
                f"URDF (fk_config.urdf_path) vs driver joint list drift — "
                f"check rl_bimanual_driver_config.yaml against the URDF."
            )
        pos_np = np.asarray(msg.position, dtype=np.float64)
        with self.joint_lock:
            np.copyto(self.latest_q, pos_np)
            self.has_joint_state = True

    def _fk_step(self) -> None:
        loop_start = time.perf_counter()

        with self.joint_lock:
            if not self.has_joint_state:
                self.get_logger().warn("No joint state available")
                return
            # self.get_logger().info("Joint state available")
            np.copyto(self._q_snapshot, self.latest_q)

        pin.forwardKinematics(self.pin_model, self.pin_data, self._q_snapshot)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        oMf = self.pin_data.oMf

        # frame_ids was built in link_names order, and the PoseArray output is
        # also in link_names order — so a single zip drives the whole packing.
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame_id
        for fid in self.frame_ids:
            T = oMf[fid]
            t = T.translation
            q = pin.Quaternion(T.rotation).coeffs()  # (x, y, z, w)
            p = Pose()
            p.position.x = float(t[0])
            p.position.y = float(t[1])
            p.position.z = float(t[2])
            p.orientation.x = float(q[0])
            p.orientation.y = float(q[1])
            p.orientation.z = float(q[2])
            p.orientation.w = float(q[3])
            msg.poses.append(p)
        self.fk_pub.publish(msg)

        elapsed = time.perf_counter() - loop_start
        # self.get_logger().info(
        #         f"total={elapsed:.4f}s"
        #     )
        period = 1.0 / self.fk_hz
        if elapsed > period:
            self.get_logger().warn(
                f"[SLOW_FK] total={elapsed:.4f}s, target_period={period:.4f}s"
            )


def main(args=None):
    rclpy.init(args=args)
    node = FKNode()
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
