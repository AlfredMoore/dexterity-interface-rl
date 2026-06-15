"""Publish a precomputed .npy trajectory to /target_joint_states in sequence."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import yaml

from robot_motion_interface.utils.qos import HIGH_RELIA_QOS

T_JS_QOS = HIGH_RELIA_QOS


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent


class RunTrajNode(Node):
    def __init__(self):
        super().__init__("run_traj_node")

        self.declare_parameter("driver_cfg_path", str(RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml"))
        self.declare_parameter("traj_path", str(RMI_ROOT / "runtime" / "traj_pregrasp_full_200.npy"))
        self.declare_parameter("publish_hz", 30.0)

        driver_cfg_path = self.get_parameter("driver_cfg_path").get_parameter_value().string_value
        traj_path = self.get_parameter("traj_path").get_parameter_value().string_value
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        with open(driver_cfg_path, "r", encoding="utf-8") as f:
            driver_cfg = yaml.safe_load(f)

        l_names = ["left_" + n for n in driver_cfg["left_panda_joint_names"] + driver_cfg["left_tesollo_joint_names"]]
        r_names = ["right_" + n for n in driver_cfg["right_panda_joint_names"] + driver_cfg["right_tesollo_joint_names"]]
        self.joint_names = l_names + r_names
        self.dof = len(self.joint_names)

        self.traj = np.load(traj_path).astype(np.float32)
        if self.traj.ndim != 2:
            raise ValueError(f"Trajectory must be 2D, got shape={self.traj.shape}")
        if self.traj.shape[1] != self.dof:
            raise ValueError(f"Trajectory DoF mismatch: traj={self.traj.shape[1]}, expected={self.dof}")

        self.target_pub = self.create_publisher(JointState, "/target_joint_states", T_JS_QOS)
        self.get_logger().info(f"Loaded traj: {traj_path}, shape={self.traj.shape}, publish_hz={self.publish_hz}")

    def run_once(self) -> None:
        # Wait for the driver to subscribe before streaming. The QoS is RELIABLE +
        # VOLATILE, so any setpoint published before DDS discovery matches the
        # subscriber (~2s across machines) is dropped and never resent. Without this
        # wait the arm receives nothing until matching completes, then jumps to a
        # setpoint well into the trajectory.
        while rclpy.ok() and self.target_pub.get_subscription_count() == 0:
            self.get_logger().info("Waiting for a subscriber on /target_joint_states ...")
            time.sleep(0.1)
        time.sleep(0.5)  # margin for the link to fully establish before streaming
        self.get_logger().info(
            f"Subscriber connected ({self.target_pub.get_subscription_count()}). Streaming trajectory."
        )

        msg = JointState()
        msg.name = self.joint_names
        dt = 1.0 / self.publish_hz
        deadline = time.perf_counter()
        for i, q in enumerate(self.traj):
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = q.tolist()
            self.target_pub.publish(msg)

            if i % 20 == 0 or i == self.traj.shape[0] - 1:
                self.get_logger().info(f"publish step {i + 1}/{self.traj.shape[0]}")

            deadline += dt
            sleep_s = deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)


def main(args=None):
    rclpy.init(args=args)
    node = RunTrajNode()
    try:
        node.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
