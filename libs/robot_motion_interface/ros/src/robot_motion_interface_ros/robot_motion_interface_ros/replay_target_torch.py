"""Publish precomputed torch trajectory targets to /target_joint_states in sequence."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import torch
import yaml

from robot_motion_interface.utils.qos import HIGH_RELIA_QOS
from robot_motion_interface.utils.sim2real import joint_mapping

T_JS_QOS = HIGH_RELIA_QOS


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent


class RunTrajNode(Node):
    def __init__(self):
        super().__init__("run_traj_node")

        self.declare_parameter("driver_cfg_path", str(RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml"))
        self.declare_parameter("runtime_cfg_path", str(RMI_ROOT / "runtime" / "runtime_cfg.yaml"))
        self.declare_parameter("traj_path", str(RMI_ROOT / "runtime" / "actions_dones_targets.pt"))
        self.declare_parameter("traj_id", 0)
        self.declare_parameter("publish_hz", 30.0)

        driver_cfg_path = self.get_parameter("driver_cfg_path").get_parameter_value().string_value
        runtime_cfg_path = self.get_parameter("runtime_cfg_path").get_parameter_value().string_value
        traj_path = self.get_parameter("traj_path").get_parameter_value().string_value
        self.traj_id = int(self.get_parameter("traj_id").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        with open(driver_cfg_path, "r", encoding="utf-8") as f:
            driver_cfg = yaml.safe_load(f)
        with open(runtime_cfg_path, "r", encoding="utf-8") as f:
            runtime_cfg = yaml.safe_load(f)

        l_names = ["left_" + n for n in driver_cfg["left_panda_joint_names"] + driver_cfg["left_tesollo_joint_names"]]
        r_names = ["right_" + n for n in driver_cfg["right_panda_joint_names"] + driver_cfg["right_tesollo_joint_names"]]
        self.joint_names = l_names + r_names
        self.dof = len(self.joint_names)

        # sim2real: targets are saved in policy joint order, reorder to real driver order
        policy_joint_names = runtime_cfg["bimanual_joint_names"]
        policy2real_idx, _ = joint_mapping(policy_joint_names, self.joint_names)
        policy2real_idx = torch.tensor(policy2real_idx, dtype=torch.long)

        data = torch.load(traj_path, map_location="cpu")
        left = data["left_dof_targets"]      # [T, N, left_dof]
        right = data["right_dof_targets"]    # [T, N, right_dof]
        dones = data["dones"]                # [T, N]
        if left.ndim != 3 or right.ndim != 3 or dones.ndim != 2:
            raise ValueError("Expected left/right targets as [T, N, D] and dones as [T, N].")

        full_targets = torch.cat([left, right], dim=-1)  # [T, N, dof] in policy order
        if full_targets.shape[-1] != self.dof:
            raise ValueError(f"Trajectory DoF mismatch: traj={full_targets.shape[-1]}, expected={self.dof}")
        if self.traj_id < 0 or self.traj_id >= full_targets.shape[1]:
            raise ValueError(f"Invalid traj_id={self.traj_id}, valid range=[0, {full_targets.shape[1]-1}]")

        # reorder policy → real driver order, then select traj_id
        self.traj = full_targets[:, self.traj_id, :][:, policy2real_idx].to(dtype=torch.float32)  # [T, dof]
        self.dones = dones[:, self.traj_id].to(dtype=torch.bool)                                   # [T]

        self.target_pub = self.create_publisher(JointState, "/target_joint_states", T_JS_QOS)
        self.get_logger().info(
            f"Loaded traj: {traj_path}, targets_shape={tuple(self.traj.shape)}, "
            f"dones_shape={tuple(self.dones.shape)}, traj_id={self.traj_id}, publish_hz={self.publish_hz}"
        )

    def run_once(self) -> None:
        msg = JointState()
        msg.name = self.joint_names
        dt = 1.0 / self.publish_hz
        deadline = time.perf_counter()
        for i in range(self.traj.shape[0]):
            q = self.traj[i]
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = q.cpu().tolist()
            self.target_pub.publish(msg)

            if i % 20 == 0 or i == self.traj.shape[0] - 1:
                self.get_logger().info(f"publish step {i + 1}/{self.traj.shape[0]}")
            if bool(self.dones[i].item()):
                self.get_logger().info(f"Done reached at step {i}. Stop replay.")
                break

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
