"""Replay policy actions and compute targets online, then publish /target_joint_states."""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import torch
import yaml

from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS

JS_QOS = HIGH_PERF_QOS
T_JS_QOS = HIGH_RELIA_QOS


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent


def compute_targets(
    dt: float,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    action_EMA: float,
    actions_scale: float,
    
    left_joint_pos: torch.Tensor,
    right_joint_pos: torch.Tensor,
    
    policy_action_indices_dict: dict[str, list[int]],
    robot_action_scale_dict: dict[str, torch.Tensor],
    robot_joint_limits_dict: dict[str, torch.Tensor],

) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    actions = actions.clamp(-1.0, 1.0) * action_EMA + prev_actions * (1.0 - action_EMA)

    left_actions = actions[:, policy_action_indices_dict["left"]]
    left_dof_targets = torch.clamp(
        left_joint_pos + left_actions * dt * robot_action_scale_dict["left_joint_vel_action"] * actions_scale,
        min=robot_joint_limits_dict["left_joint_pose_soft_lower"],
        max=robot_joint_limits_dict["left_joint_pose_soft_upper"],
    )

    right_actions = actions[:, policy_action_indices_dict["right"]]
    right_dof_targets = torch.clamp(
        right_joint_pos + right_actions * dt * robot_action_scale_dict["right_joint_vel_action"] * actions_scale,
        min=robot_joint_limits_dict["right_joint_pose_soft_lower"],
        max=robot_joint_limits_dict["right_joint_pose_soft_upper"],
    )
    return left_dof_targets, right_dof_targets, actions


class RunTrajNode(Node):
    def __init__(self):
        super().__init__("replay_action_node")
        self.lock = threading.Lock()
        self.has_joint_state = False

        self.declare_parameter("driver_cfg_path", str(RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml"))
        self.declare_parameter("runtime_cfg_path", str(RMI_ROOT / "runtime" / "runtime_cfg.yaml"))
        self.declare_parameter("traj_path", str(RMI_ROOT / "runtime" / "actions_dones_targets.pt"))
        self.declare_parameter("traj_id", 0)
        # self.declare_parameter("action_scale", 1.0)
        self.declare_parameter("save_path", str(RMI_ROOT / "runtime" / "replay_action_joint_state_log.pt"))
        self.declare_parameter("publish_hz", 30.0)

        driver_cfg_path = self.get_parameter("driver_cfg_path").get_parameter_value().string_value
        runtime_cfg_path = self.get_parameter("runtime_cfg_path").get_parameter_value().string_value
        traj_path = self.get_parameter("traj_path").get_parameter_value().string_value
        self.traj_id = int(self.get_parameter("traj_id").value)
        # cli_action_scale = float(self.get_parameter("action_scale").value)
        self.save_path = self.get_parameter("save_path").get_parameter_value().string_value
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        with open(driver_cfg_path, "r", encoding="utf-8") as f:
            driver_cfg = yaml.safe_load(f)
        with open(runtime_cfg_path, "r", encoding="utf-8") as f:
            runtime_cfg = yaml.safe_load(f)

        l_names = ["left_" + n for n in driver_cfg["left_panda_joint_names"] + driver_cfg["left_tesollo_joint_names"]]
        r_names = ["right_" + n for n in driver_cfg["right_panda_joint_names"] + driver_cfg["right_tesollo_joint_names"]]
        self.joint_names = l_names + r_names
        self.left_dof = len(l_names)
        self.right_dof = len(r_names)
        self.action_num = self.left_dof + self.right_dof

        # Node pub and sub
        self.target_pub = self.create_publisher(JointState, "/target_joint_states", T_JS_QOS)
        self.create_subscription(JointState, "/joint_states", self._sub_joint_state_cb, JS_QOS)

        # Runtime cfg
        self.dt = float(runtime_cfg["dt"])
        self.action_EMA = float(runtime_cfg["action_EMA"])
        self.action_scale = float(runtime_cfg["action_scale"])
        self.policy_action_indices_dict = runtime_cfg["policy_action_indices_dict"]
        self.robot_action_scale_dict = {
            "left_joint_vel_action": torch.tensor(
                runtime_cfg["robot_action_scale_dict"]["left_joint_vel_action"], dtype=torch.float32
            ).unsqueeze(0),
            "right_joint_vel_action": torch.tensor(
                runtime_cfg["robot_action_scale_dict"]["right_joint_vel_action"], dtype=torch.float32
            ).unsqueeze(0),
        }
        self.robot_joint_limits_dict = {
            "left_joint_pose_soft_lower": torch.tensor(
                runtime_cfg["robot_joint_limits_dict"]["left_joint_pose_soft_lower"], dtype=torch.float32
            ).unsqueeze(0),
            "left_joint_pose_soft_upper": torch.tensor(
                runtime_cfg["robot_joint_limits_dict"]["left_joint_pose_soft_upper"], dtype=torch.float32
            ).unsqueeze(0),
            "right_joint_pose_soft_lower": torch.tensor(
                runtime_cfg["robot_joint_limits_dict"]["right_joint_pose_soft_lower"], dtype=torch.float32
            ).unsqueeze(0),
            "right_joint_pose_soft_upper": torch.tensor(
                runtime_cfg["robot_joint_limits_dict"]["right_joint_pose_soft_upper"], dtype=torch.float32
            ).unsqueeze(0),
        }

        # action space
        self.actions = torch.zeros((1, self.action_num), dtype=torch.float32)
        self.prev_actions = torch.zeros((1, self.action_num), dtype=torch.float32)
        self.targets = torch.zeros((1, self.action_num), dtype=torch.float32)
        self.prev_targets = torch.zeros((1, self.action_num), dtype=torch.float32)
        
        # state space
        self.joint_poses = np.zeros(self.action_num, dtype=np.float32)
        self.prev_joint_poses = np.zeros(self.action_num, dtype=np.float32)
        self.joint_vels = np.zeros(self.action_num, dtype=np.float32)
        self.prev_joint_vels = np.zeros(self.action_num, dtype=np.float32)

        self._load_traj(traj_path)
        self.get_logger().info(
            f"Loaded action traj: {traj_path}, actions_shape={tuple(self.actions.shape)}, "
            f"dones_shape={tuple(self.dones.shape)}, traj_id={self.traj_id}, publish_hz={self.publish_hz}"
        )
        
        
    def _load_traj(self, traj_path):
        # actions shape: [T, N, A] or [T, A] -> [T, A]
        traj_data = torch.load(traj_path)
        if "actions" not in traj_data:
            raise KeyError(f"Missing key 'actions' in {traj_path}")
        actions = traj_data["actions"]  # [T, N, A] or [T, A]
        if actions.ndim == 3:
            if self.traj_id < 0 or self.traj_id >= actions.shape[1]:
                raise ValueError(f"Invalid traj_id={self.traj_id}, valid range=[0, {actions.shape[1]-1}]")
            self.actions = actions[:, self.traj_id, :].to(dtype=torch.float32).contiguous()
            if "dones" in traj_data:
                self.dones = traj_data["dones"][:, self.traj_id].to(dtype=torch.bool).contiguous()
            else:
                self.dones = torch.zeros((self.actions.shape[0],), dtype=torch.bool)
        elif actions.ndim == 2:
            self.actions = actions.to(dtype=torch.float32).contiguous()
            if "dones" in traj_data and traj_data["dones"].ndim == 1:
                self.dones = traj_data["dones"].to(dtype=torch.bool).contiguous()
            else:
                self.dones = torch.zeros((self.actions.shape[0],), dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported actions shape: {tuple(actions.shape)}")
        if self.actions.shape[1] != self.action_num:
            raise ValueError(f"Action dim mismatch: actions={self.actions.shape[1]}, expected={self.action_num}")


    def _sub_joint_state_cb(self, msg: JointState):
        with self.lock:
            self.joint_poses[:] = np.array(msg.position, dtype=np.float32)
            self.joint_vels[:] = np.array(msg.velocity, dtype=np.float32)
            self.has_joint_state = True

    def run_once(self) -> None:
        msg = JointState()
        msg.name = self.joint_names
        dt = 1.0 / self.publish_hz
        deadline = time.perf_counter()

        raw_action_log: list[torch.Tensor] = []
        ema_action_log: list[torch.Tensor] = []
        target_log: list[torch.Tensor] = []
        joint_pos_log: list[torch.Tensor] = []
        joint_vel_log: list[torch.Tensor] = []
        done_log: list[bool] = []
        t_log: list[float] = []

        self.get_logger().info("Waiting for first /joint_states message...")
        while rclpy.ok() and not self.has_joint_state:
            rclpy.spin_once(self, timeout_sec=0.1)

        t0 = time.perf_counter()
        for i in range(self.actions.shape[0]):
            rclpy.spin_once(self, timeout_sec=0.0)
            with self.lock:
                cur_q = self.joint_poses.copy()
                cur_dq = self.joint_vels.copy()

            left_joint_pos = torch.from_numpy(cur_q[: self.left_dof]).unsqueeze(0)
            right_joint_pos = torch.from_numpy(cur_q[self.left_dof :]).unsqueeze(0)
            raw_actions = self.actions[i].unsqueeze(0)  # [1, A]

            left_targets, right_targets, ema_actions = compute_targets(
                self.dt,
                raw_actions,
                self.prev_actions,
                self.action_EMA,
                self.action_scale,
                left_joint_pos,
                right_joint_pos,
                self.policy_action_indices_dict,
                self.robot_action_scale_dict,
                self.robot_joint_limits_dict,
            )
            self.prev_actions[:] = ema_actions
            full_targets = torch.cat([left_targets, right_targets], dim=-1).squeeze(0)

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = full_targets.tolist()
            self.target_pub.publish(msg)

            raw_action_log.append(raw_actions.squeeze(0).detach().cpu().clone())
            ema_action_log.append(ema_actions.squeeze(0).detach().cpu().clone())
            target_log.append(full_targets.detach().cpu().clone())
            joint_pos_log.append(torch.from_numpy(cur_q).clone())
            joint_vel_log.append(torch.from_numpy(cur_dq).clone())
            done_log.append(bool(self.dones[i].item()))
            t_log.append(time.perf_counter() - t0)

            if i % 20 == 0 or i == self.actions.shape[0] - 1:
                self.get_logger().info(f"publish step {i + 1}/{self.actions.shape[0]}")
            if bool(self.dones[i].item()):
                self.get_logger().info(f"Done reached at step {i}. Stop replay.")
                break

            deadline += dt
            sleep_s = deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        torch.save(
            {
                "joint_names": self.joint_names,
                "time_s": torch.tensor(t_log, dtype=torch.float64),
                "raw_action": torch.stack(raw_action_log, dim=0).to(torch.float32),
                "ema_action": torch.stack(ema_action_log, dim=0).to(torch.float32),
                "target": torch.stack(target_log, dim=0).to(torch.float32),
                "joint_pos": torch.stack(joint_pos_log, dim=0).to(torch.float32),
                "joint_vel": torch.stack(joint_vel_log, dim=0).to(torch.float32),
                "done": torch.tensor(done_log, dtype=torch.bool),
            },
            self.save_path,
        )
        self.get_logger().info(f"Saved replay log to: {self.save_path}")


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
