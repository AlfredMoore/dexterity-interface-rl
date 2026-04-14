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


def _format_amp_token(value: float) -> str:
    token = f"{value:.6g}"
    return token.replace("-", "m").replace(".", "p")


def _resolve_traj_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    suffix = path.suffix or ".pt"
    pattern = f"{path.stem}_wf*_amp*{suffix}"
    candidates = sorted(path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Trajectory file not found: {path}")


class RunTrajNode(Node):
    def __init__(self):
        super().__init__("replay_action_node")
        self.lock = threading.Lock()
        self.has_joint_state = False

        system_id_dir = RMI_ROOT / "runtime" / "system_id"
        self.declare_parameter("driver_cfg_path", str(RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml"))
        self.declare_parameter("runtime_cfg_path", str(RMI_ROOT / "runtime" / "runtime_cfg.yaml"))
        self.declare_parameter("traj_path", str(system_id_dir / "actions_sysid_traj.pt"))
        self.declare_parameter("combo_key", "")
        # traj_id < 0 means run all trajectories sequentially.
        self.declare_parameter("traj_id", -1)
        # self.declare_parameter("action_scale", 1.0)
        self.declare_parameter("save_path", str(system_id_dir / "replay_action_joint_state_log.pt"))
        self.declare_parameter("publish_hz", 30.0)

        driver_cfg_path = self.get_parameter("driver_cfg_path").get_parameter_value().string_value
        runtime_cfg_path = self.get_parameter("runtime_cfg_path").get_parameter_value().string_value
        traj_path = self.get_parameter("traj_path").get_parameter_value().string_value
        combo_key = self.get_parameter("combo_key").get_parameter_value().string_value
        self.combo_key = combo_key.strip() if combo_key else None
        self.traj_id = int(self.get_parameter("traj_id").value)
        # cli_action_scale = float(self.get_parameter("action_scale").value)
        self.save_path_cfg = Path(self.get_parameter("save_path").get_parameter_value().string_value)
        if self.save_path_cfg.suffix == "":
            self.save_path_cfg = self.save_path_cfg.with_suffix(".pt")
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
        self.prev_actions = torch.zeros((1, self.action_num), dtype=torch.float32)
        
        # state space
        self.joint_poses = np.zeros(self.action_num, dtype=np.float32)
        self.prev_joint_poses = np.zeros(self.action_num, dtype=np.float32)
        self.joint_vels = np.zeros(self.action_num, dtype=np.float32)
        self.prev_joint_vels = np.zeros(self.action_num, dtype=np.float32)

        self.actions_all = torch.zeros((1, 1, self.action_num), dtype=torch.float32)  # [T, N, A]
        self.dones_all = torch.zeros((1, 1), dtype=torch.bool)  # [T, N]
        self.num_traj = 1
        self.traj_ids_to_run = [0]
        self.stim_joint_indices_all = None

        traj_path_resolved = _resolve_traj_path(traj_path)
        self._load_traj(str(traj_path_resolved), self.combo_key)
        suffix = "_all" if self.traj_id < 0 else f"_id{self.traj_id}"
        if self.combo_key:
            suffix += f"_{self.combo_key}"
        if self.waveform is not None:
            suffix += f"_wf{self.waveform}"
        if self.amplitude is not None:
            suffix += f"_amp{_format_amp_token(self.amplitude)}"
        if suffix not in self.save_path_cfg.stem:
            self.save_path_cfg = self.save_path_cfg.with_name(f"{self.save_path_cfg.stem}{suffix}{self.save_path_cfg.suffix}")
        self.save_path = str(self.save_path_cfg)
        self.get_logger().info(
            f"Loaded action traj: {traj_path_resolved}, actions_shape={tuple(self.actions_all.shape)}, "
            f"dones_shape={tuple(self.dones_all.shape)}, combo_key={self.combo_key}, "
            f"traj_id={self.traj_id}, run_traj_ids={self.traj_ids_to_run}, publish_hz={self.publish_hz}"
        )
        if self.waveform is not None:
            self.get_logger().info(f"waveform={self.waveform}")
        if self.amplitude is not None:
            self.get_logger().info(f"amplitude={self.amplitude}")
        if self.stim_joint_indices_all is not None:
            self.get_logger().info(f"Using stim_joint_indices(all): {self.stim_joint_indices_all}")
        
        
    def _load_traj(self, traj_path: str, combo_key: str | None = None):
        # actions shape: [T, N, A] or [T, A] -> [T, A]
        traj_data_all = torch.load(traj_path)
        traj_data = traj_data_all
        if "actions" not in traj_data_all:
            # Grid format: top-level dict of combo_key -> payload(with actions)
            if not isinstance(traj_data_all, dict) or len(traj_data_all) == 0:
                raise KeyError(
                    f"Trajectory file has no 'actions' and no valid combo dict: {traj_path}"
                )
            valid_combos = sorted(
                [
                    k
                    for k, v in traj_data_all.items()
                    if isinstance(v, dict) and "actions" in v
                ]
            )
            if len(valid_combos) == 0:
                raise KeyError(
                    f"No valid combo payload (dict with 'actions') found in {traj_path}"
                )
            if combo_key:
                if combo_key not in valid_combos:
                    show = valid_combos[:12]
                    more = "" if len(valid_combos) <= 12 else f" ... (+{len(valid_combos) - 12} more)"
                    raise KeyError(
                        f"combo_key '{combo_key}' not found in {traj_path}. "
                        f"Available: {show}{more}"
                    )
                self.combo_key = combo_key
            else:
                # Deterministic fallback
                self.combo_key = valid_combos[0]
                self.get_logger().info(f"combo_key not set, using first key: {self.combo_key}")
            traj_data = traj_data_all[self.combo_key]
        else:
            # Legacy flat payload
            if combo_key:
                self.get_logger().warning(
                    f"combo_key='{combo_key}' is ignored because trajectory file is not a grid dict."
                )
            self.combo_key = None

        if "actions" not in traj_data:
            raise KeyError(f"Missing key 'actions' in {traj_path}")
        actions = traj_data["actions"]  # [T, N, A] or [T, A]
        self.waveform = str(traj_data["waveform"]) if "waveform" in traj_data else None
        self.amplitude = float(traj_data["amplitude"]) if "amplitude" in traj_data else None
        stim_indices_all = None
        if "stim_joint_indices" in traj_data:
            stim_indices_all = [int(i) for i in traj_data["stim_joint_indices"]]
            for idx in stim_indices_all:
                if idx < 0 or idx >= self.action_num:
                    raise ValueError(f"Invalid stim_joint_index={idx}, expected in [0, {self.action_num-1}]")
        if actions.ndim == 3:
            self.num_traj = int(actions.shape[1])
            self.actions_all = actions.to(dtype=torch.float32).contiguous()
            if "dones" in traj_data and traj_data["dones"].ndim == 2:
                self.dones_all = traj_data["dones"].to(dtype=torch.bool).contiguous()
            else:
                self.dones_all = torch.zeros((self.actions_all.shape[0], self.num_traj), dtype=torch.bool)
            if self.traj_id < 0:
                self.traj_ids_to_run = list(range(self.num_traj))
            else:
                if self.traj_id >= self.num_traj:
                    raise ValueError(f"Invalid traj_id={self.traj_id}, valid range=[0, {self.num_traj-1}]")
                self.traj_ids_to_run = [self.traj_id]
            self.stim_joint_indices_all = stim_indices_all
        elif actions.ndim == 2:
            self.num_traj = 1
            self.actions_all = actions.to(dtype=torch.float32).unsqueeze(1).contiguous()  # [T,1,A]
            if "dones" in traj_data and traj_data["dones"].ndim == 1:
                self.dones_all = traj_data["dones"].to(dtype=torch.bool).unsqueeze(1).contiguous()
            else:
                self.dones_all = torch.zeros((self.actions_all.shape[0], 1), dtype=torch.bool)
            if self.traj_id > 0:
                raise ValueError("traj_id must be 0 or -1 for 2D action traces.")
            self.traj_ids_to_run = [0]
            self.stim_joint_indices_all = stim_indices_all
        else:
            raise ValueError(f"Unsupported actions shape: {tuple(actions.shape)}")
        if self.actions_all.shape[2] != self.action_num:
            raise ValueError(f"Action dim mismatch: actions={self.actions_all.shape[2]}, expected={self.action_num}")
        if stim_indices_all is None:
            self.stim_joint_indices_all = None

        self.home_joint_pos = None
        if "home_joint_pos" in traj_data:
            home_joint_pos = torch.as_tensor(traj_data["home_joint_pos"], dtype=torch.float32).flatten()
            if home_joint_pos.numel() != self.action_num:
                raise ValueError(
                    f"home_joint_pos dim mismatch: got={home_joint_pos.numel()}, expected={self.action_num}"
                )
            self.home_joint_pos = home_joint_pos.contiguous()


    def _sub_joint_state_cb(self, msg: JointState):
        with self.lock:
            self.joint_poses[:] = np.array(msg.position, dtype=np.float32)
            self.joint_vels[:] = np.array(msg.velocity, dtype=np.float32)
            self.has_joint_state = True

    def run_once(self) -> None:
        msg = JointState()
        msg.name = self.joint_names
        dt = 1.0 / self.publish_hz
        logs_per_traj: dict[str, dict] = {}

        self.get_logger().info("Waiting for first /joint_states message...")
        while rclpy.ok() and not self.has_joint_state:
            rclpy.spin_once(self, timeout_sec=0.1)
        runtime_home_joint_pos = torch.from_numpy(self.joint_poses.copy())
        if self.home_joint_pos is not None:
            home_joint_pos_ref = self.home_joint_pos
        else:
            home_joint_pos_ref = runtime_home_joint_pos
        for run_idx, traj_id in enumerate(self.traj_ids_to_run):
            self.get_logger().info(f"Start trajectory {run_idx + 1}/{len(self.traj_ids_to_run)}: traj_id={traj_id}")
            self.prev_actions.zero_()
            deadline = time.perf_counter()

            if self.stim_joint_indices_all is not None:
                if len(self.stim_joint_indices_all) == self.num_traj:
                    stim_joint_indices = [int(self.stim_joint_indices_all[traj_id])]
                else:
                    stim_joint_indices = [int(i) for i in self.stim_joint_indices_all]
            else:
                stim_joint_indices = None
            stim_idx_tensor = (
                torch.tensor(stim_joint_indices, dtype=torch.long)
                if stim_joint_indices is not None
                else None
            )

            raw_action_log: list[torch.Tensor] = []
            ema_action_log: list[torch.Tensor] = []
            target_log: list[torch.Tensor] = []
            joint_pos_log: list[torch.Tensor] = []
            joint_vel_log: list[torch.Tensor] = []
            done_log: list[bool] = []
            t_log: list[float] = []
            t0 = time.perf_counter()

            for i in range(self.actions_all.shape[0]):
                rclpy.spin_once(self, timeout_sec=0.0)
                with self.lock:
                    cur_q = self.joint_poses.copy()
                    cur_dq = self.joint_vels.copy()

                left_joint_pos = torch.from_numpy(cur_q[: self.left_dof]).unsqueeze(0)
                right_joint_pos = torch.from_numpy(cur_q[self.left_dof :]).unsqueeze(0)
                raw_actions = self.actions_all[i, traj_id].unsqueeze(0)  # [1, A]
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

                if stim_idx_tensor is not None:
                    targets_to_publish = home_joint_pos_ref.clone()
                    targets_to_publish[stim_idx_tensor] = full_targets[stim_idx_tensor]
                else:
                    targets_to_publish = full_targets

                msg.header.stamp = self.get_clock().now().to_msg()
                msg.position = targets_to_publish.tolist()
                self.target_pub.publish(msg)

                raw_action_log.append(raw_actions.squeeze(0).detach().cpu().clone())
                ema_action_log.append(ema_actions.squeeze(0).detach().cpu().clone())
                target_log.append(targets_to_publish.detach().cpu().clone())
                joint_pos_log.append(torch.from_numpy(cur_q).clone())
                joint_vel_log.append(torch.from_numpy(cur_dq).clone())
                done_log.append(bool(self.dones_all[i, traj_id].item()))
                t_log.append(time.perf_counter() - t0)

                if i % 20 == 0 or i == self.actions_all.shape[0] - 1:
                    self.get_logger().info(f"[traj_id={traj_id}] publish step {i + 1}/{self.actions_all.shape[0]}")
                if bool(self.dones_all[i, traj_id].item()):
                    self.get_logger().info(f"[traj_id={traj_id}] Done reached at step {i}. Stop this trajectory.")
                    break

                deadline += dt
                sleep_s = deadline - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    self.get_logger().warning(
                        f"[traj_id={traj_id}] Falling behind schedule by {-sleep_s:.4f} s at step {i}. "
                        f"Consider reducing publish_hz={self.publish_hz}."
                    )

            # After each trajectory, command full HOME and wait 5s for reset.
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = home_joint_pos_ref.tolist()
            self.target_pub.publish(msg)
            self.get_logger().info(f"[traj_id={traj_id}] Published HOME target. Sleep 5s for reset.")
            time.sleep(5.0)

            logs_per_traj[f"traj_{traj_id:03d}"] = {
                "traj_id": int(traj_id),
                "time_s": torch.tensor(t_log, dtype=torch.float64),
                "raw_action": torch.stack(raw_action_log, dim=0).to(torch.float32),
                "ema_action": torch.stack(ema_action_log, dim=0).to(torch.float32),
                "target": torch.stack(target_log, dim=0).to(torch.float32),
                "joint_pos": torch.stack(joint_pos_log, dim=0).to(torch.float32),
                "joint_vel": torch.stack(joint_vel_log, dim=0).to(torch.float32),
                "done": torch.tensor(done_log, dtype=torch.bool),
                "stim_joint_indices": stim_joint_indices,
            }

        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        common = {
            "joint_names": self.joint_names,
            "home_joint_pos_ref": home_joint_pos_ref.to(torch.float32),
            "combo_key": self.combo_key,
            "waveform": self.waveform,
            "amplitude": self.amplitude,
            "traj_ids_run": [int(i) for i in self.traj_ids_to_run],
        }
        if len(self.traj_ids_to_run) == 1:
            # Keep backward-compatible format for single trajectory replay.
            only = logs_per_traj[f"traj_{self.traj_ids_to_run[0]:03d}"]
            save_payload = {**common, **only}
        else:
            save_payload = {
                **common,
                "multi_traj": True,
                "per_traj": logs_per_traj,
            }
        torch.save(save_payload, self.save_path)
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
