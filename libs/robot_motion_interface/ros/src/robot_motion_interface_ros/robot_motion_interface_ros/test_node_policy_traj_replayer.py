"""
Before running this node, make sure to:
    1. Copy the simulation run directory to your local specified path.
    2. The simulation run directory should contain the trained policy and parameters files. Refer to IsaacLab + RSL-RL documentation for details on how to train and export the policy.
"""


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import torch
import numpy as np
import threading
from pathlib import Path
import importlib.util
import yaml
import os
import time
from typing import Dict
import pinocchio as pin     # TODO: install pinocchio

# Customized Interface

# --- QoS Config: low latency (Best Effort) ---
from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
JS_QOS = HIGH_PERF_QOS
T_JS_QOS = HIGH_RELIA_QOS

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError(f"Cannot locate module spec for {__name__}")
# spec.origin  = .../libs/robot_motion_interface/src/robot_motion_interface/__init__.py
# .parent x3   = .../libs/robot_motion_interface/
# .parent x4   = .../libs/
RMI_ROOT  = Path(spec.origin).parent.parent.parent          # .../libs/robot_motion_interface/
LIBS_ROOT = RMI_ROOT.parent                                  # .../libs/

SINGLE_CHAIN_URDF_PATH = str((LIBS_ROOT / "robot_description/rl/panda_w_tesollo.urdf").resolve())
DUAL_CHAIN_URDF_PATH   = str((LIBS_ROOT / "robot_description/rl/bimanual_panda_tesollo.urdf").resolve())


class TrajectoryReplay:
    def __init__(self, traj_path: str, traj_id: int, device: torch.device):
        data = torch.load(traj_path, map_location="cpu")
        if isinstance(data, dict):
            actions = data["actions"]
            dones = data.get("dones", None)
        else:
            actions = data
            dones = None

        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim == 3:
            actions = actions[:, traj_id, :]
        self.actions = actions.to(device=device)
        self.total_steps = int(self.actions.shape[0])

        if dones is None:
            self.dones = torch.zeros((self.total_steps,), dtype=torch.bool, device=device)
        else:
            dones = torch.as_tensor(dones, dtype=torch.bool)
            if dones.ndim == 2:
                dones = dones[:, traj_id]
            self.dones = dones.to(device=device)

        self.step_idx = 0
        self._last_action = self.actions[-1].unsqueeze(0)

    def next_action(self) -> tuple[torch.Tensor, bool, int]:
        if self.step_idx >= self.total_steps:
            return self._last_action, True, self.step_idx

        action = self.actions[self.step_idx].unsqueeze(0)
        done = bool(self.dones[self.step_idx].item())
        step = self.step_idx
        self.step_idx += 1
        if self.step_idx >= self.total_steps:
            done = True
        return action, done, step

class RLPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_policy_node')
        self.lock = threading.Lock()
        self.observation_grp = MutuallyExclusiveCallbackGroup()
        self.inference_grp = MutuallyExclusiveCallbackGroup()
        
        # 1. Parameters
        self.declare_parameter('hand_env_cfg_path', str(RMI_ROOT/'runtime'/'HandEnv.yaml'))
        self.declare_parameter('runtime_cfg_path', str(RMI_ROOT/'runtime'/'runtime_cfg.yaml'))
        self.declare_parameter('traj_path', str(RMI_ROOT/'runtime'/'actions_done_trace.pt'))
        self.declare_parameter('traj_id', 0)
        hand_env_cfg_path: str = self.get_parameter('hand_env_cfg_path').value
        runtime_cfg_path: str = self.get_parameter('runtime_cfg_path').value
        traj_path: str = self.get_parameter('traj_path').value
        traj_id: int = int(self.get_parameter('traj_id').value)
        if not os.path.exists(hand_env_cfg_path):
            raise FileNotFoundError(f"HandEnv file not found at: {hand_env_cfg_path}")
        if not os.path.exists(runtime_cfg_path):
            raise FileNotFoundError(f"Runtime config file not found at: {runtime_cfg_path}")
        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Trajectory file not found at: {traj_path}")
        self.get_logger().info(f"Loaded HandEnv from: {hand_env_cfg_path}")
        self.get_logger().info(f"Loaded runtime config from: {runtime_cfg_path}")
        self.get_logger().info(f"Loaded trajectory from: {traj_path}, traj_id={traj_id}")

        # load runtime cfg and env cfg
        with open(hand_env_cfg_path, 'r') as f:
            self.env_cfg = yaml.safe_load(f)
        with open(runtime_cfg_path, 'r') as f:
            self.runtime_cfg = yaml.safe_load(f)

        self.dt: float = self.runtime_cfg['dt']  # Default to 60 Hz if not specified
        self.ema: float = self.runtime_cfg.get('action_EMA', self.env_cfg["env"]["action"]["actionEMA"])

        self.init_left_joint_pose: dict[str, float] = self.env_cfg['experiment']['pre_grasp']['left_joint_pose']
        self.init_right_joint_pose: dict[str, float] = self.env_cfg['experiment']['pre_grasp']['right_joint_pose']
        self._arm_joint_names: list[str] = self.env_cfg["env"]["robot"]["jointNames"]["arm"]
        self._hand_joint_names: list[str] = self.env_cfg["env"]["robot"]["jointNames"]["hand"]
        self._chain_joint_names: list[str] = self._arm_joint_names + self._hand_joint_names
        self._n_arm: int = self.env_cfg['env']['action']['armDof']
        self._n_hand: int = self.env_cfg['env']['action']['handDof']
        self._action_per_chain: int = self.env_cfg['env']['action']['actionPerChain']
        self._action_num: int = self.env_cfg['env']['action']['actionSpace']
        self._action_scale: float = self.env_cfg['env']['action']['actionScale']

        self.left_joint_pose_soft_lower = np.array(self.runtime_cfg['robot_joint_limits_dict']['left_joint_pose_soft_lower'], dtype=np.float32)
        self.right_joint_pose_soft_lower = np.array(self.runtime_cfg['robot_joint_limits_dict']['right_joint_pose_soft_lower'], dtype=np.float32)
        self.left_joint_pose_soft_upper = np.array(self.runtime_cfg['robot_joint_limits_dict']['left_joint_pose_soft_upper'], dtype=np.float32)
        self.right_joint_pose_soft_upper = np.array(self.runtime_cfg['robot_joint_limits_dict']['right_joint_pose_soft_upper'], dtype=np.float32)

        # 3. Replay + observation setup
        self.device = torch.device("cpu")
        self.replay = TrajectoryReplay(traj_path=traj_path, traj_id=traj_id, device=self.device)
        if self.replay.actions.shape[-1] != self._action_num:
            raise ValueError(
                f"Trajectory action dim mismatch: got {self.replay.actions.shape[-1]}, expected {self._action_num}"
            )
        self.actor_components = self.env_cfg["env"]["observation"]["actor"]
        self._obs_unstacked_space = int(sum(self.env_cfg["env"]["observation"]["obsDOF"][k] for k in self.actor_components))
        self._pinocchio_init()

        self.policy_action_indices_dict = self.runtime_cfg['policy_action_indices_dict']
        self.robot_action_scale_dict = {
            "left_joint_vel_action": torch.tensor(
                self.runtime_cfg['robot_action_scale_dict']["left_joint_vel_action"], dtype=torch.float32, device=self.device
            ),
            "right_joint_vel_action": torch.tensor(
                self.runtime_cfg['robot_action_scale_dict']["right_joint_vel_action"], dtype=torch.float32, device=self.device
            ),
        }
        self.robot_joint_limits_dict_t = {
            "left_joint_pose_soft_lower": torch.tensor(
                self.runtime_cfg['robot_joint_limits_dict']["left_joint_pose_soft_lower"], dtype=torch.float32, device=self.device
            ),
            "left_joint_pose_soft_upper": torch.tensor(
                self.runtime_cfg['robot_joint_limits_dict']["left_joint_pose_soft_upper"], dtype=torch.float32, device=self.device
            ),
            "right_joint_pose_soft_lower": torch.tensor(
                self.runtime_cfg['robot_joint_limits_dict']["right_joint_pose_soft_lower"], dtype=torch.float32, device=self.device
            ),
            "right_joint_pose_soft_upper": torch.tensor(
                self.runtime_cfg['robot_joint_limits_dict']["right_joint_pose_soft_upper"], dtype=torch.float32, device=self.device
            ),
        }

        # 4. State Management for Actions and Targets
        self.joint_poses = torch.zeros((1, self._action_num), dtype=torch.float32, device=self.device)   # [1, dof]
        self.joint_vels = torch.zeros((1, self._action_num), dtype=torch.float32, device=self.device)    # [1, dof]
        self.targets = torch.zeros((1, self._action_num), dtype=torch.float32, device=self.device)       # [1, dof]
        
        ## joint states scaled to [-1, 1] for policy input
        self.left_joint_pos_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.right_joint_pos_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.left_joint_vel_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.right_joint_vel_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)

        self.left_dof_targets = torch.zeros((1, self._action_per_chain), dtype=torch.float, device=self.device)
        self.right_dof_targets = torch.zeros((1, self._action_per_chain), dtype=torch.float, device=self.device)

        self.actions = torch.zeros((1, self._action_num), dtype=torch.float32, device=self.device)   # [1, dof]
        
        
        self.prev_actions = torch.zeros((1, self._action_num), dtype=torch.float32, device=self.device)   # [1, dof]
        self.prev_obs: torch.Tensor = torch.zeros((1, self._obs_unstacked_space), device=self.device).float()   # [env, obs_dim]

        # 5. Communication & Timers
        self.target_pub = self.create_publisher(JointState, '/target_joint_states', T_JS_QOS)
        self.done_pub = self.create_publisher(Bool, '/traj_done', T_JS_QOS)
        ## Observation mutex group
        self.create_subscription(JointState, '/joint_states', self._sub_joint_state_cb, JS_QOS, callback_group=self.observation_grp)
        # self.fk_timer = self.create_timer(1.0 / node_config['fk_rate'], self._pinocchio_forward_kinematics, callback_group=self.observation_grp)  # FK update
        
        self.policy_timer = self.create_timer(self.dt, self._policy_update_loop, callback_group=self.inference_grp)

        self.get_logger().info("RLPolicyNode initialized.")
        input("[RLPolicyNode Init] Press Enter to go to pre-grasp state...")
        self._set_pre_grasp_state()
        time.sleep(5.0)  # Allow time to reach pre-grasp state
        self.get_logger().info("RLPolicyNode is in pre-grasp state.")
        input("[RLPolicyNode Init] Press Enter to start policy inference...")


    def _pinocchio_init(self):
        # left and right hand share the dual-chain urdf with world frame base
        self.pin_model = pin.buildModelFromUrdf(DUAL_CHAIN_URDF_PATH)
        self.pin_data = self.pin_model.createData()
        assert self.pin_model.nq == self._action_num, (
            f"Pinocchio model nq ({self.pin_model.nq}) does not match expected action num ({self._action_num})"
        )

        finger_tip_links: list[str] = self.env_cfg["env"]["robot"]["linkNames"]["finger_tips"]
        hand_base_links: list[str] = self.env_cfg["env"]["robot"]["linkNames"]["hand_palm"]

        self.left_fingertip_ids: list[int] = [self.pin_model.getFrameId(f"left_{n}") for n in finger_tip_links]
        self.right_fingertip_ids: list[int] = [self.pin_model.getFrameId(f"right_{n}") for n in finger_tip_links]
        self.left_hand_base_ids: list[int] = [self.pin_model.getFrameId(f"left_{n}") for n in hand_base_links]
        self.right_hand_base_ids: list[int] = [self.pin_model.getFrameId(f"right_{n}") for n in hand_base_links]

        # self.l_hand_base_pos: np.ndarray = np.zeros((1, len(self.hand_base_id), 3))     # 1, 1, 3
        # self.r_hand_base_pos: np.ndarray = np.zeros((1, len(self.hand_base_id), 3))
        # self.l_fingertips_pos: np.ndarray = np.zeros((1, len(self.fingertip_ids), 3))   # 1, 3, 3
        # self.r_fingertips_pos: np.ndarray = np.zeros((1, len(self.fingertip_ids), 3))

    def _pinocchio_forward_kinematics(self, q) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update bimanual fingertip and hand-base positions with dual-chain urdf model."""
        q = q.astype(np.float64)
        assert q.shape == (self.pin_model.nq,), f"Pin model expects joint_pos shape ({self.pin_model.nq}), got {q.shape}"
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        l_hand_base_pos = np.array([self.pin_data.oMf[id].translation for id in self.left_hand_base_ids], dtype=np.float32)
        r_hand_base_pos = np.array([self.pin_data.oMf[id].translation for id in self.right_hand_base_ids], dtype=np.float32)
        l_fingertips_pos = np.array([self.pin_data.oMf[id].translation for id in self.left_fingertip_ids], dtype=np.float32)
        r_fingertips_pos = np.array([self.pin_data.oMf[id].translation for id in self.right_fingertip_ids], dtype=np.float32)
        return l_hand_base_pos, l_fingertips_pos, r_hand_base_pos, r_fingertips_pos

    def _set_pre_grasp_state(self):
        """Set initial joint pose from config"""
        
        with self.lock:
            left_joint_pose = np.array([self.init_left_joint_pose[n] for n in self._chain_joint_names], dtype=np.float32)
            right_joint_pose = np.array([self.init_right_joint_pose[n] for n in self._chain_joint_names], dtype=np.float32)
            init_q = np.concatenate([left_joint_pose, right_joint_pose], axis=0)
            self.joint_poses[0, :] = torch.from_numpy(init_q).to(self.device)
            self.targets[0, :] = self.joint_poses[0, :]  # [dof]
            self.left_dof_targets[0, :] = self.targets[0, :self._action_per_chain]
            self.right_dof_targets[0, :] = self.targets[0, self._action_per_chain:]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = self.targets[0].cpu().tolist()  # [dof]
        self.target_pub.publish(msg)


    def _sub_joint_state_cb(self, msg: JointState):
        """update scaled joint states in [-1,1]"""

        with self.lock:
            self.joint_poses[0, :] = torch.tensor(msg.position, dtype=torch.float32, device=self.device)
            self.joint_vels[0, :] = torch.tensor(msg.velocity, dtype=torch.float32, device=self.device)

            # self.left_joint_pos_scaled[:] = scale(
            #     target=self.joint_poses[:self._action_per_chain],
            #     lower=self.left_joint_pose_soft_lower,
            #     upper=self.left_joint_pose_soft_upper,
            # )
            # self.right_joint_pos_scaled[:] = scale(
            #     self.joint_poses[self._action_per_chain:],
            #     lower=self.right_joint_pose_soft_lower,
            #     upper=self.right_joint_pose_soft_upper,
            # )
            # self.left_joint_vel_scaled = self.joint_vels[:, :self._action_per_chain] / self.runtime_cfg['robot_joint_limits_dict']['left_joint_vel']
            # self.right_joint_vel_scaled = self.joint_vels[:, self._action_per_chain:] / self.runtime_cfg['robot_joint_limits_dict']['right_joint_vel']

    def _policy_update_loop(self):
        """Control loop: Fuses states and integrates delta actions"""
        if self.targets is None:
            return

        # waiting for assymmetric actor critic HAND policy
        with self.lock:
            cur_joint_poses = self.joint_poses[0].detach().cpu().numpy().copy()
            cur_l_joint_poses = cur_joint_poses[:self._action_per_chain].copy()
            cur_r_joint_poses = cur_joint_poses[self._action_per_chain:].copy()
            cur_l_joint_vels = self.joint_vels[0, :self._action_per_chain].detach().cpu().numpy().copy()
            cur_r_joint_vels = self.joint_vels[0, self._action_per_chain:].detach().cpu().numpy().copy()
            cur_targets = self.targets.clone()


        # has been converted to float32
        l_hand_base_pos, l_fingertips_pos, r_hand_base_pos, r_fingertips_pos = self._pinocchio_forward_kinematics(cur_joint_poses)

        left_joint_pos_scaled = scale(
            target=cur_l_joint_poses,
            lower=self.left_joint_pose_soft_lower,
            upper=self.left_joint_pose_soft_upper,
        )
        right_joint_pos_scaled = scale(
            target=cur_r_joint_poses,
            lower=self.right_joint_pose_soft_lower,
            upper=self.right_joint_pose_soft_upper,
        )
        left_joint_vel_scaled = cur_l_joint_vels / self.runtime_cfg['robot_joint_limits_dict']['left_joint_vel']
        right_joint_vel_scaled = cur_r_joint_vels / self.runtime_cfg['robot_joint_limits_dict']['right_joint_vel']
        cur_targets_np = cur_targets[0].detach().cpu().numpy()
        bottle_geom_cfg = np.zeros((4,), dtype=np.float32)
        if len(self.runtime_cfg.get("env_jar_geom_cfg", [])) > 0:
            bottle_geom_cfg = np.array(self.runtime_cfg["env_jar_geom_cfg"][0], dtype=np.float32)

        # compose actor observation
        # for single env, no batch dim
        full_obs = {
            # proprioception
            ## Q space
            "leftJointPosScaled": left_joint_pos_scaled,    # [action_per_chain,]
            "rightJointPosScaled": right_joint_pos_scaled,  # [action_per_chain,]
            "leftJointVelScaled": left_joint_vel_scaled,
            "rightJointVelScaled": right_joint_vel_scaled,
            ## targets - Cartesian space
            "leftTargets": cur_targets_np[:self._action_per_chain],   # [action_per_chain,]
            "rightTargets": cur_targets_np[self._action_per_chain:],  # [action_per_chain,]
            ## end effectors - Cartesian space
            "leftFingerTipsPos": l_fingertips_pos.flatten(),    # [_n_hand*3]
            "rightFingerTipsPos": r_fingertips_pos.flatten(),   # [_n_hand*3]
            "leftHandBasePos": l_hand_base_pos.flatten(),  # [3,]
            "rightHandBasePos": r_hand_base_pos.flatten(), # [3,]

            # exteroception placeholder (vision removed in replay mode)
            "bottleBodyPos": np.zeros((3,), dtype=np.float32),
            "bottleBodyRot": np.zeros((4,), dtype=np.float32),
            "bottleCapPos": np.zeros((3,), dtype=np.float32),
            "bottleCapRot": np.zeros((4,), dtype=np.float32),
            "deltaBottleCapJointPos": np.zeros((1,), dtype=np.float32),
            "bottleGeomCfg": bottle_geom_cfg,
        }
        
        actor_obs: torch.Tensor = torch.from_numpy(
            np.concatenate([full_obs[key] for key in self.actor_components], axis=-1)
        ).float().unsqueeze(0).to(self.device)  # [1, obs_dim]

        # cur_obs = ... # current obs, tensor
        prev_obs = self.prev_obs.clone()

        stacked_obs = torch.cat(
            (
                actor_obs,
                prev_obs,   # Previous observations
            ),
        dim=-1,
        )

        self.prev_obs = actor_obs.clone()

        # TODO: Assymmetric Actor obseervation space, align with the trained model
        observations = {"policy": torch.clamp(stacked_obs, -100.0, 100.0)}
        _ = observations
        raw_actions, traj_done, traj_step = self.replay.next_action()

        with self.lock:
            # self.prev_left_joint_vel[:] = self.actions[:, self.policy_action_indices_dict["left"]]
            # self.prev_right_joint_vel[:] = self.actions[:, self.policy_action_indices_dict["right"]]
            self.prev_actions[:] = self.actions.clone()

        left_joint_pos_t = torch.from_numpy(cur_l_joint_poses).to(dtype=torch.float32, device=self.device).unsqueeze(0)
        right_joint_pos_t = torch.from_numpy(cur_r_joint_poses).to(dtype=torch.float32, device=self.device).unsqueeze(0)

        (
            self.left_dof_targets[:],
            self.right_dof_targets[:],
            self.actions[:],
        ) = compute_targets(
            dt=self.dt,
            actions=raw_actions,
            prev_actions=self.prev_actions,
            action_EMA=self.ema,
            actions_scale=self._action_scale,
            
            left_joint_pos=left_joint_pos_t,
            right_joint_pos=right_joint_pos_t,
            policy_action_indices_dict=self.policy_action_indices_dict,
            robot_action_scale_dict=self.robot_action_scale_dict,
            robot_joint_limits_dict=self.robot_joint_limits_dict_t,
        )
        next_targets = torch.cat([self.left_dof_targets, self.right_dof_targets], dim=-1)
        with self.lock:
            self.targets[0,:] = next_targets[0,:].clone()

        # 4. Publish to Driver
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = next_targets[0].cpu().tolist()  # [dof]
        self.target_pub.publish(msg)
        done_msg = Bool()
        done_msg.data = traj_done
        self.done_pub.publish(done_msg)
        if traj_done:
            self.get_logger().info(f"Replay done at step={traj_step}. Shutting down.")
            self.policy_timer.cancel()
            rclpy.shutdown()


# utils from HAND
# transfer to numpy
def scale(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """scale to [-1, 1]"""
    return 2.0 * (target - lower) / (upper - lower) - 1.0

def compute_targets(
    dt: float,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    action_EMA: float,
    actions_scale: float,
    
    left_joint_pos: torch.Tensor,
    right_joint_pos: torch.Tensor,
    policy_action_indices_dict: Dict[str, list[int]],
    robot_action_scale_dict: Dict[str, torch.Tensor],
    robot_joint_limits_dict: Dict[str, torch.Tensor],
    
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    return: left_dof_targets, right_dof_targets, actions
    """

    # Action scaling and smoothing
    # actions: (left_arm+right_arm , )
    actions = actions.clamp(-1.0, 1.0) * action_EMA + prev_actions * (1.0 - action_EMA)   # EMA smoothing

    # Action to articulation target
    ## left_arm
    left_actions = actions[:, policy_action_indices_dict["left"]]
    left_dof_targets = torch.clamp(
        left_joint_pos + left_actions * dt * robot_action_scale_dict["left_joint_vel_action"] * actions_scale,
        min=robot_joint_limits_dict["left_joint_pose_soft_lower"], 
        max=robot_joint_limits_dict["left_joint_pose_soft_upper"]
    )
    
    ## right_arm
    right_actions = actions[:, policy_action_indices_dict["right"]]
    right_dof_targets = torch.clamp(
        right_joint_pos + right_actions * dt * robot_action_scale_dict["right_joint_vel_action"] * actions_scale,
        min=robot_joint_limits_dict["right_joint_pose_soft_lower"], 
        max=robot_joint_limits_dict["right_joint_pose_soft_upper"]
    )
    return left_dof_targets, right_dof_targets, actions


def main(args=None):
    rclpy.init(args=args)
    node = RLPolicyNode()
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
