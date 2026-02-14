"""
Before running this node, make sure to:
    1. Copy the simulation run directory to your local specified path.
    2. The simulation run directory should contain the trained policy and parameters files. Refer to IsaacLab + RSL-RL documentation for details on how to train and export the policy.
"""


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import torch
import numpy as np
import threading
import pyrealsense2 as rs
import cv2
from pathlib import Path
import importlib.util
import yaml
import os
import time
from typing import Dict
import pinocchio as pin     # TODO: install pinocchio

# Customized Interface

# --- QoS Config: low latency (Best Effort) ---
HIGH_PERF_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError(f"Cannot locate module spec for {__name__}")
RMI_ROOT = Path(spec.origin).parent.parent.parent
URDF_PATH = str((RMI_ROOT / "robot_description/rl/panda_w_tesollo.urdf").resolve())

class RLPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_policy_node')
        self.lock = threading.Lock()
        self.observation_grp = MutuallyExclusiveCallbackGroup()
        self.inference_grp = MutuallyExclusiveCallbackGroup()
        
        # 1. Parameters
        self.declare_parameter('config_path', str(RMI_ROOT/'config'/'rl_policy_node_config.yaml'))
        config_path: str = self.get_parameter('config_path').value
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as f:
            node_config = yaml.safe_load(f)
        self.get_logger().info(f"Loaded config from: {config_path}")

        # load runtime cfg, env cfg, agent cfg, policy model, cv model
        policy_run_dir: str = node_config['policy_run_dir']
        if not os.path.exists(policy_run_dir):
            raise FileNotFoundError(f"Policy run directory not found at: {policy_run_dir}")
        with open(os.path.join(policy_run_dir, 'params','env.yaml'), 'r') as f:
            self.env_cfg = yaml.safe_load(f)
        with open(os.path.join(policy_run_dir, 'params','agent.yaml'), 'r') as f:
            self.agent_cfg = yaml.safe_load(f)
        with open(os.path.join(policy_run_dir, 'exported','runtime_cfg.yaml'), 'r') as f:
            self.runtime_cfg = yaml.safe_load(f)

        self.dt: float = self.runtime_cfg['dt']  # Default to 60 Hz if not specified
        self.ema: float = self.runtime_cfg['action_EMA']

        self.init_left_joint_pose: float = self.env_cfg['experiment_settings']['setting_1_urdf']['left_joint_pose']
        self.init_right_joint_pose: float = self.env_cfg['experiment_settings']['setting_1_urdf']['right_joint_pose']
        self._n_arm: int = self.env_cfg['armDof']
        self._n_hand: int = self.env_cfg['handDof']
        self._action_per_chain: int = self.env_cfg['action_per_chain']
        self._action_num: int = self.env_cfg['action_num']
        self._action_scale: float = self.env_cfg['action_scale']
        self._obs_unstacked_space: int = self.env_cfg['obs_unstacked_space']

        self.left_joint_pose_soft_lower = np.array(self.runtime_cfg['robot_joint_limits_dict']['left_joint_pose_soft_lower'])
        self.right_joint_pose_soft_lower = np.array(self.runtime_cfg['robot_joint_limits_dict']['right_joint_pose_soft_lower'])
        self.left_joint_pose_soft_upper = np.array(self.runtime_cfg['robot_joint_limits_dict']['left_joint_pose_soft_upper'])
        self.right_joint_pose_soft_upper = np.array(self.runtime_cfg['robot_joint_limits_dict']['right_joint_pose_soft_upper'])

        # 3. Model Loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_policy = torch.jit.load(os.path.join(policy_run_dir, 'exported','policy.pt'), map_location=self.device).eval()
        self.actor_components = self.env_cfg['actors']
        self._pinocchio_init()

        # 4. State Management for Actions and Targets
        self.joint_poses: np.ndarray = np.zeros(self._action_num, dtype=np.float32)   # [dof]
        self.joint_vels: np.ndarray = np.zeros(self._action_num, dtype=np.float32)    # [dof]
        self.targets: np.ndarray = np.zeros(self._action_num, dtype=np.float32)  # [dof]
        
        ## joint states scaled to [-1, 1] for policy input
        self.left_joint_pos_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.right_joint_pos_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.left_joint_vel_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.right_joint_vel_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)

        self.prev_actions: np.ndarray = np.zeros(self._action_num, dtype=np.float32)   # [env, dof]
        self.prev_obs: torch.Tensor = torch.zeros((1, self._obs_unstacked_space), device=self.device).float()   # [env, obs_dim]

        # 5. Communication & Timers
        self.target_pub = self.create_publisher(JointState, '/target_joint_states', HIGH_PERF_QOS)
        ## Observation mutex group
        self.create_subscription(JointState, '/joint_states', self._sub_joint_state_cb, HIGH_PERF_QOS, callback_group=self.observation_grp)
        self.create_subscription(Detection3D, '/object_detection', self._sub_object_detection_cb, HIGH_PERF_QOS, callback_group=self.observation_grp)
        # self.fk_timer = self.create_timer(1.0 / node_config['fk_rate'], self._pinocchio_forward_kinematics, callback_group=self.observation_grp)  # FK update
        
        self.policy_timer = self.create_timer(self.dt, self._policy_update_loop, callback_group=self.inference_grp)

        self.get_logger().info("RLPolicyNode initialized.")
        self._set_pre_grasp_state()
        time.sleep(2.0)  # Allow time to reach pre-grasp state
        self.get_logger().info("RLPolicyNode is in pre-grasp state.")

    def _pinocchio_init(self):
        # left and right hand share the same urdf but with different pose in world frame
        self.pin_model = pin.buildModelFromUrdf(URDF_PATH)
        self.pin_data = self.pin_model.createData()
        assert self.pin_model.nq == self._action_per_chain, f"Pinocchio model nq ({self.pin_model.nq}) does not match expected single chain action num ({self._action_per_chain})"

        finger_tip_links: list[str] = self.env_cfg["hand_link_dict"]["finger_tips"]
        hand_base_link: list[str] = self.env_cfg["hand_link_dict"]["hand_base"]
        self.fingertip_ids: list[int] = [self.pin_model.getFrameId(n) for n in finger_tip_links]
        self.hand_base_id: list[int] = [self.pin_model.getFrameId(n) for n in hand_base_link]

        # self.l_hand_base_pos: np.ndarray = np.zeros((1, len(self.hand_base_id), 3))     # 1, 1, 3
        # self.r_hand_base_pos: np.ndarray = np.zeros((1, len(self.hand_base_id), 3))
        # self.l_fingertips_pos: np.ndarray = np.zeros((1, len(self.fingertip_ids), 3))   # 1, 3, 3
        # self.r_fingertips_pos: np.ndarray = np.zeros((1, len(self.fingertip_ids), 3))

    def _pinocchio_forward_kinematics(self, q) -> tuple[np.ndarray, np.ndarray]:
        """update fingertip and hand base positions with urdf model"""
        q = q.astype(np.float64)
        assert q.shape == (self.pin_model.nq,), f"Pin model expects joint_pos shape ({self.pin_model.nq}), got {q.shape}"
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        hand_base_pos = np.array([self.pin_data.oMf[id].translation for id in self.hand_base_id], dtype=np.float32)   # [1,3]
        fingertips_pos = np.array([self.pin_data.oMf[id].translation for id in self.fingertip_ids], dtype=np.float32) # [3,3]
        
        return hand_base_pos, fingertips_pos

    def _set_pre_grasp_state(self):
        """Set initial joint pose from config"""
        
        with self.lock:
            self.joint_poses[:] = np.array(list(self.init_left_joint_pose.values()) + list(self.init_right_joint_pose.values()))
            self.targets[:] = self.joint_poses.copy()  # [dof]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = self.targets.tolist()  # [dof]
        self.target_pub.publish(msg)


    def _sub_joint_state_cb(self, msg: JointState):
        """update scaled joint states in [-1,1]"""

        with self.lock:
            self.joint_poses[:] = np.array(msg.position, dtype=np.float32)
            self.joint_vels[:] = np.array(msg.velocity, dtype=np.float32)

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

    def _sub_object_detection_cb(self, msg: Detection3D):
        with self.lock:
            Detection3D
            self.object_detection: np.ndarray = None    # TODO: CV model

    def _policy_update_loop(self):
        """Control loop: Fuses states and integrates delta actions"""
        if self.object_detection is None or self.targets is None:
            return

        # waiting for assymmetric actor critic HAND policy
        with self.lock:
            cur_l_joint_poses = self.joint_poses[:self._action_per_chain].copy()
            cur_r_joint_poses = self.joint_poses[self._action_per_chain:].copy()
            cur_l_joint_vels = self.joint_vels[:self._action_per_chain].copy()
            cur_r_joint_vels = self.joint_vels[self._action_per_chain:].copy()

            object_detection = self.object_detection.copy()
            cur_targets = self.targets.copy()


        # has been converted to float32
        l_hand_base_pos, l_fingertips_pos = self._pinocchio_forward_kinematics(cur_l_joint_poses)   # update fingertip and hand base positions for observation
        r_hand_base_pos, r_fingertips_pos = self._pinocchio_forward_kinematics(cur_r_joint_poses)

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
            "leftTargets": cur_targets[:self._action_per_chain],   # [action_per_chain,]
            "rightTargets": cur_targets[self._action_per_chain:],  # [action_per_chain,]
            ## end effectors - Cartesian space
            "leftFingerTipsPos": l_fingertips_pos.flatten(),    # [_n_hand*3]
            "rightFingerTipsPos": r_fingertips_pos.flatten(),   # [_n_hand*3]
            "leftHandBasePos": l_hand_base_pos.flatten(),  # [3,]
            "rightHandBasePos": r_hand_base_pos.flatten(), # [3,]

            # exteroception (priledged information)
            # TODO: CV model
            "bottleBodyPos": object_detection[...], # [3,]
            "bottleBodyRot": object_detection[...],
            "bottleLidPos": object_detection[...],  # [3,]
            "bottleLidRot": object_detection[...],
            "bottleBjointPos": object_detection[...],   # [1,]
            "deltaBottleBjointPos": object_detection[...],  # [1,]
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

        with torch.inference_mode():
            policy_action = self.action_policy(observations)

        raw_actions = policy_action.cpu().numpy().squeeze()  # [dof]

        (
            self.targets[:], 
            self.prev_actions[:],
        ) = compute_targets(
            dt=self.dt,
            actions=raw_actions,
            prev_actions=self.prev_actions,
            action_EMA=self.ema,
            actions_scale=self._action_scale,
            
            left_dof_targets=cur_targets[:self._action_per_chain],
            right_dof_targets=cur_targets[self._action_per_chain:],
            robot_joint_indices_dict=self.runtime_cfg['robot_joint_indices_dict'],
            robot_action_scale_dict=self.runtime_cfg['robot_action_scale_dict'],
            robot_joint_limits_dict=self.runtime_cfg['robot_joint_limits_dict'],
        )   # [dof]

        # 4. Publish to Driver
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = self.targets[:].tolist()  # [dof]
        self.target_pub.publish(msg)


    def __del__(self):
        # Ensure hardware pipeline is released on shutdown
        if hasattr(self, 'rs_pipeline'):
            self.rs_pipeline.stop()


# utils from HAND
# transfer to numpy
def scale(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """scale to [-1, 1]"""
    return 2.0 * (target - lower) / (upper - lower) - 1.0

def compute_targets(
    dt: float,
    actions: np.ndarray,
    prev_actions: np.ndarray,
    action_EMA: float,
    actions_scale: float,
    
    left_dof_targets: np.ndarray,
    right_dof_targets: np.ndarray,
    robot_joint_indices_dict: Dict[str, list[int]],
    robot_action_scale_dict: Dict[str, float],
    robot_joint_limits_dict: Dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    return: targets, actions
    """

    # Action scaling and smoothing
    # actions: (left_arm+right_arm , )
    actions = np.clip(actions, -1.0, 1.0) * action_EMA + prev_actions * (1.0 - action_EMA)   # EMA smoothing

    # Action to articulation target
    ## left_arm
    left_actions = actions[robot_joint_indices_dict["left"]]
    left_dof_targets = np.clip(
        a=left_dof_targets + left_actions * dt * robot_action_scale_dict["left_joint_vel_action"] * actions_scale, 
        a_min=robot_joint_limits_dict["left_joint_pose_soft_lower"], 
        a_max=robot_joint_limits_dict["left_joint_pose_soft_upper"]
    )
    ## right_arm
    right_actions = actions[robot_joint_indices_dict["right"]]
    right_dof_targets = np.clip(
        a=right_dof_targets + right_actions * dt * robot_action_scale_dict["right_joint_vel_action"] * actions_scale, 
        a_min=robot_joint_limits_dict["right_joint_pose_soft_lower"], 
        a_max=robot_joint_limits_dict["right_joint_pose_soft_upper"]
    )

    targets = np.concatenate([left_dof_targets, right_dof_targets]) # [dof]
    return targets, actions


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