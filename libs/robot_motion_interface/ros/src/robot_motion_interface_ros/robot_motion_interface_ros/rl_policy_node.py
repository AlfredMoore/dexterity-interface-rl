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

class RLPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_policy_node')
        self.lock = threading.Lock()
        
        # 1. Parameters
        self.declare_parameter('config_path', Parameter.Type.STRING)
        config_path: str = self.get_parameter('config_path').value
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.get_logger().info(f"Loaded config from: {config_path}")

        # load runtime cfg, env cfg, agent cfg, policy model, cv model
        policy_run_dir: str = config['policy_run_dir']
        if not os.path.exists(policy_run_dir):
            raise FileNotFoundError(f"Policy run directory not found at: {policy_run_dir}")
        with open(os.path.join(policy_run_dir, 'params','env.yaml'), 'r') as f:
            self.env_cfg = yaml.safe_load(f)
        with open(os.path.join(policy_run_dir, 'params','agent.yaml'), 'r') as f:
            self.agent_cfg = yaml.safe_load(f)
        
        self.runtime_cfg = torch.load(os.path.join(policy_run_dir, 'exported','runtime_cfg.pt'))


        self.dt = self.runtime_cfg['dt']  # Default to 60 Hz if not specified
        self.ema = self.runtime_cfg['action_EMA']

        self.init_left_joint_pose = self.env_cfg['experiment_settings']['setting_1_urdf']['left_joint_pose']
        self.init_right_joint_pose = self.env_cfg['experiment_settings']['setting_1_urdf']['right_joint_pose']
        self._n_arm = self.env_cfg['armDof']
        self._n_hand = self.env_cfg['handDof']
        self._action_per_chain = self.env_cfg['action_per_chain']
        self._action_num = self.env_cfg['action_num']

        # 3. Model Loading (GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_policy = torch.jit.load(os.path.join(policy_run_dir, 'exported','policy.pt'), map_location=self.device).eval()

        # 4. State Management for Actions and Targets
        self.proprioceptive_states: torch.Tensor = None
        self.exteroceptive_states: torch.Tensor = None
        self.targets: torch.Tensor = torch.zeros((1,self._action_num), device=self.device).float()  # [env, dof]
        self.prev_actions: torch.Tensor = torch.zeros((1,self._action_num), device=self.device).float()   # [env, dof]

        # 5. Communication & Timers
        self.target_pub = self.create_publisher(JointState, '/target_joint_states', HIGH_PERF_QOS)
        self.create_subscription(JointState, '/joint_states', self.sub_joint_state_cb, HIGH_PERF_QOS)
        self.create_subscription(Detection3D, '/object_detection', self.sub_object_detection_cb, HIGH_PERF_QOS)
        
        self.policy_timer = self.create_timer(self.dt, self.policy_update_loop)

        self.get_logger().info("RLPolicyNode initialized.")
        self.set_pre_grasp_state()
        time.sleep(2.0)  # Allow time to reach pre-grasp state
        self.get_logger().info("RLPolicyNode is in pre-grasp state.")


    def set_pre_grasp_state(self):
        """Set initial joint pose from config"""
        joint_poses = list(self.init_left_joint_pose.values()) + list(self.init_right_joint_pose.values())
        self.targets[:] = torch.tensor(joint_poses).float().unsqueeze(0)   # [env, dof]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = joint_poses
        self.target_pub.publish(msg)


    def sub_joint_state_cb(self, msg: JointState):
        with self.lock:
            joint_pose = torch.tensor(msg.position).float().unsqueeze(0)    # num_envs, num_joints
            joint_vel = torch.tensor(msg.velocity).float().unsqueeze(0)  # num_envs, num_joints
            # TODO: save the HAND runtime config then load here
            left_joint_pos_scaled = scale(    
                target=joint_pose[:, :self._action_per_chain],
                lower=self.runtime_cfg['robot_joint_limits_dict']['left_joint_pose_soft_lower'],
                upper=self.runtime_cfg['robot_joint_limits_dict']['left_joint_pose_soft_upper'],
            )            
            left_joint_vel_scaled = joint_vel[:, :self._action_per_chain] / self.runtime_cfg['robot_joint_limits_dict']['left_joint_vel']
            right_joint_pos_scaled = scale(
                joint_pose[:, self._action_per_chain:],
                lower=self.runtime_cfg['robot_joint_limits_dict']['right_joint_pose_soft_lower'],
                upper=self.runtime_cfg['robot_joint_limits_dict']['right_joint_pose_soft_upper'],
            )
            right_joint_vel_scaled = joint_vel[:, self._action_per_chain:] / self.runtime_cfg['robot_joint_limits_dict']['right_joint_vel']

            # TODO: use urdf to get those positions wrt robot base the transfer to world frame or directly world frame
            leftFingerTipsPos = ...     # num_envs, fingers * 3
            rightFingerTipsPos = ...    # num_envs, fingers * 3
            leftHandBasePos = ...      # num_envs, 3
            rightHandBasePos = ...     # num_envs, 3

            proprioceptive_full_obs = {
                # proprioception
                ## Q space
                "leftJointPosScaled": left_joint_pos_scaled,    # num_envs, num_joints
                "rightJointPosScaled": right_joint_pos_scaled,  # num_envs, num_joints
                "leftJointVelScaled": left_joint_vel_scaled,
                "rightJointVelScaled": right_joint_vel_scaled,
                ## targets - Cartesian space
                "leftTargets": self.targets[:, :self._action_per_chain],  # num_envs, num_joints
                "rightTargets": self.targets[:, self._action_per_chain:],  # num_envs, num_joints
                ## end effectors - Cartesian space
                "leftFingerTipsPos": states["leftFingerTipsPos"].reshape(num_envs, -1),  # num_envs, fingers * 3
                "rightFingerTipsPos": states["rightFingerTipsPos"].reshape(num_envs, -1),  # num_envs, fingers * 3
                "leftHandBasePos": states["leftHandBasePos"],  # num_envs, 3
                "rightHandBasePos": states["rightHandBasePos"],  # num_envs, 3
            }
            
            # TODO: Assymmetric Actor obseervation space, align with the trained model
            self.proprioceptive_states = ...

    def sub_object_detection_cb(self, msg: Detection3D):
        with self.lock:
            Detection3D
            self.object_detection = torch.tensor(...)

    def policy_update_loop(self):
        """Control loop: Fuses states and integrates delta actions"""
        if self.proprioceptive_states is None or self.object_detection is None or self.targets is None:
            return

        with self.lock:
            obs = np.concatenate([self.proprioceptive_states, self.object_detection], axis=-1)   # num_envs, obs_dim
            current_targets = self.targets.clone()

        obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            # 1. Generate delta actions
            raw_action = self.action_policy(obs_tensor)

        self.targets[:] = compute_targets(
            dt=self.dt,
            actions=raw_action,
            prev_actions=self.prev_actions,
            action_EMA=self.ema,
            actions_scale=1.0,  # Could be tuned or made adaptive
            
            left_dof_targets=current_targets[:, :self._action_per_chain],
            right_dof_targets=current_targets[:, self._action_per_chain:],
            robot_joint_indices_dict=self.runtime_cfg['robot_joint_indices_dict'],
            robot_action_scale_dict=self.runtime_cfg['robot_action_scale_dict'],
            robot_joint_limits_dict=self.runtime_cfg['robot_joint_limits_dict'],
        )   # [1, dof]

        joint_poses = self.targets.squeeze().cpu().numpy().tolist()  # [dof]

        # 4. Publish to Driver
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = joint_poses
        self.target_pub.publish(msg)


    def __del__(self):
        # Ensure hardware pipeline is released on shutdown
        if hasattr(self, 'rs_pipeline'):
            self.rs_pipeline.stop()


# utils from HAND
@torch.jit.script
def scale(target: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """scale to [-1, 1]"""
    return 2.0 * (target - lower) / (upper - lower) - 1.0


@torch.jit.script
def compute_targets(
    dt: float,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    action_EMA: float,
    actions_scale: float,
    
    left_dof_targets: torch.Tensor,
    right_dof_targets: torch.Tensor,
    robot_joint_indices_dict: Dict[str, list[int]],
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
    left_actions = actions[:, robot_joint_indices_dict["left"]]
    left_dof_targets = torch.clamp(
        left_dof_targets + left_actions * dt * robot_action_scale_dict["left_joint_vel_action"] * actions_scale, 
        min=robot_joint_limits_dict["left_joint_pose_soft_lower"], 
        max=robot_joint_limits_dict["left_joint_pose_soft_upper"]
    )
    
    ## right_arm
    right_actions = actions[:, robot_joint_indices_dict["right"]]
    right_dof_targets = torch.clamp(
        right_dof_targets + right_actions * dt * robot_action_scale_dict["right_joint_vel_action"] * actions_scale, 
        min=robot_joint_limits_dict["right_joint_pose_soft_lower"], 
        max=robot_joint_limits_dict["right_joint_pose_soft_upper"]
    )
    return left_dof_targets, right_dof_targets, actions


def main(args=None):
    rclpy.init(args=args)
    node = RLPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()