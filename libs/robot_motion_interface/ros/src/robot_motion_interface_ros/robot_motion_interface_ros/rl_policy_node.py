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
        
        # 1. Parameters
        self.declare_parameter('config_path', Parameter.Type.STRING)
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
        self._pinocchio_init()

        # 4. State Management for Actions and Targets
        self.joint_poses: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)   # [dof]
        self.joint_vels: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)    # [dof]
        self.targets: np.ndarray = np.zeros(self._action_num).float()  # [dof]
        
        ## joint states scaled to [-1, 1] for policy input
        self.left_joint_pos_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.right_joint_pos_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.left_joint_vel_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.right_joint_vel_scaled: np.ndarray = np.zeros(self._action_per_chain, dtype=np.float32)
        self.prev_actions: np.ndarray = np.zeros(self._action_num, dtype=np.float32)   # [env, dof]
        
        self.prev_obs: torch.Tensor = torch.zeros((1, self._obs_unstacked_space), device=self.device).float()   # [env, obs_dim]

        # 5. Communication & Timers
        self.target_pub = self.create_publisher(JointState, '/target_joint_states', HIGH_PERF_QOS)
        self.create_subscription(JointState, '/joint_states', self._sub_joint_state_cb, HIGH_PERF_QOS)
        self.create_subscription(Detection3D, '/object_detection', self._sub_object_detection_cb, HIGH_PERF_QOS)
        
        self.policy_timer = self.create_timer(self.dt, self._policy_update_loop)
        self.fk_timer = self.create_timer(1.0 / node_config['fk_rate'], self._pinocchio_forward_kinematics)  # FK update

        self.get_logger().info("RLPolicyNode initialized.")
        self._set_pre_grasp_state()
        time.sleep(2.0)  # Allow time to reach pre-grasp state
        self.get_logger().info("RLPolicyNode is in pre-grasp state.")

    def _pinocchio_init(self):
        # left and right hand share the same urdf but with different pose in world frame
        self.pin_model = pin.buildModelFromUrdf(URDF_PATH)
        self.pin_data = self.pin_model.createData()
        assert self.pin_model.nq == self._action_num, f"Pinocchio model nq ({self.pin_model.nq}) does not match expected action num ({self._action_num})"

        finger_tip_links: list[str] = self.env_cfg["hand_link_dict"]["finger_tips"]
        hand_base_link: list[str] = self.env_cfg["hand_link_dict"]["hand_base"]
        self.fingertip_ids: list[int] = [self.pin_model.getFrameId(n) for n in finger_tip_links]
        self.hand_base_id: list[int] = [self.pin_model.getFrameId(n) for n in hand_base_link]

        self.l_hand_base_pos: np.ndarray = np.zeros((1, len(self.hand_base_id), 3))     # 1, 1, 3
        self.r_hand_base_pos: np.ndarray = np.zeros((1, len(self.hand_base_id), 3))
        self.l_fingertips_pos: np.ndarray = np.zeros((1, len(self.fingertip_ids), 3))   # 1, 3, 3
        self.r_fingertips_pos: np.ndarray = np.zeros((1, len(self.fingertip_ids), 3))

    def _pinocchio_forward_kinematics(self, joint_pos) -> tuple[np.ndarray, np.ndarray]:
        """update fingertip and hand base positions with urdf model"""

        assert joint_pos.shape == (self.pin_model.nq), f"Pin model expects joint_pos shape ({self.pin_model.nq}), got {joint_pos.shape}"
        pin.forwardKinematics(self.pin_model, self.pin_data, joint_pos)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        hand_base_pos = np.array([self.pin_data.oMf[id].translation for id in self.hand_base_id])   # [1,3]
        fingertips_pos = np.array([self.pin_data.oMf[id].translation for id in self.fingertip_ids]) # [3,3]
        
        return hand_base_pos, fingertips_pos

    def _set_pre_grasp_state(self):
        """Set initial joint pose from config"""
        
        with self.lock:
            self.joint_poses[:] = np.array(list(self.init_left_joint_pose.values()) + list(self.init_right_joint_pose.values()))
            self.targets[:] = self.joint_poses.clone()  # [dof]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = self.targets.tolist()  # [dof]
        self.target_pub.publish(msg)


    def _sub_joint_state_cb(self, msg: JointState):
        """update scaled joint states in [-1,1]"""

        with self.lock:
            self.joint_poses[:] = np.array(msg.position, dtype=np.float32)
            self.joint_vels[:] = np.array(msg.velocity, dtype=np.float32)

            self.left_joint_pos_scaled[:] = scale(
                target=self.joint_poses[:, :self._action_per_chain],
                lower=self.left_joint_pose_soft_lower,
                upper=self.left_joint_pose_soft_upper,
            )
            self.right_joint_pos_scaled[:] = scale(
                self.joint_poses[:, self._action_per_chain:],
                lower=self.right_joint_pose_soft_lower,
                upper=self.right_joint_pose_soft_upper,
            )
            self.left_joint_vel_scaled = self.joint_vels[:, :self._action_per_chain] / self.runtime_cfg['robot_joint_limits_dict']['left_joint_vel']
            self.right_joint_vel_scaled = self.joint_vels[:, self._action_per_chain:] / self.runtime_cfg['robot_joint_limits_dict']['right_joint_vel']

    def _sub_object_detection_cb(self, msg: Detection3D):
        with self.lock:
            Detection3D
            self.object_detection = torch.tensor(...)

    def _policy_update_loop(self):
        """Control loop: Fuses states and integrates delta actions"""
        if self.object_detection is None or self.targets is None:
            return

        with self.lock:
            # waiting for assymmetric actor critic HAND policy
            l_hand_base_pos, l_fingertips_pos = self._pinocchio_forward_kinematics(self.joint_poses[:self._action_per_chain])   # update fingertip and hand base positions for observation
            r_hand_base_pos, r_fingertips_pos = self._pinocchio_forward_kinematics(self.joint_poses[self._action_per_chain:])

            proprio_obs = ...   # ndarray: joint states, ee pose
            privileged_obs = ...    # tensor: object pose
            cur_obs = ... # current obs, tensor
            prev_obs = self.prev_obs.clone()

            stacked_obs = torch.cat(
                (
                    cur_obs,
                    prev_obs,   # Previous observations
                ),
            dim=-1,
            )

            self.prev_obs = cur_obs.clone()

        # TODO: Assymmetric Actor obseervation space, align with the trained model
        observations = {"policy": torch.clamp(stacked_obs, -100.0, 100.0)}

        with torch.inference_mode():
            policy_action = self.action_policy(observations)

        action = policy_action.cpu().numpy().squeeze()  # [dof]
        cur_targets = self.targets.clone()

        (
            self.targets[:], 
            actions
        ) = compute_targets(
            dt=self.dt,
            actions=action,
            prev_actions=self.prev_actions,
            action_EMA=self.ema,
            actions_scale=self._action_scale,
            
            left_dof_targets=cur_targets[:, :self._action_per_chain],
            right_dof_targets=cur_targets[:, self._action_per_chain:],
            robot_joint_indices_dict=self.runtime_cfg['robot_joint_indices_dict'],
            robot_action_scale_dict=self.runtime_cfg['robot_action_scale_dict'],
            robot_joint_limits_dict=self.runtime_cfg['robot_joint_limits_dict'],
        )   # [dof]
        self.prev_actions = actions.clone()

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    return: left_dof_targets, right_dof_targets, actions
    """

    # Action scaling and smoothing
    # actions: (left_arm+right_arm , )
    actions = actions.clamp(-1.0, 1.0) * action_EMA + prev_actions * (1.0 - action_EMA)   # EMA smoothing

    # Action to articulation target
    ## left_arm
    left_actions = actions[:, robot_joint_indices_dict["left"]]
    left_dof_targets = np.clip(
        a=left_dof_targets + left_actions * dt * robot_action_scale_dict["left_joint_vel_action"] * actions_scale, 
        a_min=robot_joint_limits_dict["left_joint_pose_soft_lower"], 
        a_max=robot_joint_limits_dict["left_joint_pose_soft_upper"]
    )
    ## right_arm
    right_actions = actions[:, robot_joint_indices_dict["right"]]
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
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()