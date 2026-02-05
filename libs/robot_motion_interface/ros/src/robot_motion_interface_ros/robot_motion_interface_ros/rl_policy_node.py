import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
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
        config_path = self.get_parameter('config_path').value
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.get_logger().info(f"Loaded config from: {config_path}")

        self.declare_parameter('policy_path', 'policy.pt')
        self.declare_parameter('cv_path', 'cv_model.pt')
        self.dt = 1.0 / config['freq']  # Default to 60 Hz if not specified
        self.ema = config['ema']
        self.rs_fps = config['rs_fps']
        self.init_joint_pose = config['init_joint_pose']

        # 2. RealSense Initialization (Mirroring pipeline/rs_config style)
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config() # Changed from self.config
        
        self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.rs_fps)
        self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.rs_fps)
        
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        self.align = rs.align(rs.stream.color) # Alignment: Depth -> Color
        self.get_logger().info("RealSense Pipeline and rs_config initialized.")

        # 3. Model Loading (GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = torch.jit.load(self.get_parameter('policy_path').value, map_location=self.device).eval()
        self.cv_model = torch.jit.load(self.get_parameter('cv_path').value, map_location=self.device).eval()

        # 4. State Management for Actions and Targets
        self.proprioceptive_states = None
        self.exteroceptive_states = None
        self.current_joint_pose_targets = None
        self.prev_actions = torch.zeros((1, 20), device='cpu')

        # 5. Communication & Timers
        self.target_pub = self.create_publisher(JointState, '/target_joint_states', 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        
        # Dual timers to keep control frequency independent of vision latency
        self.cv_timer = self.create_timer(1.0/self.rs_fps, self.cv_update_loop)
        self.policy_timer = self.create_timer(self.dt, self.policy_update_loop)

        self.get_logger().info("RLPolicyNode initialized.")
        self.pre_grasp_state()
        time.sleep(2.0)  # Allow time to reach pre-grasp state
        self.get_logger().info("RLPolicyNode is in pre-grasp state.")

    def pre_grasp_state(self):
        # TODO: Initialize current_joint_pose_targets from init_joint_pose
        self.current_joint_pose_targets = 
        # TODO: publish to target_pub to set initial pose
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = self.current_joint_pose_targets.squeeze().tolist()
        self.target_pub.publish(msg)

    def joint_cb(self, msg):
        with self.lock:
            self.proprioceptive_states = np.concatenate([msg.position, msg.velocity]).astype(np.float32)
            if self.current_joint_pose_targets is None:
                # Bootstrap from actual measured pose on first callback
                self.current_joint_pose_targets = torch.tensor(msg.position).float().unsqueeze(0)

    def cv_update_loop(self):
        """Fetch, align, and process vision data"""
        try:
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=100)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            if not color_frame: return

            # Convert to GPU tensor for JIT CV model
            img = np.asanyarray(color_frame.get_data())
            img_tensor = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float() / 255.0
            
            with torch.inference_mode():
                # Inference using JIT-loaded SAM2, YOLO, or Pose model
                pose = self.cv_model(img_tensor.unsqueeze(0))
            
            with self.lock:
                self.object_pose = pose.squeeze().cpu().numpy()
        except Exception as e:
            self.get_logger().warn(f"Vision loop error: {e}")

    def policy_update_loop(self):
        """Control loop: Fuses states and integrates delta actions"""
        if self.joint_states is None or self.object_pose is None or self.current_dof_targets is None:
            return

        with self.lock:
            obs = np.concatenate([self.joint_states, self.object_pose])
            current_targets = self.current_dof_targets.clone()

        obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            # 1. Generate delta actions
            raw_action = self.policy(obs_tensor).cpu()
            
            # 2. EMA Smoothing (based on targets.py logic)
            smooth_action = torch.clamp(raw_action, -1.0, 1.0) * self.ema + self.prev_actions * (1.0 - self.ema)
            self.prev_actions = smooth_action.clone()

            # 3. Integrate: new_target = old_target + action * dt
            dt = self.get_parameter('dt').value
            self.current_dof_targets = current_targets + (smooth_action * dt)

        # 4. Publish to Driver
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = self.current_dof_targets.squeeze().tolist()
        self.target_pub.publish(msg)

    def __del__(self):
        # Ensure hardware pipeline is released on shutdown
        if hasattr(self, 'rs_pipeline'):
            self.rs_pipeline.stop()

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