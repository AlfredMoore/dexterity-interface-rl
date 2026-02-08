import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection3D
import torch
import numpy as np
import pyrealsense2 as rs
import cv2
import yaml
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# --- QoS Config: low latency (Best Effort) ---

HIGH_PERF_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

class CVPerceptionNode(Node):
    def __init__(self):
        super().__init__('cv_perception_node')
        
        self.declare_parameter('config_path', Parameter.Type.STRING)
        config_path: str = self.get_parameter('config_path').value
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.get_logger().info(f"Loaded config from: {config_path}")

        self.rs_fps = config['rs_fps']
        
        # 2. RealSense Initialization (Mirroring pipeline/rs_config style)
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config() # Changed from self.config
        
        self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.rs_fps)
        self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.rs_fps)
        
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        self.rs_align = rs.align(rs.stream.color) # Alignment: Depth -> Color
        self.get_logger().info("RealSense Pipeline and rs_config initialized.")
        
        # 2. model loading (GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_model = torch.jit.load(self.get_parameter('cv_model_path').value, map_location=self.device).eval()
        
        # 3. publisher (publish object pose)
        self.object_detection_pub = self.create_publisher(Detection3D, '/object_detection', HIGH_PERF_QOS)
        
        # vision timer
        self.create_timer(1.0/self.get_parameter('rs_fps').value, self.cv_update_loop)

    def cv_update_loop(self):
        try:
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=10)
            color_frame = frames.get_color_frame()
            if not color_frame: return

            img = np.asanyarray(color_frame.get_data())
            img_tensor = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float() / 255.0
            
            with torch.inference_mode():
                # TODO: we need object pose and size
                pose_tensor = self.cv_model(img_tensor.unsqueeze(0))
                pose_data = pose_tensor.squeeze().cpu().numpy()

            # pub pose
            msg = Detection3D()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_optical_frame"
            msg.bbox.center.position.x, msg.bbox.center.position.y, msg.bbox.center.position.z = pose_data[:3]
            msg.bbox.size.x = float(pose_data[7]) # width
            msg.bbox.size.y = float(pose_data[8]) # height
            msg.bbox.size.z = ... # TODO: fixed depth size for simplicity
            
            self.object_detection_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"CV Error: {e}")

    def __del__(self):
        self.pipeline.stop()

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CVPerceptionNode())
    rclpy.shutdown()