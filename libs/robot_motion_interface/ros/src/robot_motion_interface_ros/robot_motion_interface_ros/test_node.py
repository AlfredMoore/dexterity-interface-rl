import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
import numpy as np
import yaml
from pathlib import Path
import importlib.util

# --- QoS Config: low latency (Best Effort) ---
HIGH_PERF_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent

# Simulated static traj (NumPy ndarray)
# dim：7(L_arm)+12(L_hand)+7(R_arm)+12(R_hand) = 38
FREQ = 60  # Hz
DURATION = 10  # seconds
T_STEPS = int(FREQ * DURATION) # total time steps
TRAJ = np.zeros((T_STEPS, 38))
# TODO: generate a more meaningful trajectory
time_steps = np.linspace(0, 2 * np.pi, T_STEPS)
TRAJ = ...

class BimanualTrajTestNode(Node):
    def __init__(self):
        super().__init__('bimanual_traj_test_node')
        
        # read config to get joint names (ensure order consistency with driver_node)
        default_config = str(RMI_ROOT / 'config' / 'rl_bimanual_driver_config.yaml')
        self.declare_parameter('config_path', default_config)
        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        
        self.get_logger().info(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # 获取关节名称列表 (确保与 driver_node 订阅的顺序完全一致)
        # 这里的逻辑通过遍历配置字典提取名称
        l_arm_names = cfg['robot_motion_interface']['panda_left']['joint_names']
        l_hand_names = cfg['robot_motion_interface']['tesollo_left']['joint_names']
        r_arm_names = cfg['robot_motion_interface']['panda_right']['joint_names']
        r_hand_names = cfg['robot_motion_interface']['tesollo_right']['joint_names']
        
        self.joint_names = l_arm_names + l_hand_names + r_arm_names + r_hand_names
        
        # 3. 发布者配置
        self.target_pub = self.create_publisher(JointState, '/target_joint_states', HIGH_PERF_QOS)
        
        # 4. 定时器 (60Hz)
        self.traj_index = 0
        self.timer = self.create_timer(1.0 / 60.0, self.timer_callback)
        
        self.get_logger().info(f"Test Node initialized with {len(self.joint_names)} joints. Ready to publish.")

    def timer_callback(self):
        # 循环播放轨迹
        if self.traj_index >= len(TRAJ):
            self.traj_index = 0
            self.get_logger().info("Trajectory loop restarted.")

        # 获取当前时刻的轨迹行
        target_q = TRAJ[self.traj_index]

        # 构建 JointState 消息
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        # 必须转换为 list 以兼容 ROS 2 Python API
        msg.position = target_q.tolist() 
        
        self.target_pub.publish(msg)
        self.traj_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = BimanualTrajTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()