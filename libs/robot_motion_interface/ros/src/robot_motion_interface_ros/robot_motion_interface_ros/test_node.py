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

HOME_Q = np.array([
    0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

PRE_GRASP_Q = np.array([
    # --- Left Joint Pose (19 dims) ---
    -0.6981317007977318, 0.9075712110370514, 0.14835298641951802, -1.8657569703819383, 
    1.3788101090755203, 1.6126842288427605, 2.0943951023931953,  # panda 1-7
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,            # F1M1-M4
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,            # F2M1-M4
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,            # F3M1-M4
    # --- Right Joint Pose (19 dims) ---
    -0.19198621771937624, 0.32986722862692824, 0.07853981633974483, -1.8936822384138476, 
    -0.059341194567807204, 2.1415189921970423, 0.8000065692366409, # panda 1-7
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,            # F1M1-M4
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,            # F2M1-M4
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988             # F3M1-M4
], dtype=np.float32)

MID_Q = HOME_Q.copy()
MID_Q[:19] = PRE_GRASP_Q[:19]
TRAJ1 = np.linspace(HOME_Q, MID_Q, T_STEPS // 2)
TRAJ2 = np.linspace(MID_Q, PRE_GRASP_Q, T_STEPS // 2)
TRAJ = np.concatenate([TRAJ1, TRAJ2], axis=0)  # [T_STEPS, 38]

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
        
        # joint names
        l_joint_names = cfg['left_panda_joint_names'] + cfg['left_tesollo_joint_names']
        l_joint_names = ['l_' + name for name in l_joint_names]
        r_joint_names = cfg['right_panda_joint_names'] + cfg['right_tesollo_joint_names']
        r_joint_names = ['r_' + name for name in r_joint_names]
        self.joint_names = l_joint_names + r_joint_names
        
        # target joint state publisher
        self.target_pub = self.create_publisher(JointState, '/target_joint_states', HIGH_PERF_QOS)
        
        # timer for publishing trajectory
        self.traj_index = 0
        self.timer = self.create_timer(1.0 / 60.0, self.timer_callback)
        
        self.get_logger().info(f"Test Node initialized with {len(self.joint_names)} joints. Ready to publish.")

    def timer_callback(self):
        if self.traj_index >= len(TRAJ):
            self.get_logger().info("Trajectory loop ended.")
            self.timer.cancel()
            self.destroy_node()
            rclpy.shutdown()
            return

        target_q = TRAJ[self.traj_index]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
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