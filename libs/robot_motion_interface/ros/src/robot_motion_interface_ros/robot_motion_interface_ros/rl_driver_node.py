import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Empty
from rclpy.parameter import Parameter
import numpy as np
import os
import yaml
from pathlib import Path
import importlib.util
import time


# Customized Interface
from robot_motion_interface.interface import Interface
from robot_motion_interface.panda.panda_interface import PandaInterface
from robot_motion_interface.tesollo.tesollo_interface import TesolloInterface
from robot_motion_interface.bimanual_interface import BimanualInterface

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


class RLDriverNode(Node):
    def __init__(self):
        super().__init__('rl_driver_node')
        
        # --- 1. config ---
        self.declare_parameter('config_path', str(RMI_ROOT/'configs'/'rl_bimanual_driver_config.yaml'))
        config_path = self.get_parameter('config_path').value
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.get_logger().info(f"Loaded config from: {config_path}")

        relative_panda_urdf_path = config["panda_urdf_path"]
        panda_urdf_path = str((RMI_ROOT / relative_panda_urdf_path).resolve())
        panda_home_joint_positions = np.array(config["panda_home_joint_positions"], dtype=float)
        panda_kp = np.array(config["panda_kp"], dtype=float)
        panda_kd = np.array(config["panda_kd"], dtype=float)
        
        tesollo_home_joint_positions = np.array(config["tesollo_home_joint_positions"], dtype=float)
        tesollo_control_loop_frequency = config["tesollo_control_loop_frequency"]
        tesollo_kp = np.array(config["tesollo_kp"], dtype=float)
        tesollo_kd = np.array(config["tesollo_kd"], dtype=float)

        # Optional
        left_panda_hostname = config.get("left_panda_hostname")
        left_panda_joint_names = config.get("left_panda_joint_names", [])
        right_panda_hostname = config.get("right_panda_hostname")
        right_panda_joint_names = config.get("right_panda_joint_names", [])

        left_tesollo_ip = config.get("left_tesollo_ip")
        left_tesollo_port = config.get("left_tesollo_port")
        left_tesollo_joint_names = config.get("left_tesollo_joint_names", [])
        right_tesollo_ip = config.get("right_tesollo_ip")
        right_tesollo_port = config.get("right_tesollo_port")
        right_tesollo_joint_names = config.get("right_tesollo_joint_names", [])
        
        # --- 2. Initialize Interface ---
        self.get_logger().info(f"Initializing RLDriverNode...")
        try:
            self._panda_left = PandaInterface(left_panda_hostname, panda_urdf_path, left_panda_joint_names, 
                panda_home_joint_positions, panda_kp, panda_kd)
            self._tesollo_left = TesolloInterface(left_tesollo_ip, left_tesollo_port, left_tesollo_joint_names, 
                tesollo_home_joint_positions, tesollo_kp,tesollo_kd, tesollo_control_loop_frequency)
            self._panda_right = PandaInterface(right_panda_hostname, panda_urdf_path, right_panda_joint_names, 
                panda_home_joint_positions, panda_kp, panda_kd)
            self._tesollo_right = TesolloInterface(right_tesollo_ip, right_tesollo_port, right_tesollo_joint_names, 
                tesollo_home_joint_positions, tesollo_kp,tesollo_kd, tesollo_control_loop_frequency)
            self._n_panda = len(self._panda_left.joint_names())
            self._n_tesollo = len(self._tesollo_left.joint_names())

            self._tesollo_left.start_loop()
            self._panda_left.start_loop()
            self._tesollo_right.start_loop()
            self._panda_right.start_loop()
            self.get_logger().info(f"Started Interface Loops.")

            self._tesollo_left.home(blocking=False)
            self._panda_left.home(blocking=False)
            self._tesollo_right.home(blocking=False)
            self._panda_right.home(blocking=False)
            time.sleep(2.0)  # wait for homing to finish
            self.get_logger().info("Robot Interface Started & Homed.")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize interface: {e}")
            raise e

        # --- 3. Publisher: joint states publisher (1kHz) ---
        self.state_pub = self.create_publisher(JointState, '/joint_states', HIGH_PERF_QOS)

        # --- 4. Subscriber: target joint states subscriber ---
        self.target_sub = self.create_subscription(
            JointState, 
            '/target_joint_states', 
            self.target_callback, 
            HIGH_PERF_QOS
        )

        # --- 5. timer: 1kHz (0.001s) read states ---
        self._timer = self.create_timer(0.001, self.joint_state_loop)

        # --- etc ---
        l_joint_names = self._panda_left.joint_names() + self._tesollo_left.joint_names()
        l_joint_names = ['l_' + name for name in l_joint_names]
        r_joint_names = self._panda_right.joint_names() + self._tesollo_right.joint_names()
        r_joint_names = ['r_' + name for name in r_joint_names]
        self.joint_names = l_joint_names + r_joint_names


    def joint_state_loop(self) -> np.ndarray:
        """ read joint states and publish to ROS """
        try:
            # joint_state() returns numpy array
            pos_state = []
            vel_state = []

            panda_left_joint_state = self._panda_left.joint_state()
            tesollo_left_joint_state = self._tesollo_left.joint_state()
            panda_right_joint_state = self._panda_right.joint_state()
            tesollo_right_joint_state = self._tesollo_right.joint_state()
            
            pos_state.extend([ 
                panda_left_joint_state[:self._n_panda],
                tesollo_left_joint_state[:self._n_tesollo],
                panda_right_joint_state[:self._n_panda],
                tesollo_right_joint_state[:self._n_tesollo]
            ])
            vel_state.extend([ 
                panda_left_joint_state[self._n_panda:],
                tesollo_left_joint_state[self._n_tesollo:],
                panda_right_joint_state[self._n_panda:],
                tesollo_right_joint_state[self._n_tesollo:]
            ])

            pos_states = np.concatenate(pos_state)
            vel_states = np.concatenate(vel_state)
            joint_names = self.joint_names

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names if joint_names else []
            msg.position = pos_states.tolist()
            msg.velocity = vel_states.tolist()
            self.state_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().warn(f"Error publishing state: {e}")
            raise e
        

    def target_callback(self, msg: JointState):
        """ receive target joint states and execute """
        try:
            target_q = np.array(msg.position, dtype=float)

            self._panda_left.set_joint_positions(target_q[:self._n_panda])
            self._tesollo_left.set_joint_positions(target_q[self._n_panda:self._n_panda+self._n_tesollo])
            offset = self._n_panda + self._n_tesollo
            self._panda_right.set_joint_positions(target_q[offset:offset+self._n_panda])
            self._tesollo_right.set_joint_positions(target_q[offset+self._n_panda:])
            
        except Exception as e:
            self.get_logger().warn(f"Error executing command: {e}")


    def shutdown(self):
        """ Shutdown the interface loops """
        self.get_logger().info("Stopping Interface Loop...")
        self._panda_left.stop_loop()
        self._tesollo_left.stop_loop()
        self._panda_right.stop_loop()
        self._tesollo_right.stop_loop()
        self.get_logger().info("Interface Loop Stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = RLDriverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()