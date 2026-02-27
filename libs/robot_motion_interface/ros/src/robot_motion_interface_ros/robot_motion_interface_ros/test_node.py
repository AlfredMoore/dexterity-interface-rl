import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import yaml
from pathlib import Path
import importlib.util

from robot_motion_interface.utils.kinematics import CuRoboBimanualMotionPlanner

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

# dim: 7(L_arm) + 12(L_hand) + 7(R_arm) + 12(R_hand) = 38
HOME_Q = np.array([
    0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

PRE_GRASP_Q = np.array([
    # --- Left Joint Pose (19 dims) ---
    -0.6981317007977318, 0.9075712110370514, 0.14835298641951802, -1.8657569703819383,
    1.3788101090755203, 1.6126842288427605, 2.0943951023931953,   # panda 1-7
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,             # F1M1-M4
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,             # F2M1-M4
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,             # F3M1-M4
    # --- Right Joint Pose (19 dims) ---
    -0.19198621771937624, 0.32986722862692824, 0.07853981633974483, -1.8936822384138476,
    -0.059341194567807204, 2.1415189921970423, 0.8000065692366409, # panda 1-7
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,             # F1M1-M4
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,             # F2M1-M4
    0.0, 0.0, 0.7853981633974483, 0.5235987755982988,             # F3M1-M4
], dtype=np.float32)


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

        # Generate collision-free trajectory using cuRobo
        result = self._build_trajectory()
        if result is None:
            self.get_logger().error("Trajectory planning failed — node will not publish.")
            return

        self.traj, traj_dt = result
        self.traj_index = 0
        self.timer = self.create_timer(traj_dt, self.timer_callback)

        self.get_logger().info(
            f"Test Node ready. Trajectory: {len(self.traj)} steps "
            f"(dt={traj_dt:.3f}s, ~{len(self.traj) * traj_dt:.1f}s total). "
            f"Publishing to /target_joint_states."
        )

    def _build_trajectory(self) -> tuple[np.ndarray, float] | None:
        """
        1. Initialise cuRobo planner.
        2. Check PRE_GRASP_Q for world and self collision.
        3. If collision-free, plan HOME_Q → PRE_GRASP_Q and return (traj, dt).
        4. If in collision or planning fails, log the reason and return None.
        """
        self.get_logger().info(
            "Initializing cuRobo motion planner (warmup may take ~30 s)..."
        )
        planner = CuRoboBimanualMotionPlanner()

        # --- Collision check ---
        self.get_logger().info("Checking PRE_GRASP_Q for collision...")
        world_col, self_col = planner.is_in_collision(PRE_GRASP_Q)
        if world_col or self_col:
            self.get_logger().error(
                f"PRE_GRASP_Q is in collision "
                f"(world={world_col}, self={self_col}). Aborting."
            )
            return None
        self.get_logger().info("PRE_GRASP_Q is collision-free.")

        # --- Trajectory planning ---
        self.get_logger().info(
            "Planning collision-free trajectory HOME_Q → PRE_GRASP_Q..."
        )
        traj, ok, status = planner.plan_to_joint(HOME_Q, PRE_GRASP_Q)
        if not ok:
            self.get_logger().error(f"cuRobo planning failed: {status}. Aborting.")
            return None

        self.get_logger().info(
            f"Trajectory planned: {traj.shape[0]} steps "
            f"(dt={planner._interpolation_dt:.3f}s, "
            f"~{traj.shape[0] * planner._interpolation_dt:.1f}s total)."
        )
        return traj, planner._interpolation_dt

    def timer_callback(self):
        if self.traj_index >= len(self.traj):
            self.get_logger().info("Trajectory complete.")
            self.timer.cancel()
            self.destroy_node()
            rclpy.shutdown()
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.traj[self.traj_index].tolist()

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
