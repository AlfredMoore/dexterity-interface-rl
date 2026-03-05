import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import yaml
from pathlib import Path
import importlib.util

from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
JS_QOS = HIGH_PERF_QOS
BBOX_QOS = HIGH_PERF_QOS
T_JS_QOS = HIGH_RELIA_QOS

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT     = Path(spec.origin).parent.parent.parent   # libs/robot_motion_interface/
PROJECT_ROOT = RMI_ROOT.parent.parent                   # dexterity-interface-rl/

TRAJ_DT  = 0.02   # must match traj_sampler interpolation_dt
PAUSE_S  = 5.0    # seconds to hold at pre-grasp before returning

_FWD  = "FWD"
_PAUSE = "PAUSE"
_RET  = "RET"


class PreGraspTestNode(Node):
    def __init__(self):
        super().__init__('pre_grasp_test_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('config_path',
                               str(RMI_ROOT / 'config' / 'rl_bimanual_driver_config.yaml'))
        self.declare_parameter('traj_index', 0)

        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        traj_idx    = self.get_parameter('traj_index').get_parameter_value().integer_value

        # ── Joint names ────────────────────────────────────────────────────────
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        l_names = ['left_' + n for n in cfg['left_panda_joint_names']  + cfg['left_tesollo_joint_names']]
        r_names = ['right_' + n for n in cfg['right_panda_joint_names'] + cfg['right_tesollo_joint_names']]
        self.joint_names = l_names + r_names

        # ── Load trajectories ──────────────────────────────────────────────────
        fwd_path = PROJECT_ROOT / 'models' / f'traj_fwd_{traj_idx}.npy'
        ret_path = PROJECT_ROOT / 'models' / f'traj_ret_{traj_idx}.npy'
        self._traj_fwd = np.load(fwd_path).astype(np.float32)
        self._traj_ret = np.load(ret_path).astype(np.float32)
        self.get_logger().info(
            f"Loaded traj_fwd={self._traj_fwd.shape}  traj_ret={self._traj_ret.shape}"
        )

        # ── State ──────────────────────────────────────────────────────────────
        self._phase = _FWD
        self._idx   = 0
        self._pause_ticks = int(PAUSE_S / TRAJ_DT)

        # ── Publisher + timer ──────────────────────────────────────────────────
        self._pub   = self.create_publisher(JointState, '/target_joint_states', T_JS_QOS)
        self._timer = self.create_timer(TRAJ_DT, self._tick)
        self.get_logger().info("Starting: HOME → PRE_GRASP")

    def _tick(self) -> None:
        if self._phase == _FWD:
            if self._idx < len(self._traj_fwd):
                self._publish(self._traj_fwd[self._idx])
                self._idx += 1
            else:
                self.get_logger().info(f"Pre-grasp reached. Holding {PAUSE_S:.0f} s...")
                self._phase = _PAUSE
                self._idx   = 0

        elif self._phase == _PAUSE:
            self._idx += 1
            if self._idx >= self._pause_ticks:
                self.get_logger().info("Returning: PRE_GRASP → HOME")
                self._phase = _RET
                self._idx   = 0

        elif self._phase == _RET:
            if self._idx < len(self._traj_ret):
                self._publish(self._traj_ret[self._idx])
                self._idx += 1
            else:
                self.get_logger().info("Done.")
                self._timer.cancel()
                rclpy.shutdown()

    def _publish(self, q: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = self.joint_names
        msg.position     = q.tolist()
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PreGraspTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
