"""
One-shot pre-grasp planner node.

Flow:
1. Subscribe /joint_states to get current q_start.
2. Read pre-grasp q_goal from runtime HandEnv.yaml (experiment.pre_grasp).
3. Plan with cuRobo.
4. Execute full trajectory to /target_joint_states.
5. Exit.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
import importlib.util

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
import yaml

from robot_motion_interface.utils.kinematics import CuRoboBimanualMotionPlanner
from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS

JS_QOS = HIGH_PERF_QOS
T_JS_QOS = HIGH_RELIA_QOS

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")

RMI_ROOT = Path(spec.origin).parent.parent.parent
LIBS_ROOT = RMI_ROOT.parent
PROJECT_ROOT = LIBS_ROOT.parent

ROBOT_DESC = LIBS_ROOT / "robot_description"
CONFIGS_CUROBO = ROBOT_DESC / "configs_curobo"

DEFAULT_BIMANUAL_URDF_PATH = str((ROBOT_DESC / "rl/bimanual_panda_tesollo.urdf").resolve())
DEFAULT_CUROBO_ROBOT_CFG_PATH = str((CONFIGS_CUROBO / "robot/bimanual_panda_tesollo.yml").resolve())
DEFAULT_COLLISION_SPHERES_PATH = str((CONFIGS_CUROBO / "robot/spheres/bimanual_panda_tesollo_spheres.yml").resolve())


class PreGraspPlanNode(Node):
    def __init__(self):
        super().__init__("pre_grasp_plan_node")

        self.declare_parameter("driver_cfg_path", str(RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml"))
        self.declare_parameter("hand_env_cfg_path", str(RMI_ROOT / "runtime" / "HandEnv.yaml"))
        self.declare_parameter("infer_rate", 30.0)
        self.declare_parameter("wait_js_timeout_s", 15.0)

        driver_cfg_path = self.get_parameter("driver_cfg_path").get_parameter_value().string_value
        hand_env_cfg_path = self.get_parameter("hand_env_cfg_path").get_parameter_value().string_value
        infer_rate = float(self.get_parameter("infer_rate").value)
        self.wait_js_timeout_s = float(self.get_parameter("wait_js_timeout_s").value)

        with open(driver_cfg_path, "r", encoding="utf-8") as f:
            driver_cfg = yaml.safe_load(f)
        with open(hand_env_cfg_path, "r", encoding="utf-8") as f:
            hand_env_cfg = yaml.safe_load(f)

        l_names = ["left_" + n for n in driver_cfg["left_panda_joint_names"] + driver_cfg["left_tesollo_joint_names"]]
        r_names = ["right_" + n for n in driver_cfg["right_panda_joint_names"] + driver_cfg["right_tesollo_joint_names"]]
        self.joint_names = l_names + r_names

        chain_joint_names = (
            hand_env_cfg["env"]["robot"]["jointNames"]["arm"] + hand_env_cfg["env"]["robot"]["jointNames"]["hand"]
        )
        left_pre_grasp = hand_env_cfg["experiment"]["pre_grasp"]["left_joint_pose"]
        right_pre_grasp = hand_env_cfg["experiment"]["pre_grasp"]["right_joint_pose"]
        q_left = np.array([left_pre_grasp[n] for n in chain_joint_names], dtype=np.float32)
        q_right = np.array([right_pre_grasp[n] for n in chain_joint_names], dtype=np.float32)
        self.q_goal = np.concatenate([q_left, q_right], axis=0)

        self._js_lock = threading.Lock()
        self._q_current: np.ndarray | None = None

        self.get_logger().info("Initializing cuRobo...")
        self._planner = CuRoboBimanualMotionPlanner(
            robot_cfg_path=DEFAULT_CUROBO_ROBOT_CFG_PATH,
            urdf_path=DEFAULT_BIMANUAL_URDF_PATH,
            spheres_path=DEFAULT_COLLISION_SPHERES_PATH,
            left_ee_link="left_delto_base_link",
            right_ee_link="right_delto_base_link",
            device="cuda:0",
            trajopt_dt=0.15,
            trajopt_tsteps=64,
            interpolation_steps=1000,
            num_ik_seeds=50,
            num_trajopt_seeds=32,
            grad_trajopt_iters=800,
            interpolation_dt=1.0 / infer_rate,
            collision_activation_distance=0.05,
        )

        self.target_pub = self.create_publisher(JointState, "/target_joint_states", T_JS_QOS)
        self.create_subscription(JointState, "/joint_states", self._js_callback, JS_QOS)

    def _js_callback(self, msg: JointState) -> None:
        q = np.array(msg.position, dtype=np.float32)
        with self._js_lock:
            self._q_current = q

    def _wait_for_joint_state(self) -> np.ndarray | None:
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < self.wait_js_timeout_s:
            with self._js_lock:
                if self._q_current is not None:
                    return self._q_current.copy()
            time.sleep(0.02)
        return None

    def _execute_traj(self, traj: np.ndarray, dt: float) -> None:
        msg = JointState()
        msg.name = self.joint_names
        deadline = time.perf_counter()
        for q in traj:
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = q.tolist()
            self.target_pub.publish(msg)
            deadline += dt
            sleep_s = deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def run_once(self) -> bool:
        q_start = self._wait_for_joint_state()
        if q_start is None:
            self.get_logger().error("No /joint_states received within timeout.")
            return False

        self.get_logger().info("Checking target pre-grasp for self-collision...")
        if self._planner.self_collision_check(self.q_goal):
            self.get_logger().error("Target pre-grasp is in self-collision.")
            return False

        self.get_logger().info("Planning to pre-grasp...")
        traj, last_tstep, ok = self._planner.plan_to_joint(q_start, self.q_goal)
        if not ok:
            self.get_logger().error("cuRobo planning failed.")
            return False

        traj = traj[: last_tstep + 1]
        dt = self._planner._interpolation_dt
        self.get_logger().info(f"Executing pre-grasp trajectory: steps={traj.shape[0]}, dt={dt:.4f}s")
        self._execute_traj(traj, dt)
        self.get_logger().info("Pre-grasp execution done.")

        npy_path = PROJECT_ROOT / "models" / "robot_next_q.npy"
        np.save(str(npy_path), self.q_goal)
        self.get_logger().info(f"Saved goal q to: {npy_path}")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PreGraspPlanNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        _ = node.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
