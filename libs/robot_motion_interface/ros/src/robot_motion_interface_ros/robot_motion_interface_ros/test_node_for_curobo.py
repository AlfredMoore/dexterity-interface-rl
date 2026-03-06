"""
Interactive bimanual trajectory test node.

Workflow
--------
1. Subscribes to /joint_states from rl_driver_node → q_start.
2. Loads pre-sampled pre-grasp joint configs from a .pt file.
3. Keyboard loop lets the user browse and select a target config → q_goal.
4. On 'g' / Enter: save target mesh, collision-check q_goal, plan a collision-free
   trajectory with cuRobo, then replay it at the configured dt.
5. On 'h': plan and execute a return trajectory to HOME_Q.

Keyboard commands (type + Enter)
---------------------------------
  n          select next config
  p          select previous config
  <number>   jump to that config index
  g / Enter  plan and execute current selection
  h          plan and execute return to HOME_Q
  q          quit
"""
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import torch
import yaml
import time
from pathlib import Path
import importlib.util

from robot_motion_interface.utils.kinematics import CuRoboBimanualMotionPlanner

# --- QoS ---
from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
JS_QOS = HIGH_PERF_QOS
BBOX_QOS = HIGH_PERF_QOS
T_JS_QOS = HIGH_RELIA_QOS


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
# spec.origin  = .../libs/robot_motion_interface/src/robot_motion_interface/__init__.py
# .parent x3   = .../libs/robot_motion_interface/   (_RMI_ROOT)
# .parent x4   = .../libs/
# .parent x5   = project root

_RMI_ROOT     = Path(spec.origin).parent.parent.parent
_LIBS_ROOT = _RMI_ROOT.parent
_PROJECT_ROOT = _LIBS_ROOT.parent

_ROBOT_DESC = _LIBS_ROOT / "robot_description"
_CONFIGS_CUROBO = _ROBOT_DESC / "configs_curobo"

DEFAULT_BIMANUAL_URDF_PATH     = str((_ROBOT_DESC / "rl/bimanual_panda_tesollo.urdf").resolve())
DEFAULT_CUROBO_ROBOT_CFG_PATH  = str((_CONFIGS_CUROBO / "robot/bimanual_panda_tesollo.yml").resolve())
DEFAULT_COLLISION_SPHERES_PATH = str((_CONFIGS_CUROBO / "robot/spheres/bimanual_panda_tesollo_spheres.yml").resolve())

_IDLE      = "IDLE"
_PLANNING  = "PLANNING"
_EXECUTING = "EXECUTING"


class BimanualTrajTestNode(Node):
    def __init__(self):
        super().__init__('bimanual_traj_test_node')

        # ── Parameters ────────────────────────────────────────────────────────
        default_drive_cfg      = str(_RMI_ROOT / 'config' / 'rl_bimanual_driver_config.yaml')
        default_policy_cfg = str(_RMI_ROOT/'config'/'rl_policy_node_config.yaml')
        default_pregrasp = str(_PROJECT_ROOT / 'models' / 'pre_grasp_q_samples.pt')
        self.declare_parameter('driver_cfg_path', default_drive_cfg)
        self.declare_parameter('policy_cfg_path',    default_policy_cfg)
        self.declare_parameter('pregrasp_path',  default_pregrasp)
        driver_cfg_path   = self.get_parameter('driver_cfg_path').get_parameter_value().string_value
        policy_cfg_path = self.get_parameter('policy_cfg_path').get_parameter_value().string_value
        pregrasp_path = self.get_parameter('pregrasp_path').get_parameter_value().string_value

        self.get_logger().info(f"Config:            {driver_cfg_path}")
        self.get_logger().info(f"policy_cfg_path:   {policy_cfg_path}")
        self.get_logger().info(f"Pregrasp:          {pregrasp_path}")

        with open(driver_cfg_path, 'r') as f:
            driver_cfg = yaml.safe_load(f)

        with open(policy_cfg_path, 'r') as f:
            policy_cfg = yaml.safe_load(f)

        infer_rate = policy_cfg["infer_rate"]

        # ── HOME_Q from driver config (panda + tesollo, both arms) ────────────
        _panda_home   = np.array(driver_cfg['panda_home_joint_positions'],   dtype=np.float32)
        _tesollo_home = np.array(driver_cfg['tesollo_home_joint_positions'], dtype=np.float32)
        self._home_q  = np.concatenate([_panda_home, _tesollo_home, _panda_home, _tesollo_home])

        # ── Joint names (must match driver_node order) ─────────────────────
        l_names = ['left_' + n for n in driver_cfg['left_panda_joint_names']  + driver_cfg['left_tesollo_joint_names']]
        r_names = ['right_' + n for n in driver_cfg['right_panda_joint_names'] + driver_cfg['right_tesollo_joint_names']]
        self.joint_names: list[str] = l_names + r_names   # 38 names

        # ── Load pre-grasp configs ──────────────────────────────────────────
        pregrasp_tensor = torch.load(pregrasp_path, weights_only=True)   # (N, 38)
        self._pregrasp_qs: np.ndarray = pregrasp_tensor.numpy().astype(np.float32)
        self._n_configs = len(self._pregrasp_qs)
        if self._n_configs == 0:
            raise RuntimeError(f"No pre-grasp configs found in {pregrasp_path}")
        self.get_logger().info(f"Loaded {self._n_configs} pre-grasp configs.")

        # ── State ─────────────────────────────────────────────────────────────
        self._lock            = threading.Lock()
        self._state           = _IDLE
        self._q_current: np.ndarray | None = None
        self._selected_idx    = 0
        self._traj: np.ndarray | None = None
        self._traj_index      = 0

        # ── cuRobo planner ────────────────────────────────────────────────────
        self.get_logger().info("Initializing cuRobo (warmup may take ~30 s)...")
        self._planner = CuRoboBimanualMotionPlanner(
            robot_cfg_path              = DEFAULT_CUROBO_ROBOT_CFG_PATH,
            urdf_path                   = DEFAULT_BIMANUAL_URDF_PATH,
            spheres_path                = DEFAULT_COLLISION_SPHERES_PATH,
            left_ee_link                = "left_delto_base_link",
            right_ee_link               = "right_delto_base_link",
            device                      = "cuda:0",
            trajopt_dt                  = 0.15,
            trajopt_tsteps              = 32,
            interpolation_steps         = 1000,
            num_ik_seeds                = 50,
            num_trajopt_seeds           = 32,
            grad_trajopt_iters          = 800,
            interpolation_dt            = 1.0 / infer_rate,
            collision_activation_distance = 0.01,
        )

        # ── ROS pub / sub ─────────────────────────────────────────────────────
        self.target_pub = self.create_publisher(JointState, '/target_joint_states', T_JS_QOS)
        self.create_subscription(JointState, '/joint_states', self._js_callback, JS_QOS)

        # Execution timer (always ticking; no-ops when not in EXECUTING state)
        self._exec_timer = self.create_timer(
            1.0 / infer_rate, self._exec_callback
        )

        # ── Keyboard thread ────────────────────────────────────────────────────
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._kb_thread.start()

        self._print_status()
        self.get_logger().info(
            "Ready. Commands (type + Enter):\n"
            "[n] next  [p] prev  [<number>] jump  [g/Enter] plan+execute  [h] home  [q] quit"
        )

    # ── Joint state subscription ───────────────────────────────────────────────
    def _js_callback(self, msg: JointState) -> None:
        """Extract current joint positions in self.joint_names order."""
        q = np.array(msg.position, dtype=np.float32)
        with self._lock:
            self._q_current = q

    # ── Execution timer callback ───────────────────────────────────────────────
    def _exec_callback(self) -> None:
        with self._lock:
            if self._state != _EXECUTING or self._traj is None:
                return
            if self._traj_index >= len(self._traj):
                self.get_logger().info("Trajectory complete. Back to IDLE.")
                self._traj  = None
                self._state = _IDLE
                return
            q = self._traj[self._traj_index]
            self._traj_index += 1
            self.get_logger().info(f"TRAJ IDX: {self._traj_index}")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = self.joint_names
        msg.position     = q.tolist()
        self.target_pub.publish(msg)

    # ── Keyboard loop (blocking, runs in daemon thread) ────────────────────────
    def _keyboard_loop(self) -> None:
        while rclpy.ok():
            try:
                prompt = (
                    f"[{self._selected_idx}/{self._n_configs - 1}]> \n"
                    "Commands (type + Enter):\n"
                    "[n] next  [p] prev  [<number>] jump  [g/Enter] plan+execute  [h] home  [q] quit\n"
                )
                raw = input(prompt).strip()
            except EOFError:
                break

            if raw == 'q':
                rclpy.shutdown()
                break
            elif raw == 'n':
                with self._lock:
                    self._selected_idx = (self._selected_idx + 1) % self._n_configs
                self._print_status()
            elif raw == 'p':
                with self._lock:
                    self._selected_idx = (self._selected_idx - 1) % self._n_configs
                self._print_status()
            elif raw.isdigit():
                idx = int(raw) % self._n_configs
                with self._lock:
                    self._selected_idx = idx
                self._print_status()
            elif raw in ('g', ''):
                self._trigger_plan()
            elif raw == 'h':
                self._trigger_home()
            else:
                self.get_logger().warn(f"Unknown command: '{raw}'")
            
            time.sleep(0.01)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _print_status(self) -> None:
        with self._lock:
            idx = self._selected_idx
        q = self._pregrasp_qs[idx]
        self.get_logger().info(
            f"\nSelected [{idx}/{self._n_configs - 1}]  "
            f"\nleft_arm={[f"{x:.3f}" for x in q[:7]]}  "
            f"\nright_arm={[f"{x:.3f}" for x in q[19:26]]}"
        )

    def _trigger_plan(self) -> None:
        with self._lock:
            if self._state == _EXECUTING:
                self.get_logger().warn("Already executing — wait for completion.")
                return
            if self._state == _PLANNING:
                self.get_logger().warn("Already planning — please wait.")
                return
            q_start = self._q_current
            q_goal  = self._pregrasp_qs[self._selected_idx].copy()
            self._state = _PLANNING

        if q_start is None:
            self.get_logger().error("No /joint_states received yet. Cannot plan.")
            with self._lock:
                self._state = _IDLE
            return

        threading.Thread(
            target=self._plan_and_execute, args=(q_start, q_goal), daemon=True
        ).start()

    def _trigger_home(self) -> None:
        with self._lock:
            if self._state == _EXECUTING:
                self.get_logger().warn("Already executing — wait for completion.")
                return
            if self._state == _PLANNING:
                self.get_logger().warn("Already planning — please wait.")
                return
            q_start = self._q_current
            self._state = _PLANNING

        if q_start is None:
            self.get_logger().error("No /joint_states received yet. Cannot plan.")
            with self._lock:
                self._state = _IDLE
            return

        self.get_logger().info("Planning return to HOME_Q...")
        threading.Thread(
            target=self._plan_and_execute, args=(q_start, self._home_q.copy()), daemon=True
        ).start()

    def _plan_and_execute(self, q_start: np.ndarray, q_goal: np.ndarray) -> None:
        # 1. Save target mesh
        stl_path = str(_PROJECT_ROOT / 'models' / 'robot_next_q.stl')
        self.get_logger().info(f"Saving target mesh to {stl_path} ...")
        self._planner.save_scene_as_mesh(q_goal, stl_path)
        self.get_logger().info("Mesh saved.")

        # 2. Collision check on target
        self.get_logger().info("Checking target config for collision...")
        self_col = self._planner.self_collision_check(q_goal)
        if self_col:
            self.get_logger().error(
                f"Target config is in self collision (self={self_col}). Aborting."
            )
            with self._lock:
                self._state = _IDLE
            return
        self.get_logger().info("Target is collision-free. Planning trajectory...")

        # 3. Plan
        traj, last_tstep, ok = self._planner.plan_to_joint(q_start, q_goal)
        if not ok:
            self.get_logger().error("cuRobo planning failed. Aborting.")
            with self._lock:
                self._state = _IDLE
            return

        traj = traj[:last_tstep + 1]
        dt = self._planner._interpolation_dt
        self.get_logger().info(
            f"Trajectory ready: {traj.shape[0]} steps "
            f"(dt={dt:.3f}s, ~{traj.shape[0] * dt:.1f}s total). Executing..."
        )

        

        # 4. Hand off to exec timer
        with self._lock:
            self._traj       = traj
            self._traj_index = 0
            self._state      = _EXECUTING


def main(args=None):
    rclpy.init(args=args)
    node = BimanualTrajTestNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
