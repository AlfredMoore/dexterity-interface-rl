"""
Interactive bimanual trajectory test node.

Workflow
--------
1. Subscribes to /joint_states from rl_driver_node → q_start.
2. Loads pre-sampled pre-grasp joint configs from a .pt file.
3. Keyboard loop lets the user browse and select a target config → q_goal.
4. On 'g' / Enter: collision-check q_goal, plan a collision-free trajectory
   with cuRobo, execute it at a fixed rate, then save the goal joint state.
5. On 'h': plan and execute a return trajectory to HOME_Q.

The executor spins in a background thread (handles /joint_states subscription).
The main thread runs a sequential keyboard loop; input is accepted only after
the current plan+execute cycle is fully complete.

Trajectory execution uses a plain time-compensated loop in the main thread —
no ROS timer, no state machine — so waypoints are sent at a precise, steady rate.

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
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
import numpy as np
import torch
import yaml
import time
from pathlib import Path
import importlib.util

from robot_motion_interface.utils.kinematics import CuRoboBimanualMotionPlanner

# --- QoS ---
from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
JS_QOS   = HIGH_PERF_QOS
T_JS_QOS = HIGH_RELIA_QOS


spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")

_RMI_ROOT     = Path(spec.origin).parent.parent.parent
_LIBS_ROOT    = _RMI_ROOT.parent
_PROJECT_ROOT = _LIBS_ROOT.parent

_ROBOT_DESC     = _LIBS_ROOT / "robot_description"
_CONFIGS_CUROBO = _ROBOT_DESC / "configs_curobo"

DEFAULT_BIMANUAL_URDF_PATH     = str((_ROBOT_DESC / "rl/bimanual_panda_tesollo.urdf").resolve())
DEFAULT_CUROBO_ROBOT_CFG_PATH  = str((_CONFIGS_CUROBO / "robot/bimanual_panda_tesollo.yml").resolve())
DEFAULT_COLLISION_SPHERES_PATH = str((_CONFIGS_CUROBO / "robot/spheres/bimanual_panda_tesollo_spheres.yml").resolve())


class BimanualTrajTestNode(Node):
    def __init__(self):
        super().__init__('bimanual_traj_test_node')

        # ── Parameters ────────────────────────────────────────────────────────
        default_drive_cfg  = str(_RMI_ROOT / 'config' / 'rl_bimanual_driver_config.yaml')
        default_policy_cfg = str(_RMI_ROOT / 'config' / 'rl_policy_node_config.yaml')
        default_pregrasp   = str(_PROJECT_ROOT / 'models' / 'pre_grasp_q_samples.pt')
        self.declare_parameter('driver_cfg_path', default_drive_cfg)
        self.declare_parameter('policy_cfg_path', default_policy_cfg)
        self.declare_parameter('pregrasp_path',   default_pregrasp)
        driver_cfg_path = self.get_parameter('driver_cfg_path').get_parameter_value().string_value
        policy_cfg_path = self.get_parameter('policy_cfg_path').get_parameter_value().string_value
        pregrasp_path   = self.get_parameter('pregrasp_path').get_parameter_value().string_value

        self.get_logger().info(f"driver_cfg:  {driver_cfg_path}")
        self.get_logger().info(f"policy_cfg:  {policy_cfg_path}")
        self.get_logger().info(f"pregrasp:    {pregrasp_path}")

        with open(driver_cfg_path, 'r') as f:
            driver_cfg = yaml.safe_load(f)
        with open(policy_cfg_path, 'r') as f:
            policy_cfg = yaml.safe_load(f)

        self._infer_rate: float = float(policy_cfg["infer_rate"])

        # ── HOME_Q ────────────────────────────────────────────────────────────
        _panda_home   = np.array(driver_cfg['panda_home_joint_positions'],   dtype=np.float32)
        _tesollo_home = np.array(driver_cfg['tesollo_home_joint_positions'], dtype=np.float32)
        self._home_q  = np.concatenate([_panda_home, _tesollo_home, _panda_home, _tesollo_home])

        # ── Joint names ───────────────────────────────────────────────────────
        l_names = ['left_'  + n for n in driver_cfg['left_panda_joint_names']  + driver_cfg['left_tesollo_joint_names']]
        r_names = ['right_' + n for n in driver_cfg['right_panda_joint_names'] + driver_cfg['right_tesollo_joint_names']]
        self.joint_names: list[str] = l_names + r_names   # 38 joints

        # ── Pre-grasp configs ─────────────────────────────────────────────────
        pregrasp_tensor = torch.load(pregrasp_path, weights_only=True)   # (N, 38)
        self._pregrasp_qs: np.ndarray = pregrasp_tensor.numpy().astype(np.float32)
        self._n_configs = len(self._pregrasp_qs)
        if self._n_configs == 0:
            raise RuntimeError(f"No pre-grasp configs found in {pregrasp_path}")
        self.get_logger().info(f"Loaded {self._n_configs} pre-grasp configs.")

        # _js_lock guards _q_current (written by background js callback thread)
        self._js_lock             = threading.Lock()
        self._q_current: np.ndarray | None = None

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
            interpolation_dt            = 1.0 / self._infer_rate,
            collision_activation_distance = 0.01,
        )

        # ── ROS pub / sub ─────────────────────────────────────────────────────
        self.target_pub = self.create_publisher(JointState, '/target_joint_states', T_JS_QOS)
        self.create_subscription(JointState, '/joint_states', self._js_callback, JS_QOS)

    # ── Callback (runs in executor background thread) ─────────────────────────

    def _js_callback(self, msg: JointState) -> None:
        q = np.array(msg.position, dtype=np.float32)
        with self._js_lock:
            self._q_current = q

    # ── Sequential plan-and-execute (called from main thread) ────────────────

    def _plan_and_execute(self, q_start: np.ndarray, q_goal: np.ndarray) -> bool:
        """
        Collision-check, plan, execute at a steady rate, save goal joint state.
        Execution uses a time-compensated loop directly in the calling thread —
        no ROS timer — to avoid scheduler-induced bursty behaviour.
        Returns True on success, False on collision or planning failure.
        """
        # 1. Collision check
        print("Checking target config for self-collision...")
        if self._planner.self_collision_check(q_goal):
            print("[ERROR] Target config is in self-collision. Aborting.")
            return False
        print("Collision-free. Planning trajectory...")

        # 2. Plan
        traj, last_tstep, ok = self._planner.plan_to_joint(q_start, q_goal)
        if not ok:
            print("[ERROR] cuRobo planning failed. Aborting.")
            return False

        traj = traj[:last_tstep + 1]
        dt   = self._planner._interpolation_dt
        T    = traj.shape[0]
        print(f"Trajectory ready: {T} steps  "
              f"(dt={dt:.3f}s, ~{T * dt:.1f}s total). Executing...")

        # 3. Execute — time-compensated loop, one waypoint per dt seconds.
        #    Using absolute deadlines so jitter in publish/sleep does not accumulate.
        msg = JointState()
        msg.name = self.joint_names
        deadline = time.perf_counter()
        for q in traj:
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position     = q.tolist()
            self.target_pub.publish(msg)

            deadline += dt
            sleep_s = deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        # 4. Save goal joint state
        npy_path = _PROJECT_ROOT / 'models' / 'robot_next_q.npy'
        np.save(str(npy_path), q_goal)
        print(f"[saved] {npy_path}")

        # 5. Export mesh in background (lazy-cached, safe to call repeatedly)
        mesh_path = str(_PROJECT_ROOT / 'models' / 'scene_goal.stl')
        def _save_mesh():
            self._planner.save_scene_as_mesh(q_goal, mesh_path)
            print(f"[saved] {mesh_path}")
        threading.Thread(target=_save_mesh, daemon=True).start()

        return True

    # ── Sequential keyboard loop (run from main thread) ───────────────────────

    def run(self) -> None:
        """
        Sequential keyboard loop.  Call from the main thread while the executor
        spins in a background thread.
        """
        selected_idx = 0
        N = self._n_configs

        print("\nReady. Commands (type + Enter):")
        print("  n          next config")
        print("  p          prev config")
        print("  <number>   jump to index")
        print("  g / Enter  plan + execute")
        print("  h          return to HOME")
        print("  q          quit\n")
        self._print_status(selected_idx)

        while rclpy.ok():
            try:
                raw = input(f"[{selected_idx}/{N - 1}]> ").strip()
            except EOFError:
                break

            if raw == 'q':
                rclpy.shutdown()
                break

            elif raw == 'n':
                selected_idx = (selected_idx + 1) % N
                self._print_status(selected_idx)

            elif raw == 'p':
                selected_idx = (selected_idx - 1) % N
                self._print_status(selected_idx)

            elif raw.isdigit():
                selected_idx = int(raw) % N
                self._print_status(selected_idx)

            elif raw in ('g', ''):
                with self._js_lock:
                    q_start = self._q_current
                if q_start is None:
                    print("[ERROR] No /joint_states received yet.")
                    continue
                q_goal = self._pregrasp_qs[selected_idx].copy()
                self._plan_and_execute(q_start, q_goal)
                self._print_status(selected_idx)

            elif raw == 'h':
                with self._js_lock:
                    q_start = self._q_current
                if q_start is None:
                    print("[ERROR] No /joint_states received yet.")
                    continue
                self._plan_and_execute(q_start, self._home_q.copy())
                self._print_status(selected_idx)

            else:
                print(f"Unknown command: '{raw}'  (n/p/<num>/g/h/q)")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _print_status(self, idx: int) -> None:
        q = self._pregrasp_qs[idx]
        print(f"\nSelected [{idx}/{self._n_configs - 1}]")
        print(f"  left_arm  = {[f'{x:.3f}' for x in q[:7]]}")
        print(f"  right_arm = {[f'{x:.3f}' for x in q[19:26]]}\n")


def main(args=None):
    rclpy.init(args=args)
    node = BimanualTrajTestNode()

    # Executor spins in background — handles /joint_states subscription only
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
