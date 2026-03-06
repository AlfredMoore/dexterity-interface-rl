"""
Kinematics utilities: Pinocchio (FK) and cuRobo (motion planning) backends.

Usage intent:
  PinocchioBimanualFK          — real-time FK in the policy control loop (CPU, ~60 Hz).
  CuRoboBimanualMotionPlanner  — offline collision-free trajectory generation and
                                  collision checking (GPU, pre-grasp / task setup).
"""
from __future__ import annotations

from pathlib import Path
import copy

import numpy as np
import time
import pinocchio as pin

import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.types.robot import RobotConfig

from curobo.util.trajectory import InterpolateType



from curobo.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)

from curobo.geom.types import Cuboid, WorldConfig

from curobo.rollout.rollout_base import Goal
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# ---------------------------------------------------------------------------
# Project-relative path resolution
#   __file__ = .../libs/robot_motion_interface/src/robot_motion_interface/utils/kinematics.py
#   parents[4] = .../libs/
# ---------------------------------------------------------------------------
_LIBS_ROOT = Path(__file__).parents[4]
_ROBOT_DESC = _LIBS_ROOT / "robot_description"
_CONFIGS_CUROBO = _ROBOT_DESC / "configs_curobo"

DEFAULT_BIMANUAL_URDF_PATH      = str((_ROBOT_DESC / "rl/bimanual_panda_tesollo.urdf").resolve())

DEFAULT_CUROBO_ROBOT_CFG_PATH   = str((_CONFIGS_CUROBO / "robot/bimanual_panda_tesollo.yml").resolve())
DEFAULT_COLLISION_SPHERES_PATH  = str((_CONFIGS_CUROBO / "robot/spheres/bimanual_panda_tesollo_spheres.yml").resolve())
DEFAULT_CUROBO_WORLD_CFG_PATH   = str((_CONFIGS_CUROBO / "world/bimanual_table.yml").resolve())



# ---------------------------------------------------------------------------
# Pinocchio backend
# ---------------------------------------------------------------------------

class PinocchioBimanualFK:
    """
    Bimanual FK via Pinocchio using a single dual-arm URDF.

    The bimanual URDF root is 'world'; both arms are attached via fixed joints,
    so all results are directly in world frame — no additional transforms needed.

    Expected joint ordering: q = [left_joints..., right_joints...]
    This must match the joint declaration order in the bimanual URDF.

    Example:
        fk = PinocchioBimanualFK(
            urdf_path          = "/path/to/bimanual_panda_tesollo.urdf",
            action_per_chain   = 19,
            left_fingertip_names  = ["left_F1_TIP_TOP", "left_F2_TIP_TOP", "left_F3_TIP_TOP"],
            right_fingertip_names = ["right_F1_TIP_TOP", "right_F2_TIP_TOP", "right_F3_TIP_TOP"],
            left_hand_base_names  = ["left_delto_base_link"],
            right_hand_base_names = ["right_delto_base_link"],
        )
        l_base, l_tips, r_base, r_tips = fk.forward(q_left, q_right)
        # each array: [n, 7] = [x, y, z, qw, qx, qy, qz]
        l_base_pos,  l_base_quat  = l_base[:, :3],  l_base[:, 3:]
        l_tips_pos,  l_tips_quat  = l_tips[:, :3],  l_tips[:, 3:]
    """

    def __init__(
        self,
        urdf_path: str,
        action_per_chain: int,
        left_fingertip_names: list[str],
        right_fingertip_names: list[str],
        left_hand_base_names: list[str],
        right_hand_base_names: list[str],
    ) -> None:
                
        self._pin = pin
        self._action_per_chain = action_per_chain

        self.model = self._pin.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()

        assert self.model.nq == action_per_chain * 2, (
            f"Bimanual model nq={self.model.nq} != 2 * action_per_chain={action_per_chain * 2}. "
            "Verify the URDF is bimanual and action_per_chain matches."
        )

        self.l_hand_base_ids = self._resolve_frame_ids(left_hand_base_names)
        self.l_fingertip_ids = self._resolve_frame_ids(left_fingertip_names)
        self.r_fingertip_ids = self._resolve_frame_ids(right_fingertip_names)
        self.r_hand_base_ids = self._resolve_frame_ids(right_hand_base_names)

    def _resolve_frame_ids(self, names: list[str]) -> list[int]:
        ids = []
        for name in names:
            fid = self.model.getFrameId(name)
            if fid == self.model.nframes:
                available = [self.model.frames[i].name for i in range(self.model.nframes)]
                raise ValueError(
                    f"Frame '{name}' not found in Pinocchio model.\n"
                    f"Available frames: {available}"
                )
            ids.append(fid)
        return ids

    def _frame_pose(self, fid: int) -> np.ndarray:
        """Return [7,] = [x, y, z, qw, qx, qy, qz] for a frame (world frame, float32).

        Pinocchio internal convention:
          data.oMf[i].rotation  -> 3×3 SO(3) matrix
          pin.Quaternion(R)      -> Quaternion with .w .x .y .z components
          .coeffs()              -> [qx, qy, qz, qw]  (xyzw, pinocchio order)
        We output wxyz to match cuRobo / Isaac convention.
        """
        pose = self.data.oMf[fid]
        t = pose.translation
        q = self._pin.Quaternion(pose.rotation)
        return np.array([t[0], t[1], t[2], q.w, q.x, q.y, q.z], dtype=np.float32)

    def forward(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run FK and return 7-DoF poses for hand bases and fingertips in world frame.

        Returns:
            l_base   : float32 [n_base, 7]  left  hand base  poses
            l_tips   : float32 [n_tips, 7]  left  fingertip  poses
            r_base   : float32 [n_base, 7]  right hand base  poses
            r_tips   : float32 [n_tips, 7]  right fingertip  poses

        Each row: [x, y, z, qw, qx, qy, qz]  (position + quaternion, wxyz convention)
        """
        q = np.concatenate([q_left, q_right]).astype(np.float64)
        self._pin.forwardKinematics(self.model, self.data, q)
        self._pin.updateFramePlacements(self.model, self.data)

        l_hand_base  = np.array([self._frame_pose(i) for i in self.l_hand_base_ids],  dtype=np.float32)
        l_fingertips = np.array([self._frame_pose(i) for i in self.l_fingertip_ids],  dtype=np.float32)
        r_hand_base  = np.array([self._frame_pose(i) for i in self.r_hand_base_ids],  dtype=np.float32)
        r_fingertips = np.array([self._frame_pose(i) for i in self.r_fingertip_ids],  dtype=np.float32)

        return l_hand_base, l_fingertips, r_hand_base, r_fingertips


# ---------------------------------------------------------------------------
# cuRobo backend
# ---------------------------------------------------------------------------

class CuRoboBimanualMotionPlanner:
    """
    GPU-accelerated bimanual collision-free trajectory planning via cuRobo.

    Uses TrajOptSolver for joint-space trajectory optimisation with self-collision
    avoidance.  World-obstacle collision is NOT enforced by the planner itself —
    use self_collision_check / check_at_planning_distance for that.

    Intended for offline / pre-grasp use, NOT the real-time 60 Hz policy loop.
    Requires a CUDA-capable GPU.

    Constructor parameters
    ----------------------
    robot_cfg_path              : absolute path to the cuRobo robot YAML
                                  (kinematics, cspace, collision spheres)
    urdf_path                   : absolute path to the bimanual URDF (overrides
                                  the path stored inside robot_cfg_path at runtime)
    spheres_path                : absolute path to the collision-sphere YAML
    left_ee_link                : name of the primary (left) EE link
    right_ee_link               : name of the secondary (right) EE link
                                  (used as a link_poses constraint in IK)
    joint_names                 : 38-DoF ordered joint name list; defaults to
                                  cspace.joint_names from robot_cfg_path
    device                      : CUDA device string, e.g. "cuda:0"
    trajopt_tsteps              : number of time steps for the trajectory optimiser
    interpolation_steps         : number of waypoints in the interpolated output
    num_ik_seeds                : parallel IK seeds
    num_trajopt_seeds           : parallel TrajOpt seeds
    grad_trajopt_iters          : gradient iterations per TrajOpt seed
    interpolation_dt            : time between consecutive output waypoints (s)
    collision_activation_distance : self-collision margin used by TrajOpt and
                                  check_at_planning_distance (metres)

    Key methods
    -----------
    solve_ik(left_pos, left_quat, right_pos, right_quat)
        → (q_full_38, success)
        Bimanual IK: arm joints solved jointly, gripper joints fixed at the
        retract_config from the robot YAML.

    plan_to_joint(q_start, q_goal)
        → (traj, success, status)
        Joint-space TrajOpt with self-collision avoidance.
        traj is (T, 38) float32 at interpolation_dt seconds per step.

    plan_to_pose(q_start, left_pos, left_quat, right_pos, right_quat)
        → (traj, success, status)
        Cartesian planning: solve_ik internally to get q_goal, then plan_to_joint.

    self_collision_check(q, verbose=False)
        → bool
        Check for actual sphere penetration (activation_distance = 0).

    check_at_planning_distance(q, label="")
        → bool
        Check against the planner's collision_activation_distance margin.

    save_scene_as_mesh(q, save_path)
        Export robot geometry to STL for offline debug visualisation.

    Example
    -------
        planner = CuRoboBimanualMotionPlanner(
            trajopt_tsteps=34,
            interpolation_dt=0.02,
            collision_activation_distance=0.025,
        )

        # Joint-space planning (most common)
        traj, last_tstep, ok = planner.plan_to_joint(HOME_Q, PRE_GRASP_Q)
        if ok:
            execute(traj)   # traj: (T, 38) float32, waypoints at 0.02 s

        # Cartesian planning
        traj, last_tstep, ok = planner.plan_to_pose(
            HOME_Q,
            left_pos=np.array([0.3, 0.3, 0.8]), left_quat=np.array([1, 0, 0, 0]),
            right_pos=np.array([0.3, -0.3, 0.8]), right_quat=np.array([1, 0, 0, 0]),
        )

        # Collision checking before planning
        if planner.self_collision_check(PRE_GRASP_Q):
            print("target is in self-collision — skipping")
        elif not planner.check_at_planning_distance(PRE_GRASP_Q):
            print("target is within the planner activation margin — may cause issues")
    """

    def __init__(
        self,
        robot_cfg_path: str = DEFAULT_CUROBO_ROBOT_CFG_PATH,
        urdf_path: str = DEFAULT_BIMANUAL_URDF_PATH,
        spheres_path: str = DEFAULT_COLLISION_SPHERES_PATH,
        left_ee_link: str = "left_delto_base_link",
        right_ee_link: str = "right_delto_base_link",
        joint_names: list[str] | None = None,
        device: str = "cuda:0",
        trajopt_dt: float = 0.15,
        trajopt_tsteps: int = 34,
        interpolation_steps: int = 2000,
        num_ik_seeds: int = 30,
        num_trajopt_seeds: int = 4,
        grad_trajopt_iters: int = 500,
        interpolation_dt: float = 0.02,
        collision_activation_distance: float = 0.01,
    ) -> None:

        self._device = device
        self._left_ee_link  = left_ee_link
        self._right_ee_link = right_ee_link
        self._interpolation_dt = interpolation_dt
        self._robot_cfg_path = robot_cfg_path
        self._urdf_path      = urdf_path
        self._spheres_path   = spheres_path
        self._collision_activation_distance = collision_activation_distance

        self.tensor_args = TensorDeviceType(device=torch.device(device))

        robot_cfg_dict = load_yaml(robot_cfg_path)
        robot_cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path
        robot_cfg_dict["robot_cfg"]["kinematics"]["collision_spheres"] = spheres_path
        self.robot_cfg = RobotConfig.from_dict(robot_cfg_dict, self.tensor_args)

        # Joint names: explicit override → YAML cspace.joint_names (single source of truth)
        self._joint_names: list[str] = (
            joint_names if joint_names is not None
            else list(robot_cfg_dict["robot_cfg"]["kinematics"]["cspace"]["joint_names"])
        )

        # Trajectory optimizer (no world model, self-collision only)
        trajopt_config = TrajOptSolverConfig.load_from_robot_config(
            robot_cfg=self.robot_cfg,
            world_model=None,
            tensor_args=self.tensor_args,
            self_collision_check=True,
            self_collision_opt=True,
            trajopt_dt=trajopt_dt,
            traj_tsteps=trajopt_tsteps,
            interpolation_dt=interpolation_dt,
            interpolation_type=InterpolateType.KUNZ_STILMAN_OPTIMAL,
            interpolation_steps=interpolation_steps,
            num_seeds=num_trajopt_seeds,
            filter_robot_command=True,
            grad_trajopt_iters=grad_trajopt_iters,
            collision_activation_distance=collision_activation_distance,
            evaluate_interpolated_trajectory=True,
            optimize_dt=False,
            use_cuda_graph=True,
        )
        self._trajopt = TrajOptSolver(trajopt_config)

        # IK solver (for plan_to_pose: EE poses → joint config)
        ik_robot_cfg_dict = copy.deepcopy(robot_cfg_dict)
        self._lock_joints = {}
        
        self._retract_config = ik_robot_cfg_dict["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
        for j_name, j_val in zip(
            self._joint_names, 
            self._retract_config
        ):
            if "panda_joint" not in j_name:
                self._lock_joints[j_name] = j_val
        ik_robot_cfg_dict["robot_cfg"]["kinematics"]["lock_joints"] = self._lock_joints
        
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg=RobotConfig.from_dict(ik_robot_cfg_dict, self.tensor_args),
            world_model=None,
            tensor_args=self.tensor_args,
            num_seeds=num_ik_seeds,
            self_collision_check=True,
            self_collision_opt=True,
            use_cuda_graph=True,
        )
        self._ik_solver = IKSolver(ik_config)

        # Warmup: triggers CUDA graph compilation for both solvers
        t_start = time.time()
        self._warmup()
        t_end = time.time()
        print(f"TrajOpt+IK warmup took {(t_end - t_start):.6f} seconds.")

        # cuRobo internal joint ordering (URDF kinematic-chain traversal order).
        # All cuRobo APIs expect joints in this order; we reorder inputs/outputs accordingly.
        self._internal_joint_names: list[str] = list(self._trajopt.rollout_fn.joint_names)

        # Collision checker at exact penetration (self_collision_check)
        robot_world_cfg = RobotWorldConfig.load_from_config(
            robot_config=self.robot_cfg,
            world_model=None,
            self_collision_activation_distance=0.0,
        )
        self._robot_world = RobotWorld(robot_world_cfg)

        # Collision checker at planning activation distance (check_at_planning_distance)
        robot_world_plan_cfg = RobotWorldConfig.load_from_config(
            robot_config=self.robot_cfg,
            world_model=None,
            self_collision_activation_distance=collision_activation_distance,
        )
        self._robot_world_plan = RobotWorld(robot_world_plan_cfg)

    def __del__(self) -> None:
        """Release internal cuRobo solvers and return freed GPU memory to the CUDA driver.

        PyTorch uses a caching allocator: memory freed by deleting tensors/modules
        stays in PyTorch's own free-block pool until empty_cache() is called.
        This only affects the current process — other processes have independent
        CUDA contexts and are unaffected.  Active tensors in the same process
        (e.g. a policy network) are also unaffected; only the pool of already-free
        blocks is returned to the driver.
        """
        for attr in ("_trajopt", "_ik_solver", "_robot_world", "_robot_world_plan"):
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _warmup(self) -> None:
        """Run dummy solves to trigger CUDA graph compilation."""
        q_ret = self._trajopt.retract_config  # [1, dof], internal order
        goal = Goal(
            goal_state=JointState.from_position(q_ret),
            current_state=JointState.from_position(q_ret),
        )
        self._trajopt.solve_single(goal)

        dummy_pos  = torch.zeros(3, dtype=self.tensor_args.dtype)
        dummy_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.tensor_args.dtype)
        self.solve_ik(
            left_pos=dummy_pos.numpy(), left_quat=dummy_quat.numpy(),
            right_pos=dummy_pos.numpy(), right_quat=dummy_quat.numpy(),
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_joint_state(self, q: np.ndarray | torch.Tensor) -> JointState:
        """Return a JointState with positions reordered to cuRobo's internal joint order."""
        q_t = torch.as_tensor(q, dtype=torch.float32, device=self._device)
        if q_t.dim() == 1:
            q_t = q_t.unsqueeze(0)
        js = JointState.from_position(q_t, joint_names=self._joint_names)
        return js.get_ordered_joint_state(self._internal_joint_names)

    def _make_pose(self, pos: np.ndarray, quat: np.ndarray) -> Pose:
        return Pose(
            position=torch.tensor(pos,  dtype=torch.float32, device=self._device).unsqueeze(0),
            quaternion=torch.tensor(quat, dtype=torch.float32, device=self._device).unsqueeze(0),
        )

    def _extract_trajectory(self, result) -> tuple[np.ndarray | None, int, bool]:
        if result.success.any().item():
            traj = result.interpolated_solution
            if traj is None:
                traj = result.solution
            if traj.joint_names is not None:
                traj = traj.get_ordered_joint_state(self._joint_names)
            pos = traj.position
            last_tstep = int(result.path_buffer_last_tstep[0].item())
            if pos.dim() == 3:
                pos = pos[0]  # [T, dof]
            return pos.cpu().numpy(), last_tstep, True
        return None, 0, False

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        left_pos: np.ndarray | torch.Tensor,   # (3,)  xyz in world frame
        left_quat: np.ndarray | torch.Tensor,  # (4,)  w x y z
        right_pos: np.ndarray | torch.Tensor,  # (3,)  xyz in world frame
        right_quat: np.ndarray | torch.Tensor, # (4,)  w x y z
    ) -> tuple[torch.Tensor | None, bool]:
        """
        Solve bimanual IK for the given left and right EE poses simultaneously.

        Left EE is the primary goal passed to IKSolver.solve_single; right EE is
        enforced as a secondary constraint via link_poses. Both arms are solved in
        a single joint optimisation pass.

        Gripper joints (non-panda joints) are locked at the retract_config values
        specified in the robot YAML and are not part of the IK optimisation.
        The returned q_full always covers all 38 DoF, with locked joints filled
        from retract_config.

        Args:
            left_pos   : (3,) xyz position of left EE in world frame.
            left_quat  : (4,) quaternion [w, x, y, z] of left EE orientation.
            right_pos  : (3,) xyz position of right EE in world frame.
            right_quat : (4,) quaternion [w, x, y, z] of right EE orientation.

        Returns:
            q_full  : (38,) float32 tensor on self._device in self._joint_names
                      order, or None if IK failed.
            success : True if a valid self-collision-free IK solution was found.
        """
        left_pose  = self._make_pose(left_pos,  left_quat)
        right_pose = self._make_pose(right_pos, right_quat)
        result = self._ik_solver.solve_single(
            left_pose,
            link_poses={self._right_ee_link: right_pose},
        )
        if not result.success.any():
            return None, False
        # 1. get IK 14 unlocked joints
        ik_active_names = self._ik_solver.kinematics.joint_names
        ik_active_pos = result.solution.squeeze().cpu().numpy() # (14,)

        full_q = np.array(self._retract_config, dtype=np.float32)
        
        for idx, name in enumerate(self._joint_names):
            if name in ik_active_names:
                full_q[idx] = ik_active_pos[ik_active_names.index(name)]

        q_full = torch.tensor(full_q, dtype=torch.float32, device=self._device)

        return q_full, True
    # ------------------------------------------------------------------
    # Trajectory planning — Cartesian EE space
    # ------------------------------------------------------------------

    def plan_to_pose(
        self,
        q_start: np.ndarray,            # (n_joints,)
        left_target_pos: np.ndarray,    # (3,)  xyz in world frame
        left_target_quat: np.ndarray,   # (4,)  w x y z
        right_target_pos: np.ndarray,   # (3,)  xyz in world frame
        right_target_quat: np.ndarray,  # (4,)  w x y z
    ) -> tuple[np.ndarray | None, int, bool]:
        """
        Plan a self-collision-free trajectory to the given bimanual EE poses.

        Pipeline: solve_ik(left_target, right_target) → q_goal → plan_to_joint.
        If IK fails, returns immediately without calling TrajOpt.

        Args:
            q_start           : (n_joints,) current joint configuration.
            left_target_pos   : (3,) target xyz for left EE in world frame.
            left_target_quat  : (4,) target orientation [w, x, y, z] for left EE.
            right_target_pos  : (3,) target xyz for right EE in world frame.
            right_target_quat : (4,) target orientation [w, x, y, z] for right EE.

        Returns:
            trajectory : (T, n_joints) float32 numpy array — waypoints at
                         interpolation_dt second intervals — or None on failure.
            success    : True if a valid trajectory was found.
        """
        q_goal, ik_ok = self.solve_ik(
            left_target_pos, left_target_quat,
            right_target_pos, right_target_quat,
        )
        if not ik_ok:
            return None, 0, False
        return self.plan_to_joint(q_start, q_goal)

    # ------------------------------------------------------------------
    # Trajectory planning — joint space
    # ------------------------------------------------------------------

    def plan_to_joint(
        self,
        q_start: np.ndarray,    # (n_joints,)
        q_goal: np.ndarray,     # (n_joints,)
    ) -> tuple[np.ndarray | None, int, bool]:
        """
        Plan a self-collision-free trajectory to the given target joint configuration.

        Uses TrajOptSolver with self-collision avoidance. World obstacles are NOT
        checked here — call self_collision_check or check_at_planning_distance
        separately to verify start/goal safety before planning.

        Args:
            q_start : (n_joints,) current joint configuration in self._joint_names order.
            q_goal  : (n_joints,) target joint configuration in self._joint_names order.

        Returns:
            trajectory : (T, n_joints) float32 numpy array — waypoints at
                         interpolation_dt second intervals — or None on failure.
            last_tstep : the last valid time step in the returned trajectory (0 if failure).
            success    : True if a valid trajectory was found.
        """
        goal = Goal(
            goal_state=self._make_joint_state(q_goal),
            current_state=self._make_joint_state(q_start),
        )
        return self._extract_trajectory(self._trajopt.solve_single(goal))

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def self_collision_check(
        self,
        q: np.ndarray,  # (n_joints,)
        verbose: bool = False,
    ) -> bool:
        """
        Check whether a joint configuration is in actual sphere-sphere self-collision.

        Uses a RobotWorld with activation_distance = 0, so only real penetration
        (cost > 0) triggers a True result. World obstacles are NOT checked.
        cuRobo cost convention: cost == 0 → free, cost > 0 → spheres overlap.

        Args:
            q       : (n_joints,) joint configuration in self._joint_names order.
            verbose : If True and collision detected, print the non-zero indices
                      and values of d_self for quick debugging.

        Returns:
            True if any sphere pair is in self-collision, False otherwise.
        """
        q_t = self._make_joint_state(q).position
        if q_t.dim() == 1:
            q_t = q_t.unsqueeze(0)
        kin_state = self._robot_world.get_kinematics(q_t)
        d_self = self._robot_world.get_self_collision(
            kin_state.link_spheres_tensor.unsqueeze(1)
        )
        self_col = bool((d_self  > 0.0).any().item())
        if verbose and self_col:
            mask = d_self > 0.0
            print(f"  d_self  nonzero idx={mask.nonzero(as_tuple=False).squeeze(-1).tolist()}"
                  f"  values={d_self[mask].tolist()}")
        return self_col

    def check_at_planning_distance(self, q: np.ndarray, label: str = "") -> bool:
        """
        Check whether q is clear of self-collision at the planner's activation margin.

        Uses a separate RobotWorld with self_collision_activation_distance set to
        collision_activation_distance (from the constructor), matching the margin
        that TrajOpt uses internally. A config that passes here is guaranteed to
        be accepted as a valid start/goal by plan_to_joint.

        Always prints a one-line status summary to stdout regardless of the result.

        Args:
            q     : (n_joints,) joint configuration in self._joint_names order.
            label : Optional string appended to the printed status line.

        Returns:
            True if the config is free of self-collision within the activation margin.
        """
        q_t = self._make_joint_state(q).position
        if q_t.dim() == 1:
            q_t = q_t.unsqueeze(0)
        kin_state = self._robot_world_plan.get_kinematics(q_t)
        d_self = self._robot_world_plan.get_self_collision(
            kin_state.link_spheres_tensor.unsqueeze(1)
        )
        valid = not bool((d_self > 0.0).any().item())
        tag = "[VALID@plan_dist]  " if valid else "[INVALID@plan_dist]"
        suffix = f" {label}" if label else ""
        print(f"{tag}{suffix}  (activation_distance={self._collision_activation_distance}m)")
        return valid

    def benchmark_traj_collision_check(
        self,
        traj: np.ndarray,
        activation_distance: float | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Run sequential (one-by-one) self-collision checks on every waypoint of a
        trajectory to measure the achievable real-time collision-check frequency.

        Each waypoint is checked independently in a serial loop — no batching —
        to mimic the condition in a live control loop where only the current
        configuration is tested.

        Args:
            traj               : (T, n_joints) float32 array of joint configurations
                                 in self._joint_names order (e.g. output of plan_to_joint).
            activation_distance: sphere-sphere activation margin (metres) used for
                                 checking.  None (default) → 0.0 (actual penetration).
                                 Pass self._collision_activation_distance to use the
                                 planner's own margin.
            verbose            : If True, print per-waypoint result.

        Returns:
            A dict with keys:
                n_steps         : int    — number of waypoints checked.
                n_collisions    : int    — number of waypoints in collision.
                collision_mask  : list[bool] — per-waypoint collision flag.
                total_time_s    : float  — wall-clock seconds for the full loop.
                mean_time_ms    : float  — mean time per check in milliseconds.
                check_freq_hz   : float  — 1 / mean_time_ms * 1000 (estimated Hz).
        """
        if activation_distance is None:
            robot_world = self._robot_world           # activation = 0
        else:
            # Build a temporary checker at the requested distance.
            cfg = RobotWorldConfig.load_from_config(
                robot_config=self.robot_cfg,
                world_model=None,
                self_collision_activation_distance=activation_distance,
            )
            robot_world = RobotWorld(cfg)

        T = traj.shape[0]
        collision_mask: list[bool] = []

        t_loop_start = time.time()
        for i in range(T):
            q = traj[i]
            q_t = self._make_joint_state(q).position
            if q_t.dim() == 1:
                q_t = q_t.unsqueeze(0)

            t0 = time.perf_counter()
            kin_state = robot_world.get_kinematics(q_t)
            d_self = robot_world.get_self_collision(
                kin_state.link_spheres_tensor.unsqueeze(1)
            )
            torch.cuda.synchronize()          # ensure GPU work is done before timing
            t1 = time.perf_counter()

            in_collision = bool((d_self > 0.0).any().item())
            collision_mask.append(in_collision)

            if verbose:
                status = "COLLISION" if in_collision else "ok"
                print(f"  step {i:4d}/{T}  {status}  ({(t1-t0)*1000:.3f} ms)")

        t_loop_end = time.time()
        total_time_s = t_loop_end - t_loop_start
        mean_time_ms = total_time_s / T * 1000.0
        check_freq_hz = 1000.0 / mean_time_ms if mean_time_ms > 0 else float("inf")
        n_collisions = sum(collision_mask)

        print(
            f"[collision benchmark] {T} waypoints | "
            f"{n_collisions} collisions | "
            f"total {total_time_s*1000:.1f} ms | "
            f"mean {mean_time_ms:.3f} ms/step | "
            f"~{check_freq_hz:.1f} Hz"
        )

        return {
            "n_steps":        T,
            "n_collisions":   n_collisions,
            "collision_mask": collision_mask,
            "total_time_s":   total_time_s,
            "mean_time_ms":   mean_time_ms,
            "check_freq_hz":  check_freq_hz,
        }

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def save_scene_as_mesh(self, q: np.ndarray, save_path: str) -> None:
        """
        Save the robot at joint configuration q together with world obstacles as an STL.

        Intended for offline debug only (not real-time). Reloads the robot config
        with mesh geometry enabled, which is separate from the planning model.

        Args:
            q         : (n_joints,) joint configuration in BIMANUAL_JOINT_NAMES order.
            save_path : Path to the output .stl file (parent directory must exist).

        Example:
            planner.save_scene_as_mesh(PRE_GRASP_Q, "/tmp/debug_scene.stl")
            # Then open with MeshLab / Blender / RViz
        """
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel  # type: ignore[import-untyped]

        # Reload robot config with link mesh geometry enabled.
        # The planning model skips this to save GPU memory.
        robot_cfg_dict = load_yaml(self._robot_cfg_path)
        robot_cfg_dict["robot_cfg"]["kinematics"]["urdf_path"]                = self._urdf_path
        robot_cfg_dict["robot_cfg"]["kinematics"]["collision_spheres"]        = self._spheres_path
        robot_cfg_dict["robot_cfg"]["kinematics"]["load_link_names_with_mesh"] = True
        robot_cfg_dict["robot_cfg"]["kinematics"]["load_meshes"]              = True

        mesh_robot_cfg = RobotConfig.from_dict(robot_cfg_dict, self.tensor_args)
        kin_model      = CudaRobotModel(mesh_robot_cfg.kinematics)

        # Build a named JointState and reorder to the model's internal joint ordering.
        # kin_model.joint_names == self._internal_joint_names (same config, same URDF traversal).
        js = JointState.from_position(
            torch.tensor(q, dtype=torch.float32, device=self._device).unsqueeze(0),
            joint_names=self._joint_names,
        )
        js_ordered   = js.get_ordered_joint_state(kin_model.joint_names)
        robot_meshes = kin_model.get_robot_as_mesh(js_ordered.position)

        scene = WorldConfig(mesh=robot_meshes[:])
        scene.save_world_as_mesh(str(Path(save_path).resolve()), process_color=False)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Smoke test: collision check + plan HOME_Q → PRE_GRASP_Q
    # ------------------------------------------------------------------
    HOME_Q = np.array([
        0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)

    PRE_GRASP_Q = np.array([
        # Left arm + gripper (19)
        -0.6981317007977318, 0.9075712110370514, 0.14835298641951802, -1.8657569703819383,
        1.3788101090755203,  1.6126842288427605, 2.0943951023931953,
        0.0, 0.0, 0.7853981633974483, 0.5235987755982988,
        0.0, 0.0, 0.7853981633974483, 0.5235987755982988,
        0.0, 0.0, 0.7853981633974483, 0.5235987755982988,
        # Right arm + gripper (19)
        -0.19198621771937624, 0.32986722862692824, 0.07853981633974483, -1.8936822384138476,
        -0.059341194567807204, 2.1415189921970423, 0.8000065692366409,
        0.0, 0.0, 0.7853981633974483, 0.5235987755982988,
        0.0, 0.0, 0.7853981633974483, 0.5235987755982988,
        0.0, 0.0, 0.7853981633974483, 0.5235987755982988,
    ], dtype=np.float32)

    INFER_RATE = 30

    def test_curobo():
        
        print("\n" + "=" * 60)
        print("Curobo smoke test")
        print("=" * 60)
        
        print("\n0. Initializing CuRoboBimanualMotionPlanner (warmup may take ~30 s)...")
        planner = CuRoboBimanualMotionPlanner(
            robot_cfg_path              = DEFAULT_CUROBO_ROBOT_CFG_PATH,
            left_ee_link                = "left_delto_base_link",
            right_ee_link               = "right_delto_base_link",
            device                      = "cuda:0",
            trajopt_tsteps              = 64,
            interpolation_steps         = 2000,
            num_ik_seeds                = 50,
            num_trajopt_seeds           = 32,
            grad_trajopt_iters          = 800,
            interpolation_dt            = 1 / INFER_RATE,
            collision_activation_distance = 0.005,
        )


        def _collision_check_test(name, q):
            t_start = time.time()
            self_col = planner.self_collision_check(q, verbose=True)
            if self_col:
                print(f"[COLLISION] {name}  self={self_col}")
            else:
                print(f"[OK] {name} is collision-free.")
            t_end = time.time()
            print(f"Collision check took {(t_end - t_start):.6f} seconds.\n")
        
        # ------------------------------------------------------------------
        # Collision detection test
        # ------------------------------------------------------------------
        print("\n1. Collision check at key configurations...")
        _collision_check_test("HOME_Q", HOME_Q)
        planner.save_scene_as_mesh(HOME_Q, str((_LIBS_ROOT.parent / "models" / "robot_world_home_q.stl").resolve()))
        print(f"[INFO] Saved HOME_Q scene mesh to {str((_LIBS_ROOT.parent / 'models' / 'robot_world_home_q.stl').resolve())}")

        _collision_check_test("PRE_GRASP_Q", PRE_GRASP_Q)
        planner.save_scene_as_mesh(PRE_GRASP_Q, str((_LIBS_ROOT.parent / "models" / "robot_world_pregrasp_q.stl").resolve()))
        print(f"[INFO] Saved PRE_GRASP_Q scene mesh to {str((_LIBS_ROOT.parent / 'models' / 'robot_world_pregrasp_q.stl').resolve())}")

        # ------------------------------------------------------------------
        # Traj Optimization test
        # ------------------------------------------------------------------
        print("\n2. Trajectory planning test...")
        print("\nChecking endpoint validity at planner's activation distance...")
        home_ok     = planner.check_at_planning_distance(HOME_Q,     "HOME_Q")
        pregrasp_ok = planner.check_at_planning_distance(PRE_GRASP_Q, "PRE_GRASP_Q")
        if not home_ok or not pregrasp_ok:
            print("[WARN] One or both endpoints are invalid at the planner's activation distance.")
            print("       Reduce collision_activation_distance or adjust the joint configuration.")

        print(f"\nPlanning trajectory from HOME_Q to PRE_GRASP_Q...")
        t_start = time.time()
        traj, last_tstep, ok = planner.plan_to_joint(HOME_Q, PRE_GRASP_Q)
        if ok:
            print(
                f"[OK] Trajectory: {traj.shape[0]} steps x {traj.shape[1]} DoF  "
                f"(dt={planner._interpolation_dt:.3f}s, "
                f"~{last_tstep * planner._interpolation_dt:.1f}s total)"
            )
            print(f"     Start : {traj[0]}")
            print(f"     End   : {traj[last_tstep]}")
        else:
            print("[FAIL] Planning failed.")
        t_end = time.time()
        print(f"Planning took {(t_end - t_start):.6f} seconds.")

        # ------------------------------------------------------------------
        # IK solver test
        # ------------------------------------------------------------------
        print("\n3. IK solver test...")
        IK_POSES = [
            {
                "label": "HOME_POSE",
                "left_pos":   np.array([-0.25110966, -0.092,     0.48428035], dtype=np.float32),
                "left_quat":  np.array([ 9.422508e-12, 1.0, -1.999867e-04, -2.980232e-08], dtype=np.float32),
                "right_pos":  np.array([ 0.25190964, -0.092,     0.48428035], dtype=np.float32),
                "right_quat": np.array([ 2.980232e-08, 1.999867e-04, 1.0, 9.422508e-12], dtype=np.float32),
            },
            {
                "label": "PREGRASP_POSE",
                "left_pos":   np.array([ 0.00075642, -0.188734,  0.05798346], dtype=np.float32),
                "left_quat":  np.array([ 0.36853078, -0.07235682, 0.70111525,  0.6061246],  dtype=np.float32),
                "right_pos":  np.array([-0.0425072,  -0.0159787,  0.21864755], dtype=np.float32),
                "right_quat": np.array([ 0.04083621,  0.04871807, 0.997932,    0.00951763], dtype=np.float32),
            },
        ]

        print("\nTesting solve_ik...")
        for pose in IK_POSES:
            t_start = time.time()
            q_sol, ok = planner.solve_ik(
                pose["left_pos"],  pose["left_quat"],
                pose["right_pos"], pose["right_quat"],
            )
            t_end = time.time()
            elapsed_ms = (t_end - t_start) * 1000.0
            if ok:
                print(f"[OK]   {pose['label']}  ({elapsed_ms:.1f} ms)")
                print(f"       q = {q_sol.cpu().numpy()}")
            else:
                print(f"[FAIL] {pose['label']}: IK failed  ({elapsed_ms:.1f} ms)")

    # ------------------------------------------------------------------
    # Pinocchio FK smoke test
    # ------------------------------------------------------------------
    # Bimanual URDF has 38 DoF (nq = 2 * action_per_chain = 2 * 19).
    # PinocchioBimanualFK takes q_left (19,) and q_right (19,) separately,
    # concatenates them internally in URDF joint declaration order.
    #
    # Frame names verified against bimanual_panda_tesollo.urdf:
    #   hand base  : left_delto_base_link / right_delto_base_link
    #   fingertips : left_F{1,2,3}_TIP_TOP / right_F{1,2,3}_TIP_TOP
    #
    # Key difference vs rl_policy_node.py (old single-chain approach):
    #   Old node: two separate pin models, each nq=19, called FK twice.
    #   New class: one bimanual model, nq=38, called FK once with both arms.
    # ------------------------------------------------------------------

    def test_pinocchio():
        
        import time
        
        ACTION_PER_CHAIN = 19  # 7 arm + 12 finger joints
        LEFT_FINGERTIP_NAMES  = ["left_F1_TIP_TOP",  "left_F2_TIP_TOP",  "left_F3_TIP_TOP"]
        RIGHT_FINGERTIP_NAMES = ["right_F1_TIP_TOP", "right_F2_TIP_TOP", "right_F3_TIP_TOP"]
        LEFT_HAND_BASE_NAMES  = ["left_delto_base_link"]
        RIGHT_HAND_BASE_NAMES = ["right_delto_base_link"]
        
        print("\n" + "=" * 60)
        print("Pinocchio FK smoke test")
        print("=" * 60)
        
        fk = PinocchioBimanualFK(
            urdf_path             = DEFAULT_BIMANUAL_URDF_PATH,
            action_per_chain      = ACTION_PER_CHAIN,
            left_fingertip_names  = LEFT_FINGERTIP_NAMES,
            right_fingertip_names = RIGHT_FINGERTIP_NAMES,
            left_hand_base_names  = LEFT_HAND_BASE_NAMES,
            right_hand_base_names = RIGHT_HAND_BASE_NAMES,
        )
        print(f"Pinocchio model nq={fk.model.nq}  (expected {ACTION_PER_CHAIN * 2})")

        for _ in range(1):
            for label, q_full in [("HOME_Q", HOME_Q), ("PRE_GRASP_Q", PRE_GRASP_Q)]:
                q_left  = q_full[:ACTION_PER_CHAIN]
                q_right = q_full[ACTION_PER_CHAIN:]
                
                t_start = time.time()
                l_base, l_tips, r_base, r_tips = fk.forward(q_left, q_right)
                t_end = time.time()
                # Each array: [n, 7] = [x, y, z, qw, qx, qy, qz]
                print(f"\n--- {label}  ({(t_end-t_start)*1000:.2f} ms) ---")
                for name, pose in zip(LEFT_HAND_BASE_NAMES, l_base):
                    print(f"  L base  {name:28s} pos={pose[:3]}  quat(wxyz)={pose[3:]}")
                for name, pose in zip(RIGHT_HAND_BASE_NAMES, r_base):
                    print(f"  R base  {name:28s} pos={pose[:3]}  quat(wxyz)={pose[3:]}")
                for name, pose in zip(LEFT_FINGERTIP_NAMES, l_tips):
                    print(f"  L tip   {name:28s} pos={pose[:3]}  quat(wxyz)={pose[3:]}")
                for name, pose in zip(RIGHT_FINGERTIP_NAMES, r_tips):
                    print(f"  R tip   {name:28s} pos={pose[:3]}  quat(wxyz)={pose[3:]}")
        
    test_pinocchio()
    test_curobo()