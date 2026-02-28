"""
Kinematics utilities: Pinocchio (FK) and cuRobo (motion planning) backends.

Usage intent:
  PinocchioBimanualFK          — real-time FK in the policy control loop (CPU, ~60 Hz).
  CuRoboBimanualMotionPlanner  — offline collision-free trajectory generation and
                                  collision checking (GPU, pre-grasp / task setup).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import time
import pinocchio as pin

import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.types.robot import RobotConfig


from curobo.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)

from curobo.geom.types import Cuboid, WorldConfig

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

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

# Finger joint lock values used during IK.
# M1=0, M2=0, M3=π/4 (≈45°), M4=π/6 (≈30°) — same for every finger on both hands.
# Only the 7+7 arm joints are optimised; fingers are held fixed at these values.
_FINGER_JOINT_LOCK: dict[str, float] = {
    j: v
    for side in ("left", "right")
    for f in (1, 2, 3)
    for j, v in [
        (f"{side}_F{f}M1", 0.0),
        (f"{side}_F{f}M2", 0.0),
        (f"{side}_F{f}M3", 0.7853981633974483),   # π/4
        (f"{side}_F{f}M4", 0.5235987755982988),   # π/6
    ]
}


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

# Default joint ordering for bimanual_panda_tesollo (38 DoF).
# Must match cspace.joint_names in the robot YAML config.
BIMANUAL_JOINT_NAMES: list[str] = [
    # Left arm (7)
    "left_panda_joint1", "left_panda_joint2", "left_panda_joint3", "left_panda_joint4",
    "left_panda_joint5", "left_panda_joint6", "left_panda_joint7",
    # Left gripper (12)
    "left_F1M1", "left_F1M2", "left_F1M3", "left_F1M4",
    "left_F2M1", "left_F2M2", "left_F2M3", "left_F2M4",
    "left_F3M1", "left_F3M2", "left_F3M3", "left_F3M4",
    # Right arm (7)
    "right_panda_joint1", "right_panda_joint2", "right_panda_joint3", "right_panda_joint4",
    "right_panda_joint5", "right_panda_joint6", "right_panda_joint7",
    # Right gripper (12)
    "right_F1M1", "right_F1M2", "right_F1M3", "right_F1M4",
    "right_F2M1", "right_F2M2", "right_F2M3", "right_F2M4",
    "right_F3M1", "right_F3M2", "right_F3M3", "right_F3M4",
]


class CuRoboBimanualMotionPlanner:
    """
    Bimanual collision-free trajectory generation and collision checking via cuRobo.

    Intended for offline / pre-grasp use, NOT the real-time policy control loop.

    Requires a CUDA-capable GPU and two config YAMLs:
      robot_cfg_path  — robot kinematics + collision spheres
                        (e.g. "robot/robot_description/configs/robot/bimanual_panda_tesollo.yml")
      world_cfg_path  — static world obstacles (table, shelves, etc.)
                        (e.g. "robot/robot_description/configs/world/bimanual_table.yml")

    Paths are relative to cuRobo's working directory (same convention as cuRobo examples).

    Example:
        planner = CuRoboBimanualMotionPlanner(
            robot_cfg_path="/workspace/libs/robot_description/configs/robot/bimanual_panda_tesollo.yml",
            world_cfg_path="/workspace/libs/robot_description/configs/world/bimanual_table.yml",
        )
        traj, ok, status = planner.plan(
            q_start           = np.zeros(38),
            left_target_pos   = np.array([0.3,  0.3, 1.0]),
            left_target_quat  = np.array([1.0, 0.0, 0.0, 0.0]),   # w x y z
            right_target_pos  = np.array([0.3, -0.3, 1.0]),
            right_target_quat = np.array([1.0, 0.0, 0.0, 0.0]),
        )
        if ok:
            # traj: (T, 38) numpy array — joint positions at each timestep
            execute(traj)

        in_world, in_self = planner.is_in_collision(q)
    """

    def __init__(
        self,
        robot_cfg_path: str = DEFAULT_CUROBO_ROBOT_CFG_PATH,
        world_cfg_path: str = DEFAULT_CUROBO_WORLD_CFG_PATH,
        urdf_path: str = DEFAULT_BIMANUAL_URDF_PATH,
        spheres_path: str = DEFAULT_COLLISION_SPHERES_PATH,
        left_ee_link: str = "left_delto_base_link",
        right_ee_link: str = "right_delto_base_link",
        joint_names: list[str] | None = None,
        device: str = "cuda:0",
        # MotionGen tuning parameters
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
        self._joint_names   = joint_names if joint_names is not None else BIMANUAL_JOINT_NAMES
        self._interpolation_dt = interpolation_dt
        self._robot_cfg_path = robot_cfg_path
        self._urdf_path      = urdf_path
        self._spheres_path   = spheres_path

        self.tensor_args = TensorDeviceType(device=torch.device(device))

        robot_cfg_dict = load_yaml(robot_cfg_path)
        robot_cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path
        robot_cfg_dict["robot_cfg"]["kinematics"]["collision_spheres"] = spheres_path
        self.robot_cfg = RobotConfig.from_dict(robot_cfg_dict, self.tensor_args)
        self.world_cfg = WorldConfig.from_dict(load_yaml(world_cfg_path))

        # Motion planner (trajectory generation)
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg=self.robot_cfg,
            world_model=self.world_cfg,
            tensor_args=self.tensor_args,
            trajopt_tsteps=trajopt_tsteps,
            interpolation_steps=interpolation_steps,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
            grad_trajopt_iters=grad_trajopt_iters,
            interpolation_dt=interpolation_dt,
            collision_activation_distance=collision_activation_distance,
            evaluate_interpolated_trajectory=True,
        )
        self._collision_activation_distance = collision_activation_distance

        self._motion_gen = MotionGen(motion_gen_cfg)
        t_start = time.time()
        self._motion_gen.warmup()
        t_end = time.time()
        print(f"MotionGen warmup took {(t_end - t_start):.6f} seconds.")

        self._motion_gen.world_coll_checker.clear_cache()
        self._motion_gen.reset(reset_seed=False)

        # IK solver — built lazily on first call to solve_ik()
        self._ik_solver: IKSolver | None = None
        self._ik_arm_joint_names: list[str] = []

        # cuRobo builds its internal joint ordering from the URDF kinematic-chain traversal,
        # which may differ from self._joint_names (our user-facing order).
        # All cuRobo APIs (MotionGen, RobotWorld, CudaRobotModel) expect joints in this
        # internal order. We cache it here so we can reorder inputs/outputs consistently.
        self._internal_joint_names: list[str] = list(self._motion_gen.rollout_fn.joint_names)

        # Standalone collision checker (same robot + world, no motion planning overhead)
        robot_world_cfg = RobotWorldConfig.load_from_config(
            robot_config=self.robot_cfg,
            world_model=self.world_cfg,
            collision_activation_distance=0.0,  # 0.0 = report actual penetration depth
        )
        self._robot_world = RobotWorld(robot_world_cfg)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_joint_state(self, q: np.ndarray) -> JointState:
        """Return a JointState with positions reordered to cuRobo's internal joint order."""
        js = JointState.from_position(
            torch.tensor(q, dtype=torch.float32, device=self._device).unsqueeze(0),
            joint_names=self._joint_names,
        )
        return js.get_ordered_joint_state(self._internal_joint_names)

    def _make_pose(self, pos: np.ndarray, quat: np.ndarray) -> Pose:
        return Pose(
            position=torch.tensor(pos,  dtype=torch.float32, device=self._device).unsqueeze(0),
            quaternion=torch.tensor(quat, dtype=torch.float32, device=self._device).unsqueeze(0),
        )

    def _extract_trajectory(self, result) -> tuple[np.ndarray | None, bool, str]:
        if result.success.item():
            traj = result.get_interpolated_plan()
            # traj is in cuRobo's internal joint order; reorder to self._joint_names order.
            if traj.joint_names is not None:
                traj = traj.get_ordered_joint_state(self._joint_names)
            return traj.position.cpu().numpy(), True, str(result.status)
        return None, False, str(result.status)

    def _make_plan_config(
        self,
        max_attempts: int,
        timeout: float,
        time_dilation_factor: float,
    ) -> MotionGenPlanConfig:
        return MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=3,
            max_attempts=max_attempts,
            timeout=timeout,
            time_dilation_factor=time_dilation_factor,
        )

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
        max_attempts: int = 10,
        timeout: float = 10.0,
        time_dilation_factor: float = 1.0,
    ) -> tuple[np.ndarray | None, bool, str]:
        """
        Plan a collision-free trajectory to the given dual-arm Cartesian EE poses.

        The primary EE (left_ee_link) is passed as the main IK goal; the secondary EE
        (right_ee_link) is constrained via link_poses — matching multi_arm_reacher.py
        convention from cuRobo examples.

        Returns:
            trajectory  : (T, n_joints) float32 numpy array, or None if planning failed.
                          T waypoints spaced interpolation_dt seconds apart.
            success     : True if a valid trajectory was found.
            status      : Human-readable status string from cuRobo.
        """
        result = self._motion_gen.plan_single(
            self._make_joint_state(q_start),
            self._make_pose(left_target_pos, left_target_quat, ),
            self._make_plan_config(max_attempts, timeout, time_dilation_factor),
            link_poses=[self._make_pose(left_target_pos, left_target_quat),
                        self._make_pose(right_target_pos, right_target_quat)],
        )
        return self._extract_trajectory(result)

    # ------------------------------------------------------------------
    # Trajectory planning — joint space
    # ------------------------------------------------------------------

    def plan_to_joint(
        self,
        q_start: np.ndarray,    # (n_joints,)
        q_goal: np.ndarray,     # (n_joints,)
        max_attempts: int = 10,
        timeout: float = 10.0,
        time_dilation_factor: float = 1.0,
    ) -> tuple[np.ndarray | None, bool, str]:
        """
        Plan a collision-free trajectory to the given target joint configuration.

        Useful for moving to a known pre-grasp joint state (e.g. from env.yaml
        init poses) while guaranteeing collision avoidance along the path.

        Returns:
            trajectory  : (T, n_joints) float32 numpy array, or None if planning failed.
                          T waypoints spaced interpolation_dt seconds apart.
            success     : True if a valid trajectory was found.
            status      : Human-readable status string from cuRobo.
        """
        result = self._motion_gen.plan_single_js(
            self._make_joint_state(q_start),
            self._make_joint_state(q_goal),
            self._make_plan_config(max_attempts, timeout, time_dilation_factor),
        )
        return self._extract_trajectory(result)

    # ------------------------------------------------------------------
    # IK solver (lazy-init, finger joints locked)
    # ------------------------------------------------------------------

    def _ensure_ik_solver(self, num_seeds: int = 30) -> None:
        """Build the IK solver on first call.

        Finger joints are locked at the values in ``_FINGER_JOINT_LOCK``
        (M1=0, M2=0, M3=π/4, M4=π/6) so the optimiser only moves the
        7+7 arm joints.  A separate RobotConfig is loaded for this purpose
        so the main MotionGen model is not affected.
        """
        if self._ik_solver is not None:
            return

        robot_cfg_dict = load_yaml(self._robot_cfg_path)
        robot_cfg_dict["robot_cfg"]["kinematics"]["urdf_path"]         = self._urdf_path
        robot_cfg_dict["robot_cfg"]["kinematics"]["collision_spheres"] = self._spheres_path
        robot_cfg_dict["robot_cfg"]["kinematics"]["lock_joints"] = _FINGER_JOINT_LOCK

        ik_robot_cfg = RobotConfig.from_dict(robot_cfg_dict, self.tensor_args)
        ik_cfg = IKSolverConfig.load_from_robot_config(
            ik_robot_cfg,
            world_model=self.world_cfg,
            num_seeds=num_seeds,
            position_threshold=0.005,
            rotation_threshold=0.05,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
        )
        self._ik_solver = IKSolver(ik_cfg)
        # Active joint names in solver's internal order (14 arm joints, no fingers)
        self._ik_arm_joint_names = list(self._ik_solver.rollout_fn.joint_names)
        print(f"IKSolver ready: {len(self._ik_arm_joint_names)} active joints "
              f"(fingers locked: M1=0, M2=0, M3=π/4, M4=π/6)  seeds={num_seeds}")

    def solve_ik(
        self,
        left_target_pos: np.ndarray,   # (3,) or (n, 3)   world frame
        left_target_quat: np.ndarray,  # (4,) or (n, 4)   wxyz
        right_target_pos: np.ndarray,  # (3,) or (n, 3)
        right_target_quat: np.ndarray, # (4,) or (n, 4)   wxyz
        num_seeds: int = 30,
        max_batch: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve bimanual IK for batched (left, right) EE pose pairs.

        Finger joints are fixed at the values in ``_FINGER_JOINT_LOCK``
        (M1=0, M2=0, M3=π/4, M4=π/6) and excluded from optimisation.
        The primary EE is ``left_delto_base_link``; ``right_delto_base_link``
        is constrained via ``link_poses``.

        Parameters
        ----------
        left_target_pos / left_target_quat:
            Desired pose of left_delto_base_link.
            Shape (3,)/(4,) for one goal or (n, 3)/(n, 4) for a batch.
        right_target_pos / right_target_quat:
            Desired pose of right_delto_base_link (same shape conventions).
        num_seeds:
            Random IK seeds per goal.  More seeds → higher success rate, slower.
        max_batch:
            Goals per GPU kernel call.  Large batches are chunked automatically.

        Returns
        -------
        q_solutions : (n, 38) float32 in BIMANUAL_JOINT_NAMES order.
                      Finger joints are set to the locked values.
                      Failed rows are filled with NaN.
        success     : (n,) bool
        """
        self._ensure_ik_solver(num_seeds)
        assert self._ik_solver is not None

        l_pos  = np.atleast_2d(left_target_pos).astype(np.float32)   # (n, 3)
        l_quat = np.atleast_2d(left_target_quat).astype(np.float32)  # (n, 4)
        r_pos  = np.atleast_2d(right_target_pos).astype(np.float32)
        r_quat = np.atleast_2d(right_target_quat).astype(np.float32)
        n = l_pos.shape[0]

        # Map IK solver's active joint names → columns in the full 38-DOF config
        full_idx    = {name: i for i, name in enumerate(self._joint_names)}
        arm_to_full = [full_idx[j] for j in self._ik_arm_joint_names if j in full_idx]

        # Pre-fill output with locked finger values; arm joints are written per-batch below.
        q_full = np.zeros((n, len(self._joint_names)), dtype=np.float32)
        for jname, jval in _FINGER_JOINT_LOCK.items():
            if jname in full_idx:
                q_full[:, full_idx[jname]] = jval
        success = np.zeros(n, dtype=bool)

        for start in range(0, n, max_batch):
            end    = min(start + max_batch, n)

            goal_pose = Pose(
                position  = torch.tensor(l_pos[start:end],  device=self._device),
                quaternion= torch.tensor(l_quat[start:end], device=self._device),
            )
            right_pose = Pose(
                position  = torch.tensor(r_pos[start:end],  device=self._device),
                quaternion= torch.tensor(r_quat[start:end], device=self._device),
            )

            result = self._ik_solver.solve_batch(
                goal_pose,
                link_poses={self._right_ee_link: right_pose},
                return_seeds=1,
            )

            succ = result.success.view(-1).cpu().numpy().astype(bool)  # (batch,)
            success[start:end] = succ

            # solution shape: [batch, n_arm_dof] (return_seeds=1)
            q_arm = result.solution.view(end - start, -1).cpu().numpy()

            # Place arm joints into the correct columns of q_full
            for arm_i, full_i in enumerate(arm_to_full):
                q_full[start:end, full_i] = q_arm[:, arm_i]

        q_full[~success] = np.nan
        n_ok = int(success.sum())
        print(f"IK: {n_ok}/{n} goals solved  "
              f"({100 * n_ok / max(n, 1):.1f}%)")
        return q_full, success

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def is_in_collision(
        self,
        q: np.ndarray,  # (n_joints,)
        verbose: bool = False,
    ) -> tuple[bool, bool]:
        """
        Check whether a joint configuration is in collision.

        cuRobo cost convention: cost == 0 → free, cost > 0 → collision.

        Args:
            q       : (n_joints,) joint configuration.
            verbose : If True and collision detected, print the non-zero entries of
                      d_world and d_self (indices + values) for quick debugging.

        Returns:
            world_in_collision : True if any sphere penetrates a world obstacle.
            self_in_collision  : True if any sphere pair penetrates each other.
        """
        q_t = self._make_joint_state(q).position
        d_world, d_self = self._robot_world.get_world_self_collision_distance_from_joints(q_t)
        world_col = bool((d_world > 0.0).any().item())
        self_col  = bool((d_self  > 0.0).any().item())
        if verbose and (world_col or self_col):
            if world_col:
                mask = d_world > 0.0
                print(f"  d_world nonzero idx={mask.nonzero(as_tuple=False).squeeze(-1).tolist()}"
                      f"  values={d_world[mask].tolist()}")
            if self_col:
                mask = d_self > 0.0
                print(f"  d_self  nonzero idx={mask.nonzero(as_tuple=False).squeeze(-1).tolist()}"
                      f"  values={d_self[mask].tolist()}")
        return world_col, self_col

    def check_at_planning_distance(self, q: np.ndarray, label: str = "") -> bool:
        """
        Check whether q is valid from the planner's perspective (using the planner's
        collision_activation_distance, not the 0.0 used by is_in_collision).

        If this returns False, plan_to_joint will ALWAYS fail regardless of other settings,
        because the start/goal itself is considered "in collision" by MotionGen.

        Returns True if the config is valid for planning.
        """
        js = self._make_joint_state(q)
        valid, status = self._motion_gen.check_start_state(js)
        tag = "[VALID@plan_dist]  " if valid else "[INVALID@plan_dist]"
        suffix = f" {label}" if label else ""
        print(f"{tag}{suffix}: {status}  (activation_distance={self._collision_activation_distance}m)")
        return bool(valid)

    # ------------------------------------------------------------------
    # Dynamic world updates
    # ------------------------------------------------------------------

    def update_world(
        self,
        cuboids: list[dict] | None = None,
    ) -> None:
        """
        Replace the current world obstacles with a new set of cuboids.

        Args:
            cuboids: List of obstacle dicts, each with:
                       "name" : str
                       "pose" : [x, y, z, qw, qx, qy, qz]
                       "dims" : [size_x, size_y, size_z]

        Example:
            planner.update_world([
                {"name": "box",   "pose": [0.0, 0.0, 1.0, 1, 0, 0, 0], "dims": [0.1, 0.1, 0.3]},
                {"name": "table", "pose": [0.0, 0.0, 0.89, 1, 0, 0, 0], "dims": [1.8, 0.6, 0.04]},
            ])
        """

        cuboid_objs = [
            Cuboid(name=c["name"], pose=c["pose"], dims=c["dims"])
            for c in (cuboids or [])
        ]
        self._motion_gen.update_world(WorldConfig(cuboid=cuboid_objs))

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

        # Combine robot link meshes with static world obstacles
        scene = WorldConfig(mesh=robot_meshes[:])
        for obj in self.world_cfg.objects:
            scene.add_obstacle(obj)

        # table_top uses a <box> primitive in the URDF (no mesh file), so it is not
        # captured by get_robot_as_mesh. Add it explicitly as a cuboid for visualization.
        # Dimensions and pose must match add_table_top() in generate_bimanual_panda_tesollo.py:
        #   box: 1.8288 × 0.62865 × 0.045 m, top face at z=0 → centre at z=-0.0225
        scene.add_obstacle(Cuboid(
            name="table_top_visual",
            pose=[0.0, 0.0, -0.0225, 1.0, 0.0, 0.0, 0.0],
            dims=[1.8288, 0.62865, 0.045],
        ))

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


    def test_curobo():
        
        print("\n" + "=" * 60)
        print("Curobo smoke test")
        print("=" * 60)
        
        print("\nInitializing CuRoboBimanualMotionPlanner (warmup may take ~30 s)...")
        planner = CuRoboBimanualMotionPlanner(
            robot_cfg_path              = DEFAULT_CUROBO_ROBOT_CFG_PATH,
            world_cfg_path              = DEFAULT_CUROBO_WORLD_CFG_PATH,
            left_ee_link                = "left_delto_base_link",
            right_ee_link               = "right_delto_base_link",
            joint_names                 = BIMANUAL_JOINT_NAMES,
            device                      = "cuda:0",
            trajopt_tsteps              = 64,
            interpolation_steps         = 2000,
            num_ik_seeds                = 50,
            num_trajopt_seeds           = 32,   # was 12; 38-DoF bimanual needs significantly more seeds
            grad_trajopt_iters          = 800,  # was 800
            interpolation_dt            = 0.02,
            collision_activation_distance = 0.005,  # was 0.005; increased to provide better gradients for optimizer
        )


        def _check_and_print(name, q):
            world_col, self_col = planner.is_in_collision(q, verbose=True)
            if world_col or self_col:
                print(f"[COLLISION] {name}  world={world_col}  self={self_col}")
            else:
                print(f"[OK] {name} is collision-free.")

        print("\nChecking HOME_Q for collision...")
        t_start = time.time()
        _check_and_print("HOME_Q", HOME_Q)
        t_end = time.time()
        print(f"Collision check took {(t_end - t_start):.6f} seconds.")
        
        print("\nSaving HOME_Q scene mesh for debug visualization...")
        t_start = time.time()
        planner.save_scene_as_mesh(HOME_Q, str((_LIBS_ROOT.parent / "models" / "robot_world_home_q.stl").resolve()))
        t_end = time.time()
        print(f"Scene mesh saved in {(t_end - t_start):.6f} seconds.")


        print("\nChecking PRE_GRASP_Q for collision...")
        t_start = time.time()
        _check_and_print("PRE_GRASP_Q", PRE_GRASP_Q)
        t_end = time.time()
        print(f"Collision check took {(t_end - t_start):.6f} seconds.")
        
        print("\nSaving pre-grasp scene mesh for debug visualization...")
        t_start = time.time()
        planner.save_scene_as_mesh(PRE_GRASP_Q, str((_LIBS_ROOT.parent / "models" / "robot_world_pregrasp_q.stl").resolve()))
        t_end = time.time()
        print(f"Scene mesh saved in {(t_end - t_start):.6f} seconds.")
            
        print("\nChecking endpoint validity at planner's activation distance...")
        home_ok     = planner.check_at_planning_distance(HOME_Q,     "HOME_Q")
        pregrasp_ok = planner.check_at_planning_distance(PRE_GRASP_Q, "PRE_GRASP_Q")
        if not home_ok or not pregrasp_ok:
            print("[WARN] One or both endpoints are invalid at the planner's activation distance.")
            print("       Reduce collision_activation_distance or adjust the joint configuration.")

        print(f"\nPlanning trajectory from HOME_Q to PRE_GRASP_Q...")
        t_start = time.time()
        traj, ok, status = planner.plan_to_joint(HOME_Q, PRE_GRASP_Q)
        if ok:
            print(
                f"[OK] Trajectory: {traj.shape[0]} steps x {traj.shape[1]} DoF  "
                f"(dt={planner._interpolation_dt:.3f}s, "
                f"~{traj.shape[0] * planner._interpolation_dt:.1f}s total)"
            )
            print(f"     Start : {traj[0]}")
            print(f"     End   : {traj[-1]}")
        else:
            print(f"[FAIL] Planning failed: {status}")
        t_end = time.time()
        print(f"Planning took {(t_end - t_start):.6f} seconds.")

        # ------------------------------------------------------------------
        # IK smoke test — 3 solve_ik calls
        # ------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("IK smoke test  (3 batches)")
        print("-" * 60)

        _N_PER_CHAIN = 19   # 7 arm + 12 finger joints per chain

        # Pinocchio FK to compute ground-truth EE poses at known joint configs.
        fk = PinocchioBimanualFK(
            urdf_path=DEFAULT_BIMANUAL_URDF_PATH,
            action_per_chain=_N_PER_CHAIN,
            left_fingertip_names =["left_F1_TIP_TOP",  "left_F2_TIP_TOP",  "left_F3_TIP_TOP"],
            right_fingertip_names=["right_F1_TIP_TOP", "right_F2_TIP_TOP", "right_F3_TIP_TOP"],
            left_hand_base_names =["left_delto_base_link"],
            right_hand_base_names=["right_delto_base_link"],
        )

        def _fk_ee(q_full):
            """Return (l_pos, l_quat, r_pos, r_quat) at q_full via Pinocchio."""
            lb, _, rb, _ = fk.forward(q_full[:_N_PER_CHAIN], q_full[_N_PER_CHAIN:])
            return lb[0, :3], lb[0, 3:], rb[0, :3], rb[0, 3:]   # all wxyz

        # Test case 1 — FK at HOME_Q  (arm in "ready" position)
        # Test case 2 — FK at PRE_GRASP_Q  (known near-goal config)
        # Test case 3 — FK at PRE_GRASP_Q + small random noise  (nearby reachable target)
        from scipy.spatial.transform import Rotation as _R
        rng = np.random.default_rng(7)
        noise_pos  = rng.uniform(-0.02, 0.02, 3).astype(np.float32)
        axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
        dq   = _R.from_rotvec(axis * np.deg2rad(5.0)).as_quat()       # xyzw
        dq_w = np.array([dq[3], dq[0], dq[1], dq[2]], dtype=np.float32)  # wxyz

        def _qmul(a, b):   # wxyz quaternion product
            aw, ax, ay, az = a;  bw, bx, by, bz = b
            return np.array([aw*bw-ax*bx-ay*by-az*bz,
                             aw*bx+ax*bw+ay*bz-az*by,
                             aw*by-ax*bz+ay*bw+az*bx,
                             aw*bz+ax*by-ay*bx+az*bw], dtype=np.float32)

        lp_h, lq_h, rp_h, rq_h = _fk_ee(HOME_Q)
        lp_p, lq_p, rp_p, rq_p = _fk_ee(PRE_GRASP_Q)

        test_cases = [
            ("HOME_Q    → IK",
             lp_h, lq_h, rp_h, rq_h),
            ("PRE_GRASP → IK",
             lp_p, lq_p, rp_p, rq_p),
            ("PRE_GRASP + noise → IK",
             lp_p + noise_pos, _qmul(dq_w, lq_p),
             rp_p + noise_pos * 0.5, _qmul(dq_w, rq_p)),
        ]

        for idx, (label, lp, lq, rp, rq) in enumerate(test_cases):
            print(f"\n  [IK test {idx+1}/3]  {label}")
            print(f"    left  target  pos={np.round(lp, 3)}  quat(wxyz)={np.round(lq, 3)}")
            print(f"    right target  pos={np.round(rp, 3)}  quat(wxyz)={np.round(rq, 3)}")
            t_start = time.time()
            q_sols, succ = planner.solve_ik(
                left_target_pos=lp,   left_target_quat=lq,
                right_target_pos=rp,  right_target_quat=rq,
            )
            t_end = time.time()
            print(f"    solve_ik took {(t_end - t_start):.3f} s")
            if succ[0]:
                q_sol = q_sols[0]
                # Verify via FK: position error at the IK solution
                lb2, _, rb2, _ = fk.forward(q_sol[:_N_PER_CHAIN], q_sol[_N_PER_CHAIN:])
                l_err = np.linalg.norm(lb2[0, :3] - lp)
                r_err = np.linalg.norm(rb2[0, :3] - rp)
                print(f"    [OK] left pos-err={l_err*1000:.1f} mm   right pos-err={r_err*1000:.1f} mm")
                print(f"    q_sol arm-left  = {np.round(q_sol[:7], 3)}")
                print(f"    q_sol arm-right = {np.round(q_sol[19:26], 3)}")
                _check_and_print(f"IK_sol_{idx+1}", q_sol)
            else:
                print(f"    [FAIL] IK did not converge for this target.")

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