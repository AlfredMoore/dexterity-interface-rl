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
        urdf_path: str = DEFAULT_BIMANUAL_URDF_PATH,
        spheres_path: str = DEFAULT_COLLISION_SPHERES_PATH,
        left_ee_link: str = "left_delto_base_link",
        right_ee_link: str = "right_delto_base_link",
        joint_names: list[str] | None = None,
        device: str = "cuda:0",
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
        self._collision_activation_distance = collision_activation_distance

        self.tensor_args = TensorDeviceType(device=torch.device(device))

        robot_cfg_dict = load_yaml(robot_cfg_path)
        robot_cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path
        robot_cfg_dict["robot_cfg"]["kinematics"]["collision_spheres"] = spheres_path
        self.robot_cfg = RobotConfig.from_dict(robot_cfg_dict, self.tensor_args)

        # Trajectory optimizer (no world model, self-collision only)
        trajopt_config = TrajOptSolverConfig.load_from_robot_config(
            robot_cfg=self.robot_cfg,
            world_model=None,
            tensor_args=self.tensor_args,
            self_collision_check=True,
            self_collision_opt=True,
            traj_tsteps=trajopt_tsteps,
            interpolation_steps=interpolation_steps,
            num_seeds=num_trajopt_seeds,
            grad_trajopt_iters=grad_trajopt_iters,
            interpolation_dt=interpolation_dt,
            collision_activation_distance=collision_activation_distance,
            evaluate_interpolated_trajectory=True,
            use_cuda_graph=True,
        )
        self._trajopt = TrajOptSolver(trajopt_config)

        # IK solver (for plan_to_pose: EE poses → joint config)
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg=self.robot_cfg,
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

    def _extract_trajectory(self, result) -> tuple[np.ndarray | None, bool, str]:
        if result.success.any().item():
            traj = result.interpolated_solution
            if traj is None:
                traj = result.solution
            if traj.joint_names is not None:
                traj = traj.get_ordered_joint_state(self._joint_names)
            pos = traj.position
            if pos.dim() == 3:
                pos = pos[0]  # [T, dof]
            return pos.cpu().numpy(), True, "success"
        return None, False, "trajopt_failed"

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
        Solve bimanual IK for given left and right EE poses simultaneously.

        Left EE is the primary goal; right EE is enforced via link_poses constraint.
        Both are solved jointly in a single optimization pass.

        Returns:
            q_solution : (n_joints,) float32 tensor on self._device, in self._joint_names order,
                         or None if IK failed.
            success    : True if a valid solution was found.
        """
        left_pose  = self._make_pose(left_pos,  left_quat)
        right_pose = self._make_pose(right_pos, right_quat)
        result = self._ik_solver.solve_single(
            left_pose,
            link_poses={self._right_ee_link: right_pose},
        )
        if not result.success.any():
            return None, False
        idx = result.success.flatten().nonzero(as_tuple=False)[0, 0].item()
        q_js = JointState.from_position(
            result.js_solution.position[idx].unsqueeze(0),
            joint_names=self._internal_joint_names,
        ).get_ordered_joint_state(self._joint_names)
        return q_js.position.squeeze(0), True

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
    ) -> tuple[np.ndarray | None, bool, str]:
        """
        Plan a self-collision-free trajectory to the given dual-arm Cartesian EE poses.

        Calls solve_ik to obtain q_goal, then runs TrajOpt joint-to-joint.

        Returns:
            trajectory  : (T, n_joints) float32 numpy array, or None if planning failed.
                          T waypoints spaced interpolation_dt seconds apart.
            success     : True if a valid trajectory was found.
            status      : Human-readable status string.
        """
        q_goal, ik_ok = self.solve_ik(
            left_target_pos, left_target_quat,
            right_target_pos, right_target_quat,
        )
        if not ik_ok:
            return None, False, "ik_failed"
        return self.plan_to_joint(q_start, q_goal)

    # ------------------------------------------------------------------
    # Trajectory planning — joint space
    # ------------------------------------------------------------------

    def plan_to_joint(
        self,
        q_start: np.ndarray,    # (n_joints,)
        q_goal: np.ndarray,     # (n_joints,)
    ) -> tuple[np.ndarray | None, bool, str]:
        """
        Plan a self-collision-free trajectory to the given target joint configuration.

        Returns:
            trajectory  : (T, n_joints) float32 numpy array, or None if planning failed.
                          T waypoints spaced interpolation_dt seconds apart.
            success     : True if a valid trajectory was found.
            status      : Human-readable status string.
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
        Check whether q is free of self-collision within the planner's activation distance.

        Uses a separate RobotWorld instance with self_collision_activation_distance set to
        collision_activation_distance, so the result matches what TrajOpt considers valid.

        Returns True if the config is valid for planning.
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


    def test_curobo():
        
        print("\n" + "=" * 60)
        print("Curobo smoke test")
        print("=" * 60)
        
        print("\nInitializing CuRoboBimanualMotionPlanner (warmup may take ~30 s)...")
        planner = CuRoboBimanualMotionPlanner(
            robot_cfg_path              = DEFAULT_CUROBO_ROBOT_CFG_PATH,
            left_ee_link                = "left_delto_base_link",
            right_ee_link               = "right_delto_base_link",
            joint_names                 = BIMANUAL_JOINT_NAMES,
            device                      = "cuda:0",
            trajopt_tsteps              = 64,
            interpolation_steps         = 2000,
            num_ik_seeds                = 50,
            num_trajopt_seeds           = 32,
            grad_trajopt_iters          = 800,
            interpolation_dt            = 0.02,
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
            
        _collision_check_test("HOME_Q", HOME_Q)
        planner.save_scene_as_mesh(HOME_Q, str((_LIBS_ROOT.parent / "models" / "robot_world_home_q.stl").resolve()))
        print(f"[INFO] Saved HOME_Q scene mesh to {str((_LIBS_ROOT.parent / 'models' / 'robot_world_home_q.stl').resolve())}")

        _collision_check_test("PRE_GRASP_Q", PRE_GRASP_Q)
        planner.save_scene_as_mesh(PRE_GRASP_Q, str((_LIBS_ROOT.parent / "models" / "robot_world_pregrasp_q.stl").resolve()))
        print(f"[INFO] Saved PRE_GRASP_Q scene mesh to {str((_LIBS_ROOT.parent / 'models' / 'robot_world_pregrasp_q.stl').resolve())}")

        # ------------------------------------------------------------------
        # Traj Optimization test
        # ------------------------------------------------------------------
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
        # IK solver test
        # ------------------------------------------------------------------
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