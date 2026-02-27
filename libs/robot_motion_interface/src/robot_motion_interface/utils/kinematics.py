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

import pinocchio as pin

import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState

from curobo.geom.types import Cuboid, WorldConfig

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# ---------------------------------------------------------------------------
# Project-relative path resolution
#   __file__ = .../libs/robot_motion_interface/src/robot_motion_interface/utils/kinematics.py
#   parents[4] = .../libs/
# ---------------------------------------------------------------------------
_LIBS_ROOT = Path(__file__).parents[4]
_ROBOT_DESC = _LIBS_ROOT / "robot_description"

DEFAULT_BIMANUAL_URDF_PATH      = str((_ROBOT_DESC / "rl/bimanual_panda_tesollo.urdf").resolve())
DEFAULT_CUROBO_ROBOT_CFG_PATH   = str((_ROBOT_DESC / "configs_curobo/robot/bimanual_panda_tesollo.yml").resolve())
DEFAULT_CUROBO_WORLD_CFG_PATH   = str((_ROBOT_DESC / "configs_curobo/world/bimanual_table.yml").resolve())


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
        l_base, r_base, l_tips, r_tips = fk.forward(q_left, q_right)
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

        self.model = pin.buildModelFromUrdf(urdf_path)
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

    def forward(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pin = self._pin
        q = np.concatenate([q_left, q_right]).astype(np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        l_hand_base  = np.array([self.data.oMf[i].translation for i in self.l_hand_base_ids],  dtype=np.float32)
        l_fingertips = np.array([self.data.oMf[i].translation for i in self.l_fingertip_ids],  dtype=np.float32)
        r_hand_base  = np.array([self.data.oMf[i].translation for i in self.r_hand_base_ids],  dtype=np.float32)
        r_fingertips = np.array([self.data.oMf[i].translation for i in self.r_fingertip_ids],  dtype=np.float32)

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
            robot_cfg_path="robot/robot_description/configs/robot/bimanual_panda_tesollo.yml",
            world_cfg_path="robot/robot_description/configs/world/bimanual_table.yml",
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
        collision_activation_distance: float = 0.025,
    ) -> None:        

        self._device = device
        self._left_ee_link  = left_ee_link
        self._right_ee_link = right_ee_link
        self._joint_names   = joint_names if joint_names is not None else BIMANUAL_JOINT_NAMES
        self._interpolation_dt = interpolation_dt

        tensor_args = TensorDeviceType(device=torch.device(device))

        # Motion planner (trajectory generation)
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg=robot_cfg_path,
            world_cfg=world_cfg_path,
            tensor_args=tensor_args,
            trajopt_tsteps=trajopt_tsteps,
            interpolation_steps=interpolation_steps,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
            grad_trajopt_iters=grad_trajopt_iters,
            interpolation_dt=interpolation_dt,
            collision_activation_distance=collision_activation_distance,
            evaluate_interpolated_trajectory=True,
        )
        self._motion_gen = MotionGen(motion_gen_cfg)
        self._motion_gen.warmup()
        self._motion_gen.world_coll_checker.clear_cache()
        self._motion_gen.reset(reset_seed=False)

        # Standalone collision checker (same robot + world, no motion planning overhead)
        robot_world_cfg = RobotWorldConfig.load_from_config(
            robot_cfg_path,
            world_cfg_path,
            collision_activation_distance=0.0,  # 0.0 = report actual penetration depth
        )
        self._robot_world = RobotWorld(robot_world_cfg)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_joint_state(self, q: np.ndarray) -> "JointState":
        return JointState.from_position(
            torch.tensor(q, dtype=torch.float32, device=self._device).unsqueeze(0),
            joint_names=self._joint_names,
        )

    def _make_pose(self, pos: np.ndarray, quat: np.ndarray) -> "Pose":
        return Pose(
            position=torch.tensor(pos,  dtype=torch.float32, device=self._device).unsqueeze(0),
            quaternion=torch.tensor(quat, dtype=torch.float32, device=self._device).unsqueeze(0),
        )

    def _extract_trajectory(self, result) -> tuple[np.ndarray | None, bool, str]:
        if result.success.item():
            # get_full_js fills in any joints cuRobo omitted from planning
            traj = self._motion_gen.get_full_js(result.get_interpolated_plan())
            return traj.position.cpu().numpy(), True, str(result.status)
        return None, False, str(result.status)

    def _make_plan_config(
        self,
        max_attempts: int,
        timeout: float,
        time_dilation_factor: float,
    ) -> "MotionGenPlanConfig":
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
            self._make_pose(left_target_pos, left_target_quat),
            self._make_plan_config(max_attempts, timeout, time_dilation_factor),
            link_poses={self._right_ee_link: self._make_pose(right_target_pos, right_target_quat)},
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
    # Collision checking
    # ------------------------------------------------------------------

    def is_in_collision(
        self,
        q: np.ndarray,  # (n_joints,)
    ) -> tuple[bool, bool]:
        """
        Check whether a joint configuration is in collision.

        Uses collision_activation_distance=0.0, so returns True only on
        actual penetration (distance <= 0). Increase the threshold in
        __init__ for a safety margin.

        Returns:
            world_in_collision : True if any link sphere penetrates a world obstacle.
            self_in_collision  : True if any link-sphere pair penetrates each other.
        """

        q_t = torch.tensor(q, dtype=torch.float32, device=self._device).unsqueeze(0)
        d_world, d_self = self._robot_world.get_world_self_collision_distance_from_joints(q_t)
        return bool((d_world <= 0.0).any().item()), bool((d_self <= 0.0).any().item())

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

    print("Initializing CuRoboBimanualMotionPlanner (warmup may take ~30 s)...")
    planner = CuRoboBimanualMotionPlanner(
        robot_cfg_path              = DEFAULT_CUROBO_ROBOT_CFG_PATH,
        world_cfg_path              = DEFAULT_CUROBO_WORLD_CFG_PATH,
        left_ee_link                = "left_delto_base_link",
        right_ee_link               = "right_delto_base_link",
        joint_names                 = BIMANUAL_JOINT_NAMES,
        device                      = "cuda:0",
        trajopt_tsteps              = 34,
        interpolation_steps         = 2000,
        num_ik_seeds                = 30,
        num_trajopt_seeds           = 4,
        grad_trajopt_iters          = 500,
        interpolation_dt            = 0.02,
        collision_activation_distance = 0.025,
    )

    print("Checking PRE_GRASP_Q for collision...")
    world_col, self_col = planner.is_in_collision(PRE_GRASP_Q)
    if world_col or self_col:
        print(f"[ABORT] PRE_GRASP_Q is in collision (world={world_col}, self={self_col}).")
    else:
        print("PRE_GRASP_Q is collision-free. Planning HOME_Q → PRE_GRASP_Q...")
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