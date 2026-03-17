"""38-DOF full-chain IK solver for fingertip retargeting using cuRobo.

Solves 6 fingertip position targets simultaneously for bimanual Panda + Tesollo
robot. Uses warm-start from previous frame for trajectory continuity.

Must run inside handrl-policy Docker container (cuRobo + CUDA required).

Usage:
    solver = CuRoboRetargetIK()
    q, ok = solver.solve({"left_F1_TIP": pos1, "left_F2_TIP": pos2, ...})
    q, ok = solver.solve(tips, q_prev=q)  # warm-start from previous frame
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.cuda_robot_model.cuda_robot_model import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]  # dexterity-interface-rl/
_ROBOT_DESC = _REPO_ROOT / "libs" / "robot_description"

DEFAULT_ROBOT_CFG = str(_ROBOT_DESC / "configs_curobo" / "robot" / "bimanual_panda_tesollo-retargeting.yml")
DEFAULT_URDF = str(_ROBOT_DESC / "rl" / "bimanual_panda_tesollo.urdf")
DEFAULT_SPHERES = str(_ROBOT_DESC / "configs_curobo" / "robot" / "spheres" / "bimanual_panda_tesollo_spheres.yml")
# gradient_file is resolved relative to cuRobo's content/configs/task/ directory
DEFAULT_GRADIENT_CFG = "gradient_ik_retargeting.yml"

# ee_link is left_F1_TIP; remaining 5 fingertips are secondary targets
_SECONDARY_TIPS = ["left_F2_TIP", "left_F3_TIP", "right_F1_TIP", "right_F2_TIP", "right_F3_TIP"]
_ALL_TIPS = ["left_F1_TIP"] + _SECONDARY_TIPS

# Dummy quaternion for position-only IK (rotation_weight=0 so value doesn't matter)
_DUMMY_QUAT = [1.0, 0.0, 0.0, 0.0]


class CuRoboRetargetIK:
    """38-DOF full-chain IK for fingertip position retargeting."""

    def __init__(
        self,
        robot_cfg_path: str = DEFAULT_ROBOT_CFG,
        urdf_path: str = DEFAULT_URDF,
        spheres_path: str = DEFAULT_SPHERES,
        gradient_cfg_path: str = DEFAULT_GRADIENT_CFG,
        base_cfg_path: str = "base_cfg_retargeting.yml",
        num_seeds: int = 32,
        device: str = "cuda:0",
    ) -> None:
        self._device = device
        self.tensor_args = TensorDeviceType(device=torch.device(device))

        # Load robot config
        robot_cfg_dict = load_yaml(robot_cfg_path)
        robot_cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path
        robot_cfg_dict["robot_cfg"]["kinematics"]["collision_spheres"] = spheres_path
        robot_cfg = RobotConfig.from_dict(robot_cfg_dict, self.tensor_args)

        # Joint names from config (38 DOF)
        self._joint_names: list[str] = list(
            robot_cfg_dict["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
        )
        self._retract_config: list[float] = list(
            robot_cfg_dict["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
        )
        self._n_dof = len(self._joint_names)

        # Create IK solver with position-only gradient + base config
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=None,
            tensor_args=self.tensor_args,
            num_seeds=num_seeds,
            self_collision_check=True,
            self_collision_opt=True,
            use_cuda_graph=True,
            gradient_file=gradient_cfg_path,
            base_cfg_file=base_cfg_path,
            regularization=True,
        )
        self._ik_solver = IKSolver(ik_config)

        # Get internal joint ordering for reorder
        self._internal_joint_names: list[str] = list(self._ik_solver.kinematics.joint_names)

        # Build reorder map: config order -> internal order and back
        self._cfg_to_internal = [
            self._internal_joint_names.index(n) for n in self._joint_names
        ]
        self._internal_to_cfg = [
            self._joint_names.index(n) for n in self._internal_joint_names
        ]

        # Warmup CUDA graphs
        self._warmup()
        print(f"[CuRoboRetargetIK] Ready: {self._n_dof} DOF, {num_seeds} seeds, device={device}")

    def _warmup(self) -> None:
        """Run a dummy solve to trigger CUDA graph compilation.

        Must include seed_config and retract_config to match production calls,
        since CUDA graph capture is fixed to the first call's signature.
        """
        dummy_pos = torch.zeros(1, 3, device=self._device, dtype=torch.float32)
        dummy_quat = torch.tensor([_DUMMY_QUAT], device=self._device, dtype=torch.float32)
        goal = Pose(position=dummy_pos, quaternion=dummy_quat)
        link_poses = {
            name: Pose(position=dummy_pos, quaternion=dummy_quat)
            for name in _SECONDARY_TIPS
        }
        # Provide seed/retract so CUDA graph captures the full call signature
        retract = torch.tensor(
            [self._retract_config], dtype=torch.float32, device=self._device
        )
        retract_internal = retract[:, self._cfg_to_internal]
        seed = retract_internal.unsqueeze(0)  # (1, 1, dof)
        self._ik_solver.solve_single(
            goal,
            retract_config=retract_internal,
            seed_config=seed,
            link_poses=link_poses,
        )
        print("[CuRoboRetargetIK] CUDA warmup done.")

    def _make_pose(self, position: np.ndarray) -> Pose:
        """Create a Pose from a (3,) position array with dummy quaternion."""
        pos_t = torch.tensor(position, dtype=torch.float32, device=self._device).unsqueeze(0)
        quat_t = torch.tensor([_DUMMY_QUAT], dtype=torch.float32, device=self._device)
        return Pose(position=pos_t, quaternion=quat_t)

    def _q_to_internal(self, q: np.ndarray) -> torch.Tensor:
        """Reorder (n_dof,) config-order array to internal cuRobo order."""
        q_internal = q[self._cfg_to_internal]
        return torch.tensor(q_internal, dtype=torch.float32, device=self._device)

    def _q_to_config(self, q_internal: torch.Tensor) -> np.ndarray:
        """Reorder internal cuRobo order tensor to (n_dof,) config-order array."""
        q_np = q_internal.cpu().numpy()
        return q_np[self._internal_to_cfg]

    def solve(
        self,
        fingertip_positions: dict[str, np.ndarray],
        q_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Solve full-chain IK for 6 fingertip position targets.

        Args:
            fingertip_positions: {link_name: (3,) xyz in world frame} for all 6 fingertips.
                Required keys: left_F1_TIP, left_F2_TIP, left_F3_TIP,
                               right_F1_TIP, right_F2_TIP, right_F3_TIP
            q_prev: (38,) previous frame joint angles for warm-start.
                If None, uses retract_config.

        Returns:
            q: (38,) joint angles in config order.
            success: True if IK converged.
        """
        # Primary goal: left_F1_TIP
        goal_pose = self._make_pose(fingertip_positions["left_F1_TIP"])

        # Secondary targets
        link_poses = {
            name: self._make_pose(fingertip_positions[name])
            for name in _SECONDARY_TIPS
        }

        # Warm-start: always provide seed/retract to keep CUDA graph signature consistent
        if q_prev is not None:
            q_internal = self._q_to_internal(q_prev)
        else:
            q_internal = torch.tensor(
                self._retract_config, dtype=torch.float32, device=self._device
            )[self._cfg_to_internal]
        seed_config = q_internal.unsqueeze(0).unsqueeze(0)   # (1, 1, dof)
        retract_config = q_internal.unsqueeze(0)              # (1, dof)

        result = self._ik_solver.solve_single(
            goal_pose,
            retract_config=retract_config,
            seed_config=seed_config,
            link_poses=link_poses,
        )

        if not result.success.any():
            # Return retract or previous config on failure
            q_fallback = q_prev if q_prev is not None else np.array(self._retract_config, dtype=np.float32)
            return q_fallback, False

        # Extract solution in internal order, convert to config order
        q_solution_internal = result.solution.squeeze()  # (dof,)
        q_config = self._q_to_config(q_solution_internal)

        return q_config.astype(np.float32), True

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def retract_config(self) -> np.ndarray:
        return np.array(self._retract_config, dtype=np.float32)

    @property
    def n_dof(self) -> int:
        return self._n_dof


if __name__ == "__main__":
    # Quick sanity test: use FK from retract config to get reachable tip positions
    solver = CuRoboRetargetIK()
    print(f"Joint names ({solver.n_dof}): {solver.joint_names[:7]} ...")

    # Compute FK at retract config to get reachable fingertip positions
    q_retract = solver.retract_config
    q_internal = solver._q_to_internal(q_retract).unsqueeze(0)
    kin_state = solver._ik_solver.kinematics.get_state(q_internal)

    # ee_link position (left_F1_TIP)
    ee_pos = kin_state.ee_position.squeeze().cpu().numpy()
    print(f"FK retract -> ee_link (left_F1_TIP): {ee_pos}")

    # Inspect link_pose type/shape
    link_names = solver._ik_solver.kinematics.link_names
    print(f"Tracked links: {link_names}")
    print(f"link_pose type: {type(kin_state.link_pose)}")
    if isinstance(kin_state.link_pose, dict):
        for k, v in kin_state.link_pose.items():
            print(f"  {k}: type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
    elif hasattr(kin_state.link_pose, 'shape'):
        print(f"  shape: {kin_state.link_pose.shape}")

    # Try to get link positions
    tips = {"left_F1_TIP": ee_pos}
    link_state = kin_state.link_pose
    if isinstance(link_state, dict):
        for ln in _SECONDARY_TIPS:
            pose = link_state[ln]
            pos = pose.position.squeeze().cpu().numpy() if hasattr(pose, 'position') else pose[:3].cpu().numpy()
            tips[ln] = pos
            print(f"  {ln}: {pos}")
    else:
        # Assume tensor (batch, n_links, 7) or similar
        for idx, ln in enumerate(link_names):
            if ln in _SECONDARY_TIPS:
                pos = link_state[..., idx, :3].squeeze().cpu().numpy()
                tips[ln] = pos
                print(f"  {ln}: {pos}")

    print(f"\nSolving IK with FK-derived targets (should be trivial)...")
    q, ok = solver.solve(tips)
    print(f"IK success: {ok}")
    if ok:
        delta = np.abs(q - q_retract)
        print(f"Max joint delta from retract: {delta.max():.6f} rad")

    # Perturb slightly and test warm-start
    if ok:
        tips2 = {k: v + np.array([0.005, 0, 0]) for k, v in tips.items()}
        q2, ok2 = solver.solve(tips2, q_prev=q)
        print(f"\nWarm-start IK with +5mm X offset: success={ok2}")
        if ok2:
            delta2 = np.abs(q2 - q)
            print(f"Max joint delta: {delta2.max():.4f} rad, mean: {delta2.mean():.4f} rad")
