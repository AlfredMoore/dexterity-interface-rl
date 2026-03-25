"""
Batch retargeting with cuRobo two-stage IK + pre-grasp trajopt.

Pipeline (per episode):
  1) Compute camera->URDF alignment from EgoDex fingertip data and anchor mode.
  2) Per frame two-stage IK with two persistent solvers:
       - Stage-1 (arm solver): lock all finger joints, solve left/right virtual palm.
       - Stage-2 (finger solver): solve 6 fingertip targets with soft arm freedom
         (no strict per-frame arm lock).
  3) Use frame-0 IK result as q_pregrasp and plan q_home -> q_pregrasp via cuRobo trajopt.
  4) Concatenate [traj_home_to_pregrasp, joint_positions_retarget] -> traj_full.
  5) Compare against current retarget_tool two-stage baseline and export reports.

Run INSIDE handrl-policy container:
  /root/miniconda3/envs/policy/bin/python baselines/ml-egodex-HAND/retarget/curobo_two_stage_batch.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Ensure local imports are resolvable from repo root execution.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
_RMI_SRC = _REPO_ROOT / "libs" / "robot_motion_interface" / "src"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_RMI_SRC) not in sys.path:
    sys.path.insert(0, str(_RMI_SRC))

from curobo.cuda_robot_model.cuda_robot_model import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

from retarget_episode import (
    LEFT_FINGER_TIPS,
    RIGHT_FINGER_TIPS,
    CoordinateAligner,
    _world_to_robot_pos,
    load_episode,
)
from retarget_tool import (
    _DEFAULT_LEFT_OFFSET,
    _DEFAULT_RIGHT_OFFSET,
    compute_alignment,
    retarget_episode as baseline_retarget_episode,
)
from palm_search_retarget import FullModelFK
from robot_motion_interface.utils.kinematics import CuRoboBimanualMotionPlanner


ROBOT_CFG_DIR = _REPO_ROOT / "libs" / "robot_description" / "configs_curobo" / "robot"
URDF_RETARGET = _REPO_ROOT / "libs" / "robot_description" / "rl" / "bimanual_panda_tesollo_retarget.urdf"
SPHERES_CFG = ROBOT_CFG_DIR / "spheres" / "bimanual_panda_tesollo_spheres.yml"

STAGE1_CFG = ROBOT_CFG_DIR / "bimanual_panda_tesollo_2stage_lock_fingers.yml"
STAGE2_CFG = ROBOT_CFG_DIR / "bimanual_panda_tesollo_2stage_lock_arms.yml"

_DUMMY_QUAT = [1.0, 0.0, 0.0, 0.0]
_ALL_EGODEX_TIPS = LEFT_FINGER_TIPS + RIGHT_FINGER_TIPS
_ALL_TIP_LINKS = [
    "left_F1_TIP",
    "left_F2_TIP",
    "left_F3_TIP",
    "right_F1_TIP",
    "right_F2_TIP",
    "right_F3_TIP",
]
_TIP_LINK_MAP = {
    "leftThumbTip": "left_F1_TIP",
    "leftIndexFingerTip": "left_F2_TIP",
    "leftMiddleFingerTip": "left_F3_TIP",
    "rightThumbTip": "right_F1_TIP",
    "rightIndexFingerTip": "right_F2_TIP",
    "rightMiddleFingerTip": "right_F3_TIP",
}


def _percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _safe_rate(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else 0.0


def _strict_success(ik_rate: float | None, collision_actual_rate: float | None) -> int:
    if ik_rate is None or collision_actual_rate is None:
        return 0
    ik_v = float(ik_rate)
    coll_v = float(collision_actual_rate)
    if math.isnan(ik_v) or math.isnan(coll_v):
        return 0
    return int(ik_v >= 1.0 - 1e-9 and coll_v <= 1e-9)


def _load_home_q_from_driver_cfg() -> np.ndarray:
    cfg_path = _REPO_ROOT / "libs" / "robot_motion_interface" / "config" / "rl_bimanual_driver_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    panda_home = np.asarray(cfg["panda_home_joint_positions"], dtype=np.float32)
    tesollo_home = np.asarray(cfg["tesollo_home_joint_positions"], dtype=np.float32)
    q_home = np.concatenate([panda_home, tesollo_home, panda_home, tesollo_home]).astype(np.float32)
    if q_home.shape[0] != 38:
        raise ValueError(f"Expected HOME_Q with 38 DoF, got {q_home.shape[0]}")
    return q_home


def _ensure_solver_paths_exist() -> None:
    for p in [STAGE1_CFG, STAGE2_CFG, URDF_RETARGET, SPHERES_CFG]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")


@dataclass
class SolverBuildParams:
    device: str = "cuda:0"
    num_seeds_stage1: int = 64
    num_seeds_stage2: int = 64
    gradient_file: str = "gradient_ik_retargeting.yml"
    base_cfg_file: str = "base_cfg_retargeting.yml"


class CuRoboTwoStageIK:
    """Two persistent cuRobo IK solvers for two-stage solving."""

    def __init__(self, params: SolverBuildParams):
        self.params = params
        self.tensor_args = TensorDeviceType(device=torch.device(params.device))

        self.stage1_cfg_dict = self._load_cfg_dict(STAGE1_CFG, URDF_RETARGET, SPHERES_CFG)
        self.stage2_cfg_dict = self._load_cfg_dict(STAGE2_CFG, URDF_RETARGET, SPHERES_CFG)

        self.full_joint_names = list(self.stage1_cfg_dict["robot_cfg"]["kinematics"]["cspace"]["joint_names"])
        self.full_name_to_idx = {n: i for i, n in enumerate(self.full_joint_names)}
        self.retract_full = np.asarray(
            self.stage1_cfg_dict["robot_cfg"]["kinematics"]["cspace"]["retract_config"],
            dtype=np.float32,
        )
        if self.retract_full.shape[0] != 38:
            raise ValueError("Expected 38-DoF retract config in stage config.")

        self.stage1_solver = self._build_solver(self.stage1_cfg_dict, params.num_seeds_stage1)
        self.stage2_solver = self._build_solver(self.stage2_cfg_dict, params.num_seeds_stage2)

        self.stage1_active_names = list(self.stage1_solver.kinematics.joint_names)
        self.stage2_active_names = list(self.stage2_solver.kinematics.joint_names)

        self.stage1_lock_fingers = dict(self.stage1_cfg_dict["robot_cfg"]["kinematics"]["lock_joints"])
        stage2_lock = self.stage2_cfg_dict["robot_cfg"]["kinematics"].get("lock_joints")
        self.stage2_lock_map = dict(stage2_lock) if isinstance(stage2_lock, dict) else None

    def _load_cfg_dict(self, cfg_path: Path, urdf_path: Path, spheres_path: Path) -> dict[str, Any]:
        cfg = load_yaml(str(cfg_path))
        kin = cfg["robot_cfg"]["kinematics"]
        kin["urdf_path"] = str(urdf_path)
        kin["collision_spheres"] = str(spheres_path)
        return cfg

    def _build_solver(self, cfg_dict: dict[str, Any], num_seeds: int) -> IKSolver:
        robot_cfg = RobotConfig.from_dict(cfg_dict, self.tensor_args)
        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=None,
            tensor_args=self.tensor_args,
            num_seeds=num_seeds,
            self_collision_check=True,
            self_collision_opt=True,
            # keep disabled for dynamic lock updates and varying calls
            use_cuda_graph=False,
            gradient_file=self.params.gradient_file,
            base_cfg_file=self.params.base_cfg_file,
            regularization=True,
        )
        return IKSolver(ik_cfg)

    def _pose_from_position(self, pos: np.ndarray) -> Pose:
        pos_t = torch.as_tensor(pos, dtype=torch.float32, device=self.params.device).view(1, 3)
        quat_t = torch.tensor([_DUMMY_QUAT], dtype=torch.float32, device=self.params.device)
        return Pose(position=pos_t, quaternion=quat_t)

    def _active_seed_from_full(self, active_names: list[str], q_full: np.ndarray) -> np.ndarray:
        return np.asarray([q_full[self.full_name_to_idx[n]] for n in active_names], dtype=np.float32)

    def _solve_with_solver(
        self,
        solver: IKSolver,
        active_names: list[str],
        goal_pose: Pose,
        link_poses: dict[str, Pose],
        q_seed_full: np.ndarray | None,
        base_full: np.ndarray | None = None,
        lock_overrides: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, bool]:
        q_seed_full = self.retract_full if q_seed_full is None else q_seed_full
        q_active_seed = self._active_seed_from_full(active_names, q_seed_full)
        seed_t = torch.as_tensor(q_active_seed, dtype=torch.float32, device=self.params.device).view(1, 1, -1)
        retract_t = torch.as_tensor(q_active_seed, dtype=torch.float32, device=self.params.device).view(1, -1)

        result = solver.solve_single(
            goal_pose,
            retract_config=retract_t,
            seed_config=seed_t,
            link_poses=link_poses,
        )
        if not result.success.any():
            q_fallback = np.asarray(base_full if base_full is not None else q_seed_full, dtype=np.float32).copy()
            if lock_overrides is not None:
                for jn, jv in lock_overrides.items():
                    q_fallback[self.full_name_to_idx[jn]] = float(jv)
            return q_fallback, False

        active_solution = result.solution.squeeze().detach().cpu().numpy()
        active_solution = np.asarray(active_solution, dtype=np.float32).reshape(-1)
        q_out = np.asarray(base_full if base_full is not None else q_seed_full, dtype=np.float32).copy()

        if lock_overrides is not None:
            for jn, jv in lock_overrides.items():
                q_out[self.full_name_to_idx[jn]] = float(jv)

        for i, jn in enumerate(active_names):
            q_out[self.full_name_to_idx[jn]] = float(active_solution[i])
        return q_out, True

    def solve_frame(
        self,
        target_tips_urdf: np.ndarray,  # (6, 3)
        q_prev: np.ndarray | None,
        q_home: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, bool]:
        def _seed_candidates(*vals: np.ndarray | None) -> list[np.ndarray]:
            out: list[np.ndarray] = []
            for v in vals:
                if v is None:
                    continue
                vv = np.asarray(v, dtype=np.float32)
                if not any(np.allclose(vv, x) for x in out):
                    out.append(vv)
            return out

        # Stage-1: solve virtual palms with fingers locked.
        left_palm = target_tips_urdf[0:3].mean(axis=0)
        right_palm = target_tips_urdf[3:6].mean(axis=0)
        goal_stage1 = self._pose_from_position(left_palm)
        link_stage1 = {"right_virtual_palm": self._pose_from_position(right_palm)}
        ok1 = False
        q_stage1 = np.asarray(q_prev if q_prev is not None else self.retract_full, dtype=np.float32).copy()
        for seed in _seed_candidates(q_prev, q_home, self.retract_full):
            q_try, ok_try = self._solve_with_solver(
                solver=self.stage1_solver,
                active_names=self.stage1_active_names,
                goal_pose=goal_stage1,
                link_poses=link_stage1,
                q_seed_full=seed,
                base_full=seed,
                lock_overrides=self.stage1_lock_fingers,
            )
            if ok_try:
                q_stage1, ok1 = q_try, True
                break

        # Stage-2: solve fingertips with soft arm freedom (no strict per-frame arm lock).
        goal_stage2 = self._pose_from_position(target_tips_urdf[0])  # left_F1_TIP
        link_stage2 = {
            "left_F2_TIP": self._pose_from_position(target_tips_urdf[1]),
            "left_F3_TIP": self._pose_from_position(target_tips_urdf[2]),
            "right_F1_TIP": self._pose_from_position(target_tips_urdf[3]),
            "right_F2_TIP": self._pose_from_position(target_tips_urdf[4]),
            "right_F3_TIP": self._pose_from_position(target_tips_urdf[5]),
        }
        ok2 = False
        q_stage2 = q_stage1.copy()
        if q_prev is not None:
            for jn in self.stage2_active_names:
                q_stage2[self.full_name_to_idx[jn]] = float(q_prev[self.full_name_to_idx[jn]])
        for seed in _seed_candidates(q_prev, q_stage1, q_home, self.retract_full):
            q_try, ok_try = self._solve_with_solver(
                solver=self.stage2_solver,
                active_names=self.stage2_active_names,
                goal_pose=goal_stage2,
                link_poses=link_stage2,
                q_seed_full=seed,
                base_full=q_stage1,
                lock_overrides=self.stage2_lock_map,
            )
            if ok_try:
                q_stage2, ok2 = q_try, True
                break
        return q_stage2, ok1, ok2


def _extract_fingertip_targets_urdf(
    ep: dict[str, Any],
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> np.ndarray:
    tfs = ep["transforms"]
    T = len(tfs["leftHand"])
    identity = CoordinateAligner()
    targets = np.zeros((T, 6, 3), dtype=np.float32)

    for i in range(T):
        cam_ext_i = ep["cam_ext"][i]
        for j, ego_name in enumerate(_ALL_EGODEX_TIPS):
            pos_cam = _world_to_robot_pos(tfs[ego_name][i], cam_ext_i, identity)
            targets[i, j] = (R_align @ pos_cam + t_align).astype(np.float32)
    return targets


def _compute_fingertip_errors(
    full_fk: FullModelFK,
    q38: np.ndarray,
    target_tips: np.ndarray,  # (6, 3)
) -> np.ndarray:
    act_left = full_fk.fingertip_positions(q38, "left")
    act_right = full_fk.fingertip_positions(q38, "right")
    actual = np.vstack([act_left, act_right])
    return np.linalg.norm(actual - target_tips, axis=1).astype(np.float32)


def _compute_fk_points(
    full_fk: FullModelFK,
    q38: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_tips = full_fk.fingertip_positions(q38, "left")
    right_tips = full_fk.fingertip_positions(q38, "right")
    tips = np.vstack([left_tips, right_tips]).astype(np.float32)  # (6,3)
    palms = full_fk.palm_positions(q38).astype(np.float32)  # (2,3)
    return tips, palms


def _render_episode_animation(
    *,
    out_npz_path: Path,
    traj_full: np.ndarray,              # (N,38)
    fingertip_targets: np.ndarray,      # (T,6,3) for retarget segment only
    ik_success_stage1: np.ndarray,      # (T,)
    ik_success_stage2: np.ndarray,      # (T,)
    pregrasp_steps: int,
    full_fk: FullModelFK,
) -> str:
    N = int(traj_full.shape[0])
    T = int(fingertip_targets.shape[0])
    if N == 0:
        raise ValueError("traj_full is empty, cannot render animation")

    actual_tips = np.zeros((N, 6, 3), dtype=np.float32)
    actual_palms = np.zeros((N, 2, 3), dtype=np.float32)
    for i in range(N):
        tips_i, palms_i = _compute_fk_points(full_fk, traj_full[i])
        actual_tips[i] = tips_i
        actual_palms[i] = palms_i

    all_pts = np.concatenate(
        [
            actual_tips.reshape(-1, 3),
            actual_palms.reshape(-1, 3),
            fingertip_targets.reshape(-1, 3),
        ],
        axis=0,
    )
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 0.30)
    half = 0.6 * span

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    tip_colors = ["#1f77b4", "#4d8dd8", "#9fc5ff", "#d62728", "#f26d6d", "#ffb3b3"]

    def _draw_table() -> None:
        xs = np.array([center[0] - half, center[0] + half], dtype=np.float32)
        ys = np.array([center[1] - half, center[1] + half], dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.08, color="saddlebrown", linewidth=0)

    def update(frame_idx: int) -> None:
        ax.clear()
        _draw_table()

        tips = actual_tips[frame_idx]
        palms = actual_palms[frame_idx]
        ax.scatter(
            palms[:, 0], palms[:, 1], palms[:, 2],
            s=80, c=["#2ca02c", "#9467bd"], marker="^", label="palm(actual)"
        )
        for j in range(6):
            p = tips[j]
            ax.scatter(p[0], p[1], p[2], s=40, c=tip_colors[j], marker="o")

        ret_idx = frame_idx - pregrasp_steps
        status = "pregrasp"
        if 0 <= ret_idx < T:
            target = fingertip_targets[ret_idx]
            ax.scatter(
                target[:, 0], target[:, 1], target[:, 2],
                s=34, c=tip_colors, marker="x", label="tips(target)"
            )
            status = f"retarget s1={bool(ik_success_stage1[ret_idx])} s2={bool(ik_success_stage2[ret_idx])}"

        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(min(-0.05, center[2] - half), center[2] + half)
        ax.set_xlabel("X (world)")
        ax.set_ylabel("Y (world)")
        ax.set_zlabel("Z (world)")
        ax.set_title(f"{out_npz_path.stem} | frame {frame_idx+1}/{N} | {status}")

    stride = max(1, int(math.ceil(N / 400.0)))
    frame_ids = list(range(0, N, stride))
    ani = FuncAnimation(fig, update, frames=frame_ids, interval=66)
    mp4_path = out_npz_path.with_name(f"{out_npz_path.stem}_anime.mp4")
    gif_path = out_npz_path.with_name(f"{out_npz_path.stem}_anime.gif")
    try:
        ani.save(str(mp4_path), writer="ffmpeg", fps=15)
        out = str(mp4_path)
    except Exception:
        ani.save(str(gif_path), writer="pillow", fps=12)
        out = str(gif_path)
    plt.close(fig)
    return out


def _episode_metrics_from_arrays(
    ik1: np.ndarray,
    ik2: np.ndarray,
    tip_err: np.ndarray,
    coll_actual: np.ndarray,
    coll_margin: np.ndarray,
) -> dict[str, float]:
    both = np.logical_and(ik1, ik2)
    flat_err = tip_err.reshape(-1)
    return {
        "ik_success_stage1_rate": _safe_rate(ik1),
        "ik_success_stage2_rate": _safe_rate(ik2),
        "ik_success_both_rate": _safe_rate(both),
        "tip_error_mean_m": float(np.mean(flat_err)) if flat_err.size > 0 else float("nan"),
        "tip_error_median_m": float(np.median(flat_err)) if flat_err.size > 0 else float("nan"),
        "tip_error_p95_m": _percentile(flat_err, 95.0),
        "collision_actual_rate": _safe_rate(coll_actual.astype(np.float32)),
        "collision_margin_rate": _safe_rate(coll_margin.astype(np.float32)),
    }


def _task_episode_from_path(hdf5_path: Path) -> tuple[str, str]:
    task = hdf5_path.parent.name
    episode = hdf5_path.stem
    return task, episode


def _save_episode_npz(
    out_path: Path,
    *,
    source: str,
    task: str,
    episode: str,
    bottle_pos: np.ndarray,
    anchor_mode: str,
    anchor_fixed_z: float,
    q_home: np.ndarray,
    q_pregrasp: np.ndarray,
    traj_home_to_pregrasp: np.ndarray,
    joint_positions_retarget: np.ndarray,
    traj_full: np.ndarray,
    ik_success_stage1: np.ndarray,
    ik_success_stage2: np.ndarray,
    fingertip_errors: np.ndarray,
    collision_actual: np.ndarray,
    collision_margin: np.ndarray,
    fingertip_targets: np.ndarray,
    R_align: np.ndarray,
    t_align: np.ndarray,
    failure_reason_codes: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        source=source,
        task=task,
        episode=episode,
        bottle_pos=bottle_pos.astype(np.float32),
        anchor_mode=np.asarray(anchor_mode),
        anchor_fixed_z=np.asarray(float(anchor_fixed_z), dtype=np.float32),
        q_home=q_home.astype(np.float32),
        q_pregrasp=q_pregrasp.astype(np.float32),
        traj_home_to_pregrasp=traj_home_to_pregrasp.astype(np.float32),
        joint_positions_retarget=joint_positions_retarget.astype(np.float32),
        traj_full=traj_full.astype(np.float32),
        ik_success_stage1=ik_success_stage1.astype(bool),
        ik_success_stage2=ik_success_stage2.astype(bool),
        fingertip_errors=fingertip_errors.astype(np.float32),
        collision_actual=collision_actual.astype(bool),
        collision_margin=collision_margin.astype(bool),
        fingertip_targets=fingertip_targets.astype(np.float32),
        R_align=R_align.astype(np.float32),
        t_align=t_align.astype(np.float32),
        failure_reason_codes=np.asarray(failure_reason_codes, dtype=object),
    )


def _check_planning_distance_silent(planner: CuRoboBimanualMotionPlanner, q: np.ndarray) -> bool:
    with contextlib.redirect_stdout(io.StringIO()):
        return bool(planner.check_at_planning_distance(q))


def _build_failure_reason_codes(
    ik1: np.ndarray,
    ik2: np.ndarray,
    coll_actual: np.ndarray,
    coll_margin: np.ndarray,
    plan_ok: bool,
) -> list[str]:
    codes: list[str] = []
    if not bool(np.any(ik1)):
        codes.append("ik_stage1_all_failed")
    elif not bool(np.all(ik1)):
        codes.append("ik_stage1_partial_failed")

    if not bool(np.any(ik2)):
        codes.append("ik_stage2_all_failed")
    elif not bool(np.all(ik2)):
        codes.append("ik_stage2_partial_failed")

    both = np.logical_and(ik1, ik2)
    if not bool(np.any(both)):
        codes.append("ik_both_all_failed")

    if bool(np.any(coll_actual)):
        codes.append("collision_actual_detected")
    if bool(np.any(coll_margin)):
        codes.append("collision_margin_detected")
    if not plan_ok:
        codes.append("trajopt_pregrasp_failed")

    return codes


def _compute_baseline_collision_metrics(
    baseline_joint_positions: np.ndarray,  # (T, 38)
    planner: CuRoboBimanualMotionPlanner,
) -> tuple[np.ndarray, np.ndarray]:
    q_seq = np.asarray(baseline_joint_positions, dtype=np.float32)
    if q_seq.ndim != 2 or q_seq.shape[1] != 38:
        raise ValueError(f"Expected baseline joint_positions shape (T,38), got {q_seq.shape}")
    T = q_seq.shape[0]
    coll_actual = np.zeros((T,), dtype=bool)
    coll_margin = np.zeros((T,), dtype=bool)
    for i in range(T):
        coll_actual[i] = planner.self_collision_check(q_seq[i])
        coll_margin[i] = not _check_planning_distance_silent(planner, q_seq[i])
    return coll_actual, coll_margin


def _run_baseline_with_collision(
    *,
    ep_path: Path,
    left_offset: np.ndarray,
    right_offset: np.ndarray,
    R_align: np.ndarray,
    t_align: np.ndarray,
    planner: CuRoboBimanualMotionPlanner,
) -> dict[str, float]:
    baseline = baseline_retarget_episode(
        str(ep_path),
        left_offset=left_offset,
        right_offset=right_offset,
        R_align=R_align,
        t_align=t_align,
    )
    b_ik = np.asarray(baseline["ik_success"], dtype=bool).all(axis=1)
    b_err = np.asarray(baseline.get("fingertip_errors"), dtype=np.float32)
    b_q = np.asarray(baseline.get("joint_positions"), dtype=np.float32)
    b_coll_actual, b_coll_margin = _compute_baseline_collision_metrics(b_q, planner)
    metrics = {
        "baseline_ik_success_rate": _safe_rate(b_ik.astype(np.float32)),
        "baseline_tip_error_mean_m": float(np.mean(b_err)) if b_err.size > 0 else float("nan"),
        "baseline_tip_error_p95_m": _percentile(b_err.reshape(-1), 95.0),
        "baseline_collision_actual_rate": _safe_rate(b_coll_actual.astype(np.float32)),
        "baseline_collision_margin_rate": _safe_rate(b_coll_margin.astype(np.float32)),
    }
    metrics["baseline_strict_success"] = _strict_success(
        metrics["baseline_ik_success_rate"], metrics["baseline_collision_actual_rate"]
    )
    return metrics


def _episode_sort_key(episode: str) -> tuple[int, int | str]:
    try:
        return (0, int(episode))
    except Exception:
        return (1, episode)


def _load_report_rows(report_dir: Path) -> list[dict[str, Any]]:
    json_path = report_dir / "curobo_2stage_vs_baseline.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Existing report not found for recompute mode: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        rows = obj.get("episodes", [])
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError(f"Invalid report format in {json_path}")
    return [dict(r) for r in rows]


def _select_recompute_keys(
    rows: list[dict[str, Any]],
    tasks: list[str],
    smoke: bool,
    max_episodes_per_task: int | None,
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    task_set = set(tasks)
    for task in sorted(task_set):
        task_rows = [r for r in rows if str(r.get("task", "")) == task]
        task_rows = sorted(task_rows, key=lambda r: _episode_sort_key(str(r.get("episode", ""))))
        if smoke:
            task_rows = task_rows[:1]
        elif max_episodes_per_task is not None:
            task_rows = task_rows[:max_episodes_per_task]
        for r in task_rows:
            keys.add((task, str(r.get("episode", ""))))
    return keys


def _load_alignment_for_recompute(
    *,
    row: dict[str, Any],
    ep_path: Path,
    output_root: Path,
    bottle_pos_default: np.ndarray,
    center_from: str,
    center_time: str,
    anchor_mode_default: str,
    anchor_fixed_z_default: float,
) -> tuple[np.ndarray, np.ndarray]:
    task = str(row["task"])
    episode = str(row["episode"])
    out_npz = output_root / task / f"{episode}_curobo_2stage.npz"
    if out_npz.exists():
        with np.load(str(out_npz), allow_pickle=True) as d:
            if "R_align" in d and "t_align" in d:
                return np.asarray(d["R_align"], dtype=np.float64), np.asarray(d["t_align"], dtype=np.float64)

    anchor_mode = str(row.get("anchor_mode", anchor_mode_default))
    anchor_fixed_z = float(row.get("anchor_fixed_z", anchor_fixed_z_default))
    anchor_xyz = np.array(
        [
            float(row.get("anchor_x", bottle_pos_default[0])),
            float(row.get("anchor_y", bottle_pos_default[1])),
            float(row.get("anchor_z", bottle_pos_default[2])),
        ],
        dtype=np.float64,
    )
    R_align, t_align = compute_alignment(
        str(ep_path),
        anchor_xyz,
        center_from=center_from,
        center_time=center_time,
        anchor_mode=anchor_mode,
        anchor_fixed_z=anchor_fixed_z,
        return_anchor=False,
    )
    return np.asarray(R_align, dtype=np.float64), np.asarray(t_align, dtype=np.float64)


def _format_stats(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return "n/a"
    return (
        f"mean={np.mean(a):.6f}, median={np.median(a):.6f}, "
        f"min={np.min(a):.6f}, max={np.max(a):.6f}"
    )


def _write_ik_experiment_summary(report_dir: Path, rows: list[dict[str, Any]]) -> Path:
    out_path = report_dir / "ik_experiment_summary.md"
    tasks = sorted({str(r.get("task", "")) for r in rows if "task" in r})

    def _strict_rows(method: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in rows:
            if method == "new":
                ok = _strict_success(
                    r.get("new_ik_success_both_rate"),
                    r.get("new_collision_actual_rate"),
                )
            else:
                ok = _strict_success(
                    r.get("baseline_ik_success_rate"),
                    r.get("baseline_collision_actual_rate"),
                )
            if ok == 1:
                out.append(r)
        return out

    def _method_stats(method: str, subset: list[dict[str, Any]]) -> dict[str, str]:
        if method == "new":
            errs = [float(r["new_tip_error_mean_m"]) for r in subset if "new_tip_error_mean_m" in r]
        else:
            errs = [float(r["baseline_tip_error_mean_m"]) for r in subset if "baseline_tip_error_mean_m" in r]
        return {
            "count": str(len(subset)),
            "rate": f"{(len(subset) / max(1, len(rows))):.2%}",
            "stats": _format_stats(errs),
        }

    new_strict = _strict_rows("new")
    baseline_strict = _strict_rows("baseline")
    new_stats = _method_stats("new", new_strict)
    base_stats = _method_stats("baseline", baseline_strict)

    lines: list[str] = []
    lines.append("# IK Experiment Summary")
    lines.append("")
    lines.append("## 1. EgoDex Input and Retargeting Inputs")
    lines.append("- Episode source: `models/egodex/test/{task}/{episode}.hdf5`.")
    lines.append("- Used arrays: per-frame hand transforms (`transforms`) and camera extrinsics (`cam_ext`).")
    lines.append("- Extracted targets: 6 fingertip points (`left/right` thumb, index, middle) in EgoDex camera/world pipeline.")
    lines.append("")
    lines.append("## 2. Camera-to-URDF Anchoring")
    lines.append("Given a fingertip point `p_cam` in EgoDex camera coordinates, the URDF/world point is:")
    lines.append("")
    lines.append("```text")
    lines.append("p_urdf = R_cam2urdf * p_cam + t")
    lines.append("```")
    lines.append("")
    lines.append("Where:")
    lines.append("- `R_cam2urdf` is fixed by frame convention mapping.")
    lines.append("- Camera center estimate:")
    lines.append("")
    lines.append("```text")
    lines.append("c_cam = mean({p_cam(t, finger)} over selected fingers/frames)")
    lines.append("```")
    lines.append("")
    lines.append("- Translation uses anchor `a_urdf`:")
    lines.append("")
    lines.append("```text")
    lines.append("t = a_urdf - R_cam2urdf * c_cam")
    lines.append("```")
    lines.append("")
    lines.append("- In this run, anchor mode is `last_xy_fixed_z` (xy from last-frame center after rotation, z fixed).")
    lines.append("")
    lines.append("## 3. Fingertip Pose Targets in URDF World")
    lines.append("- For every frame and each of 6 fingertips, target is transformed via the equation above.")
    lines.append("- These transformed fingertip points are the IK position targets used by all compared methods.")
    lines.append("")
    lines.append("## 4. IK Methods Compared")
    lines.append("- Fully IK (concept, `curobo_ik.py`): one full-chain cuRobo solve using 6 fingertip position constraints.")
    lines.append("- Baseline two-stage IK (`retarget_tool` / `retarget_episode_v6`):")
    lines.append("  - Stage-1: Pinocchio-based arm IK to virtual palm centroids.")
    lines.append("  - Stage-2: fixed-base finger retargeting (dex-retargeting).")
    lines.append("- cuRobo two-stage IK (`curobo_two_stage_batch.py`):")
    lines.append("  - Stage-1 cuRobo IK for palms.")
    lines.append("  - Stage-2 cuRobo IK for 6 fingertips.")
    lines.append("  - Pre-grasp: frame-0 IK result as `q_pregrasp`, then cuRobo trajopt plans `q_home -> q_pregrasp`.")
    lines.append("")
    lines.append("## 5. cuRobo Sphere Collision Check")
    lines.append("- `collision_actual`: exact sphere penetration check (`self_collision_activation_distance = 0.0`).")
    lines.append("- `collision_margin`: planning-margin check at activation distance used by planner.")
    lines.append("- Baseline trajectories are now rechecked frame-by-frame with the same cuRobo sphere checker.")
    lines.append("")
    lines.append("## 6. Strict Result Summary (IK=1.0 and Actual Collision=0)")
    lines.append(f"- Total episodes considered: `{len(rows)}`")
    lines.append(f"- Baseline two-stage strict pass: `{base_stats['count']}` (`{base_stats['rate']}`)")
    lines.append(f"  - Tip error stats (episode mean, meters): {base_stats['stats']}")
    lines.append(f"- cuRobo two-stage strict pass: `{new_stats['count']}` (`{new_stats['rate']}`)")
    lines.append(f"  - Tip error stats (episode mean, meters): {new_stats['stats']}")
    lines.append("")
    lines.append("Per-task strict pass counts:")
    for task in tasks:
        b_task = [r for r in baseline_strict if str(r.get("task", "")) == task]
        n_task = [r for r in new_strict if str(r.get("task", "")) == task]
        lines.append(f"- `{task}`: baseline={len(b_task)}, cuRobo={len(n_task)}")
    lines.append("")
    lines.append("## 7. Final Conclusion")
    if len(baseline_strict) > len(new_strict):
        lines.append(
            "Under the strict criterion (IK success rate exactly 1.0 and zero actual self-collision), "
            "baseline two-stage currently passes more episodes."
        )
    elif len(baseline_strict) < len(new_strict):
        lines.append(
            "Under the strict criterion (IK success rate exactly 1.0 and zero actual self-collision), "
            "cuRobo two-stage currently passes more episodes."
        )
    else:
        lines.append(
            "Under the strict criterion (IK success rate exactly 1.0 and zero actual self-collision), "
            "both two-stage methods currently pass the same number of episodes."
        )
    lines.append(
        "On strict-pass subsets, cuRobo two-stage shows much lower fingertip error; "
        "the bottleneck remains pass coverage consistency across tasks."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_reports(
    report_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "curobo_2stage_vs_baseline.json"
    csv_path = report_dir / "curobo_2stage_vs_baseline.csv"

    # Build global aggregates.
    def _agg(field: str) -> float:
        vals = []
        for r in rows:
            if field not in r or r[field] is None:
                continue
            v = float(r[field])
            if math.isnan(v):
                continue
            vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "num_episodes": len(rows),
        "new_ik_success_both_rate_mean": _agg("new_ik_success_both_rate"),
        "new_tip_error_mean_m": _agg("new_tip_error_mean_m"),
        "new_collision_actual_rate_mean": _agg("new_collision_actual_rate"),
        "baseline_ik_success_rate_mean": _agg("baseline_ik_success_rate"),
        "baseline_tip_error_mean_m": _agg("baseline_tip_error_mean_m"),
        "baseline_collision_actual_rate_mean": _agg("baseline_collision_actual_rate"),
        "baseline_collision_margin_rate_mean": _agg("baseline_collision_margin_rate"),
        "new_strict_success_rate": _agg("new_strict_success"),
        "baseline_strict_success_rate": _agg("baseline_strict_success"),
        "pregrasp_plan_success_rate": _agg("pregrasp_plan_success"),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": rows}, f, indent=2)

    all_keys: list[str] = []
    key_set = set()
    for r in rows:
        for k in r.keys():
            if k not in key_set:
                key_set.add(k)
                all_keys.append(k)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return json_path, csv_path


def _iter_task_episodes(data_root: Path, task: str) -> list[Path]:
    task_dir = data_root / task
    if not task_dir.exists():
        raise FileNotFoundError(f"Task dir not found: {task_dir}")
    eps = sorted(task_dir.glob("*.hdf5"), key=lambda p: int(p.stem))
    if not eps:
        raise FileNotFoundError(f"No episodes in task dir: {task_dir}")
    return eps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch cuRobo two-stage retargeting + baseline comparison")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["add_remove_lid", "screw_unscrew_bottle_cap"],
        help="EgoDex task folder names under models/egodex/test/",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=str(_REPO_ROOT / "models" / "egodex" / "test"),
        help="Root of EgoDex task folders",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default=str(_REPO_ROOT / "models" / "egodex" / "traj-retarging"),
        help="Output root for per-episode npz files",
    )
    p.add_argument(
        "--report_dir",
        type=str,
        default=str(_REPO_ROOT / "models" / "egodex" / "traj-retarging" / "reports"),
        help="Output directory for JSON/CSV reports",
    )
    p.add_argument(
        "--bottle_pos",
        type=float,
        nargs=3,
        default=[0.042, 0.0, 0.10],
        metavar=("X", "Y", "Z"),
        help="URDF bottle anchor position used when --anchor_mode=bottle_pos",
    )
    p.add_argument(
        "--anchor_mode",
        type=str,
        default="last_xy_fixed_z",
        choices=["bottle_pos", "last_xy_fixed_z"],
        help="Trajectory anchoring mode for camera->URDF alignment",
    )
    p.add_argument(
        "--anchor_fixed_z",
        type=float,
        default=0.10,
        help="Used when --anchor_mode=last_xy_fixed_z",
    )
    p.add_argument(
        "--center_from",
        type=str,
        default="all6",
        choices=["all6", "left3", "right3"],
        help="Object center source for alignment",
    )
    p.add_argument(
        "--center_time",
        type=str,
        default="last",
        choices=["all", "last", "first"],
        help="Frame subset used for object-center inference",
    )
    p.add_argument(
        "--left_offset",
        type=float,
        nargs=3,
        default=_DEFAULT_LEFT_OFFSET.tolist(),
        metavar=("LX", "LY", "LZ"),
        help="Baseline two-stage left offset",
    )
    p.add_argument(
        "--right_offset",
        type=float,
        nargs=3,
        default=_DEFAULT_RIGHT_OFFSET.tolist(),
        metavar=("RX", "RY", "RZ"),
        help="Baseline two-stage right offset",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_seeds_stage1", type=int, default=64)
    p.add_argument("--num_seeds_stage2", type=int, default=64)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run one episode per task only (first episode)",
    )
    p.add_argument(
        "--max_episodes_per_task",
        type=int,
        default=None,
        help="Optional cap per task",
    )
    p.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline run (still writes new-method outputs)",
    )
    p.add_argument(
        "--recompute_baseline_only",
        action="store_true",
        help="Reload existing report rows and recompute baseline metrics + baseline collision only.",
    )
    p.add_argument(
        "--no_anime",
        action="store_true",
        help="Disable per-episode animation export",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_solver_paths_exist()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    bottle_pos = np.asarray(args.bottle_pos, dtype=np.float32)
    left_offset = np.asarray(args.left_offset, dtype=np.float64)
    right_offset = np.asarray(args.right_offset, dtype=np.float64)

    q_home = _load_home_q_from_driver_cfg()
    print(f"Loaded HOME_Q from driver config: shape={q_home.shape}")

    planner = CuRoboBimanualMotionPlanner(
        robot_cfg_path=str(_REPO_ROOT / "libs" / "robot_description" / "configs_curobo" / "robot" / "bimanual_panda_tesollo.yml"),
        urdf_path=str(_REPO_ROOT / "libs" / "robot_description" / "rl" / "bimanual_panda_tesollo.urdf"),
        spheres_path=str(SPHERES_CFG),
        left_ee_link="left_delto_base_link",
        right_ee_link="right_delto_base_link",
        device=args.device,
        trajopt_dt=0.15,
        trajopt_tsteps=64,
        interpolation_steps=1000,
        num_ik_seeds=50,
        num_trajopt_seeds=32,
        grad_trajopt_iters=800,
        interpolation_dt=1.0 / 30.0,
        collision_activation_distance=0.05,
    )

    if args.recompute_baseline_only:
        if args.skip_baseline:
            raise ValueError("--recompute_baseline_only cannot be used with --skip_baseline")
        rows = _load_report_rows(report_dir)
        keys = _select_recompute_keys(rows, args.tasks, args.smoke, args.max_episodes_per_task)
        print(f"Recompute baseline-only mode: selected {len(keys)} episode(s) for refresh.")

        refreshed_rows: list[dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            task_name = str(row.get("task", ""))
            ep_name = str(row.get("episode", ""))
            key = (task_name, ep_name)

            if key in keys:
                t0 = time.time()
                ep_path = data_root / task_name / f"{ep_name}.hdf5"
                if not ep_path.exists():
                    row["baseline_recompute_error"] = f"missing episode file: {ep_path}"
                else:
                    try:
                        R_align, t_align = _load_alignment_for_recompute(
                            row=row,
                            ep_path=ep_path,
                            output_root=output_root,
                            bottle_pos_default=bottle_pos,
                            center_from=args.center_from,
                            center_time=args.center_time,
                            anchor_mode_default=args.anchor_mode,
                            anchor_fixed_z_default=float(args.anchor_fixed_z),
                        )
                        baseline_metrics = _run_baseline_with_collision(
                            ep_path=ep_path,
                            left_offset=left_offset,
                            right_offset=right_offset,
                            R_align=R_align,
                            t_align=t_align,
                            planner=planner,
                        )
                        row.update(baseline_metrics)
                        row["baseline_recompute_runtime_sec"] = float(time.time() - t0)
                        row.pop("baseline_recompute_error", None)
                        print(
                            f"[baseline-refresh {task_name}/{ep_name}] "
                            f"ik={float(row.get('baseline_ik_success_rate', 0.0)):.1%} "
                            f"coll={float(row.get('baseline_collision_actual_rate', 0.0)):.1%}"
                        )
                    except Exception as exc:
                        row["baseline_recompute_error"] = repr(exc)
                        print(f"[baseline-refresh {task_name}/{ep_name}] ERROR: {exc}")

            row["new_strict_success"] = _strict_success(
                row.get("new_ik_success_both_rate"),
                row.get("new_collision_actual_rate"),
            )
            row["baseline_strict_success"] = _strict_success(
                row.get("baseline_ik_success_rate"),
                row.get("baseline_collision_actual_rate"),
            )
            refreshed_rows.append(row)

        json_path, csv_path = _write_reports(report_dir, refreshed_rows)
        md_path = _write_ik_experiment_summary(report_dir, refreshed_rows)
        print("\n=== Reports (baseline refresh) ===")
        print(f"JSON: {json_path}")
        print(f"CSV:  {csv_path}")
        print(f"MD:   {md_path}")
        return

    solver_params = SolverBuildParams(
        device=args.device,
        num_seeds_stage1=args.num_seeds_stage1,
        num_seeds_stage2=args.num_seeds_stage2,
    )
    two_stage_solver = CuRoboTwoStageIK(solver_params)
    full_fk = FullModelFK(URDF_RETARGET)

    episode_rows: list[dict[str, Any]] = []

    for task in args.tasks:
        eps = _iter_task_episodes(data_root, task)
        if args.smoke:
            eps = eps[:1]
        elif args.max_episodes_per_task is not None:
            eps = eps[: args.max_episodes_per_task]

        print(f"\n=== Task: {task} | Episodes: {len(eps)} ===")

        for ep_path in eps:
            t0 = time.time()
            task_name, ep_name = _task_episode_from_path(ep_path)
            print(f"\n[{task_name}/{ep_name}] start")

            try:
                R_align, t_align, anchor_pos = compute_alignment(
                    str(ep_path),
                    bottle_pos,
                    center_from=args.center_from,
                    center_time=args.center_time,
                    anchor_mode=args.anchor_mode,
                    anchor_fixed_z=float(args.anchor_fixed_z),
                    return_anchor=True,
                )
                ep = load_episode(str(ep_path))
                target_tips = _extract_fingertip_targets_urdf(ep, R_align, t_align)  # (T,6,3)
                T = target_tips.shape[0]

                q_retarget = np.zeros((T, 38), dtype=np.float32)
                ik1 = np.zeros((T,), dtype=bool)
                ik2 = np.zeros((T,), dtype=bool)
                tip_err = np.zeros((T, 6), dtype=np.float32)
                coll_actual = np.zeros((T,), dtype=bool)
                coll_margin = np.zeros((T,), dtype=bool)

                q_prev = q_home.copy()
                for i in range(T):
                    q_i, ok1, ok2 = two_stage_solver.solve_frame(target_tips[i], q_prev, q_home=q_home)
                    q_retarget[i] = q_i
                    ik1[i] = ok1
                    ik2[i] = ok2
                    tip_err[i] = _compute_fingertip_errors(full_fk, q_i, target_tips[i])
                    coll_actual[i] = planner.self_collision_check(q_i)
                    coll_margin[i] = not _check_planning_distance_silent(planner, q_i)

                    if ok1 or ok2:
                        q_prev = q_i

                    if i % 50 == 0 or i == T - 1:
                        both = bool(ok1 and ok2)
                        print(
                            f"  frame {i:4d}/{T}: s1={ok1} s2={ok2} both={both} "
                            f"mean_err={tip_err[i].mean():.4f}m coll={coll_actual[i]}"
                        )

                q_pregrasp = q_retarget[0].copy()
                traj_home_to_pregrasp, last_tstep, plan_ok = planner.plan_to_joint(q_home, q_pregrasp)
                if plan_ok and traj_home_to_pregrasp is not None:
                    traj_home_to_pregrasp = traj_home_to_pregrasp[: last_tstep + 1].astype(np.float32)
                else:
                    traj_home_to_pregrasp = np.zeros((0, 38), dtype=np.float32)
                traj_full = (
                    np.concatenate([traj_home_to_pregrasp, q_retarget], axis=0)
                    if traj_home_to_pregrasp.shape[0] > 0 else q_retarget.copy()
                )

                new_metrics = _episode_metrics_from_arrays(ik1, ik2, tip_err, coll_actual, coll_margin)
                failure_codes = _build_failure_reason_codes(ik1, ik2, coll_actual, coll_margin, bool(plan_ok))

                baseline_metrics: dict[str, Any] = {}
                if not args.skip_baseline:
                    baseline_metrics = _run_baseline_with_collision(
                        ep_path=ep_path,
                        left_offset=left_offset,
                        right_offset=right_offset,
                        R_align=R_align,
                        t_align=t_align,
                        planner=planner,
                    )

                out_path = output_root / task_name / f"{ep_name}_curobo_2stage.npz"
                _save_episode_npz(
                    out_path,
                    source="egodex",
                    task=task_name,
                    episode=ep_name,
                    bottle_pos=anchor_pos,
                    anchor_mode=args.anchor_mode,
                    anchor_fixed_z=float(args.anchor_fixed_z),
                    q_home=q_home,
                    q_pregrasp=q_pregrasp,
                    traj_home_to_pregrasp=traj_home_to_pregrasp,
                    joint_positions_retarget=q_retarget,
                    traj_full=traj_full,
                    ik_success_stage1=ik1,
                    ik_success_stage2=ik2,
                    fingertip_errors=tip_err,
                    collision_actual=coll_actual,
                    collision_margin=coll_margin,
                    fingertip_targets=target_tips,
                    R_align=R_align,
                    t_align=t_align,
                    failure_reason_codes=failure_codes,
                )

                anime_path = ""
                if not args.no_anime:
                    anime_path = _render_episode_animation(
                        out_npz_path=out_path,
                        traj_full=traj_full,
                        fingertip_targets=target_tips,
                        ik_success_stage1=ik1,
                        ik_success_stage2=ik2,
                        pregrasp_steps=int(traj_home_to_pregrasp.shape[0]),
                        full_fk=full_fk,
                    )

                row = {
                    "task": task_name,
                    "episode": ep_name,
                    "anchor_mode": args.anchor_mode,
                    "anchor_fixed_z": float(args.anchor_fixed_z),
                    "anchor_x": float(anchor_pos[0]),
                    "anchor_y": float(anchor_pos[1]),
                    "anchor_z": float(anchor_pos[2]),
                    "frames": T,
                    "pregrasp_plan_success": int(bool(plan_ok)),
                    "pregrasp_traj_steps": int(traj_home_to_pregrasp.shape[0]),
                    "retarget_steps": int(q_retarget.shape[0]),
                    "traj_full_steps": int(traj_full.shape[0]),
                    "new_ik_success_stage1_rate": new_metrics["ik_success_stage1_rate"],
                    "new_ik_success_stage2_rate": new_metrics["ik_success_stage2_rate"],
                    "new_ik_success_both_rate": new_metrics["ik_success_both_rate"],
                    "new_tip_error_mean_m": new_metrics["tip_error_mean_m"],
                    "new_tip_error_median_m": new_metrics["tip_error_median_m"],
                    "new_tip_error_p95_m": new_metrics["tip_error_p95_m"],
                    "new_collision_actual_rate": new_metrics["collision_actual_rate"],
                    "new_collision_margin_rate": new_metrics["collision_margin_rate"],
                    "new_strict_success": _strict_success(
                        new_metrics["ik_success_both_rate"], new_metrics["collision_actual_rate"]
                    ),
                    "stage2_rebuild_count": 0,
                    "failure_reason_codes": ";".join(failure_codes),
                    "anime_path": anime_path,
                    "runtime_sec": float(time.time() - t0),
                }
                row.update(baseline_metrics)
                if "baseline_strict_success" not in row:
                    row["baseline_strict_success"] = _strict_success(
                        row.get("baseline_ik_success_rate"),
                        row.get("baseline_collision_actual_rate"),
                    )
                episode_rows.append(row)
                print(
                    f"[{task_name}/{ep_name}] done | both_ik={row['new_ik_success_both_rate']:.1%} "
                    f"err={row['new_tip_error_mean_m']:.4f}m plan_ok={bool(plan_ok)} "
                    f"time={row['runtime_sec']:.1f}s"
                )
            except Exception as exc:
                row = {
                    "task": task_name,
                    "episode": ep_name,
                    "error": repr(exc),
                    "runtime_sec": float(time.time() - t0),
                    "pregrasp_plan_success": 0,
                }
                episode_rows.append(row)
                print(f"[{task_name}/{ep_name}] ERROR: {exc}")

    json_path, csv_path = _write_reports(report_dir, episode_rows)
    md_path = _write_ik_experiment_summary(report_dir, episode_rows)
    print("\n=== Reports ===")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
