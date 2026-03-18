"""
Batch retargeting with cuRobo two-stage IK + pre-grasp trajopt.

Pipeline (per episode):
  1) Compute camera->URDF alignment from EgoDex fingertip data and bottle_pos.
  2) Per frame two-stage IK with two persistent solvers:
       - Stage-1 (arm solver): lock all finger joints, solve left/right virtual palm.
       - Stage-2 (finger solver): strict lock both arms at Stage-1 result, solve 6 fingertip targets.
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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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
    """Two persistent cuRobo IK solvers for strict two-stage solving."""

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
        self.stage2_arm_lock_keys = list(self.stage2_cfg_dict["robot_cfg"]["kinematics"]["lock_joints"].keys())
        self.stage2_current_lock_map = dict(self.stage2_cfg_dict["robot_cfg"]["kinematics"]["lock_joints"])

        self._stage2_update_mode = "kinematics_update"
        self._stage2_rebuild_count = 0

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

    def _rebuild_stage2_solver(self) -> None:
        self.stage2_solver = self._build_solver(self.stage2_cfg_dict, self.params.num_seeds_stage2)
        self.stage2_active_names = list(self.stage2_solver.kinematics.joint_names)
        self._stage2_rebuild_count += 1

    def _update_stage2_arm_locks(self, q_stage1: np.ndarray) -> None:
        new_lock_map = {
            jn: float(q_stage1[self.full_name_to_idx[jn]])
            for jn in self.stage2_arm_lock_keys
        }
        if new_lock_map == self.stage2_current_lock_map:
            return

        self.stage2_cfg_dict["robot_cfg"]["kinematics"]["lock_joints"] = new_lock_map
        self.stage2_current_lock_map = new_lock_map

        if self._stage2_update_mode == "kinematics_update":
            try:
                robot_cfg = RobotConfig.from_dict(self.stage2_cfg_dict, self.tensor_args)
                self.stage2_solver.kinematics.update_kinematics_config(
                    robot_cfg.kinematics.kinematics_config
                )
                self.stage2_active_names = list(self.stage2_solver.kinematics.joint_names)
                return
            except Exception:
                # fallback to robust mode
                self._stage2_update_mode = "rebuild"

        self._rebuild_stage2_solver()

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

        # Stage-2: strict arm lock at stage-1 result, solve fingertips.
        self._update_stage2_arm_locks(q_stage1)
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
                lock_overrides=self.stage2_current_lock_map,
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


def _write_reports(
    report_dir: Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "curobo_2stage_vs_baseline.json"
    csv_path = report_dir / "curobo_2stage_vs_baseline.csv"

    # Build global aggregates.
    def _agg(field: str) -> float:
        vals = [float(r[field]) for r in rows if field in r and r[field] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "num_episodes": len(rows),
        "new_ik_success_both_rate_mean": _agg("new_ik_success_both_rate"),
        "new_tip_error_mean_m": _agg("new_tip_error_mean_m"),
        "baseline_ik_success_rate_mean": _agg("baseline_ik_success_rate"),
        "baseline_tip_error_mean_m": _agg("baseline_tip_error_mean_m"),
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
        default=[0.042, 0.0, -0.0215],
        metavar=("X", "Y", "Z"),
        help="URDF bottle position used for alignment",
    )
    p.add_argument(
        "--center_from",
        type=str,
        default="all6",
        choices=["all6", "left3", "right3"],
        help="Object center source for alignment",
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

    solver_params = SolverBuildParams(
        device=args.device,
        num_seeds_stage1=args.num_seeds_stage1,
        num_seeds_stage2=args.num_seeds_stage2,
    )
    two_stage_solver = CuRoboTwoStageIK(solver_params)
    full_fk = FullModelFK(URDF_RETARGET)

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
                R_align, t_align = compute_alignment(str(ep_path), bottle_pos, center_from=args.center_from)
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
                rebuild_before = two_stage_solver._stage2_rebuild_count
                for i in range(T):
                    q_i, ok1, ok2 = two_stage_solver.solve_frame(target_tips[i], q_prev, q_home=q_home)
                    q_retarget[i] = q_i
                    ik1[i] = ok1
                    ik2[i] = ok2
                    tip_err[i] = _compute_fingertip_errors(full_fk, q_i, target_tips[i])
                    coll_actual[i] = planner.self_collision_check(q_i)
                    coll_margin[i] = not _check_planning_distance_silent(planner, q_i)

                    if ok1:
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
                    baseline = baseline_retarget_episode(
                        str(ep_path),
                        left_offset=left_offset,
                        right_offset=right_offset,
                        R_align=R_align,
                        t_align=t_align,
                    )
                    b_ik = np.asarray(baseline["ik_success"], dtype=bool).all(axis=1)
                    b_err = np.asarray(baseline.get("fingertip_errors"), dtype=np.float32)
                    baseline_metrics = {
                        "baseline_ik_success_rate": _safe_rate(b_ik.astype(np.float32)),
                        "baseline_tip_error_mean_m": float(np.mean(b_err)) if b_err.size > 0 else float("nan"),
                        "baseline_tip_error_p95_m": _percentile(b_err.reshape(-1), 95.0),
                    }

                out_path = output_root / task_name / f"{ep_name}_curobo_2stage.npz"
                _save_episode_npz(
                    out_path,
                    source="egodex",
                    task=task_name,
                    episode=ep_name,
                    bottle_pos=bottle_pos,
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

                row = {
                    "task": task_name,
                    "episode": ep_name,
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
                    "stage2_rebuild_count": two_stage_solver._stage2_rebuild_count - rebuild_before,
                    "failure_reason_codes": ";".join(failure_codes),
                    "runtime_sec": float(time.time() - t0),
                }
                row.update(baseline_metrics)
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
    print("\n=== Reports ===")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
