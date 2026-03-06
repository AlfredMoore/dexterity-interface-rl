"""
traj_sampler.py — Generate and save go/return trajectories for a pre-grasp config.

Usage
-----
    python traj_sampler.py [--index 0] [--pregrasp models/pre_grasp_q_samples.pt]

For a given pre-grasp config (selected by index from the .pt file):
  1. Collision-check the pre-grasp Q.
  2. Plan  HOME_Q → PRE_GRASP_Q  (forward trajectory).
  3. Plan  PRE_GRASP_Q → HOME_Q  (return trajectory).
  4. Save both as NumPy arrays in models/.

Output files
------------
  models/traj_fwd_{index}.npy   — (T, 38) float32, HOME → PRE_GRASP
  models/traj_ret_{index}.npy   — (T, 38) float32, PRE_GRASP → HOME
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path resolution
#   __file__ = .../libs/robot_motion_interface/src/robot_motion_interface/utils/traj_sampler.py
#   parents[4] = .../libs/
#   parents[5] = project root  (dexterity-interface-rl/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_MODELS_DIR   = _PROJECT_ROOT / "models"

# dim: 7(L_arm) + 12(L_hand) + 7(R_arm) + 12(R_hand) = 38
HOME_Q = np.array([
    0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index",    type=int, default=0,
                        help="Index of the pre-grasp config to use (default: 0).")
    parser.add_argument("--pregrasp", type=str,
                        default=str(_MODELS_DIR / "pre_grasp_q_samples.pt"),
                        help="Path to the pre-grasp .pt file.")
    args = parser.parse_args()

    pregrasp_path = Path(args.pregrasp)
    if not pregrasp_path.exists():
        raise FileNotFoundError(f"Pre-grasp file not found: {pregrasp_path}")

    # ── Load pre-grasp configs ──────────────────────────────────────────────
    pregrasp_tensor = torch.load(pregrasp_path, weights_only=True)   # (N, 38)
    pregrasp_qs: np.ndarray = pregrasp_tensor.numpy().astype(np.float32)
    n_configs = len(pregrasp_qs)

    idx = args.index % n_configs
    pre_grasp_q = pregrasp_qs[idx]

    print(f"Loaded {n_configs} pre-grasp configs from {pregrasp_path}")
    print(f"Using index {idx}: left_arm={np.round(pre_grasp_q[:7], 3).tolist()}")

    # ── Planner ──────────────────────────────────────────────────────────────
    try:
        from .kinematics import CuRoboBimanualMotionPlanner, DEFAULT_CUROBO_ROBOT_CFG_PATH
    except ImportError:
        from kinematics import CuRoboBimanualMotionPlanner, DEFAULT_CUROBO_ROBOT_CFG_PATH

    print("\nInitializing CuRoboBimanualMotionPlanner (warmup may take ~30 s)...")
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
        interpolation_dt            = 0.02,
        collision_activation_distance = 0.005,
    )

    # ── Collision check ───────────────────────────────────────────────────────
    print(f"\nChecking pre-grasp Q [{idx}] for self-collision...")
    if planner.self_collision_check(pre_grasp_q, verbose=True):
        print("[ABORT] Pre-grasp Q is in self-collision. Exiting.")
        return
    print("[OK] Collision-free.")

    # ── Forward: HOME → PRE_GRASP ─────────────────────────────────────────────
    print("\nPlanning HOME_Q → PRE_GRASP_Q...")
    traj_fwd, last_tstep_fwd, ok_fwd = planner.plan_to_joint(HOME_Q, pre_grasp_q)
    if not ok_fwd:
        print("[FAIL] Forward planning failed. Exiting.")
        return
    traj_fwd = traj_fwd[:last_tstep_fwd + 1]
    dt = planner._interpolation_dt
    print(f"[OK] Forward: {traj_fwd.shape[0]} steps  "
          f"(dt={dt:.3f}s, ~{traj_fwd.shape[0] * dt:.1f}s total)")

    # ── Return: PRE_GRASP → HOME ──────────────────────────────────────────────
    print("\nPlanning PRE_GRASP_Q → HOME_Q...")
    traj_ret, last_tstep_ret, ok_ret = planner.plan_to_joint(pre_grasp_q, HOME_Q)
    if not ok_ret:
        print("[FAIL] Return planning failed. Exiting.")
        return
    traj_ret = traj_ret[:last_tstep_ret + 1]
    print(f"[OK] Return:  {traj_ret.shape[0]} steps  "
          f"(dt={dt:.3f}s, ~{traj_ret.shape[0] * dt:.1f}s total)")

    # ── Save ──────────────────────────────────────────────────────────────────
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fwd_path = _MODELS_DIR / f"traj_fwd_{idx}.npy"
    ret_path = _MODELS_DIR / f"traj_ret_{idx}.npy"
    np.save(fwd_path, traj_fwd)
    np.save(ret_path, traj_ret)

    print(f"\n[SAVED] {fwd_path}  shape={traj_fwd.shape}")
    print(f"[SAVED] {ret_path}  shape={traj_ret.shape}")

    # ── Mesh export — pre-grasp config at selected index ─────────────────────
    mesh_path = str(_MODELS_DIR / f"scene_pregrasp_{idx}.stl")
    print(f"\nExporting pre-grasp mesh for index {idx}...")
    planner.save_scene_as_mesh(pre_grasp_q, mesh_path)
    print(f"[SAVED] {mesh_path}")

    # ── Collision-check benchmark on forward trajectory ───────────────────────
    print(f"\nBenchmarking sequential collision checks on forward trajectory ({traj_fwd.shape[0]} steps)...")
    planner.benchmark_traj_collision_check(traj_fwd, verbose=False)

    print(f"\nBenchmarking with planner activation distance ({planner._collision_activation_distance} m)...")
    planner.benchmark_traj_collision_check(
        traj_fwd,
        activation_distance=planner._collision_activation_distance,
        verbose=False,
    )


if __name__ == "__main__":
    main()
