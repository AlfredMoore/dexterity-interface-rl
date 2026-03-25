"""
Step 2': CoordinateAligner calibration.

Analyzes hand positions from all retargeted episodes (in camera frame),
then estimates R/t that maps them into the Panda's reachable workspace.

Strategy:
  1. Load all EgoDex episodes, extract wrist positions in camera frame.
  2. Compute centroid and principal axes (PCA).
  3. Propose R/t: map centroid → robot workspace center, align axes.
  4. Re-run retargeting with the proposed R/t and measure IK success rate.
  5. Print the final R/t for use in run_retarget.py.

Run:
    docker exec handrl-policy bash -c "
        cd /workspace &&
        /root/miniconda3/envs/policy/bin/python \
            baselines/ml-egodex-HAND/retarget/calibrate_aligner.py \
            --data_dir /workspace/models/egodex/test/add_remove_lid
    "
"""

import sys as _sys
_sys.path = [p for p in _sys.path if "openrobots" not in p]

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).parent
if str(_HERE.parent) not in _sys.path:
    _sys.path.insert(0, str(_HERE.parent))

from retarget.retarget_episode import (
    CoordinateAligner,
    build_retargeters,
    retarget_episode,
)


# ---------------------------------------------------------------------------
# Panda reachable workspace center (robot base frame, metres)
# Franka Panda: reach ~855mm, comfortable task space ~0.3-0.6m from base
# ---------------------------------------------------------------------------
_ROBOT_WORKSPACE_CENTER = np.array([0.45, 0.0, 0.40])


def load_wrist_positions_camera_frame(hdf5_path: str) -> np.ndarray:
    """
    Return (T, 3) right-wrist positions in camera frame for one episode.
    """
    with h5py.File(hdf5_path, "r") as f:
        cam_ext   = f["/transforms/camera"][:]       # (T, 4, 4)
        rh_world  = f["/transforms/rightHand"][:]    # (T, 4, 4)
        lh_world  = f["/transforms/leftHand"][:]     # (T, 4, 4)

    T = len(cam_ext)
    r_pos = np.zeros((T, 3))
    l_pos = np.zeros((T, 3))
    for i in range(T):
        tf_r = np.linalg.inv(cam_ext[i]) @ rh_world[i]
        tf_l = np.linalg.inv(cam_ext[i]) @ lh_world[i]
        r_pos[i] = tf_r[:3, 3]
        l_pos[i] = tf_l[:3, 3]
    return r_pos, l_pos


def collect_all_positions(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Collect all wrist positions across all episodes."""
    files = sorted(Path(data_dir).glob("*.hdf5"))
    all_r, all_l = [], []
    for f in files:
        r, l = load_wrist_positions_camera_frame(str(f))
        all_r.append(r)
        all_l.append(l)
        print(f"  {f.name}: right centroid={r.mean(0).round(3)}  left centroid={l.mean(0).round(3)}")
    return np.concatenate(all_r), np.concatenate(all_l)


def estimate_calibration(all_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate R/t from camera frame to robot base frame.

    Approach:
      - t: shifts the centroid of all hand positions to _ROBOT_WORKSPACE_CENTER
      - R: computed from PCA — maps the dominant motion axes to robot frame axes

    For a typical egocentric camera looking at a tabletop workspace:
      camera +Z (depth/forward)  → robot +X (forward toward robot)
      camera +X (right)          → robot -Y (right hand side of robot)
      camera -Y (up in image)    → robot +Z (up)

    This canonical rotation is used as the initial guess; PCA refines the scale.
    """
    centroid = all_pos.mean(axis=0)
    print(f"\nAll-episode hand centroid (camera frame): {centroid.round(3)}")
    print(f"Hand position std:                         {all_pos.std(0).round(3)}")
    print(f"Hand position range X: [{all_pos[:,0].min():.3f}, {all_pos[:,0].max():.3f}]")
    print(f"Hand position range Y: [{all_pos[:,1].min():.3f}, {all_pos[:,1].max():.3f}]")
    print(f"Hand position range Z: [{all_pos[:,2].min():.3f}, {all_pos[:,2].max():.3f}]")

    # Canonical rotation: camera → robot
    # Adjust based on observed data distribution axes
    # EgoDex camera is head-mounted, looking forward and slightly down.
    # Typical axes after inv(cam_ext):
    #   X: lateral (right positive)
    #   Y: vertical (down positive in camera)
    #   Z: depth (forward positive)
    # Robot base frame:
    #   X: forward (away from robot base)
    #   Y: left
    #   Z: up
    R_canonical = np.array([
        [ 0,  0,  1],   # robot X = camera Z (depth → forward)
        [-1,  0,  0],   # robot Y = -camera X (right → left)
        [ 0, -1,  0],   # robot Z = -camera Y (down → up)
    ], dtype=np.float64)

    # After applying R_canonical, new centroid in robot frame
    centroid_robot = R_canonical @ centroid
    t = _ROBOT_WORKSPACE_CENTER - centroid_robot
    print(f"\nProposed R (canonical camera→robot):\n{R_canonical}")
    print(f"Proposed t: {t.round(3)}")

    return R_canonical, t


def evaluate_ik_rate(
    data_dir: str,
    R: np.ndarray,
    t: np.ndarray,
    max_episodes: int = None,
) -> dict:
    """Re-run retargeting with given R/t, return per-episode IK rates."""
    print("\nBuilding retargeters for evaluation ...")
    left_ret, right_ret, left_ik, right_ik = build_retargeters()
    aligner = CoordinateAligner(R, t)

    files = sorted(Path(data_dir).glob("*.hdf5"))
    if max_episodes:
        files = files[:max_episodes]

    results = {}
    for f in files:
        try:
            res = retarget_episode(str(f), left_ret, right_ret, left_ik, right_ik, aligner)
            ik = res["ik_success"]
            results[f.name] = {
                "left":  float(ik[:, 0].mean()),
                "right": float(ik[:, 1].mean()),
                "T":     int(ik.shape[0]),
            }
        except Exception as e:
            results[f.name] = {"left": 0.0, "right": 0.0, "T": 0, "error": str(e)}

    return results


def print_ik_summary(results: dict, label: str) -> float:
    print(f"\n{'='*60}")
    print(f"IK success rates — {label}")
    print(f"{'File':<15} {'Left':>8} {'Right':>8} {'T':>6}")
    print("-" * 40)
    lefts, rights = [], []
    for name, r in sorted(results.items()):
        print(f"  {name:<13} {r['left']:>7.1%} {r['right']:>8.1%} {r['T']:>6}")
        lefts.append(r["left"])
        rights.append(r["right"])
    mean_ik = (np.mean(lefts) + np.mean(rights)) / 2
    print(f"\n  Mean left={np.mean(lefts):.1%}  Mean right={np.mean(rights):.1%}  "
          f"Overall={mean_ik:.1%}")
    print("=" * 60)
    return mean_ik


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--eval_episodes", type=int, default=None,
                   help="Limit evaluation episodes (default: all)")
    args = p.parse_args()

    print("=== Step 2': CoordinateAligner Calibration ===\n")
    print("Phase 1: Collecting hand positions (camera frame) across all episodes ...")
    all_r, all_l = collect_all_positions(args.data_dir)
    all_pos = np.concatenate([all_r, all_l])

    # Phase 2: Estimate calibration
    print("\nPhase 2: Estimating R/t ...")
    R, t = estimate_calibration(all_pos)

    # Phase 3: Baseline (identity)
    print("\nPhase 3: Baseline evaluation (identity aligner) ...")
    base_results = evaluate_ik_rate(args.data_dir, None, None, args.eval_episodes)
    base_score   = print_ik_summary(base_results, "identity (baseline)")

    # Phase 4: Calibrated evaluation
    print("\nPhase 4: Calibrated evaluation ...")
    cal_results = evaluate_ik_rate(args.data_dir, R, t, args.eval_episodes)
    cal_score   = print_ik_summary(cal_results, "calibrated R/t")

    improvement = cal_score - base_score
    print(f"\nImprovement: {base_score:.1%} → {cal_score:.1%}  "
          f"({'+'if improvement>=0 else ''}{improvement:.1%})")

    # Phase 5: Print final R/t for use in run_retarget.py
    print("\n=== Final calibration parameters ===")
    print(f"--cam_R '{json.dumps(R.flatten().tolist())}'")
    print(f"--cam_t '{json.dumps(t.tolist())}'")


if __name__ == "__main__":
    main()
