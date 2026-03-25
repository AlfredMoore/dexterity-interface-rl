"""
Batch retargeting script: EgoDex dataset → bimanual robot joint trajectories.

Filters episodes by language annotation keywords (simple string match),
retargets each episode, and saves results as .npz files.

Usage:
    cd /workspace
    python baselines/ml-egodex-HAND/retarget/run_retarget.py \\
        --data_dir /path/to/egodex \\
        --output_dir /path/to/output \\
        --keywords "cap,lid,twist" \\
        --max_episodes 50

    # Check joint limits / smoothness only (no file save)
    python ... --check_limits

Output structure:
    output_dir/
      <task_name>/
        <idx>_retargeted.npz
          joint_positions  (T, 38) float64
          ik_success       (T, 2)  bool
          episode_meta     dict    (hdf5_path, task, llm_description, ...)
"""

import sys as _sys
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Ensure this module is importable from the repo root
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR.parent) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPT_DIR.parent))

from retarget.retarget_episode import (
    CoordinateAligner,
    HandRetargeter,
    PandaArmIKSolver,
    build_retargeters,
    retarget_episode,
)

# Try to import compute_metrics for optional evaluation
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "compute_metrics", _SCRIPT_DIR.parent / "compute_metrics.py"
    )
    _cm_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cm_mod)
    evaluate_distance = _cm_mod.evaluate_distance
    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False


# ---------------------------------------------------------------------------
# Episode discovery & filtering
# ---------------------------------------------------------------------------

def find_hdf5_files(data_dir: str) -> list[str]:
    """Recursively find all HDF5 files under data_dir."""
    found = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".hdf5"):
                found.append(os.path.join(root, f))
    found.sort()
    print(f"Found {len(found)} HDF5 files in {data_dir}")
    return found


def get_language_description(hdf5_path: str) -> str:
    """Return the episode's language description (GPT-4 annotation)."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            llm_type = f.attrs.get("llm_type", "")
            if llm_type == "reversible":
                which = f.attrs.get("which_llm_description", "1")
                key = "llm_description" if which == "1" else "llm_description2"
            else:
                key = "llm_description"
            desc = f.attrs.get(key, "")
            return str(desc)
    except Exception:
        return ""


def filter_episodes(
    hdf5_files: list[str],
    keywords: list[str],
) -> list[str]:
    """
    Keep files whose language description contains at least one keyword.
    If keywords is empty, return all files.
    """
    if not keywords:
        return hdf5_files
    kws = [k.lower().strip() for k in keywords if k.strip()]
    kept = []
    for path in tqdm(hdf5_files, desc="Filtering by keywords"):
        desc = get_language_description(path).lower()
        if any(k in desc for k in kws):
            kept.append(path)
    print(f"After filtering: {len(kept)} / {len(hdf5_files)} episodes match keywords {kws}")
    return kept


# ---------------------------------------------------------------------------
# Joint-limit / smoothness diagnostics
# ---------------------------------------------------------------------------

def _check_trajectory(
    joint_positions: np.ndarray,
    ik_success:      np.ndarray,
    joint_limits_lo: np.ndarray,
    joint_limits_hi: np.ndarray,
    fps:             float = 30.0,
) -> dict:
    T = joint_positions.shape[0]
    limit_violations = np.mean(
        (joint_positions < joint_limits_lo) | (joint_positions > joint_limits_hi)
    )
    ik_rate = ik_success.mean(axis=0)
    vel = np.diff(joint_positions, axis=0) * fps     # joint/s
    max_vel = np.abs(vel).max(axis=0)
    mean_vel = np.abs(vel).mean(axis=0)
    return {
        "T":                    T,
        "limit_violation_rate": float(limit_violations),
        "ik_success_rate":      ik_rate.tolist(),
        "max_joint_vel":        max_vel.tolist(),
        "mean_joint_vel":       mean_vel.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch EgoDex → robot retargeting")
    p.add_argument("--data_dir",    required=True,
                   help="Root directory of EgoDex dataset")
    p.add_argument("--output_dir",  required=True,
                   help="Where to save retargeted .npz files")
    p.add_argument("--keywords",    default="cap,lid,twist,open,close,unscrew,screw",
                   help="Comma-separated keywords to filter episodes by language annotation")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Process at most this many episodes (for quick tests)")
    p.add_argument("--check_limits", action="store_true",
                   help="Print joint-limit / velocity diagnostics per episode")
    p.add_argument("--no_filter",   action="store_true",
                   help="Process all episodes without keyword filtering")
    p.add_argument("--cam_R", type=str, default=None,
                   help="3x3 rotation matrix (row-major JSON) camera→robot base frame")
    p.add_argument("--cam_t", type=str, default=None,
                   help="3-element translation JSON camera→robot base frame")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Coordinate aligner (identity by default)
    R = np.array(json.loads(args.cam_R)).reshape(3, 3) if args.cam_R else None
    t = np.array(json.loads(args.cam_t)) if args.cam_t else None
    aligner = CoordinateAligner(R, t)

    # Build retargeters (one-time setup)
    print("Initialising retargeters and IK solvers...")
    left_ret, right_ret, left_ik, right_ik = build_retargeters()
    print("Ready.\n")

    # Gather and filter episodes
    hdf5_files = find_hdf5_files(args.data_dir)
    if not args.no_filter:
        keywords = [k for k in args.keywords.split(",") if k.strip()]
        hdf5_files = filter_episodes(hdf5_files, keywords)

    if args.max_episodes is not None:
        hdf5_files = hdf5_files[: args.max_episodes]
    print(f"Processing {len(hdf5_files)} episodes...\n")

    # Build joint limits array for diagnostics (38-DOF)
    lo = np.concatenate([
        left_ik.model.lowerPositionLimit,
        np.full(12, -np.pi),          # Tesollo (no strict limits in reduced model)
        right_ik.model.lowerPositionLimit,
        np.full(12, -np.pi),
    ])
    hi = np.concatenate([
        left_ik.model.upperPositionLimit,
        np.full(12,  np.pi),
        right_ik.model.upperPositionLimit,
        np.full(12,  np.pi),
    ])

    n_ok = n_fail = 0
    for hdf5_path in tqdm(hdf5_files, desc="Retargeting"):
        try:
            result = retarget_episode(
                hdf5_path, left_ret, right_ret, left_ik, right_ik, aligner
            )
        except Exception as exc:
            print(f"  ERROR {hdf5_path}: {exc}")
            n_fail += 1
            continue

        joint_positions = result["joint_positions"]
        ik_success      = result["ik_success"]

        # Diagnostics
        if args.check_limits:
            diag = _check_trajectory(joint_positions, ik_success, lo, hi)
            tqdm.write(
                f"  {Path(hdf5_path).name}  T={diag['T']}  "
                f"limit_viol={diag['limit_violation_rate']:.3f}  "
                f"ik_rate={diag['ik_success_rate']}"
            )

        # Determine output path: mirror the task/index structure of the input
        rel = Path(hdf5_path).relative_to(args.data_dir)
        out_path = output_dir / rel.parent / (rel.stem + "_retargeted.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Episode metadata
        meta = {
            "hdf5_path":       hdf5_path,
            "task":            rel.parts[0] if len(rel.parts) > 1 else "",
            "llm_description": get_language_description(hdf5_path),
        }

        np.savez_compressed(
            out_path,
            joint_positions=joint_positions,
            ik_success=ik_success,
            episode_meta=np.array([json.dumps(meta)]),   # stored as 1-element array
        )
        n_ok += 1

    print(f"\nDone. {n_ok} saved, {n_fail} failed.")
    if n_fail > 0:
        print("  (Re-run with a smaller --max_episodes to debug failures)")


if __name__ == "__main__":
    main()
