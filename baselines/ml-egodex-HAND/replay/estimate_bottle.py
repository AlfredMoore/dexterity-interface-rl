"""
Estimate bottle/cup dimensions from EgoDex fingertip annotations.

EgoDex has no explicit object annotations, only 68 hand/body joint SE(3) poses.
This script estimates object dimensions by analyzing fingertip positions during
grasping and lid-manipulation phases of add_remove_lid episodes.

Approach:
  1. Load all episodes, convert fingertip positions to object-centric frame
  2. Separate left hand (lid/cap manipulation) and right hand (body grasp)
  3. Estimate radii from horizontal distance to object center
  4. Estimate heights from vertical (Z) extent of fingertips
  5. Aggregate across episodes using median

Output: cap_r, cap_h, body_r, body_h (meters), plus a simple cylinder URDF.

Usage:
    python estimate_bottle.py --data_dir /path/to/add_remove_lid/
    python estimate_bottle.py  # uses default path
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add retarget directory to path for imports
_THIS_DIR = Path(__file__).resolve().parent
_RETARGET_DIR = _THIS_DIR.parent / "retarget"
sys.path.insert(0, str(_RETARGET_DIR))

from retarget_episode import (
    LEFT_FINGER_TIPS,
    RIGHT_FINGER_TIPS,
    CoordinateAligner,
    load_episode,
    _world_to_robot_pos,
)

_REPO_ROOT = _THIS_DIR.parents[2]  # dexterity-interface-rl/
_DEFAULT_DATA_DIR = _REPO_ROOT / "models" / "egodex" / "test" / "add_remove_lid"

# Camera -> URDF rotation (same as retarget_for_sim.py)
_R_CAM_TO_URDF = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0],
], dtype=np.float64)


def extract_fingertip_geometry(hdf5_path: str) -> dict:
    """
    Extract fingertip positions in object-centric URDF frame for one episode.

    Returns dict with:
        left_tips: (T, 3, 3) left hand fingertip positions [thumb, index, middle]
        right_tips: (T, 3, 3) right hand fingertip positions
        object_center_cam: (3,) estimated object center in camera frame
    """
    ep = load_episode(hdf5_path)
    tfs = ep["transforms"]
    T = len(tfs["leftHand"])
    identity_aligner = CoordinateAligner()

    # Collect all fingertip positions in camera frame
    all_cam_positions = []
    for i in range(T):
        cam_ext_i = ep["cam_ext"][i]
        for tip_name in LEFT_FINGER_TIPS + RIGHT_FINGER_TIPS:
            pos = _world_to_robot_pos(
                tfs[tip_name][i], cam_ext_i, identity_aligner
            )
            all_cam_positions.append(pos)
    all_cam_positions = np.array(all_cam_positions)
    object_center_cam = all_cam_positions.mean(axis=0)

    # Transform to object-centric URDF frame (object at origin)
    R = _R_CAM_TO_URDF
    t = -R @ object_center_cam  # center at origin

    left_tips = np.zeros((T, 3, 3), dtype=np.float64)
    right_tips = np.zeros((T, 3, 3), dtype=np.float64)

    for i in range(T):
        cam_ext_i = ep["cam_ext"][i]
        for j, tip_name in enumerate(LEFT_FINGER_TIPS):
            pos_cam = _world_to_robot_pos(
                tfs[tip_name][i], cam_ext_i, identity_aligner
            )
            left_tips[i, j] = R @ pos_cam + t
        for j, tip_name in enumerate(RIGHT_FINGER_TIPS):
            pos_cam = _world_to_robot_pos(
                tfs[tip_name][i], cam_ext_i, identity_aligner
            )
            right_tips[i, j] = R @ pos_cam + t

    return {
        "left_tips": left_tips,
        "right_tips": right_tips,
        "object_center_cam": object_center_cam,
    }


def estimate_dimensions(
    left_tips: np.ndarray,
    right_tips: np.ndarray,
) -> dict:
    """
    Estimate object dimensions from fingertip positions in object-centric frame.

    In add_remove_lid task:
      - Left hand typically manipulates the cap/lid (upper part)
      - Right hand typically holds the body (lower part)

    Radii are estimated from horizontal (XY) distance to center.
    Heights are estimated from vertical (Z) extent.
    """
    # Horizontal distance from center for each fingertip
    left_xy_dist = np.linalg.norm(left_tips[:, :, :2], axis=2)   # (T, 3)
    right_xy_dist = np.linalg.norm(right_tips[:, :, :2], axis=2)  # (T, 3)

    # Z positions (vertical)
    left_z = left_tips[:, :, 2]   # (T, 3)
    right_z = right_tips[:, :, 2]  # (T, 3)

    # Cap radius: median horizontal distance of left hand fingertips
    # (fingers wrap around the cap, so distance ≈ cap radius + finger thickness)
    cap_r_raw = np.median(left_xy_dist)
    # Body radius: median horizontal distance of right hand fingertips
    body_r_raw = np.median(right_xy_dist)

    # Heights from Z extent
    # Left hand operates on the upper portion (cap)
    left_z_range = np.percentile(left_z, 90) - np.percentile(left_z, 10)
    # Right hand holds the lower portion (body)
    right_z_range = np.percentile(right_z, 90) - np.percentile(right_z, 10)

    # The cap height is roughly the Z range of the left hand
    cap_h_raw = left_z_range
    # The body height is roughly the Z range of the right hand
    body_h_raw = right_z_range

    # Z center of each hand's interaction zone
    left_z_center = np.median(left_z)
    right_z_center = np.median(right_z)

    return {
        "cap_r_raw": cap_r_raw,
        "body_r_raw": body_r_raw,
        "cap_h_raw": cap_h_raw,
        "body_h_raw": body_h_raw,
        "left_z_center": left_z_center,
        "right_z_center": right_z_center,
        "left_z_range": left_z_range,
        "right_z_range": right_z_range,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate bottle/cup dimensions from EgoDex fingertip data"
    )
    parser.add_argument(
        "--data_dir", type=str, default=str(_DEFAULT_DATA_DIR),
        help="Directory containing EgoDex .hdf5 files",
    )
    parser.add_argument(
        "--finger_thickness", type=float, default=0.012,
        help="Approximate finger thickness offset to subtract from radius (m)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    episodes = sorted(data_dir.glob("*.hdf5"))
    if not episodes:
        print(f"No .hdf5 files found in {data_dir}")
        return

    print(f"Found {len(episodes)} episodes in {data_dir}")

    all_estimates = []
    for ep_path in episodes:
        print(f"\nProcessing {ep_path.name}...")
        geo = extract_fingertip_geometry(str(ep_path))
        est = estimate_dimensions(geo["left_tips"], geo["right_tips"])
        all_estimates.append(est)
        print(f"  cap_r={est['cap_r_raw']:.4f}m, body_r={est['body_r_raw']:.4f}m, "
              f"cap_h={est['cap_h_raw']:.4f}m, body_h={est['body_h_raw']:.4f}m")
        print(f"  left_z_center={est['left_z_center']:.4f}m, "
              f"right_z_center={est['right_z_center']:.4f}m")

    # Aggregate across episodes
    print(f"\n{'='*60}")
    print("AGGREGATE (median across episodes)")
    print(f"{'='*60}")

    keys = ["cap_r_raw", "body_r_raw", "cap_h_raw", "body_h_raw",
            "left_z_center", "right_z_center"]
    agg = {}
    for k in keys:
        values = [e[k] for e in all_estimates]
        agg[k] = np.median(values)
        print(f"  {k:>20s}: median={agg[k]:.4f}m, "
              f"std={np.std(values):.4f}m, "
              f"range=[{np.min(values):.4f}, {np.max(values):.4f}]")

    # Apply finger thickness correction for actual object dimensions
    ft = args.finger_thickness
    cap_r = max(agg["cap_r_raw"] - ft, 0.01)
    body_r = max(agg["body_r_raw"] - ft, 0.01)
    cap_h = agg["cap_h_raw"]
    body_h = agg["body_h_raw"]

    print(f"\n{'='*60}")
    print(f"ESTIMATED OBJECT DIMENSIONS (finger_thickness={ft:.3f}m)")
    print(f"{'='*60}")
    print(f"  cap_r  = {cap_r:.4f}m  (cap/lid radius)")
    print(f"  cap_h  = {cap_h:.4f}m  (cap/lid height)")
    print(f"  body_r = {body_r:.4f}m  (body radius)")
    print(f"  body_h = {body_h:.4f}m  (body height)")
    print(f"  total_h ≈ {body_h + cap_h:.4f}m")

    print(f"\nFor reference, HAND-gat bottle (unscaled):")
    print(f"  cap_r=0.032m, cap_h=0.021m, body_r=0.043m, body_h=0.063m")
    print(f"HAND-gat bottle (scaled 1.5x):")
    print(f"  cap_r=0.048m, cap_h=0.032m, body_r=0.064m, body_h=0.095m")

    # Save results
    results_path = _THIS_DIR / "bottle_estimate.json"
    import json
    results = {
        "cap_r": round(cap_r, 4),
        "cap_h": round(cap_h, 4),
        "body_r": round(body_r, 4),
        "body_h": round(body_h, 4),
        "finger_thickness": ft,
        "n_episodes": len(episodes),
        "raw": {k: round(float(agg[k]), 4) for k in keys},
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
