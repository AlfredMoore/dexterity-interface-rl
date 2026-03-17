"""
Retarget EgoDex episodes to HAND-gat IsaacSim environment (v5: cuRobo full-chain IK).

Pipeline:
  1. Extract fingertip positions from EgoDex HDF5 (camera frame)
  2. Compute object-centric alignment: camera frame -> bimanual URDF frame
  3. Solve 38-DOF full-chain IK with cuRobo (6 fingertip position targets)
  4. Save trajectories for HAND-gat replay

Key difference from v4: uses cuRobo 38-DOF simultaneous IK (arm + finger)
instead of separate pinocchio arm IK + dex-retargeting hand optimization.
This eliminates the wrist orientation mismatch that caused 15-25cm errors.

Coordinate frames:
  - EgoDex camera frame (ARKit): X-right, Y-up, Z-toward-viewer
  - Bimanual URDF frame: X-right, Y-back, Z-up
  - Bottle in URDF frame: (0.042, 0, -0.022)

Output joint order: [left_panda*7, left_tesollo*12, right_panda*7, right_tesollo*12]

Usage (in handrl-policy container):
    python retarget_for_sim.py --episode /path/to/0.hdf5 --output /path/to/out/
    python retarget_for_sim.py --input_dir /path/to/add_remove_lid/ --output /path/to/out/
"""

import argparse
from pathlib import Path

import numpy as np

from retarget_episode import (
    LEFT_FINGER_TIPS,
    RIGHT_FINGER_TIPS,
    CoordinateAligner,
    load_episode,
    _world_to_robot_pos,
)
from curobo_ik import CuRoboRetargetIK

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RETARGET_DIR = Path(__file__).parent

# Bottle position in bimanual URDF frame.
_BOTTLE_POS_URDF = np.array([0.042, 0.0, -0.0215])

# Rotation from EgoDex camera frame to bimanual URDF frame.
# Camera (ARKit): X-right, Y-up, Z-toward-viewer
# URDF: X-right, Y-back, Z-up
_R_CAM_TO_URDF = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0],
    [0.0,  1.0,  0.0],
], dtype=np.float64)

# Mapping from EgoDex fingertip names to cuRobo link names
_TIP_NAME_MAP = {
    "leftThumbTip": "left_F1_TIP",
    "leftIndexFingerTip": "left_F2_TIP",
    "leftMiddleFingerTip": "left_F3_TIP",
    "rightThumbTip": "right_F1_TIP",
    "rightIndexFingerTip": "right_F2_TIP",
    "rightMiddleFingerTip": "right_F3_TIP",
}


# ---------------------------------------------------------------------------
# Main retargeting with cuRobo full-chain IK
# ---------------------------------------------------------------------------
def retarget_episode_for_sim(
    hdf5_path: str,
    ik_solver: CuRoboRetargetIK,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> dict:
    """
    Retarget one EgoDex episode using cuRobo 38-DOF full-chain IK.

    For each frame:
      1. Extract 6 fingertip positions from EgoDex data
      2. Transform to URDF world frame
      3. Solve 38-DOF IK with warm-start from previous frame

    Args:
        hdf5_path: path to EgoDex .hdf5 file
        ik_solver: CuRoboRetargetIK instance
        R_align: (3,3) rotation from camera frame to URDF frame
        t_align: (3,) translation from camera frame to URDF frame

    Returns dict with:
        joint_positions: (T, 38) robot joint angles
        ik_success: (T, 2) bool [left_ok, right_ok] (both set to same value)
        fingertip_targets: (T, 6, 3) target fingertip positions in URDF frame
    """
    ep = load_episode(hdf5_path)
    tfs = ep["transforms"]
    T = len(tfs["leftHand"])

    identity_aligner = CoordinateAligner()

    joint_positions = np.zeros((T, 38), dtype=np.float64)
    ik_success = np.zeros((T, 2), dtype=bool)
    fingertip_targets = np.zeros((T, 6, 3), dtype=np.float64)

    q_prev = None
    n_success = 0

    egodex_tip_names = LEFT_FINGER_TIPS + RIGHT_FINGER_TIPS

    for i in range(T):
        cam_ext_i = ep["cam_ext"][i]

        # Extract all 6 fingertip positions in URDF frame
        tips_urdf = {}
        for j, egodex_name in enumerate(egodex_tip_names):
            pos_cam = _world_to_robot_pos(
                tfs[egodex_name][i], cam_ext_i, identity_aligner
            )
            pos_urdf = R_align @ pos_cam + t_align
            fingertip_targets[i, j] = pos_urdf
            curobo_name = _TIP_NAME_MAP[egodex_name]
            tips_urdf[curobo_name] = pos_urdf

        # Solve 38-DOF IK with warm-start
        q, ok = ik_solver.solve(tips_urdf, q_prev=q_prev)
        joint_positions[i] = q
        ik_success[i] = [ok, ok]  # single solve covers both arms

        if ok:
            q_prev = q
            n_success += 1

        if i % 50 == 0 or i == T - 1:
            print(f"  Frame {i:4d}/{T}: IK {'OK' if ok else 'FAIL'} "
                  f"(cumulative: {n_success}/{i+1} = {n_success/(i+1):.1%})")

    return {
        "joint_positions": joint_positions,
        "ik_success": ik_success,
        "fingertip_targets": fingertip_targets,
    }


# ---------------------------------------------------------------------------
# Compute object-centric alignment from an episode
# ---------------------------------------------------------------------------
def compute_alignment(reference_hdf5: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute R, t from EgoDex camera frame -> bimanual URDF frame.

    Strategy: object-centric alignment.
    1. Estimate object center as mean of all fingertip positions (camera frame)
    2. Analytical rotation: camera -> URDF
    3. Translation: align rotated object center to URDF bottle position

    Returns (R, t) where urdf_pos = R @ cam_pos + t.
    """
    ep = load_episode(reference_hdf5)
    identity_aligner = CoordinateAligner()
    T = len(ep["transforms"]["leftHand"])

    all_tips = LEFT_FINGER_TIPS + RIGHT_FINGER_TIPS
    all_positions = []
    for i in range(T):
        cam_ext_i = ep["cam_ext"][i]
        for tip_name in all_tips:
            pos = _world_to_robot_pos(
                ep["transforms"][tip_name][i], cam_ext_i, identity_aligner
            )
            all_positions.append(pos)
    all_positions = np.array(all_positions)
    cam_object_center = all_positions.mean(axis=0)

    R = _R_CAM_TO_URDF.copy()
    t = _BOTTLE_POS_URDF - R @ cam_object_center

    # Verify
    cam_ext_0 = ep["cam_ext"][0]
    for tip_name in ["leftThumbTip", "rightThumbTip"]:
        pos_cam = _world_to_robot_pos(
            ep["transforms"][tip_name][0], cam_ext_0, identity_aligner
        )
        pos_urdf = R @ pos_cam + t
        dist = np.linalg.norm(pos_urdf - _BOTTLE_POS_URDF)
        print(f"  {tip_name} in URDF: ({pos_urdf[0]:.4f}, {pos_urdf[1]:.4f}, {pos_urdf[2]:.4f}), "
              f"dist to bottle: {dist:.4f}m")

    print(f"  Object center (cam): ({cam_object_center[0]:.4f}, {cam_object_center[1]:.4f}, {cam_object_center[2]:.4f})")
    print(f"  Object center (urdf): ({_BOTTLE_POS_URDF[0]:.4f}, {_BOTTLE_POS_URDF[1]:.4f}, {_BOTTLE_POS_URDF[2]:.4f})")
    return R, t


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_episodes(
    input_dir: Path,
    output_dir: Path,
    reference_episode: int = 0,
    min_ik_rate: float = 0.5,
):
    """Process all episodes in input_dir, save retargeted trajectories to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted(input_dir.glob("*.hdf5"))
    if not episodes:
        print(f"No .hdf5 files found in {input_dir}")
        return

    print(f"Found {len(episodes)} episodes")

    # Compute alignment from reference episode
    ref_path = input_dir / f"{reference_episode}.hdf5"
    if not ref_path.exists():
        ref_path = episodes[0]
    print(f"\nComputing alignment from {ref_path.name}...")
    R_align, t_align = compute_alignment(str(ref_path))

    # Build cuRobo IK solver
    print("\nInitializing cuRobo 38-DOF IK solver...")
    ik_solver = CuRoboRetargetIK()

    # Process each episode
    results_summary = []
    for ep_path in episodes:
        print(f"\n{'='*60}")
        print(f"Processing {ep_path.name}...")

        result = retarget_episode_for_sim(
            str(ep_path), ik_solver, R_align, t_align,
        )

        ik = result["ik_success"]
        both_rate = ik.all(axis=1).mean()
        print(f"  IK success: {both_rate:.1%}")
        print(f"  Frames: {len(ik)}")

        # Save
        out_name = ep_path.stem + "_sim.npz"
        out_path = output_dir / out_name
        np.savez_compressed(
            str(out_path),
            joint_positions=result["joint_positions"],
            ik_success=result["ik_success"],
            fingertip_targets=result["fingertip_targets"],
            episode=ep_path.stem,
            R_align=R_align,
            t_align=t_align,
        )
        print(f"  Saved: {out_path}")

        results_summary.append({
            "episode": ep_path.stem,
            "frames": len(ik),
            "ik_rate": both_rate,
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    good = 0
    for r in results_summary:
        status = "GOOD" if r["ik_rate"] >= min_ik_rate else "BAD"
        if status == "GOOD":
            good += 1
        print(f"  {r['episode']:>4s}: IK={r['ik_rate']:.1%} [{status}]")
    print(f"\nGood episodes (>={min_ik_rate:.0%} IK): {good}/{len(results_summary)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Retarget EgoDex to HAND-gat sim (v5: cuRobo)")
    parser.add_argument("--episode", type=str, help="Single episode .hdf5 path")
    parser.add_argument("--input_dir", type=str, help="Directory with .hdf5 episodes")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--ref_episode", type=int, default=0,
                        help="Reference episode for alignment")
    parser.add_argument("--min_ik_rate", type=float, default=0.5,
                        help="Minimum IK success rate to count as good")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.episode:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Computing alignment...")
        R_align, t_align = compute_alignment(args.episode)
        print("\nInitializing cuRobo 38-DOF IK solver...")
        ik_solver = CuRoboRetargetIK()
        print(f"\nProcessing {args.episode}...")
        result = retarget_episode_for_sim(
            args.episode, ik_solver, R_align, t_align,
        )
        ik = result["ik_success"]
        print(f"IK success: {ik.all(axis=1).mean():.1%}")
        out_path = output_dir / (Path(args.episode).stem + "_sim.npz")
        np.savez_compressed(
            str(out_path),
            joint_positions=result["joint_positions"],
            ik_success=result["ik_success"],
            fingertip_targets=result["fingertip_targets"],
            R_align=R_align,
            t_align=t_align,
        )
        print(f"Saved: {out_path}")
    elif args.input_dir:
        process_episodes(
            Path(args.input_dir), output_dir,
            reference_episode=args.ref_episode,
            min_ik_rate=args.min_ik_rate,
        )
    else:
        parser.error("Provide --episode or --input_dir")


if __name__ == "__main__":
    main()
