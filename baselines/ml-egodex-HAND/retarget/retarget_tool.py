"""
Retarget tool: given a bottle position and episode index, output a 30Hz
38-DOF joint state trajectory for the bimanual Panda + Tesollo robot.

This is a thin wrapper around the existing retargeting pipeline
(retarget_for_sim.py) with a parameterizable bottle position instead of the
hardcoded _BOTTLE_POS_URDF.

Must run inside handrl-policy Docker container (cuRobo + CUDA required).

Usage:
    # Single episode
    python retarget_tool.py --bottle_pos 0.042 0.0 -0.0215 \
        --episode_idx 0 --output traj_0.npz

    # All episodes
    python retarget_tool.py --bottle_pos 0.042 0.0 -0.0215 \
        --all --output_dir trajs/

    # Custom data directory
    python retarget_tool.py --bottle_pos 0.1 0.0 0.0 \
        --episode_idx 5 --data_dir /path/to/add_remove_lid/ --output traj.npz
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
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]  # dexterity-interface-rl/
_DEFAULT_DATA_DIR = _REPO_ROOT / "models" / "egodex" / "test" / "add_remove_lid"

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
# Compute alignment with custom bottle position
# ---------------------------------------------------------------------------
def compute_alignment(
    hdf5_path: str,
    bottle_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (R, t) from EgoDex camera frame -> bimanual URDF frame,
    centering the object at the given bottle_pos.

    Strategy: object-centric alignment.
    1. Estimate object center as mean of all fingertip positions (camera frame)
    2. Analytical rotation: camera -> URDF
    3. Translation: align rotated object center to bottle_pos

    Returns (R, t) where urdf_pos = R @ cam_pos + t.
    """
    ep = load_episode(hdf5_path)
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
    t = bottle_pos - R @ cam_object_center

    print(f"  Object center (cam): ({cam_object_center[0]:.4f}, "
          f"{cam_object_center[1]:.4f}, {cam_object_center[2]:.4f})")
    print(f"  Bottle pos (urdf):   ({bottle_pos[0]:.4f}, "
          f"{bottle_pos[1]:.4f}, {bottle_pos[2]:.4f})")
    return R, t


# ---------------------------------------------------------------------------
# Retarget one episode
# ---------------------------------------------------------------------------
def retarget_episode(
    hdf5_path: str,
    ik_solver: CuRoboRetargetIK,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> dict:
    """
    Retarget one EgoDex episode using cuRobo 38-DOF full-chain IK.

    Returns dict with:
        joint_positions: (T, 38) robot joint angles
        ik_success: (T, 2) bool
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

        tips_urdf = {}
        for j, egodex_name in enumerate(egodex_tip_names):
            pos_cam = _world_to_robot_pos(
                tfs[egodex_name][i], cam_ext_i, identity_aligner
            )
            pos_urdf = R_align @ pos_cam + t_align
            fingertip_targets[i, j] = pos_urdf
            curobo_name = _TIP_NAME_MAP[egodex_name]
            tips_urdf[curobo_name] = pos_urdf

        q, ok = ik_solver.solve(tips_urdf, q_prev=q_prev)
        joint_positions[i] = q
        ik_success[i] = [ok, ok]

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
# Process single episode
# ---------------------------------------------------------------------------
def process_single(
    hdf5_path: Path,
    bottle_pos: np.ndarray,
    ik_solver: CuRoboRetargetIK,
    output_path: Path,
) -> dict:
    """Retarget a single episode and save to output_path."""
    print(f"\nComputing alignment from {hdf5_path.name} "
          f"with bottle_pos=({bottle_pos[0]:.4f}, {bottle_pos[1]:.4f}, {bottle_pos[2]:.4f})...")
    R_align, t_align = compute_alignment(str(hdf5_path), bottle_pos)

    print(f"Retargeting {hdf5_path.name}...")
    result = retarget_episode(str(hdf5_path), ik_solver, R_align, t_align)

    ik = result["ik_success"]
    ik_rate = ik.all(axis=1).mean()
    print(f"  IK success: {ik_rate:.1%} ({ik.all(axis=1).sum()}/{len(ik)} frames)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        joint_positions=result["joint_positions"],
        ik_success=result["ik_success"],
        fingertip_targets=result["fingertip_targets"],
        bottle_pos=bottle_pos,
        R_align=R_align,
        t_align=t_align,
        episode=hdf5_path.stem,
        fps=30,
    )
    print(f"  Saved: {output_path}")

    return {
        "episode": hdf5_path.stem,
        "frames": len(ik),
        "ik_rate": ik_rate,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Retarget EgoDex episodes with custom bottle position"
    )
    parser.add_argument(
        "--bottle_pos", type=float, nargs=3, required=True,
        metavar=("X", "Y", "Z"),
        help="Bottle position in URDF world frame (meters)",
    )
    parser.add_argument(
        "--episode_idx", type=int, default=None,
        help="Single episode index to retarget",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Retarget all episodes in data_dir",
    )
    parser.add_argument(
        "--data_dir", type=str, default=str(_DEFAULT_DATA_DIR),
        help="Directory containing EgoDex .hdf5 files "
             f"(default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .npz path (for single episode mode)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (for --all mode)",
    )
    args = parser.parse_args()

    bottle_pos = np.array(args.bottle_pos, dtype=np.float64)
    data_dir = Path(args.data_dir)

    if not args.all and args.episode_idx is None:
        parser.error("Provide --episode_idx or --all")

    # Initialize cuRobo IK solver
    print("Initializing cuRobo 38-DOF IK solver...")
    ik_solver = CuRoboRetargetIK()

    if args.episode_idx is not None:
        # Single episode
        hdf5_path = data_dir / f"{args.episode_idx}.hdf5"
        if not hdf5_path.exists():
            parser.error(f"Episode not found: {hdf5_path}")

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"{args.episode_idx}_retarget.npz")

        process_single(hdf5_path, bottle_pos, ik_solver, output_path)

    elif args.all:
        # All episodes
        episodes = sorted(data_dir.glob("*.hdf5"))
        if not episodes:
            parser.error(f"No .hdf5 files in {data_dir}")

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path("retargeted_trajs")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(episodes)} episodes in {data_dir}")
        results = []
        for ep_path in episodes:
            out_path = output_dir / f"{ep_path.stem}_retarget.npz"
            r = process_single(ep_path, bottle_pos, ik_solver, out_path)
            results.append(r)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        good = sum(1 for r in results if r["ik_rate"] >= 0.9)
        for r in results:
            status = "GOOD" if r["ik_rate"] >= 0.9 else "BAD"
            print(f"  {r['episode']:>4s}: {r['frames']:4d} frames, "
                  f"IK={r['ik_rate']:.1%} [{status}]")
        print(f"\nGood episodes (>=90% IK): {good}/{len(results)}")
        print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
