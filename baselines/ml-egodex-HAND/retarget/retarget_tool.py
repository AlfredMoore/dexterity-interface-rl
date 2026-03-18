"""
Retarget tool: given a bottle position and episode index, output a 30Hz
38-DOF joint state trajectory for the bimanual Panda + Tesollo robot.

This tool uses the palm-offset two-stage IK pipeline:
  1) 7-DOF arm IK to a virtual palm target
  2) 12-DOF fixed-base finger retargeting

The object alignment is parameterized by --bottle_pos.

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
from palm_search_retarget import retarget_episode_v6

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

_CENTER_TIP_SETS = {
    "all6": LEFT_FINGER_TIPS + RIGHT_FINGER_TIPS,
    "left3": LEFT_FINGER_TIPS,
    "right3": RIGHT_FINGER_TIPS,
}

# v6 palm-offset search results (also encoded in
# libs/robot_description/rl/bimanual_panda_tesollo_retarget.urdf):
#   left_virtual_palm_joint  xyz="-0.011 -0.011 0.037"
#   right_virtual_palm_joint xyz="-0.001 -0.015 0.043"
_DEFAULT_LEFT_OFFSET = np.array([-0.011, -0.011, 0.037], dtype=np.float64)
_DEFAULT_RIGHT_OFFSET = np.array([-0.001, -0.015, 0.043], dtype=np.float64)


# ---------------------------------------------------------------------------
# Compute alignment with custom bottle position
# ---------------------------------------------------------------------------
def compute_alignment(
    hdf5_path: str,
    bottle_pos: np.ndarray,
    center_from: str = "all6",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (R, t) from EgoDex camera frame -> bimanual URDF frame,
    centering the object at the given bottle_pos.

    Strategy: object-centric alignment.
    1. Estimate object center as mean fingertip positions (camera frame)
    2. Analytical rotation: camera -> URDF
    3. Translation: align rotated object center to bottle_pos

    Returns (R, t) where urdf_pos = R @ cam_pos + t.
    """
    if center_from not in _CENTER_TIP_SETS:
        raise ValueError(
            f"Invalid center_from='{center_from}', expected one of {list(_CENTER_TIP_SETS.keys())}"
        )
    ep = load_episode(hdf5_path)
    identity_aligner = CoordinateAligner()
    T = len(ep["transforms"]["leftHand"])

    all_tips = _CENTER_TIP_SETS[center_from]
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

    print(f"  Object center source: {center_from} ({len(all_tips)} tips)")
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
    left_offset: np.ndarray,
    right_offset: np.ndarray,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> dict:
    """
    Retarget one EgoDex episode using palm-offset two-stage IK.

    Returns dict with:
        joint_positions: (T, 38) robot joint angles
        ik_success: (T, 2) bool
        fingertip_errors: (T, 6) per-fingertip error in URDF frame
        fingertip_targets: (T, 6, 3) target fingertip positions in URDF frame
    """
    result = retarget_episode_v6(
        str(hdf5_path),
        np.asarray(left_offset, dtype=np.float64),
        np.asarray(right_offset, dtype=np.float64),
        R_align,
        t_align,
    )

    # Preserve fingertip target output for visualization/debug compatibility.
    ep = load_episode(hdf5_path)
    tfs = ep["transforms"]
    T = len(tfs["leftHand"])
    identity_aligner = CoordinateAligner()
    fingertip_targets = np.zeros((T, 6, 3), dtype=np.float64)
    egodex_tip_names = LEFT_FINGER_TIPS + RIGHT_FINGER_TIPS
    for i in range(T):
        cam_ext_i = ep["cam_ext"][i]
        for j, egodex_name in enumerate(egodex_tip_names):
            pos_cam = _world_to_robot_pos(
                tfs[egodex_name][i], cam_ext_i, identity_aligner
            )
            pos_urdf = R_align @ pos_cam + t_align
            fingertip_targets[i, j] = pos_urdf
    result["fingertip_targets"] = fingertip_targets
    return result


# Visualization (for debugging)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize_traj(result: dict, bottle_pos: np.ndarray, output_path: Path):
    """
    生成 3D 动画：
    - 金色星号：输入的 bottle_pos (目标瓶子位置)
    - 蓝色系点：左手指尖 (Thumb, Index, Middle)
    - 红色系点：右手指尖 (Thumb, Index, Middle)
    """
    targets = result["fingertip_targets"]  # (T, 6, 3)
    T = targets.shape[0]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义手指颜色
    # 顺序：leftThumb, leftIndex, leftMiddle, rightThumb, rightIndex, rightMiddle
    colors = ['#1f77b4', '#6699cc', '#aaccff', '#d62728', '#ff6666', '#ffcccc']
    labels = ["L_Thumb", "L_Idx", "L_Mid", "R_Thumb", "R_Idx", "R_Mid"]

    def update(frame):
        ax.clear()
        # 1. 绘制瓶子 (这是你在命令行输入的 bottle_pos)
        ax.scatter(bottle_pos[0], bottle_pos[1], bottle_pos[2], 
                   color='gold', s=200, marker='*', label='Bottle Target', edgecolors='black')
        
        # 2. 绘制 6 个指尖
        for i in range(6):
            pos = targets[frame, i]
            ax.scatter(pos[0], pos[1], pos[2], color=colors[i], s=50, 
                       label=labels[i] if frame == 0 else "")
            
        # 3. 设置范围 (以瓶子为中心前后 20cm)
        ax.set_xlim(bottle_pos[0] - 0.2, bottle_pos[0] + 0.2)
        ax.set_ylim(bottle_pos[1] - 0.2, bottle_pos[1] + 0.2)
        ax.set_zlim(bottle_pos[2] - 0.2, bottle_pos[2] + 0.2)
        
        ax.set_title(f"Frame {frame}/{T} | IK: {'OK' if result['ik_success'][frame].all() else 'FAIL'}")
        ax.set_xlabel("X (Robot)")
        ax.set_ylabel("Y (Robot)")
        ax.set_zlabel("Z (Robot)")
        if frame == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # 抽取帧数以加快生成速度（例如每隔 2 帧抽一次）
    ani = FuncAnimation(fig, update, frames=range(0, T, 2), interval=66)
    
    # 保存为 mp4 (需要容器内安装 ffmpeg) 或 gif
    save_path = output_path.with_suffix('.mp4')
    try:
        ani.save(str(save_path), writer='ffmpeg', fps=15)
        print(f"  [Visualization] Animation saved to: {save_path}")
    except Exception as e:
        save_path = output_path.with_suffix('.gif')
        ani.save(str(save_path), writer='pillow', fps=15)
        print(f"  [Visualization] FFmpeg failed, saved as GIF instead: {save_path}")
    
    plt.close(fig)

# ---------------------------------------------------------------------------
# Process single episode
# ---------------------------------------------------------------------------
def process_single(
    hdf5_path: Path,
    bottle_pos: np.ndarray,
    left_offset: np.ndarray,
    right_offset: np.ndarray,
    output_path: Path,
    center_from: str = "all6",
    visualize: bool = False,
    anim_path: Path | None = None,
) -> dict:
    """Retarget a single episode and save to output_path."""
    print(f"\nComputing alignment from {hdf5_path.name} "
          f"with bottle_pos=({bottle_pos[0]:.4f}, {bottle_pos[1]:.4f}, {bottle_pos[2]:.4f})...")
    R_align, t_align = compute_alignment(str(hdf5_path), bottle_pos, center_from=center_from)

    print(f"Using palm offsets: left={left_offset}, right={right_offset}")
    print(f"Retargeting {hdf5_path.name}...")
    result = retarget_episode(
        str(hdf5_path),
        left_offset=left_offset,
        right_offset=right_offset,
        R_align=R_align,
        t_align=t_align,
    )

    ik = result["ik_success"]
    ik_rate = ik.all(axis=1).mean()
    print(f"  IK success: {ik_rate:.1%} ({ik.all(axis=1).sum()}/{len(ik)} frames)")
    if "fingertip_errors" in result:
        print(f"  Mean tip error: {result['fingertip_errors'].mean():.4f}m")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        joint_positions=result["joint_positions"],
        ik_success=result["ik_success"],
        fingertip_targets=result["fingertip_targets"],
        fingertip_errors=result.get("fingertip_errors"),
        bottle_pos=bottle_pos,
        R_align=R_align,
        t_align=t_align,
        left_offset=left_offset,
        right_offset=right_offset,
        episode=hdf5_path.stem,
        fps=30,
    )
    print(f"  Saved: {output_path}")
    if visualize:
        anim_path = output_path.with_suffix(".mp4") if anim_path is None else anim_path
        print(f"  Generating visualization: {anim_path}")
        visualize_traj(result, bottle_pos, anim_path)

    return {
        "episode": hdf5_path.stem,
        "frames": len(ik),
        "ik_rate": ik_rate,
        "mean_tip_error": float(result["fingertip_errors"].mean())
        if "fingertip_errors" in result else None,
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
        "--center_from", type=str, default="all6", choices=["all6", "left3", "right3"],
        help=(
            "Which fingertip set to use for object-center inference: "
            "all6=both hands, left3=left thumb/index/middle, right3=right thumb/index/middle"
        ),
    )
    parser.add_argument(
        "--left_offset", type=float, nargs=3, default=_DEFAULT_LEFT_OFFSET.tolist(),
        metavar=("LX", "LY", "LZ"),
        help=(
            "Left palm offset (meters) in left_delto_base_link frame "
            f"(default: {_DEFAULT_LEFT_OFFSET.tolist()})"
        ),
    )
    parser.add_argument(
        "--right_offset", type=float, nargs=3, default=_DEFAULT_RIGHT_OFFSET.tolist(),
        metavar=("RX", "RY", "RZ"),
        help=(
            "Right palm offset (meters) in right_delto_base_link frame "
            f"(default: {_DEFAULT_RIGHT_OFFSET.tolist()})"
        ),
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
    parser.add_argument(
        "--viz", action="store_true",
        help="3D animation to validate bottle position and fingertip trajectories",
    )
    args = parser.parse_args()

    bottle_pos = np.array(args.bottle_pos, dtype=np.float64)
    left_offset = np.array(args.left_offset, dtype=np.float64)
    right_offset = np.array(args.right_offset, dtype=np.float64)
    data_dir = Path(args.data_dir)

    if not args.all and args.episode_idx is None:
        parser.error("Provide --episode_idx or --all")

    if args.episode_idx is not None:
        # Single episode
        hdf5_path = data_dir / f"{args.episode_idx}.hdf5"
        if not hdf5_path.exists():
            parser.error(f"Episode not found: {hdf5_path}")

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"{args.episode_idx}_retarget.npz")

        process_single(
            hdf5_path,
            bottle_pos,
            left_offset,
            right_offset,
            output_path,
            center_from=args.center_from,
            visualize=args.viz,
        )

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
            r = process_single(
                ep_path,
                bottle_pos,
                left_offset,
                right_offset,
                out_path,
                center_from=args.center_from,
                visualize=args.viz,
            )
            results.append(r)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        good = sum(1 for r in results if r["ik_rate"] >= 0.9)
        for r in results:
            status = "GOOD" if r["ik_rate"] >= 0.9 else "BAD"
            msg = (f"  {r['episode']:>4s}: {r['frames']:4d} frames, "
                   f"IK={r['ik_rate']:.1%} [{status}]")
            if r["mean_tip_error"] is not None:
                msg += f", mean_err={r['mean_tip_error']:.4f}m"
            print(msg)
        print(f"\nGood episodes (>=90% IK): {good}/{len(results)}")
        print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
