"""
Standalone IsaacLab replay environment for retargeted EgoDex trajectories.

Uses SimulationContext + InteractiveScene (no RL env) to replay 38-DOF
joint trajectories with correct physics. Robot joints are teleported
directly via write_joint_state_to_sim().

Usage:
    # Single trajectory with video
    python standalone_replay.py --trajectory path/to/0_sim.npz --headless --video

    # Batch replay
    python standalone_replay.py --traj_dir path/to/retargeted_sim_v6/ --headless --video

    # Custom bottle position
    python standalone_replay.py --trajectory 0_sim.npz --bottle_pos 0.042 0.0 -0.0215
"""

import argparse
import os
import sys
from pathlib import Path
from math import pi

# ---------------------------------------------------------------------------
# IsaacLab AppLauncher (must come before any other isaaclab import)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Standalone replay of retargeted trajectories")
parser.add_argument("--trajectory", type=str, help="Single .npz trajectory file")
parser.add_argument("--traj_dir", type=str, help="Directory with *_sim.npz files")
parser.add_argument("--bottle_pos", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Bottle position in URDF frame (default: 0.042, 0, -0.0215)")
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--output_dir", type=str, default="replay_output",
                    help="Output directory for videos and metrics")
parser.add_argument("--fps", type=int, default=30, help="Video FPS")
parser.add_argument("--sim_dt", type=float, default=1.0/120,
                    help="Simulation timestep (default: 1/120)")
parser.add_argument("--decimation", type=int, default=4,
                    help="Physics steps per trajectory frame (default: 4)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras for video recording
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# IsaacLab imports (after AppLauncher)
# ---------------------------------------------------------------------------
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]  # dexterity-interface-rl/
_HAND_GAT = Path("/workspace/simulation/HAND-gat")
_ASSETS_DIR = _HAND_GAT / "assets"

_RIGHT_ARM_URDF = str(_ASSETS_DIR / "DexRL_description/urdf/panda_w_tesollo_right.urdf")
_LEFT_ARM_URDF = str(_ASSETS_DIR / "DexRL_description/urdf/panda_w_tesollo_left.urdf")
_BOTTLE_URDF = str(_THIS_DIR / "bottle_egodex" / "model.urdf")

# Default bottle position in URDF world frame
_DEFAULT_BOTTLE_POS = (0.042, 0.0, -0.0215)

# ---------------------------------------------------------------------------
# Joint name ordering (must match retarget output: 38 DOF)
# [left_panda*7, left_tesollo*12, right_panda*7, right_tesollo*12]
# ---------------------------------------------------------------------------
LEFT_PANDA_JOINTS = [f"left_panda_joint{i}" for i in range(1, 8)]
LEFT_TESOLLO_JOINTS = [
    "left_F1M1", "left_F1M2", "left_F1M3", "left_F1M4",
    "left_F2M1", "left_F2M2", "left_F2M3", "left_F2M4",
    "left_F3M1", "left_F3M2", "left_F3M3", "left_F3M4",
]
RIGHT_PANDA_JOINTS = [f"right_panda_joint{i}" for i in range(1, 8)]
RIGHT_TESOLLO_JOINTS = [
    "right_F1M1", "right_F1M2", "right_F1M3", "right_F1M4",
    "right_F2M1", "right_F2M2", "right_F2M3", "right_F2M4",
    "right_F3M1", "right_F3M2", "right_F3M3", "right_F3M4",
]

ALL_JOINT_NAMES = LEFT_PANDA_JOINTS + LEFT_TESOLLO_JOINTS + RIGHT_PANDA_JOINTS + RIGHT_TESOLLO_JOINTS


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
def make_scene_cfg(bottle_pos: tuple) -> type:
    """Create scene config class with the given bottle position."""

    # Quaternion for left arm: 180 deg yaw (w, x, y, z) = (0, 0, 0, 1) for IsaacSim convention
    # IsaacSim uses (w, x, y, z). 180° around Z => (cos(90°), 0, 0, sin(90°)) = (0, 0, 0, 1)
    left_rot = (0.0, 0.0, 0.0, 1.0)  # 180° yaw
    right_rot = (1.0, 0.0, 0.0, 0.0)  # identity

    # Workstation top surface at z=0 in URDF frame
    # workstation thickness = 0.045, so center at z = -0.0225
    ws_z = -0.0225

    # Bottle on top of workstation: workstation_top(0) + bottle_body_h/2(0.041)
    bottle_spawn_z = 0.041 + 0.005  # small margin above surface
    bottle_init = (bottle_pos[0], bottle_pos[1], bottle_spawn_z)

    @configclass
    class ReplaySceneCfg(InteractiveSceneCfg):

        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
        )

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(
                color=(0.92, 0.95, 1.0),
                intensity=350.0,
            ),
        )

        sun_light = AssetBaseCfg(
            prim_path="/World/SunLight",
            spawn=sim_utils.DistantLightCfg(
                color=(1.0, 0.97, 0.9),
                intensity=1200.0,
                angle=1.5,
            ),
        )

        right_arm: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/RightArm",
            spawn=UrdfFileCfg(
                asset_path=_RIGHT_ARM_URDF,
                fix_base=True,
                force_usd_conversion=True,
                merge_fixed_joints=False,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    target_type="none",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0.0, damping=0.0,
                    ),
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.558, -0.092, 0.0),
                rot=right_rot,
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -pi/4,
                    "panda_joint3": 0.0,
                    "panda_joint4": -3*pi/4,
                    "panda_joint5": 0.0,
                    "panda_joint6": pi/2,
                    "panda_joint7": pi/4,
                },
            ),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0,
                    damping=0.0,
                    effort_limit_sim=1e6,
                ),
            },
        )

        left_arm: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/LeftArm",
            spawn=UrdfFileCfg(
                asset_path=_LEFT_ARM_URDF,
                fix_base=True,
                force_usd_conversion=True,
                merge_fixed_joints=False,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    target_type="none",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0.0, damping=0.0,
                    ),
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.558, -0.092, 0.0),
                rot=left_rot,
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -pi/4,
                    "panda_joint3": 0.0,
                    "panda_joint4": -3*pi/4,
                    "panda_joint5": 0.0,
                    "panda_joint6": pi/2,
                    "panda_joint7": pi/4,
                },
            ),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0,
                    damping=0.0,
                    effort_limit_sim=1e6,
                ),
            },
        )

        workstation = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Workstation",
            spawn=sim_utils.CuboidCfg(
                size=(1.8288, 0.62865, 0.045),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, ws_z)),
        )

        bottle: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Bottle",
            spawn=UrdfFileCfg(
                asset_path=_BOTTLE_URDF,
                fix_base=False,
                force_usd_conversion=True,
                merge_fixed_joints=False,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    target_type="none",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0.0, damping=0.0,
                    ),
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=bottle_init,
            ),
            actuators={
                "joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0,
                    damping=0.0,
                    effort_limit_sim=1e6,
                ),
            },
        )

        # Camera for video recording (overhead view)
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,  # every step
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 1.5, 1.0),
                rot=(0.8536, -0.3536, 0.1464, 0.3536),  # looking down at table
                convention="world",
            ),
        )

    return ReplaySceneCfg


# ---------------------------------------------------------------------------
# Build joint index mapping
# ---------------------------------------------------------------------------
def build_joint_index_map(articulation, joint_names_subset: list[str]) -> list[int]:
    """Map joint names to articulation's internal joint indices."""
    art_joint_names = articulation.joint_names
    indices = []
    for name in joint_names_subset:
        if name in art_joint_names:
            indices.append(art_joint_names.index(name))
        else:
            print(f"  WARNING: joint '{name}' not found in articulation "
                  f"(available: {art_joint_names[:5]}...)")
            indices.append(-1)
    return indices


# ---------------------------------------------------------------------------
# Replay one trajectory
# ---------------------------------------------------------------------------
def replay_trajectory(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    traj_data: dict,
    args: argparse.Namespace,
) -> dict:
    """
    Replay one trajectory, optionally recording video.

    Returns dict with:
        max_cap_rotation: float (radians)
        frames: int
        video_path: str or None
    """
    joint_positions = traj_data["joint_positions"]  # (T, 38)
    T = joint_positions.shape[0]
    device = sim.device

    # Get articulations
    right_arm = scene["right_arm"]
    left_arm = scene["left_arm"]
    bottle = scene["bottle"]

    # Build joint index maps for each arm.
    # URDF joint names have no left_/right_ prefix (separate URDFs).
    # Strip prefix to match articulation joint names.
    right_urdf_names = [n.replace("right_", "") for n in RIGHT_PANDA_JOINTS + RIGHT_TESOLLO_JOINTS]
    left_urdf_names = [n.replace("left_", "") for n in LEFT_PANDA_JOINTS + LEFT_TESOLLO_JOINTS]
    right_idx_map = build_joint_index_map(right_arm, right_urdf_names)
    left_idx_map = build_joint_index_map(left_arm, left_urdf_names)

    # Trajectory slices: [left_panda*7, left_tesollo*12, right_panda*7, right_tesollo*12]
    left_traj_slice = slice(0, 19)   # left_panda(0:7) + left_tesollo(7:19)
    right_traj_slice = slice(19, 38)  # right_panda(19:26) + right_tesollo(26:38)

    # Get bottle b_joint index for cap rotation tracking
    b_joint_idx = None
    if "b_joint" in bottle.joint_names:
        b_joint_idx = bottle.joint_names.index("b_joint")

    max_cap_rotation = 0.0
    frames = []

    print(f"  Replaying {T} frames...")
    for t in range(T):
        # Extract joint targets for this frame
        left_q = joint_positions[t, left_traj_slice]
        right_q = joint_positions[t, right_traj_slice]

        # Build full joint position tensors (matching articulation's joint order)
        left_pos = torch.zeros(1, left_arm.num_joints, device=device)
        right_pos = torch.zeros(1, right_arm.num_joints, device=device)
        left_vel = torch.zeros(1, left_arm.num_joints, device=device)
        right_vel = torch.zeros(1, right_arm.num_joints, device=device)

        for i, idx in enumerate(left_idx_map):
            if idx >= 0:
                left_pos[0, idx] = float(left_q[i])
        for i, idx in enumerate(right_idx_map):
            if idx >= 0:
                right_pos[0, idx] = float(right_q[i])

        # Teleport joints
        left_arm.write_joint_state_to_sim(left_pos, left_vel)
        right_arm.write_joint_state_to_sim(right_pos, right_vel)

        # Step physics (decimation steps per trajectory frame)
        for _ in range(args.decimation):
            scene.write_data_to_sim()
            sim.step()
            scene.update(args.sim_dt)

        # Track bottle cap rotation
        if b_joint_idx is not None:
            cap_rot = abs(float(bottle.data.joint_pos[0, b_joint_idx]))
            max_cap_rotation = max(max_cap_rotation, cap_rot)

        # Capture frame for video
        if args.video:
            camera = scene["camera"]
            rgb = camera.data.output["rgb"]  # (1, H, W, 4)
            if rgb is not None and rgb.numel() > 0:
                frame = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
                frames.append(frame)

        if t % 30 == 0 or t == T - 1:
            cap_str = f", cap_rot={max_cap_rotation:.3f}rad" if b_joint_idx is not None else ""
            print(f"    Frame {t}/{T}{cap_str}")

    # Save video
    video_path = None
    if args.video and frames:
        import imageio
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ep_name = str(traj_data.get("episode", "unknown"))
        video_path = str(output_dir / f"replay_{ep_name}.mp4")
        writer = imageio.get_writer(video_path, fps=args.fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  Video saved: {video_path} ({len(frames)} frames)")

    return {
        "max_cap_rotation": max_cap_rotation,
        "frames": T,
        "video_path": video_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    bottle_pos = tuple(args_cli.bottle_pos) if args_cli.bottle_pos else _DEFAULT_BOTTLE_POS

    # Create simulation
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, 1.5, 1.0], target=[0.0, 0.0, 0.0])

    # Create scene
    SceneCfg = make_scene_cfg(bottle_pos)
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Reset
    sim.reset()
    print("[INFO] Scene ready.")

    # Collect trajectories
    traj_files = []
    if args_cli.trajectory:
        traj_files = [Path(args_cli.trajectory)]
    elif args_cli.traj_dir:
        traj_files = sorted(Path(args_cli.traj_dir).glob("*_sim.npz"))
    else:
        print("ERROR: Provide --trajectory or --traj_dir")
        simulation_app.close()
        return

    if not traj_files:
        print(f"No trajectory files found.")
        simulation_app.close()
        return

    print(f"Found {len(traj_files)} trajectories")

    # Replay each
    results = []
    for traj_path in traj_files:
        print(f"\n{'='*50}")
        print(f"Trajectory: {traj_path.name}")
        data = dict(np.load(str(traj_path), allow_pickle=True))

        # Reset scene for each episode
        sim.reset()

        result = replay_trajectory(sim, scene, data, args_cli)
        result["file"] = traj_path.name
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("REPLAY SUMMARY")
    print(f"{'='*60}")
    n_success = 0
    for r in results:
        success = r["max_cap_rotation"] > pi
        if success:
            n_success += 1
        status = "SUCCESS" if success else "---"
        print(f"  {r['file']:>20s}: cap_rot={r['max_cap_rotation']:.3f}rad "
              f"({r['frames']} frames) [{status}]")
    print(f"\nSuccess (cap_rot > pi): {n_success}/{len(results)} "
          f"({n_success/len(results)*100:.0f}%)")

    # Save metrics
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import json
    metrics = {
        "bottle_pos": list(bottle_pos),
        "n_trajectories": len(results),
        "n_success": n_success,
        "results": [{
            "file": r["file"],
            "max_cap_rotation": r["max_cap_rotation"],
            "frames": r["frames"],
        } for r in results],
    }
    metrics_path = output_dir / "replay_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
