# Tool: standalone_replay.py

## Purpose

Replay retargeted 38-DOF joint trajectories in an IsaacLab standalone environment
(no RL env). Teleports joint positions each frame, records video, and measures
bottle cap rotation.

**File:** `baselines/ml-egodex-HAND/replay/standalone_replay.py`

## Usage

```bash
# Must run on host machine with env_isaaclab conda env
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd /workspace/simulation/dexterity-interface-rl

# Single trajectory (headless + video)
python baselines/ml-egodex-HAND/replay/standalone_replay.py \
    --trajectory models/egodex/retargeted_sim_v6/0_sim.npz \
    --headless --video

# Batch replay (all trajectories in directory)
python baselines/ml-egodex-HAND/replay/standalone_replay.py \
    --traj_dir models/egodex/retargeted_sim_v6/ \
    --headless --video

# With GUI (only works with display)
python baselines/ml-egodex-HAND/replay/standalone_replay.py \
    --trajectory models/egodex/retargeted_sim_v6/0_sim.npz
```

## Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--trajectory` | str | Path to single .npz trajectory file |
| `--traj_dir` | str | Directory of .npz files for batch replay |
| `--headless` | flag | Run without GUI (required for remote/SSH) |
| `--video` | flag | Record MP4 video of replay |
| `--output_dir` | str | Output directory (default: `replay_output/`) |

## Scene Configuration

| Component | Position | Details |
|-----------|----------|---------|
| Right arm | (0.558, -0.092, 0.0), identity rot | `panda_w_tesollo_right.urdf` |
| Left arm | (-0.558, -0.092, 0.0), 180° yaw | `panda_w_tesollo_left.urdf` |
| Workstation | (0.0, 0.0, -0.0225) | Kinematic cuboid 1.8288×0.6287×0.045 |
| Bottle | (0.042, 0.0, 0.065) | EgoDex cylinder bottle URDF |
| Ground | z=0 plane | |
| Camera | overhead, looking down | 640×480, for video recording |
| Sim dt | 1/120s | 4 sim steps per trajectory frame (30Hz→120Hz) |

## Output

- **Video:** `replay_output/replay_{episode}.mp4` (30fps)
- **Metrics:** `replay_output/replay_{episode}_metrics.json`
  - `max_cap_rotation_rad`: maximum b_joint rotation
  - `frames`: number of trajectory frames
  - `episode`: episode name

## Architecture

Uses IsaacLab standalone mode (see [skill_isaaclab_standalone.md](skill_isaaclab_standalone.md)):
- `SimulationContext` + `InteractiveScene` (no `ManagerBasedEnv` or `DirectRLEnv`)
- Each arm loaded as separate `ArticulationCfg` with its own URDF
- Bottle loaded as `ArticulationCfg` (has revolute b_joint for cap)
- Teleport: stiffness=0, damping=0, `write_joint_state_to_sim()` each frame
- Camera sensor captures RGB frames for video

## Joint Name Mapping

The 38-DOF trajectory uses prefixed names: `left_panda_joint1`, `right_F1M1`, etc.
Each arm URDF has unprefixed names: `panda_joint1`, `F1M1`, etc.
The script strips `left_`/`right_` prefix when mapping trajectory columns to
per-arm articulation joints.

## Known Behaviors

- **Cap rotation ≈ 0 in teleport mode:** Expected — teleported fingers don't generate
  physical contact forces. This tool is for visualization/validation, not physics evaluation.
- **First run slow:** Shader compilation on first `sim.reset()` with cameras takes minutes.
- **Bottle on table:** Bottle stays stable due to kinematic workstation + proper collision.
