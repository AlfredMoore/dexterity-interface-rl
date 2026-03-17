# Tool: retarget_tool.py

## Purpose

Given a bottle position (x,y,z) in URDF world frame and an episode index, produce a
30Hz 38-DOF joint state trajectory for the bimanual Panda + Tesollo robot.

**File:** `baselines/ml-egodex-HAND/retarget/retarget_tool.py`

## Usage

```bash
# Must run inside handrl-policy container (cuRobo + CUDA required)

# Single episode
docker exec handrl-policy bash -c "cd /workspace && \
    /root/miniconda3/envs/policy/bin/python \
    baselines/ml-egodex-HAND/retarget/retarget_tool.py \
    --bottle_pos 0.042 0.0 -0.0215 \
    --episode_idx 0 --output models/egodex/traj_0.npz"

# All episodes
docker exec handrl-policy bash -c "cd /workspace && \
    /root/miniconda3/envs/policy/bin/python \
    baselines/ml-egodex-HAND/retarget/retarget_tool.py \
    --bottle_pos 0.042 0.0 -0.0215 \
    --all --output_dir models/egodex/trajs_realbot/"

# Custom data directory
docker exec handrl-policy bash -c "cd /workspace && \
    /root/miniconda3/envs/policy/bin/python \
    baselines/ml-egodex-HAND/retarget/retarget_tool.py \
    --bottle_pos 0.1 0.0 0.0 \
    --episode_idx 5 --data_dir /path/to/add_remove_lid/ --output traj.npz"
```

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--bottle_pos` | 3 floats | Yes | Bottle position (X Y Z) in URDF world frame (meters) |
| `--episode_idx` | int | No* | Single episode index to retarget |
| `--all` | flag | No* | Retarget all episodes in data_dir |
| `--data_dir` | str | No | Directory with EgoDex .hdf5 files (default: `models/egodex/test/add_remove_lid/`) |
| `--output` | str | No | Output .npz path (single episode mode) |
| `--output_dir` | str | No | Output directory (--all mode) |

*Must provide either `--episode_idx` or `--all`.

## Output Format (.npz)

| Key | Shape | Description |
|-----|-------|-------------|
| `joint_positions` | (T, 38) | Robot joint angles at 30Hz |
| `ik_success` | (T, 2) | IK success per frame (left, right) |
| `fingertip_targets` | (T, 6, 3) | Target fingertip positions in URDF frame |
| `bottle_pos` | (3,) | Bottle position used for alignment |
| `R_align` | (3, 3) | Rotation matrix: camera → URDF |
| `t_align` | (3,) | Translation vector: camera → URDF |
| `episode` | str | Episode name |
| `fps` | int | 30 |

## How It Works

1. **Object-centric alignment:** Computes R, t from EgoDex camera frame to URDF frame
   - `R = _R_CAM_TO_URDF` (analytical rotation: ARKit XYZ → URDF XYZ)
   - `t = bottle_pos - R @ cam_object_center`
   - `cam_object_center` = mean of all fingertip positions across all frames
2. **Per-frame IK:** For each of the T frames (30Hz):
   - Transform 6 fingertip positions from camera frame to URDF frame
   - cuRobo 38-DOF IK with warm-start from previous frame
3. **Output:** Save NPZ with joint positions and metadata

## Dependencies

- `retarget_episode.py` — `load_episode()`, `_world_to_robot_pos()`, `CoordinateAligner`
- `curobo_ik.py` — `CuRoboRetargetIK` (38-DOF full-chain IK, 6 fingertip targets)
- cuRobo + CUDA (only available in `handrl-policy` Docker container)

## Key Difference from retarget_for_sim.py

`retarget_for_sim.py` has a hardcoded `_BOTTLE_POS_URDF = np.array([0.042, 0.0, -0.0215])`.
`retarget_tool.py` accepts `--bottle_pos` as a parameter, allowing retargeting for
arbitrary bottle placements (e.g., real robot table positions).
