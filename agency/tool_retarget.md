# Tool: retarget_tool.py

## Purpose

Given a bottle position (x,y,z) in URDF world frame and an episode index, produce a
30Hz 38-DOF joint state trajectory for the bimanual Panda + Tesollo robot.

**File:** `baselines/ml-egodex-HAND/retarget/retarget_tool.py`

## Usage

```bash
# Must run inside handrl-policy container (pinocchio + dex-retargeting required)

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
| `--left_offset` | 3 floats | No | Left palm offset in `left_delto_base_link` frame (default: `-0.011 -0.011 0.037`) |
| `--right_offset` | 3 floats | No | Right palm offset in `right_delto_base_link` frame (default: `-0.001 -0.015 0.043`) |
| `--center_from` | str | No | Object center estimation source: `all6`, `left3`, or `right3` |
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
| `fingertip_errors` | (T, 6) | Per-fingertip FK error in URDF frame |
| `bottle_pos` | (3,) | Bottle position used for alignment |
| `R_align` | (3, 3) | Rotation matrix: camera → URDF |
| `t_align` | (3,) | Translation vector: camera → URDF |
| `left_offset` | (3,) | Left palm offset used by two-stage solver |
| `right_offset` | (3,) | Right palm offset used by two-stage solver |
| `episode` | str | Episode name |
| `fps` | int | 30 |

## How It Works

1. **Object-centric alignment:** Computes R, t from EgoDex camera frame to URDF frame
   - `R = _R_CAM_TO_URDF` (analytical rotation: ARKit XYZ → URDF XYZ)
   - `t = bottle_pos - R @ cam_object_center`
   - `cam_object_center` = mean of all fingertip positions across all frames
2. **Per-frame IK:** For each of the T frames (30Hz):
   - Transform 6 fingertip positions from camera frame to URDF frame
   - Palm-offset two-stage IK:
     - 7-DOF arm position IK to palm centroid
     - FK hand base pose
     - 12-DOF fixed-base finger retargeting in hand frame
3. **Output:** Save NPZ with joint positions and metadata

## Dependencies

- `retarget_episode.py` — `load_episode()`, `_world_to_robot_pos()`, `CoordinateAligner`
- `palm_search_retarget.py` — `retarget_episode_v6` (palm-offset two-stage IK)
- pinocchio + dex-retargeting (inside `handrl-policy` Docker container)

## Key Difference from retarget_for_sim.py

`retarget_for_sim.py` has a hardcoded `_BOTTLE_POS_URDF = np.array([0.042, 0.0, -0.0215])`.
`retarget_tool.py` accepts `--bottle_pos` as a parameter, allowing retargeting for
arbitrary bottle placements (e.g., real robot table positions).

## 2026-03 cuRobo Batch Retargeting Pipeline (Implemented)

This section documents the production pipeline implemented for
`add_remove_lid` + `screw_unscrew_bottle_cap`, with the following files:

- `baselines/ml-egodex-HAND/retarget/curobo_two_stage_batch.py`
- `libs/robot_description/configs_curobo/robot/bimanual_panda_tesollo_2stage_lock_fingers.yml`
- `libs/robot_description/configs_curobo/robot/bimanual_panda_tesollo_2stage_lock_arms.yml`

### 1) Runtime Environment

- Must run inside container: `docker exec handrl-policy ...`
- Dependencies: cuRobo + CUDA, pinocchio, dex-retargeting

### 2) Inputs and Task Scope

- Data source: `models/egodex/test/{task}/*.hdf5`
- Batch tasks:
  - `add_remove_lid`
  - `screw_unscrew_bottle_cap`
- Alignment parameters:
  - `bottle_pos = (0.042, 0.0, -0.0215)`
  - `center_from = all6`

### 3) Per-Episode Processing Flow

1. Load episode and compute camera->URDF alignment `(R_align, t_align)`.
2. Extract 6 fingertip targets per frame in URDF world frame.
3. Run two-stage IK with two solvers per frame:
   - **stage-1**: lock all finger joints, solve both arms to left/right virtual palms.
   - **stage-2**: strictly lock arm joints at stage-1 result, solve finger joints to 6 tip targets.
   - Warm-start strategy: use previous frame `q_prev`, then retry with `q_home/retract` fallback.
4. Use frame-0 IK result as `q_pregrasp`, then run cuRobo trajopt from `q_home -> q_pregrasp`.
5. Concatenate full trajectory:
   - `traj_full = traj_home_to_pregrasp + joint_positions_retarget`
6. Run baseline in the same episode (`retarget_tool` current two-stage) for same-metric comparison.
7. Save per-episode `.npz` and global JSON/CSV reports.

### 4) Output Paths (Updated)

> Default output root has been switched to `models/egodex/traj-retarging`

- Per-episode trajectories:
  - `models/egodex/traj-retarging/{task}/{episode}_curobo_2stage.npz`
- Aggregate reports:
  - `models/egodex/traj-retarging/reports/curobo_2stage_vs_baseline.json`
  - `models/egodex/traj-retarging/reports/curobo_2stage_vs_baseline.csv`

### 5) Key NPZ Fields Per Episode

- `joint_positions_retarget`
- `traj_home_to_pregrasp`
- `traj_full`
- `ik_success_stage1`
- `ik_success_stage2`
- `fingertip_errors`
- `collision_actual`
- `collision_margin`
- `q_home`
- `q_pregrasp`
- `task`, `episode`, `source="egodex"`, `bottle_pos`
- `failure_reason_codes`

### 6) Failure Reason Codes (for Debug/Review)

- `ik_stage1_all_failed` / `ik_stage1_partial_failed`
- `ik_stage2_all_failed` / `ik_stage2_partial_failed`
- `ik_both_all_failed`
- `collision_actual_detected`
- `collision_margin_detected`
- `trajopt_pregrasp_failed`
