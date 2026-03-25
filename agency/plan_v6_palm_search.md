# Sub-Plan: v6 Palm Offset Search + Decoupled IK

## Problem

Previous retargeting versions failed because the arm IK target frame (`panda_link8`) has no correct correspondence to the human palm center. Different wrist orientations lead to vastly different finger reachability. We need to find the optimal "virtual palm point" on the robot hand that minimizes end-to-end fingertip error.

## Approach

1. Define a virtual reference point at offset (dx, dy, dz) from `delto_base_link` (palm hub)
2. Arm IK (7-DOF): position this virtual point at the human palm centroid
3. Finger retargeting (12-DOF): optimize finger angles in hand-base frame
4. FK verify: measure actual fingertip positions vs human targets
5. Grid search over (dx, dy, dz) to minimize fingertip error

## Robot Kinematic Chain

```
left_panda_link8 (wrist, 7-DOF arm endpoint)
    ↓ [fixed: xyz=0 0 0.106, rpy=0 0 -0.785]
    left_delto_base_link (palm hub)              ← search origin
        ├─ F1_01 [xyz=0.0265 0 0]               ← thumb base
        ├─ F2_01 [xyz=-0.01334 0.023 0]          ← index base
        └─ F3_01 [xyz=-0.01334 -0.023 0]         ← middle base
        Each finger: 4 revolute joints, chain length ~80-100mm
```

Verified: all frames (delto_base_link, F*_TIP) exist in pinocchio reduced model (left arm only, 7 active joints).

## Implementation

### New file: `baselines/ml-egodex-HAND/retarget/palm_search_retarget.py`

#### 1. PalmOffsetIKSolver

```python
class PalmOffsetIKSolver:
    """7-DOF arm IK targeting a virtual palm frame offset from delto_base_link."""

    def __init__(self, side: str, palm_offset: np.ndarray):
        # Build PandaArmIKSolver (loads reduced 7-DOF model)
        # Find delto_base_link frame in reduced model
        # pin.addFrame() at delto_base_link + palm_offset
        # model.createData() to refresh

    def solve_position(self, target_pos, q_init=None) -> (np.ndarray, bool):
        # Position-only IK to virtual palm frame (reuse solve_position_only logic)

    def fk_hand_base(self, q) -> np.ndarray:
        # Returns (4,4) SE3 of delto_base_link (for transforming fingertip targets)
```

Key detail: `addFrame` placement = `delto_base_link.placement * SE3(I, offset)`.

#### 2. evaluate_candidate()

```python
def evaluate_candidate(side, palm_offset, human_palm_positions,
                       human_tips_urdf, frame_indices) -> dict:
    """
    For one candidate offset, evaluate fingertip error across sampled frames.

    Per frame:
      1. arm IK: virtual palm → human palm centroid
      2. FK: get delto_base_link SE3
      3. Transform human fingertips to hand-base frame
      4. FixedBaseHandRetargeter → 12-DOF finger angles
      5. Full FK (full pinocchio model) → actual fingertip world positions
      6. Error = L2(actual - human targets)

    Returns: {mean_tip_error, max_tip_error, ik_success_rate, per_frame_errors}
    """
```

#### 3. Grid Search

**Coarse pass** (delto_base_link local frame):
- x: [-0.02, 0.04], step 1cm → 7 values
- y: [-0.03, 0.03], step 1cm → 7 values
- z: [0.00, 0.08], step 1cm → 9 values
- Total: 441 candidates
- Evaluate every 10th frame from reference episode (~40 frames)
- **Est. time: ~2 min per hand**

**Fine pass:**
- Top-5 candidates from coarse, ±5mm range, step 2mm
- 125 × 5 = 625 candidates, every 5th frame
- **Est. time: ~3 min per hand**

**Left/right hands searched independently** (different task roles).

**Palm centroid = mean of 3 fingertip positions per hand per frame** (URDF world frame).

#### 4. Full Retargeting

```python
def retarget_episode_v6(hdf5_path, left_offset, right_offset, R_align, t_align):
    # Per frame:
    #   arm IK with warm-start (q_init = q_prev) for trajectory continuity
    #   finger retargeting
    # Output: (T, 38) joint_positions + (T, 2) ik_success + (T, 6) fingertip_errors
```

#### 5. CLI

```bash
# Search only
docker exec handrl-policy ... python palm_search_retarget.py \
    --input_dir /path/to/episodes --search_only --ref_episode 0

# Full retarget with known offsets
docker exec handrl-policy ... python palm_search_retarget.py \
    --input_dir /path/to/episodes --output /path/to/out \
    --left_offset "[0.01, 0.0, 0.04]" --right_offset "[0.01, 0.0, 0.04]"

# Replay (host, env_isaaclab)
python replay_trajectory.py --teleport --video --traj_dir models/egodex/retargeted_sim_v6/
```

## Dependencies (reuse existing code)

| Component | Source | Usage |
|-----------|--------|-------|
| `PandaArmIKSolver` | `retarget_episode.py:109-246` | Extend with virtual frame for arm IK |
| `FixedBaseHandRetargeter` | `retarget_episode.py:327-375` | Direct reuse for finger retargeting |
| `compute_alignment()` | `retarget_for_sim.py:148-191` | Camera → URDF coordinate alignment |
| `load_episode()` | `retarget_episode.py` | EgoDex HDF5 data loading |
| `_world_to_robot_pos()` | `retarget_episode.py` | ARKit → camera frame transform |
| `PinocchioBimanualFK` | `libs/.../kinematics.py:65-167` | Full FK verification (optional) |
| `tesollo_{left,right}_fixed.yaml` | `retarget/config/` | FixedBaseHandRetargeter configs |

## Verification

1. **FK error < 3cm** for all 6 fingertips vs human targets
2. **IK success rate > 80%** (arm + finger combined)
3. **Trajectory continuity**: adjacent frame joint delta < 0.1 rad
4. **HAND-gat replay**: bottle cap rotation > π rad
5. **Cross-episode validation**: evaluate best offset on 3-5 episodes before committing

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Position-only arm IK gives inconsistent wrist orientation | Frame-to-frame warm-start (q_init = q_prev) |
| Optimal offset varies per episode | Evaluate on 3-5 episodes, use robust average |
| Search minimum on grid boundary | Expand search range if detected |
| FixedBaseHandRetargeter can't reach targets in hand frame | Log per-finger error to diagnose; consider fallback to cuRobo (no CUDA graph) |

## Fallback

If v6 palm search still can't achieve < 3cm error, revert to cuRobo 38-DOF IK with `use_cuda_graph=False` (verified working with 3e-08m precision on trivial targets). Code: `baselines/ml-egodex-HAND/retarget/curobo_ik.py`.
