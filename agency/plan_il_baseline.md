# IL Baseline Plan: EgoDex Retargeting → HAND-gat Replay

## Goal

Build an **Imitation Learning baseline** for the bimanual unscrew-lid task, to compare against the RL policy. The IL baseline takes human hand manipulation episodes (EgoDex dataset) and retargets them into 38-DOF robot joint trajectories for replay in IsaacSim (HAND-gat environment).

**Success criterion:** bottle lid rotation > π rad in HAND-gat IsaacSim replay.

## Pipeline Overview

```
EgoDex .hdf5 episodes (human hand SE3 poses, 30Hz)
    ↓ coordinate alignment (camera → URDF world frame)
    ↓ fingertip extraction (3 per hand → 6 total)
    ↓ retargeting (arm IK + finger optimization)
    ↓ 38-DOF joint trajectories (.npz)
    ↓ HAND-gat replay (teleport mode + video)
    ↓ evaluate bottle cap rotation
```

## Robot Configuration

- **2× Franka Panda** (7 DOF each) — arm positioning
- **2× Tesollo DG-3F** (12 DOF each, 3 fingers × 4 joints) — dexterous manipulation
- **Total: 38 DOF** = [left_panda×7, left_tesollo×12, right_panda×7, right_tesollo×12]
- **URDF:** `libs/robot_description/rl/bimanual_panda_tesollo.urdf`
- **Bottle position in URDF frame:** (0.042, 0, -0.0215)

## Version History

| Version | Method | Fingertip Error | Result |
|---------|--------|----------------|--------|
| v2 | arm-first + free-joint hand retargeting | 15-25cm | No bottle contact |
| v3 | arm-first + fixed-base hand retargeting | Unreachable | Workspace incompatible |
| v4 | hand-first + arm-second | 3-14cm (best) | IK rate 20/26, sim tracking error, max_rot=0.01π |
| v5 | cuRobo 38-DOF full-chain IK | vec_weight bug | CUDA graph failure; works without CUDA graph but unverified on real data |
| **v6** | **Palm offset search + decoupled IK** | **TBD** | **Current plan — see sub-plan** |

### v5 Failure Analysis

cuRobo `PoseCost.vec_weight` order is `[rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]` (source: `dep/curobo/src/curobo/rollout/cost/pose_cost.py:164-165`). Our config had `[1,1,1,0,0,0]` which optimized rotation and ignored position entirely. After fixing to `[0,0,0,1,1,1]`:
- `use_cuda_graph=False`: perfect (position error ~3e-08m)
- `use_cuda_graph=True`: fails (~0.7m error, CUDA graph bakes wrong computation)

v5 code preserved in `baselines/ml-egodex-HAND/retarget/curobo_ik.py` as fallback.

## Current Plan: v6 Palm Offset Search

**See:** [plan_v6_palm_search.md](plan_v6_palm_search.md)

Core idea: systematically search for the optimal "correspondence point" on the robot palm that minimizes end-to-end fingertip error when used as the arm IK target.

## Future: Route Z (ManipTrans)

If Route A v6 fails, consider ManipTrans — a two-stage framework not dependent on direct retargeting:
- **Z-a:** DEXMANIPNET dataset (pre-retargeted robot trajectories for cap-twisting) → train Diffusion Policy
- **Z-b:** Stage 1 pretrained imitator + Stage 2 residual fine-tuning

## Data Locations

| Data | Path |
|------|------|
| EgoDex raw episodes | External dataset (26 episodes, `add_remove_lid` task) |
| Retargeted v4 | `models/egodex/retargeted_sim_v4/` |
| Retargeted v5 | `models/egodex/retargeted_sim_v5/` |
| Retargeted v6 (planned) | `models/egodex/retargeted_sim_v6/` |
| Replay videos | `models/vis/` |

## Key Code

| File | Purpose |
|------|---------|
| `baselines/ml-egodex-HAND/retarget/retarget_episode.py` | Core: PandaArmIKSolver, FixedBaseHandRetargeter, data loading |
| `baselines/ml-egodex-HAND/retarget/retarget_for_sim.py` | v5 retargeting (cuRobo-based) |
| `baselines/ml-egodex-HAND/retarget/curobo_ik.py` | v5 cuRobo 38-DOF IK wrapper |
| `baselines/ml-egodex-HAND/retarget/palm_search_retarget.py` | v6 palm search (to be created) |
| `HAND-gat/.../replay_trajectory.py` | IsaacSim replay with --teleport --video |
