# IL Baseline Plan: EgoDex Retargeting → Standalone Replay

## Goal

Build an **Imitation Learning baseline** for the bimanual unscrew-lid task, to compare against the RL policy. The IL baseline takes human hand manipulation episodes (EgoDex dataset) and retargets them into 38-DOF robot joint trajectories for replay in IsaacSim and on real hardware.

**Success criterion:** bottle lid rotation > π rad in IsaacSim replay; real robot replay produces contact-rich manipulation.

## Pipeline Overview

```
EgoDex .hdf5 episodes (human hand SE3 poses, 30Hz)
    ↓ coordinate alignment (camera → URDF world frame, object-centric)
    ↓ fingertip extraction (3 per hand → 6 total)
    ↓ retargeting: cuRobo 38-DOF full-chain IK (retarget_tool.py)
    ↓ 38-DOF joint trajectories (.npz, 30Hz)
    ↓ standalone IsaacLab replay (standalone_replay.py) OR real robot replay
    ↓ evaluate bottle cap rotation / video
```

## Robot Configuration

- **2× Franka Panda** (7 DOF each) — arm positioning
- **2× Tesollo DG-3F** (12 DOF each, 3 fingers × 4 joints) — dexterous manipulation
- **Total: 38 DOF** = [left_panda×7, left_tesollo×12, right_panda×7, right_tesollo×12]
- **URDF (combined):** `libs/robot_description/rl/bimanual_panda_tesollo.urdf`
- **URDFs (separate):** `HAND-gat/assets/DexRL_description/urdf/panda_w_tesollo_{left,right}.urdf`
- **Default bottle position in URDF frame:** (0.042, 0, -0.0215)

## Version History

| Version | Method | Fingertip Error | Result |
|---------|--------|----------------|--------|
| v2 | arm-first + free-joint hand retargeting | 15-25cm | No bottle contact |
| v3 | arm-first + fixed-base hand retargeting | Unreachable | Workspace incompatible |
| v4 | hand-first + arm-second | 3-14cm (best) | IK rate 20/26, sim tracking error, max_rot=0.01π |
| v5 | cuRobo 38-DOF full-chain IK | vec_weight bug | CUDA graph failure; works without CUDA graph but unverified on real data |
| **v6** | **cuRobo 38-DOF (vec_weight fixed)** | **1.82cm mean** | **26/26 episodes, IK 99.1%, standalone replay tested** |

### v5 Failure Analysis

cuRobo `PoseCost.vec_weight` order is `[rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]` (source: `dep/curobo/src/curobo/rollout/cost/pose_cost.py:164-165`). Our config had `[1,1,1,0,0,0]` which optimized rotation and ignored position entirely. After fixing to `[0,0,0,1,1,1]`:
- `use_cuda_graph=False`: perfect (position error ~3e-08m)
- `use_cuda_graph=True`: fails (~0.7m error, CUDA graph bakes wrong computation)

### v6 Results

- **Method:** cuRobo 38-DOF full-chain IK with `use_cuda_graph=False`, 6 fingertip position targets
- **IK success:** 99.1% across 26 episodes (26/26 retargeted)
- **Mean fingertip error:** 1.82cm (reasonable for position-only IK without orientation)
- **Output format:** `(T, 38)` joint positions at 30Hz + IK success flags + fingertip targets
- **Standalone replay test:** Episode 0 replayed successfully, bottle stays on table, video generated
- **Cap rotation in teleport mode ≈ 0** (expected — teleported fingers don't create physical contact forces)

## Replay Approaches

### A. Standalone IsaacLab Replay (current)
- `standalone_replay.py` — `SimulationContext` + `InteractiveScene`, no RL env
- Teleport mode: stiffness=0, damping=0, `write_joint_state_to_sim()`
- Pro: clean, no RL pipeline overhead, reliable teleport
- Con: teleported fingers don't generate contact forces → cap won't rotate
- Use case: visualization, trajectory validation, video recording

### B. HAND-gat DirectRLEnv Replay (abandoned)
- `HAND-gat/.../replay_trajectory.py` with `--teleport`
- Failed: `scene.write_data_to_sim()` caused bottle to clip through table
- PD mode: too much tracking lag from EMA + velocity scaling + clamp

### C. Real Robot Replay (planned)
- Use `retarget_tool.py` to generate trajectories with real bottle position
- Send 30Hz joint targets via ROS 2 `/target_joint_states`
- Real physics → actual contact forces → cap can rotate

## Tools

| Tool | File | Usage |
|------|------|-------|
| **retarget_tool** | `baselines/ml-egodex-HAND/retarget/retarget_tool.py` | Given bottle_pos + episode_idx → 30Hz 38-DOF trajectory |
| **standalone_replay** | `baselines/ml-egodex-HAND/replay/standalone_replay.py` | IsaacLab standalone trajectory replay + video |
| **estimate_bottle** | `baselines/ml-egodex-HAND/replay/estimate_bottle.py` | Estimate bottle dimensions from EgoDex fingertip data |

See: [tool_retarget.md](tool_retarget.md), [tool_standalone_replay.md](tool_standalone_replay.md), [tool_bottle_estimation.md](tool_bottle_estimation.md)

## Future: Route Z (ManipTrans)

If Route A doesn't achieve cap rotation on real robot, consider ManipTrans — a two-stage framework not dependent on direct retargeting:
- **Z-a:** DEXMANIPNET dataset (pre-retargeted robot trajectories for cap-twisting) → train Diffusion Policy
- **Z-b:** Stage 1 pretrained imitator + Stage 2 residual fine-tuning

## Data Locations

| Data | Path |
|------|------|
| EgoDex raw episodes | `models/egodex/test/add_remove_lid/` (26 episodes) |
| Retargeted v4 | `models/egodex/retargeted_sim_v4/` |
| Retargeted v5 | `models/egodex/retargeted_sim_v5/` |
| Retargeted v6 | `models/egodex/retargeted_sim_v6/` (26 episodes, cuRobo 38-DOF) |
| Replay videos | `baselines/ml-egodex-HAND/replay/replay_output/` |
| Bottle estimate | `baselines/ml-egodex-HAND/replay/bottle_estimate.json` |
| EgoDex bottle URDF | `baselines/ml-egodex-HAND/replay/bottle_egodex/model.urdf` |

## Key Code

| File | Purpose |
|------|---------|
| `baselines/ml-egodex-HAND/retarget/retarget_episode.py` | Core: PandaArmIKSolver, FixedBaseHandRetargeter, data loading |
| `baselines/ml-egodex-HAND/retarget/retarget_for_sim.py` | v6 retargeting (cuRobo-based, hardcoded bottle pos) |
| `baselines/ml-egodex-HAND/retarget/retarget_tool.py` | Retarget tool: parameterizable bottle position |
| `baselines/ml-egodex-HAND/retarget/curobo_ik.py` | cuRobo 38-DOF IK wrapper (CuRoboRetargetIK) |
| `baselines/ml-egodex-HAND/replay/standalone_replay.py` | IsaacLab standalone replay |
| `baselines/ml-egodex-HAND/replay/estimate_bottle.py` | Bottle dimension estimation |
| `baselines/ml-egodex-HAND/replay/bottle_egodex/model.urdf` | EgoDex-sized bottle URDF |
