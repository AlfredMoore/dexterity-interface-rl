# External Repos & Dependencies

## EgoDex Dataset (`baselines/ml-egodex-HAND/`)

Human hand manipulation dataset (829 hours, 30Hz, 1080p + 3D SE3 poses for 68 joints).

- **HDF5 format:** `transforms/<joint_name>` → (N, 4, 4) SE3 in ARKit world frame
- **Key joints:** `leftThumbTip`, `leftIndexFingerTip`, `leftMiddleFingerTip` (same for right)
- **Camera intrinsics:** `[[736.63, 0, 960], [0, 736.63, 540], [0,0,1]]`
- **Task used:** `add_remove_lid` (26 episodes)
- **Data loading:** `baselines/ml-egodex-HAND/retarget/retarget_episode.py` → `load_episode()`

## dex-retargeting (`baselines/dex-retargeting/`)

Finger retargeting library using NLopt optimization.

- **Key class:** `SeqRetargeting` (wraps pinocchio FK + NLopt SLSQP optimizer)
- **Config format:** YAML with joint names, target links, optimization type
- **Our configs:** `baselines/ml-egodex-HAND/retarget/config/tesollo_{left,right}_fixed.yaml`
- **Requires:** standalone Tesollo URDF in `retarget/assets/tesollo_dg3f_{left,right}.urdf`

## cuRobo (`dep/curobo/`)

NVIDIA GPU-accelerated robotics library for IK, motion planning, collision checking.

- **IK API:** `IKSolver.solve_single(goal_pose, retract_config, seed_config, link_poses)`
- **Config files:** Robot config in `libs/robot_description/configs_curobo/robot/`, task config in `dep/curobo/src/curobo/content/configs/task/`
- **Custom retargeting config:** `bimanual_panda_tesollo-retargeting.yml` (38-DOF, 6 fingertip targets)
- **Known issue:** `vec_weight` in `PoseCost` is `[rot, rot, rot, pos, pos, pos]`, NOT `[pos, rot]`
- **Known issue:** CUDA graph mode breaks position-only IK; use `use_cuda_graph=False`

## HAND-gat (`/workspace/simulation/HAND-gat`)

IsaacLab RL environment for bimanual bottle cap manipulation.

- **Robot:** 2× Panda + 2× Tesollo = 38 DOF
- **Action space:** 38-D velocity control, EMA smoothed (α=0.75)
- **Observation:** 212-D (joint pos/target, fingertip pos, bottle/cap pos)
- **Task:** Unscrew bottle cap, dense reward, 67 bottle variants
- **Replay script:** `source/HAND/HAND/tasks/direct/hand/replay_trajectory.py`
  - `--teleport`: direct joint position write (no PD control)
  - `--video`: record video to `logs/replay/videos/`
- **Branch:** `baseline`
- **Runs on:** Host machine, `conda activate env_isaaclab`

## PromptDA (`dep/PromptDA-HAND`)

Prompt-based metric depth estimation. Used in `cv_node` for real robot deployment, not relevant for IL baseline retargeting.

## SAM2 (`dep/sam2-HAND`)

Segment Anything Model for object detection. Used in real robot pipeline, not for IL baseline.
