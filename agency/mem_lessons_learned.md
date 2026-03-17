# Lessons Learned from v2–v5 Retargeting

## v2–v4: Decoupled Arm + Hand IK Doesn't Work

**Root cause:** Solving arm IK (7-DOF) and finger retargeting (12-DOF) independently means the wrist orientation from arm IK may be incompatible with finger reachability. The human wrist and robot wrist are fundamentally different structures.

**Key insight:** The arm IK target frame matters enormously. Targeting `panda_link8` (wrist flange) directly from human wrist position gives poor results because the human "wrist" (ARKit `leftHand` joint) doesn't correspond to any specific robot frame.

## v5: cuRobo Configuration Pitfalls

1. **vec_weight ordering:** cuRobo `PoseCost.vec_weight` is `[rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]` — the first 3 are rotation, last 3 are position. Source: `dep/curobo/src/curobo/rollout/cost/pose_cost.py:164-165`:
   ```python
   self.rot_weight = self.vec_weight[0:3]
   self.pos_weight = self.vec_weight[3:6]
   ```

2. **CUDA graph captures wrong computation:** When `use_cuda_graph=True`, the warmup call bakes the computation graph. For position-only IK with non-standard configs, this produces wrong results. Always test with `use_cuda_graph=False` first.

3. **Config file resolution:** `gradient_file` and `base_cfg_file` in `IKSolverConfig.load_from_robot_config()` are resolved relative to `cuRobo/content/configs/task/`, NOT as absolute paths. Custom configs must be copied there.

4. **CUDA graph signature must match:** If warmup call has different argument pattern (e.g., no `seed_config`) than production call, you get `ValueError: changing goal type, cuda graph reset not available`.

5. **CostBase disables if weight sum = 0:** In `cost_base.py:81`: `if torch.sum(self.weight) == 0.0: self.disable_cost()`. Setting all weight values to 0 disables the entire cost function silently.

## General Retargeting Lessons

- **Always FK-verify IK solutions:** Never trust IK success flags alone. Compute FK on the solution and compare with targets.
- **Human 5-finger → Robot 3-finger:** Only thumb/index/middle map to Tesollo F1/F2/F3. Ring and pinky are lost.
- **Warm-start is critical:** Frame-to-frame trajectory continuity depends on passing previous solution as `q_init`/`seed_config`.
- **Object-centric alignment:** Align coordinate frames using the object (bottle) center, not the hand itself. Human and robot hand geometry are too different for direct wrist alignment.
