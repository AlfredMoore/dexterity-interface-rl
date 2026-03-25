# cuRobo Skill (Official API + Project Mapping)

## Scope

This note captures the cuRobo APIs used in this repo for:

1. Collision-aware IK (`IKSolver`)
2. Collision-aware trajectory optimization (`TrajOptSolver`)
3. Collision querying (`RobotWorld`)
4. Optional dynamic lock-joint updates (`MotionGen.update_locked_joints`)

Primary references (official cuRobo docs):

- https://curobo.org/get_started/2a_python_examples.html
- https://curobo.org/_api/curobo.wrap.reacher.ik_solver.html
- https://curobo.org/_api/curobo.wrap.reacher.trajopt.html
- https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html
- https://curobo.org/_api/curobo.wrap.model.robot_world.html

Project code references:

- `libs/robot_motion_interface/src/robot_motion_interface/utils/kinematics.py`
- `baselines/ml-egodex-HAND/retarget/curobo_ik.py`
- `libs/robot_motion_interface/ros/src/robot_motion_interface_ros/robot_motion_interface_ros/test_node_for_curobo.py`

## API Cheat Sheet

### 1) IKSolver

Core construction pattern:

```python
ik_cfg = IKSolverConfig.load_from_robot_config(
    robot_cfg=robot_cfg,
    world_model=None,
    tensor_args=tensor_args,
    num_seeds=...,
    self_collision_check=True,
    self_collision_opt=True,
    use_cuda_graph=...,
    gradient_file=...,     # optional task gradient cfg
    base_cfg_file=...,     # optional base cfg
    regularization=True,
)
ik = IKSolver(ik_cfg)
```

Solve pattern:

```python
result = ik.solve_single(
    goal_pose,                 # primary ee_link target
    retract_config=...,        # strongly recommended for deterministic behavior
    seed_config=...,           # warm-start, continuity
    link_poses={...},          # secondary link goals
)
```

Key points:

- `solve_single` optimizes active joints in solver kinematic order.
- If `lock_joints` exists in robot cfg, returned solution is for unlocked DoF; you must merge back to full joint vector.
- For sequential retargeting, pass previous solution as `seed_config`.

### 2) TrajOptSolver

Core construction pattern:

```python
traj_cfg = TrajOptSolverConfig.load_from_robot_config(
    robot_cfg=robot_cfg,
    world_model=None,
    tensor_args=tensor_args,
    self_collision_check=True,
    self_collision_opt=True,
    trajopt_dt=...,
    traj_tsteps=...,
    interpolation_dt=...,
    interpolation_steps=...,
    num_seeds=...,
    grad_trajopt_iters=...,
    collision_activation_distance=...,
    use_cuda_graph=...,
)
trajopt = TrajOptSolver(traj_cfg)
```

Usage pattern:

- Build `Goal(current_state, goal_state)` with ordered `JointState`.
- Call `trajopt.solve_single(goal)`.
- Extract interpolated trajectory and `path_buffer_last_tstep`.

### 3) RobotWorld / RobotWorldConfig

For direct collision queries:

```python
rw_cfg = RobotWorldConfig.load_from_config(
    robot_config=robot_cfg,
    world_model=None,
    self_collision_activation_distance=...,
)
rw = RobotWorld(rw_cfg)
```

Use two configs in practice:

- `activation_distance=0.0`: actual penetration collision.
- `activation_distance=planning_margin`: planning safety margin collision.

### 4) MotionGen.update_locked_joints

`MotionGen` exposes `update_locked_joints(lock_joints, robot_config_dict)` for changing lock values between calls.

Practical constraint from official API doc + behavior:

- Changing lock values is supported.
- Changing lock topology (different number of locked joints) is only safe when tensor shape assumptions remain compatible.

## CUDA Graph Notes (Important)

From official API behavior and in-repo experience:

- `use_cuda_graph=True` is fast but sensitive to call signature and tensor shapes.
- If runtime call shape/signature changes (common in staged IK or changing constraints), prefer:
  - `use_cuda_graph=False` for robustness, or
  - strict fixed signatures + explicit graph reset handling.
- Warmup first call should match production argument structure.

## Project Mapping (Decision)

For EgoDex two-stage retargeting with collision constraints:

1. Stage-1 IK solver (`lock_fingers` cfg):
   - Goal links: virtual palms (`left_virtual_palm`, `right_virtual_palm`)
   - Active joints: dual-arm Panda joints.
2. Stage-2 IK solver (`lock_arms` cfg):
   - Goal links: six fingertips
   - Active joints: dual-hand finger joints.
   - Arm lock values updated to stage-1 output each frame (strict lock-arm behavior).
3. Frame-0 output is `q_pregrasp`.
4. `TrajOptSolver` plans `q_home -> q_pregrasp` collision-free path.
5. Then stream retargeted sequence frame-by-frame with warm-started seeds.

## Config Practices

For both stage configs:

- Keep same `collision_link_names`, `self_collision_ignore`, `self_collision_buffer`, sphere file.
- Only change:
  - `ee_link` / `link_names`
  - `lock_joints`
- Keep a single full 38-DoF `cspace.joint_names` and `retract_config` as canonical ordering.

## Validation Checklist

Per-episode minimum validation:

1. Stage-1 IK success rate
2. Stage-2 IK success rate
3. Combined success rate
4. Fingertip position error (mean / median / p95)
5. Collision rates (actual + margin)
6. `q_home -> q_pregrasp` trajopt success
