# Execution Environment

## Docker Container: `handrl-policy`

All Python/IK/retargeting code runs inside this container.

```bash
# Run a script
docker exec handrl-policy bash -c "cd /workspace/baselines/ml-egodex-HAND/retarget && \
    /root/miniconda3/envs/policy/bin/python script.py"

# Interactive shell
docker exec -it handrl-policy bash
```

- **Python:** `/root/miniconda3/envs/policy/bin/python` (Python 3.11)
- **Mount:** Host repo at `/workspace/simulation/dexterity-interface-rl` → Container `/workspace`
- **Available in container:** pinocchio, dex-retargeting, cuRobo, torch (CUDA), numpy, scipy
- **NOT in container:** IsaacSim, IsaacLab

## IsaacSim: Host Machine

IsaacSim runs on the host (not Docker), requires GPU + specific system deps.

```bash
conda activate env_isaaclab

# Run replay
cd /workspace/simulation/HAND-gat
python source/HAND/HAND/tasks/direct/hand/replay_trajectory.py \
    --teleport --video --traj_dir /path/to/trajectories
```

- **Conda env:** `env_isaaclab`
- **Scene config:** `/workspace/simulation/HAND-gat`
- **Video output:** `HAND-gat/logs/replay/videos/`
- **Do NOT use `python -m`** — import order matters (SimulationApp must init first)

## cuRobo

- Lives in `dep/curobo/` (editable install inside container)
- Task configs resolved at: `dep/curobo/src/curobo/content/configs/task/`
- Custom configs must be COPIED there for cuRobo to find them
- **CUDA graph bug:** `use_cuda_graph=True` bakes wrong computation for position-only IK. Use `False` for correctness.
- **vec_weight order:** `[rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]` (source: `pose_cost.py:164-165`)
