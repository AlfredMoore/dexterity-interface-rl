# Tool: estimate_bottle.py

## Purpose

Estimate bottle/cup dimensions from EgoDex fingertip annotations. EgoDex has no
explicit object annotations, so dimensions are inferred from fingertip positions
during grasping and lid manipulation.

**File:** `baselines/ml-egodex-HAND/replay/estimate_bottle.py`

## Usage

```bash
docker exec handrl-policy bash -c "cd /workspace && \
    /root/miniconda3/envs/policy/bin/python \
    baselines/ml-egodex-HAND/replay/estimate_bottle.py \
    --data_dir models/egodex/test/add_remove_lid/"
```

## Method

1. Load all EgoDex episodes, extract 6 fingertip positions per frame
2. Compute object center as mean of all fingertip positions (camera frame)
3. Transform to object-centric URDF frame (object at origin)
4. Left hand → cap manipulation: estimate cap radius from XY distance, cap height from Z range
5. Right hand → body grasp: estimate body radius from XY distance, body height from Z range
6. Aggregate across episodes using median
7. Subtract finger thickness offset (~12mm) from radii

## Results (26 episodes)

| Dimension | Value | Description |
|-----------|-------|-------------|
| cap_r | 0.055m | Cap/lid radius |
| cap_h | 0.030m | Cap/lid height (tighter estimate) |
| body_r | 0.060m | Body radius |
| body_h | 0.082m | Body height |

**Comparison with HAND-gat bottle (1.5x scale):**
- HAND-gat: cap_r=0.048m, cap_h=0.032m, body_r=0.064m, body_h=0.095m
- EgoDex: slightly wider cap, slightly narrower body, shorter overall

## Output Files

- `baselines/ml-egodex-HAND/replay/bottle_estimate.json` — JSON with all estimates
- `baselines/ml-egodex-HAND/replay/bottle_egodex/model.urdf` — Cylinder-based URDF

## Bottle URDF Structure

Matches HAND-gat bottle structure:
- `link2` (body): cylinder r=0.060, l=0.082
- `link1` (cap): cylinder r=0.055, l=0.030
- `b_joint`: revolute z-axis at z=0.056 (body_h/2 + cap_h/2)
- `brake` + `brake_joint`: friction element (prismatic z)
