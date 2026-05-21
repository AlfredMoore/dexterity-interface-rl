"""
Fit an affine compensation map for depth_feat bottle predictions:

      target = X @ W + b

where
    X = (depth_feat_body, depth_feat_cap)   ∈ R^6   (sim-trained predictor output)
    target = (apriltag_body, apriltag_cap) + z_bias_per_axis  ∈ R^6
    z_bias = 0.9144 + 0.045/2 = 0.93690 m

The apriltag origin is the tabletop top; the policy obs frame's z=0 is at
"tabletop top + workstation_top thickness / 2". Adding z_bias to apriltag z
lifts the ground truth into the same frame the depth_feat predictor already
lives in, so a single LSQ pass learns both the systematic position bias
and the depth_feat geometry distortion at once.

W (6x6) and b (6,) come out of np.linalg.lstsq on the augmented system
    [X | 1]  @  [W ; b]^T  =  Y

Run:
    /workspace/miniconda3_data/envs/env_isaaclab_da3/bin/python \
        -m robot_motion_interface.utils.lsq

The script reads data.npy from <dexterity-interface-rl>/data/apriltag_depthfeat_collect/
(produced by realsense_apriltag_collect.py + depth_feat_node_collect.py),
prints before/after residuals, then dumps W and b as numpy literals you can
paste straight into depth_feat_node.py's hardcoded compensation block.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


# z-bias: apriltag z is measured from tabletop top; policy obs (and the
# depth_feat predictor's output) lives in a frame raised by workstation_top
# thickness / 2 above the tabletop. To fit predictor output to "what policy
# wants to see" we add this to apriltag z before LSQ.
Z_BIAS = 0.9144 + 0.045 / 2.0    # = 0.93690


# Path resolution — same pattern as the other utils. data.npy lives at the
# project root: <dexterity-interface-rl>/data/apriltag_depthfeat_collect/.
spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent
PROJECT_ROOT = RMI_ROOT.parent.parent
DEFAULT_DATA_NPY = PROJECT_ROOT / "data" / "apriltag_depthfeat_collect" / "data.npy"


def _stack_pos(data: dict[str, np.ndarray], body_key: str, cap_key: str) -> np.ndarray:
    """Concat body[:, 3] and cap[:, 3] into a (N, 6) row-vector layout."""
    return np.concatenate([data[body_key], data[cap_key]], axis=1).astype(np.float64)


def _per_axis_summary(name: str, mat: np.ndarray) -> None:
    """Print per-axis mean / std / abs_max + body/cap 3D distance stats."""
    labels = ("body.x", "body.y", "body.z", "cap.x", "cap.y", "cap.z")
    body_dist = np.linalg.norm(mat[:, 0:3], axis=1)
    cap_dist  = np.linalg.norm(mat[:, 3:6], axis=1)
    print(f"  {name}")
    for i, label in enumerate(labels):
        col = mat[:, i]
        print(
            f"    {label}: mean={col.mean():+.5f}  std={col.std():.5f}  "
            f"abs_max={np.abs(col).max():.5f}"
        )
    print(f"    body 3D-dist:  mean={body_dist.mean():.5f}  max={body_dist.max():.5f}")
    print(f"    cap  3D-dist:  mean={cap_dist.mean():.5f}  max={cap_dist.max():.5f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LSQ fit of depth_feat -> (apriltag + z_bias) compensation."
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_NPY),
        help=f"data.npy path (default: {DEFAULT_DATA_NPY})",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = np.load(data_path, allow_pickle=True).item()
    n_apr = len(data["apriltag_body"])
    n_df  = len(data["depth_feat_body"])
    n = min(n_apr, n_df)
    if n < 7:
        raise ValueError(
            f"Need at least 7 paired samples for a 6x6 + bias affine fit, got {n}."
        )
    if n_apr != n_df:
        print(
            f"[warn] apriltag rows ({n_apr}) != depth_feat rows ({n_df}). "
            f"Using first {n} rows from each."
        )

    # X = predictor output (1.0~1.1 m on z, already in policy frame).
    X = _stack_pos(data, "depth_feat_body", "depth_feat_cap")[:n]   # (N, 6)
    # Y = apriltag gt mapped into the policy frame: add Z_BIAS to both z cols.
    apr = _stack_pos(data, "apriltag_body", "apriltag_cap")[:n]
    Y = apr.copy()
    Y[:, 2] += Z_BIAS    # body z column
    Y[:, 5] += Z_BIAS    # cap  z column

    # Augmented LSQ:  [X | 1] @ [W ; b]^T = Y   →   sol shape (7, 6).
    X_aug = np.concatenate([X, np.ones((n, 1))], axis=1)           # (N, 7)
    sol, _residuals_lsq, rank, sv = np.linalg.lstsq(X_aug, Y, rcond=None)
    W = sol[:6]    # (6, 6)
    b = sol[6]     # (6,)

    Y_hat_fit = X @ W + b
    residual_after = Y - Y_hat_fit        # what's left after compensation
    residual_before = Y - X               # identity baseline (no compensation)

    print(f"=== Setup ===")
    print(f"  data file:     {data_path}")
    print(f"  N samples:     {n}")
    print(f"  Z_BIAS:        {Z_BIAS:.5f}  (added to apriltag z so Y is in policy frame)")
    print(f"  LSQ rank:      {rank}/7   (full rank = 7)")
    print(f"  singular vals: {sv}")
    print()

    print(f"=== Residual BEFORE compensation (Y - X, identity W) ===")
    _per_axis_summary("identity baseline", residual_before)
    print()

    print(f"=== Residual AFTER compensation (Y - (X @ W + b)) ===")
    _per_axis_summary("LSQ-fitted", residual_after)
    print()

    print(f"=== Worst 5 rows after fit (by 6-D L2 of residual) ===")
    per_row = np.linalg.norm(residual_after, axis=1)
    worst = np.argsort(per_row)[-5:][::-1]
    for idx in worst:
        print(
            f"  row {idx+1:3d}  L2={per_row[idx]:.4f}  "
            f"residual={np.array2string(residual_after[idx], precision=4, suppress_small=True)}"
        )
    print()

    # ----- Copy-paste block -----
    print("=" * 70)
    print("Copy-paste into depth_feat_node.py to hardcode the compensation:")
    print("=" * 70)
    print()
    print("# LSQ-fitted affine compensation: corrected_pos = pred_pos @ _LSQ_W + _LSQ_B")
    print("# Input/output layout: [body_x, body_y, body_z, cap_x, cap_y, cap_z]  metres,")
    print("# in the policy obs frame (z = 0 at tabletop + workstation_top/2).")
    print(f"# Fitted from {n} samples in {data_path.name}.")
    print(f"# Z_BIAS applied to apriltag during fit: {Z_BIAS:.5f} m")
    print()
    print(f"_LSQ_W = np.array({W.tolist()!r}, dtype=np.float32)")
    print()
    print(f"_LSQ_B = np.array({b.tolist()!r}, dtype=np.float32)")
    print()
    print("# tensor form (depth_feat_node is torch-based):")
    print("# self._lsq_W = torch.tensor(_LSQ_W, device=self.device)")
    print("# self._lsq_B = torch.tensor(_LSQ_B, device=self.device)")
    print("# corrected = pred_pos @ self._lsq_W + self._lsq_B")


if __name__ == "__main__":
    main()
