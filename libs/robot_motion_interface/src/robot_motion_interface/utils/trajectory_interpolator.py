"""
Velocity-limited trajectory interpolation utility.

Given a trajectory (T, DoF) at a fixed rate, insert linear waypoints between two
consecutive points whenever the implied per-step joint velocity exceeds a limit.

CLI example:
    python trajectory_interpolator.py \
        --traj_path models/egodex/traj-retarging/add_remove_lid/0_curobo_2stage.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def compute_max_step_velocity(traj: np.ndarray, hz: float) -> float:
    """Return the maximum implied per-step joint speed (rad/s)."""
    if traj.ndim != 2:
        raise ValueError(f"traj must be 2D, got shape={traj.shape}")
    if traj.shape[0] <= 1:
        return 0.0
    dt = 1.0 / hz
    return float(np.max(np.abs(np.diff(traj, axis=0)) / dt))


def interpolate_traj_by_velocity_limit(
    traj: np.ndarray,
    max_vel_rad_s: float = 1.0,
    hz: float = 30.0,
) -> np.ndarray:
    """
    Insert linear waypoints so each per-step joint speed is <= max_vel_rad_s.

    Args:
        traj: (T, D) trajectory array.
        max_vel_rad_s: maximum allowed per-step joint speed (rad/s).
        hz: sampling frequency of input trajectory.

    Returns:
        (T_new, D) interpolated trajectory.
    """
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 2:
        raise ValueError(f"traj must be 2D, got shape={traj.shape}")
    if traj.shape[0] == 0:
        return traj.copy()
    if max_vel_rad_s <= 0.0:
        raise ValueError(f"max_vel_rad_s must be > 0, got {max_vel_rad_s}")
    if hz <= 0.0:
        raise ValueError(f"hz must be > 0, got {hz}")

    if traj.shape[0] == 1:
        return traj.copy()

    dt = 1.0 / hz
    out: list[np.ndarray] = [traj[0]]

    for i in range(traj.shape[0] - 1):
        q0 = traj[i]
        q1 = traj[i + 1]
        delta = q1 - q0
        v_max = float(np.max(np.abs(delta) / dt))
        n_sub = max(1, int(math.ceil(v_max / max_vel_rad_s)))
        for k in range(1, n_sub + 1):
            alpha = float(k) / float(n_sub)
            out.append(q0 + alpha * delta)

    return np.asarray(out, dtype=np.float32)


def _make_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() != ".npz":
        raise ValueError(f"Input file must be .npz, got: {input_path}")
    stem = input_path.stem
    if stem.endswith("-interpolated"):
        out_stem = stem
    else:
        out_stem = f"{stem}-interpolated"
    return input_path.with_name(out_stem + input_path.suffix)


def interpolate_traj_file(
    traj_path: Path,
    max_vel_rad_s: float = 1.0,
    hz: float = 30.0,
) -> tuple[Path, int, int, float, float]:
    """
    Interpolate traj_full from an input NPZ and save sibling NPZ with suffix.

    Returns:
        output_path, original_steps, new_steps, vmax_before, vmax_after
    """
    if not traj_path.exists():
        raise FileNotFoundError(f"traj_path not found: {traj_path}")

    with np.load(str(traj_path), allow_pickle=True) as npz:
        data = {k: npz[k] for k in npz.files}

    if "traj_full" not in data:
        raise KeyError(f"traj_full not found in {traj_path}")

    traj = np.asarray(data["traj_full"], dtype=np.float32)
    if traj.ndim != 2:
        raise ValueError(f"traj_full must be 2D, got shape={traj.shape}")

    vmax_before = compute_max_step_velocity(traj, hz)
    traj_interp = interpolate_traj_by_velocity_limit(
        traj=traj,
        max_vel_rad_s=max_vel_rad_s,
        hz=hz,
    )
    vmax_after = compute_max_step_velocity(traj_interp, hz)

    data["traj_full"] = traj_interp.astype(np.float32)
    data["interpolation_applied"] = np.array(True)
    data["interpolation_max_vel_rad_s"] = np.array(float(max_vel_rad_s), dtype=np.float32)
    data["interpolation_hz"] = np.array(float(hz), dtype=np.float32)
    data["interpolation_original_steps"] = np.array(int(traj.shape[0]), dtype=np.int32)
    data["interpolation_new_steps"] = np.array(int(traj_interp.shape[0]), dtype=np.int32)

    out_path = _make_output_path(traj_path)
    np.savez_compressed(str(out_path), **data)

    return out_path, int(traj.shape[0]), int(traj_interp.shape[0]), vmax_before, vmax_after


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Velocity-limited interpolation for retargeted trajectory NPZ")
    p.add_argument("--traj_path", type=str, required=True, help="Path to input .npz trajectory file")
    p.add_argument("--max_vel_rad_s", type=float, default=1.0, help="Max allowed joint speed (rad/s)")
    p.add_argument("--hz", type=float, default=30.0, help="Trajectory sample frequency")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    traj_path = Path(args.traj_path).expanduser().resolve()
    out_path, n_old, n_new, vmax_before, vmax_after = interpolate_traj_file(
        traj_path=traj_path,
        max_vel_rad_s=float(args.max_vel_rad_s),
        hz=float(args.hz),
    )
    print(f"[saved] {out_path}")
    print(
        f"steps: {n_old} -> {n_new} | "
        f"max_step_vel(rad/s): {vmax_before:.4f} -> {vmax_after:.4f}"
    )


if __name__ == "__main__":
    main()

