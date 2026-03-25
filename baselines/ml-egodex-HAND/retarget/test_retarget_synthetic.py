"""
Synthetic smoke-test for the retargeting pipeline.

Creates a fake EgoDex-format HDF5 with T=60 frames of plausible hand motion,
then runs the full retargeting pipeline and prints diagnostics.

No real EgoDex data needed.

Run:
    docker exec handrl-policy bash -c "
        cd /workspace &&
        /root/miniconda3/envs/policy/bin/python \
            baselines/ml-egodex-HAND/retarget/test_retarget_synthetic.py
    "
"""

import sys as _sys
_sys.path = [p for p in _sys.path if "openrobots" not in p]

import tempfile
from pathlib import Path

import h5py
import numpy as np

# Make the retarget package importable when run from repo root
_HERE = Path(__file__).parent
if str(_HERE.parent) not in _sys.path:
    _sys.path.insert(0, str(_HERE.parent))

import pinocchio as pin

from retarget.retarget_episode import (
    CoordinateAligner,
    PandaArmIKSolver,
    build_retargeters,
    retarget_episode,
)

# ---------------------------------------------------------------------------
# Synthetic HDF5 generator
# ---------------------------------------------------------------------------

T = 60   # number of frames
FPS = 30.0

# Fingertip offsets relative to wrist (rough hand geometry)
_TIP_OFFSETS = np.array([
    [ 0.00,  0.02,  0.10],   # thumb  (F1)
    [ 0.00, -0.02,  0.12],   # index  (F2)
    [ 0.00, -0.05,  0.12],   # middle (F3)
])  # shape (3, 3)


def _fk_ee_pose(ik_solver: PandaArmIKSolver) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, rot) of the EE at the default (ready) joint configuration."""
    q = ik_solver.q_default.copy()
    pin.forwardKinematics(ik_solver.model, ik_solver.data, q)
    pin.updateFramePlacements(ik_solver.model, ik_solver.data)
    se3 = ik_solver.data.oMf[ik_solver.ee_frame_id]
    return se3.translation.copy(), se3.rotation.copy()


def _make_se3(pos: np.ndarray, rot: np.ndarray = None) -> np.ndarray:
    """Build a (4, 4) SE(3) matrix from a 3D position and optional 3×3 rotation."""
    tf = np.eye(4)
    tf[:3, :3] = rot if rot is not None else np.eye(3)
    tf[:3,  3] = pos
    return tf


def _small_rot(amplitude: float = 0.05) -> np.ndarray:
    """Return a random near-identity 3×3 rotation (Rodriguez formula)."""
    axis  = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-9
    angle = np.random.uniform(-amplitude, amplitude)
    K = np.array([
        [        0, -axis[2],  axis[1]],
        [ axis[2],         0, -axis[0]],
        [-axis[1],  axis[0],        0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def make_synthetic_hdf5(
    path: str,
    left_ik: PandaArmIKSolver,
    right_ik: PandaArmIKSolver,
    seed: int = 0,
) -> None:
    """Write a minimal EgoDex-compatible HDF5 to *path*.

    Wrist nominal poses are derived from each arm's FK at the neutral
    configuration, so they are guaranteed to be reachable.
    """
    rng = np.random.default_rng(seed)

    left_nom_pos,  left_nom_rot  = _fk_ee_pose(left_ik)
    right_nom_pos, right_nom_rot = _fk_ee_pose(right_ik)
    print(f"  Left  EE nominal pos (FK): {left_nom_pos.round(3)}")
    print(f"  Right EE nominal pos (FK): {right_nom_pos.round(3)}")

    # Camera extrinsics: identity for all frames (world == camera in this test).
    cam_ext = np.tile(np.eye(4), (T, 1, 1))

    # Camera intrinsics (EgoDex default).
    cam_int = np.array([
        [736.63,   0.0, 960.0],
        [  0.0, 736.63, 540.0],
        [  0.0,   0.0,    1.0],
    ])

    # Build per-frame wrist SE(3) with gentle position-only drift from FK nominal pose.
    # Rotation is kept fixed (realistic: wrist orientation changes slowly relative to position).
    def _wrist_traj(nom_pos: np.ndarray, nom_rot: np.ndarray) -> np.ndarray:
        traj = np.zeros((T, 4, 4))
        pos  = nom_pos.copy()
        for i in range(T):
            pos += rng.uniform(-0.003, 0.003, 3)   # 3 mm random walk
            traj[i] = _make_se3(pos, nom_rot)       # rotation fixed at FK nominal
        return traj

    left_wrist  = _wrist_traj(left_nom_pos,  left_nom_rot)
    right_wrist = _wrist_traj(right_nom_pos, right_nom_rot)

    # Fingertip trajectories: wrist pos + fixed offset + tiny noise.
    def _tip_traj(wrist_traj: np.ndarray, offset: np.ndarray) -> np.ndarray:
        traj = np.zeros((T, 4, 4))
        for i in range(T):
            pos = wrist_traj[i, :3, 3] + offset + rng.uniform(-0.005, 0.005, 3)
            traj[i] = _make_se3(pos)
        return traj

    with h5py.File(path, "w") as f:
        f.attrs["llm_description"] = "synthetic: unscrew bottle cap"
        f.attrs["llm_type"]        = "other"

        f.create_dataset("camera/intrinsic", data=cam_int)

        tfs = f.require_group("transforms")
        tfs.create_dataset("camera",  data=cam_ext)
        tfs.create_dataset("leftHand",  data=left_wrist)
        tfs.create_dataset("rightHand", data=right_wrist)

        for side, wrist_traj in [("left", left_wrist), ("right", right_wrist)]:
            prefix  = side   # "left" / "right" — EgoDex uses camelCase: leftThumbTip
            joints  = [f"{prefix}ThumbTip", f"{prefix}IndexFingerTip", f"{prefix}MiddleFingerTip"]
            offsets = _TIP_OFFSETS
            for j, off in zip(joints, offsets):
                sign = 1 if side == "left" else -1   # mirror Y for right hand
                tfs.create_dataset(j, data=_tip_traj(wrist_traj, off * np.array([1, sign, 1])))

    print(f"[synthetic HDF5] written to {path}  (T={T} frames)")


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------

def _print_diagnostics(joint_positions: np.ndarray, ik_success: np.ndarray) -> None:
    T_ep = joint_positions.shape[0]
    print(f"\n{'='*60}")
    print(f"Output shape:     {joint_positions.shape}   (expected ({T_ep}, 38))")
    print(f"IK success rate:  left={ik_success[:, 0].mean():.1%}  "
          f"right={ik_success[:, 1].mean():.1%}")

    vel = np.diff(joint_positions, axis=0) * FPS
    max_vel  = np.abs(vel).max(axis=0)
    mean_vel = np.abs(vel).mean(axis=0)

    slices = {
        "left_panda  [0:7]  ": slice(0,  7),
        "left_tesollo[7:19] ": slice(7,  19),
        "right_panda [19:26]": slice(19, 26),
        "right_tesollo[26:38]": slice(26, 38),
    }
    print(f"\n{'Group':<25}  {'max |vel| (rad/s)':<20}  {'mean |vel|':<15}  range [min, max]")
    print("-" * 80)
    for label, sl in slices.items():
        q_sl  = joint_positions[:, sl]
        mv_sl = max_vel[sl]
        av_sl = mean_vel[sl]
        print(f"  {label}  {mv_sl.max():>8.3f} (worst jt)       "
              f"{av_sl.mean():>8.3f}        "
              f"[{q_sl.min():.3f}, {q_sl.max():.3f}]")

    nans = np.isnan(joint_positions).sum()
    print(f"\nNaN count: {nans}  {'← OK' if nans == 0 else '← WARNING!'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Retargeting pipeline smoke-test (synthetic data) ===\n")

    # 1. Reserve a temp file path (filled after IK solvers are built)
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        hdf5_path = tmp.name

    # 2. Initialise retargeters and IK solvers
    print("\nInitialising retargeters and IK solvers ...")
    left_ret, right_ret, left_ik, right_ik = build_retargeters()
    print("  Done.")

    # 1b. Re-create synthetic HDF5 now that we have IK solvers (FK-based nominal poses)
    print("\nGenerating synthetic HDF5 with FK-based nominal EE poses ...")
    make_synthetic_hdf5(hdf5_path, left_ik, right_ik)

    # 3. CoordinateAligner — identity (camera == robot base in synthetic data)
    aligner = CoordinateAligner()

    # 4. Run full retargeting
    print(f"\nRetargeting {T} frames ...")
    result = retarget_episode(hdf5_path, left_ret, right_ret, left_ik, right_ik, aligner)

    # 5. Print diagnostics
    _print_diagnostics(result["joint_positions"], result["ik_success"])

    # Cleanup
    Path(hdf5_path).unlink(missing_ok=True)
    print("\nSMOKE TEST PASSED — pipeline end-to-end functional.\n")


if __name__ == "__main__":
    main()
