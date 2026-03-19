"""
v6 Palm Offset Search + Decoupled IK retargeting.

Pipeline:
  1. Define virtual palm point at offset (dx, dy, dz) from delto_base_link
  2. 7-DOF arm IK: position virtual palm at human palm centroid
  3. 12-DOF finger retargeting: match fingertips in hand-base frame
  4. Full FK verification: measure actual vs target fingertip positions
  5. Grid search over offsets to minimize fingertip error

Usage (in handrl-policy container):
    # Search only (find optimal offsets)
    python palm_search_retarget.py --input_dir /path/to/episodes --search_only --ref_episode 0

    # Full retarget with known offsets
    python palm_search_retarget.py --input_dir /path/to/episodes --output /path/to/out \
        --left_offset "0.01,0.0,0.04" --right_offset "0.01,0.0,0.04"
"""

# Ensure conda-env pinocchio takes priority over /opt/openrobots
import sys as _sys
_sys.path = [p for p in _sys.path if "openrobots" not in p]

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

from retarget_episode import (
    LEFT_FINGER_TIPS,
    LEFT_PANDA_JOINTS,
    LEFT_TESOLLO_JOINTS,
    RIGHT_FINGER_TIPS,
    RIGHT_PANDA_JOINTS,
    RIGHT_TESOLLO_JOINTS,
    CoordinateAligner,
    FixedBaseHandRetargeter,
    PandaArmIKSolver,
    _BIMANUAL_URDF,
    _PANDA_READY_Q,
    _world_to_robot_pos,
    load_episode,
)
from retarget_for_sim import _R_CAM_TO_URDF, _BOTTLE_POS_URDF, compute_alignment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RETARGET_DIR = Path(__file__).parent

# Fingertip names per side (EgoDex → robot mapping)
_SIDE_TIPS = {
    "left": LEFT_FINGER_TIPS,     # [leftThumbTip, leftIndexFingerTip, leftMiddleFingerTip]
    "right": RIGHT_FINGER_TIPS,
}
_SIDE_TIP_FRAMES = {
    "left": ["left_F1_TIP", "left_F2_TIP", "left_F3_TIP"],
    "right": ["right_F1_TIP", "right_F2_TIP", "right_F3_TIP"],
}
_SIDE_PALM_FRAMES = {
    "left": ["left_virtual_palm", "left_delto_base_link"],
    "right": ["right_virtual_palm", "right_delto_base_link"],
}
_SIDE_ARM_JOINTS = {
    "left": LEFT_PANDA_JOINTS,
    "right": RIGHT_PANDA_JOINTS,
}
_SIDE_FINGER_JOINTS = {
    "left": LEFT_TESOLLO_JOINTS,
    "right": RIGHT_TESOLLO_JOINTS,
}
# Joint position slices in the 38-DOF vector
_SIDE_ARM_SLICE = {"left": slice(0, 7), "right": slice(19, 26)}
_SIDE_FINGER_SLICE = {"left": slice(7, 19), "right": slice(26, 38)}


# ---------------------------------------------------------------------------
# PalmOffsetIKSolver
# ---------------------------------------------------------------------------

class PalmOffsetIKSolver:
    """
    7-DOF arm IK targeting a virtual palm frame at offset from delto_base_link.

    Builds a pinocchio reduced model (7 active arm joints), adds a virtual
    frame at delto_base_link + palm_offset (in delto_base_link local frame),
    and solves position-only IK to that frame.
    """

    def __init__(self, side: str, palm_offset: np.ndarray,
                 urdf_path: Path = _BIMANUAL_URDF) -> None:
        assert side in ("left", "right")
        self.side = side
        self.palm_offset = np.asarray(palm_offset, dtype=np.float64)
        arm_joint_names = _SIDE_ARM_JOINTS[side]

        # Build reduced model (7-DOF arm only)
        full_model = pin.buildModelFromUrdf(str(urdf_path))
        q_neutral = pin.neutral(full_model)
        arm_id_set = {full_model.getJointId(n) for n in arm_joint_names
                      if full_model.existJointName(n)}
        joints_to_lock = [
            j.id for j in full_model.joints
            if j.id != 0 and j.id not in arm_id_set
        ]
        result = pin.buildReducedModel(full_model, joints_to_lock, q_neutral)
        self.model = result if isinstance(result, pin.Model) else result[0]

        # Find delto_base_link frame
        db_name = f"{side}_delto_base_link"
        self.db_frame_id = self.model.getFrameId(db_name)
        assert self.db_frame_id < len(self.model.frames), \
            f"Frame '{db_name}' not found in reduced model"
        db_frame = self.model.frames[self.db_frame_id]

        # Add virtual palm frame: delto_base_link placement + local offset
        offset_se3 = pin.SE3(np.eye(3), self.palm_offset)
        vp_frame = pin.Frame(
            f"{side}_virtual_palm",
            db_frame.parentJoint,
            self.db_frame_id,
            db_frame.placement * offset_se3,
            pin.FrameType.OP_FRAME,
        )
        self.vp_frame_id = self.model.addFrame(vp_frame)

        # Also store panda_link8 frame id for reference
        ee_name = f"{side}_panda_link8"
        self.ee_frame_id = self.model.getFrameId(ee_name)

        # Refresh data
        self.data = self.model.createData()
        self.q_default = np.clip(
            _PANDA_READY_Q.copy(),
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
        )

    def solve_position(
        self,
        target_pos: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tol: float = 1e-3,
        damping: float = 1e-4,
    ) -> tuple[np.ndarray, bool]:
        """Position-only IK targeting the virtual palm frame."""
        q = q_init.copy() if q_init is not None else self.q_default.copy()
        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            curr_pos = self.data.oMf[self.vp_frame_id].translation
            err = target_pos - curr_pos
            if np.linalg.norm(err) < tol:
                return q, True
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, self.vp_frame_id,
                pin.LOCAL_WORLD_ALIGNED,
            )
            J_pos = J_full[:3, :]
            dq = np.linalg.solve(
                J_pos.T @ J_pos + damping * np.eye(self.model.nv),
                J_pos.T @ err,
            )
            q = pin.integrate(self.model, q, dq * 0.5)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        return q, False

    def fk_hand_base(self, q: np.ndarray) -> np.ndarray:
        """Returns (4,4) SE3 of delto_base_link given 7-DOF arm joints."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.db_frame_id].homogeneous.copy()


# ---------------------------------------------------------------------------
# FullModelFK — full 38-DOF forward kinematics for verification
# ---------------------------------------------------------------------------

class FullModelFK:
    """Full bimanual model FK for fingertip and palm position verification."""

    def __init__(self, urdf_path: Path = _BIMANUAL_URDF) -> None:
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()

        # Cache frame IDs for all fingertips
        self.tip_frame_ids = {}
        for side in ("left", "right"):
            for tip_name in _SIDE_TIP_FRAMES[side]:
                fid = self.model.getFrameId(tip_name)
                assert fid < len(self.model.frames), f"Frame '{tip_name}' not found"
                self.tip_frame_ids[tip_name] = fid

        self.palm_frame_ids = {}
        for side in ("left", "right"):
            chosen = None
            for frame_name in _SIDE_PALM_FRAMES[side]:
                fid = self.model.getFrameId(frame_name)
                if fid < len(self.model.frames):
                    chosen = fid
                    break
            assert chosen is not None, f"Palm frame not found for side='{side}'"
            self.palm_frame_ids[side] = chosen

    def fingertip_positions(self, q38: np.ndarray, side: str) -> np.ndarray:
        """
        Compute 3 fingertip world positions for the given side.

        Args:
            q38: (38,) full joint angles
            side: 'left' or 'right'
        Returns:
            (3, 3) array — rows are [thumb_tip, index_tip, middle_tip]
        """
        pin.forwardKinematics(self.model, self.data, q38)
        pin.updateFramePlacements(self.model, self.data)
        tips = np.zeros((3, 3), dtype=np.float64)
        for i, tip_name in enumerate(_SIDE_TIP_FRAMES[side]):
            tips[i] = self.data.oMf[self.tip_frame_ids[tip_name]].translation.copy()
        return tips

    def palm_positions(self, q38: np.ndarray) -> np.ndarray:
        """
        Compute left/right palm world positions from full joint state.

        Returns:
            (2, 3) array — rows are [left_palm, right_palm]
        """
        pin.forwardKinematics(self.model, self.data, q38)
        pin.updateFramePlacements(self.model, self.data)
        palms = np.zeros((2, 3), dtype=np.float64)
        palms[0] = self.data.oMf[self.palm_frame_ids["left"]].translation.copy()
        palms[1] = self.data.oMf[self.palm_frame_ids["right"]].translation.copy()
        return palms


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(
    side: str,
    palm_offset: np.ndarray,
    human_palm_positions: np.ndarray,
    human_tips_urdf: np.ndarray,
    frame_indices: np.ndarray,
    full_fk: FullModelFK,
    finger_retargeter: FixedBaseHandRetargeter,
) -> dict:
    """
    Evaluate one candidate palm offset on sampled frames.

    Args:
        side: 'left' or 'right'
        palm_offset: (3,) offset in delto_base_link local frame
        human_palm_positions: (T, 3) palm centroid in URDF world frame
        human_tips_urdf: (T, 3, 3) fingertip positions in URDF world frame
        frame_indices: indices into the T dimension to evaluate
        full_fk: FullModelFK instance for verification
        finger_retargeter: FixedBaseHandRetargeter instance

    Returns:
        dict with mean_tip_error, max_tip_error, ik_success_rate, per_frame_errors
    """
    ik_solver = PalmOffsetIKSolver(side, palm_offset)

    errors = []
    ik_successes = 0
    q_prev = None

    for idx in frame_indices:
        target_palm = human_palm_positions[idx]
        target_tips = human_tips_urdf[idx]  # (3, 3)

        # Step 1: Arm IK — position virtual palm at human palm centroid
        q_arm, ok = ik_solver.solve_position(target_palm, q_init=q_prev)
        if ok:
            q_prev = q_arm
        ik_successes += int(ok)

        # Step 2: FK to get hand base SE3
        T_hand_base = ik_solver.fk_hand_base(q_arm)
        R_hb = T_hand_base[:3, :3]
        t_hb = T_hand_base[:3, 3]

        # Step 3: Transform fingertip targets to hand-base frame
        tips_hand_frame = ((target_tips - t_hb) @ R_hb).astype(np.float64)  # (3, 3)

        # Step 4: Finger retargeting in hand-base frame
        q_fingers = finger_retargeter.retarget(tips_hand_frame)

        # Step 5: Full FK verification
        q38 = np.zeros(38, dtype=np.float64)
        q38[_SIDE_ARM_SLICE[side]] = q_arm
        q38[_SIDE_FINGER_SLICE[side]] = q_fingers
        actual_tips = full_fk.fingertip_positions(q38, side)

        # Step 6: Error
        tip_errors = np.linalg.norm(actual_tips - target_tips, axis=1)  # (3,)
        errors.append(tip_errors)

    errors = np.array(errors)  # (N_frames, 3)
    return {
        "mean_tip_error": errors.mean(),
        "max_tip_error": errors.max(),
        "median_tip_error": np.median(errors),
        "ik_success_rate": ik_successes / len(frame_indices),
        "per_frame_errors": errors,
        "palm_offset": palm_offset.copy(),
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def extract_episode_data(
    hdf5_path: str,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> dict:
    """Extract human palm centroids and fingertip positions in URDF frame."""
    ep = load_episode(hdf5_path)
    tfs = ep["transforms"]
    T = len(tfs["leftHand"])
    identity_aligner = CoordinateAligner()

    result = {}
    for side in ("left", "right"):
        tip_names = _SIDE_TIPS[side]
        tips_urdf = np.zeros((T, 3, 3), dtype=np.float64)
        palm_positions = np.zeros((T, 3), dtype=np.float64)

        for i in range(T):
            cam_ext_i = ep["cam_ext"][i]
            frame_tips = np.zeros((3, 3), dtype=np.float64)
            for j, tn in enumerate(tip_names):
                pos_cam = _world_to_robot_pos(tfs[tn][i], cam_ext_i, identity_aligner)
                pos_urdf = R_align @ pos_cam + t_align
                frame_tips[j] = pos_urdf
            tips_urdf[i] = frame_tips
            palm_positions[i] = frame_tips.mean(axis=0)

        result[side] = {
            "tips_urdf": tips_urdf,
            "palm_positions": palm_positions,
        }

    return result


def coarse_grid_search(
    side: str,
    episode_data: dict,
    full_fk: FullModelFK,
    finger_retargeter: FixedBaseHandRetargeter,
    frame_step: int = 10,
) -> list[dict]:
    """
    Coarse grid search over palm offsets.

    Search space (delto_base_link local frame):
      x: [-0.02, 0.04], step 1cm
      y: [-0.03, 0.03], step 1cm
      z: [0.00, 0.08], step 1cm
    """
    xs = np.arange(-0.02, 0.041, 0.01)
    ys = np.arange(-0.03, 0.031, 0.01)
    zs = np.arange(0.00, 0.081, 0.01)

    T = len(episode_data[side]["palm_positions"])
    frame_indices = np.arange(0, T, frame_step)

    candidates = list(product(xs, ys, zs))
    print(f"  Coarse search: {len(candidates)} candidates, "
          f"{len(frame_indices)} frames (step={frame_step})")

    results = []
    t0 = time.time()
    for i, (x, y, z) in enumerate(candidates):
        offset = np.array([x, y, z])
        r = evaluate_candidate(
            side, offset,
            episode_data[side]["palm_positions"],
            episode_data[side]["tips_urdf"],
            frame_indices,
            full_fk,
            finger_retargeter,
        )
        results.append(r)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(candidates) - i - 1)
            print(f"    [{i+1}/{len(candidates)}] "
                  f"offset=({x:.3f},{y:.3f},{z:.3f}) "
                  f"mean_err={r['mean_tip_error']:.4f}m "
                  f"ik_rate={r['ik_success_rate']:.0%} "
                  f"ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Coarse search done in {elapsed:.1f}s")

    # Sort by mean error
    results.sort(key=lambda r: r["mean_tip_error"])
    return results


def fine_grid_search(
    side: str,
    top_offsets: list[np.ndarray],
    episode_data: dict,
    full_fk: FullModelFK,
    finger_retargeter: FixedBaseHandRetargeter,
    frame_step: int = 5,
    radius: float = 0.005,
    step: float = 0.002,
) -> list[dict]:
    """Fine grid search around top candidates from coarse pass."""
    T = len(episode_data[side]["palm_positions"])
    frame_indices = np.arange(0, T, frame_step)

    # Build unique candidate set around each top offset
    offsets_grid = np.arange(-radius, radius + step * 0.5, step)
    seen = set()
    candidates = []
    for center in top_offsets:
        for dx, dy, dz in product(offsets_grid, offsets_grid, offsets_grid):
            o = (round(center[0] + dx, 4), round(center[1] + dy, 4), round(center[2] + dz, 4))
            if o not in seen:
                seen.add(o)
                candidates.append(np.array(o))

    print(f"  Fine search: {len(candidates)} candidates, "
          f"{len(frame_indices)} frames (step={frame_step})")

    results = []
    t0 = time.time()
    for i, offset in enumerate(candidates):
        r = evaluate_candidate(
            side, offset,
            episode_data[side]["palm_positions"],
            episode_data[side]["tips_urdf"],
            frame_indices,
            full_fk,
            finger_retargeter,
        )
        results.append(r)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(candidates) - i - 1)
            print(f"    [{i+1}/{len(candidates)}] "
                  f"best_so_far={min(r['mean_tip_error'] for r in results):.4f}m "
                  f"ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Fine search done in {elapsed:.1f}s")

    results.sort(key=lambda r: r["mean_tip_error"])
    return results


def run_search(
    input_dir: Path,
    ref_episode: int = 0,
    n_top: int = 5,
) -> dict:
    """Run full coarse-to-fine grid search for both hands."""
    # Compute alignment
    ref_path = input_dir / f"{ref_episode}.hdf5"
    print(f"\nComputing alignment from {ref_path.name}...")
    R_align, t_align = compute_alignment(str(ref_path))

    # Extract episode data
    print(f"\nExtracting data from {ref_path.name}...")
    episode_data = extract_episode_data(str(ref_path), R_align, t_align)
    for side in ("left", "right"):
        T = len(episode_data[side]["palm_positions"])
        print(f"  {side}: {T} frames")

    # Build shared resources
    print("\nBuilding FK model and finger retargeters...")
    full_fk = FullModelFK()

    best_offsets = {}
    for side in ("left", "right"):
        print(f"\n{'='*60}")
        print(f"Searching {side} hand...")
        print(f"{'='*60}")

        finger_ret = FixedBaseHandRetargeter(side)

        # Coarse pass
        print("\n[Coarse pass]")
        coarse_results = coarse_grid_search(
            side, episode_data, full_fk, finger_ret,
        )

        # Print top results
        print(f"\n  Top-{n_top} coarse results:")
        for i, r in enumerate(coarse_results[:n_top]):
            o = r["palm_offset"]
            print(f"    {i+1}. offset=({o[0]:.3f},{o[1]:.3f},{o[2]:.3f}) "
                  f"mean={r['mean_tip_error']:.4f}m "
                  f"max={r['max_tip_error']:.4f}m "
                  f"ik={r['ik_success_rate']:.0%}")

        # Fine pass
        print("\n[Fine pass]")
        top_offsets = [r["palm_offset"] for r in coarse_results[:n_top]]
        fine_results = fine_grid_search(
            side, top_offsets, episode_data, full_fk, finger_ret,
        )

        best = fine_results[0]
        o = best["palm_offset"]
        print(f"\n  BEST {side}: offset=({o[0]:.4f},{o[1]:.4f},{o[2]:.4f}) "
              f"mean={best['mean_tip_error']:.4f}m "
              f"max={best['max_tip_error']:.4f}m "
              f"median={best['median_tip_error']:.4f}m "
              f"ik={best['ik_success_rate']:.0%}")

        best_offsets[side] = best["palm_offset"]

    print(f"\n{'='*60}")
    print("SEARCH RESULTS")
    print(f"{'='*60}")
    for side in ("left", "right"):
        o = best_offsets[side]
        print(f"  {side}: ({o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f})")

    return {
        "left_offset": best_offsets["left"],
        "right_offset": best_offsets["right"],
        "R_align": R_align,
        "t_align": t_align,
    }


# ---------------------------------------------------------------------------
# Full episode retargeting with optimal offsets
# ---------------------------------------------------------------------------

def retarget_episode_v6(
    hdf5_path: str,
    left_offset: np.ndarray,
    right_offset: np.ndarray,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> dict:
    """
    Retarget one episode using v6 palm offset + decoupled IK.

    Returns dict with:
        joint_positions: (T, 38) robot joint angles
        ik_success: (T, 2) bool [left_ok, right_ok]
        fingertip_errors: (T, 6) per-fingertip errors in meters
    """
    ep = load_episode(hdf5_path)
    tfs = ep["transforms"]
    T = len(tfs["leftHand"])
    identity_aligner = CoordinateAligner()

    # Build solvers
    ik_solvers = {
        "left": PalmOffsetIKSolver("left", left_offset),
        "right": PalmOffsetIKSolver("right", right_offset),
    }
    finger_rets = {
        "left": FixedBaseHandRetargeter("left"),
        "right": FixedBaseHandRetargeter("right"),
    }
    full_fk = FullModelFK()

    joint_positions = np.zeros((T, 38), dtype=np.float64)
    ik_success = np.zeros((T, 2), dtype=bool)
    fingertip_errors = np.zeros((T, 6), dtype=np.float64)

    q_prev = {"left": None, "right": None}

    for i in range(T):
        cam_ext_i = ep["cam_ext"][i]

        for si, side in enumerate(("left", "right")):
            tip_names = _SIDE_TIPS[side]

            # Extract fingertip positions in URDF frame
            target_tips = np.zeros((3, 3), dtype=np.float64)
            for j, tn in enumerate(tip_names):
                pos_cam = _world_to_robot_pos(tfs[tn][i], cam_ext_i, identity_aligner)
                target_tips[j] = R_align @ pos_cam + t_align

            # Palm centroid
            palm_pos = target_tips.mean(axis=0)

            # Arm IK
            q_arm, ok = ik_solvers[side].solve_position(
                palm_pos, q_init=q_prev[side],
            )
            ik_success[i, si] = ok
            if ok:
                q_prev[side] = q_arm

            # Hand base FK
            T_hb = ik_solvers[side].fk_hand_base(q_arm)
            R_hb = T_hb[:3, :3]
            t_hb = T_hb[:3, 3]

            # Transform tips to hand frame
            tips_hand = (target_tips - t_hb) @ R_hb  # (3, 3)

            # Finger retargeting
            q_fingers = finger_rets[side].retarget(tips_hand)

            # Store
            joint_positions[i, _SIDE_ARM_SLICE[side]] = q_arm
            joint_positions[i, _SIDE_FINGER_SLICE[side]] = q_fingers

        # Full FK verification
        actual_left = full_fk.fingertip_positions(joint_positions[i], "left")
        actual_right = full_fk.fingertip_positions(joint_positions[i], "right")

        # Compute errors for this frame
        for j, tn in enumerate(LEFT_FINGER_TIPS):
            pos_cam = _world_to_robot_pos(tfs[tn][i], cam_ext_i, identity_aligner)
            target = R_align @ pos_cam + t_align
            fingertip_errors[i, j] = np.linalg.norm(actual_left[j] - target)

        for j, tn in enumerate(RIGHT_FINGER_TIPS):
            pos_cam = _world_to_robot_pos(tfs[tn][i], cam_ext_i, identity_aligner)
            target = R_align @ pos_cam + t_align
            fingertip_errors[i, 3 + j] = np.linalg.norm(actual_right[j] - target)

        if i % 50 == 0 or i == T - 1:
            mean_err = fingertip_errors[i].mean()
            ik_ok = ik_success[i].all()
            print(f"  Frame {i:4d}/{T}: mean_tip_err={mean_err:.4f}m IK={'OK' if ik_ok else 'FAIL'}")

    return {
        "joint_positions": joint_positions,
        "ik_success": ik_success,
        "fingertip_errors": fingertip_errors,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_all_episodes(
    input_dir: Path,
    output_dir: Path,
    left_offset: np.ndarray,
    right_offset: np.ndarray,
    ref_episode: int = 0,
):
    """Retarget all episodes with given offsets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted(input_dir.glob("*.hdf5"), key=lambda p: int(p.stem))
    print(f"Found {len(episodes)} episodes")

    # Compute alignment
    ref_path = input_dir / f"{ref_episode}.hdf5"
    if not ref_path.exists():
        ref_path = episodes[0]
    print(f"\nComputing alignment from {ref_path.name}...")
    R_align, t_align = compute_alignment(str(ref_path))

    results_summary = []
    for ep_path in episodes:
        print(f"\n{'='*60}")
        print(f"Processing {ep_path.name}...")

        result = retarget_episode_v6(
            str(ep_path), left_offset, right_offset, R_align, t_align,
        )

        ik = result["ik_success"]
        both_rate = ik.all(axis=1).mean()
        mean_err = result["fingertip_errors"].mean()
        max_err = result["fingertip_errors"].max()

        print(f"  IK success: {both_rate:.1%}")
        print(f"  Mean tip error: {mean_err:.4f}m, Max: {max_err:.4f}m")

        # Save
        out_name = ep_path.stem + "_sim.npz"
        out_path = output_dir / out_name
        np.savez_compressed(
            str(out_path),
            joint_positions=result["joint_positions"],
            ik_success=result["ik_success"],
            fingertip_errors=result["fingertip_errors"],
            episode=ep_path.stem,
            R_align=R_align,
            t_align=t_align,
            left_offset=left_offset,
            right_offset=right_offset,
        )
        print(f"  Saved: {out_path}")

        results_summary.append({
            "episode": ep_path.stem,
            "frames": len(ik),
            "ik_rate": both_rate,
            "mean_err": mean_err,
            "max_err": max_err,
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results_summary:
        print(f"  ep {r['episode']:>2s}: IK={r['ik_rate']:.1%} "
              f"mean_err={r['mean_err']:.4f}m max_err={r['max_err']:.4f}m")

    mean_ik = np.mean([r["ik_rate"] for r in results_summary])
    mean_err = np.mean([r["mean_err"] for r in results_summary])
    print(f"\nOverall: IK={mean_ik:.1%}, mean_tip_error={mean_err:.4f}m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="v6 Palm Offset Search + Decoupled IK retargeting",
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with EgoDex .hdf5 episodes")
    parser.add_argument("--output", type=str,
                        help="Output directory for retargeted trajectories")
    parser.add_argument("--search_only", action="store_true",
                        help="Only run grid search, don't retarget all episodes")
    parser.add_argument("--ref_episode", type=int, default=0,
                        help="Reference episode for alignment and search")
    parser.add_argument("--left_offset", type=str,
                        help="Known left offset, comma-separated: '0.01,0.0,0.04'")
    parser.add_argument("--right_offset", type=str,
                        help="Known right offset, comma-separated: '0.01,0.0,0.04'")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if args.search_only:
        # Search mode
        search_result = run_search(input_dir, ref_episode=args.ref_episode)
        # Save search results
        out_path = input_dir.parent / "palm_search_results.json"
        save_dict = {
            "left_offset": search_result["left_offset"].tolist(),
            "right_offset": search_result["right_offset"].tolist(),
        }
        with open(out_path, "w") as f:
            json.dump(save_dict, f, indent=2)
        print(f"\nSearch results saved to {out_path}")

    elif args.left_offset and args.right_offset:
        # Retarget mode with known offsets
        left_offset = np.array([float(x) for x in args.left_offset.split(",")])
        right_offset = np.array([float(x) for x in args.right_offset.split(",")])

        if not args.output:
            parser.error("--output required for retargeting")

        print(f"Left offset:  {left_offset}")
        print(f"Right offset: {right_offset}")

        process_all_episodes(
            input_dir, Path(args.output),
            left_offset, right_offset,
            ref_episode=args.ref_episode,
        )

    else:
        # Search + retarget
        search_result = run_search(input_dir, ref_episode=args.ref_episode)

        if args.output:
            process_all_episodes(
                input_dir, Path(args.output),
                search_result["left_offset"],
                search_result["right_offset"],
                ref_episode=args.ref_episode,
            )


if __name__ == "__main__":
    main()
