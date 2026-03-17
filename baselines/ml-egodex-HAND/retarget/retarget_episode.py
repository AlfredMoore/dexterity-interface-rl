"""
Core retargeting module: EgoDex human hand SE(3) → Bimanual Panda + Tesollo DG-3F joint angles.

Output joint order (38-DOF, matches rl_driver convention):
  [left_panda×7, left_tesollo×12, right_panda×7, right_tesollo×12]

Usage:
    from retarget.retarget_episode import build_retargeters, retarget_episode

    left_ret, right_ret, left_ik, right_ik = build_retargeters()
    aligner = CoordinateAligner()          # identity by default
    result = retarget_episode("0.hdf5", left_ret, right_ret, left_ik, right_ik, aligner)
    joint_positions = result["joint_positions"]  # (T, 38)
"""

# Ensure conda-env pinocchio takes priority over /opt/openrobots
import sys as _sys
_sys.path = [p for p in _sys.path if "openrobots" not in p]

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pinocchio as pin

from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RETARGET_DIR = Path(__file__).parent                          # .../retarget/
_REPO_ROOT    = _RETARGET_DIR.parents[2]                       # .../dexterity-interface-rl/ (or /workspace/ in container)
_ASSETS_DIR   = _RETARGET_DIR / "assets"
_CONFIG_DIR   = _RETARGET_DIR / "config"
_BIMANUAL_URDF = _REPO_ROOT / "libs/robot_description/rl/bimanual_panda_tesollo.urdf"

# ---------------------------------------------------------------------------
# Panda default joint configuration (standard "ready" pose, avoids all-zero singularity)
# ---------------------------------------------------------------------------
_PANDA_READY_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# ---------------------------------------------------------------------------
# Joint name constants
# ---------------------------------------------------------------------------
LEFT_PANDA_JOINTS  = [f"left_panda_joint{i}"  for i in range(1, 8)]
RIGHT_PANDA_JOINTS = [f"right_panda_joint{i}" for i in range(1, 8)]

LEFT_TESOLLO_JOINTS = [
    "left_F1M1",  "left_F1M2",  "left_F1M3",  "left_F1M4",
    "left_F2M1",  "left_F2M2",  "left_F2M3",  "left_F2M4",
    "left_F3M1",  "left_F3M2",  "left_F3M3",  "left_F3M4",
]
RIGHT_TESOLLO_JOINTS = [
    "right_F1M1", "right_F1M2", "right_F1M3", "right_F1M4",
    "right_F2M1", "right_F2M2", "right_F2M3", "right_F2M4",
    "right_F3M1", "right_F3M2", "right_F3M3", "right_F3M4",
]

# EgoDex ARKit joint names → rows in the (3,3) fingertip array
# Row 0 = thumb analog (F1), row 1 = index analog (F2), row 2 = middle analog (F3)
RIGHT_FINGER_TIPS = ["rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerTip"]
LEFT_FINGER_TIPS  = ["leftThumbTip",  "leftIndexFingerTip",  "leftMiddleFingerTip"]


# ---------------------------------------------------------------------------
# CoordinateAligner
# ---------------------------------------------------------------------------

class CoordinateAligner:
    """
    Applies a fixed SE(3) transform to convert positions/poses from the
    EgoDex camera frame to the robot base frame.

    Default is identity (keeps the camera frame), which is fine for
    visualisation and relative-motion analysis.  For real-robot deployment,
    provide a calibrated (R, t) pair.

    Args:
        R: (3,3) rotation matrix, camera→robot.
        t: (3,) translation, camera→robot.
    """

    def __init__(
        self,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
    ) -> None:
        self.R = np.eye(3, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
        self.t = np.zeros(3, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
        self._T = np.eye(4, dtype=np.float64)
        self._T[:3, :3] = self.R
        self._T[:3,  3] = self.t

    def transform_position(self, pos: np.ndarray) -> np.ndarray:
        """pos: (3,) or (N,3) → transformed (3,) or (N,3)."""
        return (self.R @ np.asarray(pos, dtype=np.float64).T).T + self.t

    def transform_se3(self, tf: np.ndarray) -> np.ndarray:
        """tf: (4,4) SE(3) in camera frame → (4,4) SE(3) in robot base frame."""
        return self._T @ np.asarray(tf, dtype=np.float64)


# ---------------------------------------------------------------------------
# PandaArmIKSolver
# ---------------------------------------------------------------------------

class PandaArmIKSolver:
    """
    Single-arm Jacobian IK (Levenberg–Marquardt / DLS) for Franka Panda.

    Uses a pinocchio reduced model containing only the 7 arm joints, derived
    from the bimanual URDF.  All other joints (Tesollo + the other arm) are
    locked at their URDF neutral values.

    Args:
        side:      'left' or 'right'.
        urdf_path: path to the bimanual URDF (default: auto-detected).
    """

    def __init__(self, side: str, urdf_path: Path = _BIMANUAL_URDF) -> None:
        assert side in ("left", "right")
        self.side = side
        arm_joint_names = LEFT_PANDA_JOINTS if side == "left" else RIGHT_PANDA_JOINTS

        # Load the full bimanual model
        full_model = pin.buildModelFromUrdf(str(urdf_path))
        q_neutral  = pin.neutral(full_model)

        # IDs of joints to LOCK (everything except this arm's 7 joints)
        arm_id_set = {full_model.getJointId(n) for n in arm_joint_names
                      if full_model.existJointName(n)}
        joints_to_lock = [
            j.id for j in full_model.joints
            if j.id != 0 and j.id not in arm_id_set
        ]
        result = pin.buildReducedModel(full_model, joints_to_lock, q_neutral)
        # pinocchio 2.x returns (model, data), 3.x returns model directly
        self.model = result if isinstance(result, pin.Model) else result[0]
        self.data      = self.model.createData()
        self.q_neutral = pin.neutral(self.model)
        # Use a non-singular ready pose as the default IK starting point.
        # pin.neutral() (all zeros) puts the Panda in a singular upright config.
        self.q_default = np.clip(
            _PANDA_READY_Q.copy(),
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
        )

        ee_link = f"{side}_panda_link8"
        self.ee_frame_id = self.model.getFrameId(ee_link)
        assert self.ee_frame_id < len(self.model.frames), \
            f"EE frame '{ee_link}' not found in reduced model"

    def solve(
        self,
        target_se3: pin.SE3,
        q_init:     Optional[np.ndarray] = None,
        max_iter:   int   = 200,
        tol:        float = 1e-4,
        damping:    float = 1e-4,
    ) -> tuple[np.ndarray, bool]:
        """
        Solve IK for the EE to reach target_se3.

        Returns:
            q:       (7,) joint angles.
            success: True if converged within tol.
        """
        q = q_init.copy() if q_init is not None else self.q_default.copy()
        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            curr  = self.data.oMf[self.ee_frame_id]
            err   = pin.log6(curr.inverse() * target_se3).vector
            if np.linalg.norm(err) < tol:
                return q, True
            J  = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.LOCAL
            )
            dq = np.linalg.solve(J.T @ J + damping * np.eye(self.model.nv), J.T @ err)
            q  = pin.integrate(self.model, q, dq * 0.5)
            q  = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        return q, False

    def fk_hand_base(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the SE(3) pose of the Tesollo hand base (delto_base_link)
        given the 7-DOF arm joint angles.

        The chain is: base → panda_link8 → (fixed joint: xyz=0,0,0.106 rpy=0,0,-0.785) → delto_base_link.
        We use pinocchio FK to get panda_link8 and apply the fixed offset.

        Returns: (4, 4) SE(3) hand base pose in the arm base frame.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T_link8 = self.data.oMf[self.ee_frame_id].homogeneous

        # Fixed joint: panda_link8 → delto_base_link
        # From URDF: origin rpy="0 0 -0.785" xyz="0 0 0.106"
        cos_a = np.cos(-0.785)
        sin_a = np.sin(-0.785)
        T_mount = np.eye(4)
        T_mount[:3, :3] = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ])
        T_mount[2, 3] = 0.106

        return T_link8 @ T_mount

    def solve_position_only(
        self,
        target_pos: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tol: float = 1e-3,
        damping: float = 1e-4,
    ) -> tuple[np.ndarray, bool]:
        """
        Position-only IK (3-DOF task, ignores orientation).
        Useful as fallback when full 6-DOF IK fails due to unreachable orientations.

        Returns:
            q:       (7,) joint angles.
            success: True if position error < tol.
        """
        q = q_init.copy() if q_init is not None else self.q_default.copy()
        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            curr_pos = self.data.oMf[self.ee_frame_id].translation
            err = target_pos - curr_pos
            if np.linalg.norm(err) < tol:
                return q, True
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            J_pos = J_full[:3, :]  # position-only Jacobian
            dq = np.linalg.solve(J_pos.T @ J_pos + damping * np.eye(self.model.nv), J_pos.T @ err)
            q = pin.integrate(self.model, q, dq * 0.5)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        return q, False


# ---------------------------------------------------------------------------
# HandRetargeter
# ---------------------------------------------------------------------------

class HandRetargeter:
    """
    Wraps dex-retargeting SeqRetargeting for one Tesollo DG-3F hand.

    Input:  (3, 3) array of [thumb_tip, index_tip, middle_tip] absolute 3D
            positions in the robot base frame.
    Output: (12,) Tesollo joint angles in order
            [F1M1..F1M4, F2M1..F2M4, F3M1..F3M4].

    The underlying optimizer uses add_dummy_free_joint=true so the hand base
    can float; only the 12 finger angles are returned.

    Args:
        side: 'left' or 'right'.
    """

    _FINGER_JOINT_COUNT = 12

    def __init__(self, side: str) -> None:
        assert side in ("left", "right")
        self.side = side

        RetargetingConfig.set_default_urdf_dir(_ASSETS_DIR)
        cfg = RetargetingConfig.load_from_file(
            str(_CONFIG_DIR / f"tesollo_{side}.yaml")
        )
        self._seq: SeqRetargeting = cfg.build()

        # Indices of finger joints in the full qpos output (skip 6 dummy DOFs)
        expected = LEFT_TESOLLO_JOINTS if side == "left" else RIGHT_TESOLLO_JOINTS
        names    = self._seq.joint_names
        self._finger_idx = [names.index(j) for j in expected if j in names]
        assert len(self._finger_idx) == self._FINGER_JOINT_COUNT, (
            f"Expected {self._FINGER_JOINT_COUNT} finger joints, "
            f"found {len(self._finger_idx)} in {names}"
        )

    @property
    def joint_names(self) -> list[str]:
        expected = LEFT_TESOLLO_JOINTS if self.side == "left" else RIGHT_TESOLLO_JOINTS
        return expected

    def retarget(self, fingertip_positions: np.ndarray) -> np.ndarray:
        """
        fingertip_positions: (3, 3) — rows are [thumb_pos, index_pos, middle_pos]
                             in robot base frame (or camera frame if no aligner).
        Returns (12,) finger joint angles.
        """
        qpos = self._seq.retarget(fingertip_positions.astype(np.float64))
        return qpos[self._finger_idx]

    def warm_start(self, wrist_pos: np.ndarray, wrist_quat: np.ndarray) -> None:
        """
        Warm-start the free joint from the wrist position/orientation.
        Call once at the beginning of each episode for faster convergence.

        wrist_pos:  (3,) position in robot base frame.
        wrist_quat: (4,) quaternion [w, x, y, z].
        """
        self._seq.warm_start(
            wrist_pos.astype(np.float64),
            wrist_quat.astype(np.float64),
            hand_type=self.side,
            is_mano_convention=False,
        )

    def reset(self) -> None:
        self._seq.reset()


# ---------------------------------------------------------------------------
# FixedBaseHandRetargeter
# ---------------------------------------------------------------------------

class FixedBaseHandRetargeter:
    """
    Hand retargeting with the base fixed (no free joint).

    Fingertip positions must be given in the hand base frame (delto_base_link).
    This avoids the mismatch when using separate arm IK + free-floating hand
    retargeting: the finger angles are optimized for the actual wrist pose.

    Args:
        side: 'left' or 'right'.
    """

    _FINGER_JOINT_COUNT = 12

    def __init__(self, side: str) -> None:
        assert side in ("left", "right")
        self.side = side

        RetargetingConfig.set_default_urdf_dir(_ASSETS_DIR)
        cfg = RetargetingConfig.load_from_file(
            str(_CONFIG_DIR / f"tesollo_{side}_fixed.yaml")
        )
        self._seq: SeqRetargeting = cfg.build()

        expected = LEFT_TESOLLO_JOINTS if side == "left" else RIGHT_TESOLLO_JOINTS
        names    = self._seq.joint_names
        self._finger_idx = [names.index(j) for j in expected if j in names]
        assert len(self._finger_idx) == self._FINGER_JOINT_COUNT, (
            f"Expected {self._FINGER_JOINT_COUNT} finger joints, "
            f"found {len(self._finger_idx)} in {names}"
        )

    @property
    def joint_names(self) -> list[str]:
        expected = LEFT_TESOLLO_JOINTS if self.side == "left" else RIGHT_TESOLLO_JOINTS
        return expected

    def retarget(self, fingertip_positions_hand_frame: np.ndarray) -> np.ndarray:
        """
        fingertip_positions_hand_frame: (3, 3) — rows are [thumb, index, middle]
            positions in the hand base frame (delto_base_link).
        Returns (12,) finger joint angles.
        """
        qpos = self._seq.retarget(fingertip_positions_hand_frame.astype(np.float64))
        return qpos[self._finger_idx]

    def reset(self) -> None:
        self._seq.reset()


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

_JOINTS_NEEDED = (
    ["leftHand", "rightHand"]
    + LEFT_FINGER_TIPS
    + RIGHT_FINGER_TIPS
)


def load_episode(hdf5_path: str) -> dict:
    """
    Load all relevant transforms from an EgoDex HDF5 file.

    Returns:
        transforms: dict joint_name → (T, 4, 4) SE(3) in ARKit world frame.
        cam_ext:    (T, 4, 4) camera extrinsics in ARKit world frame.
        cam_int:    (3, 3) fixed camera intrinsics.
    """
    transforms: dict[str, np.ndarray] = {}
    with h5py.File(hdf5_path, "r") as f:
        cam_int = f["/camera/intrinsic"][:]
        cam_ext = f["/transforms/camera"][:]
        for j in _JOINTS_NEEDED:
            transforms[j] = f[f"/transforms/{j}"][:]
    return {"transforms": transforms, "cam_ext": cam_ext, "cam_int": cam_int}


# ---------------------------------------------------------------------------
# Per-frame coordinate helpers (module-level for clarity)
# ---------------------------------------------------------------------------

def _world_to_robot_se3(
    tf_world: np.ndarray,
    cam_ext:  np.ndarray,
    aligner:  CoordinateAligner,
) -> np.ndarray:
    """ARKit world frame SE(3) → robot base frame SE(3)."""
    tf_cam = np.linalg.inv(cam_ext) @ tf_world
    return aligner.transform_se3(tf_cam)


def _world_to_robot_pos(
    tf_world: np.ndarray,
    cam_ext:  np.ndarray,
    aligner:  CoordinateAligner,
) -> np.ndarray:
    """ARKit world frame SE(3) → robot base frame 3D position."""
    tf_cam = np.linalg.inv(cam_ext) @ tf_world
    pos_cam = tf_cam[:3, 3]
    return aligner.transform_position(pos_cam)


# ---------------------------------------------------------------------------
# Main retargeting function
# ---------------------------------------------------------------------------

def retarget_episode(
    hdf5_path:       str,
    left_retargeter: HandRetargeter,
    right_retargeter: HandRetargeter,
    left_ik:         PandaArmIKSolver,
    right_ik:        PandaArmIKSolver,
    aligner:         CoordinateAligner,
) -> dict:
    """
    Retarget one EgoDex episode to bimanual robot joint angles.

    The output joint order matches the rl_driver convention:
        [left_panda×7, left_tesollo×12, right_panda×7, right_tesollo×12]

    Args:
        hdf5_path:        path to EgoDex .hdf5 file.
        left/right_retargeter: HandRetargeter instances.
        left/right_ik:         PandaArmIKSolver instances.
        aligner:          CoordinateAligner (camera frame → robot base frame).

    Returns dict with:
        joint_positions: (T, 38) float64 robot joint angles.
        ik_success:      (T, 2)  bool [left_ok, right_ok] per frame.
    """
    ep = load_episode(hdf5_path)
    tfs = ep["transforms"]
    T   = len(tfs["leftHand"])

    joint_positions = np.zeros((T, 38), dtype=np.float64)
    ik_success      = np.zeros((T, 2),  dtype=bool)

    # Warm-start retargeters from the first frame
    for side, retargeter, wrist_key in [
        ("left",  left_retargeter,  "leftHand"),
        ("right", right_retargeter, "rightHand"),
    ]:
        retargeter.reset()
        tf0 = _world_to_robot_se3(tfs[wrist_key][0], ep["cam_ext"][0], aligner)
        rot = tf0[:3, :3]
        quat_xyzw = pin.Quaternion(rot).coeffs()      # [x, y, z, w]
        quat_wxyz  = np.array([quat_xyzw[3], *quat_xyzw[:3]])  # [w, x, y, z]
        retargeter.warm_start(tf0[:3, 3], quat_wxyz)

    left_q_prev  = None
    right_q_prev = None

    for i in range(T):
        cam_ext_i = ep["cam_ext"][i]

        # ── Panda arm IK ─────────────────────────────────────────────────────
        for col, wrist_key, ik_solver, q_prev_ref, out_slice in [
            (0, "leftHand",  left_ik,  left_q_prev,  slice(0,  7)),
            (1, "rightHand", right_ik, right_q_prev, slice(19, 26)),
        ]:
            tf_robot  = _world_to_robot_se3(tfs[wrist_key][i], cam_ext_i, aligner)
            target    = pin.SE3(tf_robot[:3, :3], tf_robot[:3, 3])
            q, ok     = ik_solver.solve(target, q_init=q_prev_ref)
            joint_positions[i, out_slice] = q
            ik_success[i, col] = ok
            if col == 0:
                left_q_prev  = q
            else:
                right_q_prev = q

        # ── Tesollo finger retargeting ────────────────────────────────────────
        for tips, retargeter, out_slice in [
            (LEFT_FINGER_TIPS,  left_retargeter,  slice(7,  19)),
            (RIGHT_FINGER_TIPS, right_retargeter, slice(26, 38)),
        ]:
            tip_positions = np.stack([
                _world_to_robot_pos(tfs[t][i], cam_ext_i, aligner)
                for t in tips
            ])  # (3, 3)
            joint_positions[i, out_slice] = retargeter.retarget(tip_positions)

    return {"joint_positions": joint_positions, "ik_success": ik_success}


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_retargeters(
    urdf_path: Path = _BIMANUAL_URDF,
) -> tuple[HandRetargeter, HandRetargeter, PandaArmIKSolver, PandaArmIKSolver]:
    """Build and return (left_ret, right_ret, left_ik, right_ik)."""
    left_ret   = HandRetargeter("left")
    right_ret  = HandRetargeter("right")
    left_ik    = PandaArmIKSolver("left",  urdf_path)
    right_ik   = PandaArmIKSolver("right", urdf_path)
    return left_ret, right_ret, left_ik, right_ik
