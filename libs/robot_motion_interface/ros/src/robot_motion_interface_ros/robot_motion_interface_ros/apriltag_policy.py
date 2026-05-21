"""Policy node that consumes AprilTag-derived bottle pose via ROS subscribers.

Variant of depth_feat_policy: same actor obs composition, same JIT-exported
policy, same target-publish path — only the bottle-pose source differs.

Where depth_feat_policy pulled a pre-computed (body, cap, geom) feature
tensor over CUDA IPC from depth_feat_node (sim-trained predictor),
this node subscribes to two plain ROS topics produced by
bottle_apriltag_node:
    * `output.poses_topic` (geometry_msgs/PoseArray, [body, cap]) — world frame
    * `output.geom_topic`  (std_msgs/Float32MultiArray, 4 floats) — static dims

Inside `_policy_update_loop` the latest (body, cap, geom) is packed into a
(1, 10) device tensor with exactly the same layout as the old IPC tensor,
so `_compose_actor_obs` is unchanged — the JIT actor sees the same obs it
saw in training, just with a different upstream geometry source.

Why bypass the predictor:
  * sim-trained DepthFeatureNetFiLM had ~8–15 cm sim2real error in real
    deployment; LSQ post-fit didn't fully close the gap because of frame
    misalignment between training and apriltag world frames.
  * AprilTag gives mm-level metric truth as long as the tag is in view.

Topic names + IPC-free contract live in
`libs/robot_motion_interface/config/april_tag_node_config.yaml`, so
producer and consumer can't drift.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import torch
import yaml
from geometry_msgs.msg import PoseArray
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
from robot_motion_interface.utils.sim2real import joint_mapping


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = yaml.unsafe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def _scale_torch(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Clip then min-max scale [lower, upper] -> [-1, 1] elementwise.

    Clipping first guarantees the output stays inside [-1, 1] even if `value`
    strays outside [lower, upper] — e.g. a depth-feature geom prediction
    landing outside the policy-side bounds when the predictor was trained
    with slightly different bounds, or a real joint snapshot momentarily
    exceeding the soft limits. Matches observations.py behaviour so
    deployment obs has the same range the actor saw during training.
    """
    clipped = value.clamp(min=lower, max=upper)
    return 2.0 * (clipped - lower) / (upper - lower) - 1.0


def _rot6d_from_axis(
    body_pos: torch.Tensor,
    cap_pos: torch.Tensor,
    world_z: torch.Tensor,
    world_x: torch.Tensor,
) -> torch.Tensor:
    """Derive bottle rotation 6D from (cap_pos - body_pos) — covers 2/3 DOF.

    The vector from body to cap is the bottle's principal axis = R[:, 2]
    (third column of the rotation matrix). Rotation around this axis (roll)
    is unobservable from positions alone, so we pick a deterministic world
    reference and Gram-Schmidt the remaining two columns. The bottle is
    roughly rotationally symmetric around its axis, so this loses little
    info in practice — definitely better than identity, which carries zero
    axis information.

    Reference axis = world Z, unless the bottle's principal axis is too
    close to parallel with world Z (cos > 0.95), in which case fall back to
    world X to keep the cross product well-conditioned.

    Output convention matches `_quat_wxyz_to_6d`: first two columns of R
    flattened (Zhou et al. 2019).

    Args:
        body_pos: (B, 3) bottle body position, metres.
        cap_pos:  (B, 3) bottle cap position, metres.
        world_z:  (1, 3) or (B, 3) cached world-Z reference.
        world_x:  (1, 3) or (B, 3) cached world-X reference (fallback).

    Returns:
        (B, 6) rot6d.
    """
    z = cap_pos - body_pos                                       # (B, 3)
    z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-9)         # principal axis
    cos_z = (z * world_z).sum(dim=-1, keepdim=True).abs()        # (B, 1)
    ref = torch.where(cos_z < 0.95, world_z, world_x)            # broadcast (B, 3)
    x = torch.cross(ref, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)         # R[:, 0]
    y = torch.cross(z, x, dim=-1)                                # R[:, 1]
    return torch.cat([x, y], dim=-1)                             # (B, 6)


def _quat_wxyz_to_6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Quaternion (..., 4) wxyz -> 6D rotation (..., 6).

    First two columns of the rotation matrix, flattened. Continuous,
    free of q ≡ -q antipodal ambiguity (Zhou et al., 2019). Mirrors
    `observations.py::quat_to_6d` exactly — duplicated here to avoid
    a cross-repo import.
    """
    r, i, j, k = torch.unbind(quat_wxyz, -1)
    two_s = 2.0 / (quat_wxyz * quat_wxyz).sum(-1)
    c0_x = 1.0 - two_s * (j * j + k * k)
    c0_y = two_s * (i * j + k * r)
    c0_z = two_s * (i * k - j * r)
    c1_x = two_s * (i * j - k * r)
    c1_y = 1.0 - two_s * (i * i + k * k)
    c1_z = two_s * (j * k + i * r)
    return torch.stack((c0_x, c0_y, c0_z, c1_x, c1_y, c1_z), dim=-1)


# Bottle-geom bounds, MUST match observations.py:147-151. Used to convert
# depth_feat_node's metres-space geom prediction into the [-1, 1] scaled
# 12-d `bottleGeomCfg` slot the actor consumes.
_BOTTLE_GEOM_LOWER = (0.03, 0.10, 0.02, 0.02)
_BOTTLE_GEOM_UPPER = (0.05, 0.20, 0.055, 0.04)

# ---------------------------------------------------------------------------
# ApriltagPolicyNode
# ---------------------------------------------------------------------------
class ApriltagPolicyNode(Node):
    def __init__(self):
        super().__init__("apriltag_policy_node")

        # -- 1. params & cfgs --
        self._init_callback_mutex_groups()
        self._declare_parameters()
        self._load_configs()
        self._init_device()

        # -- 2. locks --
        self._init_threading_locks()

        # -- 3. components --
        # No rsl_rl import: the policy comes in as a TorchScript file with
        # actor + EmpiricalNormalization already baked together (see
        # IsaacLab's export_policy_as_jit). The node loads it with
        # torch.jit.load and calls it as `actions = self.policy(obs)`.
        self._build_joint_mappings()
        self._read_hand_link_counts()
        self._build_policy_and_normalizer()
        self._init_state_buffers()

        # -- 4. subs & pubs --
        self._init_pub_sub()

        # -- 5. timer + summary --
        self._init_policy_timer()
        self.node_cfg = self._build_node_cfg_summary()
        self.get_logger().info("ApriltagPolicyNode ready:\n" + yaml.safe_dump(self.node_cfg, sort_keys=False))

        self.get_logger().info("ApriltagPolicyNode initialization complete. Waiting for 2 seconds before spinning node.")
        time.sleep(2.0)

    # ------------------------------------------------------------------
    # Step 1: params & cfgs
    # ------------------------------------------------------------------
    def _init_callback_mutex_groups(self) -> None:
        # Four independent mutex groups so /joint_states, fk_topic, the two
        # apriltag topics, and the policy timer don't serialize each other at
        # the executor level. Each callback's own lock (joint_lock / fk_lock /
        # apriltag_lock) still protects shared state vs the timer reader.
        self.js_grp = MutuallyExclusiveCallbackGroup()
        self.fk_grp = MutuallyExclusiveCallbackGroup()
        self.apriltag_grp = MutuallyExclusiveCallbackGroup()
        self.infer_grp = MutuallyExclusiveCallbackGroup()

    def _declare_parameters(self) -> None:
        """Declare ROS parameters: paths + da3 model override only.

        All paths must be absolute. Frequencies / runtime knobs come from cfg files.
        Missing files will fail naturally at cfg-load time.
        """
        # All training-side artifacts live under a single policy_log dir,
        # mirroring rsl_rl's checkpoint folder layout. Each retraining run
        # drops in a new policy.pt + cfg yamls; deployment just rsync's the
        # log dir over and points this parameter at it. Defaults to
        # <RMI_ROOT>/runtime/policy_log/ — change via launch arg to swap
        # between deployed runs without touching the node code.
        #
        # Expected layout (mirrors rsl_rl + rslppo_play.py export output):
        #   <policy_log_dir>/
        #     exported/
        #       policy.pt              # TorchScript actor + normalizer
        #       runtime_cfg_play.yaml  # joint mappings + dt/action_scale/EMA
        #     params/
        #       agent.yaml             # PPO agent config (read for logging)
        #       env.yaml               # actor obs key list + dims
        self.declare_parameter("policy_log_dir", str((RMI_ROOT / "runtime" / "policy_log").resolve()))
        self.declare_parameter("hand_env_cfg_path", str((RMI_ROOT / "runtime" / "HandEnv.yaml").resolve()))
        self.declare_parameter("driver_cfg_path", str((RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml").resolve()))
        self.declare_parameter("policy_node_cfg_path", str((RMI_ROOT / "config" / "rl_policy_node_config.yaml").resolve()))
        self.declare_parameter("fk_cfg_path", str((RMI_ROOT / "config" / "fk_config.yaml").resolve()))
        self.declare_parameter("da3_cfg_path", str((RMI_ROOT / "config" / "da3_compile_config.yaml").resolve()))
        self.declare_parameter(
            "apriltag_cfg_path",
            str((RMI_ROOT / "config" / "april_tag_node_config.yaml").resolve()),
        )

        # Derive per-file paths from the policy_log root. If your deployment
        # layout differs, override `policy_log_dir` rather than each path.
        self.policy_log_dir = Path(self.get_parameter("policy_log_dir").value)
        self.jit_policy_path = self.policy_log_dir / "exported" / "policy.pt"
        self.runtime_cfg_path = self.policy_log_dir / "exported" / "runtime_cfg_play.yaml"
        self.agent_cfg_path = self.policy_log_dir / "params" / "agent.yaml"
        self.env_cfg_path = self.policy_log_dir / "params" / "env.yaml"

        self.hand_env_cfg_path = Path(self.get_parameter("hand_env_cfg_path").value)
        self.driver_cfg_path = Path(self.get_parameter("driver_cfg_path").value)
        self.policy_node_cfg_path = Path(self.get_parameter("policy_node_cfg_path").value)
        self.fk_cfg_path = Path(self.get_parameter("fk_cfg_path").value)
        self.da3_cfg_path = Path(self.get_parameter("da3_cfg_path").value)
        self.apriltag_cfg_path = Path(self.get_parameter("apriltag_cfg_path").value)

    def _load_configs(self) -> None:
        """Load all YAML cfgs. Missing files raise FileNotFoundError naturally."""
        self.runtime_cfg = _load_yaml(self.runtime_cfg_path)
        self.env_cfg = _load_yaml(self.env_cfg_path)
        # agent.yaml is loaded for diagnostics (PPO hparams, normalization
        # flag) — the JIT policy doesn't actually need it at runtime since
        # actor + normalizer are baked into the ScriptModule.
        self.agent_cfg = _load_yaml(self.agent_cfg_path)
        self.hand_env_cfg = _load_yaml(self.hand_env_cfg_path)
        self.driver_cfg = _load_yaml(self.driver_cfg_path)
        self.policy_node_cfg = _load_yaml(self.policy_node_cfg_path)
        self.fk_cfg = _load_yaml(self.fk_cfg_path)
        self.da3_full_cfg = _load_yaml(self.da3_cfg_path)
        self.apriltag_full_cfg = _load_yaml(self.apriltag_cfg_path)

        self.get_logger().info("#### Apriltag policy node configs: ####")
        self.get_logger().info(f"policy_log_dir:       {self.policy_log_dir}")
        self.get_logger().info(f"  jit_policy_path:    {self.jit_policy_path}")
        self.get_logger().info(f"  runtime_cfg_path:   {self.runtime_cfg_path}")
        self.get_logger().info(f"  agent_cfg_path:     {self.agent_cfg_path}")
        self.get_logger().info(f"  env_cfg_path:       {self.env_cfg_path}")
        self.get_logger().info(f"hand_env_cfg_path:    {self.hand_env_cfg_path}")
        self.get_logger().info(f"driver_cfg_path:      {self.driver_cfg_path}")
        self.get_logger().info(f"policy_node_cfg_path: {self.policy_node_cfg_path}")
        self.get_logger().info(f"da3_cfg_path:         {self.da3_cfg_path}")
        self.get_logger().info(f"fk_cfg_path:          {self.fk_cfg_path}")
        self.get_logger().info(f"apriltag_cfg_path:    {self.apriltag_cfg_path}")

    def _init_device(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available.")
        self.device = torch.device("cuda")
        self.get_logger().info(f"Aux policy device: {self.device}")

    # ------------------------------------------------------------------
    # Step 2: locks
    # ------------------------------------------------------------------
    def _init_threading_locks(self) -> None:
        self.joint_lock = threading.Lock()
        self.fk_lock = threading.Lock()
        self.apriltag_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Step 3: components
    # ------------------------------------------------------------------
    def _to_dev_t(self, data: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, device=self.device)

    def _build_joint_mappings(self) -> None:
        # Real driver order: left panda+tesollo, then right panda+tesollo.
        left_real = [
            "left_" + n
            for n in self.driver_cfg["left_panda_joint_names"] + self.driver_cfg["left_tesollo_joint_names"]
        ]
        right_real = [
            "right_" + n
            for n in self.driver_cfg["right_panda_joint_names"] + self.driver_cfg["right_tesollo_joint_names"]
        ]
        self.real_joint_names = left_real + right_real
        self.action_num = len(self.real_joint_names)

        # Bidirectional index map between policy order and real driver order.
        policy_joint_names = self.runtime_cfg["bimanual_joint_names"]
        policy2real_idx, real2policy_idx = joint_mapping(policy_joint_names, self.real_joint_names)
        self.policy2real_idx = self._to_dev_t(policy2real_idx, dtype=torch.long)
        self.real2policy_idx = self._to_dev_t(real2policy_idx, dtype=torch.long)

        # Per-arm policy-order indices into the bimanual action vector.
        policy_action_indices = self.runtime_cfg["policy_action_indices_dict"]
        self.left_policy_indices = self._to_dev_t(policy_action_indices["left"], dtype=torch.long)
        self.right_policy_indices = self._to_dev_t(policy_action_indices["right"], dtype=torch.long)

        # Per-arm scales / soft limits / vel limits (policy order, per-arm slices).
        scale_dict = self.runtime_cfg["robot_action_scale_dict"]
        limits_dict = self.runtime_cfg["robot_joint_limits_dict"]
        vel_scale_left = self._to_dev_t(scale_dict["left_joint_vel_action"])
        vel_scale_right = self._to_dev_t(scale_dict["right_joint_vel_action"])
        lower_left = self._to_dev_t(limits_dict["left_joint_pose_soft_lower"])
        lower_right = self._to_dev_t(limits_dict["right_joint_pose_soft_lower"])
        upper_left = self._to_dev_t(limits_dict["left_joint_pose_soft_upper"])
        upper_right = self._to_dev_t(limits_dict["right_joint_pose_soft_upper"])
        vel_limit_left = self._to_dev_t(limits_dict["left_joint_vel"])
        vel_limit_right = self._to_dev_t(limits_dict["right_joint_vel"])

        # Scatter per-arm slices into full bimanual policy-order vectors.
        # Integration runs in policy order; conversion to real order only happens
        # at the publish boundary in _policy_update_loop.
        def _scatter_full(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            full = torch.zeros(self.action_num, dtype=torch.float32, device=self.device)
            full[self.left_policy_indices] = left
            full[self.right_policy_indices] = right
            return full

        self.full_scale_policy = _scatter_full(vel_scale_left, vel_scale_right).unsqueeze(0)
        self.full_lower_policy = _scatter_full(lower_left, lower_right).unsqueeze(0)
        self.full_upper_policy = _scatter_full(upper_left, upper_right).unsqueeze(0)

        # Per-arm bounds kept for obs scaling (policy order, per-arm).
        self.left_soft_lower = lower_left.unsqueeze(0)
        self.left_soft_upper = upper_left.unsqueeze(0)
        self.right_soft_lower = lower_right.unsqueeze(0)
        self.right_soft_upper = upper_right.unsqueeze(0)
        self.left_vel_limit = vel_limit_left.unsqueeze(0)
        self.right_vel_limit = vel_limit_right.unsqueeze(0)

    def _read_hand_link_counts(self) -> None:
        """Resolve fk topic + per-group prefixed link names used to look up
        positions inside fk_pose_dict during compose_obs.

        FK itself runs in the external fk_node. fk_cfg.link_names is the source
        of truth for both the published PoseArray order and the fk_pose_dict
        keys.
        """
        hand_link_dict = self.hand_env_cfg["env"]["robot"]["linkNames"]
        finger_tip_links = list(hand_link_dict["finger_tips"])
        hand_palm_links = list(hand_link_dict["hand_palm"])

        # Prefixed names matching pinocchio frames in fk_node / fk_cfg.link_names.
        self.left_fingertip_names = [f"left_{n}" for n in finger_tip_links]
        self.right_fingertip_names = [f"right_{n}" for n in finger_tip_links]
        self.left_hand_base_names = [f"left_{n}" for n in hand_palm_links]
        self.right_hand_base_names = [f"right_{n}" for n in hand_palm_links]

        self.fk_link_names = list(self.fk_cfg["link_names"])
        self.fk_n_links = len(self.fk_link_names)
        self.fk_topic = str(self.fk_cfg["fk_topic"])

    def _build_policy_and_normalizer(self) -> None:
        """Resolve obs dims from env.yaml, then load the JIT-exported policy.

        The TorchScript file at `self.jit_policy_path` was produced by
        IsaacLab's `export_policy_as_jit`, which wraps:
            forward(obs) = actor(normalizer(obs))
        So inference is just `actions = self.policy(obs_clamped)` — no
        separate EmpiricalNormalization, no rsl-rl class import, no
        state_dict surgery. Re-export from rslppo_play.py whenever the
        upstream training run changes.

        The depth feature still gets spliced into the obs vector at the
        `bottleBodyPosNoised` / `bottleCapPosNoised` / `bottleGeomCfg` slots
        inside `_compose_actor_obs`; that part is independent of how the
        actor weights happen to be packaged.
        """
        # 1. Cache scaling tensors used by _compose_actor_obs.
        # Bottle-geom bounds (1, 4) — MUST match observations.py:147-151.
        self._geom_lower_t = self._to_dev_t(list(_BOTTLE_GEOM_LOWER)).unsqueeze(0)
        self._geom_upper_t = self._to_dev_t(list(_BOTTLE_GEOM_UPPER)).unsqueeze(0)

        # World reference axes used by _rot6d_from_axis to Gram-Schmidt the
        # bottle's rotation from (cap_pos - body_pos). World Z is the primary
        # reference; world X is the fallback when the bottle axis is nearly
        # parallel to world Z (avoids degenerate cross-product). Shape (1, 3)
        # so they broadcast against the (B, 3) pos tensors.
        self._world_z = self._to_dev_t([0.0, 0.0, 1.0]).unsqueeze(0)
        self._world_x = self._to_dev_t([1.0, 0.0, 0.0]).unsqueeze(0)

        # 2. Read obs dims / actor key order from env.yaml. The JIT file
        # carries the actor weights but NOT the obs key list, so we still
        # need env_cfg as the single source of truth for the obs layout.
        self.n_stack_frame = int(self.env_cfg["n_stack_frame"])
        self.actor_obs_keys: list[str] = list(self.env_cfg["actors"])
        self.actor_obs_unstacked_space = int(self.env_cfg["actor_obs_unstacked_space"])
        self.critic_obs_unstacked_space = int(self.env_cfg["critic_obs_unstacked_space"])
        self.num_actor_obs = self.n_stack_frame * self.actor_obs_unstacked_space
        self.num_critic_obs = self.n_stack_frame * self.critic_obs_unstacked_space

        # Mirror aux_policy_v2's `obs_dim` / `state_dim` names so the rest of
        # the class (buffers, summary, etc.) keeps working unchanged.
        self.obs_dim = self.num_actor_obs
        self.state_dim = self.num_critic_obs

        # cfg-internal sanity: obs_DOF summed over the actor keys must equal
        # actor_obs_unstacked_space — catches yaml drift early.
        obs_dof = self.env_cfg["obs_DOF"]
        inferred_unstacked = sum(int(obs_dof[k]) for k in self.actor_obs_keys)
        if inferred_unstacked != self.actor_obs_unstacked_space:
            raise ValueError(
                f"sum(obs_DOF[actor_obs_keys])={inferred_unstacked} "
                f"!= actor_obs_unstacked_space={self.actor_obs_unstacked_space}"
            )

        # 3. Load the JIT-exported policy. The exporter bakes the
        # EmpiricalNormalization in as the first module of `forward`, so
        # calling `self.policy(raw_obs)` returns actions directly.
        if not self.jit_policy_path.exists():
            raise FileNotFoundError(
                f"JIT policy not found at {self.jit_policy_path}. "
                f"Re-export via rslppo_play.py (export_policy_as_jit) or "
                f"point --jit_policy_path at the correct exported/ dir."
            )
        self.policy = torch.jit.load(
            str(self.jit_policy_path), map_location=self.device
        ).to(self.device)
        self.policy.eval()
        self.get_logger().info(
            f"Loaded JIT policy: {self.jit_policy_path}  "
            f"(num_actor_obs={self.num_actor_obs}, num_actions={self.action_num})"
        )

    def _init_state_buffers(self) -> None:
        """Allocate all reusable np / torch buffers used by the policy loop.

        Sized off cfg-derived dims; allocated once here so the policy loop never
        does per-step allocations.
        """
        A = self.action_num

        # /joint_states latest snapshot (CPU, written by sub callback).
        self.latest_joint_pos_real_np: np.ndarray = np.zeros((A,), dtype=np.float32)
        self.latest_joint_vel_real_np: np.ndarray = np.zeros((A,), dtype=np.float32)
        self.has_joint_state: bool = False

        # Per-step CPU snapshots copied under joint_lock.
        self._joint_pos_snapshot_np: np.ndarray = np.zeros((A,), dtype=np.float32)
        self._joint_vel_snapshot_np: np.ndarray = np.zeros((A,), dtype=np.float32)

        # GPU mirrors used by the policy loop (filled from CPU snapshots).
        # All in policy order — joint_pos/vel are converted from real at the loop top.
        self._joint_pos_policy_t: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        self._joint_vel_policy_t: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        self._targets_snapshot_t: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        self._prev_actions_snapshot_t: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)

        # Persistent policy state (policy order; lives across steps).
        self.targets_policy: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        self.targets_initialized: bool = False
        self.prev_actions_policy: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        # Previous unstacked actor obs, used for the second frame in the
        # stacked obs (n_stack_frame=2 case).
        self.prev_actor_obs: torch.Tensor = torch.zeros(
            (1, self.actor_obs_unstacked_space), dtype=torch.float32, device=self.device
        )

        # Stacked-obs working buffers (reused every step).
        self._stacked_obs_buf: torch.Tensor = torch.zeros((1, self.obs_dim), dtype=torch.float32, device=self.device)
        self._obs_clamped_buf: torch.Tensor = torch.zeros((1, self.obs_dim), dtype=torch.float32, device=self.device)

        # FK results: a single dict keyed by "<linkname>_pos" (xyz, float32[3]) and
        # "<linkname>_wxyz" (float32[4]). Pre-allocated in fk_cfg.link_names order;
        self.fk_pose_dict: dict[str, np.ndarray] = {}
        for name in self.fk_link_names:
            self.fk_pose_dict[name + "_pos"] = np.zeros(3, dtype=np.float32)
            self.fk_pose_dict[name + "_wxyz"] = np.zeros(4, dtype=np.float32)
        self.has_fk: bool = False

        # AprilTag-derived bottle pose + geom (filled by ROS callbacks). All
        # CPU buffers — the policy loop copies them under apriltag_lock,
        # packs into a (1, 10) device tensor once per step, and from then on
        # the rest of the obs path is identical to the IPC variant.
        # Layout matches the old IPC tensor:
        #     [body(3), cap(3), geom(4)]   metres in world frame
        # so _compose_actor_obs is unchanged.
        self.latest_apriltag_body_np: np.ndarray = np.zeros(3, dtype=np.float32)
        self.latest_apriltag_cap_np:  np.ndarray = np.zeros(3, dtype=np.float32)
        self.latest_apriltag_geom_np: np.ndarray = np.zeros(4, dtype=np.float32)
        self.has_apriltag_pose: bool = False
        self.has_apriltag_geom: bool = False
        # Per-step working buffer + device tensor (reused, no per-step alloc).
        self._apriltag_feat_np: np.ndarray = np.zeros(10, dtype=np.float32)
        self._apriltag_feat_t: torch.Tensor = torch.zeros(
            (1, 10), dtype=torch.float32, device=self.device
        )

    # ------------------------------------------------------------------
    # Step 4: subs & pubs
    # ------------------------------------------------------------------
    def _init_pub_sub(self) -> None:
        """Wire ROS I/O: subscribe to /joint_states + fk_topic + apriltag
        bottle topics; publish /target_joint_states."""
        self.target_pub = self.create_publisher(
            JointState, "/target_joint_states", HIGH_RELIA_QOS
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._sub_joint_state_cb,
            HIGH_PERF_QOS,
            callback_group=self.js_grp,
        )
        # Single FK PoseArray from fk_node. Pose order matches fk_cfg.link_names.
        self.create_subscription(
            PoseArray,
            self.fk_topic,
            self._sub_fk_cb,
            HIGH_PERF_QOS,
            callback_group=self.fk_grp,
        )
        # AprilTag bottle pose + geom from bottle_apriltag_node. Topic names
        # come from the same yaml the producer reads, so they can't drift.
        out_cfg = self.apriltag_full_cfg.get("output", {})
        self._apriltag_poses_topic = str(out_cfg.get("poses_topic", "/bottle_apriltag/poses"))
        self._apriltag_geom_topic  = str(out_cfg.get("geom_topic",  "/bottle_apriltag/geom"))
        self.create_subscription(
            PoseArray,
            self._apriltag_poses_topic,
            self._sub_apriltag_poses_cb,
            HIGH_RELIA_QOS,
            callback_group=self.apriltag_grp,
        )
        self.create_subscription(
            Float32MultiArray,
            self._apriltag_geom_topic,
            self._sub_apriltag_geom_cb,
            HIGH_RELIA_QOS,
            callback_group=self.apriltag_grp,
        )

    def _sub_apriltag_poses_cb(self, msg: PoseArray) -> None:
        """Snapshot body/cap world positions into the latest buffer.

        Expected message convention (matches bottle_apriltag_node):
          poses[0] = body
          poses[1] = cap
        Both in `output.world_frame_id` (default "world"), metres.
        """
        if len(msg.poses) < 2:
            self.get_logger().error(
                f"{self._apriltag_poses_topic} expected >=2 poses (body, cap), "
                f"got {len(msg.poses)}"
            )
            return
        body = msg.poses[0].position
        cap  = msg.poses[1].position
        with self.apriltag_lock:
            self.latest_apriltag_body_np[0] = body.x
            self.latest_apriltag_body_np[1] = body.y
            self.latest_apriltag_body_np[2] = body.z
            self.latest_apriltag_cap_np[0]  = cap.x
            self.latest_apriltag_cap_np[1]  = cap.y
            self.latest_apriltag_cap_np[2]  = cap.z
            self.has_apriltag_pose = True

    def _sub_apriltag_geom_cb(self, msg: Float32MultiArray) -> None:
        """Snapshot the static [body_r, body_h, cap_r, cap_h] override."""
        if len(msg.data) != 4:
            self.get_logger().error(
                f"{self._apriltag_geom_topic} expected 4 floats, got {len(msg.data)}"
            )
            return
        with self.apriltag_lock:
            for i in range(4):
                self.latest_apriltag_geom_np[i] = float(msg.data[i])
            self.has_apriltag_geom = True

    def _sub_fk_cb(self, msg: PoseArray) -> None:
        """Write the inbound PoseArray into fk_pose_dict in fk_cfg.link_names order.

        Per link, two entries are stored:
          - "<linkname>_pos":  float32[3]  (x, y, z)
          - "<linkname>_wxyz": float32[4]  (w, x, y, z)
        fk_lock protects writers vs the compose_obs reader.
        """
        if len(msg.poses) != self.fk_n_links:
            self.get_logger().error(
                f"{self.fk_topic} pose count mismatch: {len(msg.poses)} vs "
                f"expected {self.fk_n_links} (fk_cfg.link_names)"
            )
            rclpy.shutdown()
            return

        with self.fk_lock:
            for name, p in zip(self.fk_link_names, msg.poses):
                pos = self.fk_pose_dict[name + "_pos"]
                pos[0] = p.position.x
                pos[1] = p.position.y
                pos[2] = p.position.z
                wxyz = self.fk_pose_dict[name + "_wxyz"]
                wxyz[0] = p.orientation.w
                wxyz[1] = p.orientation.x
                wxyz[2] = p.orientation.y
                wxyz[3] = p.orientation.z
            self.has_fk = True

    def _sub_joint_state_cb(self, msg: JointState) -> None:
        """Snapshot the latest joint state into pre-allocated CPU buffers."""
        if len(msg.position) != self.action_num:
            # Fail loudly — driver_cfg's joint name list and the actual
            # /joint_states msg must agree, otherwise the policy/real index
            # maps in _build_joint_mappings are all wrong.
            raise RuntimeError(
                f"/joint_states DoF mismatch: msg.position has "
                f"{len(msg.position)} entries, driver_cfg expects "
                f"{self.action_num}."
            )
        # Convert outside the lock to keep the critical section to a memcpy.
        pos_np = np.asarray(msg.position, dtype=np.float32)
        vel_np = np.asarray(msg.velocity, dtype=np.float32)
        with self.joint_lock:
            np.copyto(self.latest_joint_pos_real_np, pos_np)
            np.copyto(self.latest_joint_vel_real_np, vel_np)
            self.has_joint_state = True

    # ------------------------------------------------------------------
    # Step 5: timer + summary 
    # helpers: FK, obs composition, target integration, timing log
    # ------------------------------------------------------------------
    def _init_policy_timer(self) -> None:
        """Initialize policy-step runtime scalars and start the policy ROS timer."""
        self.policy_hz = float(self.policy_node_cfg["infer_rate"])
        if self.policy_hz <= 0.0:
            raise ValueError(f"infer_rate must be > 0, got {self.policy_hz}")

        self.dt = float(self.runtime_cfg["dt"])
        self.action_ema = float(self.runtime_cfg["action_EMA"])
        # action_scale comes straight from runtime_cfg now (1.0 at time of
        # writing per logs/.../runtime_cfg_play.yaml); aux_policy_v2 had a
        # hardcoded 0.25 override that's been removed — must stay in sync
        # with the training-side env's action_scale or target integration
        # will drift away from the trained dynamics.
        self.action_scale = float(self.runtime_cfg["action_scale"])

        timer_dt = 1.0 / self.policy_hz
        if abs(timer_dt - self.dt) > 1e-5:
            self.get_logger().warn(
                f"Policy timer dt ({timer_dt:.6f}) != runtime dt ({self.dt:.6f}). "
                "Target integration uses runtime dt."
            )

        self.policy_timer = self.create_timer(
            timer_dt,
            self._policy_update_loop,
            callback_group=self.infer_grp,
        )
        self.get_logger().info(
            f"Policy timer started @ {self.policy_hz:.3f}Hz (period={timer_dt:.6f}s)."
        )

    def _compose_actor_obs(
        self,
        joint_pos_policy: torch.Tensor,  # [1, A] policy order
        joint_vel_policy: torch.Tensor,  # [1, A] policy order
        targets_policy: torch.Tensor,    # [1, A] policy order
        feat_t: torch.Tensor,            # [1, 10] depth-feature: body(3) + cap(3) + geom(4) in metres
    ) -> torch.Tensor:
        """Build the unstacked actor observation vector in the policy's expected key order.

        Three families of slots are filled here:
          - proprio (joints, targets):   from /joint_states (already in policy order).
          - FK (fingertips, hand base):  from fk_pose_dict, populated by fk_node.
                                         Hand-base quaternions get converted to 6D.
          - bottle (pos + geom):         from feat_t, the depth-feature node's
                                         IPC tensor. The 4-d raw geom is rescaled
                                         to [-1, 1] using observations.py bounds
                                         and repeated 3x to fill the 12-d slot.

        FK from real sensors is naturally noisy, so the FK-derived obs fill the
        *Noised slots in actor_obs_keys without any extra noise injection.
        Same goes for the depth feature — the trained predictor's error stands
        in for the env's `bottle_pos_noise` term.
        """
        left_jp = joint_pos_policy[:, self.left_policy_indices]
        right_jp = joint_pos_policy[:, self.right_policy_indices]
        left_jv = joint_vel_policy[:, self.left_policy_indices]
        right_jv = joint_vel_policy[:, self.right_policy_indices]
        left_tgt = targets_policy[:, self.left_policy_indices]
        right_tgt = targets_policy[:, self.right_policy_indices]

        left_jp_s = _scale_torch(left_jp, self.left_soft_lower, self.left_soft_upper)
        right_jp_s = _scale_torch(right_jp, self.right_soft_lower, self.right_soft_upper)
        left_jv_s = left_jv / self.left_vel_limit
        right_jv_s = right_jv / self.right_vel_limit

        # Snapshot FK positions + quats out of fk_pose_dict (concatenate into
        # flat ndarrays under fk_lock), then H2D outside the lock. Quaternions
        # are stored wxyz (see _sub_fk_cb), which matches observations.py's
        # quat_to_6d convention.
        with self.fk_lock:
            l_ft_pos_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.left_fingertip_names])
            r_ft_pos_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.right_fingertip_names])
            l_hb_pos_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.left_hand_base_names])
            l_hb_wxyz_np = np.concatenate([self.fk_pose_dict[n + "_wxyz"] for n in self.left_hand_base_names])
            r_hb_pos_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.right_hand_base_names])
            r_hb_wxyz_np = np.concatenate([self.fk_pose_dict[n + "_wxyz"] for n in self.right_hand_base_names])
        l_ft_pos = torch.from_numpy(l_ft_pos_np).unsqueeze(0).to(self.device, non_blocking=True)
        r_ft_pos = torch.from_numpy(r_ft_pos_np).unsqueeze(0).to(self.device, non_blocking=True)
        l_hb_pos = torch.from_numpy(l_hb_pos_np).unsqueeze(0).to(self.device, non_blocking=True)
        r_hb_pos = torch.from_numpy(r_hb_pos_np).unsqueeze(0).to(self.device, non_blocking=True)

        # quat (wxyz) -> 6D. Single hand-base link per side, so input/output are flat.
        l_hb_wxyz = torch.from_numpy(l_hb_wxyz_np).unsqueeze(0).to(self.device, non_blocking=True)
        r_hb_wxyz = torch.from_numpy(r_hb_wxyz_np).unsqueeze(0).to(self.device, non_blocking=True)
        l_hb_rot6d = _quat_wxyz_to_6d(l_hb_wxyz)   # (1, 6)
        r_hb_rot6d = _quat_wxyz_to_6d(r_hb_wxyz)   # (1, 6)

        # Split depth feature tensor: [body(3), cap(3), geom_raw(4)] in metres.
        body_pos_m = feat_t[:, 0:3]                # (1, 3)
        cap_pos_m  = feat_t[:, 3:6]                # (1, 3)
        geom_raw_m = feat_t[:, 6:10]               # (1, 4) raw metres — scaled inline below

        # Inline scaling matches observations.py:147-151 — bottle geom is
        # scaled to [-1, 1] and repeated 3x at obs-dict construction time,
        # not pre-computed.
        full_obs = {
            # proprio (clean + noised both point to the same scaled vector
            # because real sensors already carry the noise)
            "leftJointPosScaled": left_jp_s,
            "rightJointPosScaled": right_jp_s,
            "leftJointPosScaledNoised": left_jp_s,
            "rightJointPosScaledNoised": right_jp_s,
            "leftJointVelScaled": left_jv_s,
            "rightJointVelScaled": right_jv_s,
            "leftTargets": left_tgt,
            "rightTargets": right_tgt,
            # FK proprio
            "leftFingerTipsPos": l_ft_pos,
            "rightFingerTipsPos": r_ft_pos,
            "leftFingerTipsPosNoised": l_ft_pos,
            "rightFingerTipsPosNoised": r_ft_pos,
            "leftHandBasePos": l_hb_pos,
            "rightHandBasePos": r_hb_pos,
            "leftHandBasePosNoised": l_hb_pos,
            "rightHandBasePosNoised": r_hb_pos,
            "leftHandBaseRot6D": l_hb_rot6d,
            "rightHandBaseRot6D": r_hb_rot6d,
            "leftHandBaseRot6DNoised": l_hb_rot6d,
            "rightHandBaseRot6DNoised": r_hb_rot6d,
            # bottle (from depth feature predictor)
            "bottleBodyPos": body_pos_m,
            "bottleCapPos": cap_pos_m,
            "bottleBodyPosNoised": body_pos_m,
            "bottleCapPosNoised": cap_pos_m,
            "bottleGeomCfg": _scale_torch(
                geom_raw_m, self._geom_lower_t, self._geom_upper_t
            ).repeat(1, 3),
            # bottle body rot6d: derived from (cap_pos - body_pos) via
            # _rot6d_from_axis. Covers 2/3 DOF — roll around the bottle's
            # principal axis is unobservable from positions alone, but the
            # bottle is roughly rotationally symmetric so this is a
            # reasonable approximation. Replaces the previous identity-rot6d
            # placeholder, which carried zero axis information.
            "bottleBodyRot6DNoised": _rot6d_from_axis(
                body_pos_m, cap_pos_m, self._world_z, self._world_x
            ),
        }
        actor_obs = torch.cat([full_obs[k] for k in self.actor_obs_keys], dim=-1)
        if int(actor_obs.shape[-1]) != self.actor_obs_unstacked_space:
            raise ValueError(
                f"composed actor obs dim={int(actor_obs.shape[-1])} "
                f"!= actor_obs_unstacked_space={self.actor_obs_unstacked_space}"
            )
        return actor_obs

    def _compute_next_targets(
        self,
        actions_policy: torch.Tensor,        # [1, A] policy order
        prev_actions_policy: torch.Tensor,   # [1, A] policy order
        joint_pos_policy: torch.Tensor,      # [1, A] policy order
        prev_targets_policy: torch.Tensor,
        alpha: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EMA-smooth actions and integrate to next joint targets (policy order), clamped to soft limits."""
        ema_actions = (
            actions_policy.clamp(-1.0, 1.0) * self.action_ema
            + prev_actions_policy * (1.0 - self.action_ema)
        )
        next_targets_policy = torch.clamp(
            joint_pos_policy * (1.0 - alpha) + prev_targets_policy * alpha + ema_actions * self.dt * self.full_scale_policy * self.action_scale,
            min=self.full_lower_policy,
            max=self.full_upper_policy,
        )
        return next_targets_policy, ema_actions

    def _log_policy_timing(
        self,
        feat_copy_s: float,
        js_lock_s: float,
        prev_var_s: float,
        fk_compose_obs_s: float,
        policy_inference_s: float,
        policy_inference_sync_s: float,
        total_s: float,
    ) -> None:
        """Warn when the policy step blows past per-stage / total budgets."""
        stage_threshold_s = 0.01
        total_threshold_s = 0.02
        stages = {
            "feat_copy": feat_copy_s,
            "js_lock": js_lock_s,
            "prev_var": prev_var_s,
            "fk_lock_compose_obs": fk_compose_obs_s,
            "policy_inference": policy_inference_s,
            "policy_inference_sync": policy_inference_sync_s,
        }
        timing = ", ".join(f"{n}={v:.4f}s" for n, v in stages.items())
        slow_stages = [f"{n}={v:.4f}s" for n, v in stages.items() if v > stage_threshold_s]
        total_slow = total_s > total_threshold_s
        if total_slow or slow_stages:
            parts = []
            if total_slow:
                parts.append(f"total={total_s:.4f}s>{total_threshold_s:.3f}s")
            if slow_stages:
                parts.append(f"stages>{stage_threshold_s:.3f}s: " + ", ".join(slow_stages))
            parts.append(timing)
            self.get_logger().warn("[SLOW_POLICY] " + " | ".join(parts))
        else:
            self.get_logger().debug(f"[POLICY_TIMING] total={total_s:.4f}s | {timing}")

    def _policy_update_loop(self) -> None:
        """Single policy step: snapshot sensors, run inference, integrate targets, publish."""
        t0 = time.perf_counter()
        if not self.has_joint_state:
            return

        # 1) Snapshot apriltag (body, cap, geom) under apriltag_lock and pack
        # into the same (1, 10) device tensor layout the old IPC path used,
        # so _compose_actor_obs is unchanged. If a callback hasn't fired yet
        # we can't run the actor — skip this tick and warn.
        if not self.has_apriltag_pose or not self.has_apriltag_geom:
            self.get_logger().warn(
                f"AprilTag data not yet received "
                f"(pose={self.has_apriltag_pose}, geom={self.has_apriltag_geom}) "
                f"— skipping policy tick."
            )
            return
        with self.apriltag_lock:
            self._apriltag_feat_np[0:3] = self.latest_apriltag_body_np
            self._apriltag_feat_np[3:6] = self.latest_apriltag_cap_np
            self._apriltag_feat_np[6:10] = self.latest_apriltag_geom_np
        # CPU -> GPU once per step; reuses the pre-allocated buffer so no per-
        # step alloc. unsqueeze(0) restores the (1, 10) shape the actor expects.
        self._apriltag_feat_t[0].copy_(
            torch.from_numpy(self._apriltag_feat_np), non_blocking=True
        )
        feat_t = self._apriltag_feat_t
        # torch.cuda.synchronize()
        t01 = time.perf_counter()

        # 2) Snapshot latest joint arrays (real order) under joint_lock.
        with self.joint_lock:
            if not self.has_joint_state:
                return
            np.copyto(self._joint_pos_snapshot_np, self.latest_joint_pos_real_np)
            np.copyto(self._joint_vel_snapshot_np, self.latest_joint_vel_real_np)
        # torch.cuda.synchronize()
        t02 = time.perf_counter()

        # 3) Wait until fk_node has produced at least one sample for all four groups.
        if not self.has_fk:
            return

        # 4) CPU snapshots -> GPU tensors, then real->policy reorder.
        joint_pos_real_t = torch.from_numpy(self._joint_pos_snapshot_np).to(self.device, non_blocking=True)
        joint_vel_real_t = torch.from_numpy(self._joint_vel_snapshot_np).to(self.device, non_blocking=True)
        self._joint_pos_policy_t[0, :].copy_(joint_pos_real_t[self.real2policy_idx])
        self._joint_vel_policy_t[0, :].copy_(joint_vel_real_t[self.real2policy_idx])

        if not self.targets_initialized:
            self.targets_policy.copy_(self._joint_pos_policy_t)
            self.targets_initialized = True
        self._targets_snapshot_t.copy_(self.targets_policy)
        self._prev_actions_snapshot_t.copy_(self.prev_actions_policy)
        # torch.cuda.synchronize()
        t03 = time.perf_counter()
        # 5) Compose obs (policy order), build stacked obs, run policy.
        cur_actor_obs = self._compose_actor_obs(
            joint_pos_policy=self._joint_pos_policy_t,
            joint_vel_policy=self._joint_vel_policy_t,
            targets_policy=self._targets_snapshot_t,
            feat_t=feat_t,
        )
        # torch.cuda.synchronize()
        t1 = time.perf_counter()

        if self.n_stack_frame == 1:
            self._stacked_obs_buf.copy_(cur_actor_obs)
        elif self.n_stack_frame == 2:
            self._stacked_obs_buf[:, : self.actor_obs_unstacked_space].copy_(cur_actor_obs)
            self._stacked_obs_buf[:, self.actor_obs_unstacked_space :].copy_(self.prev_actor_obs)
            self.prev_actor_obs.copy_(cur_actor_obs)
        else:
            raise ValueError(
                f"n_stack_frame={self.n_stack_frame} is not supported by depth_feat_policy."
            )
        obs_clamped = torch.clamp(self._stacked_obs_buf, -100.0, 100.0, out=self._obs_clamped_buf)

        with torch.inference_mode():
            # JIT-exported policy: forward(obs) = actor(normalizer(obs)). The
            # obs_normalizer was baked into the ScriptModule by
            # export_policy_as_jit, so we hand it the un-normalized obs
            # vector directly. Depth feature is already spliced into the
            # bottle slots inside _compose_actor_obs.
            actions_policy = self.policy(obs_clamped)
        
        # torch.cuda.synchronize()
        t2 = time.perf_counter()

        # 6) Integrate and publish (convert policy->real only at publish boundary).
        next_targets_policy, ema_actions_policy = self._compute_next_targets(
            actions_policy=actions_policy,
            prev_actions_policy=self._prev_actions_snapshot_t,
            joint_pos_policy=self._joint_pos_policy_t,
            prev_targets_policy=self._targets_snapshot_t,
        )
        self.targets_policy.copy_(next_targets_policy)
        self.prev_actions_policy.copy_(ema_actions_policy)

        next_targets_real = next_targets_policy[:, self.policy2real_idx]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.real_joint_names
        msg.position = next_targets_real[0].detach().cpu().tolist()
        self.target_pub.publish(msg)
        # torch.cuda.synchronize()
        t_end = time.perf_counter()
        self._log_policy_timing(
            feat_copy_s=t01 - t0,
            js_lock_s=t02 - t01,
            prev_var_s=t03 - t02,
            fk_compose_obs_s=t1 - t03,
            policy_inference_s=t2 - t1,
            policy_inference_sync_s=t_end - t2,
            total_s=t_end - t0,
        )

    def _build_node_cfg_summary(self) -> dict[str, Any]:
        """Build a deterministic runtime summary for the final ready log."""
        return {
            "paths": {
                "policy_log_dir": str(self.policy_log_dir),
                "jit_policy_path": str(self.jit_policy_path),
                "runtime_cfg_path": str(self.runtime_cfg_path),
                "agent_cfg_path": str(self.agent_cfg_path),
                "env_cfg_path": str(self.env_cfg_path),
                "hand_env_cfg_path": str(self.hand_env_cfg_path),
                "driver_cfg_path": str(self.driver_cfg_path),
                "policy_node_cfg_path": str(self.policy_node_cfg_path),
                "fk_cfg_path": str(self.fk_cfg_path),
                "da3_cfg_path": str(self.da3_cfg_path),
                "apriltag_cfg_path": str(self.apriltag_cfg_path),
            },
            "rates_hz": {
                "policy_hz": float(self.policy_hz),
            },
            "apriltag_subs": {
                "poses_topic": self._apriltag_poses_topic,
                "geom_topic":  self._apriltag_geom_topic,
                "has_pose":    bool(self.has_apriltag_pose),
                "has_geom":    bool(self.has_apriltag_geom),
            },
            "runtime": {
                "dt": float(self.dt),
                "action_ema": float(self.action_ema),
                "action_scale": float(self.action_scale),
            },
            "obs": {
                "n_stack_frame": int(self.n_stack_frame),
                "actor_obs_unstacked_space": int(self.actor_obs_unstacked_space),
                "critic_obs_unstacked_space": int(self.critic_obs_unstacked_space),
                "num_actor_obs": int(self.num_actor_obs),
                "num_critic_obs": int(self.num_critic_obs),
                "actor_obs_keys": list(self.actor_obs_keys),
            },
            "policy": {
                "source": "torch.jit.load",
                "jit_path": str(self.jit_policy_path),
                "action_num": int(self.action_num),
                "note": "actor + obs_normalizer baked together via export_policy_as_jit",
            },
            "bottle_geom_bounds": {
                "lower": list(_BOTTLE_GEOM_LOWER),
                "upper": list(_BOTTLE_GEOM_UPPER),
                "note": "must match observations.py:147-151",
            },
        }

    def destroy_node(self) -> bool:
        """Cancel the policy timer before destroying the ROS node.

        RealSense / FK / DepthFeatureNetFiLM all live in their own nodes
        (cam_node / fk_node / depth_feat_node — the last is TODO);
        nothing to clean up here besides the timer.
        """
        if hasattr(self, "policy_timer") and self.policy_timer is not None:
            self.policy_timer.cancel()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ApriltagPolicyNode()
    # executor = MultiThreadedExecutor(num_threads=3)
    executor = SingleThreadedExecutor()
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
