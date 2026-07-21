"""Policy node that consumes a pre-computed depth feature vector via CUDA IPC.

Refactored for mjlab-trained policies. The obs layout follows mjlab's per-term
history convention: proprio_noised (112d x history=2 = 224d) concatenated with
extero_noised (10d x history=1 = 10d) = 234d total actor obs.

Key differences from the HAND/IsaacLab version:
  - No joint velocity in actor obs (critic-only in mjlab)
  - No bottleBodyRot6D in actor obs
  - jar_geom normalized to [-1,1] (not repeated 3x)
  - Targets are RAW positions (not scaled)
  - History is per-term [oldest, newest], not global frame stacking
  - Config comes from mjlab_policy_runtime.yaml, not runtime_cfg_play.yaml
"""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any

import base64
import pickle
import re
import numpy as np
import rclpy
import torch
import torch.multiprocessing  # noqa: F401  # registers CUDA IPC reducers in ForkingPickler
import yaml
from geometry_msgs.msg import PoseArray
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
from robot_motion_interface.utils.sim2real import joint_mapping
from robot_motion_interface.utils.mjlab_yaml import load_mjlab_yaml


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
    """Clip then min-max scale [lower, upper] -> [-1, 1] elementwise."""
    clipped = value.clamp(min=lower, max=upper)
    return 2.0 * (clipped - lower) / (upper - lower) - 1.0


def _quat_wxyz_to_6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Quaternion (..., 4) wxyz -> 6D rotation (..., 6).

    First two columns of the rotation matrix, flattened (Zhou et al., 2019).
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


# ---------------------------------------------------------------------------
# DepthFeatPolicyNode
# ---------------------------------------------------------------------------
class DepthFeatPolicyNode(Node):
    def __init__(self):
        super().__init__("depth_feat_policy_node")

        # -- 1. params & cfgs --
        self._init_callback_mutex_groups()
        self._declare_parameters()
        self._load_configs()
        self._init_device()

        # -- 2. locks --
        self._init_threading_locks()

        # -- 3. components --
        self._build_joint_mappings()
        self._read_hand_link_counts()
        self._build_policy_and_normalizer()
        self._init_state_buffers()
        # BYPASS perception (depth_sam_feat_node off): hardcode the jar-pose feature
        # instead of fetching the CUDA-IPC handle. Uncomment the call below to restore.
        # self._fetch_depth_feature_handle()
        _hardcoded_jar_feat = torch.tensor(
            [[0.0, 0.0, 1.0169, 0.0, 0.0, 1.1169, 0.04, 0.16, 0.035, 0.025]],    # green
            # [[0.0, 0.0, 1.0169, 0.0, 0.0, 1.1069, 0.04, 0.16, 0.035, 0.02]],    # printed
            dtype=torch.float32,
            device=self.device,
        )  # body_pos(3)+cap_pos(3)+jar_geom(4), metres, env-local frame (= sim extero)
        self.latest_depth_feature = _hardcoded_jar_feat
        self._feat_snapshot_t = torch.empty_like(_hardcoded_jar_feat)

        # -- 4. subs & pubs --
        self._init_pub_sub()

        # -- 5. timer + summary --
        self._init_policy_timer()
        self.node_cfg = self._build_node_cfg_summary()
        self.get_logger().info("DepthFeatPolicyNode ready:\n" + yaml.safe_dump(self.node_cfg, sort_keys=False))

        self.get_logger().info("DepthFeatPolicyNode initialization complete. Waiting for 2 seconds before spinning node.")
        time.sleep(2.0)

    # ------------------------------------------------------------------
    # Step 1: params & cfgs
    # ------------------------------------------------------------------
    def _init_callback_mutex_groups(self) -> None:
        self.js_grp = MutuallyExclusiveCallbackGroup()
        self.fk_grp = MutuallyExclusiveCallbackGroup()
        self.infer_grp = MutuallyExclusiveCallbackGroup()

    def _declare_parameters(self) -> None:
        self.declare_parameter("policy_log_dir", str((RMI_ROOT / "runtime" / "policy").resolve()))
        self.declare_parameter("mjlab_runtime_cfg_path", str((RMI_ROOT / "config" / "mjlab_policy_runtime.yaml").resolve()))
        self.declare_parameter("driver_cfg_path", str((RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml").resolve()))
        self.declare_parameter("fk_cfg_path", str((RMI_ROOT / "config" / "fk_config.yaml").resolve()))
        self.declare_parameter(
            "depth_feat_cfg_path",
            str((RMI_ROOT / "config" / "depth_feat_config.yaml").resolve()),
        )

        self.policy_log_dir = Path(self.get_parameter("policy_log_dir").value)
        self.jit_policy_path = self.policy_log_dir / "exported" / "policy.pt"

        self.mjlab_runtime_cfg_path = Path(self.get_parameter("mjlab_runtime_cfg_path").value)
        self.driver_cfg_path = Path(self.get_parameter("driver_cfg_path").value)
        self.fk_cfg_path = Path(self.get_parameter("fk_cfg_path").value)
        self.depth_feat_cfg_path = Path(self.get_parameter("depth_feat_cfg_path").value)

    def _load_configs(self) -> None:
        self.mjlab_rt = _load_yaml(self.mjlab_runtime_cfg_path)
        self.driver_cfg = _load_yaml(self.driver_cfg_path)
        self.fk_cfg = _load_yaml(self.fk_cfg_path)
        self.depth_feat_full_cfg = _load_yaml(self.depth_feat_cfg_path)
        # Deployed policy's training env.yaml -> vel_scale etc. (auto-synced with the
        # exported policy). See utils/mjlab_yaml: tag-ignoring loader (mjlab not importable).
        self._env_cfg_path = self.policy_log_dir / "params" / "env.yaml"
        self.env_cfg = load_mjlab_yaml(self._env_cfg_path)

        self.get_logger().info("#### Depth-feat policy node configs (mjlab): ####")
        self.get_logger().info(f"policy_log_dir:          {self.policy_log_dir}")
        self.get_logger().info(f"  jit_policy_path:       {self.jit_policy_path}")
        self.get_logger().info(f"mjlab_runtime_cfg_path:  {self.mjlab_runtime_cfg_path}")
        self.get_logger().info(f"driver_cfg_path:         {self.driver_cfg_path}")
        self.get_logger().info(f"fk_cfg_path:             {self.fk_cfg_path}")
        self.get_logger().info(f"depth_feat_cfg_path:     {self.depth_feat_cfg_path}")
        self.get_logger().info(f"env_cfg_path:            {self._env_cfg_path}")

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

    # ------------------------------------------------------------------
    # Step 3: components
    # ------------------------------------------------------------------
    def _to_dev_t(self, data: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, device=self.device)

    def _build_joint_mappings(self) -> None:
        # Real driver order: left panda+tesollo, then right panda+tesollo.
        left_panda = self.driver_cfg["left_panda_joint_names"]
        left_tesollo = self.driver_cfg["left_tesollo_joint_names"]
        right_panda = self.driver_cfg["right_panda_joint_names"]
        right_tesollo = self.driver_cfg["right_tesollo_joint_names"]

        left_real = ["left_" + n for n in left_panda + left_tesollo]
        right_real = ["right_" + n for n in right_panda + right_tesollo]
        self.real_joint_names = left_real + right_real
        self.action_num = len(self.real_joint_names)

        # Policy order = driver's all_joint_names (same 38-d bimanual ordering).
        policy_joint_names = list(self.driver_cfg["all_joint_names"])
        policy2real_idx, real2policy_idx = joint_mapping(policy_joint_names, self.real_joint_names)
        self.policy2real_idx = self._to_dev_t(policy2real_idx, dtype=torch.long)
        self.real2policy_idx = self._to_dev_t(real2policy_idx, dtype=torch.long)

        # Per-arm indices: first 19 joints are left (7 panda + 12 tesollo),
        # next 19 are right.
        n_per_arm = len(left_panda) + len(left_tesollo)
        self.left_policy_indices = self._to_dev_t(list(range(n_per_arm)), dtype=torch.long)
        self.right_policy_indices = self._to_dev_t(list(range(n_per_arm, 2 * n_per_arm)), dtype=torch.long)

        # Vel scale: arm joints get arm_vel_scale, finger joints get finger_vel_scale.
        # Read from the deployed policy's env.yaml (auto-synced with training), NOT
        # mjlab_policy_runtime.yaml which is hand-maintained and drifts stale.
        _act = self.env_cfg["actions"]["left"]  # left/right share these integration params
        arm_vel = float(_act["arm_vel_scale"])
        finger_vel = float(_act["finger_vel_scale"])
        arm_expr = self.mjlab_rt["arm_joint_expr"]
        finger_expr = self.mjlab_rt["finger_joint_expr"]

        def _vel_scale_for(unprefixed_names: list[str]) -> list[float]:
            scales = []
            for n in unprefixed_names:
                if re.fullmatch(arm_expr, n):
                    scales.append(arm_vel)
                elif re.fullmatch(finger_expr, n):
                    scales.append(finger_vel)
                else:
                    raise ValueError(f"Joint '{n}' matches neither arm_joint_expr nor finger_joint_expr")
            return scales

        unprefixed_left = left_panda + left_tesollo
        unprefixed_right = right_panda + right_tesollo
        vel_scale_left = self._to_dev_t(_vel_scale_for(unprefixed_left))
        vel_scale_right = self._to_dev_t(_vel_scale_for(unprefixed_right))

        # Soft limits (same for both arms, from mjlab_runtime_cfg).
        soft_lower = self._to_dev_t(self.mjlab_rt["soft_lower"])
        soft_upper = self._to_dev_t(self.mjlab_rt["soft_upper"])

        # Scatter per-arm slices into full bimanual policy-order vectors.
        def _scatter_full(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            full = torch.zeros(self.action_num, dtype=torch.float32, device=self.device)
            full[self.left_policy_indices] = left
            full[self.right_policy_indices] = right
            return full

        self.full_scale_policy = _scatter_full(vel_scale_left, vel_scale_right).unsqueeze(0)
        self.full_lower_policy = _scatter_full(soft_lower, soft_lower).unsqueeze(0)
        self.full_upper_policy = _scatter_full(soft_upper, soft_upper).unsqueeze(0)

        # Per-arm soft limits kept for obs scaling.
        self.left_soft_lower = soft_lower.unsqueeze(0)
        self.left_soft_upper = soft_upper.unsqueeze(0)
        self.right_soft_lower = soft_lower.unsqueeze(0)
        self.right_soft_upper = soft_upper.unsqueeze(0)

        # Frozen-joint mask per arm.
        left_frozen_expr = self.mjlab_rt.get("left_frozen_joint_expr")
        right_frozen_expr = self.mjlab_rt.get("right_frozen_joint_expr")
        frozen = torch.zeros(self.action_num, dtype=torch.bool, device=self.device)

        if left_frozen_expr:
            for i, name in enumerate(unprefixed_left):
                if re.fullmatch(left_frozen_expr, name):
                    frozen[i] = True
        if right_frozen_expr:
            for i, name in enumerate(unprefixed_right):
                if re.fullmatch(right_frozen_expr, name):
                    frozen[n_per_arm + i] = True

        if frozen.any():
            frozen_names = [n for n, f in zip(policy_joint_names, frozen) if f]
            self.get_logger().info(f"Frozen joints: {frozen_names}")
        self._frozen_mask = frozen

    def _read_hand_link_counts(self) -> None:
        """Resolve FK link names from mjlab_runtime_cfg."""
        finger_tip_links = list(self.mjlab_rt["finger_tip_links"])
        hand_palm_links = list(self.mjlab_rt["hand_palm_links"])

        self.left_fingertip_names = [f"left_{n}" for n in finger_tip_links]
        self.right_fingertip_names = [f"right_{n}" for n in finger_tip_links]
        self.left_hand_base_names = [f"left_{n}" for n in hand_palm_links]
        self.right_hand_base_names = [f"right_{n}" for n in hand_palm_links]

        self.fk_link_names = list(self.fk_cfg["link_names"])
        self.fk_n_links = len(self.fk_link_names)
        self.fk_topic = str(self.fk_cfg["fk_topic"])

    def _build_policy_and_normalizer(self) -> None:
        """Load obs dims from mjlab_runtime_cfg, cache scaling tensors, load JIT policy."""
        # Jar geom bounds from runtime config.
        self._geom_lower_t = self._to_dev_t(self.mjlab_rt["jar_geom_lo"]).unsqueeze(0)
        self._geom_upper_t = self._to_dev_t(self.mjlab_rt["jar_geom_hi"]).unsqueeze(0)

        # Obs dims.
        self.obs_dim = int(self.mjlab_rt["actor_obs_dim"])
        self.proprio_history = int(self.mjlab_rt["proprio_history"])

        if not self.jit_policy_path.exists():
            raise FileNotFoundError(
                f"JIT policy not found at {self.jit_policy_path}. "
                f"Re-export or point --policy_log_dir at the correct run."
            )
        self.policy = torch.jit.load(
            str(self.jit_policy_path), map_location=self.device
        ).to(self.device)
        self.policy.eval()
        self.get_logger().info(
            f"Loaded JIT policy: {self.jit_policy_path}  "
            f"(actor_obs_dim={self.obs_dim}, num_actions={self.action_num})"
        )

    def _init_state_buffers(self) -> None:
        A = self.action_num

        # /joint_states latest snapshot (CPU, written by sub callback).
        self.latest_joint_pos_real_np: np.ndarray = np.zeros((A,), dtype=np.float32)
        self.latest_joint_vel_real_np: np.ndarray = np.zeros((A,), dtype=np.float32)
        self.has_joint_state: bool = False

        # Per-step CPU snapshots copied under joint_lock.
        self._joint_pos_snapshot_np: np.ndarray = np.zeros((A,), dtype=np.float32)
        self._joint_vel_snapshot_np: np.ndarray = np.zeros((A,), dtype=np.float32)

        # GPU mirrors (policy order).
        self._joint_pos_policy_t: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        self._joint_vel_policy_t: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        self._targets_snapshot_t: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        self._prev_actions_snapshot_t: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)

        # Persistent policy state (policy order).
        self.targets_policy: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        self.targets_initialized: bool = False
        self.prev_actions_policy: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)
        self._frozen_default_targets: torch.Tensor = torch.zeros((1, A), dtype=torch.float32, device=self.device)

        # Per-term proprio history ring buffer (mjlab CircularBuffer semantics):
        # holds the last N frames (N = proprio_history), oldest at [0], newest at [-1].
        # 112d per frame = 19+19+19+19+9+9+3+6+3+6 per side. On the first frame the
        # whole buffer is back-filled with that frame (NOT zeros), matching mjlab.
        self._PROPRIO_DIM = 112
        self._proprio_hist: torch.Tensor = torch.zeros(
            (self.proprio_history, 1, self._PROPRIO_DIM), dtype=torch.float32, device=self.device
        )
        self._proprio_initialized: bool = False

        # Working buffers.
        self._obs_buf: torch.Tensor = torch.zeros((1, self.obs_dim), dtype=torch.float32, device=self.device)
        self._obs_clamped_buf: torch.Tensor = torch.zeros((1, self.obs_dim), dtype=torch.float32, device=self.device)

        # FK results dict.
        self.fk_pose_dict: dict[str, np.ndarray] = {}
        for name in self.fk_link_names:
            self.fk_pose_dict[name + "_pos"] = np.zeros(3, dtype=np.float32)
            self.fk_pose_dict[name + "_wxyz"] = np.zeros(4, dtype=np.float32)
        self.has_fk: bool = False

        # Depth feature IPC tensor.
        self.latest_depth_feature: torch.Tensor | None = None

        # Recording buffer (auto-saves after N steps).
        self._rec_max_steps = 800
        self._rec_step = 0
        self._rec_actor_obs = torch.zeros((self._rec_max_steps, self.obs_dim), dtype=torch.float32, device=self.device)
        self._rec_actions = torch.zeros((self._rec_max_steps, A), dtype=torch.float32, device=self.device)
        self._rec_targets = torch.zeros((self._rec_max_steps, A), dtype=torch.float32, device=self.device)
        self._rec_joint_pos = torch.zeros((self._rec_max_steps, A), dtype=torch.float32, device=self.device)
        self._rec_feat = torch.zeros((self._rec_max_steps, 10), dtype=torch.float32, device=self.device)
        self._rec_saved = False

    def _fetch_depth_feature_handle(self) -> None:
        """One-shot: call depth_feat_node's service, decode CUDA IPC handle,
        attach self.latest_depth_feature to the persistent feature buffer."""
        service_name = str(self.depth_feat_full_cfg["ipc"]["handle_service_name"])

        client = self.create_client(Trigger, service_name)
        self.get_logger().info(f"Waiting for depth_feat_node service: {service_name}")
        while not client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                raise RuntimeError(
                    "rclpy shutting down while waiting for depth_feat_node service"
                )
            self.get_logger().info(f"...still waiting for {service_name}")

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is None or not resp.success:
            raise RuntimeError(f"depth_feat_node service call failed: {resp}")

        payload = base64.b64decode(resp.message.encode("ascii"))
        feature_view = pickle.loads(payload)
        if not isinstance(feature_view, torch.Tensor):
            raise TypeError(
                f"IPC payload did not unpickle to torch.Tensor: {type(feature_view)}"
            )
        if feature_view.device.type != "cuda":
            raise RuntimeError(
                f"IPC depth-feature tensor is not on CUDA: {feature_view.device}"
            )
        expected_dim = int(self.depth_feat_full_cfg["depth_feature"]["output_dim"])
        if (
            feature_view.ndim != 2
            or feature_view.shape[0] != 1
            or int(feature_view.shape[1]) != expected_dim
        ):
            raise RuntimeError(
                f"Unexpected feature tensor shape: {tuple(feature_view.shape)}; "
                f"expected (1, {expected_dim})"
            )

        self.latest_depth_feature = feature_view
        self._feat_snapshot_t = torch.empty_like(feature_view)
        self.get_logger().info(
            f"Depth-feature IPC handle received: shape={tuple(feature_view.shape)} "
            f"dtype={feature_view.dtype} device={feature_view.device}"
        )

    # ------------------------------------------------------------------
    # Step 4: subs & pubs
    # ------------------------------------------------------------------
    def _init_pub_sub(self) -> None:
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
        self.create_subscription(
            PoseArray,
            self.fk_topic,
            self._sub_fk_cb,
            HIGH_PERF_QOS,
            callback_group=self.fk_grp,
        )

    def _sub_fk_cb(self, msg: PoseArray) -> None:
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
        if len(msg.position) != self.action_num:
            raise RuntimeError(
                f"/joint_states DoF mismatch: msg.position has "
                f"{len(msg.position)} entries, driver_cfg expects "
                f"{self.action_num}."
            )
        pos_np = np.asarray(msg.position, dtype=np.float32)
        vel_np = np.asarray(msg.velocity, dtype=np.float32)
        with self.joint_lock:
            np.copyto(self.latest_joint_pos_real_np, pos_np)
            np.copyto(self.latest_joint_vel_real_np, vel_np)
            self.has_joint_state = True

    # ------------------------------------------------------------------
    # Step 5: timer + summary
    # ------------------------------------------------------------------
    def _init_policy_timer(self) -> None:
        self.policy_hz = float(self.mjlab_rt["infer_rate"])
        if self.policy_hz <= 0.0:
            raise ValueError(f"infer_rate must be > 0, got {self.policy_hz}")

        self.dt = float(self.mjlab_rt["dt"])
        self.action_ema = float(self.mjlab_rt["action_ema"])
        self.action_scale = float(self.mjlab_rt["action_scale"])
        self.target_alpha = float(self.mjlab_rt["target_alpha"])

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
        joint_pos_policy: torch.Tensor,  # [1, A]
        targets_policy: torch.Tensor,    # [1, A]
        feat_t: torch.Tensor,            # [1, 10] body(3)+cap(3)+geom(4) in metres
    ) -> torch.Tensor:
        """Build the actor obs in mjlab's per-term history layout.

        proprio_noised (112 * proprio_history) = 10 terms, each emitted as its
        history_length (=proprio_history) frames oldest->newest, term-major:
          left_joint_pos(19) | right_joint_pos(19) |
          left_targets(19)   | right_targets(19)   |
          left_fingertips(9) | right_fingertips(9) |
          left_palm_pos(3)   | left_palm_rot6d(6)  |
          right_palm_pos(3)  | right_palm_rot6d(6)
          = 112 per frame, x proprio_history frames.

        extero_noised (10d, no history):
          bottle_pos(3) | cap_pos(3) | jar_geom(4)

        Total actor obs = 112 * proprio_history + 10, checked against obs_dim.
        """
        left_jp = joint_pos_policy[:, self.left_policy_indices]
        right_jp = joint_pos_policy[:, self.right_policy_indices]
        left_tgt = targets_policy[:, self.left_policy_indices]
        right_tgt = targets_policy[:, self.right_policy_indices]

        # Scale joint positions to [-1, 1] via soft limits.
        left_jp_s = _scale_torch(left_jp, self.left_soft_lower, self.left_soft_upper)
        right_jp_s = _scale_torch(right_jp, self.right_soft_lower, self.right_soft_upper)

        # FK positions + palm quats.
        with self.fk_lock:
            l_ft_pos_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.left_fingertip_names])
            r_ft_pos_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.right_fingertip_names])
            l_palm_pos_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.left_hand_base_names])
            r_palm_pos_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.right_hand_base_names])
            l_palm_wxyz_np = np.concatenate([self.fk_pose_dict[n + "_wxyz"] for n in self.left_hand_base_names])
            r_palm_wxyz_np = np.concatenate([self.fk_pose_dict[n + "_wxyz"] for n in self.right_hand_base_names])

        l_ft_pos = torch.from_numpy(l_ft_pos_np).unsqueeze(0).to(self.device, non_blocking=True)    # (1, 9)
        r_ft_pos = torch.from_numpy(r_ft_pos_np).unsqueeze(0).to(self.device, non_blocking=True)    # (1, 9)
        l_palm_pos = torch.from_numpy(l_palm_pos_np).unsqueeze(0).to(self.device, non_blocking=True) # (1, 3)
        r_palm_pos = torch.from_numpy(r_palm_pos_np).unsqueeze(0).to(self.device, non_blocking=True) # (1, 3)

        l_palm_wxyz = torch.from_numpy(l_palm_wxyz_np).unsqueeze(0).to(self.device, non_blocking=True)
        r_palm_wxyz = torch.from_numpy(r_palm_wxyz_np).unsqueeze(0).to(self.device, non_blocking=True)
        l_palm_rot6d = _quat_wxyz_to_6d(l_palm_wxyz)  # (1, 6)
        r_palm_rot6d = _quat_wxyz_to_6d(r_palm_wxyz)  # (1, 6)

        # Current proprio frame (112d): the 10 terms concatenated.
        cur_proprio = torch.cat([
            left_jp_s,      # 19
            right_jp_s,     # 19
            left_tgt,       # 19  (RAW targets, not scaled)
            right_tgt,      # 19
            l_ft_pos,       # 9
            r_ft_pos,       # 9
            l_palm_pos,     # 3
            l_palm_rot6d,   # 6
            r_palm_pos,     # 3
            r_palm_rot6d,   # 6
        ], dim=-1)          # total = 112

        if int(cur_proprio.shape[-1]) != self._PROPRIO_DIM:
            raise ValueError(
                f"proprio dim={int(cur_proprio.shape[-1])} != expected {self._PROPRIO_DIM}"
            )

        # Update the ring buffer (mjlab CircularBuffer semantics). On the first call,
        # back-fill ALL N frames with the current frame (not zeros); afterwards, roll and
        # write the newest frame to [-1] (oldest at [0], newest at [-1]).
        if not self._proprio_initialized:
            self._proprio_hist[:] = cur_proprio
            self._proprio_initialized = True
        else:
            self._proprio_hist = torch.roll(self._proprio_hist, -1, dims=0)
            self._proprio_hist[-1] = cur_proprio

        # Per-term history: for each term emit its N frames oldest->newest, term-major
        # (mjlab flattens each term's history separately, then concatenates terms).
        # Term boundaries within the 112d proprio vector:
        #   left_jp(19), right_jp(19), left_tgt(19), right_tgt(19),
        #   left_ft(9), right_ft(9), left_palm_pos(3), left_palm_rot6d(6),
        #   right_palm_pos(3), right_palm_rot6d(6)
        term_sizes = [19, 19, 19, 19, 9, 9, 3, 6, 3, 6]
        proprio_parts = []
        offset = 0
        for sz in term_sizes:
            for f in range(self.proprio_history):  # oldest -> newest
                proprio_parts.append(self._proprio_hist[f, :, offset:offset + sz])
            offset += sz
        proprio_obs = torch.cat(proprio_parts, dim=-1)  # 112 * proprio_history

        # Extero (10d, no history): body_pos(3) + cap_pos(3) + jar_geom(4).
        body_pos_m = feat_t[:, 0:3]
        cap_pos_m = feat_t[:, 3:6]
        geom_raw = feat_t[:, 6:10]
        # body_pos_m = torch.tensor([[0.0, 0.0, 1.0169]], device=self.device)       # jar body
        # cap_pos_m  = torch.tensor([[0.0, 0.0, 1.1069]], device=self.device)       # jar cap
        # geom_raw   = torch.tensor([[0.04, 0.16, 0.029, 0.02]], device=self.device)  # meter
        jar_geom_scaled = _scale_torch(geom_raw, self._geom_lower_t, self._geom_upper_t)  # (1, 4)

        extero_obs = torch.cat([body_pos_m, cap_pos_m, jar_geom_scaled], dim=-1)  # 10d

        actor_obs = torch.cat([proprio_obs, extero_obs], dim=-1)  # 234d

        if int(actor_obs.shape[-1]) != self.obs_dim:
            raise ValueError(
                f"composed actor obs dim={int(actor_obs.shape[-1])} "
                f"!= actor_obs_dim={self.obs_dim}"
            )
        return actor_obs

    def _compute_next_targets(
        self,
        actions_policy: torch.Tensor,
        prev_actions_policy: torch.Tensor,
        joint_pos_policy: torch.Tensor,
        prev_targets_policy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EMA-smooth actions and integrate to next joint targets (policy order), clamped to soft limits."""
        ema_actions = (
            actions_policy.clamp(-1.0, 1.0) * self.action_ema
            + prev_actions_policy * (1.0 - self.action_ema)
        )
        alpha = self.target_alpha
        next_targets_policy = torch.clamp(
            joint_pos_policy * (1.0 - alpha) + prev_targets_policy * alpha + ema_actions * self.dt * self.full_scale_policy * self.action_scale,
            min=self.full_lower_policy,
            max=self.full_upper_policy,
        )
        if self._frozen_mask.any():
            next_targets_policy[:, self._frozen_mask] = self._frozen_default_targets[:, self._frozen_mask]
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

        # 1) Snapshot depth-feature IPC tensor.
        if self.latest_depth_feature is None:
            self.get_logger().error("Depth-feature IPC handle not yet attached.")
            return
        self._feat_snapshot_t.copy_(self.latest_depth_feature, non_blocking=True)
        feat_t = self._feat_snapshot_t
        t01 = time.perf_counter()
        if feat_t.ndim != 2 or feat_t.shape[0] != 1:
            self.get_logger().error(
                f"Unexpected cached feature shape: {tuple(feat_t.shape)}; "
                f"expected [1, feature_dim]"
            )
            rclpy.shutdown()
            return

        # 2) Snapshot latest joint arrays (real order) under joint_lock.
        with self.joint_lock:
            if not self.has_joint_state:
                return
            np.copyto(self._joint_pos_snapshot_np, self.latest_joint_pos_real_np)
            np.copyto(self._joint_vel_snapshot_np, self.latest_joint_vel_real_np)
        t02 = time.perf_counter()

        # 3) Wait for FK.
        if not self.has_fk:
            return

        # 4) CPU -> GPU, real -> policy order.
        joint_pos_real_t = torch.from_numpy(self._joint_pos_snapshot_np).to(self.device, non_blocking=True)
        joint_vel_real_t = torch.from_numpy(self._joint_vel_snapshot_np).to(self.device, non_blocking=True)
        self._joint_pos_policy_t[0, :].copy_(joint_pos_real_t[self.real2policy_idx])
        self._joint_vel_policy_t[0, :].copy_(joint_vel_real_t[self.real2policy_idx])

        if not self.targets_initialized:
            self.targets_policy.copy_(self._joint_pos_policy_t)
            self._frozen_default_targets.copy_(self._joint_pos_policy_t)
            self.targets_initialized = True
        self._targets_snapshot_t.copy_(self.targets_policy)
        self._prev_actions_snapshot_t.copy_(self.prev_actions_policy)
        t03 = time.perf_counter()

        # 5) Compose 234d obs, run policy.
        actor_obs = self._compose_actor_obs(
            joint_pos_policy=self._joint_pos_policy_t,
            targets_policy=self._targets_snapshot_t,
            feat_t=feat_t,
        )
        t1 = time.perf_counter()

        obs_clamped = torch.clamp(actor_obs, -100.0, 100.0, out=self._obs_clamped_buf)

        with torch.inference_mode():
            actions_policy = self.policy(obs_clamped)
        t2 = time.perf_counter()

        # 6) Integrate and publish.
        next_targets_policy, ema_actions_policy = self._compute_next_targets(
            actions_policy=actions_policy,
            prev_actions_policy=self._prev_actions_snapshot_t,
            joint_pos_policy=self._joint_pos_policy_t,
            prev_targets_policy=self._targets_snapshot_t,
        )
        self.targets_policy.copy_(next_targets_policy)
        self.prev_actions_policy.copy_(ema_actions_policy)

        # Record step.
        if self._rec_step < self._rec_max_steps:
            i = self._rec_step
            self._rec_actor_obs[i].copy_(actor_obs[0])
            self._rec_actions[i].copy_(actions_policy[0])
            self._rec_targets[i].copy_(next_targets_policy[0])
            self._rec_joint_pos[i].copy_(self._joint_pos_policy_t[0])
            self._rec_feat[i].copy_(feat_t[0])
            self._rec_step += 1
            if self._rec_step == self._rec_max_steps and not self._rec_saved:
                self._save_recording()

        next_targets_real = next_targets_policy[:, self.policy2real_idx]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.real_joint_names
        msg.position = next_targets_real[0].detach().cpu().tolist()
        self.target_pub.publish(msg)
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

    def _save_recording(self) -> None:
        out = self.policy_log_dir / "policy_recording.npz"
        np.savez_compressed(
            str(out),
            actor_obs=self._rec_actor_obs.cpu().numpy(),
            actions=self._rec_actions.cpu().numpy(),
            targets=self._rec_targets.cpu().numpy(),
            joint_pos=self._rec_joint_pos.cpu().numpy(),
            feat=self._rec_feat.cpu().numpy(),
            joint_names=np.array(self.real_joint_names),
        )
        self._rec_saved = True
        self.get_logger().info(f"Recording saved: {out} ({self._rec_max_steps} steps)")

    def _build_node_cfg_summary(self) -> dict[str, Any]:
        return {
            "paths": {
                "policy_log_dir": str(self.policy_log_dir),
                "jit_policy_path": str(self.jit_policy_path),
                "mjlab_runtime_cfg_path": str(self.mjlab_runtime_cfg_path),
                "driver_cfg_path": str(self.driver_cfg_path),
                "fk_cfg_path": str(self.fk_cfg_path),
                "depth_feat_cfg_path": str(self.depth_feat_cfg_path),
            },
            "rates_hz": {
                "policy_hz": float(self.policy_hz),
            },
            "depth_feature_ipc": {
                "shape": list(self.latest_depth_feature.shape) if self.latest_depth_feature is not None else None,
                "dtype": str(self.latest_depth_feature.dtype) if self.latest_depth_feature is not None else None,
                "service": str(self.depth_feat_full_cfg["ipc"]["handle_service_name"]),
            },
            "runtime": {
                "dt": float(self.dt),
                "action_ema": float(self.action_ema),
                "action_scale": float(self.action_scale),
                "target_alpha": float(self.target_alpha),
            },
            "obs": {
                "actor_obs_dim": int(self.obs_dim),
                "proprio_history": int(self.proprio_history),
                "proprio_dim_per_frame": self._PROPRIO_DIM,
            },
            "policy": {
                "source": "torch.jit.load",
                "jit_path": str(self.jit_policy_path),
                "action_num": int(self.action_num),
            },
            "jar_geom_bounds": {
                "lower": self.mjlab_rt["jar_geom_lo"],
                "upper": self.mjlab_rt["jar_geom_hi"],
            },
        }

    def destroy_node(self) -> bool:
        if hasattr(self, "policy_timer") and self.policy_timer is not None:
            self.policy_timer.cancel()
        if not self._rec_saved and self._rec_step > 0:
            self._rec_actor_obs = self._rec_actor_obs[:self._rec_step]
            self._rec_actions = self._rec_actions[:self._rec_step]
            self._rec_targets = self._rec_targets[:self._rec_step]
            self._rec_joint_pos = self._rec_joint_pos[:self._rec_step]
            self._rec_feat = self._rec_feat[:self._rec_step]
            self._save_recording()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthFeatPolicyNode()
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
