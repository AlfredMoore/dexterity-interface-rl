from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any

import base64
import io
import pickle
import numpy as np
import rclpy
import torch
import torch.multiprocessing  # noqa: F401  # registers CUDA IPC reducers in ForkingPickler
import yaml
from geometry_msgs.msg import PoseArray
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

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
    """Min-max scale [lower, upper] -> [-1, 1] elementwise."""
    return 2.0 * (value - lower) / (upper - lower) - 1.0

# ---------------------------------------------------------------------------
# AuxPolicyNode
# ---------------------------------------------------------------------------
class AuxPolicyNode(Node):
    def __init__(self):
        super().__init__("aux_policy_node")

        # -- 1. params & cfgs --
        self._init_callback_mutex_groups()
        self._declare_parameters()
        self._load_configs()
        self._init_device()

        # -- 2. locks --
        self._init_threading_locks()

        # -- 3. components --
        self._import_rsl_rl()
        self._build_joint_mappings()
        self._read_hand_link_counts()
        self._build_policy_and_normalizer()
        self._init_state_buffers()
        self._fetch_depth_handle()

        # -- 4. subs & pubs --
        self._init_pub_sub()

        # -- 5. timer + summary --
        self._init_policy_timer()
        self.node_cfg = self._build_node_cfg_summary()
        self.get_logger().info("AuxPolicyNode ready:\n" + yaml.safe_dump(self.node_cfg, sort_keys=False))
        
        self.get_logger().info("AuxPolicyNode initialization complete. Waiting for 2 seconds before spinning node.")
        time.sleep(2.0)

    # ------------------------------------------------------------------
    # Step 1: params & cfgs
    # ------------------------------------------------------------------
    def _init_callback_mutex_groups(self) -> None:
        self.obs_mutex_grp = MutuallyExclusiveCallbackGroup()
        self.infer_mutex_grp = MutuallyExclusiveCallbackGroup()

    def _declare_parameters(self) -> None:
        """Declare ROS parameters: paths + da3 model override only.

        All paths must be absolute. Frequencies / runtime knobs come from cfg files.
        Missing files will fail naturally at cfg-load time.
        """
        self.declare_parameter("checkpoint_path", str((RMI_ROOT / "runtime" / "model.pt").resolve()))
        self.declare_parameter("runtime_cfg_path", str((RMI_ROOT / "runtime" / "runtime_cfg.yaml").resolve()))
        self.declare_parameter("env_cfg_path", str((RMI_ROOT / "runtime" / "env.yaml").resolve()))
        self.declare_parameter("hand_env_cfg_path", str((RMI_ROOT / "runtime" / "HandEnv.yaml").resolve()))
        self.declare_parameter("agent_cfg_path", str((RMI_ROOT / "runtime" / "agent.yaml").resolve()))
        self.declare_parameter("driver_cfg_path", str((RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml").resolve()))
        self.declare_parameter("policy_node_cfg_path", str((RMI_ROOT / "config" / "rl_policy_node_config.yaml").resolve()))
        self.declare_parameter("fk_cfg_path", str((RMI_ROOT / "config" / "fk_config.yaml").resolve()))
        self.declare_parameter("da3_cfg_path", str((RMI_ROOT / "config" / "da3_compile_config.yaml").resolve()))
        self.declare_parameter("rsl_rl_root", str((RMI_ROOT / "dep" / "rsl_rl-HAND").resolve()))

        self.checkpoint_path = Path(self.get_parameter("checkpoint_path").value)
        self.runtime_cfg_path = Path(self.get_parameter("runtime_cfg_path").value)
        self.env_cfg_path = Path(self.get_parameter("env_cfg_path").value)
        self.hand_env_cfg_path = Path(self.get_parameter("hand_env_cfg_path").value)
        self.agent_cfg_path = Path(self.get_parameter("agent_cfg_path").value)
        self.driver_cfg_path = Path(self.get_parameter("driver_cfg_path").value)
        self.policy_node_cfg_path = Path(self.get_parameter("policy_node_cfg_path").value)
        self.fk_cfg_path = Path(self.get_parameter("fk_cfg_path").value)
        self.da3_cfg_path = Path(self.get_parameter("da3_cfg_path").value)
        self.rsl_rl_root = Path(self.get_parameter("rsl_rl_root").value)

    def _load_configs(self) -> None:
        """Load all YAML cfgs. Missing files raise FileNotFoundError naturally."""
        self.runtime_cfg = _load_yaml(self.runtime_cfg_path)
        self.env_cfg = _load_yaml(self.env_cfg_path)
        self.hand_env_cfg = _load_yaml(self.hand_env_cfg_path)
        self.agent_cfg = _load_yaml(self.agent_cfg_path)
        self.driver_cfg = _load_yaml(self.driver_cfg_path)
        self.policy_node_cfg = _load_yaml(self.policy_node_cfg_path)
        self.fk_cfg = _load_yaml(self.fk_cfg_path)
        self.da3_full_cfg = _load_yaml(self.da3_cfg_path)

        self.get_logger().info("#### Aux policy node configs: ####")
        self.get_logger().info(f"runtime_cfg_path:     {self.runtime_cfg_path}")
        self.get_logger().info(f"env_cfg_path:         {self.env_cfg_path}")
        self.get_logger().info(f"hand_env_cfg_path:    {self.hand_env_cfg_path}")
        self.get_logger().info(f"agent_cfg_path:       {self.agent_cfg_path}")
        self.get_logger().info(f"driver_cfg_path:      {self.driver_cfg_path}")
        self.get_logger().info(f"policy_node_cfg_path: {self.policy_node_cfg_path}")
        self.get_logger().info(f"fk_cfg_path:          {self.fk_cfg_path}")
        self.get_logger().info(f"checkpoint_path:      {self.checkpoint_path}")

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

    def _import_rsl_rl(self) -> None:
        try:
            from rsl_rl.modules.normalizer import EmpiricalNormalization
            from rsl_rl.modules.student_teacher_aux import StudentTeacherAux
        except Exception as exc:
            raise ImportError(
                f"Failed to import rsl_rl modules from {self.rsl_rl_root}"
            ) from exc
        self.EmpiricalNormalization = EmpiricalNormalization
        self.StudentTeacherAux = StudentTeacherAux

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
        policy_cfg = self.agent_cfg["policy"]

        # 1. Vision settings (agent_cfg).
        self.vision_backbone_dim = int(policy_cfg["vision_backbone_dim"])
        self.vision_target_dim = int(policy_cfg["vision_target_dim"])
        self.vision_input_height = int(policy_cfg["vision_input_height"])
        self.vision_input_width = int(policy_cfg["vision_input_width"])
        self.depth_clip_min = float(policy_cfg["depth_clip_min"])
        self.depth_clip_max = float(policy_cfg["depth_clip_max"])
        self.student_vision_modality = str(policy_cfg["student_vision_modality"])
        if self.student_vision_modality != "depth":
            raise ValueError(
                f"Aux policy supports only 'depth' modality; got '{self.student_vision_modality}'"
            )

        # 2. Observation / action dims (env_cfg is the single source of truth).
        self.n_stack_frame = int(self.env_cfg["n_stack_frame"])
        self.student_obs_keys = [k for k in self.env_cfg["student"] if k != "visionFeatures"]
        self.student_obs_unstacked_space = int(self.env_cfg["student_obs_unstacked_space"])
        self.obs_dim = int(self.env_cfg["observation_space"])
        self.state_dim = int(self.env_cfg["state_space"])

        # cfg-internal sanity: stacked obs dim must equal observation_space,
        # and obs_DOF summed over student keys must equal the unstacked dim.
        if self.n_stack_frame * self.student_obs_unstacked_space != self.obs_dim:
            raise ValueError(
                f"obs_dim mismatch: n_stack_frame({self.n_stack_frame}) * "
                f"student_obs_unstacked_space({self.student_obs_unstacked_space}) "
                f"!= observation_space({self.obs_dim})"
            )
        obs_dof = self.env_cfg["obs_DOF"]
        inferred_unstacked = sum(int(obs_dof[k]) for k in self.student_obs_keys)
        if inferred_unstacked != self.student_obs_unstacked_space:
            raise ValueError(
                f"sum(obs_DOF[student_obs_keys])={inferred_unstacked} "
                f"!= student_obs_unstacked_space={self.student_obs_unstacked_space}"
            )

        # 3. Build student-teacher model from cfg dims.
        self.policy = self.StudentTeacherAux(
            num_student_obs=self.obs_dim,
            num_teacher_obs=self.state_dim,
            num_actions=self.action_num,
            student_hidden_dims=policy_cfg["student_hidden_dims"],
            teacher_hidden_dims=policy_cfg["teacher_hidden_dims"],
            activation=policy_cfg["activation"],
            init_noise_std=float(policy_cfg["init_noise_std"]),
            vision_backbone_dim=self.vision_backbone_dim,
            vision_target_dim=self.vision_target_dim,
            student_vision_modality=self.student_vision_modality,
            vision_input_height=self.vision_input_height,
            vision_input_width=self.vision_input_width,
            depth_clip_min=self.depth_clip_min,
            depth_clip_max=self.depth_clip_max,
        ).to(self.device)
        self.policy.eval()

        # 4. Load checkpoint; strict=True enforces every weight shape match,
        # which validates obs_dim / state_dim / action_num against the ckpt.
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_sd = ckpt.get("model_state_dict", ckpt)
        self.policy.load_state_dict(model_sd, strict=True)

        # 5. Observation normalizer. agent_cfg dictates whether the policy was
        # trained with empirical normalization; if so, ckpt must carry its state.
        self.use_obs_norm = bool(self.agent_cfg["empirical_normalization"])
        if self.use_obs_norm:
            if "obs_norm_state_dict" not in ckpt:
                raise KeyError(
                    "agent_cfg.empirical_normalization=true but ckpt has no "
                    "'obs_norm_state_dict'."
                )
            self.obs_normalizer = self.EmpiricalNormalization(shape=self.obs_dim).to(self.device)
            self.obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
            self.obs_normalizer.eval()
        else:
            self.obs_normalizer = None
            raise ValueError(
                "Observation normalization disabled. Ensure this matches the training cfg."
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
        self.prev_student_obs: torch.Tensor = torch.zeros(
            (1, self.student_obs_unstacked_space), dtype=torch.float32, device=self.device
        )

        # Stacked-obs working buffers (reused every step).
        self._stacked_obs_buf: torch.Tensor = torch.zeros((1, self.obs_dim), dtype=torch.float32, device=self.device)
        self._obs_clamped_buf: torch.Tensor = torch.zeros((1, self.obs_dim), dtype=torch.float32, device=self.device)

        # FK results: a single dict keyed by "<linkname>_pos" (xyz, float32[3]) and
        # "<linkname>_wxyz" (float32[4]). Pre-allocated in fk_cfg.link_names order;
        # the fk_topic subscriber np.copyto's into these arrays under fk_lock.
        self.fk_pose_dict: dict[str, np.ndarray] = {}
        for name in self.fk_link_names:
            self.fk_pose_dict[name + "_pos"] = np.zeros(3, dtype=np.float32)
            self.fk_pose_dict[name + "_wxyz"] = np.zeros(4, dtype=np.float32)
        self.has_fk: bool = False

        # Depth tensor: a CUDA IPC view onto cam_node's persistent depth buffer.
        # Filled by _fetch_depth_handle(); read directly by the policy loop with
        # no lock — tearing is rare and the cost of a torn read is negligible.
        self.latest_depth: torch.Tensor | None = None

    def _fetch_depth_handle(self) -> None:
        """One-shot: call cam_node's service, decode the CUDA IPC handle, and
        attach self.latest_depth to cam_node's persistent depth buffer.

        Blocks until cam_node advertises the service; afterwards reads of
        self.latest_depth are direct GPU reads of the IPC-shared tensor — no
        lock, no clone. Cam_node's DA3 timer keeps writing into that same
        memory via copy_, so latest_depth always reflects the most recent
        completed inference. Torn reads are possible but rare and tolerated.
        """
        ipc_cfg = self.da3_full_cfg["ipc"] if hasattr(self, "da3_full_cfg") else {}
        service_name = str(ipc_cfg["handle_service_name"])

        client = self.create_client(Trigger, service_name)
        self.get_logger().info(f"Waiting for cam_node service: {service_name}")
        while not client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                raise RuntimeError("rclpy shutting down while waiting for cam_node service")
            self.get_logger().info(f"...still waiting for {service_name}")

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is None or not resp.success:
            raise RuntimeError(f"cam_node service call failed: {resp}")

        # Decode and import the CUDA IPC handle. torch.multiprocessing was
        # imported at module load so reduce_tensor / rebuild_cuda_tensor are
        # registered and pickle.loads can rebuild a tensor that aliases
        # cam_node's GPU memory.
        payload = base64.b64decode(resp.message.encode("ascii"))
        depth_view = pickle.loads(payload)
        if not isinstance(depth_view, torch.Tensor):
            raise TypeError(f"IPC payload did not unpickle to torch.Tensor: {type(depth_view)}")
        if depth_view.device.type != "cuda":
            raise RuntimeError(f"IPC depth tensor is not on CUDA: {depth_view.device}")

        self.latest_depth = depth_view
        self.get_logger().info(
            f"Depth IPC handle received: shape={tuple(depth_view.shape)} "
            f"dtype={depth_view.dtype} device={depth_view.device}"
        )

    # ------------------------------------------------------------------
    # Step 4: subs & pubs
    # ------------------------------------------------------------------
    def _init_pub_sub(self) -> None:
        """Wire ROS I/O: subscribe to /joint_states + fk_topic, publish /target_joint_states."""
        self.target_pub = self.create_publisher(
            JointState, "/target_joint_states", HIGH_RELIA_QOS
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._sub_joint_state_cb,
            HIGH_PERF_QOS,
            callback_group=self.obs_mutex_grp,
        )
        # Single FK PoseArray from fk_node. Pose order matches fk_cfg.link_names.
        self.create_subscription(
            PoseArray,
            self.fk_topic,
            self._sub_fk_cb,
            HIGH_PERF_QOS,
            callback_group=self.obs_mutex_grp,
        )

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
            self.get_logger().error(
                f"/joint_states DoF mismatch: {len(msg.position)} vs expected {self.action_num}"
            )
            rclpy.shutdown()
            return
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
            callback_group=self.infer_mutex_grp,
        )
        self.get_logger().info(
            f"Policy timer started @ {self.policy_hz:.3f}Hz (period={timer_dt:.6f}s)."
        )

    def _compose_student_obs(
        self,
        joint_pos_policy: torch.Tensor,  # [1, A] policy order
        joint_vel_policy: torch.Tensor,  # [1, A] policy order
        targets_policy: torch.Tensor,    # [1, A] policy order
    ) -> torch.Tensor:
        """Build the unstacked student observation vector in the policy's expected key order.

        Inputs are already in policy order. FK results are read from pre-computed
        buffers maintained by the FK loop — no FK call here.
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

        # Snapshot FK positions out of fk_pose_dict (concatenate into a flat
        # ndarray under fk_lock), then H2D outside the lock. The lookups go
        # through "<linkname>_pos" keys; orientation entries are written by the
        # callback but unused here (positions only feed the policy).
        with self.fk_lock:
            l_ft_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.left_fingertip_names])
            r_ft_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.right_fingertip_names])
            l_hb_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.left_hand_base_names])
            r_hb_np = np.concatenate([self.fk_pose_dict[n + "_pos"] for n in self.right_hand_base_names])
        l_ft = torch.from_numpy(l_ft_np).unsqueeze(0).to(self.device, non_blocking=True)
        r_ft = torch.from_numpy(r_ft_np).unsqueeze(0).to(self.device, non_blocking=True)
        l_hb = torch.from_numpy(l_hb_np).unsqueeze(0).to(self.device, non_blocking=True)
        r_hb = torch.from_numpy(r_hb_np).unsqueeze(0).to(self.device, non_blocking=True)

        full_obs = {
            "leftJointPosScaled": left_jp_s,
            "rightJointPosScaled": right_jp_s,
            "leftJointVelScaled": left_jv_s,
            "rightJointVelScaled": right_jv_s,
            "leftTargets": left_tgt,
            "rightTargets": right_tgt,
            "leftFingerTipsPos": l_ft,
            "rightFingerTipsPos": r_ft,
            "leftHandBasePos": l_hb,
            "rightHandBasePos": r_hb,
        }
        student_obs = torch.cat([full_obs[k] for k in self.student_obs_keys], dim=-1)
        if int(student_obs.shape[-1]) != self.student_obs_unstacked_space:
            raise ValueError(
                f"composed student obs dim={int(student_obs.shape[-1])} "
                f"!= student_obs_unstacked_space={self.student_obs_unstacked_space}"
            )
        return student_obs

    def _compute_next_targets(
        self,
        actions_policy: torch.Tensor,        # [1, A] policy order
        prev_actions_policy: torch.Tensor,   # [1, A] policy order
        joint_pos_policy: torch.Tensor,      # [1, A] policy order
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EMA-smooth actions and integrate to next joint targets (policy order), clamped to soft limits."""
        ema_actions = (
            actions_policy.clamp(-1.0, 1.0) * self.action_ema
            + prev_actions_policy * (1.0 - self.action_ema)
        )
        next_targets_policy = torch.clamp(
            joint_pos_policy + ema_actions * self.dt * self.full_scale_policy * self.action_scale,
            min=self.full_lower_policy,
            max=self.full_upper_policy,
        )
        return next_targets_policy, ema_actions

    def _log_policy_timing(
        self,
        joint_lock_s: float,
        compose_obs_s: float,
        inference_s: float,
        total_s: float,
    ) -> None:
        """Warn when the policy step blows past per-stage / total budgets."""
        stage_threshold_s = 0.01
        total_threshold_s = 0.02
        stages = {
            "joint_lock": joint_lock_s,
            "compose_obs": compose_obs_s,
            "inference": inference_s,
        }
        slow_stages = [f"{n}={v:.4f}s" for n, v in stages.items() if v > stage_threshold_s]
        total_slow = total_s > total_threshold_s
        if total_slow or slow_stages:
            parts = []
            if total_slow:
                parts.append(f"total={total_s:.4f}s>{total_threshold_s:.3f}s")
            if slow_stages:
                parts.append(f"stages>{stage_threshold_s:.3f}s: " + ", ".join(slow_stages))
            self.get_logger().warn("[SLOW_POLICY] " + " | ".join(parts))

    def _policy_update_loop(self) -> None:
        """Single policy step: snapshot sensors, run inference, integrate targets, publish."""
        t0 = time.perf_counter()
        if not self.has_joint_state:
            return

        # 1) Read the IPC-shared depth tensor directly. cam_node writes via
        # copy_ into this same GPU memory, so latest_depth always points to
        # the freshest completed inference. No lock, no clone — torn reads
        # are tolerated.
        depth_t = self.latest_depth
        if depth_t is None:
            self.get_logger().error("Depth IPC handle not yet attached.")
            return
        if depth_t.ndim != 3 or depth_t.shape[0] != 1:
            self.get_logger().error(
                f"Unexpected cached depth shape: {tuple(depth_t.shape)}; expected [1, H, W]"
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

        # 5) Compose obs (policy order), build stacked obs, run policy.
        cur_student_obs = self._compose_student_obs(
            joint_pos_policy=self._joint_pos_policy_t,
            joint_vel_policy=self._joint_vel_policy_t,
            targets_policy=self._targets_snapshot_t,
        )
        t1 = time.perf_counter()

        if self.n_stack_frame == 1:
            self._stacked_obs_buf.copy_(cur_student_obs)
        elif self.n_stack_frame == 2:
            self._stacked_obs_buf[:, : self.student_obs_unstacked_space].copy_(cur_student_obs)
            self._stacked_obs_buf[:, self.student_obs_unstacked_space :].copy_(self.prev_student_obs)
            self.prev_student_obs.copy_(cur_student_obs)
        else:
            raise ValueError(
                f"n_stack_frame={self.n_stack_frame} is not supported by aux_policy_v2 policy loop."
            )
        obs_clamped = torch.clamp(self._stacked_obs_buf, -100.0, 100.0, out=self._obs_clamped_buf)

        with torch.inference_mode():
            obs_normed = self.obs_normalizer(obs_clamped)
            actions_policy = self.policy.act_inference(obs_normed, vision_input=depth_t)
        t2 = time.perf_counter()

        # 6) Integrate and publish (convert policy->real only at publish boundary).
        next_targets_policy, ema_actions_policy = self._compute_next_targets(
            actions_policy=actions_policy,
            prev_actions_policy=self._prev_actions_snapshot_t,
            joint_pos_policy=self._joint_pos_policy_t,
        )
        self.targets_policy.copy_(next_targets_policy)
        self.prev_actions_policy.copy_(ema_actions_policy)

        next_targets_real = next_targets_policy[:, self.policy2real_idx]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.real_joint_names
        msg.position = next_targets_real[0].detach().cpu().tolist()
        self.target_pub.publish(msg)

        t_end = time.perf_counter()
        self._log_policy_timing(
            joint_lock_s=t02 - t0,
            compose_obs_s=t1 - t02,
            inference_s=t2 - t1,
            total_s=t_end - t0,
        )

    def _build_node_cfg_summary(self) -> dict[str, Any]:
        """Build a deterministic runtime summary for the final ready log."""
        return {
            "paths": {
                "checkpoint_path": str(self.checkpoint_path),
                "runtime_cfg_path": str(self.runtime_cfg_path),
                "env_cfg_path": str(self.env_cfg_path),
                "hand_env_cfg_path": str(self.hand_env_cfg_path),
                "agent_cfg_path": str(self.agent_cfg_path),
                "driver_cfg_path": str(self.driver_cfg_path),
                "policy_node_cfg_path": str(self.policy_node_cfg_path),
                "fk_cfg_path": str(self.fk_cfg_path),
                "da3_cfg_path": str(self.da3_cfg_path),
                "rsl_rl_root": str(self.rsl_rl_root),
            },
            "rates_hz": {
                "policy_hz": float(self.policy_hz),
            },
            "depth_ipc": {
                "shape": list(self.latest_depth.shape) if self.latest_depth is not None else None,
                "dtype": str(self.latest_depth.dtype) if self.latest_depth is not None else None,
            },
            "runtime": {
                "dt": float(self.dt),
                "action_ema": float(self.action_ema),
                "action_scale": float(self.action_scale),
            },
            "obs": {
                "n_stack_frame": int(self.n_stack_frame),
                "student_obs_unstacked_space": int(self.student_obs_unstacked_space),
                "observation_space": int(self.obs_dim),
                "state_space": int(self.state_dim),
                "student_obs_keys": list(self.student_obs_keys),
            },
            "policy": {
                "action_num": int(self.action_num),
                "vision_backbone_dim": int(self.vision_backbone_dim),
                "vision_target_dim": int(self.vision_target_dim),
                "student_vision_modality": str(self.student_vision_modality),
                "vision_input_height": int(self.vision_input_height),
                "vision_input_width": int(self.vision_input_width),
                "depth_clip_min": float(self.depth_clip_min),
                "depth_clip_max": float(self.depth_clip_max),
            },
        }

    def destroy_node(self) -> bool:
        """Cancel the policy timer before destroying the ROS node.

        RealSense / DA3 / FK live in cam_node + fk_node; nothing to clean up
        here besides the timer.
        """
        if hasattr(self, "policy_timer") and self.policy_timer is not None:
            self.policy_timer.cancel()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AuxPolicyNode()
    executor = MultiThreadedExecutor(num_threads=8)
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
