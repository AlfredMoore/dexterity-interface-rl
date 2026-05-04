from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pinocchio as pin
import pyrealsense2 as rs
import rclpy
import torch
import yaml
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from robot_motion_interface.utils.da3_compile_utils import DA3Inference
from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
from robot_motion_interface.utils.sim2real import joint_mapping


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent
LIBS_ROOT = RMI_ROOT.parent
DUAL_CHAIN_URDF_PATH = str((LIBS_ROOT / "robot_description/rl/bimanual_panda_tesollo.urdf").resolve())


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
        self._build_pinocchio()
        self._build_policy_and_normalizer()
        self._init_state_buffers()
        self._setup_realsense_and_da3()

        # -- 4. subs & pubs --
        self._init_pub_sub()
        self._setup_fk()

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
        self.declare_parameter("rsl_rl_root", str((RMI_ROOT / "dep" / "rsl_rl-HAND").resolve()))

        self.checkpoint_path = Path(self.get_parameter("checkpoint_path").value)
        self.runtime_cfg_path = Path(self.get_parameter("runtime_cfg_path").value)
        self.env_cfg_path = Path(self.get_parameter("env_cfg_path").value)
        self.hand_env_cfg_path = Path(self.get_parameter("hand_env_cfg_path").value)
        self.agent_cfg_path = Path(self.get_parameter("agent_cfg_path").value)
        self.driver_cfg_path = Path(self.get_parameter("driver_cfg_path").value)
        self.policy_node_cfg_path = Path(self.get_parameter("policy_node_cfg_path").value)
        self.rsl_rl_root = Path(self.get_parameter("rsl_rl_root").value)

    def _load_configs(self) -> None:
        """Load all YAML cfgs. Missing files raise FileNotFoundError naturally."""
        self.runtime_cfg = _load_yaml(self.runtime_cfg_path)
        self.env_cfg = _load_yaml(self.env_cfg_path)
        self.hand_env_cfg = _load_yaml(self.hand_env_cfg_path)
        self.agent_cfg = _load_yaml(self.agent_cfg_path)
        self.driver_cfg = _load_yaml(self.driver_cfg_path)
        self.policy_node_cfg = _load_yaml(self.policy_node_cfg_path)

        self.get_logger().info("#### Aux policy node configs: ####")
        self.get_logger().info(f"runtime_cfg_path:     {self.runtime_cfg_path}")
        self.get_logger().info(f"env_cfg_path:         {self.env_cfg_path}")
        self.get_logger().info(f"hand_env_cfg_path:    {self.hand_env_cfg_path}")
        self.get_logger().info(f"agent_cfg_path:       {self.agent_cfg_path}")
        self.get_logger().info(f"driver_cfg_path:      {self.driver_cfg_path}")
        self.get_logger().info(f"policy_node_cfg_path: {self.policy_node_cfg_path}")
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
        self.vision_lock = threading.Lock()
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

    def _build_pinocchio(self) -> None:
        self.pin_model = pin.buildModelFromUrdf(DUAL_CHAIN_URDF_PATH)
        self.pin_data = self.pin_model.createData()
        if self.pin_model.nq != self.action_num:
            raise ValueError(
                f"Pinocchio nq mismatch: nq={self.pin_model.nq}, action_num={self.action_num}"
            )

        hand_link_dict = self.hand_env_cfg["env"]["robot"]["linkNames"]
        finger_tip_links = hand_link_dict["finger_tips"]
        hand_palm_links = hand_link_dict["hand_palm"]
        self.left_fingertip_ids = [self.pin_model.getFrameId(f"left_{n}") for n in finger_tip_links]
        self.right_fingertip_ids = [self.pin_model.getFrameId(f"right_{n}") for n in finger_tip_links]
        self.left_hand_base_ids = [self.pin_model.getFrameId(f"left_{n}") for n in hand_palm_links]
        self.right_hand_base_ids = [self.pin_model.getFrameId(f"right_{n}") for n in hand_palm_links]

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

        # FK results: CPU numpy. Pre-allocated; FK loop writes via np.copyto under fk_lock.
        # Compose_obs takes its own snapshot copy (also via np.copyto) before doing H2D.
        n_palm = len(self.left_hand_base_ids)
        n_tips = len(self.left_fingertip_ids)
        self.latest_left_hand_base_np = np.zeros((n_palm, 3), dtype=np.float32)
        self.latest_right_hand_base_np = np.zeros((n_palm, 3), dtype=np.float32)
        self.latest_left_fingertips_np = np.zeros((n_tips, 3), dtype=np.float32)
        self.latest_right_fingertips_np = np.zeros((n_tips, 3), dtype=np.float32)
        # Snapshots used by compose_obs to copy FK out of the shared buffers.
        self._fk_l_hb_snapshot_np = np.zeros((n_palm, 3), dtype=np.float32)
        self._fk_r_hb_snapshot_np = np.zeros((n_palm, 3), dtype=np.float32)
        self._fk_l_ft_snapshot_np = np.zeros((n_tips, 3), dtype=np.float32)
        self._fk_r_ft_snapshot_np = np.zeros((n_tips, 3), dtype=np.float32)
        self.has_fk: bool = False

        # Vision (color frame on CPU; depth tensor on GPU).
        self.latest_color_rgb: np.ndarray | None = None
        self.latest_depth: torch.Tensor | None = None
        self.latest_depth_stamp: float = 0.0

        # Background thread handles + run flags (set up by _setup_realsense_and_da3 / _setup_fk).
        self._capture_running: bool = False
        self._capture_thread: threading.Thread | None = None
        self._da3_running: bool = False
        self._da3_thread: threading.Thread | None = None
        self._fk_running: bool = False
        self._fk_thread: threading.Thread | None = None

    def _setup_realsense_and_da3(self) -> None:
        """Start RealSense color stream + DA3 depth inference.

        Both are mandatory; the policy loop blocks on `latest_depth` being available.
        """
        realsense_cfg = self.policy_node_cfg["realsense"]
        da3_cfg = self.policy_node_cfg["da3_cfg"]
        env_process_res = int(self.env_cfg["da3_process_res"])
        da3_process_res = int(da3_cfg["process_res"])
        if env_process_res != da3_process_res:
            raise ValueError(
                "process_res mismatch between env_cfg and da3_cfg: "
                f"env_cfg.da3_process_res={env_process_res}, "
                f"da3_cfg.process_res={da3_process_res}"
            )
        color_intrinsics = realsense_cfg["color_intrinsics"]
        sensor_settings = realsense_cfg.get("sensor_settings", {})

        self.capture_hz = float(realsense_cfg["rs_fps"])
        self.da3_hz = float(da3_cfg["rate"])
        self.rs_width = int(color_intrinsics["width"])
        self.rs_height = int(color_intrinsics["height"])

        # -- RealSense color pipeline --
        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.color, self.rs_width, self.rs_height, rs.format.bgr8, int(self.capture_hz)
        )
        rs_profile = self.rs_pipeline.start(rs_config)
        self._apply_sensor_settings(rs_profile, sensor_settings)
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._cam_capture_loop, daemon=True, name="aux_rs_capture"
        )
        self._capture_thread.start()
        self.get_logger().info(
            f"RealSense capture started: {self.rs_width}x{self.rs_height}@{int(self.capture_hz)}Hz"
        )
        time.sleep(0.5)

        # -- DA3 depth inference --
        self.get_logger().info(f"DA3 compilation started.")
        self.da3 = DA3Inference.from_dict(da3_cfg)
        self._da3_running = True
        self._da3_thread = threading.Thread(target=self._da3_loop, daemon=True, name="aux_da3")
        self._da3_thread.start()
        self.get_logger().info(
            f"DA3 ready: model={self.da3.model_name}, process_res={self.da3.process_res}, rate={self.da3_hz}Hz"
        )

    def _apply_sensor_settings(self, rs_profile, sensor_settings: dict[str, Any]) -> None:
        """Apply optional RealSense sensor settings (auto_exposure / exposure / gain)."""
        if not sensor_settings:
            return
        sensors = rs_profile.get_device().query_sensors()
        auto_exposure = sensor_settings.get("auto_exposure", False)
        exposure = sensor_settings.get("exposure")
        gain = sensor_settings.get("gain")
        for sensor in sensors:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
            if not auto_exposure:
                if exposure is not None and sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, float(exposure))
                if gain is not None and sensor.supports(rs.option.gain):
                    sensor.set_option(rs.option.gain, float(gain))

    def _cam_capture_loop(self) -> None:
        """RealSense color-frame capture; writes BGR ndarray under vision_lock."""
        period = 1.0 / max(self.capture_hz, 1e-3)
        while self._capture_running:
            loop_start = time.perf_counter()
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()

                # cvtColor allocates a fresh ndarray, also detaching from
                # librealsense's internal buffer (recycled next iteration).
                color_rgb = cv2.cvtColor(
                    np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB
                )
                with self.vision_lock:
                    self.latest_color_rgb = color_rgb
                    
            except Exception as exc:
                self.get_logger().warn(f"RealSense capture error: {exc}")
                self._capture_running = False
                rclpy.shutdown()
                return

            elapsed = time.perf_counter() - loop_start
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _da3_loop(self) -> None:
        """Run DA3 depth inference on the latest color frame; writes depth tensor under vision_lock."""
        period = 1.0 / max(self.da3_hz, 1e-3)
        while self._da3_running:
            loop_start = time.perf_counter()
            try:
                with self.vision_lock:
                    color_rgb = self.latest_color_rgb.copy() if self.latest_color_rgb is not None else None
                if color_rgb is None:
                    self.get_logger().error("DA3 loop: no color frame, shutting down node.")
                    self._da3_running = False
                    rclpy.shutdown()
                    return

                infer_start = time.perf_counter()
                color_rgb_t = torch.from_numpy(color_rgb).to(self.device, non_blocking=True).unsqueeze(0)
                depth_t = self.da3.infer_no_chunk(color_rgb_t)
                if depth_t.dim() == 2:
                    depth_t = depth_t.unsqueeze(0)
                elif depth_t.dim() == 3 and depth_t.shape[0] == 1:
                    pass
                else:
                    raise ValueError(f"Unexpected DA3 depth shape: {tuple(depth_t.shape)}")
                depth_t = depth_t.to(dtype=torch.float32).clamp(
                    min=self.depth_clip_min, max=self.depth_clip_max
                ).to(torch.float16)
                
                with self.vision_lock:
                    self.latest_depth = depth_t
                    self.latest_depth_stamp = time.time()
                    
            except Exception as exc:
                self.get_logger().warn(f"DA3 inference failed: {exc}")
                self._da3_running = False
                rclpy.shutdown()
                return

            elapsed = time.perf_counter() - loop_start
            
            sleep_s = period - elapsed
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                self.get_logger().warn(
                    f"[SLOW_DA3] total={elapsed:.4f}s, "
                    f"target_period={period:.4f}s"
                )

    # ------------------------------------------------------------------
    # Step 3.5: FK background loop
    # ------------------------------------------------------------------
    def _setup_fk(self) -> None:
        """Spawn the FK background thread.

        FK runs at policy_node_cfg.fk_rate (independent of policy / da3 / capture)
        and produces fingertip + palm positions on GPU for the policy loop to read.
        The thread itself waits for /joint_states to populate before producing FK.
        """
        self.fk_hz = float(self.policy_node_cfg["fk_rate"])
        self._fk_running = True
        self._fk_thread = threading.Thread(target=self._fk_loop, daemon=True, name="aux_fk")
        self._fk_thread.start()
        self.get_logger().info(f"FK loop started @ {self.fk_hz}Hz")

    def _fk_loop(self) -> None:
        """Read latest joint_pos, run pinocchio FK, push results to GPU under fk_lock."""
        period = 1.0 / max(self.fk_hz, 1e-3)
        q_local = np.zeros((self.action_num,), dtype=np.float32)
        while self._fk_running:
            loop_start = time.perf_counter()

            # Snapshot latest joint_pos out of joint_lock fast.
            ready = False
            with self.joint_lock:
                if self.has_joint_state:
                    np.copyto(q_local, self.latest_joint_pos_real_np)
                    ready = True
            if not ready:
                time.sleep(period)
                continue

            # CPU-only FK; H2D is deferred to the policy step (avoids wasted
            # transfers when fk_hz > policy_hz).
            l_hb_np, l_ft_np, r_hb_np, r_ft_np = self._pinocchio_fk(q_local)

            # Copy into pre-allocated shared buffers under fk_lock (zero alloc).
            with self.fk_lock:
                np.copyto(self.latest_left_hand_base_np, l_hb_np)
                np.copyto(self.latest_right_hand_base_np, r_hb_np)
                np.copyto(self.latest_left_fingertips_np, l_ft_np)
                np.copyto(self.latest_right_fingertips_np, r_ft_np)
                self.has_fk = True

            elapsed = time.perf_counter() - loop_start
            sleep_s = period - elapsed
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                self.get_logger().warn(
                    f"[SLOW_FK] total={elapsed:.4f}s, target_period={period:.4f}s"
                )

    # ------------------------------------------------------------------
    # Step 4: subs & pubs
    # ------------------------------------------------------------------
    def _init_pub_sub(self) -> None:
        """Wire ROS I/O: subscribe to /joint_states, publish /target_joint_states."""
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

    def _pinocchio_fk(self, q_real_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward kinematics on the bimanual chain.

        Args:
            q_real_np: (A,) joint positions in real driver order.

        Returns:
            (left_hand_base, left_fingertips, right_hand_base, right_fingertips), each
            (n_links, 3) float32 in world frame.
        """
        q = q_real_np.astype(np.float64, copy=False)
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        oMf = self.pin_data.oMf
        left_hand_base = np.array([oMf[i].translation for i in self.left_hand_base_ids], dtype=np.float32)
        right_hand_base = np.array([oMf[i].translation for i in self.right_hand_base_ids], dtype=np.float32)
        left_fingertips = np.array([oMf[i].translation for i in self.left_fingertip_ids], dtype=np.float32)
        right_fingertips = np.array([oMf[i].translation for i in self.right_fingertip_ids], dtype=np.float32)
        return left_hand_base, left_fingertips, right_hand_base, right_fingertips

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

        # Snapshot FK out of shared buffers (np.copyto under fk_lock), then H2D
        # outside the lock. This is the only FK transfer per policy step.
        with self.fk_lock:
            np.copyto(self._fk_l_hb_snapshot_np, self.latest_left_hand_base_np)
            np.copyto(self._fk_r_hb_snapshot_np, self.latest_right_hand_base_np)
            np.copyto(self._fk_l_ft_snapshot_np, self.latest_left_fingertips_np)
            np.copyto(self._fk_r_ft_snapshot_np, self.latest_right_fingertips_np)
        l_hb = torch.from_numpy(self._fk_l_hb_snapshot_np.reshape(1, -1)).to(self.device, non_blocking=True)
        r_hb = torch.from_numpy(self._fk_r_hb_snapshot_np.reshape(1, -1)).to(self.device, non_blocking=True)
        l_ft = torch.from_numpy(self._fk_l_ft_snapshot_np.reshape(1, -1)).to(self.device, non_blocking=True)
        r_ft = torch.from_numpy(self._fk_r_ft_snapshot_np.reshape(1, -1)).to(self.device, non_blocking=True)

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
        vision_lock_s: float,
        joint_lock_s: float,
        compose_obs_s: float,
        inference_s: float,
        total_s: float,
    ) -> None:
        """Warn when the policy step blows past per-stage / total budgets."""
        stage_threshold_s = 0.01
        total_threshold_s = 0.02
        stages = {
            "vision_lock": vision_lock_s,
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

        # 1) Snapshot latest depth ref quickly under vision_lock.
        with self.vision_lock:
            depth_t = self.latest_depth.clone() if self.latest_depth is not None else None
        t01 = time.perf_counter()
        if depth_t is None:
            self.get_logger().error("No latest depth available.")
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

        # 3) Wait until FK loop has produced at least one sample.
        # with self.fk_lock:
        #     fk_ready = self.has_fk
        # if not fk_ready:
        #     return

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
            vision_lock_s=t01 - t0,
            joint_lock_s=t02 - t01,
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
                "rsl_rl_root": str(self.rsl_rl_root),
            },
            "rates_hz": {
                "capture_hz": float(self.capture_hz),
                "da3_hz": float(self.da3_hz),
                "fk_hz": float(self.fk_hz),
                "policy_hz": float(self.policy_hz),
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
            "realsense": {
                "width": int(self.rs_width),
                "height": int(self.rs_height),
            },
            "da3": {
                "process_res": int(self.da3.process_res),
            },
        }

    def destroy_node(self) -> bool:
        """Stop background workers and sensors before destroying the ROS node."""
        self._capture_running = False
        self._da3_running = False
        self._fk_running = False

        if hasattr(self, "policy_timer") and self.policy_timer is not None:
            self.policy_timer.cancel()

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self._da3_thread is not None and self._da3_thread.is_alive():
            self._da3_thread.join(timeout=1.0)
        if self._fk_thread is not None and self._fk_thread.is_alive():
            self._fk_thread.join(timeout=1.0)

        if hasattr(self, "rs_pipeline") and self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except Exception as exc:
                self.get_logger().error(f"Error stopping RealSense pipeline: {exc}")

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
