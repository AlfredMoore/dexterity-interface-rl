from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any

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

from robot_motion_interface.utils.da3_utils import DA3Inference, DEFAULT_DA3_MODEL
from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
from robot_motion_interface.utils.sim2real import joint_mapping

JS_QOS = HIGH_PERF_QOS
T_JS_QOS = HIGH_RELIA_QOS

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent
LIBS_ROOT = RMI_ROOT.parent
DUAL_CHAIN_URDF_PATH = str((LIBS_ROOT / "robot_description/rl/bimanual_panda_tesollo.urdf").resolve())


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


def _scale_torch(target: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (target - lower) / (upper - lower) - 1.0


def compute_targets(
    dt: float,
    actions: torch.Tensor,  # [1, A], policy order
    prev_actions: torch.Tensor,  # [1, A], policy order
    action_ema: float,
    action_scale: float,
    joint_pos_real: torch.Tensor,  # [1, A], real order
    policy2real_idx: torch.Tensor,  # [A], long
    full_scale_real: torch.Tensor,  # [1, A], real order
    full_lower_real: torch.Tensor,  # [1, A], real order
    full_upper_real: torch.Tensor,  # [1, A], real order
) -> tuple[torch.Tensor, torch.Tensor]:
    ema_actions = actions.clamp(-1.0, 1.0) * action_ema + prev_actions * (1.0 - action_ema)
    actions_real = ema_actions[:, policy2real_idx]
    next_targets_real = torch.clamp(
        joint_pos_real + actions_real * dt * full_scale_real * action_scale,
        min=full_lower_real,
        max=full_upper_real,
    )
    return next_targets_real, ema_actions


class AuxPolicyNode(Node):
    def __init__(self):
        super().__init__("aux_policy_node")
        self.observation_grp = MutuallyExclusiveCallbackGroup()
        self.inference_grp = MutuallyExclusiveCallbackGroup()

        # Paths and runtime setup
        runtime_dir_default = (RMI_ROOT / "runtime").resolve()
        self.declare_parameter("runtime_dir", str(runtime_dir_default))
        self.declare_parameter("checkpoint_path", str((runtime_dir_default / "model.pt").resolve()))
        self.declare_parameter("runtime_cfg_path", str((runtime_dir_default / "runtime_cfg_train.yaml").resolve()))
        self.declare_parameter("env_cfg_path", str((runtime_dir_default / "env.yaml").resolve()))
        self.declare_parameter("agent_cfg_path", str((runtime_dir_default / "agent.yaml").resolve()))
        self.declare_parameter("driver_cfg_path", str((RMI_ROOT / "config" / "rl_bimanual_driver_config.yaml").resolve()))
        self.declare_parameter("policy_node_cfg_path", str((RMI_ROOT / "config" / "rl_policy_node_config.yaml").resolve()))
        self.declare_parameter("rsl_rl_root", str((RMI_ROOT / "dep" / "rsl_rl-HAND").resolve()))

        # Frequencies
        self.declare_parameter("capture_hz", 60.0)
        self.declare_parameter("da3_hz", 30.0)
        self.declare_parameter("policy_hz", 30.0)

        # Optional static-init mode
        self.declare_parameter("enable_realsense", True)
        self.declare_parameter("enable_da3_thread", True)
        self.declare_parameter("enable_policy_timer", True)

        # DA3 override (None => config/default)
        self.declare_parameter("da3_model", "")

        runtime_dir = Path(self.get_parameter("runtime_dir").value).expanduser().resolve()
        self.checkpoint_path = Path(self.get_parameter("checkpoint_path").value).expanduser()
        self.runtime_cfg_path = Path(self.get_parameter("runtime_cfg_path").value).expanduser()
        self.env_cfg_path = Path(self.get_parameter("env_cfg_path").value).expanduser()
        self.agent_cfg_path = Path(self.get_parameter("agent_cfg_path").value).expanduser()
        self.driver_cfg_path = Path(self.get_parameter("driver_cfg_path").value).expanduser()
        self.policy_node_cfg_path = Path(self.get_parameter("policy_node_cfg_path").value).expanduser()
        self.rsl_rl_root = Path(self.get_parameter("rsl_rl_root").value).expanduser().resolve()

        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = (runtime_dir / self.checkpoint_path).resolve()
        if not self.runtime_cfg_path.is_absolute():
            self.runtime_cfg_path = (runtime_dir / self.runtime_cfg_path).resolve()
        if not self.env_cfg_path.is_absolute():
            self.env_cfg_path = (runtime_dir / self.env_cfg_path).resolve()
        if not self.agent_cfg_path.is_absolute():
            self.agent_cfg_path = (runtime_dir / self.agent_cfg_path).resolve()
        if not self.driver_cfg_path.is_absolute():
            self.driver_cfg_path = (runtime_dir / self.driver_cfg_path).resolve()
        if not self.policy_node_cfg_path.is_absolute():
            self.policy_node_cfg_path = (runtime_dir / self.policy_node_cfg_path).resolve()

        self.capture_hz = float(self.get_parameter("capture_hz").value)
        self.da3_hz = float(self.get_parameter("da3_hz").value)
        self.policy_hz = float(self.get_parameter("policy_hz").value)
        self.enable_realsense = bool(self.get_parameter("enable_realsense").value)
        self.enable_da3_thread = bool(self.get_parameter("enable_da3_thread").value)
        self.enable_policy_timer = bool(self.get_parameter("enable_policy_timer").value)

        da3_model_override = str(self.get_parameter("da3_model").value).strip()

        for p in (self.checkpoint_path, self.runtime_cfg_path, self.env_cfg_path, self.agent_cfg_path, self.driver_cfg_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")

        self.runtime_cfg = _load_yaml(self.runtime_cfg_path)
        self.env_cfg = _load_yaml(self.env_cfg_path)
        self.agent_cfg = _load_yaml(self.agent_cfg_path)
        self.driver_cfg = _load_yaml(self.driver_cfg_path)
        self.policy_node_cfg = _load_yaml(self.policy_node_cfg_path) if self.policy_node_cfg_path.exists() else {}

        self.get_logger().info(f"runtime_cfg_path: {self.runtime_cfg_path}")
        self.get_logger().info(f"env_cfg_path: {self.env_cfg_path}")
        self.get_logger().info(f"agent_cfg_path: {self.agent_cfg_path}")
        self.get_logger().info(f"checkpoint_path: {self.checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            raise RuntimeError("CUDA not available.")
        self.get_logger().info(f"Aux policy device: {self.device}")

        self._import_rsl_rl()
        self._build_joint_mappings()
        self._build_policy_and_normalizer()
        self._build_pinocchio()

        # State buffers
        self.joint_lock = threading.Lock()
        self.vision_lock = threading.Lock()
        self.latest_joint_pos_real = torch.zeros((1, self.action_num), dtype=torch.float32, device=self.device)
        self.latest_joint_vel_real = torch.zeros((1, self.action_num), dtype=torch.float32, device=self.device)
        self.has_joint_state = False
        self.targets_real = torch.zeros((1, self.action_num), dtype=torch.float32, device=self.device)
        self.targets_initialized = False
        self.prev_actions_policy = torch.zeros((1, self.action_num), dtype=torch.float32, device=self.device)
        self.prev_student_obs = torch.zeros((1, self.student_obs_unstacked_space), dtype=torch.float32, device=self.device)

        self.latest_color_bgr: np.ndarray | None = None
        self.latest_depth: torch.Tensor | None = None
        self.latest_depth_stamp = 0.0

        self._capture_running = False
        self._capture_thread: threading.Thread | None = None
        self._da3_running = False
        self._da3_thread: threading.Thread | None = None
        self._da3_busy = False
        self._da3_busy_lock = threading.Lock()

        self.target_pub = self.create_publisher(JointState, "/target_joint_states", T_JS_QOS)
        self.create_subscription(
            JointState,
            "/joint_states",
            self._sub_joint_state_cb,
            JS_QOS,
            callback_group=self.observation_grp,
        )

        self.action_ema = float(self.runtime_cfg["action_EMA"])
        self.action_scale = float(self.runtime_cfg["action_scale"])
        self.dt = float(self.runtime_cfg["dt"])
        timer_dt = 1.0 / self.policy_hz
        if abs(timer_dt - self.dt) > 1e-5:
            self.get_logger().warn(
                f"Policy timer dt ({timer_dt:.6f}) != runtime dt ({self.dt:.6f}). "
                "Target integration uses runtime dt."
            )

        self._setup_realsense_and_da3(da3_model_override=da3_model_override)

        if self.enable_policy_timer:
            self.policy_timer = self.create_timer(
                timer_dt,
                self._policy_update_loop,
                callback_group=self.inference_grp,
            )
        else:
            self.policy_timer = None

        self.get_logger().info(
            "AuxPolicyNode ready. "
            f"capture_hz={self.capture_hz}, da3_hz={self.da3_hz}, policy_hz={self.policy_hz}, "
            f"enable_realsense={self.enable_realsense}, enable_da3_thread={self.enable_da3_thread}, "
            f"enable_policy_timer={self.enable_policy_timer}"
        )

    def _import_rsl_rl(self) -> None:
        if not self.rsl_rl_root.exists():
            raise FileNotFoundError(f"rsl_rl root not found: {self.rsl_rl_root}")
        if str(self.rsl_rl_root) not in sys.path:
            sys.path.insert(0, str(self.rsl_rl_root))
        try:
            from rsl_rl.modules.normalizer import EmpiricalNormalization
            from rsl_rl.modules.student_teacher_aux import StudentTeacherAux
        except Exception as exc:
            raise ImportError(
                "Failed to import rsl_rl StudentTeacherAux/EmpiricalNormalization "
                f"from {self.rsl_rl_root}"
            ) from exc
        self.EmpiricalNormalization = EmpiricalNormalization
        self.StudentTeacherAux = StudentTeacherAux

    def _build_joint_mappings(self) -> None:
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

        policy_joint_names = self.runtime_cfg["bimanual_joint_names"]
        policy2real_idx, real2policy_idx = joint_mapping(policy_joint_names, self.real_joint_names)
        self.policy2real_idx = torch.tensor(policy2real_idx, dtype=torch.long, device=self.device)
        self.real2policy_idx = torch.tensor(real2policy_idx, dtype=torch.long, device=self.device)

        pai = self.runtime_cfg["policy_action_indices_dict"]
        self.left_policy_indices = torch.tensor(pai["left"], dtype=torch.long, device=self.device)
        self.right_policy_indices = torch.tensor(pai["right"], dtype=torch.long, device=self.device)

        scale_left = torch.tensor(
            self.runtime_cfg["robot_action_scale_dict"]["left_joint_vel_action"],
            dtype=torch.float32,
            device=self.device,
        )
        scale_right = torch.tensor(
            self.runtime_cfg["robot_action_scale_dict"]["right_joint_vel_action"],
            dtype=torch.float32,
            device=self.device,
        )
        lower_left = torch.tensor(
            self.runtime_cfg["robot_joint_limits_dict"]["left_joint_pose_soft_lower"],
            dtype=torch.float32,
            device=self.device,
        )
        lower_right = torch.tensor(
            self.runtime_cfg["robot_joint_limits_dict"]["right_joint_pose_soft_lower"],
            dtype=torch.float32,
            device=self.device,
        )
        upper_left = torch.tensor(
            self.runtime_cfg["robot_joint_limits_dict"]["left_joint_pose_soft_upper"],
            dtype=torch.float32,
            device=self.device,
        )
        upper_right = torch.tensor(
            self.runtime_cfg["robot_joint_limits_dict"]["right_joint_pose_soft_upper"],
            dtype=torch.float32,
            device=self.device,
        )
        vel_left = torch.tensor(
            self.runtime_cfg["robot_joint_limits_dict"]["left_joint_vel"],
            dtype=torch.float32,
            device=self.device,
        )
        vel_right = torch.tensor(
            self.runtime_cfg["robot_joint_limits_dict"]["right_joint_vel"],
            dtype=torch.float32,
            device=self.device,
        )

        full_scale_policy = torch.zeros(self.action_num, dtype=torch.float32, device=self.device)
        full_lower_policy = torch.zeros(self.action_num, dtype=torch.float32, device=self.device)
        full_upper_policy = torch.zeros(self.action_num, dtype=torch.float32, device=self.device)
        full_vel_policy = torch.zeros(self.action_num, dtype=torch.float32, device=self.device)

        full_scale_policy[self.left_policy_indices] = scale_left
        full_scale_policy[self.right_policy_indices] = scale_right
        full_lower_policy[self.left_policy_indices] = lower_left
        full_lower_policy[self.right_policy_indices] = lower_right
        full_upper_policy[self.left_policy_indices] = upper_left
        full_upper_policy[self.right_policy_indices] = upper_right
        full_vel_policy[self.left_policy_indices] = vel_left
        full_vel_policy[self.right_policy_indices] = vel_right

        self.full_scale_real = full_scale_policy[self.policy2real_idx].unsqueeze(0)
        self.full_lower_real = full_lower_policy[self.policy2real_idx].unsqueeze(0)
        self.full_upper_real = full_upper_policy[self.policy2real_idx].unsqueeze(0)
        self.left_soft_lower = lower_left.unsqueeze(0)
        self.left_soft_upper = upper_left.unsqueeze(0)
        self.right_soft_lower = lower_right.unsqueeze(0)
        self.right_soft_upper = upper_right.unsqueeze(0)
        self.left_vel_limit = vel_left.unsqueeze(0)
        self.right_vel_limit = vel_right.unsqueeze(0)

    def _build_policy_and_normalizer(self) -> None:
        policy_cfg = self.agent_cfg["policy"]

        self.obs_dim = int(self.env_cfg["observation_space"])
        self.state_dim = int(self.env_cfg["state_space"])
        self.student_obs_unstacked_space = int(self.env_cfg["student_obs_unstacked_space"])
        self.vision_backbone_dim = int(policy_cfg["vision_backbone_dim"])
        self.vision_target_dim = int(policy_cfg["vision_target_dim"])
        self.vision_input_height = int(policy_cfg["vision_input_height"])
        self.vision_input_width = int(policy_cfg["vision_input_width"])
        self.depth_clip_min = float(policy_cfg["depth_clip_min"])
        self.depth_clip_max = float(policy_cfg["depth_clip_max"])
        self.student_vision_modality = str(policy_cfg["student_vision_modality"])
        self.da3_process_res = int(self.env_cfg.get("da3_process_res", 308))
        if self.student_vision_modality != "depth":
            raise ValueError(
                "Aux policy deployment currently supports depth modality only, "
                f"got student_vision_modality={self.student_vision_modality}"
            )

        self.student_obs_keys = list(self.env_cfg.get("student", []))
        self.student_obs_keys = [k for k in self.student_obs_keys if k != "visionFeatures"]
        if not self.student_obs_keys:
            self.student_obs_keys = [
                "leftJointPosScaled",
                "rightJointPosScaled",
                "leftTargets",
                "rightTargets",
                "leftFingerTipsPos",
                "rightFingerTipsPos",
                "leftHandBasePos",
                "rightHandBasePos",
            ]

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

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.policy.load_state_dict(model_sd, strict=True)

        self.obs_normalizer = self.EmpiricalNormalization(shape=self.obs_dim).to(self.device)
        if "obs_norm_state_dict" in ckpt:
            self.obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
        self.obs_normalizer.eval()

        self._run_consistency_checks()

    def _run_consistency_checks(self) -> None:
        first_student_layer = self.policy.student[0]
        if not hasattr(first_student_layer, "in_features"):
            raise ValueError("Student first layer is not Linear; cannot validate input dimension.")
        student_in_expected = self.obs_dim + self.vision_backbone_dim
        student_in_actual = int(first_student_layer.in_features)
        if student_in_actual != student_in_expected:
            raise ValueError(
                "student first layer input mismatch: "
                f"actual={student_in_actual}, expected={student_in_expected}"
            )
        if int(self.policy.vision_backbone_dim) != self.vision_backbone_dim:
            raise ValueError(
                "vision_backbone_dim mismatch: "
                f"model={self.policy.vision_backbone_dim}, cfg={self.vision_backbone_dim}"
            )
        if int(self.policy.num_proprio_obs) != self.obs_dim:
            raise ValueError(
                f"obs_dim mismatch: model={self.policy.num_proprio_obs}, cfg={self.obs_dim}"
            )
        if self.policy.student[-1].out_features != self.action_num:
            raise ValueError(
                f"action dim mismatch: model={self.policy.student[-1].out_features}, runtime={self.action_num}"
            )
        if int(self.obs_normalizer._mean.shape[-1]) != self.obs_dim:
            raise ValueError(
                f"obs normalizer shape mismatch: {self.obs_normalizer._mean.shape} vs obs_dim={self.obs_dim}"
            )
        obs_dof = self.env_cfg.get("obs_dof", {})
        if isinstance(obs_dof, dict) and self.student_obs_keys:
            inferred_unstacked = int(sum(int(obs_dof[k]) for k in self.student_obs_keys))
            if inferred_unstacked != self.student_obs_unstacked_space:
                raise ValueError(
                    "student_obs_unstacked_space mismatch: "
                    f"inferred={inferred_unstacked}, cfg={self.student_obs_unstacked_space}"
                )
        n_stack_frame = int(self.env_cfg.get("n_stack_frame", 2))
        if n_stack_frame * self.student_obs_unstacked_space != self.obs_dim:
            raise ValueError(
                "stacked obs dim mismatch: "
                f"n_stack_frame({n_stack_frame}) * student_obs_unstacked_space({self.student_obs_unstacked_space}) "
                f"!= observation_space({self.obs_dim})"
            )

    def _build_pinocchio(self) -> None:
        self.pin_model = pin.buildModelFromUrdf(DUAL_CHAIN_URDF_PATH)
        self.pin_data = self.pin_model.createData()
        if self.pin_model.nq != self.action_num:
            raise ValueError(
                f"Pinocchio nq mismatch: nq={self.pin_model.nq}, action_num={self.action_num}"
            )

        hand_link_dict = self.env_cfg["hand_link_dict"]
        finger_tip_links = hand_link_dict["finger_tips"]
        hand_palm_links = hand_link_dict["hand_palm"]
        self.left_fingertip_ids = [self.pin_model.getFrameId(f"left_{n}") for n in finger_tip_links]
        self.right_fingertip_ids = [self.pin_model.getFrameId(f"right_{n}") for n in finger_tip_links]
        self.left_hand_base_ids = [self.pin_model.getFrameId(f"left_{n}") for n in hand_palm_links]
        self.right_hand_base_ids = [self.pin_model.getFrameId(f"right_{n}") for n in hand_palm_links]

    def _setup_realsense_and_da3(self, da3_model_override: str) -> None:
        self.rs_pipeline = None
        self.da3 = None

        realsense_cfg = self.policy_node_cfg.get("realsense", {})
        color_intrinsics = realsense_cfg.get("color_intrinsics", {})
        sensor_settings = realsense_cfg.get("sensor_settings", {})

        self.rs_fps = int(realsense_cfg.get("rs_fps", int(self.capture_hz)))
        self.rs_width = int(color_intrinsics.get("width", 640))
        self.rs_height = int(color_intrinsics.get("height", 480))

        if self.enable_realsense:
            self.rs_pipeline = rs.pipeline()
            rs_config = rs.config()
            rs_config.enable_stream(rs.stream.color, self.rs_width, self.rs_height, rs.format.bgr8, self.rs_fps)
            rs_profile = self.rs_pipeline.start(rs_config)
            self._apply_sensor_settings(rs_profile, sensor_settings)
            self._capture_running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="aux_rs_capture")
            self._capture_thread.start()
            self.get_logger().info(
                f"RealSense capture started: {self.rs_width}x{self.rs_height}@{self.rs_fps}Hz"
            )

        model_name = da3_model_override or self._resolve_da3_model()
        fx = float(color_intrinsics.get("fx", 0.0))
        fy = float(color_intrinsics.get("fy", 0.0))
        focal = 0.5 * (fx + fy) if fx > 0 and fy > 0 else None
        self.da3 = DA3Inference(
            model=model_name,
            focal=focal,
            fx=float(color_intrinsics.get("fx")) if "fx" in color_intrinsics else None,
            fy=float(color_intrinsics.get("fy")) if "fy" in color_intrinsics else None,
            cx=float(color_intrinsics.get("cx")) if "cx" in color_intrinsics else None,
            cy=float(color_intrinsics.get("cy")) if "cy" in color_intrinsics else None,
            device=str(self.device),
            process_res=self.da3_process_res,
        )
        self.get_logger().info(f"DA3 ready: model={model_name}, process_res={self.da3_process_res}")

        if self.enable_da3_thread:
            self._da3_running = True
            self._da3_thread = threading.Thread(target=self._da3_loop, daemon=True, name="aux_da3")
            self._da3_thread.start()

    def _resolve_da3_model(self) -> str:
        cv_model = self.policy_node_cfg.get("cv_model", {})
        da3_cfg = cv_model.get("da3", {})
        model = str(da3_cfg.get("model", "")).strip()
        return model or DEFAULT_DA3_MODEL

    def _apply_sensor_settings(self, rs_profile, sensor_settings: dict[str, Any]) -> None:
        if not sensor_settings:
            return
        try:
            device = rs_profile.get_device()
            sensors = device.query_sensors()
        except Exception:
            return

        auto_exposure = sensor_settings.get("auto_exposure", False)
        exposure = sensor_settings.get("exposure", 350)
        gain = sensor_settings.get("gain", 16)

        for sensor in sensors:
            if auto_exposure is not None and sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
            if auto_exposure is False:
                if exposure is not None and sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, float(exposure))
                if gain is not None and sensor.supports(rs.option.gain):
                    sensor.set_option(rs.option.gain, float(gain))

    def _capture_loop(self) -> None:
        period = 1.0 / max(self.capture_hz, 1e-3)
        while self._capture_running and self.rs_pipeline is not None:
            loop_start = time.perf_counter()
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_bgr = np.asanyarray(color_frame.get_data())
                with self.vision_lock:
                    self.latest_color_bgr = color_bgr
            except Exception:
                pass

            elapsed = time.perf_counter() - loop_start
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _try_enter_da3(self) -> bool:
        with self._da3_busy_lock:
            if self._da3_busy:
                return False
            self._da3_busy = True
            return True

    def _leave_da3(self) -> None:
        with self._da3_busy_lock:
            self._da3_busy = False

    def _da3_loop(self) -> None:
        if self.da3 is None:
            return
        period = 1.0 / max(self.da3_hz, 1e-3)
        while self._da3_running:
            loop_start = time.perf_counter()
            if not self._try_enter_da3():
                time.sleep(0.001)
                continue
            try:
                with self.vision_lock:
                    color_bgr = None if self.latest_color_bgr is None else self.latest_color_bgr.copy()
                if color_bgr is not None:
                    depth = self.da3.infer(color_bgr)
                    depth_np = np.asarray(depth, dtype=np.float32)
                    depth_t = torch.from_numpy(depth_np).to(self.device, non_blocking=True)
                    depth_t = depth_t.clamp(min=self.depth_clip_min, max=self.depth_clip_max).to(torch.float16)
                    with self.vision_lock:
                        self.latest_depth = depth_t
                        self.latest_depth_stamp = time.time()
            except Exception as exc:
                self.get_logger().warn(f"DA3 inference failed: {exc}")
            finally:
                self._leave_da3()

            elapsed = time.perf_counter() - loop_start
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _sub_joint_state_cb(self, msg: JointState) -> None:
        if len(msg.position) != self.action_num:
            self.get_logger().warn(
                f"Received /joint_states with unexpected DoF: {len(msg.position)} (expected {self.action_num})"
            )
            return
        vel = msg.velocity if len(msg.velocity) == self.action_num else [0.0] * self.action_num
        with self.joint_lock:
            self.latest_joint_pos_real[0, :] = torch.as_tensor(msg.position, dtype=torch.float32, device=self.device)
            self.latest_joint_vel_real[0, :] = torch.as_tensor(vel, dtype=torch.float32, device=self.device)
            self.has_joint_state = True
            if not self.targets_initialized:
                self.targets_real[0, :] = self.latest_joint_pos_real[0, :]
                self.targets_initialized = True

    def _pinocchio_forward_kinematics(
        self,
        q_real_np: np.ndarray,  # [A]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = q_real_np.astype(np.float64, copy=False)
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        left_hand_base_pos = np.array([self.pin_data.oMf[i].translation for i in self.left_hand_base_ids], dtype=np.float32)
        right_hand_base_pos = np.array([self.pin_data.oMf[i].translation for i in self.right_hand_base_ids], dtype=np.float32)
        left_fingertips_pos = np.array([self.pin_data.oMf[i].translation for i in self.left_fingertip_ids], dtype=np.float32)
        right_fingertips_pos = np.array([self.pin_data.oMf[i].translation for i in self.right_fingertip_ids], dtype=np.float32)
        return left_hand_base_pos, left_fingertips_pos, right_hand_base_pos, right_fingertips_pos

    def _compose_student_obs(
        self,
        joint_pos_real: torch.Tensor,  # [1, A]
        joint_vel_real: torch.Tensor,  # [1, A]
        targets_real: torch.Tensor,  # [1, A]
    ) -> torch.Tensor:
        # Convert to policy action order first for consistency with runtime cfg limits/scales.
        joint_pos_policy = joint_pos_real[:, self.real2policy_idx]
        joint_vel_policy = joint_vel_real[:, self.real2policy_idx]
        targets_policy = targets_real[:, self.real2policy_idx]

        left_joint_pos = joint_pos_policy[:, self.left_policy_indices]
        right_joint_pos = joint_pos_policy[:, self.right_policy_indices]
        left_joint_vel = joint_vel_policy[:, self.left_policy_indices]
        right_joint_vel = joint_vel_policy[:, self.right_policy_indices]
        left_targets = targets_policy[:, self.left_policy_indices]
        right_targets = targets_policy[:, self.right_policy_indices]

        left_joint_pos_scaled = _scale_torch(left_joint_pos, self.left_soft_lower, self.left_soft_upper)
        right_joint_pos_scaled = _scale_torch(right_joint_pos, self.right_soft_lower, self.right_soft_upper)
        left_joint_vel_scaled = left_joint_vel / self.left_vel_limit
        right_joint_vel_scaled = right_joint_vel / self.right_vel_limit

        # FK expects real driver order; one torch->numpy conversion here.
        q_real_np = joint_pos_real[0].detach().cpu().numpy()
        left_hand_base_np, left_fingertips_np, right_hand_base_np, right_fingertips_np = self._pinocchio_forward_kinematics(
            q_real_np
        )

        left_hand_base_pos = torch.from_numpy(left_hand_base_np.reshape(1, -1)).to(self.device)
        right_hand_base_pos = torch.from_numpy(right_hand_base_np.reshape(1, -1)).to(self.device)
        left_fingertips_pos = torch.from_numpy(left_fingertips_np.reshape(1, -1)).to(self.device)
        right_fingertips_pos = torch.from_numpy(right_fingertips_np.reshape(1, -1)).to(self.device)

        full_obs = {
            "leftJointPosScaled": left_joint_pos_scaled,
            "rightJointPosScaled": right_joint_pos_scaled,
            "leftJointVelScaled": left_joint_vel_scaled,
            "rightJointVelScaled": right_joint_vel_scaled,
            "leftTargets": left_targets,
            "rightTargets": right_targets,
            "leftFingerTipsPos": left_fingertips_pos,
            "rightFingerTipsPos": right_fingertips_pos,
            "leftHandBasePos": left_hand_base_pos,
            "rightHandBasePos": right_hand_base_pos,
        }

        return torch.cat([full_obs[k] for k in self.student_obs_keys], dim=-1)

    def _policy_update_loop(self) -> None:
        if not self.has_joint_state or not self.targets_initialized:
            return

        with self.vision_lock:
            depth_t = None if self.latest_depth is None else self.latest_depth.clone()
        if depth_t is None:
            return
        if depth_t.ndim == 2:
            depth_t = depth_t.unsqueeze(0)  # [1, H, W]
        elif depth_t.ndim == 3:
            pass
        else:
            self.get_logger().warn(f"Unexpected depth shape: {tuple(depth_t.shape)}")
            return

        with self.joint_lock:
            joint_pos_real = self.latest_joint_pos_real.clone()
            joint_vel_real = self.latest_joint_vel_real.clone()
            targets_real = self.targets_real.clone()
            prev_actions_policy = self.prev_actions_policy.clone()

        cur_student_obs = self._compose_student_obs(
            joint_pos_real=joint_pos_real,
            joint_vel_real=joint_vel_real,
            targets_real=targets_real,
        )
        stacked_obs = torch.cat((cur_student_obs, self.prev_student_obs), dim=-1)
        self.prev_student_obs = cur_student_obs.detach()

        obs_clamped = torch.clamp(stacked_obs, -100.0, 100.0)

        with torch.inference_mode():
            obs_normed = self.obs_normalizer(obs_clamped)
            actions_policy = self.policy.act_inference(obs_normed, vision_input=depth_t)

        next_targets_real, ema_actions_policy = compute_targets(
            dt=self.dt,
            actions=actions_policy,
            prev_actions=prev_actions_policy,
            action_ema=self.action_ema,
            action_scale=self.action_scale,
            joint_pos_real=joint_pos_real,
            policy2real_idx=self.policy2real_idx,
            full_scale_real=self.full_scale_real,
            full_lower_real=self.full_lower_real,
            full_upper_real=self.full_upper_real,
        )

        with self.joint_lock:
            self.targets_real.copy_(next_targets_real)
            self.prev_actions_policy.copy_(ema_actions_policy)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.real_joint_names
        msg.position = next_targets_real[0].detach().cpu().tolist()
        self.target_pub.publish(msg)

    def destroy_node(self) -> bool:
        self._capture_running = False
        self._da3_running = False
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self._da3_thread is not None and self._da3_thread.is_alive():
            self._da3_thread.join(timeout=1.0)
        if self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AuxPolicyNode()
    try:
        executor = MultiThreadedExecutor(num_threads=8)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
