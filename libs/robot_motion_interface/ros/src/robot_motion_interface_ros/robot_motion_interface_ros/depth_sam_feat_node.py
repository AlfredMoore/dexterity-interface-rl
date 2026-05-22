"""ROS 2 node: live RealSense + SAM3-init + SAM2 dual-bank tracking + depth-feature inference.

Drop-in replacement for depth_feat_node.py where the YOLO seg masker is swapped
out for SAM2P3DualBankTracker (SAM3 text-prompted bbox/mask anchor + SAM2 video
predictor with periodic re-anchoring via two overlapping memory-bank states).

Capture / inference are decoupled (mirrors depth_feat_node.py):

    ┌─ capture thread ──────────────────────┐
    │ wait_for_frames at rs_fps             │   <- realsense_config.yaml::rs_fps
    │   -> decimation_filter                │
    │   -> hole_filling_filter              │
    │   -> align(depth -> color frame)      │
    │   -> latest_color_bgr / latest_depth  │   (vision_lock-protected ndarrays)
    └───────────────────────────────────────┘
    ┌─ inference timer ─────────────────────┐
    │ ROS timer at runtime.inference_rate   │   <- depth_feat_config.yaml
    │   -> snapshot latest_color/depth      │
    │   -> (lazy on first frame:)           │
    │      tracker.warmup + tracker.init    │   SAM3 ViT-H ~150 ms one-shot
    │   -> tracker.step(color)              │   SAM2 video propagation ~50 ms
    │   -> mask -> masked_depth (np.where)  │
    │   -> uint16 mm -> float32 metres      │
    │   -> DepthFeatureExtractor.step(...)  │
    │   -> un-normalize geom slab           │
    │   -> copy_ into CUDA IPC buffer       │
    │   -> optional Image publish for vis   │   (rqt_image_view / RViz)
    └───────────────────────────────────────┘

Decoupling lets the camera keep running at native FPS while the (heavier)
SAM2P3 + DepthFeatureNetFiLM path runs at whatever rate the GPU can sustain.

(*) Segmentation is kept inline (vs a separate node) so depth and mask stay
    in lockstep on the same captured frame.

(*) DepthFeatureNetFiLM, trained offline by
    HAND-policy/scripts/sim2real/train_depth_predictor.py. Inference mode:
    cfg.train=False (no optimizer, net.eval()), cfg.load_checkpoint=True
    (latest depth_net_*.pth picked up from `depth_feature.log_dir`).

FK conditioning vector (24-d for a 3-finger hand) is assembled per-step from
the latest PoseArray on fk_config.fk_topic. Layout matches train_depth_predictor.py:
    [leftHandBasePos(3), rightHandBasePos(3),
     leftFingerTipsPos(F*3), rightFingerTipsPos(F*3)]
Link names come from the `fk:` section of depth_feat_config.yaml.

Output buffer is a persistent (1, 10) CUDA tensor with layout
    [bottleBodyPos(3), bottleCapPos(3), bottleGeomCfg_metres(4)]
all in metres. The geom slab is un-normalized inside this node
(un_preprocess_geom) so downstream consumers see metres, not [0, 1]. The
buffer's CUDA IPC handle is encoded once at startup and served via the
Trigger service ipc.handle_service_name (default
/depth_feat_node/get_feature_handle); depth_feat_policy_node attaches a
view onto this same memory at startup.

RealSense settings (resolution / fps / intrinsics / exposure) are read from
`libs/robot_motion_interface/config/realsense_config.yaml`.
SAM2P3 + DepthFeatureExtractor + FK link selection + IPC service name +
inference_rate are read from
`libs/robot_motion_interface/config/depth_feat_config.yaml`.

The depth_feat_config.yaml must contain a `sam2p3:` section like:

    sam2p3:
      sam2_ckpt: "models/sam2/sam2.1_hiera_small.pt"   # workspace-relative
      sam2_cfg:  "configs/sam2.1/sam2.1_hiera_s.yaml"  # hydra-resolved by sam2
      sam3_ckpt: "models/sam3/sam3.pt"                 # workspace-relative
      prompt:    "bottle."
      prompt_type: "mask"                              # "mask" | "box"
      cycle_K:   100                                   # primary duration (step() calls)
      warmup_W:  7                                     # overlap window before swap
      sam3_conf: 0.3
      offload_video_to_cpu: false                      # set true on <16 GB GPUs

Memory budget on 4090 (offload_video_to_cpu=false):
    SAM3 ViT-H              ~3.0 GB
    SAM2.1-hiera-small      ~0.2 GB
    Two banks' images bufs  ~3.6 GB  (peak during warmup overlap)
    Memory bank entries     ~0.4 GB
    -------------------------------
    sam2p3 total            ~7.2 GB
    DepthFeatureNetFiLM     ~0.5 GB
"""

from __future__ import annotations

import base64
import io
import threading
import time
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import torch
import torch.multiprocessing  # noqa: F401  registers CUDA IPC reducers in ForkingPickler
import yaml
from geometry_msgs.msg import Pose, PoseArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger

from robot_motion_interface.utils.depth_feature import (
    DepthFeatureExtractor,
    DepthFeatureExtractorCfg,
)
from robot_motion_interface.utils.qos import HIGH_PERF_QOS, HIGH_RELIA_QOS
from robot_motion_interface.utils.sam2p3_dualbank_util import SAM2P3DualBankTracker


# Inside the handrl-policy docker, libs/robot_motion_interface is mounted at
# /workspace/libs/..., so relative paths below resolve correctly. Mirrors the
# convention used by gsam2_node.py — relative defaults + ROS-parameter overrides.
WORKSPACE_ROOT = Path("/workspace")
REALSENSE_CONFIG_PATH = "libs/robot_motion_interface/config/realsense_config.yaml"
DEPTH_FEAT_CONFIG_PATH = "libs/robot_motion_interface/config/depth_feat_config.yaml"
FK_CONFIG_PATH = "libs/robot_motion_interface/config/fk_config.yaml"

# # ── LSQ compensation for body/cap (fitted offline, 40 samples) ──────────
# # corrected_pos = pred_pos @ _LSQ_W + _LSQ_B
# # Input/output layout: [body_x, body_y, body_z, cap_x, cap_y, cap_z] m,
# # in the policy obs frame (z=0 at tabletop + workstation_top/2).
# # Mean residual ≈ 1.88 cm; see libs/.../utils/lsq.py to re-fit when the
# # bottle / camera mount / training run changes.
# _LSQ_W = np.array([
#     [ 0.34887896641805943,  0.5759479808107646,   0.06614204741424203,  0.34887896641805943,  0.575947980810764,    0.06614182648443188 ],
#     [-0.3973368369406706,   0.8405134497492819,   0.01997501091804399, -0.3973368369406706,   0.8405134497492821,   0.019975154132943485],
#     [ 2.2356774912943127,   0.2701747221227452,   0.03863010411512774,  2.2356774912943127,   0.270174722122746,    0.03863030075067703 ],
#     [ 1.499541154062977,   -0.5197405499934648,  -0.03657934233377573,  1.499541154062977,   -0.5197405499934643,  -0.0365791896497728  ],
#     [ 0.02313018766017715,  0.7080967439829899,  -0.05670362713046395,  0.02313018766017715,  0.7080967439829897,  -0.056703724825511594],
#     [-1.3527293168223906,   0.6504247049806191,   0.04017506682628019, -1.3527293168223906,   0.650424704980618,    0.040174965768886906],
# ], dtype=np.float32)
# _LSQ_B = np.array([
#     -0.9315139006546089, -0.9493223558367246, 0.8603479782973352,
#     -0.9315139006546089, -0.9493223558367241, 0.9558478742668937,
# ], dtype=np.float32)

# # ── Geom override (predictor's geom branch is too noisy in real, override
# # with the measured value of the specific bottle in use). Order matches
# # train_depth_predictor.py geom slab:
# #     [body_radius, body_height, cap_radius, cap_height]  metres
# _GEOM_OVERRIDE = np.array([0.04, 0.16, 0.029, 0.02], dtype=np.float32)


def _resolve_workspace_path(value: object) -> Path:
    """Resolve relative paths against /workspace; pass absolute paths through."""
    text = str(value).strip()
    if not text:
        raise ValueError("Path value must not be empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve(strict=False)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = _resolve_workspace_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


def _encode_cuda_ipc(tensor: torch.Tensor) -> str:
    """Serialize a CUDA tensor's IPC handle to a base64 ASCII string.

    Uses ForkingPickler so torch.multiprocessing.reductions.reduce_tensor is
    invoked, which produces a payload carrying a cudaIpcMemHandle instead of
    the tensor data itself.
    """
    buf = io.BytesIO()
    ForkingPickler(buf).dump(tensor)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class DepthSamFeatNode(Node):
    def __init__(self) -> None:
        super().__init__("depth_sam_feat_node")

        self._declare_parameters()
        self._load_configs()
        self._init_device()

        # vision_lock protects the latest captured frames vs the inference
        # timer. fk_lock protects the FK PoseArray buffer vs assembly.
        self.vision_lock = threading.Lock()
        self.fk_lock = threading.Lock()
        self.latest_color_bgr: np.ndarray | None = None
        self.latest_depth_u16: np.ndarray | None = None

        self._setup_realsense()
        self._setup_sam2p3()
        self._setup_depth_feature()
        self._setup_fk_state()
        self._setup_ipc_buffer()

        # IPC handle is encoded once — depth_feat_policy_node decodes it on
        # its single startup call and then reads the buffer in-place forever.
        self._encoded_handle = _encode_cuda_ipc(self.feature_buffer)

        self._init_vis_publisher()
        self._start_capture_thread()
        self._init_fk_subscription()
        self._init_inference_timer()
        self._init_handle_service()

        self.get_logger().info(
            f"DepthSamFeatNode ready:\n"
            f"  rs        = {self.rs_color_width}x{self.rs_color_height}@{int(self.capture_hz)}Hz "
            f"-> {self.policy_width}x{self.policy_height} (CPU resize)\n"
            f"  inf_rate  = {self.inference_hz}Hz\n"
            f"  sam2p3    = prompt={self._sam_prompt!r} type={self._sam_prompt_type} "
            f"K={self._sam_cycle_K} W={self._sam_warmup_W}\n"
            f"             sam2={Path(self._sam_sam2_ckpt).name}  sam3={Path(self._sam_sam3_ckpt).name}\n"
            f"  clip      = [{self.clip_near}, {self.clip_far}]m\n"
            f"  depth_net = {self._extractor_cfg.backbone}  fk_dim={self.fk_dim}  "
            f"film={self._extractor_cfg.use_film}  out={self.output_dim}\n"
            f"  ckpt_dir  = {self._extractor_cfg.log_dir}\n"
            f"  fk_topic  = {self.fk_topic}  ({self.fk_n_links} links)\n"
            f"  ipc_svc   = {self.ipc_service_name}"
        )

    # ------------------------------------------------------------------
    # init: parameters / configs / device
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        # Workspace-relative defaults; absolute overrides passed via launch /
        # `ros2 run --ros-args` still work because _resolve_workspace_path
        # leaves absolute paths untouched.
        self.declare_parameter("depth_feat_config_path", DEPTH_FEAT_CONFIG_PATH)
        self.declare_parameter("realsense_config_path", REALSENSE_CONFIG_PATH)
        self.declare_parameter("fk_config_path", FK_CONFIG_PATH)
        self.depth_feat_config_path = _resolve_workspace_path(
            self.get_parameter("depth_feat_config_path").value
        )
        self.realsense_config_path = _resolve_workspace_path(
            self.get_parameter("realsense_config_path").value
        )
        self.fk_config_path = _resolve_workspace_path(
            self.get_parameter("fk_config_path").value
        )

    def _load_configs(self) -> None:
        cfg = _load_yaml(self.depth_feat_config_path)
        rs_cfg_full = _load_yaml(self.realsense_config_path)
        fk_cfg = _load_yaml(self.fk_config_path)

        self._cfg = cfg
        self._rs_cfg = rs_cfg_full["realsense"]
        self._fk_cfg = fk_cfg

        # Pre-pull the scalar knobs the rest of __init__ needs.
        feat_cfg = cfg["depth_feature"]
        runtime_cfg = cfg.get("runtime", {})
        disp_cfg = cfg.get("display", {})

        self.clip_near = float(feat_cfg["near"])
        self.clip_far = float(feat_cfg["far"])
        self.inference_hz = float(runtime_cfg.get("inference_rate", 15.0))
        self.publish_masked_depth = bool(disp_cfg.get("publish_masked_depth", True))
        self.masked_depth_topic = str(disp_cfg.get("masked_depth_topic", "/depth_feat_node/depth"))
        self.publish_bottle_poses = bool(disp_cfg.get("publish_bottle_poses", True))
        self.bottle_poses_topic = str(disp_cfg.get("bottle_poses_topic", "/depth_feat_node/bottle_pose"))
        self.publish_bottle_geom = bool(disp_cfg.get("publish_bottle_geom", True))
        self.bottle_geom_topic = str(disp_cfg.get("bottle_geom_topic", "/depth_feat_node/bottle_geom"))
        # fk_cfg defines the world frame that fk_node publishes its PoseArray in.
        # We piggy-back on it so RViz can overlay our debug poses on the FK ones
        # without an extra TF.
        self.world_frame_id = str(fk_cfg.get("world_frame_id", "world"))

        c_intr = self._rs_cfg["color_intrinsics"]
        self.rs_color_width = int(c_intr["width"])
        self.rs_color_height = int(c_intr["height"])
        self.capture_hz = float(self._rs_cfg["rs_fps"])

        # Policy / YOLO / DepthFeatureNet input resolution. RealSense captures
        # at rs_color_width x rs_color_height (640x480 now, so SDK filters get
        # high-res input), then the capture loop CPU-downsamples to this size.
        # Hardcoded to the training input — change only if you retrain.
        self.policy_width = 320
        self.policy_height = 240

    def _init_device(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DepthFeatureExtractor.")
        self.device = torch.device("cuda")

    # ------------------------------------------------------------------
    # RealSense (capture thread + filters)
    # ------------------------------------------------------------------
    def _setup_realsense(self) -> None:
        rs_cfg = self._rs_cfg
        c_intr = rs_cfg["color_intrinsics"]
        d_intr = rs_cfg["depth_intrinsics"]
        sens_set = rs_cfg.get("sensor_settings", {})

        # Filters: decimation(2) -> hole_filling(2). Both run BEFORE align,
        # matching realsense_record.py and collect_policy_realsense.py.
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 2)

        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.color, c_intr["width"], c_intr["height"], rs.format.bgr8, int(self.capture_hz)
        )
        rs_config.enable_stream(
            rs.stream.depth, d_intr["width"], d_intr["height"], rs.format.z16, int(self.capture_hz)
        )
        rs_profile = self.rs_pipeline.start(rs_config)
        self.align = rs.align(rs.stream.color)

        self._apply_sensor_settings(rs_profile, sens_set)

        try:
            self.depth_scale = float(
                rs_profile.get_device().first_depth_sensor().get_depth_scale()
            )
        except Exception:
            self.depth_scale = 0.001
        # Training (collect_policy_realsense.py + train_depth_predictor.py) is
        # hard-coded to "raw depth value * 1e-3 = metres" (depth stored as mm
        # uint16, trainer divides by 1000). Any sensor whose factory depth
        # scale isn't 0.001 would silently break that contract — fail loudly
        # rather than feed mismatched units into the FiLM net.
        if abs(self.depth_scale - 0.001) > 1e-6:
            raise RuntimeError(
                f"RealSense depth_scale={self.depth_scale} != 0.001 m/unit; "
                f"training assumed mm-to-metres conversion. Either re-collect "
                f"data at the new scale or pick a sensor with depth_scale=1e-3."
            )

        self.get_logger().info(
            f"RealSense capture started: {c_intr['width']}x{c_intr['height']}@{int(self.capture_hz)}Hz "
            f"(depth_scale={self.depth_scale})"
        )
        time.sleep(0.5)

    def _apply_sensor_settings(self, rs_profile, sensor_settings: dict[str, Any]) -> None:
        if not sensor_settings:
            return
        auto_exposure = sensor_settings.get("auto_exposure", False)
        exposure = sensor_settings.get("exposure")
        gain = sensor_settings.get("gain")
        emitter_enabled = sensor_settings.get("emitter_enabled", None)
        laser_power = sensor_settings.get("laser_power", None)
        for sensor in rs_profile.get_device().query_sensors():
            sensor_name = sensor.get_info(rs.camera_info.name)
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
            if not auto_exposure:
                if exposure is not None and sensor.supports(rs.option.exposure):
                    sensor.set_option(rs.option.exposure, float(exposure))
                if gain is not None and sensor.supports(rs.option.gain):
                    sensor.set_option(rs.option.gain, float(gain))
            # Emitter / laser power live on the stereo sensor only; supports()
            # filters color sensor automatically. laser_power is clamped to
            # the sensor's reported range so the same yaml value works across
            # D435 (max=360) / D405 (max=100) / L515 (max=200).
            if emitter_enabled is not None and sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, float(emitter_enabled))
                self.get_logger().info(
                    f"[{sensor_name}] emitter_enabled -> "
                    f"{sensor.get_option(rs.option.emitter_enabled)}"
                )
            if laser_power is not None and sensor.supports(rs.option.laser_power):
                lp_range = sensor.get_option_range(rs.option.laser_power)
                clamped = max(lp_range.min, min(float(laser_power), lp_range.max))
                sensor.set_option(rs.option.laser_power, clamped)
                self.get_logger().info(
                    f"[{sensor_name}] laser_power -> "
                    f"{sensor.get_option(rs.option.laser_power)} mW "
                    f"(yaml={laser_power}, range {lp_range.min}..{lp_range.max})"
                )

    def _start_capture_thread(self) -> None:
        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="depth_feat_rs_capture"
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Pull aligned (color, depth) frames at rs_fps; cache under vision_lock."""
        period = 1.0 / max(self.capture_hz, 1e-3)
        while self._capture_running and rclpy.ok():
            loop_start = time.perf_counter()
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError as exc:
                self.get_logger().warn(f"RealSense wait_for_frames timed out: {exc}")
                continue
            except Exception as exc:
                self.get_logger().error(f"RealSense capture error: {exc}")
                self._capture_running = False
                rclpy.shutdown()
                return

            # decimation -> hole_filling -> align (depth -> color frame).
            frames = self.decimation.process(frames).as_frameset()
            frames = self.hole_filling.process(frames).as_frameset()
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # SDK gave us frames at capture resolution (e.g. 640x480) — that
            # high pixel count was needed by decimation / hole_filling / align
            # above. Now downsample on CPU to the policy input size (320x240).
            color_hires = np.asanyarray(color_frame.get_data())
            depth_hires = np.asanyarray(depth_frame.get_data())
            if (color_hires.shape[0] != self.policy_height or
                    color_hires.shape[1] != self.policy_width):
                # Different interpolations per modality:
                #   * color: INTER_AREA — local-average is the right thing
                #     for RGB downsampling (clean low-res, no aliasing).
                #   * depth: INTER_NEAREST — DO NOT average depth across
                #     edges; mean creates phantom "in-between" depth at
                #     object boundaries (a thin ring of mid-distance pixels
                #     between near and far). Nearest matches the spirit of
                #     RealSense's SDK decimation_filter (which uses non-zero
                #     median, not mean) so behaviour is consistent with the
                #     training-data collection pipeline.
                color_bgr = cv2.resize(
                    color_hires, (self.policy_width, self.policy_height),
                    interpolation=cv2.INTER_AREA,
                )
                depth_u16 = cv2.resize(
                    depth_hires, (self.policy_width, self.policy_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                # Already at policy resolution — just .copy() to detach from
                # the RealSense-owned buffer (the SDK reuses it next frame).
                color_bgr = color_hires.copy()
                depth_u16 = depth_hires.copy()

            with self.vision_lock:
                self.latest_color_bgr = color_bgr
                self.latest_depth_u16 = depth_u16

            elapsed = time.perf_counter() - loop_start
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    # ------------------------------------------------------------------
    # SAM3-init + SAM2 dual-bank tracker + DepthFeatureExtractor
    # ------------------------------------------------------------------
    def _setup_sam2p3(self) -> None:
        """Build SAM2P3DualBankTracker; lazy-init (tracker.init()) is deferred
        to the first _inference_step that has a captured frame."""
        sam_cfg = self._cfg["sam2p3"]
        # ckpt paths are workspace-relative ("models/sam2/..."); sam2_cfg is
        # the SAM2 hydra config name (resolved internally by the sam2 package,
        # NOT a filesystem path) — pass it through untouched.
        self._sam_sam2_ckpt = str(_resolve_workspace_path(sam_cfg["sam2_ckpt"]))
        self._sam_sam3_ckpt = str(_resolve_workspace_path(sam_cfg["sam3_ckpt"]))
        self._sam_sam2_cfg  = str(sam_cfg["sam2_cfg"])
        self._sam_prompt        = str(sam_cfg["prompt"])
        self._sam_prompt_type   = str(sam_cfg["prompt_type"])
        self._sam_cycle_K       = int(sam_cfg["cycle_K"])
        self._sam_warmup_W      = int(sam_cfg["warmup_W"])
        self._sam_sam3_conf     = float(sam_cfg["sam3_conf"])
        self._sam_offload_video = bool(sam_cfg["offload_video_to_cpu"])

        # Builds SAM3 + SAM2 immediately (loads ~3 GB of weights to GPU).
        # tracker.init(first_frame) is deferred until we have a real RealSense
        # frame to feed it (see _inference_step's lazy-init branch).
        self.tracker = SAM2P3DualBankTracker(
            sam2_ckpt=self._sam_sam2_ckpt,
            sam2_cfg=self._sam_sam2_cfg,
            sam3_ckpt=self._sam_sam3_ckpt,
            prompt=self._sam_prompt,
            prompt_type=self._sam_prompt_type,
            cycle_K=self._sam_cycle_K,
            warmup_W=self._sam_warmup_W,
            device="cuda",
            offload_video_to_cpu=self._sam_offload_video,
            sam3_conf=self._sam_sam3_conf,
            verbose=False,                   # don't spam the ROS log on every spawn
        )
        self._tracker_initialized = False    # tracker.init() runs on first frame

    def _setup_depth_feature(self) -> None:
        feat_cfg = self._cfg["depth_feature"]
        # log_dir is workspace-relative by default; absolute host paths pass
        # through _resolve_workspace_path untouched. checkpoint_path follows
        # the same resolution rules; None / empty stays None so the extractor
        # falls back to scanning log_dir.
        log_dir = str(_resolve_workspace_path(feat_cfg["log_dir"]))
        ckpt_path_raw = feat_cfg.get("checkpoint_path", None)
        checkpoint_path = (
            str(_resolve_workspace_path(ckpt_path_raw)) if ckpt_path_raw else None
        )
        self._extractor_cfg = DepthFeatureExtractorCfg(
            enable=True,
            train=False,
            load_checkpoint=bool(feat_cfg.get("load_checkpoint", True)),
            backbone=str(feat_cfg["backbone"]),
            num_keypoints=int(feat_cfg["num_keypoints"]),
            mlp_hidden=int(feat_cfg["mlp_hidden"]),
            near=self.clip_near,
            far=self.clip_far,
            log_dir=log_dir,
            checkpoint_path=checkpoint_path,
            use_film=bool(feat_cfg.get("use_film", True)),
            fk_dim=int(feat_cfg["fk_dim"]),
            film_hidden=int(feat_cfg["film_hidden"]),
            fk_embed_dim=int(feat_cfg["fk_embed_dim"]),
        )
        self.output_dim = int(feat_cfg["output_dim"])
        self.fk_dim = int(feat_cfg["fk_dim"])
        self.extractor = DepthFeatureExtractor(
            self._extractor_cfg, output_dim=self.output_dim, device=self.device
        )
        
        # # Cache LSQ W/b and geom override on the inference device so the hot
        # # path is pure tensor math.
        # self._lsq_W = torch.from_numpy(_LSQ_W).to(self.device)              # (6, 6)
        # self._lsq_B = torch.from_numpy(_LSQ_B).to(self.device)              # (6,)
        # self._geom_override = torch.from_numpy(_GEOM_OVERRIDE).to(self.device)  # (4,)

    def _setup_fk_state(self) -> None:
        fk_sel = self._cfg["fk"]
        self.fk_link_names: list[str] = list(self._fk_cfg["link_names"])
        self.fk_topic: str = str(self._fk_cfg["fk_topic"])
        self.fk_n_links = len(self.fk_link_names)
        self.left_hand_base_names = list(fk_sel["left_hand_base_links"])
        self.right_hand_base_names = list(fk_sel["right_hand_base_links"])
        self.left_fingertip_names = list(fk_sel["left_fingertip_links"])
        self.right_fingertip_names = list(fk_sel["right_fingertip_links"])

        # Sanity: cfg fk_dim must match the link-selection sizes.
        n_left_base = 3 * len(self.left_hand_base_names)
        n_right_base = 3 * len(self.right_hand_base_names)
        n_left_tips = 3 * len(self.left_fingertip_names)
        n_right_tips = 3 * len(self.right_fingertip_names)
        derived_fk_dim = n_left_base + n_right_base + n_left_tips + n_right_tips
        if derived_fk_dim != self.fk_dim:
            raise ValueError(
                f"fk_dim mismatch: depth_feature.fk_dim={self.fk_dim} vs "
                f"link-derived={derived_fk_dim} "
                f"(left_base={n_left_base}, right_base={n_right_base}, "
                f"left_tips={n_left_tips}, right_tips={n_right_tips})"
            )

        self.fk_pose_dict: dict[str, np.ndarray] = {
            name + "_pos": np.zeros(3, dtype=np.float32) for name in self.fk_link_names
        }
        self.has_fk = False
        # Pre-allocated CPU buffer for FK vector assembly each step.
        self._fk_cpu = np.zeros(self.fk_dim, dtype=np.float32)
        self._fk_gpu = torch.zeros((1, self.fk_dim), dtype=torch.float32, device=self.device)

    def _setup_ipc_buffer(self) -> None:
        ipc_cfg = self._cfg["ipc"]
        # Persistent (1, output_dim) float CUDA tensor — depth_feat_policy_node
        # attaches a view onto this same memory at startup, so every copy_
        # below is observable on the consumer side. The handle is encoded once
        # in __init__ after this method returns.
        self.feature_buffer: torch.Tensor = torch.zeros(
            (1, self.output_dim), dtype=torch.float32, device=self.device
        )
        self.ipc_service_name = str(ipc_cfg["handle_service_name"])

    # ------------------------------------------------------------------
    # Visualization publisher
    # ------------------------------------------------------------------
    def _init_vis_publisher(self) -> None:
        """Set up the debug publishers: masked depth (Image) + bottle pred (PoseArray).

        Each publisher is gated by its own cfg toggle; when off, the field
        is set to None and the inference loop short-circuits its publish
        branch. Masked-depth msg construction mirrors aux_policy_v2.py's
        /depth_vis pattern (no cv_bridge dep, hand-built from a contiguous
        numpy buffer). The bottle PoseArray uses fk_cfg's world frame as
        header.frame_id so it overlays cleanly on the FK PoseArray in RViz.
        """
        if self.publish_masked_depth:
            self.depth_vis_pub = self.create_publisher(
                Image, self.masked_depth_topic, HIGH_RELIA_QOS
            )
        else:
            self.depth_vis_pub = None

        if self.publish_bottle_poses:
            self.bottle_poses_pub = self.create_publisher(
                PoseArray, self.bottle_poses_topic, HIGH_RELIA_QOS
            )
        else:
            self.bottle_poses_pub = None

        if self.publish_bottle_geom:
            self.bottle_geom_pub = self.create_publisher(
                Float32MultiArray, self.bottle_geom_topic, HIGH_RELIA_QOS
            )
        else:
            self.bottle_geom_pub = None

    # ------------------------------------------------------------------
    # FK PoseArray subscriber
    # ------------------------------------------------------------------
    def _init_fk_subscription(self) -> None:
        self.create_subscription(
            PoseArray, self.fk_topic, self._sub_fk_cb, HIGH_PERF_QOS
        )

    def _sub_fk_cb(self, msg: PoseArray) -> None:
        """Write the inbound PoseArray into fk_pose_dict in fk_cfg.link_names order."""
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
            self.has_fk = True

    def _assemble_fk_vector(self) -> torch.Tensor | None:
        """Concat [left_base, right_base, left_tips, right_tips] -> (1, fk_dim) GPU.

        Returns None until at least one FK PoseArray has been received.
        """
        if not self.has_fk:
            return None
        with self.fk_lock:
            offset = 0
            for name in self.left_hand_base_names:
                self._fk_cpu[offset:offset + 3] = self.fk_pose_dict[name + "_pos"]
                offset += 3
            for name in self.right_hand_base_names:
                self._fk_cpu[offset:offset + 3] = self.fk_pose_dict[name + "_pos"]
                offset += 3
            for name in self.left_fingertip_names:
                self._fk_cpu[offset:offset + 3] = self.fk_pose_dict[name + "_pos"]
                offset += 3
            for name in self.right_fingertip_names:
                self._fk_cpu[offset:offset + 3] = self.fk_pose_dict[name + "_pos"]
                offset += 3
        self._fk_gpu.copy_(
            torch.from_numpy(self._fk_cpu).unsqueeze(0), non_blocking=True
        )
        return self._fk_gpu

    # ------------------------------------------------------------------
    # Inference timer
    # ------------------------------------------------------------------
    def _init_inference_timer(self) -> None:
        self.inference_timer = self.create_timer(
            1.0 / max(self.inference_hz, 1e-3), self._inference_step
        )

    def _inference_step(self) -> None:
        loop_start = time.perf_counter()
        with self.vision_lock:
            color_bgr = self.latest_color_bgr.copy() if self.latest_color_bgr is not None else None
            depth_u16 = self.latest_depth_u16.copy() if self.latest_depth_u16 is not None else None
        if color_bgr is None or depth_u16 is None:
            return  # capture thread hasn't produced a frame yet

        # ── Lazy init on the first real frame ────────────────────────────────
        # SAM3 bootstrap (warmup + init) needs a real image to seed the cond
        # frame, which we only have once the capture thread has produced one.
        # If SAM3 misses the target on the bootstrap frame we retry on the
        # next tick; we never call tracker.step() before init() succeeds.
        if not self._tracker_initialized:
            try:
                self.tracker.warmup(color_bgr, n=3)
                self.tracker.init(color_bgr)
                self._tracker_initialized = True
                self.get_logger().info(
                    f"[tracker] initialized on first frame (prompt={self._sam_prompt!r})"
                )
            except RuntimeError as exc:
                self.get_logger().warn(
                    f"[tracker] init failed (SAM3 missed target?), retrying: {exc}"
                )
            return  # always skip this step; pipeline starts on the next call

        # ── SAM2P3 streaming step → (H, W) bool mask ─────────────────────────
        try:
            mask = self.tracker.step(color_bgr)
        except Exception as exc:
            self.get_logger().error(f"tracker.step failed: {exc}")
            return

        # mask-out the depth: pixels outside the mask -> 0 (sentinel for
        # "no signal"), pixels inside keep their raw uint16 sensor units.
        masked_depth = np.where(mask, depth_u16, 0).astype(np.uint16)

        if not mask.any():
            # Tracker produced an empty mask this frame — likely full
            # occlusion or the bottle left the FOV.  Same fallback as the
            # YOLO branch: leave the previous good feature in the IPC
            # buffer so the policy isn't blindsided by a meaningless pred.
            self.get_logger().warn(
                "tracker produced empty mask this frame — skipping depth-feature "
                "update; policy continues to consume the last good feature."
            )
            if self.depth_vis_pub is not None:
                self._publish_masked_depth(masked_depth)
            return

        fk_t = self._assemble_fk_vector()
        if fk_t is None:
            self.get_logger().warn(
                f"No FK PoseArray received yet on '{self.fk_topic}' — "
                f"skipping depth-feature update. Is fk_node running?"
            )
            if self.depth_vis_pub is not None:
                self._publish_masked_depth(masked_depth)
            return

        try:
            # masked_depth (uint16, sensor-native units) -> metres tensor on GPU.
            # DepthFeatureExtractor expects (B, H, W) or (B, 1, H, W) input and
            # handles its own [near, far] clip + normalisation. depth_scale
            # converts raw sensor values to metres in the same convention
            # training used (collect_policy_realsense.py stores depth in mm
            # uint16; trainer divides by 1000 to metres).
            depth_t = (
                torch.from_numpy(masked_depth).to(self.device).float() * self.depth_scale
            ).unsqueeze(0)   # (1, H, W) float32 metres

            # target=None -> wrapper returns (feature, pred, None) and skips
            # the loss/optimizer path entirely.
            _feature, pred, _loss = self.extractor.step(
                raw_depth=depth_t, target=None, fk=fk_t
            )

            # pred[..., 0:6] is metres (positions); pred[..., 6:10] is in [0,1].
            # Un-normalize the geom slab so the IPC buffer is uniformly in metres.
            pred_metres = pred.clone()
            pred_metres[..., 6:10] = self.extractor.un_preprocess_geom(pred[..., 6:10])

            # # (a) LSQ-compensate body+cap (6-d affine fitted from real apriltag data).
            # pred_metres[..., 0:6] = pred_metres[..., 0:6] @ self._lsq_W + self._lsq_B

            # # (b) Override the geom slab — predictor's geom branch is too noisy in real.
            # # Broadcast (4,) into (1, 4) row. If pred has batch > 1 this also broadcasts.
            # pred_metres[..., 6:10] = self._geom_override
            # In-place into the IPC-shared buffer; consumers read this same
            # GPU memory.
            self.feature_buffer.copy_(pred_metres, non_blocking=True)
        except Exception as exc:
            self.get_logger().error(f"DepthFeatureExtractor inference failed: {exc}")
            rclpy.shutdown()
            return

        if self.depth_vis_pub is not None:
            self._publish_masked_depth(masked_depth)
        if self.bottle_poses_pub is not None:
            self._publish_bottle_poses(pred_metres)
        if self.bottle_geom_pub is not None:
            self._publish_bottle_geom(pred_metres)

        elapsed = time.perf_counter() - loop_start
        period = 1.0 / max(self.inference_hz, 1e-3)
        if elapsed > period:
            self.get_logger().warn(
                f"[SLOW_INFER] total={elapsed:.4f}s, target_period={period:.4f}s"
            )

    def _publish_masked_depth(self, masked_depth: np.ndarray) -> None:
        """Colormap the masked depth and publish it as a BGR8 sensor_msgs/Image.

        masked depth (uint16, sensor units) -> metres -> clip [near, far]
        -> uint8 -> INFERNO colormap -> Image msg on self.masked_depth_topic.
        Out-of-mask / no-detection pixels (depth <= 0) are forced to black so
        they read as "no signal" in the viewer instead of mapping to colormap's
        low end.
        """
        if self.depth_vis_pub is None:
            return
        d_m = masked_depth.astype(np.float32) * self.depth_scale
        invalid = d_m <= 0
        norm = np.clip(
            (d_m - self.clip_near) / (self.clip_far - self.clip_near + 1e-6), 0, 1
        )
        depth_u8 = (norm * 255).astype(np.uint8)
        depth_u8[invalid] = 0
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        depth_color[invalid] = 0
        depth_color = np.ascontiguousarray(depth_color)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = depth_color.shape[0]
        msg.width = depth_color.shape[1]
        msg.encoding = "bgr8"
        msg.step = depth_color.shape[1] * 3
        msg.data = depth_color.tobytes()
        self.depth_vis_pub.publish(msg)

    def _publish_bottle_poses(self, pred_metres: torch.Tensor) -> None:
        """Publish the network's predicted body/cap positions as a PoseArray.

        Layout matches the IPC buffer: poses[0] = bottleBodyPos,
        poses[1] = bottleCapPos. Orientation is left identity — the policy
        derives 6D rot downstream from (cap - body) via _rot6d_from_axis;
        here we just want the two points in RViz, optionally overlaid on
        fk_node's PoseArray (same world_frame_id).

        Geom (pred_metres[..., 6:10]) isn't a pose, so it's not in this msg;
        if you need to inspect it, attach to the IPC tensor or add a
        std_msgs/Float32MultiArray side channel.
        """
        if self.bottle_poses_pub is None:
            return
        # One D->H copy of 6 floats is cheap; pulling individual elements is
        # less code than slicing into a pre-pinned CPU buffer.
        pts = pred_metres[0, :6].detach().cpu().numpy()
        body = Pose()
        body.position.x = float(pts[0])
        body.position.y = float(pts[1])
        body.position.z = float(pts[2])
        body.orientation.w = 1.0   # identity quaternion (xyz default to 0)
        cap = Pose()
        cap.position.x = float(pts[3])
        cap.position.y = float(pts[4])
        cap.position.z = float(pts[5])
        cap.orientation.w = 1.0

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame_id
        msg.poses = [body, cap]
        self.bottle_poses_pub.publish(msg)

    def _publish_bottle_geom(self, pred_metres: torch.Tensor) -> None:
        """Publish the 4 geom dims as Float32MultiArray.

        Data order matches the IPC tensor and observations.py::bottleGeomCfg:
            [body_radius, body_height, cap_radius, cap_height]  (metres)

        Float32MultiArray has no header.stamp; if you need synchronization
        with the pose/depth streams, correlate by the inference-step cadence
        (all three publishers fire in the same _inference_step).
        """
        if self.bottle_geom_pub is None:
            return
        geom = pred_metres[0, 6:10].detach().cpu().numpy().astype(np.float32)
        msg = Float32MultiArray()
        msg.data = geom.tolist()
        self.bottle_geom_pub.publish(msg)

    # ------------------------------------------------------------------
    # IPC handle service
    # ------------------------------------------------------------------
    def _init_handle_service(self) -> None:
        self.handle_srv = self.create_service(
            Trigger, self.ipc_service_name, self._on_get_handle
        )

    def _on_get_handle(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        response.success = True
        response.message = self._encoded_handle
        self.get_logger().info("Served depth-feature IPC handle to a client.")
        return response

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def destroy_node(self) -> bool:
        self._capture_running = False
        if hasattr(self, "inference_timer") and self.inference_timer is not None:
            self.inference_timer.cancel()
        if getattr(self, "_capture_thread", None) is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if hasattr(self, "rs_pipeline") and self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except Exception as exc:
                self.get_logger().error(f"Error stopping RealSense pipeline: {exc}")
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthSamFeatNode()
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
