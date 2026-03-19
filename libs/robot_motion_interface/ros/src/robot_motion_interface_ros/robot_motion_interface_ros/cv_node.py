import os
import threading
import time
import importlib.util
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import torch
import yaml
from rclpy.node import Node

spec = importlib.util.find_spec("robot_motion_interface")
if spec is None or spec.origin is None:
    raise RuntimeError("Cannot locate robot_motion_interface")
RMI_ROOT = Path(spec.origin).parent.parent.parent
DEFAULT_CONFIG_PATH = RMI_ROOT / "config" / "rl_policy_node_config.yaml"

MODEL_TIMER_HZ = 30.0
COMPOSITOR_HZ = 30.0
SAM2_WARMUP_TIMEOUT_SEC = 5.0


class CVPerceptionNode(Node):
    def __init__(self):
        super().__init__("cv_perception_node")

        self.declare_parameter("config_path", str(DEFAULT_CONFIG_PATH.resolve()))
        config_path: str = self.get_parameter("config_path").value
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._node_verbose = bool(config.get("cv_verbose", False))
        self._log_info(f"Loaded config from: {config_path}")

        self._cv_model_cfg = config.get("cv_model", {})
        required_models = ["promptda", "sam2_w_prompt", "da3", "ultralytics"]
        self._model_enabled: dict[str, bool] = {}
        self._model_verbose: dict[str, bool] = {}
        for model_name in required_models:
            model_cfg = self._cv_model_cfg.get(model_name)
            if not isinstance(model_cfg, dict) or "enable" not in model_cfg:
                raise KeyError(
                    f"cv_model.{model_name}.enable is required; 'enbale' is no longer supported"
                )
            self._model_enabled[model_name] = bool(model_cfg.get("enable", False))
            self._model_verbose[model_name] = bool(model_cfg.get("verbose", False))

        # ── RealSense init ──────────────────────────────────────────────────
        rs_cfg = config["realsense"]
        rs_fps = int(rs_cfg["rs_fps"])
        sensor_settings = rs_cfg.get("sensor_settings", {})
        c_intrinsics = rs_cfg["color_intrinsics"]
        d_intrinsics = rs_cfg["depth_intrinsics"]

        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color,
            int(c_intrinsics["width"]),
            int(c_intrinsics["height"]),
            rs.format.bgr8,
            rs_fps,
        )
        self.rs_config.enable_stream(
            rs.stream.depth,
            int(d_intrinsics["width"]),
            int(d_intrinsics["height"]),
            rs.format.z16,
            rs_fps,
        )

        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        self.rs_align = rs.align(rs.stream.color)
        self._apply_sensor_settings(sensor_settings)

        try:
            depth_sensor = self.rs_profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
        except Exception:
            self._depth_scale = 0.001

        self._depth_filters = self.build_rs_depth_filters()
        rs_device = self.rs_profile.get_device()
        self._log_info(
            "RealSense initialized: "
            f"device={rs_device.get_info(rs.camera_info.name)} "
            f"serial={rs_device.get_info(rs.camera_info.serial_number)} "
            f"color={c_intrinsics['width']}x{c_intrinsics['height']}@{rs_fps} "
            f"depth={d_intrinsics['width']}x{d_intrinsics['height']}@{rs_fps} "
            f"depth_scale={self._depth_scale}"
        )

        # ── Shared frame buffers ─────────────────────────────────────────────
        self._latest_color: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._img_lock = threading.Lock()

        # ── Model panel buffers ──────────────────────────────────────────────
        self._panel_lock = threading.Lock()
        self._model_panels: dict[str, np.ndarray | None] = {
            "promptda": None,
            "da3": None,
            "sam2_w_prompt": None,
            "ultralytics": None,
        }
        self._busy_lock = threading.Lock()
        self._model_busy: dict[str, bool] = {
            "promptda": False,
            "da3": False,
            "sam2_w_prompt": False,
            "ultralytics": False,
        }

        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="rs_capture",
            daemon=True,
        )
        self._capture_thread.start()

        # ── Model init ───────────────────────────────────────────────────────
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log_info(f"CV device: {self.device}")

        self.promptda: Any | None = None
        self.da3: Any | None = None
        self.sam2_tracker: Any | None = None
        self.ultralytics: Any | None = None
        self._sam2_draw_results = None
        self._sam2_ready = False

        if self._model_enabled["promptda"]:
            self._init_promptda_model()
        if self._model_enabled["da3"]:
            self._init_da3_model(c_intrinsics)
        if self._model_enabled["sam2_w_prompt"]:
            self._init_sam2_model()
        if self._model_enabled["ultralytics"]:
            self._init_ultralytics_model()

        # ── Warmup/init (startup only) ──────────────────────────────────────
        self._warmup_models()

        # ── Timers (all 30 Hz) ──────────────────────────────────────────────
        self._timers = []
        if self._model_enabled["promptda"] and self.promptda is not None:
            self._timers.append(self.create_timer(1.0 / MODEL_TIMER_HZ, self._promptda_timer_cb))
        if self._model_enabled["da3"] and self.da3 is not None:
            self._timers.append(self.create_timer(1.0 / MODEL_TIMER_HZ, self._da3_timer_cb))
        if self._model_enabled["sam2_w_prompt"] and self.sam2_tracker is not None:
            self._timers.append(self.create_timer(1.0 / MODEL_TIMER_HZ, self._sam2_timer_cb))
        if self._model_enabled["ultralytics"] and self.ultralytics is not None:
            self._timers.append(self.create_timer(1.0 / MODEL_TIMER_HZ, self._ultralytics_timer_cb))

        self._timers.append(self.create_timer(1.0 / COMPOSITOR_HZ, self._compositor_timer_cb))
        self._log_info(
            "CV node ready: "
            f"promptda={self._model_enabled['promptda']} "
            f"da3={self._model_enabled['da3']} "
            f"sam2_w_prompt={self._model_enabled['sam2_w_prompt']} "
            f"ultralytics={self._model_enabled['ultralytics']} "
            f"model_hz={MODEL_TIMER_HZ} compositor_hz={COMPOSITOR_HZ}"
        )

    def _log_info(self, msg: str) -> None:
        if self._node_verbose:
            self.get_logger().info(msg)

    @contextmanager
    def _maybe_silence_model_output(self, model_name: str):
        if self._model_verbose.get(model_name, False):
            with nullcontext():
                yield
            return
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield

    # ── Model init helpers ──────────────────────────────────────────────────

    def _init_promptda_model(self) -> None:
        from robot_motion_interface.utils.promptda_utils import (
            DEFAULT_PROMPTDA_ENCODER,
            PromptDAInference,
        )

        cfg = self._cv_model_cfg["promptda"]
        ckpt = cfg.get("checkpoint")
        encoder = str(cfg.get("encoder", DEFAULT_PROMPTDA_ENCODER))
        with self._maybe_silence_model_output("promptda"):
            self.promptda = PromptDAInference(
                ckpt_path=ckpt,
                encoder=encoder,
                device=self.device,
                depth_scale=self._depth_scale,
            )
        self._log_info(
            f"PromptDA enabled: ckpt={self.promptda.ckpt_path} encoder={self.promptda.encoder}"
        )

    def _init_da3_model(self, color_intrinsics: dict[str, Any]) -> None:
        from robot_motion_interface.utils.da3_utils import DA3Inference, DEFAULT_DA3_MODEL

        cfg = self._cv_model_cfg["da3"]
        model_name = str(cfg.get("model") or DEFAULT_DA3_MODEL)
        process_res = int(cfg.get("process_res", 504))
        fx = float(color_intrinsics["fx"])
        fy = float(color_intrinsics["fy"])
        focal = 0.5 * (fx + fy)
        with self._maybe_silence_model_output("da3"):
            self.da3 = DA3Inference(
                model=model_name,
                focal=focal,
                device=self.device,
                process_res=process_res,
            )
        self._log_info(
            f"DA3 enabled: model={model_name} process_res={process_res} focal={focal:.3f}"
        )

    def _init_sam2_model(self) -> None:
        from robot_motion_interface.utils.sam2_w_prompt import (
            DEFAULT_SAM2_CFG,
            DEFAULT_SAM2_CKPT,
            DEFAULT_SAM3_CKPT,
            SAM2WithPrompt,
            draw_results,
        )

        cfg = self._cv_model_cfg["sam2_w_prompt"]
        sam3_ckpt = str(cfg.get("sam3_ckpt") or DEFAULT_SAM3_CKPT)
        sam2_ckpt = str(cfg.get("sam2_ckpt") or DEFAULT_SAM2_CKPT)
        sam2_cfg = str(cfg.get("sam2_cfg") or DEFAULT_SAM2_CFG)
        compile_flag = bool(cfg.get("compile", True))

        with self._maybe_silence_model_output("sam2_w_prompt"):
            self.sam2_tracker = SAM2WithPrompt(
                sam3_ckpt=sam3_ckpt,
                sam2_ckpt=sam2_ckpt,
                sam2_cfg=sam2_cfg,
                device=self.device,
                compile=compile_flag,
                verbose=self._model_verbose["sam2_w_prompt"],
            )
        self._sam2_draw_results = draw_results
        self._log_info(
            f"SAM2_w_prompt enabled: sam3_ckpt={sam3_ckpt} sam2_ckpt={sam2_ckpt} "
            f"sam2_cfg={sam2_cfg} compile={compile_flag}"
        )

    def _init_ultralytics_model(self) -> None:
        from robot_motion_interface.utils.ultralytics_utils import (
            DEFAULT_ULTRALYTICS_MODEL,
            UltralyticsTracker,
        )

        cfg = self._cv_model_cfg["ultralytics"]
        model_path = str(cfg.get("model_path") or DEFAULT_ULTRALYTICS_MODEL)
        classes = self._parse_ultralytics_classes(cfg.get("classes"))
        conf = float(cfg.get("conf", 0.3))
        iou = float(cfg.get("iou", 0.5))

        with self._maybe_silence_model_output("ultralytics"):
            self.ultralytics = UltralyticsTracker(
                model_path=model_path,
                classes=classes,
                device=self.device,
                conf=conf,
                iou=iou,
            )
        self._log_info(
            f"Ultralytics enabled: model={model_path} classes={classes} conf={conf} iou={iou}"
        )

    def _parse_ultralytics_classes(self, raw: Any) -> list[str] | None:
        if raw is None:
            return None
        if isinstance(raw, list):
            cls = [str(x).strip() for x in raw if str(x).strip()]
            return cls or None
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            if "." in text:
                cls = [x.strip().rstrip(".") for x in text.split(".") if x.strip()]
                return cls or None
            cls = [x.strip() for x in text.split(",") if x.strip()]
            return cls or None
        return None

    # ── Startup warmup ──────────────────────────────────────────────────────

    def _warmup_models(self) -> None:
        if not self._model_enabled.get("sam2_w_prompt", False) or self.sam2_tracker is None:
            return

        first_color = self._wait_for_first_color(timeout_sec=SAM2_WARMUP_TIMEOUT_SEC)
        if first_color is None:
            self.get_logger().warn(
                f"SAM2 warmup skipped: no frame within {SAM2_WARMUP_TIMEOUT_SEC:.1f}s"
            )
            return

        try:
            self._log_info("SAM2 warmup start...")
            with self._maybe_silence_model_output("sam2_w_prompt"):
                self.sam2_tracker.warmup(first_color, n=3)
                init_objects = self.sam2_tracker.init_prompt(first_color)
            if len(init_objects) == 0:
                self._sam2_ready = False
                self.get_logger().warn(
                    "SAM2 warmup done but init_prompt found 0 objects; disabling SAM2 tracking path."
                )
            else:
                self._sam2_ready = True
                self._log_info(
                    f"SAM2 warmup done; init objects={len(init_objects)}"
                )
        except Exception as exc:
            self._sam2_ready = False
            self.get_logger().error(f"SAM2 warmup/init failed: {exc}")

    def _wait_for_first_color(self, timeout_sec: float) -> np.ndarray | None:
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            with self._img_lock:
                color = None if self._latest_color is None else self._latest_color.copy()
            if color is not None:
                return color
            time.sleep(0.01)
        return None

    # ── Capture thread ──────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        while self._capture_running:
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue

            frames = self.apply_depth_filters(frames, self._depth_filters)
            aligned_frames = self.rs_align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            with self._img_lock:
                self._latest_color = color
                self._latest_depth = depth
            time.sleep(0.001)

    # ── Timer callbacks ─────────────────────────────────────────────────────

    def _promptda_timer_cb(self) -> None:
        if self.promptda is None or not self._try_enter_model("promptda"):
            return
        try:
            color, depth = self._get_latest_frames()
            if color is None or depth is None:
                return
            with self._maybe_silence_model_output("promptda"):
                metric_depth = self.promptda.infer(color, depth)
            if isinstance(metric_depth, torch.Tensor):
                metric_depth_np = metric_depth.squeeze().detach().cpu().numpy()
            else:
                metric_depth_np = np.asarray(metric_depth).squeeze()
            panel = self._depth_to_colormap(metric_depth_np, color.shape[1], color.shape[0])
            self._set_model_panel("promptda", panel)
        except Exception as exc:
            self.get_logger().warn(f"PromptDA callback failed: {exc}")
        finally:
            self._leave_model("promptda")

    def _da3_timer_cb(self) -> None:
        if self.da3 is None or not self._try_enter_model("da3"):
            return
        try:
            color, _ = self._get_latest_frames()
            if color is None:
                return
            with self._maybe_silence_model_output("da3"):
                da3_depth = self.da3.infer(color)
            panel = self._depth_to_colormap(np.asarray(da3_depth), color.shape[1], color.shape[0])
            self._set_model_panel("da3", panel)
        except Exception as exc:
            self.get_logger().warn(f"DA3 callback failed: {exc}")
        finally:
            self._leave_model("da3")

    def _sam2_timer_cb(self) -> None:
        if self.sam2_tracker is None or not self._sam2_ready:
            return
        if not self._try_enter_model("sam2_w_prompt"):
            return
        try:
            color, _ = self._get_latest_frames()
            if color is None:
                return
            with self._maybe_silence_model_output("sam2_w_prompt"):
                results = self.sam2_tracker.sam2_infer(color)
            if self._sam2_draw_results is None:
                return
            panel = self._sam2_draw_results(color, results)
            self._set_model_panel("sam2_w_prompt", panel)
        except Exception as exc:
            self.get_logger().warn(f"SAM2_w_prompt callback failed: {exc}")
        finally:
            self._leave_model("sam2_w_prompt")

    def _ultralytics_timer_cb(self) -> None:
        if self.ultralytics is None or not self._try_enter_model("ultralytics"):
            return
        try:
            color, _ = self._get_latest_frames()
            if color is None:
                return
            with self._maybe_silence_model_output("ultralytics"):
                detections = self.ultralytics.infer(color)
            panel = self._draw_ultralytics_overlay(color, detections)
            self._set_model_panel("ultralytics", panel)
        except Exception as exc:
            self.get_logger().warn(f"Ultralytics callback failed: {exc}")
        finally:
            self._leave_model("ultralytics")

    def _compositor_timer_cb(self) -> None:
        color, depth = self._get_latest_frames()
        if color is None or depth is None:
            return

        h, w = color.shape[:2]
        color_tile = self._label_tile(color.copy(), "RGB")
        depth_m = depth.astype(np.float32) * self._depth_scale
        depth_tile = self._label_tile(self._depth_to_colormap(depth_m, w, h), "Depth")

        with self._panel_lock:
            promptda = None if self._model_panels["promptda"] is None else self._model_panels["promptda"].copy()
            da3 = None if self._model_panels["da3"] is None else self._model_panels["da3"].copy()
            sam2p = None if self._model_panels["sam2_w_prompt"] is None else self._model_panels["sam2_w_prompt"].copy()
            ultra = None if self._model_panels["ultralytics"] is None else self._model_panels["ultralytics"].copy()

        p_tile = self._tile_or_black(promptda, w, h, "PromptDA")
        d_tile = self._tile_or_black(da3, w, h, "DA3")
        s_tile = self._tile_or_black(sam2p, w, h, "SAM2_w_prompt")
        u_tile = self._tile_or_black(ultra, w, h, "Ultralytics")

        row1 = np.concatenate([color_tile, depth_tile, p_tile], axis=1)
        row2 = np.concatenate([d_tile, s_tile, u_tile], axis=1)
        preview = np.concatenate([row1, row2], axis=0)

        cv2.imshow("CV 2x3 Panel", preview)
        cv2.waitKey(1)

    # ── Buffer helpers ──────────────────────────────────────────────────────

    def _try_enter_model(self, model_name: str) -> bool:
        with self._busy_lock:
            if self._model_busy[model_name]:
                return False
            self._model_busy[model_name] = True
            return True

    def _leave_model(self, model_name: str) -> None:
        with self._busy_lock:
            self._model_busy[model_name] = False

    def _get_latest_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._img_lock:
            color = None if self._latest_color is None else self._latest_color.copy()
            depth = None if self._latest_depth is None else self._latest_depth.copy()
        return color, depth

    def _set_model_panel(self, model_name: str, panel_bgr: np.ndarray) -> None:
        with self._panel_lock:
            self._model_panels[model_name] = panel_bgr

    def _tile_or_black(self, panel: np.ndarray | None, width: int, height: int, label: str) -> np.ndarray:
        if panel is None:
            tile = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            tile = panel
            if tile.shape[:2] != (height, width):
                tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_LINEAR)
            if tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        return self._label_tile(tile, label)

    def _depth_to_colormap(self, depth_f32: np.ndarray, width: int, height: int) -> np.ndarray:
        depth = np.nan_to_num(depth_f32.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)

        positive = depth[depth > 0.0]
        if positive.size > 0:
            lo, hi = np.percentile(positive, (2, 98))
            if hi <= lo:
                hi = lo + 1e-6
        else:
            lo, hi = 0.0, 1.0

        depth_u8 = np.clip((depth - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        depth_u8 = (depth_u8 * 255.0).astype(np.uint8)
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

    def _label_tile(self, tile: np.ndarray, text: str) -> np.ndarray:
        out = tile.copy()
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _draw_ultralytics_overlay(self, color_bgr: np.ndarray, detections: list[Any]) -> np.ndarray:
        vis = color_bgr.copy()
        palette = [(0, 80, 255), (255, 80, 0), (0, 200, 0), (200, 0, 200), (0, 200, 200)]
        for det in detections:
            color = palette[int(det.track_id) % len(palette)]
            x1, y1, x2, y2 = np.asarray(det.box).astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{det.label}#{det.track_id} {det.score:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return vis

    # ── RealSense helpers ───────────────────────────────────────────────────

    def _apply_sensor_settings(self, sensor_settings: dict[str, Any]) -> None:
        if not sensor_settings:
            return
        try:
            device = self.rs_profile.get_device()
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

    def build_rs_depth_filters(self) -> list:
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)

        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, 2)

        return [decimation, hole_filling]

    def apply_depth_filters(self, frames, filters: list):
        filtered_frames = frames
        for filt in filters:
            filtered_frames = filt.process(filtered_frames).as_frameset()
        return filtered_frames

    # ── Shutdown ────────────────────────────────────────────────────────────

    def destroy_node(self) -> bool:
        self._capture_running = False
        if hasattr(self, "_capture_thread") and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        try:
            self.rs_pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CVPerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
