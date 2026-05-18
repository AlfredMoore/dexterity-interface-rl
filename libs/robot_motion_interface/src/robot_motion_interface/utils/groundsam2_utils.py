"""
Grounded-SAM-2 streaming wrapper for per-frame depth masking.

Pipeline:
    GroundingDINO (text prompt -> bbox)  ─┐
                                          ├─►  SAM2VideoPredictor (streaming)
    SAM2 image predictor (bbox -> mask) ──┘                │
                                                           ▼
                                                     per-frame mask

GroundingDINO + SAM2-image are invoked every `detection_interval` frames to
(re-)seed the video predictor's memory bank; in between, only the SAM2 video
predictor's `add_new_frame` + `infer_single_frame` runs — ~25-80 ms/frame on a
4090 depending on backbone size.

Usage:
    masker = GroundedSAM2Masker(prompt="bottle.", sam2_variant="l")

    for bgr, depth in stream:
        masked_depth, mask = masker.mask_depth(bgr, depth)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Per variant: (hydra config name, HF repo id, ckpt filename inside the repo).
_SAM2_VARIANTS = {
    "t": ("configs/sam2.1/sam2.1_hiera_t.yaml",  "facebook/sam2.1-hiera-tiny",      "sam2.1_hiera_tiny.pt"),
    "s": ("configs/sam2.1/sam2.1_hiera_s.yaml",  "facebook/sam2.1-hiera-small",     "sam2.1_hiera_small.pt"),
    "b": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "facebook/sam2.1-hiera-base-plus", "sam2.1_hiera_base_plus.pt"),
    "l": ("configs/sam2.1/sam2.1_hiera_l.yaml",  "facebook/sam2.1-hiera-large",     "sam2.1_hiera_large.pt"),
}


class GroundedSAM2Masker:
    """
    Text-prompt-driven streaming segmentation + depth masker.

    Episode lifecycle:
      - Construct once.
      - Optionally call `reset(prompt=...)` to change the text prompt.
      - Call `mask_depth(bgr, depth)` (or `step(bgr)` for mask only) per frame.

    Internals:
      - Every `detection_interval` frames the GroundingDINO+SAM2-image branch
        re-detects and re-seeds the video predictor — recovers from drift,
        and avoids unbounded GPU-memory growth in the streaming `images` tensor.
      - In between, only the SAM2 video predictor runs (single-frame inference
        backed by its memory bank).
    """

    def __init__(
        self,
        prompt: str = "bottle.",
        sam2_variant: str = "l",
        sam2_ckpt_path: Optional[str] = None,
        sam2_cache_dir: Optional[str] = None,
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        grounding_cache_dir: Optional[str] = None,
        device: str = "cuda",
        detection_interval: int = 20,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        fill_value: float = 0.0,
        mask_dilation: int = 0,
    ):
        if sam2_variant not in _SAM2_VARIANTS:
            raise ValueError(
                f"sam2_variant must be one of {list(_SAM2_VARIANTS)}, got {sam2_variant!r}"
            )
        if detection_interval < 1:
            raise ValueError("detection_interval must be >= 1")

        self.device = device
        self.prompt = prompt.strip()
        if not self.prompt.endswith("."):
            self.prompt = self.prompt + "."  # GroundingDINO expects '.'-separated phrases.
        self.detection_interval = int(detection_interval)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.fill_value = float(fill_value)
        self.mask_dilation = int(mask_dilation)

        sam2_cfg, sam2_hf_id, sam2_ckpt_name = _SAM2_VARIANTS[sam2_variant]

        # Resolve SAM2 checkpoint: explicit path > cache_dir > HF default cache.
        # `hf_hub_download` is a no-op when the file already exists in `cache_dir`.
        if sam2_ckpt_path is not None:
            if not Path(sam2_ckpt_path).is_file():
                raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_ckpt_path}")
        else:
            from huggingface_hub import hf_hub_download
            sam2_ckpt_path = hf_hub_download(
                repo_id=sam2_hf_id, filename=sam2_ckpt_name, cache_dir=sam2_cache_dir,
            )

        # GroundingDINO: `grounding_model_id` may be a HF Hub id (auto-downloaded
        # into `grounding_cache_dir`, or HF's default ~/.cache/huggingface when
        # None) OR an absolute path to an already-downloaded snapshot directory.
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self._gd_processor = AutoProcessor.from_pretrained(
            grounding_model_id, cache_dir=grounding_cache_dir,
        )
        self._gd_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                grounding_model_id, cache_dir=grounding_cache_dir,
            )
            .to(device)
        )

        # SAM2 image predictor — turns bbox into a clean mask for seeding.
        self._sam2_image = SAM2ImagePredictor(
            build_sam2(sam2_cfg, str(sam2_ckpt_path), device=device)
        )

        # SAM2 video predictor — streaming. Grounded-SAM-2's fork adds
        # `init_state(video_path=None)` / `add_new_frame` / `infer_single_frame`.
        self._sam2_video = build_sam2_video_predictor(
            sam2_cfg, str(sam2_ckpt_path), device=device
        )

        self._frame_idx = 0          # global frame counter across episode
        self._has_init_mask = False  # True once the video predictor has a prompt
        self._state = self._fresh_state()

    # ------------------------------------------------------------------ utils

    def _fresh_state(self, image_hw: tuple[int, int] | None = None) -> dict:
        state = self._sam2_video.init_state()
        # 1024 is SAM2's image_size; matches process_stream_frame's resize target.
        state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
        if image_hw is None:
            state["video_height"] = None
            state["video_width"] = None
        else:
            state["video_height"] = int(image_hw[0])
            state["video_width"] = int(image_hw[1])
        return state

    def reset(self, prompt: Optional[str] = None) -> None:
        """Reset video memory; optionally change the text prompt."""
        if prompt is not None:
            self.prompt = prompt.strip()
            if not self.prompt.endswith("."):
                self.prompt = self.prompt + "."
        self._state = self._fresh_state()
        self._frame_idx = 0
        self._has_init_mask = False

    def _detect_and_seed(self, rgb: np.ndarray) -> bool:
        """Run GroundingDINO + SAM2-image to produce a clean mask, then seed video state.

        Returns True iff a detection passed thresholds and seeding succeeded.
        """
        from PIL import Image
        img_pil = Image.fromarray(rgb)

        inputs = self._gd_processor(images=img_pil, text=self.prompt, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self._gd_model(**inputs)
        results = self._gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img_pil.size[::-1]],
        )
        boxes = results[0]["boxes"]
        if boxes.shape[0] == 0:
            return False

        # SAM2 image branch: bbox -> mask. Only keep the top-1 detection.
        self._sam2_image.set_image(rgb)
        # GroundingDINO returns boxes as float tensor on `device`; SAM2 wants numpy/list.
        top_box = boxes[:1].detach().cpu().numpy()
        masks, _scores, _logits = self._sam2_image.predict(
            point_coords=None,
            point_labels=None,
            box=top_box,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)         # (1, H, W)
        elif masks.ndim == 2:
            masks = masks[None]              # (1, H, W)
        seed_mask = masks[0].astype(bool)
        if not seed_mask.any():
            return False

        # Start a fresh short tracking window on every successful re-detection.
        # SAM2's streaming state keeps all added frames in `images`; rebuilding
        # the state here prevents unbounded GPU memory growth on long streams.
        self._state = self._fresh_state(image_hw=rgb.shape[:2])
        frame_idx = self._sam2_video.add_new_frame(self._state, rgb)
        self._sam2_video.reset_state(self._state)
        self._sam2_video.add_new_mask(
            self._state,
            frame_idx=frame_idx,
            obj_id=1,
            mask=torch.from_numpy(seed_mask).to(self.device),
        )
        return True

    # ------------------------------------------------------------------- API

    def step(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """Process one frame. Returns boolean mask (H, W) or None if no detection yet."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        need_detect = (
            (self._frame_idx % self.detection_interval == 0)
            or (not self._has_init_mask)
        )
        detected_this_step = False
        if need_detect:
            detected_this_step = self._detect_and_seed(rgb)
            if detected_this_step:
                self._has_init_mask = True

        if not self._has_init_mask:
            self._frame_idx += 1
            return None

        # `_detect_and_seed` adds the frame itself on success; otherwise add now.
        if not detected_this_step:
            self._sam2_video.add_new_frame(self._state, rgb)

        frame_idx = self._state["num_frames"] - 1
        _, _, video_res_masks = self._sam2_video.infer_single_frame(
            inference_state=self._state, frame_idx=frame_idx
        )
        # video_res_masks: (num_obj, 1, H, W) in (-inf, +inf) logits.
        mask = (video_res_masks[0, 0] > 0.0).detach().cpu().numpy()
        if self.mask_dilation > 0:
            k = 2 * self.mask_dilation + 1
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

        self._frame_idx += 1
        return mask

    def mask_depth(
        self, bgr: np.ndarray, depth: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Mask out depth outside the SAM2 mask.

        Args:
            bgr:   (H, W, 3) uint8 BGR colour frame.
            depth: (H, W) aligned depth frame, any numeric dtype.

        Returns:
            masked_depth: same shape & dtype as `depth`; pixels outside the
                          mask replaced with `fill_value`. All-`fill_value`
                          when no detection has succeeded yet.
            mask:         (H, W) bool mask, or None if no detection yet.
        """
        mask = self.step(bgr)
        out = np.full_like(depth, self.fill_value)
        if mask is None:
            return out, None

        # Resize mask if SAM2 returned a different resolution (shouldn't happen
        # with default settings, but guard anyway).
        if mask.shape != depth.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        out[mask] = depth[mask]
        return out, mask


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _main():
    """Run on a JPG+PNG frame folder; write masked depth + mask overlay PNGs."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Grounded-SAM-2 streaming smoke test")
    parser.add_argument("--frames_dir", required=True,
                        help="Directory with color/*.jpg and depth/*.png")
    parser.add_argument("--sam2_ckpt", default=None,
                        help="Path to SAM2 checkpoint. If omitted, auto-downloaded via HF.")
    parser.add_argument("--sam2_cache_dir", default=None,
                        help="HF cache dir for SAM2 auto-download (default: ~/.cache/huggingface).")
    parser.add_argument("--prompt", default="bottle.",
                        help="GroundingDINO text prompt (period-separated phrases)")
    parser.add_argument("--variant", default="l", choices=list(_SAM2_VARIANTS))
    parser.add_argument("--detection_interval", type=int, default=20)
    parser.add_argument("--clip", nargs=2, type=float, default=[0.1, 1.1],
                        metavar=("NEAR_M", "FAR_M"),
                        help="Depth clip in metres for visualisation only.")
    parser.add_argument("--depth_scale", type=float, default=0.001,
                        help="Multiply uint16 depth by this to get metres.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    color_dir = frames_dir / "color"
    depth_dir = frames_dir / "depth"
    out_dir   = frames_dir / "gsam2_masked"
    out_dir.mkdir(parents=True, exist_ok=True)

    color_paths = sorted(color_dir.glob("*.jpg"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    if not color_paths or len(color_paths) != len(depth_paths):
        raise RuntimeError(
            f"color/depth count mismatch: {len(color_paths)} vs {len(depth_paths)}"
        )

    masker = GroundedSAM2Masker(
        prompt=args.prompt,
        sam2_variant=args.variant,
        sam2_ckpt_path=args.sam2_ckpt,
        sam2_cache_dir=args.sam2_cache_dir,
        detection_interval=args.detection_interval,
    )

    near, far = float(args.clip[0]), float(args.clip[1])
    total_t = 0.0
    for i, (c_path, d_path) in enumerate(zip(color_paths, depth_paths)):
        bgr = cv2.imread(str(c_path), cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED)

        t0 = time.time()
        masked_u16, mask = masker.mask_depth(bgr, depth_u16)
        dt = time.time() - t0
        total_t += dt

        # Colourise masked depth for inspection.
        depth_m = masked_u16.astype(np.float32) * args.depth_scale
        depth_m = np.clip(depth_m, near, far)
        norm = ((depth_m - near) / (far - near) * 255).astype(np.uint8)
        vis = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        if mask is not None:
            vis[~mask] = 0
        cv2.imwrite(str(out_dir / f"{i:06d}.png"), vis)

        if i % 30 == 0:
            print(f"[{i:5d}] dt={dt*1000:.1f} ms  mask={'OK' if mask is not None else 'MISS'}")

    print(f"\nDone. {len(color_paths)} frames, mean {total_t / len(color_paths) * 1000:.1f} ms/frame")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    _main()
