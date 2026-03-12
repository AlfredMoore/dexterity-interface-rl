"""
SAM 2 inference wrapper for per-frame image segmentation.
Encapsulates model loading, BGR preprocessing, and prompt-based inference.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Inference:
    """
    Wraps SAM 2 model loading and per-frame segmentation for a live camera stream.

    Accepts BGR frames (as returned by RealSense / cv2), converts to RGB internally,
    and supports point, box, and mask prompts.

    The image encoder can be compiled via torch.compile for a significant
    per-frame speedup at the cost of a one-time warm-up on the first call.

    Usage:
        sam = SAM2Inference(
            ckpt_path="/path/to/sam2.1_hiera_small.pt",
            model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
            device="cuda",
            compile_image_encoder=True,
        )

        # --- prompt will be filled in by caller ---
        masks, scores, logits = sam.infer(
            bgr_uint8,
            point_coords=np.array([[cx, cy]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
        )
        # masks:  (N_masks, H, W)  bool
        # scores: (N_masks,)       float32 IoU confidence
        # logits: (N_masks, H/4, W/4) float32  -- pass back as mask_input for refinement
    """

    def __init__(
        self,
        ckpt_path: str,
        model_cfg: str,
        device: str = "cuda",
        compile_image_encoder: bool = True,
        multimask_output: bool = True,
    ):
        """
        Args:
            ckpt_path:             Path to the SAM 2 .pt checkpoint file.
            model_cfg:             Hydra config name resolved relative to the sam2 package,
                                   e.g. "configs/sam2.1/sam2.1_hiera_s.yaml".
            device:                torch device string ("cuda" or "cpu").
            compile_image_encoder: Apply torch.compile to the image encoder for faster
                                   per-frame inference. First call triggers a warm-up.
                                   Requires CUDA; automatically disabled on CPU.
            multimask_output:      If True, SAM 2 returns 3 ranked mask candidates;
                                   if False, returns 1 mask (more suitable for iterative
                                   refinement with mask_input).
        """
        self.device = torch.device(device)
        self.multimask_output = multimask_output

        # Disable compilation on CPU -- torch.compile gains are CUDA-specific
        _compile = compile_image_encoder and self.device.type == "cuda"

        hydra_overrides: list[str] = []
        if _compile:
            hydra_overrides.append("++model.compile_image_encoder=True")

        sam_model = build_sam2(
            config_file=model_cfg,
            ckpt_path=ckpt_path,
            device=str(self.device),
            hydra_overrides_extra=hydra_overrides,
        )
        self.predictor = SAM2ImagePredictor(sam_model)

    @torch.inference_mode()
    def infer(
        self,
        bgr: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment a single BGR frame with optional prompts.

        At least one prompt (point_coords/box/mask_input) must be provided;
        if all are None the call will raise inside SAM 2.

        Args:
            bgr:          (H, W, 3) uint8 BGR frame from RealSense / cv2.
            point_coords: (N, 2) float32 prompt points in pixel (x, y) coordinates.
            point_labels: (N,)   int32  1 = foreground, 0 = background click.
            box:          (4,)   float32 bounding box [x_min, y_min, x_max, y_max].
            mask_input:   (1, H/4, W/4) float32 low-res mask logits from a previous
                          call's `logits` output (enables iterative refinement).

        Returns:
            masks:  (N_masks, H, W) bool   -- binary segmentation masks.
            scores: (N_masks,)      float32 -- IoU confidence scores, highest first.
            logits: (N_masks, H/4, W/4) float32 -- raw logits for iterative refinement.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            self.predictor.set_image(rgb)
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                mask_input=mask_input,
                multimask_output=self.multimask_output,
                return_logits=True,  # always return logits for potential refinement
            )

        return masks, scores, logits
