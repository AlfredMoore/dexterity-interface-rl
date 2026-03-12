"""
SAM 3 inference wrapper for per-frame semantic segmentation.
Encapsulates model loading, BGR preprocessing, and multi-concept text-prompted inference.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, Optional

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# Object IDs assigned to each semantic concept (fixed at init, used downstream)
CONCEPT_LEFT_ARM  = 1
CONCEPT_RIGHT_ARM = 2
CONCEPT_CUP       = 3


class SAM3Inference:
    """
    Wraps SAM 3 image model for per-frame semantic segmentation.

    Concepts (text prompts + their object IDs) are registered once at init.
    Each call to infer() runs detection for all registered concepts on a
    single BGR frame and returns per-concept mask/box/score results.

    The vision encoder can be compiled via torch.compile (compile=True) for
    significantly faster per-frame inference at the cost of a one-time warm-up.

    Usage:
        sam3 = SAM3Inference(
            ckpt_path="/path/to/sam3.pt",
            device="cuda",
            compile=True,
        )
        results = sam3.infer(bgr_uint8)
        # results: dict[obj_id -> {"concept", "masks", "boxes", "scores"}]
        #   masks:  (N, 1, H, W) bool   -- N detections for this concept
        #   boxes:  (N, 4)       float32 -- pixel-space [x0, y0, x1, y1]
        #   scores: (N,)         float32 -- confidence scores
    """

    # Default concept map for this project: text_prompt -> object_id
    DEFAULT_CONCEPT_MAP: Dict[str, int] = {
        "left robot arm":  CONCEPT_LEFT_ARM,
        "right robot arm": CONCEPT_RIGHT_ARM,
        "cup":             CONCEPT_CUP,
    }

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        compile: bool = True,
        concept_map: Optional[Dict[str, int]] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            ckpt_path:            Path to the sam3.pt checkpoint file.
            device:               torch device string ("cuda" or "cpu").
            compile:              Apply torch.compile to the ViT encoder and segmentation
                                  head for faster per-frame inference. First call triggers
                                  a warm-up. Requires CUDA.
            concept_map:          Optional override for {text_prompt: object_id}. Defaults
                                  to DEFAULT_CONCEPT_MAP ("left robot arm"=1, "right robot
                                  arm"=2, "cup"=3).
            confidence_threshold: Detection score threshold; lower → more recalls.
        """
        self.device = device
        self.concept_map = concept_map if concept_map is not None else self.DEFAULT_CONCEPT_MAP

        # Disable compile on CPU -- no benefit without CUDA
        _compile = compile and torch.device(device).type == "cuda"

        model = build_sam3_image_model(
            checkpoint_path=ckpt_path,
            device=device,
            eval_mode=True,
            load_from_HF=False,
            compile=_compile,
        )
        self.processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    @torch.inference_mode()
    def infer(self, bgr: np.ndarray) -> Dict[int, dict]:
        """
        Run SAM 3 semantic segmentation on a single BGR frame.

        Vision encoding is performed once; each registered concept then runs its
        own text-conditioned grounding forward pass over the shared visual features.

        Args:
            bgr: (H, W, 3) uint8 BGR frame from RealSense / cv2.

        Returns:
            dict keyed by object_id (int), each value is a dict with:
                "concept": str                  -- the text prompt used
                "masks":   (N, 1, H, W) bool    -- per-detection binary masks (None if no detections)
                "boxes":   (N, 4)       float32 -- pixel-space [x0, y0, x1, y1] (None if no detections)
                "scores":  (N,)         float32 -- confidence scores (None if no detections)
        """
        # BGR numpy -> PIL RGB (Sam3Processor expects PIL Image or tensor)
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # Encode visual features once for all concepts
        base_state = self.processor.set_image(pil_img)

        results: Dict[int, dict] = {}
        for concept, obj_id in self.concept_map.items():
            # Shallow-copy the state so that each concept's text features
            # do not bleed into the next concept's grounding pass.
            # backbone_out is also shallow-copied: set_text_prompt only adds
            # new language keys (language_features, language_mask, language_embeds)
            # without modifying the shared vision tensors.
            state = dict(base_state)
            state["backbone_out"] = dict(base_state["backbone_out"])

            state = self.processor.set_text_prompt(prompt=concept, state=state)

            masks  = state.get("masks")   # (N, 1, H, W) bool tensor or absent
            boxes  = state.get("boxes")   # (N, 4) float tensor or absent
            scores = state.get("scores")  # (N,) float tensor or absent

            # Convert to numpy for downstream consumers; None if no detections
            results[obj_id] = {
                "concept": concept,
                "masks":   masks.cpu().numpy()  if masks  is not None and masks.numel()  > 0 else None,
                "boxes":   boxes.cpu().numpy()  if boxes  is not None and boxes.numel()  > 0 else None,
                "scores":  scores.cpu().numpy() if scores is not None and scores.numel() > 0 else None,
            }

        return results
