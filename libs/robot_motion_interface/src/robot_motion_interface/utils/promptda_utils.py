"""
PromptDA inference wrapper for RealSense input.
Encapsulates model loading, preprocessing, and inference.
"""

import cv2
import numpy as np
import torch

from promptda.promptda import PromptDA

class PromptDAInference:
    """
    Wraps PromptDA model loading, RealSense preprocessing, and inference.

    Usage:
        pda = PromptDAInference(ckpt_path="/path/to/model.ckpt", encoder="vits", device="cuda")
        metric_depth = pda.infer(bgr_uint8, depth_uint16)  # → [H, W] float32 metres on GPU
    """

    def __init__(self,
                 ckpt_path: str,
                 encoder: str = "vits",
                 device: str = "cuda",
                 max_size: int = 1008,
                 multiple_of: int = 14,
                 depth_scale: float = 0.001):
        """
        Args:
            ckpt_path:    path to PromptDA .ckpt checkpoint file
            encoder:      "vits" or "vitl"
            device:       torch device string
            max_size:     longest edge of color image is capped here
            multiple_of:  ViT patch size requirement (DINOv2 = 14)
            depth_scale:  metres per raw depth unit (RealSense D-series default: 0.001)
        """

        self.device = torch.device(device)
        self.max_size = max_size
        self.multiple_of = multiple_of
        self.depth_scale = depth_scale

        self.model = PromptDA.from_pretrained(
            ckpt_path, model_kwargs={"encoder": encoder}
        ).to(self.device).eval()

    def _preprocess_color(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 HxWx3 → [1, 3, H', W'] float32 on device."""
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            h = int(h * scale) // self.multiple_of * self.multiple_of
            w = int(w * scale) // self.multiple_of * self.multiple_of
        else:
            h = h // self.multiple_of * self.multiple_of
            w = w // self.multiple_of * self.multiple_of
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _preprocess_depth(self, depth_u16: np.ndarray) -> torch.Tensor:
        """uint16 HxW → [1, 1, H, W] float32 metres on device."""
        depth = depth_u16.astype(np.float32) * self.depth_scale
        return torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def infer(self, bgr: np.ndarray, depth_u16: np.ndarray) -> torch.Tensor:
        """
        Run PromptDA inference on a RealSense frame pair.

        Args:
            bgr:       HxWx3 uint8 BGR color frame
            depth_u16: HxW uint16 depth frame (aligned to color, raw units)

        Returns:
            [H', W'] float32 tensor of metric depth in metres, on self.device
            H'/W' = input H/W rounded down to nearest multiple of 14
        """
        image_tensor = self._preprocess_color(bgr)
        depth_tensor = self._preprocess_depth(depth_u16)
        return self.model.predict(image_tensor, depth_tensor)
