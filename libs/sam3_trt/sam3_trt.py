"""SAM3 TensorRT inference: image + text prompt -> binary mask.

The compiled TensorRT engine (built by build_engine_sam3.py) IS the model, so we
do NOT load Sam3Model here -- that would run the slow PyTorch model and defeat
TensorRT. We still use Sam3Processor for the two lightweight CPU steps it owns:
tokenizing the text prompt and preprocessing the image to SAM3's 1008x1008
pixel_values. Both are fed to the engine, so preprocessing matches training
exactly (no hand-rolled resize / normalisation).

Only the `semantic_seg` head is used (one mask for the whole concept, e.g. the
bottle); `presence_logits` is returned so callers can gate on concept presence.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from transformers.models.sam3 import Sam3Processor

HERE = Path(__file__).resolve().parent
DEFAULT_PROCESSOR_PATH = HERE / "cache" / "sam3_processor"

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


class Sam3TRT:
    def __init__(self, engine_path, processor_path=DEFAULT_PROCESSOR_PATH, device="cuda"):
        self.device = torch.device(device)
        self.processor = Sam3Processor.from_pretrained(
            str(processor_path),
            local_files_only=True,
        )

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        # One persistent GPU buffer per binding; TRT reads/writes these addresses.
        self.buf = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            self.buf[name] = torch.empty(shape, dtype=dtype, device=self.device)
            self.ctx.set_tensor_address(name, self.buf[name].data_ptr())

    @torch.no_grad()
    def infer(self, bgr: np.ndarray, prompt: str = "bottle", threshold: float = 0.5):
        """Return a CUDA uint8 mask with values {0, 255} and a presence score."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, text=prompt, return_tensors="pt", device=self.device)
        for name in ("pixel_values", "input_ids", "attention_mask"):
            self.buf[name].copy_(inputs[name])  # copy_ moves to CUDA + casts dtype
        stream = torch.cuda.current_stream(self.device)
        self.ctx.execute_async_v3(stream.cuda_stream)
        prob = torch.sigmoid(self.buf["semantic_seg"].float())      # 1,1,288,288
        prob = F.interpolate(prob, (h, w), mode="bilinear", align_corners=False)[0, 0]
        mask = (prob > threshold).to(torch.uint8) * 255
        presence = torch.sigmoid(self.buf["presence_logits"].float()).item()
        return mask, presence

    @torch.no_grad()
    def infer_bench(self, bgr: np.ndarray, prompt: str = "bottle", threshold: float = 0.5):
        """Return a CUDA uint8 mask with values {0, 255} and a presence score. With detailed latency breakdown for benchmarking."""
        total_start = time.perf_counter()

        stage_start = time.perf_counter()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        color_ms = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        inputs = self.processor(images=rgb, text=prompt, return_tensors="pt", device=self.device)
        processor_ms = (time.perf_counter() - stage_start) * 1000

        stream = torch.cuda.current_stream(self.device)

        stage_start = time.perf_counter()
        for name in ("pixel_values", "input_ids", "attention_mask"):
            self.buf[name].copy_(inputs[name])  # copy_ moves to CUDA + casts dtype
        stream.synchronize()
        input_copy_ms = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        tensorrt_ms = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        prob = torch.sigmoid(self.buf["semantic_seg"].float())      # 1,1,288,288
        prob = F.interpolate(prob, (h, w), mode="bilinear", align_corners=False)[0, 0]
        mask = (prob > threshold).to(torch.uint8) * 255
        stream.synchronize()
        mask_post_ms = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        presence = torch.sigmoid(self.buf["presence_logits"].float()).item()
        presence_post_ms = (time.perf_counter() - stage_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000
        print(
            "infer latency: "
            f"color={color_ms:.3f} ms, "
            f"processor={processor_ms:.3f} ms, "
            f"input_copy={input_copy_ms:.3f} ms, "
            f"tensorrt={tensorrt_ms:.3f} ms, "
            f"mask_post={mask_post_ms:.3f} ms, "
            f"presence_post={presence_post_ms:.3f} ms, "
            f"total={total_ms:.3f} ms"
        )
        return mask, presence
