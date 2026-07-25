"""Run Sam3TRT on bottle_0000.png: report presence + latency, save an overlay."""

import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from sam3_trt import Sam3TRT

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "onnx_weights" / "sam3_fp16.plan"
IMAGE = HERE / "bottle_0000.png"
OUT = HERE / "bottle_0000_seg.png"
WARMUP_RUNS = 5
MEASURE_RUNS = 200
STEADY_STATE_RUNS = 10

sam = Sam3TRT(str(ENGINE))
bgr = cv2.imread(str(IMAGE))

for i in range(WARMUP_RUNS):
    t0 = time.perf_counter()
    mask, presence = sam.infer_bench(bgr)
    ms = (time.perf_counter() - t0) * 1000
    print(f"warmup {i + 1:02d}/{WARMUP_RUNS}: {ms:.2f} ms")

latencies = []
for i in range(MEASURE_RUNS):
    t0 = time.perf_counter()
    mask, presence = sam.infer_bench(bgr)
    ms = (time.perf_counter() - t0) * 1000
    latencies.append(ms)
    print(f"run {i + 1:02d}/{MEASURE_RUNS}: {ms:.2f} ms")

steady_state_ms = statistics.median(latencies[-STEADY_STATE_RUNS:])
print(
    f"steady-state latency: {steady_state_ms:.2f} ms "
    f"(median of final {STEADY_STATE_RUNS} runs)"
)

mask_np = mask.cpu().numpy()
print(f"presence={presence:.3f}  mask_px={int((mask_np > 0).sum())}/{mask_np.size}")

overlay = bgr.copy()
overlay[mask_np > 0] = (
    0.5 * overlay[mask_np > 0] + 0.5 * np.array([0, 0, 255])
).astype(np.uint8)
cv2.imwrite(str(OUT), np.hstack([bgr, overlay]))
print(f"wrote {OUT}")
