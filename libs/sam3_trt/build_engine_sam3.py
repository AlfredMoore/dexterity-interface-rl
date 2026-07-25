"""Build a TensorRT FP16 engine from the exported SAM3 ONNX.

Uses the TensorRT Python API (not trtexec) so the engine is built with the same
TensorRT the ROS node will load it with -- engines are version-locked.

The SAM3 processor/tokenizer is also saved locally so TensorRT inference never
needs Hugging Face authentication.

Usage: python build_engine_sam3.py [onnx_path] [plan_path]
"""

import sys
from pathlib import Path

import tensorrt as trt
from transformers.models.sam3 import Sam3Processor

HERE = Path(__file__).resolve().parent
ONNX_WEIGHTS = HERE / "onnx_weights"
PROCESSOR_CACHE = HERE / "cache" / "sam3_processor"
MODEL_ID = "facebook/sam3"

ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    ONNX_WEIGHTS / "sam3_dynamic.onnx"
PLAN_PATH = sys.argv[2] if len(sys.argv) > 2 else \
    ONNX_WEIGHTS / "sam3_fp16.plan"

if not (PROCESSOR_CACHE / "processor_config.json").is_file():
    print(f"caching {MODEL_ID} processor in {PROCESSOR_CACHE}")
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    processor.save_pretrained(PROCESSOR_CACHE)
else:
    print(f"using cached processor at {PROCESSOR_CACHE}")

Sam3Processor.from_pretrained(str(PROCESSOR_CACHE), local_files_only=True)
print("validated local processor cache")

logger = trt.Logger(trt.Logger.WARNING)
print(f"TensorRT {trt.__version__}")

builder = trt.Builder(logger)
network = builder.create_network(0)  # TRT 10 is explicit-batch only; no flags needed
parser = trt.OnnxParser(network, logger)

# parse_from_file, NOT parse(bytes): SAM3 has EXTERNAL weight shards that the
# parser resolves relative to the .onnx directory. parse(f.read()) would miss them.
if not parser.parse_from_file(str(ONNX_PATH)):
    for i in range(parser.num_errors):
        print("  parse error:", parser.get_error(i))
    sys.exit(1)

print("inputs:")
for i in range(network.num_inputs):
    t = network.get_input(i)
    print(f"  {t.name}: {t.shape} {t.dtype}")
print("outputs:")
for i in range(network.num_outputs):
    t = network.get_output(i)
    print(f"  {t.name}: {t.shape} {t.dtype}")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

print("building engine (fp16)... this can take several minutes")
plan = builder.build_serialized_network(network, config)
if plan is None:
    print("engine build failed")
    sys.exit(1)

with open(PLAN_PATH, "wb") as f:
    f.write(plan)
print(f"wrote {PLAN_PATH} ({plan.nbytes / 1e6:.1f} MB)")
