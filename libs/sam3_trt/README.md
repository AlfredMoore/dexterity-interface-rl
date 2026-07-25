0. Install
```bash
pip install -e /workspace/libs/sam3_trt   # exposes `import sam3_trt`
```

1. Export ONNX
```bash
cd /workspace/libs/sam3_trt
export HF_TOKEN=<huggingface token>
python onnxexport_sam3.py
```
Get ONNX trace in `onnx_weights` and `onnx_weights/sam3_dynamic.onnx`.

2. Build TRT
```bash
python build_engine_sam3.py
```
Get TRT plan `onnx_weights/sam3_fp16.plan`.

3. Test SAM3 TRT
```bash
python test_sam3_trt.py
```