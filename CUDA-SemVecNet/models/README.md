# Models Directory

The three ONNX models in this directory run directly in ONNX Runtime. See
[ONNX inference and pipeline notes](../docs/onnx_inference.md) for model inputs,
outputs, export steps, and where CUDA kernels might help.

## TensorRT Version Compatibility Issue

**Problem:** `.plan` files are NOT portable across TensorRT versions.

The BEVFusion `.plan` files were created with:
- TensorRT 8.5.3
- CUDA 11.8

This project's Docker container has:
- TensorRT 10+ (with CUDA 13.4.3)

**Error you'll see:**
```
Serialization assertion header.magicTag == kEXPECTED_MAGIC_TAG failed.
Trying to load an engine created with incompatible serialization version
```

## Solutions

### Option 1: Match BEVFusion's Environment
Update `docker/Dockerfile` to use the same TensorRT version:
```dockerfile
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04 AS dev
# Install TensorRT 8.5.3 (matches BEVFusion)
```

### Option 2: Create a New .plan File
Convert an ONNX model to .plan with the current environment:
```bash
# Need trtexec tool (not in current Docker image)
trtexec --onnx=model.onnx --saveEngine=model.plan
```

### Option 3: Use Python tensorrt Package
```python
import tensorrt as trt
# Convert ONNX -> TensorRT engine
```

## For Testing

`TensorRtEngine::load()` in `src/common/common.cpp` deserializes a TensorRT engine and
returns an owner that keeps its runtime alive.
The engine must be compatible with the TensorRT version and GPU in use.
