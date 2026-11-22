# ONNX inference

Run the `.onnx` files in `CUDA-SemVecNet/models/` with ONNX Runtime. No TensorRT
engine or custom model plugin is required. The exports use batch size 1.

| File | Input | Output |
| --- | --- | --- |
| `hrnet_ocrnet_mapillary.onnx` | `input.1`: FP32 RGB `[1,3,720,960]` | `5776`: INT64 class IDs `[1,720,960]`, no extra argmax |
| `semvecnet.onnx` | `sem_map`: FP32 RGB `[1,3,300,150]` | `class_logits`: FP32 `[1,70,4]`; `normalized_points`: FP32 `[1,70,20,2]` |
| `semvecnet_encoder.onnx` | `sem_map`: FP32 RGB `[1,3,300,150]` | `features`: FP32 `[1,256,10,5]` |
| `semvecnet_decoder.onnx` | `features`: encoder output | `class_logits`: FP32 `[1,70,4]`; `normalized_points`: FP32 `[1,70,20,2]` |

Use `semvecnet.onnx` for one inference session, or the two split files if you
need access to intermediate features. The main Python inference path calls
`model.simple_test(...)` once; it runs the encoder and decoder internally,
then calls `get_bboxes`. Both ONNX layouts stop before `get_bboxes`, so polyline
postprocessing is still the caller's responsibility. HRNet is separate.

For OCRNet, resize RGB to width 960, height 720. Divide by 255, then subtract
`[0.485, 0.456, 0.406]` and divide by `[0.229, 0.224, 0.225]`. Match camera
intrinsics to the mask resolution, or resize the mask back with nearest
neighbor before projection.

### C++ input from the lanelet node

`semvecnet::forward()` accepts the host buffers used by the BEVFusion wrapper:
one pointer per decoded, contiguous RGB8 image in configured camera order, and
five FP16 values (`x, y, z, intensity, time_lag`) per LiDAR point. Pass the actual
shared image width and height, before any BEVFusion network crop. For example:

```cpp
semvecnet::PreparedFrame prepared;
semvecnet::forward(camera_images.data(), lidar_fp16, num_points, 8,
                   image_width, image_height, prepared);
```

Run HRNet once for each `prepared.hrnet_images[i]`, binding it as FP32
`[1, 3, 720, 960]`. `prepared.lidar_points` holds decoded FP32 points for
projection after inference; LiDAR is not an HRNet input. This call only prepares
the buffers on the CPU, using OpenCV `INTER_AREA` on RGB8 before FP32
normalization and HWC-to-CHW conversion, like the Python reference. The build
requires OpenCV `core` and `imgproc` (`libopencv-dev` on Ubuntu), included in
the `semvecnet:dev-ort` image. The ROS wrapper still
owns camera ordering, synchronization, and calibration; it does not pass a CUDA
stream.

### SemVecNet encoder

For SemVecNet, render an ego-centric RGB semantic map, not a camera image or a
class-ID tensor. Match the nuScenes grid: 0.2 m/pixel, 60 m forward/back span
and 30 m lateral span. Use the palette and orientation in
[`tools/infer_bevfusion.py`](../../tools/infer_bevfusion.py). On the 0–255 RGB
values, subtract `[123.675, 116.28, 103.53]` and divide by
`[58.395, 57.12, 57.375]`. Both models take NCHW tensors.

For a smoke test, run this from `CUDA-SemVecNet/` with NumPy and `onnxruntime`
installed. Replace zeros with the normalized inputs above for real inference.

```python
from pathlib import Path

import numpy as np
import onnxruntime as ort

models = Path("models")
providers = ["CPUExecutionProvider"]

def load(name):
    return ort.InferenceSession(str(models / name), providers=providers)

ocr = load("hrnet_ocrnet_mapillary.onnx")
semvecnet = load("semvecnet.onnx")
mask, = ocr.run(None, {"input.1": np.zeros((1, 3, 720, 960), np.float32)})
logits, points = semvecnet.run(None, {"sem_map": np.zeros((1, 3, 300, 150), np.float32)})
print(mask.shape, logits.shape, points.shape)
```

For GPU inference, use `onnxruntime-gpu` instead, with matching CUDA and cuDNN
libraries, and request `CUDAExecutionProvider`. Verify the session lists that
provider. Use runtime profiling to distinguish CUDA execution from CPU fallback.
For the tested Python setup, install `onnxruntime-gpu[cuda,cudnn]==1.23.2`,
call `ort.preload_dlls()` before loading sessions, and set
`providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]`.
On the RTX 3070 Ti, all three graphs executed on CUDA. On the synthetic BEV
fixture, maximum absolute error versus PyTorch was `0.00116` for normalized
points on CUDA and `0.000046` on CPU. This is a smoke test, not a validation
of map or lanelet quality.

In C++, use ONNX Runtime's C++ API and bind GPU tensors directly if transfers
between stages become a bottleneck.

## C++ and CUDA responsibilities

Start with CPU code for the non-model stages. Profile the full pipeline before
moving them to CUDA.

| Stage | First implementation | When CUDA helps |
| --- | --- | --- |
| Load sensor data, synchronize timestamps, load calibration and poses | C++ or existing sensor stack | No custom kernel. Use pinned host buffers and async copies if transfers dominate. |
| Resize and normalize camera images | OpenCV or existing GPU preprocessing | Fuse resize, normalize, and HWC-to-NCHW conversion if CPU cost or transfers matter. |
| Project LiDAR into cameras and sample labels | Port `DynamicMap.project_pcd` | Parallelize by point and camera. Match coordinate transforms, bounds, and mask sampling. |
| Rasterize and update BEV | Port `DynamicMap.update_map` | Parallelize grid updates. Test duplicate-cell behavior before replacing NumPy updates with GPU atomics or reductions. |
| Render and normalize ego BEV | Port `render_local_map` and `render_bev_map` | Fuse class selection, palette lookup, crop/warp, and normalization if profiling favors it. Match whether the reference enables filtering. |
| Decode polylines | Port `MapTRNMSFreeCoder` to C++ | No custom kernel needed initially: 280 class scores, top 50, score/range filtering, and conversion to metres. Classes are divider, pedestrian crossing, boundary, and centerline in that order. |
| Group lanelets | New CPU geometry code | No custom kernel needed initially. Pair oriented left and right boundaries, use centerlines where available, and validate geometry before constructing topology. |

If using BEVFusion instead of OCRNet and `DynamicMap`, port the class remap,
resampling, and coordinate conversion from `tools/infer_bevfusion.py` instead.

## Rebuild the exports

The OCRNet file is the existing export of the trained Mapillary snapshot.
See the [upstream export recipe](../../docs/prepare_dataset.md#prepare-dataset).
Do not export the ImageNet initialization checkpoint as a standalone model.

To rebuild the combined SemVecNet graph, run from the repository root:

```bash
docker build -t semvecnet-bevfusion:latest -f docker/Dockerfile .
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=4 \
  --entrypoint python \
  -v "$PWD/SemVecNet/ckpts:/workspace/SemVecNet/ckpts:ro" \
  -v "$PWD/CUDA-SemVecNet/tool:/export:ro" \
  -v "$PWD/CUDA-SemVecNet/models:/artifacts" \
  semvecnet-bevfusion:latest /export/export_semvecnet_onnx.py --combined --output /artifacts
```

Omit `--combined` to export the encoder and decoder as two separate graphs.

The exporter loads `mAP_epoch_99.pth` strictly with the nuScenes model
configuration. It replaces only the decoder's bilinear sampler during export
and compares the replacement with the PyTorch reference before writing the
decoder graph. The trained attention projections stay unchanged.
