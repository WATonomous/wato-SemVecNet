# CUDA-SemVecNet

C++/CUDA/TensorRT inference for the SemVecNet model. It generates a BEV semantic map and produces vector map polylines.

## File Structure

```
CUDA-SemVecNet/
├── CMakeLists.txt     builds libsemvecnet_core.so (+ plugins, + demo exe if present)
├── cmake/             FindTensorRT.cmake, install config
├── include/
│   └── semvecnet/     <- all headers (.hpp/.cuh): #include "semvecnet/foo.hpp"
├── src/
│   ├── semvecnet/     <- implementation: .cpp (host code) and .cu (CUDA kernels)
│   ├── common/        <- shared helpers (CUDA checks, tensors, timers)
│   └── main.cpp       <- demo executable (build/semvecnet)
├── models/            <- generated ONNX exports and TensorRT engines (git-ignored)
├── tests/             <- ctest unit tests, one binary per source file
├── tool/
│   ├── build.sh           configure + build (writes compile_commands.json)
│   ├── docker_run.sh      run a command in the dev container
│   ├── sync_headers.sh    headers for host IntelliSense
│   ├── export/            HRNet and SemVecNet model export, TensorRT engines
│   └── eval/              rosbag smoke test, output comparison, profiling report
└── docker/            Dockerfile (dev, build stages) + docker-compose.yaml
```

Headers go in `include/` and are included as `#include "semvecnet/foo.hpp"`. `include/` is what
gets installed for the monorepo; `src/` is also on the include path (privately) for internal headers.

## Develop (VS Code)

1. Open the repository in VS Code and **allow automatic tasks** when prompted.
2. Build: **Ctrl+Shift+B**
3. Run: **F5** (outputs to Terminal panel)
4. Debug: Run and Debug panel > "semvecnet (debugger)"

Builds and runs in a GPU container (Docker + NVIDIA Container Toolkit required).
Edit on the host; headers are synced automatically for IntelliSense.

## Rosbag smoke test

From the repository root:

```bash
CUDA-SemVecNet/tool/eval/run_rosbag_smoke.py
```

Downloads the sample bag into `bags/` on first run (shows a progress bar;
skipped if already downloaded), then runs two synchronized frames through the
C++ pipeline using all eight cameras and the merged LiDAR. Needs Docker with
NVIDIA GPU support, the `semvecnet-mcap-hrnet:latest` reader image,
`CUDA-SemVecNet/models/hrnet_ocrnet_mapillary_b8_s1.plan`, and
`CUDA-SemVecNet/models/semvecnet.onnx`.

Results are written per frame (matching six-digit filename stems) to:

- `polylines/`: JSON with timestamp, pose, and scored vectors.
- `semantic_maps/`: the 150 × 300 RGB semantic map.
- `combined/`: the semantic map next to the predicted polylines.

Use `--frames 0` to process the whole bag, `--out` to pick an output
directory, or `--help` for all options (including pointing `bag` at a
different MCAP file or directory).

## HRNet engine

From `CUDA-SemVecNet/`, run on the machine whose GPU will load the engine, then copy the `.plan` into `models/`:

```
tool/export/export_hrnet.sh --gpu 3000      # RTX 30-series; use --gpu 4090 for a 4090
cp hrnet_build/hrnet_ocrnet_mapillary_b8_s1.plan models/
```

## Engine

```
tool/export/build_trt_engine.sh path/to/semvecnet.onnx   # -> models/semvecnet.plan
```
A `.plan` only works with the TensorRT version and GPU it was built on.

## Testing

From `CUDA-SemVecNet/`:

```bash
tool/docker_run.sh bash -c 'cmake -S . -B build -G Ninja -DBUILD_TESTING=ON && tool/build.sh && ctest --test-dir build --output-on-failure'
```

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

## Exporting models from checkpoints

Store generated artifacts for both models in `CUDA-SemVecNet/models/` (they
are git-ignored). Trained `.pth` checkpoints are exported to ONNX; the pipeline
loads HRNet's TensorRT `.plan` and SemVecNet's `.onnx` via ONNX Runtime.
`tool/export/build_trt_engine.sh` can also build a SemVecNet `.plan`.
TensorRT plans are specific to the GPU and TensorRT version used to build them.

HRNet's `.onnx` in `models/` is the existing export of the trained Mapillary
snapshot; see the [upstream export recipe](../docs/prepare_dataset.md#prepare-dataset)
to rebuild it from a checkpoint. Do not export the ImageNet initialization
checkpoint as a standalone model.

For a fine-tuned SemVecNet encoder/decoder checkpoint, export it to `.onnx`:

```bash
docker build -t semvecnet-bevfusion:latest -f docker/Dockerfile .
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=4 \
  --entrypoint python \
  -v "$PWD/SemVecNet/ckpts:/workspace/SemVecNet/ckpts:ro" \
  -v "$PWD/CUDA-SemVecNet/tool/export:/export:ro" \
  -v "$PWD/CUDA-SemVecNet/models:/artifacts" \
  semvecnet-bevfusion:latest /export/export_semvecnet_onnx.py --combined --output /artifacts
```

Omit `--combined` to export the encoder and decoder as two separate graphs
instead (useful if you need the intermediate features). The exporter loads
the checkpoint strictly with the nuScenes model configuration, replaces only
the decoder's bilinear sampler during export, and compares the replacement
with the PyTorch reference before writing the decoder graph. The trained
attention projections stay unchanged.

## Monorepo integration

`integration/cuda_semvecnet_vendor` is an `ament_cmake` vendor package that builds
`CUDA-SemVecNet/` as an `ExternalProject` and installs it, exposing the imported
target `cuda_semvecnet_vendor::semvecnet_core`. A ROS wrapper package depends on
it with:

```cmake
find_package(cuda_semvecnet_vendor REQUIRED)
target_link_libraries(<target> PUBLIC cuda_semvecnet_vendor::semvecnet_core)
```

The `.plan` engine is not baked into the image; it is mounted and its path is
passed in as a ROS parameter, since engines are tied to the TensorRT version
and GPU they were built on.

`SemVecNet::forward` (`include/semvecnet/semvecnet.hpp`) is the entry point the
wrapper package calls. It takes:

- tightly packed RGB8 camera images and a flat array of five FP16 values per
  LiDAR point (x, y, z, intensity, time_lag) — the same host buffers BEVFusion
  uses
- per-camera calibrations (input-image intrinsics, camera order) and the
  shared image width/height
- both `lidar_to_map` and `ego_to_map` row-major 4x4 poses: the mapper uses the
  LiDAR pose to place points, and the renderer uses the ego pose to orient and
  center the model input (they may differ when the LiDAR is offset from the
  vehicle's ego frame)
- an optional output buffer to receive the rendered semantic map before
  normalization

and returns the predicted vector map polylines.

Note: the vendor package needs at least one source file in
`CUDA-SemVecNet/src/semvecnet/`, otherwise there is no `libsemvecnet_core.so`
to import.

## Docker image

```
docker build -f docker/Dockerfile --target dev -t semvecnet:dev-ort .   # from CUDA-SemVecNet/
```
