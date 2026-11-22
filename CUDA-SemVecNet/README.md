# CUDA-SemVecNet

C++/CUDA/TensorRT inference for the SemVecNet model: **ResNet-50 + FPN encoder and MapTRv2 decoder**,
taking a BEV semantic map and producing vector map polylines.

Same approach as `wato-cuda-bevfusion`: load a prebuilt TensorRT `.plan`, run the whole network
in one forward call, then post-process on the GPU/CPU.

```
CUDA-SemVecNet/
├── CMakeLists.txt     builds libsemvecnet_core.so (+ plugins, + demo exe if present)
├── cmake/             FindTensorRT.cmake, install config
├── include/
│   └── semvecnet/     <- all headers (.hpp/.cuh): #include "semvecnet/foo.hpp"
├── src/
│   ├── semvecnet/     <- implementation: .cpp (host code) and .cu (CUDA kernels)
│   ├── common/        <- shared helpers (CUDA checks, tensors, timers)
│   ├── plugins/       <- optional TensorRT plugins -> libsemvecnet_plugins.so
│   └── main.cpp       <- demo executable (build/semvecnet)
├── model/             <- put semvecnet.plan here (git-ignored)
├── tool/
│   ├── build.sh               configure + build (writes compile_commands.json)
│   └── build_trt_engine.sh    ONNX -> .plan via trtexec
└── docker/            Dockerfile (dev, build stages) + docker-compose.yaml
```

Headers go in `include/` and are included as `#include "semvecnet/foo.hpp"`. `include/` is what
gets installed for the monorepo; `src/` is also on the include path (privately) for internal headers.

`include/common/dtype.hpp` is copied unchanged from `wato-cuda-bevfusion`
(`CUDA-BEVFusion/src/common/dtype.hpp` at `f5c954d4f06f09e9caf4eafb8f19884661508fc7`).
It provides the shared `nvtype` types without linking BEVFusion. Keep it in sync
with BEVFusion when changing the shared type definitions.

Use `.cpp` for normal host code and `.cu` only for files with CUDA kernels (`__global__`, `<<<>>>`).
Both compile into the same `libsemvecnet_core.so`, so a `.cpp` calls a kernel through a plain C++
function declared in a header and defined in the `.cu` file (see `hello.hpp` / `hello.cu`).
Nothing is compiled until you add sources; until then `semvecnet_core` is an empty INTERFACE target.

## Develop (VS Code)

Edit on the host; build and run in a throwaway GPU container (no dev container needed).
Needs Docker + NVIDIA Container Toolkit.

1. Open the repository folder in VS Code on the host and allow automatic tasks when prompted.
   The header-sync task builds `semvecnet:dev-ort` if needed (CUDA 13.4, TensorRT 11.4,
   OpenCV and the ONNX Runtime GPU C++ SDK).
2. The task copies the development headers into `.sysroot/` (git-ignored) so host
   IntelliSense and clangd can find them. No Dev Container is needed. If automatic
   tasks are disabled, run `semvecnet: sync IntelliSense headers` once instead.
3. Build: `tool/build.sh` (or Ctrl+Shift+B). On the host it re-runs itself inside the container.
4. Run the [rosbag smoke test](#rosbag-smoke-test), or use `tool/docker_run.sh` for a shell.
5. In VS Code, build and run are separate: **Ctrl+Shift+B** builds. Run with the task
   "semvecnet: run (docker)" (Terminal > Run Task, or Run Test Task); it prints in the Terminal panel
   and ends when the program ends. **F5** is bound to this task for C++/CUDA files in the user
   keybindings (`~/.config/Code/User/keybindings.json`, see below), so F5 prints in the Terminal.
   For breakpoints use the green play button in Run and Debug ("semvecnet (debugger)"). Neither rebuilds.
   Program arguments: edit `command` in `.vscode/tasks.json` (run) or `args` in `launch.json` (F5).
   `SEMVECNET_QUIET=1` on `tool/docker_run.sh` hides NVIDIA's startup banner.

   The F5 binding (user-level, not in the repo) is:
   `{ "key": "f5", "command": "workbench.action.tasks.runTask", "args": "semvecnet: run (docker)", "when": "!inDebugMode && resourceExtname =~ /\\.(cpp|cu|hpp|cuh|h|cc)$/" }`

The repo is mounted at the same absolute path inside the container, so compiler errors are clickable.
`.devcontainer/` still works if you ever want a full in-container VS Code.

GPU architecture defaults to `sm_86` (RTX 30xx) in the image; use
`--build-arg CUDASM=89` or `CUDASM=89 tool/build.sh` for other GPUs.

## Rosbag smoke test

From the repository root:

```bash
CUDA-SemVecNet/tool/run_rosbag_smoke.py
```

This reads two synchronized frames from
`~/Desktop/data/bags/may_30_ring_road_test_5/` and runs the C++ pipeline with
all eight cameras and the merged LiDAR. To process every synchronized LiDAR
sweep from the start of the recording:

```bash
CUDA-SemVecNet/tool/run_rosbag_smoke.py --frames 0 --out /path/to/new/results
```

Each frame has the same six-digit filename stem in three directories:

- `polylines/`: JSON with timestamp, local odometry pose, and scored vectors
  in rear-axle coordinates (meters, x forward and y left), like the Python
  `vectors/` files.
- `semantic_maps/`: the 150 × 300 RGB semantic map before normalization.
- `combined/`: the semantic map next to the predicted polylines on the same
  extent, using the Python comparison view's class colors.

The C++ mapper accumulates sweeps throughout the run. It paints LiDAR returns;
the Python pipeline can additionally project camera pixels onto the road
plane, so the C++ maps can be sparser. Predictions use the C++ score filter,
so a vector list can be empty. The full run reports sweeps skipped for
missing camera synchronization or odometry rather than filling in data.

The script uses Docker with NVIDIA GPU support, the existing
`semvecnet-mcap-hrnet:latest` reader image, `model/hrnet_ocrnet_mapillary.plan`,
and `models/semvecnet.onnx` (model paths relative to `CUDA-SemVecNet/`). Use
`--start`, `--frames`, or `--out` to change the sample or output directory;
see `--help` for model overrides. It is configured for the sensor topics and
calibration of this recording, not a generic ROS bag player.

## Engine

```
tool/build_trt_engine.sh path/to/semvecnet.onnx   # -> model/semvecnet.plan
```
A `.plan` only works with the TensorRT version and GPU it was built on.

## Monorepo

See `../integration/README.md` for linking into `wato_monorepo`.

`SemVecNet::forward` takes both `lidar_to_map` and `ego_to_map` row-major 4x4
poses. The mapper uses the LiDAR pose to place points; the renderer uses the
ego pose to orient and center the model input. They may differ when the LiDAR
is offset from the vehicle's ego frame.

## Docker image

```
docker build -f docker/Dockerfile --target dev -t semvecnet:dev-ort .   # from CUDA-SemVecNet/
```
