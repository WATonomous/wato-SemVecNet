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

1. One-time: `tool/docker_run.sh true` builds the `semvecnet:dev` image (CUDA 13.4 + TensorRT 11.4, same as the monorepo bevfusion build).
2. Run `tool/sync_headers.sh` to copy CUDA, TensorRT, and OpenCV headers into `.sysroot/`
   (git-ignored), which host IntelliSense reads (`.vscode/c_cpp_properties.json` for the
   Microsoft C/C++ extension, generated `.clangd` for clangd). Reload the VS Code window after.
   If an older image lacks OpenCV, rebuild it or set `SEMVECNET_OPENCV_IMAGE` to a local build
   image that already has the OpenCV development headers.
3. Build: `tool/build.sh` (or Ctrl+Shift+B). On the host it re-runs itself inside the container.
4. Run: `tool/docker_run.sh build/semvecnet model/semvecnet.plan`, or `tool/docker_run.sh` for a shell.
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

## Engine

```
tool/build_trt_engine.sh path/to/semvecnet.onnx   # -> model/semvecnet.plan
```
A `.plan` only works with the TensorRT version and GPU it was built on.

## Monorepo

See `../integration/README.md` for linking into `wato_monorepo`.

## Docker image

```
docker build -f docker/Dockerfile --target build -t semvecnet:build .   # from CUDA-SemVecNet/
```
