# SemVecNet Build Guide

## Quick Start

Press **Ctrl+Shift+B** in VSCode to build (uses Docker automatically).

---

## Build Methods

### 1. VSCode (Recommended)
```
Ctrl+Shift+B  →  builds in Docker with Ninja
```

**First time:** Builds the `semvecnet:dev` Docker image (~5-10 minutes)  
**After that:** Fast incremental builds

### 2. Command Line (Docker)
```bash
cd CUDA-SemVecNet
tool/build.sh              # RelWithDebInfo (optimized + debug symbols)
tool/build.sh Release      # Release (optimized, no debug)
tool/build.sh Debug        # Debug (no optimization)
```

### 3. Command Line (Host - if TensorRT is installed)
```bash
cd CUDA-SemVecNet
mkdir -p build && cd build
CUDACXX=/opt/cuda/bin/nvcc cmake ..
make -j$(nproc)
```

---

## Running Programs

### Test the .plan loader:
```bash
# In Docker:
tool/docker_run.sh build/test_plan_loader ~/Desktop/code/wato-cuda-bevfusion/CUDA-BEVFusion/model/seg/build/camera.vtransform.plan

# Or on host (if TensorRT installed):
./CUDA-SemVecNet/build/test_plan_loader ~/Desktop/code/wato-cuda-bevfusion/CUDA-BEVFusion/model/seg/build/camera.vtransform.plan
```

### Interactive shell in Docker:
```bash
tool/docker_run.sh
# Now you're inside the container with TensorRT
```

---

## Troubleshooting

### "generator : Ninja does not match Unix Makefiles"
**Fix:** Clean the build directory
```bash
rm -rf CUDA-SemVecNet/build/*
```

### "CMAKE_CUDA_COMPILER not found"
**Fix:** Set CUDACXX environment variable
```bash
CUDACXX=/opt/cuda/bin/nvcc cmake ..
```
Or use Docker (already configured).

### "Could NOT find TensorRT"
**Solution:** Use Docker - TensorRT is already installed in the container.

---

## Project Structure

```
CUDA-SemVecNet/
├── src/
│   ├── semvecnet/           # Model code (engine loading, inference)
│   ├── common/              # Shared utilities
│   ├── plugins/             # TensorRT plugins
│   ├── main.cpp             # Demo executable
│   └── test_plan_loader.cpp # Test program for loading .plan files
├── include/                 # Public headers
├── docker/
│   └── Dockerfile           # Dev environment (CUDA 11.8 + TensorRT 8.5.3)
├── tool/
│   ├── build.sh             # Build script (runs in Docker)
│   └── docker_run.sh        # Docker wrapper
├── build/                   # Build output (created by CMake)
└── CMakeLists.txt           # Build configuration
```

---

## Available Targets

After building, you'll have:
- **`libsemvecnet_core.so`** - Main library
- **`libsemvecnet_plugins.so`** - TensorRT plugins (if any in src/plugins/)
- **`semvecnet`** - Demo executable (if src/main.cpp exists)
- **`test_plan_loader`** - Test program for loading .plan files

---

## VS Code Tasks

| Task | Shortcut | Description |
|------|----------|-------------|
| Build | `Ctrl+Shift+B` | Build in Docker with Ninja |
| Run | `Ctrl+Shift+P` → "Run Task" | Run the demo |
| Shell | `Ctrl+Shift+P` → "Run Task" | Interactive Docker shell |

---

## Docker Image Details

**Base:** `nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04`

**Includes:**
- CUDA 11.8
- TensorRT 8.5.3
- cuDNN 8.6
- CMake, Ninja, GCC
- Matches wato-cuda-bevfusion environment

**Rebuild after Dockerfile changes:**
```bash
docker build -f docker/Dockerfile --target dev -t semvecnet:dev .
```
