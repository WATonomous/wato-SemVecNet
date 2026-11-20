#!/bin/bash
# Build and develop SemVecNet inside Docker with TensorRT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="semvecnet-dev:latest"

echo "Building Docker image..."
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/docker/Dockerfile" "$SCRIPT_DIR"

echo ""
echo "Starting development container..."
docker run --rm -it \
    --gpus all \
    -v "$SCRIPT_DIR:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" bash -c "
        mkdir -p build && cd build
        cmake .. && make -j\$(nproc)
        echo ''
        echo '=== Build complete ==='
        echo 'Binaries:'
        ls -lh test_plan_loader 2>/dev/null || true
        ls -lh semvecnet 2>/dev/null || true
    "
