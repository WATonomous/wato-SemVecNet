#!/bin/bash
# Configure + build into CUDA-SemVecNet/build.  Usage: tool/build.sh [RelWithDebInfo|Release|Debug]
# Run on the host or in a container: on the host it re-runs itself inside Docker.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${SEMVECNET_IN_DOCKER:-}" ] && [ ! -f /.dockerenv ]; then
  exec tool/docker_run.sh tool/build.sh "$@"
fi

# CMake's Ninja generator and Ninja both update the same build log.
# Keep overlapping build tasks from configuring or building this directory together.
mkdir -p build
exec 9>build/.build.lock
flock 9

# RelWithDebInfo = optimized + debug symbols, so F5 breakpoints work. Pass Release for a lean build.
BUILD_TYPE="${1:-RelWithDebInfo}"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build build
