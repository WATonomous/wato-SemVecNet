#!/bin/bash
# Configure + build into CUDA-SemVecNet/build.  Usage: tool/build.sh [RelWithDebInfo|Release|Debug]
# Run on the host or in a container: on the host it re-runs itself inside semvecnet:dev.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${SEMVECNET_IN_DOCKER:-}" ] && [ ! -f /.dockerenv ]; then
  exec tool/docker_run.sh tool/build.sh "$@"
fi

# RelWithDebInfo = optimized + debug symbols, so F5 breakpoints work. Pass Release for a lean build.
BUILD_TYPE="${1:-RelWithDebInfo}"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build build
