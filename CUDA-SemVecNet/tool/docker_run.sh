#!/bin/bash
# Run a command in a throwaway GPU container (no dev container needed).
# The repo is mounted at the SAME absolute path as on the host, so paths in build output,
# compiler errors and compile_commands.json match what VS Code sees.
#
#   tool/docker_run.sh                     # interactive shell
#   tool/docker_run.sh tool/build.sh       # build
#   tool/docker_run.sh build/semvecnet model/semvecnet.plan
#
# Builds the image first if it does not exist. Rebuild manually after Dockerfile changes:
#   docker build -f docker/Dockerfile --target dev -t semvecnet:dev-ort .
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_DIR="$(dirname "$PROJECT_DIR")"
IMAGE="${SEMVECNET_IMAGE:-semvecnet:dev-ort}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE not found, building it (one-time, a few minutes)..." >&2
  docker build -f "$PROJECT_DIR/docker/Dockerfile" --target dev -t "$IMAGE" "$PROJECT_DIR"
fi

# Header sync only needs the image, not a running GPU container.
if [ "${1:-}" = "--prepare-image" ]; then
  exit 0
fi

# -i always (stdin is forwarded, e.g. for the VS Code debugger); -t only on a real terminal.
TTY_FLAGS=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_FLAGS=(-it)
[ $# -eq 0 ] && set -- bash

# SEMVECNET_QUIET=1 skips the NVIDIA entrypoint banner (keeps stdout clean for the VS Code debugger).
EXTRA_FLAGS=()
[ -n "${SEMVECNET_QUIET:-}" ] && EXTRA_FLAGS=(--entrypoint "")

# Run as the host user so build/ files are not owned by root.
exec docker run --rm "${TTY_FLAGS[@]}" "${EXTRA_FLAGS[@]}" \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e SEMVECNET_IN_DOCKER=1 \
  -v "$REPO_DIR:$REPO_DIR" \
  -w "$PROJECT_DIR" \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --network host \
  "$IMAGE" "$@"
