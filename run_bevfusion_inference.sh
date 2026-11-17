#!/usr/bin/env bash
# One-shot: BEVFusion segmentation maps -> SemVecNet vector maps.
#
#   ./run_bevfusion_inference.sh [INPUT_DIR_OR_PNG] [OUTPUT_DIR] [extra infer_bevfusion.py args...]
#
# Defaults:
#   INPUT  = ~/Desktop/code/wato-cuda-bevfusion/CUDA-BEVFusion/build/outputs
#   OUTPUT = ./outputs
# Default covers the full +-50 m map by tiling; --mode crop = central 60x30 m only.
# e.g.  ./run_bevfusion_inference.sh "" "" --score-thr 0.3 --mode crop
set -euo pipefail

REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
INPUT="$(realpath "${1:-$HOME/Desktop/code/wato-cuda-bevfusion/CUDA-BEVFusion/build/outputs}")"
OUTPUT="$(realpath -m "${2:-$REPO/outputs}")"
shift $(( $# < 2 ? $# : 2 ))
IMAGE=semvecnet-bevfusion:latest
CKPT="$REPO/SemVecNet/ckpts/mAP_epoch_99.pth"

[ -e "$INPUT" ] || { echo "ERROR: input not found: $INPUT"; exit 1; }
[ -f "$CKPT" ]  || { echo "ERROR: missing $CKPT"
                     echo "Download SemVecNet weights from the link in README.md into SemVecNet/ckpts/"; exit 1; }

# Mount a directory either way; pass a file path through if a single PNG was given.
if [ -d "$INPUT" ]; then IN_DIR="$INPUT"; IN_ARG=/input
else IN_DIR="$(dirname "$INPUT")"; IN_ARG="/input/$(basename "$INPUT")"; fi
mkdir -p "$OUTPUT"

echo ">> Building $IMAGE (cached after first run)"
docker build -q -t "$IMAGE" -f "$REPO/docker/Dockerfile" "$REPO" >/dev/null

echo ">> Running SemVecNet on $INPUT"
docker run --rm --gpus all --shm-size 2g \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e PYTHONWARNINGS=ignore \
    -v "$IN_DIR:/input:ro" \
    -v "$CKPT:/workspace/SemVecNet/ckpts/mAP_epoch_99.pth:ro" \
    -v "$OUTPUT:/output" \
    "$IMAGE" --input "$IN_ARG" --output /output "$@" 2>&1 \
    | sed -n '/^Loaded\|^  scene\|^  .*\.png:\|^Done\|Error\|Traceback\|File "/p'

echo ">> Outputs in $OUTPUT  (<name>_vectors.png / <name>_vectors.json)"
