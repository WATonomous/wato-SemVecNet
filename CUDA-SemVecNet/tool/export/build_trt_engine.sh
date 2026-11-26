#!/bin/bash
# Build a TensorRT engine (.plan) from the exported ONNX model.
# Run inside the dev container.  Usage:
#   tool/export/build_trt_engine.sh path/to/semvecnet.onnx [model/semvecnet.plan]
set -euo pipefail
cd "$(dirname "$0")/../.."

ONNX="${1:?usage: $0 <model.onnx> [output.plan]}"
PLAN="${2:-model/semvecnet.plan}"
TRTEXEC="${TensorRT_Bin:-/usr/src/tensorrt/bin}/trtexec"

mkdir -p "$(dirname "$PLAN")"
"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$PLAN" \
  --fp16 \
  --memPoolSize=workspace:2048 \
  --profilingVerbosity=detailed \
  --exportLayerInfo="${PLAN%.plan}.json" \
  2>&1 | tee "${PLAN%.plan}.log"
echo "Wrote $PLAN"
