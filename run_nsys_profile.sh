#!/usr/bin/env bash
# Profile a few frames from the middle of a ROS 2 MCAP bag with Nsight Systems.
#
#   ./run_nsys_profile.sh [BAG] [--frames N] [--start SECONDS] [--out DIR]
#
# Defaults:
#   BAG    = ~/Desktop/data/bags/may_30_ring_road_test_5
#   FRAMES = 3
#   START  = middle of the bag (first synchronized frame at or after it)
#   OUT    = ./nsys_reports/<timestamp>-<bag name>
#
# The bag is decoded by semvecnet-mcap-hrnet, then only the C++ inference
# executable is profiled (bag decoding is not in the nsys trace):
#
#   nsys profile --stats=true --trace=cuda,cudnn,cublas,osrt ./build/semvecnet
#
# The same extracted frames are also run once without nsys to write the usual
# smoke-test outputs: semantic_maps/, polylines/ and combined/ (semantic map
# next to the predicted polylines).
#
# CPU sampling is on by default (process-tree), so the timeline also shows the
# C++ call stacks for the forward() stages while the stats log gives the
# kernel/CUDA/CuDNN/cuBLAS/OS-runtime totals.
set -euo pipefail

REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
PROJECT="$REPO/CUDA-SemVecNet"
BAG="${BAG:-$HOME/Desktop/data/bags/may_30_ring_road_test_5}"
FRAMES="${FRAMES:-3}"
START="${START:-}"
OUT="${OUT:-}"
IMAGE="${SEMVECNET_IMAGE:-semvecnet:dev-ort}"
READER_IMAGE="${SEMVECNET_MCAP_IMAGE:-semvecnet-mcap-hrnet:latest}"
PLAN="${SEMVECNET_HRNET_PLAN:-$PROJECT/model/hrnet_ocrnet_mapillary.plan}"
MODEL="${SEMVECNET_ONNX:-$PROJECT/models/semvecnet.onnx}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [BAG] [options]

  BAG                  MCAP file or directory containing one .mcap
                       (default: ~/Desktop/data/bags/may_30_ring_road_test_5)
  --frames N           number of synchronized frames to profile (default: 3)
  --start SECONDS      first frame, as seconds from bag start (default: middle)
  --out DIR            report + frame-image directory
                       (default: ./nsys_reports/<timestamp>-<bag>)
  --image IMAGE        inference image (default: $IMAGE)
  --reader-image IMG   MCAP reader image (default: $READER_IMAGE)
  -h, --help           show this help
EOF
}

POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --frames) FRAMES="${2:?--frames needs a value}"; shift 2 ;;
    --start) START="${2:?--start needs a value}"; shift 2 ;;
    --out) OUT="${2:?--out needs a value}"; shift 2 ;;
    --image) IMAGE="${2:?--image needs a value}"; shift 2 ;;
    --reader-image) READER_IMAGE="${2:?--reader-image needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

[ ${#POSITIONAL[@]} -ge 1 ] && BAG="${POSITIONAL[0]}"
[ ${#POSITIONAL[@]} -ge 2 ] && FRAMES="${POSITIONAL[1]}"
[ ${#POSITIONAL[@]} -ge 3 ] && START="${POSITIONAL[2]}"
[ ${#POSITIONAL[@]} -le 3 ] || { echo "ERROR: too many arguments" >&2; exit 2; }

case "$FRAMES" in
  ''|*[!0-9]*|0) echo "ERROR: --frames must be a positive integer" >&2; exit 2 ;;
esac
if [ -n "$START" ]; then
  awk -v v="$START" 'BEGIN { exit !(v ~ /^[0-9]+([.][0-9]+)?$/) }' \
    || { echo "ERROR: --start must be a nonnegative number of seconds" >&2; exit 2; }
fi

BAG="${BAG/#\~/$HOME}"
[ -e "$BAG" ] || { echo "ERROR: bag not found: $BAG" >&2; exit 1; }
BAG="$(realpath "$BAG")"
if [ -d "$BAG" ]; then
  shopt -s nullglob
  MCAPS=("$BAG"/*.mcap)
  shopt -u nullglob
  [ ${#MCAPS[@]} -eq 1 ] || {
    echo "ERROR: expected exactly one .mcap in $BAG, found ${#MCAPS[@]}" >&2
    echo "       pass the .mcap file directly" >&2
    exit 1
  }
  BAG="${MCAPS[0]}"
fi
[ -f "$BAG" ] || { echo "ERROR: bag is not a file: $BAG" >&2; exit 1; }
[ -f "$PLAN" ] || { echo "ERROR: missing HRNet plan: $PLAN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: missing SemVecNet ONNX: $MODEL" >&2; exit 1; }

docker image inspect "$READER_IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: reader image '$READER_IMAGE' not found." >&2
  echo "       Set SEMVECNET_MCAP_IMAGE to a local image with Python, numpy, scipy," >&2
  echo "       mcap>=1.6 and mcap-ros2-support, or build semvecnet-mcap-hrnet:latest." >&2
  exit 1
}

echo ">> Building $IMAGE and CUDA-SemVecNet/build/semvecnet"
SEMVECNET_QUIET=1 "$PROJECT/tool/build.sh"

docker run --rm --entrypoint bash "$IMAGE" -lc 'command -v nsys >/dev/null' || {
  echo "ERROR: nsys is not installed in $IMAGE." >&2
  echo "       Rebuild the image or install Nsight Systems in it." >&2
  exit 1
}

# Bag time range. The MCAP summary is indexed, so this does not read the whole bag.
TIMES="$(docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$BAG:/bag.mcap:ro" --entrypoint python3 "$READER_IMAGE" -c '
import sys
from mcap.reader import make_reader

with open(sys.argv[1], "rb") as stream:
    summary = make_reader(stream).get_summary()
    if summary is None or summary.statistics is None:
        raise SystemExit("bag has no MCAP summary statistics")
    stats = summary.statistics
    print(stats.message_start_time / 1e9, stats.message_end_time / 1e9)
' /bag.mcap)"

read -r BAG_BEGIN BAG_END <<< "$TIMES"
[ -n "${BAG_END:-}" ] || { echo "ERROR: could not read bag time range" >&2; exit 1; }
if [ -z "$START" ]; then
  # run_rosbag_smoke.py --start is an offset from the first bag message.
  START="$(awk -v a="$BAG_BEGIN" -v b="$BAG_END" 'BEGIN { printf "%.3f", (b - a) / 2 }')"
  echo ">> Bag is $(awk -v a="$BAG_BEGIN" -v b="$BAG_END" 'BEGIN { printf "%.1f", b - a }')s long; using middle start ${START}s from bag start"
else
  echo ">> Using start ${START}s from bag start"
fi

if [ -z "$OUT" ]; then
  OUT="$REPO/nsys_reports/$(date +%Y%m%d-%H%M%S)-$(basename "$BAG" .mcap)"
fi
OUT="$(realpath -m "$OUT")"
mkdir -p "$OUT"

FRAMES_BIN="$(mktemp "${TMPDIR:-/tmp}/semvecnet-nsys-frames.XXXXXX.bin")"
trap 'rm -f "$FRAMES_BIN"' EXIT

echo ">> Extracting $FRAMES frame(s) from $START s"
docker run --rm --network none --log-driver none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PROJECT/tool/run_rosbag_smoke.py:/extract.py:ro" \
  -v "$BAG:/bag.mcap:ro" \
  --entrypoint python3 "$READER_IMAGE" \
  /extract.py --extract /bag.mcap "$START" "$FRAMES" > "$FRAMES_BIN"

[ -s "$FRAMES_BIN" ] || {
  echo "ERROR: no synchronized frames were extracted around $START s" >&2
  echo "       try a different --start (e.g. ${START%%.*} or 0.5)" >&2
  exit 1
}

REPORT_NAME="semvecnet-$(basename "$BAG" .mcap)"
FRAME_OUT="$OUT/frames"
mkdir -p "$FRAME_OUT"

echo ">> Rendering semantic maps, polylines and combined view to $FRAME_OUT"
docker run --rm -i --gpus all --shm-size 2g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
  -v "$PROJECT:$PROJECT:ro" \
  -v "$PLAN:/hrnet.plan:ro" \
  -v "$MODEL:/semvecnet.onnx:ro" \
  -v "$FRAME_OUT:/output" \
  -w "$PROJECT" \
  "$IMAGE" \
  "$PROJECT/build/semvecnet" /hrnet.plan /semvecnet.onnx /output \
  < "$FRAMES_BIN"

echo ">> Running nsys (cuda,cudnn,cublas,osrt) on $FRAMES frame(s)"
docker run --rm -i --gpus all --shm-size 2g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$PROJECT:$PROJECT:ro" \
  -v "$PLAN:/hrnet.plan:ro" \
  -v "$MODEL:/semvecnet.onnx:ro" \
  -v "$OUT:/reports" \
  -w "$PROJECT" \
  "$IMAGE" \
  nsys profile --stats=true --trace=cuda,cudnn,cublas,osrt \
    --force-overwrite=true -o "/reports/$REPORT_NAME" \
    "$PROJECT/build/semvecnet" /hrnet.plan /semvecnet.onnx /tmp/semvecnet-nsys-unused --profile \
  < "$FRAMES_BIN" 2>&1 | tee "$OUT/nsys-stats.txt"

echo
echo ">> Semantic maps:      $FRAME_OUT/semantic_maps/*.png"
echo ">> Polylines:          $FRAME_OUT/polylines/*.json"
echo ">> Combined views:     $FRAME_OUT/combined/*.png"
echo ">> Nsight Systems report: $OUT/$REPORT_NAME.nsys-rep"
echo ">> Stats log:             $OUT/nsys-stats.txt"
echo ">> Open the timeline with: nsys-ui $OUT/$REPORT_NAME.nsys-rep"
