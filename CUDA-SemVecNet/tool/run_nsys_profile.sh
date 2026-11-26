#!/usr/bin/env bash
# Profile a few frames from the middle of a ROS 2 MCAP bag with Nsight Systems.
#
#   CUDA-SemVecNet/tool/run_nsys_profile.sh [BAG] [--frames N] [--start SECONDS] [--out DIR]
#
# Defaults:
#   BAG    = ~/Desktop/data/bags/may_30_ring_road_test_5
#   FRAMES = 10 (frame 0 is a warm-up and is left out of the statistics)
#   START  = middle of the bag (first synchronized frame at or after it)
#   OUT    = ./nsys_reports/<timestamp>-<bag name>
#
# The bag is decoded by semvecnet-mcap-hrnet, then only the C++ inference
# executable is profiled (bag decoding is not in the nsys trace):
#
#   nsys profile --trace=nvtx,cuda,cudnn,cublas,osrt ./build/semvecnet
#
# SemVecNet::forward() and the stage implementations use NVTX ranges.
# tool/eval/nsys_report.py turns those ranges into:
#
#   OUT/report.html          open in a browser: end-to-end time per input,
#                            per-stage breakdown, every frame, top CUDA kernels
#   OUT/stage_timings.csv    per-frame, per-stage milliseconds
#   OUT/frames/              semantic_maps/, polylines/, combined/ images
#   OUT/raw/                 .nsys-rep (open with nsys-ui), full nsys stats text
#
# A short summary is also printed to the terminal.
set -euo pipefail

PROJECT="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
REPO="$(dirname "$PROJECT")"
BAG="${BAG:-$HOME/Desktop/data/bags/may_30_ring_road_test_5}"
FRAMES="${FRAMES:-10}"
START="${START:-}"
OUT="${OUT:-}"
IMAGE="${SEMVECNET_IMAGE:-semvecnet:dev-ort}"
READER_IMAGE="${SEMVECNET_MCAP_IMAGE:-semvecnet-mcap-hrnet:latest}"
PLAN="${SEMVECNET_HRNET_PLAN:-$PROJECT/models/hrnet_ocrnet_mapillary_b8_s1.plan}"
MODEL="${SEMVECNET_ONNX:-$PROJECT/models/semvecnet.onnx}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [BAG] [options]

  BAG                  MCAP file or directory containing one .mcap
                       (default: ~/Desktop/data/bags/may_30_ring_road_test_5)
  --frames N           number of synchronized frames to profile (default: 10;
                       frame 0 is a warm-up excluded from the statistics)
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
  -v "$PROJECT/tool/eval/run_rosbag_smoke.py:/extract.py:ro" \
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

RAW="$OUT/raw"
mkdir -p "$RAW"
NSYS_LOG="$RAW/nsys-profile.log"

echo ">> Profiling $FRAMES frame(s) with nsys (nvtx,cuda,cudnn,cublas,osrt)"
docker run --rm -i --gpus all --shm-size 2g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$PROJECT:$PROJECT:ro" \
  -v "$PLAN:/hrnet.plan:ro" \
  -v "$MODEL:/semvecnet.onnx:ro" \
  -v "$RAW:/reports" \
  -w "$PROJECT" \
  "$IMAGE" \
  nsys profile --stats=false --trace=nvtx,cuda,cudnn,cublas,osrt \
    --force-overwrite=true -o "/reports/$REPORT_NAME" \
    "$PROJECT/build/semvecnet" /hrnet.plan /semvecnet.onnx /tmp/semvecnet-nsys-unused --profile \
  < "$FRAMES_BIN" > "$NSYS_LOG" 2>&1 || {
    echo "ERROR: nsys profile failed; last lines of $NSYS_LOG:" >&2
    tail -n 20 "$NSYS_LOG" >&2
    exit 1
  }

echo ">> Exporting nsys stats"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$RAW:/reports" -w /reports --entrypoint bash "$IMAGE" -c "
    set -e
    nsys stats --force-export=true --format csv --output stats \
      --report nvtx_pushpop_trace,cuda_gpu_kern_sum $REPORT_NAME.nsys-rep >/dev/null
    nsys stats $REPORT_NAME.nsys-rep > nsys-stats.txt 2>&1
  "

META="$(printf '{"bag": "%s", "start": "%s", "created": "%s", "files": [["frames/", "semantic_maps/, polylines/ and combined/ outputs"], ["stage_timings.csv", "per-frame, per-stage ms"], ["raw/%s.nsys-rep", "open with nsys-ui for the full timeline"], ["raw/nsys-stats.txt", "every nsys stats report as text"]]}' \
  "$(basename "$BAG")" "$START" "$(date '+%Y-%m-%d %H:%M')" "$REPORT_NAME")"

docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PROJECT/tool/eval/nsys_report.py:/nsys_report.py:ro" \
  -v "$OUT:/out" \
  --entrypoint python3 "$READER_IMAGE" -I /nsys_report.py \
    --nvtx "/out/raw/stats_nvtx_pushpop_trace.csv" \
    --kernels "/out/raw/stats_cuda_gpu_kern_sum.csv" \
    --out /out --meta "$META"

echo ">> Report:          $OUT/report.html"
echo ">> Stage timings:   $OUT/stage_timings.csv"
echo ">> Frame images:    $FRAME_OUT/{semantic_maps,polylines,combined}/"
echo ">> nsys timeline:   nsys-ui $RAW/$REPORT_NAME.nsys-rep"
