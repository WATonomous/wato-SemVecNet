#!/usr/bin/env bash
# Old vs fine-tuned SemVecNet weights on a rosbag, on the WATcloud SLURM cluster, in one command from a login node:
#
#   ./compare-weights.sh [BAG] [stage-1 options of run_mcap_pipeline.sh, e.g. --cameras a,b]
#
#   BAG  path inside the shared Drive "Bags" folder (downloaded with rclone), or a .mcap file on the server.
#        Default: recording_20260224_213947/recording_20260224_213947_1.mcap, a held-out test drive (never trained on).
#
# Runs the same pipeline as run_mcap_pipeline.sh, once for stage 1 and twice for stages 2+3+viz:
#   1. stage 1 (HRNet + LiDAR semantic maps) on a slice of the bag, with the fine-tuning data settings
#      (--map-hz 5 --near-map; the eight pano cameras for the default bag)
#   2. a pose/map offset from the semantic maps (tools/mcap_estimate_offset.py), so both models are scored against
#      the same, aligned ground truth
#   3. run_mcap_pipeline.sh --stages 23 with the old weights, then with the new ones (inference, Lanelet2 accuracy,
#      all the usual videos)
#   4. tools/mcap_compare_weights.py: old vs new metrics table and a side-by-side video
#
# Environment overrides:
#   START (0) DURATION (300)   bag seconds to process (DURATION=0: the whole bag)
#   SEMVECNET_OLD_CKPT   default <repo>/SemVecNet/ckpts/mAP_epoch_99.pth (fetched with rclone if missing)
#   SEMVECNET_NEW_CKPT   default the newest ~/semvecnet-finetune/*/ringroad_finetuned.pth
#   SEMVECNET_CMP_OUT    output dir (default ~/semvecnet-compare/<job id>, ~1-2 GB for 5 minutes of bag)
#   SEMVECNET_WORK       stage-1 assets (HRNet weights, TensorRT engines) from run_stage1_slurm.sh (~/semvecnet-phase1)
#   CLASS_THR            score thresholds for drawing / precision / recall (default run_mcap_pipeline.sh's divider=0.1)
#   SBATCH_EXTRA         extra sbatch options, e.g. SBATCH_EXTRA="--time=8:00:00"
#
# GPU: RTX 3090 only (the SemVecNet image cannot initialise CUDA on the RTX 4090 node; see fine-tune.sh).
#
#SBATCH --job-name=semvecnet-compare
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:rtx_3090:16384,tmpdisk:102400
#SBATCH --time=4:00:00
set -euo pipefail

DEFAULT_BAG=recording_20260224_213947/recording_20260224_213947_1.mcap
PANO8=camera_pano_nn,camera_pano_ne,camera_pano_ee,camera_pano_se,camera_pano_ss,camera_pano_sw,camera_pano_ww,camera_pano_nw
BAGS_FOLDER_ID=127Kw3o7Org474rkK1wwMDRZKiYHaR0Db      # Drive folder "Bags"
WEIGHTS_FOLDER_ID=1bV7mnjSOYg2M-LBbM59C_eFDIvqvZZ7i   # Drive weights folder
SVN_IMAGE=naruarjun/semvecnet:v2
HRNET_BASE=naruarjun/tensorrt-nvidia-semseg
HRNET_IMAGE=semvecnet-mcap-hrnet:phase1
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # ------------------------------------------------------------ login node: check, submit, follow
    REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
    [ -f "$REPO/rclone.conf" ] && export RCLONE_CONFIG="$REPO/rclone.conf"
    command -v sbatch >/dev/null || { echo "ERROR: sbatch not found - run this on a WATcloud login node" >&2; exit 1; }
    BAG="$DEFAULT_BAG"
    if [ $# -gt 0 ] && [[ "$1" != -* ]]; then BAG="$1"; shift; fi
    [ -e "$BAG" ] && BAG="$(realpath "$BAG")"
    OLD="$(realpath -m "${SEMVECNET_OLD_CKPT:-$REPO/SemVecNet/ckpts/mAP_epoch_99.pth}")"
    NEW="${SEMVECNET_NEW_CKPT:-$(ls -t "$HOME"/semvecnet-finetune/*/ringroad_finetuned.pth 2>/dev/null | head -1 || true)}"
    [ -n "$NEW" ] && [ -s "$NEW" ] || { echo "ERROR: no fine-tuned weights (run ./fine-tune.sh or set SEMVECNET_NEW_CKPT)" >&2; exit 1; }
    NEW="$(realpath "$NEW")"
    if [ ! -e "$BAG" ] || [ ! -s "$OLD" ]; then
        command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -qx 'gdrive:' \
            || { echo "ERROR: rclone remote gdrive is needed (see run_stage1_slurm.sh)" >&2; exit 1; }
    fi
    if [ ! -s "$OLD" ]; then
        [ -z "${SEMVECNET_OLD_CKPT:-}" ] || { echo "ERROR: $OLD not found" >&2; exit 1; }
        echo ">> Downloading the original weights to $OLD"
        mkdir -p "$(dirname "$OLD")"
        rclone copyto --drive-root-folder-id "$WEIGHTS_FOLDER_ID" \
            'gdrive:SemVecNet checkpoints/nuscenes-semantic/mAP_epoch_99.pth' "$OLD.part"
        mv "$OLD.part" "$OLD"
    fi
    if [ ! -e "$BAG" ]; then
        [ -n "$(rclone lsf --drive-root-folder-id "$BAGS_FOLDER_ID" "gdrive:$BAG" 2>/dev/null)" ] \
            || { echo "ERROR: $BAG is neither a file here nor in the Drive Bags folder" >&2; exit 1; }
    fi
    echo ">> Bag $BAG (from ${START:-0} s, $([ "${DURATION:-300}" = 0 ] && echo 'to the end' || echo "${DURATION:-300} s"))"
    echo "   old weights $OLD"
    echo "   new weights $NEW"
    export SEMVECNET_OLD_CKPT="$OLD" SEMVECNET_NEW_CKPT="$NEW"
    mkdir -p "$REPO/logs"
    # shellcheck disable=SC2086
    JOB="$(sbatch --parsable --output="$REPO/logs/compare-%j.out" --export="ALL,SEMVECNET_REPO=$REPO" ${SBATCH_EXTRA:-} \
        "$REPO/compare-weights.sh" "$BAG" "$@")"
    JOB="${JOB%%;*}"
    LOG="$REPO/logs/compare-$JOB.out"
    echo ">> Submitted job $JOB. Log: $LOG"
    echo "   Ctrl-C only stops following the log. Later: tail -f $LOG   |   squeue -u $USER   |   scancel $JOB"
    while [ ! -f "$LOG" ] && [ -n "$(squeue -h -j "$JOB" 2>/dev/null)" ]; do
        squeue -h -j "$JOB" -o '   job %i: %T %R' 2>/dev/null
        sleep 30
    done
    [ -f "$LOG" ] || { echo "Job $JOB left the queue without a log; see: sacct -j $JOB" >&2; exit 1; }
    tail -n +1 -F "$LOG" &
    TAIL=$!
    trap 'kill $TAIL 2>/dev/null' EXIT
    while [ -n "$(squeue -h -j "$JOB" 2>/dev/null)" ]; do sleep 30; done
    sleep 5
    sacct -j "$JOB" --format=JobID,State,ExitCode,Elapsed -X 2>/dev/null || true
    exit 0
fi

# ---------------------------------------------------------------- inside the job
REPO="${SEMVECNET_REPO:?submit with ./compare-weights.sh from the repo}"
[ -f "$REPO/rclone.conf" ] && export RCLONE_CONFIG="$REPO/rclone.conf"
BAG="$1"; shift
WORK="${SEMVECNET_WORK:-$HOME/semvecnet-phase1}"
OUT="$(realpath -m "${SEMVECNET_CMP_OUT:-$HOME/semvecnet-compare/$SLURM_JOB_ID}")"
OLD="${SEMVECNET_OLD_CKPT:?}" NEW="${SEMVECNET_NEW_CKPT:?}"
CACHE=/tmp/semvecnet_bags
cd "$REPO"
step() { echo; echo "==== [$(date '+%F %T')] $*"; }

# Stage-1 options: the fine-tuning data settings (tools/phase1_plan_b.txt), the slice, then the user's own options.
S1=(--map-hz 5 --near-map --start "${START:-0}")
[ "${DURATION:-300}" != 0 ] && S1+=(--duration "${DURATION:-300}")
[ "$BAG" = "$DEFAULT_BAG" ] && S1+=(--cameras "$PANO8" --expect-cameras 8)
S1+=("$@")
S23=()
[ -n "${CLASS_THR:-}" ] && S23+=(--class-thr "$CLASS_THR")

step "Job $SLURM_JOB_ID on $(hostname); repo $REPO ($(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "  bag $BAG; stage-1 options ${S1[*]}"
echo "  old $OLD"
echo "  new $NEW"
echo "  output $OUT"
df -h /tmp "$HOME" | sed 's/^/  /'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU /' || true
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
mkdir -p "$OUT"
[ -z "$(ls -A "$OUT")" ] || { echo "ERROR: $OUT is not empty" >&2; exit 1; }

step "Starting rootless Docker"
slurm-start-dockerd.sh
if ! docker info >/dev/null 2>&1; then
    for s in "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock" /tmp/run/docker.sock "/run/user/$(id -u)/docker.sock"; do
        if [ -S "$s" ] && DOCKER_HOST="unix://$s" docker info >/dev/null 2>&1; then export DOCKER_HOST="unix://$s"; break; fi
    done
fi
docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not reachable after slurm-start-dockerd.sh" >&2; exit 1; }
USER_ARGS=(--user "$(id -u):$(id -g)")
docker info --format '{{.SecurityOptions}}' | grep -q rootless && USER_ARGS=()

if [ -f "$BAG" ]; then
    BAGFILE="$BAG"
else
    step "Downloading $BAG"
    mkdir -p "$CACHE"
    BAGFILE="$CACHE/$(basename "$BAG")"
    rclone copyto --drive-root-folder-id "$BAGS_FOLDER_ID" "gdrive:$BAG" "$BAGFILE" --multi-thread-streams 8 \
        --stats 30s --stats-one-line --stats-log-level NOTICE
    ls -lh "$BAGFILE" | sed 's/^/  /'
fi

step "Pulling the HRNet image $HRNET_BASE (~10 GB download, ~31 GB unpacked in /tmp)"
docker pull -q "$HRNET_BASE"
SEEN="$(docker run --rm --gpus all --entrypoint nvidia-smi "$HRNET_BASE" -L)"
echo "$SEEN" | sed 's/^/  /'
if [ "$(echo "$SEEN" | grep -c '^GPU')" -gt 1 ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export SEMVECNET_DOCKER_GPUS="\"device=$CUDA_VISIBLE_DEVICES\""
    echo "  using --gpus $SEMVECNET_DOCKER_GPUS"
fi
GPU_NAME="$(docker run --rm --gpus "${SEMVECNET_DOCKER_GPUS:-all}" --entrypoint nvidia-smi "$HRNET_BASE" \
    --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_')"
export SEMVECNET_ENGINE="$WORK/assets/hrnet_${GPU_NAME}.engine" SEMVECNET_ASSETS="$WORK/assets" \
       SEMVECNET_HRNET_IMAGE="$HRNET_IMAGE"
if [ -f "$SEMVECNET_ENGINE" ]; then echo "  TensorRT engine for $GPU_NAME already built"
else echo "  TensorRT engine for $GPU_NAME will be built first (~15 min, once per GPU model)"; fi

step "Stage 1: semantic maps (kept in $OUT/stage1, shared by both models)"
"$REPO/run_mcap_pipeline.sh" "$BAGFILE" -o "$OUT/stage1" --stages 1 "${S1[@]}"
N="$(grep -c . "$OUT/stage1/frames.jsonl" 2>/dev/null || true)"
[ "${N:-0}" -gt 0 ] || { echo "ERROR: stage 1 wrote no keyframes (is this slice of the bag on the Ring Road map?)" >&2; exit 1; }
echo "  $N keyframes"

step "Freeing /tmp: the bag and the HRNet images are no longer needed"
[ "$BAGFILE" = "$BAG" ] || rm -f "$BAGFILE"
docker image rm "$HRNET_IMAGE" "$HRNET_BASE" >/dev/null 2>&1 || true
df -h /tmp | sed 's/^/  /'

step "Pulling the SemVecNet image $SVN_IMAGE (~7 GB download, ~24 GB unpacked)"
docker pull -q "$SVN_IMAGE"
R="$(docker run --rm --gpus "${SEMVECNET_DOCKER_GPUS:-all}" "$SVN_IMAGE" python -c \
    'import torch; torch.cuda.init(); print("OK", torch.cuda.get_device_name(0))' 2>&1 | grep -E '^OK|Error' | tail -1 || true)"
echo "  ${R:-no output}"
[[ "$R" == OK* ]] || { echo "ERROR: PyTorch cannot use the GPU in the SemVecNet image (it needs an RTX 3090)" >&2; exit 1; }

step "Pose/map offset from the semantic maps (same aligned ground truth for both models)"
docker run --rm --entrypoint python "${USER_ARGS[@]}" -e HOME=/tmp -e PYTHONWARNINGS=ignore \
    -v "$REPO/tools:/repo/tools:ro" -v "$REPO/ring_road.osm:/gt/map.osm:ro" -v "$OUT/stage1:/run" "$SVN_IMAGE" \
    /repo/tools/mcap_estimate_offset.py --run /run --osm /gt/map.osm || true
OFFSET="$(python3 -c 'import json, sys; print(*json.load(open(sys.argv[1]))["run_offset"])' \
    "$OUT/stage1/offset/offset.json" 2>/dev/null || true)"
if [ -n "$OFFSET" ]; then
    echo "  using --pose-offset $OFFSET for both models"
    # shellcheck disable=SC2206
    S23+=(--pose-offset $OFFSET)
else
    echo "  WARNING: no offset estimate; each model's evaluation estimates its own offset"
fi

for W in old new; do
    CK="$OLD"; [ "$W" = new ] && CK="$NEW"
    step "Stages 2+3 with the $W weights ($CK)"
    cp -al "$OUT/stage1" "$OUT/$W"   # hard links: no extra space; stages 2/3 only add new files
    # The bag argument is only used by stage 1; the stage-1 run directory stands in for it here.
    SEMVECNET_CKPT="$CK" "$REPO/run_mcap_pipeline.sh" "$OUT/stage1" -o "$OUT/$W" --stages 23 "${S23[@]}"
done

step "Old vs new"
docker run --rm --entrypoint python "${USER_ARGS[@]}" -e HOME=/tmp -v "$REPO/tools:/repo/tools:ro" -v "$OUT:/cmp" \
    "$SVN_IMAGE" /repo/tools/mcap_compare_weights.py --old /cmp/old --new /cmp/new --out /cmp

step "Done. Outputs in $OUT ($(du -sh "$OUT" | cut -f1))"
cat <<EOF
  old_vs_new.mp4, old_vs_new/        cameras | semantic map | OLD | NEW | ground truth | OLD overlay | NEW overlay
  metrics_old_vs_new.md              accuracy table (also printed above as [RESULT] lines)
  old/, new/                         everything run_mcap_pipeline.sh makes, per model:
      compare.mp4                    cameras + LiDAR | semantic map | SemVecNet | ground truth | overlay
      semvecnet.mp4                  camera | semantic map | predicted polylines
      eval/eval.mp4, eval/junctions.png, eval/global_overlay.png, eval/metrics.json
Copy to your laptop:  scp -r -i ~/.ssh/watcloud $USER@wato-login1.ext.watonomous.ca:$OUT ~/Desktop/data/
EOF
