#!/usr/bin/env bash
# Fine-tune SemVecNet (ResNet-50 encoder + MapTRv2 decoder) on the Ring Road dataset, on the WATcloud SLURM
# cluster, in one command from a login node:
#
#   ./fine-tune.sh [extra sbatch options, e.g. --gres shard:rtx_3090:16384,tmpdisk:102400]
#   EPOCHS=60 BATCH=4 LR=1e-4 ./fine-tune.sh      # hyperparameter overrides
#   ./fine-tune.sh --local                         # run the job body here (local Docker + GPU), no SLURM
#
# On the login node it checks the dataset and the start checkpoint (downloads it with rclone if missing),
# submits itself as a batch job and follows the job's log (Ctrl-C stops following; the job keeps running).
# Inside the job it:
#   1. starts rootless Docker (data in the job's /tmp tmpdisk) and pulls the SemVecNet image
#   2. finds the GPU this job was given: every candidate is tested with PyTorch inside the container
#   3. runs tools/finetune_ringroad.py: baseline test/train mAP of the start weights, then training with a
#      [TRAIN] line every 10 iterations and a [SUMMARY] line after every epoch (train loss, train mAP, test mAP,
#      per-class AP, best so far, ETA)
#   4. leaves best.pth, latest.pth, ringroad_finetuned.pth (= best) and metrics.json/csv in the output dir
#
# Environment overrides:
#   SEMVECNET_DATASET  dataset from tools/build_finetune_dataset.py (default ~/ringroad_dataset_large)
#   SEMVECNET_FT_OUT   output dir (default ~/semvecnet-finetune/<job id>, ~1 GB; /home quota is 20 GiB)
#   SEMVECNET_CKPT     start weights (default <repo>/SemVecNet/ckpts/mAP_epoch_99.pth, fetched if missing)
#   SEMVECNET_DOCKER_GPUS  force the docker --gpus value (skips the GPU search)
#   EPOCHS (40) BATCH (4) LR (1e-4) WORKERS (4) AUGMENT (1)
#
#SBATCH --job-name=semvecnet-finetune
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:16384,tmpdisk:102400
#SBATCH --time=12:00:00
set -euo pipefail

LOCAL=0
if [ "${1:-}" = "--local" ]; then LOCAL=1; shift; fi

if [ -z "${SLURM_JOB_ID:-}" ] || [ "$LOCAL" = 1 ]; then
    REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
else
    REPO="${SEMVECNET_REPO:?submit with ./fine-tune.sh from the repo}"
fi
DATASET="$(realpath -m "${SEMVECNET_DATASET:-$HOME/ringroad_dataset_large}")"
CKPT="$(realpath -m "${SEMVECNET_CKPT:-$REPO/SemVecNet/ckpts/mAP_epoch_99.pth}")"
IMAGE=naruarjun/semvecnet:v2
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
[ -f "$REPO/rclone.conf" ] && export RCLONE_CONFIG="$REPO/rclone.conf"   # gitignored credentials in the checkout

check_inputs() {
    for s in train test; do
        [ -f "$DATASET/$s/$s.pkl" ] || { echo "ERROR: $DATASET/$s/$s.pkl not found (set SEMVECNET_DATASET)" >&2; exit 1; }
    done
    if [ ! -s "$CKPT" ]; then
        [ -z "${SEMVECNET_CKPT:-}" ] || { echo "ERROR: checkpoint $CKPT not found" >&2; exit 1; }
        command -v rclone >/dev/null || { echo "ERROR: $CKPT missing and rclone not installed (run ./run_stage1_slurm.sh once, or copy the checkpoint there)" >&2; exit 1; }
        echo ">> Downloading the start checkpoint to $CKPT"
        mkdir -p "$(dirname "$CKPT")"
        rclone copyto --drive-root-folder-id 1bV7mnjSOYg2M-LBbM59C_eFDIvqvZZ7i \
            'gdrive:SemVecNet checkpoints/nuscenes-semantic/mAP_epoch_99.pth' "$CKPT.part"
        mv "$CKPT.part" "$CKPT"
    fi
}

if [ -z "${SLURM_JOB_ID:-}" ] && [ "$LOCAL" = 0 ]; then
    # ------------------------------------------------------------ login node: check, submit, follow
    command -v sbatch >/dev/null || { echo "ERROR: sbatch not found - run this on a WATcloud login node" >&2; exit 1; }
    check_inputs
    mkdir -p "$REPO/logs"
    JOB="$(sbatch --parsable --output="$REPO/logs/finetune-%j.out" --export="ALL,SEMVECNET_REPO=$REPO" "$@" "$REPO/fine-tune.sh")"
    JOB="${JOB%%;*}"
    LOG="$REPO/logs/finetune-$JOB.out"
    echo ">> Submitted job $JOB. Log: $LOG"
    echo "   Ctrl-C only stops following the log. Later: tail -f $LOG   |   squeue -u $USER   |   scancel $JOB"
    echo "   Progress only: grep -E '\\[(SUMMARY|TRAIN)\\]' $LOG"
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

# ---------------------------------------------------------------- inside the job (or --local)
OUT="$(realpath -m "${SEMVECNET_FT_OUT:-$HOME/semvecnet-finetune/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}}")"
cd "$REPO"
step() { echo; echo "==== [$(date '+%F %T')] $*"; }

step "Job ${SLURM_JOB_ID:-local} on $(hostname); repo $REPO ($(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "  dataset $DATASET, start weights $CKPT, output $OUT"
check_inputs
df -h /tmp "$HOME" | sed 's/^/  /'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader | sed 's/^/  GPU /' || true
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ "$LOCAL" = 0 ]; then
    step "Starting rootless Docker"
    slurm-start-dockerd.sh
    if ! docker info >/dev/null 2>&1; then
        for s in "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock" /tmp/run/docker.sock "/run/user/$(id -u)/docker.sock"; do
            if [ -S "$s" ] && DOCKER_HOST="unix://$s" docker info >/dev/null 2>&1; then export DOCKER_HOST="unix://$s"; break; fi
        done
    fi
fi
docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not reachable" >&2; exit 1; }
ROOTLESS="$(docker info --format '{{.SecurityOptions}}' | grep -o rootless || true)"
echo "  docker $(docker version --format '{{.Server.Version}}'), ${ROOTLESS:-rootful}"
# Rootless Docker maps container root to the calling user; rootful Docker needs --user to keep outputs ours.
USER_ARGS=(--user "$(id -u):$(id -g)")
[ -n "$ROOTLESS" ] && USER_ARGS=()

if docker image inspect "$IMAGE" >/dev/null 2>&1; then echo "  image $IMAGE present"
else step "Pulling the SemVecNet image $IMAGE (~7 GB download, ~24 GB unpacked)"; docker pull "$IMAGE"; fi

step "Finding this job's GPU inside Docker"
# The image's own entrypoint (/opt/nvidia/nvidia_entrypoint.sh) is kept, as run_mcap_pipeline.sh does.
# nvidia-smi can list GPUs that CUDA cannot open (another job's GPU on a shared node gives "CUDA unknown error"),
# so every candidate is tested with PyTorch and the first that works is used.
torch_ok() {
    docker run --rm --gpus "$1" "$IMAGE" python -c \
        'import torch; torch.cuda.init(); x = torch.ones(8, device="cuda"); print("OK", torch.cuda.get_device_name(0), "torch", torch.__version__, "cuda", torch.version.cuda, float(x.sum()))' \
        2>&1 | grep -E '^OK|Error|error' | tail -1
}
UUIDS="$(docker run --rm --gpus all --entrypoint nvidia-smi "$IMAGE" --query-gpu=uuid --format=csv,noheader 2>/dev/null || true)"
echo "  GPUs visible to Docker: $(echo "$UUIDS" | grep -c . || true)"
CANDIDATES=()
if [ -n "${SEMVECNET_DOCKER_GPUS:-}" ]; then
    CANDIDATES=("$SEMVECNET_DOCKER_GPUS")
else
    CVD="${CUDA_VISIBLE_DEVICES:-}"
    for d in ${CVD//,/ }; do CANDIDATES+=("\"device=$d\""); done
    for u in $UUIDS; do CANDIDATES+=("\"device=$u\""); done
    CANDIDATES+=(all)
fi
GPUS=""
for c in "${CANDIDATES[@]}"; do
    r="$(torch_ok "$c" || true)"
    echo "  --gpus $c: ${r:-no output}"
    if [[ "$r" == OK* ]]; then GPUS="$c"; break; fi
done
[ -n "$GPUS" ] || { echo "ERROR: PyTorch cannot use any GPU inside Docker (see above)" >&2; ls -l /dev/nvidia* >&2 || true; exit 1; }
echo "  using --gpus $GPUS"

step "Fine-tuning (EPOCHS=${EPOCHS:-40} BATCH=${BATCH:-4} LR=${LR:-1e-4} AUGMENT=${AUGMENT:-1}) -> $OUT"
mkdir -p "$OUT"
docker run --rm --gpus "$GPUS" --ipc=host "${USER_ARGS[@]}" \
    -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
    -v "$REPO/SemVecNet/projects:/workspace/SemVecNet/projects:ro" -v "$REPO/tools:/repo/tools:ro" \
    -v "$CKPT:/weights.pth:ro" -v "$DATASET:/dataset:ro" -v "$OUT:/out" -w /workspace/SemVecNet \
    "$IMAGE" python -u /repo/tools/finetune_ringroad.py --data-root /dataset --output /out --checkpoint /weights.pth \
        --epochs "${EPOCHS:-40}" --batch "${BATCH:-4}" --lr "${LR:-1e-4}" --workers "${WORKERS:-4}" \
        --augment "${AUGMENT:-1}" 2>&1 \
    | grep --line-buffered -v -E '^(=+|NVIDIA Release|PyTorch Version|Container image|Copyright|All rights|Various files|This container|By pulling|https://|NOTE: |   insufficient|   docker run|   Use the NVIDIA|WARNING: The NVIDIA)'

step "Done. Fine-tuned weights: $OUT/ringroad_finetuned.pth"
ls -la "$OUT" | sed 's/^/  /'
