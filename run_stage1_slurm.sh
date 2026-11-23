#!/usr/bin/env bash
# Stage 1 (Ring Road fine-tuning data) on the WATcloud SLURM cluster, in one command from a login node:
#
#   ./run_stage1_slurm.sh [extra sbatch options, e.g. --gres shard:rtx_3090:16384,tmpdisk:102400]
#
# On the login node it checks rclone, submits itself as a batch job and follows the job's log (Ctrl-C stops
# following; the job keeps running). Inside the job it:
#   1. starts rootless Docker (data in the job's /tmp tmpdisk), pulls the HRNet image and checks the GPU
#   2. builds the HRNet TensorRT engine for this GPU model once (kept in $SEMVECNET_WORK/assets)
#   3. for each split of tools/phase1_plan_all.txt (SEMVECNET_PLAN overrides): downloads it to /tmp, makes the semantic maps, aligns them to
#      ring_road.osm, deletes the split
#   4. writes the train/test dataset (semantic maps + Ring Road ground-truth vectors) to <repo>/data
# Rerunning resumes: finished splits are skipped.
#
# Environment overrides:
#   SEMVECNET_WORK   assets + per-split stage-1 runs (default ~/semvecnet-phase1, ~3 GB; /home quota is 20 GiB)
#   SEMVECNET_DATA   dataset output (default <repo>/data)
#   rclone uses <repo>/rclone.conf if it exists (gitignored - it holds Google Drive credentials), else ~/.config/rclone
#
#SBATCH --job-name=semvecnet-stage1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:16384,tmpdisk:102400
#SBATCH --time=1-00:00:00
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # ------------------------------------------------------------ login node: check, submit, follow
    REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
    export PATH="$HOME/.local/bin:$PATH"
    [ -f "$REPO/rclone.conf" ] && export RCLONE_CONFIG="$REPO/rclone.conf"
    command -v sbatch >/dev/null || { echo "ERROR: sbatch not found - run this on a WATcloud login node" >&2; exit 1; }
    python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' \
        || { echo "ERROR: python3 >= 3.9 is needed (found $(python3 --version 2>&1))" >&2; exit 1; }
    if ! command -v rclone >/dev/null; then
        echo ">> Installing rclone into ~/.local/bin"
        mkdir -p "$HOME/.local/bin"
        tmp="$(mktemp -d)"
        curl -fsSL -o "$tmp/rclone.zip" https://downloads.rclone.org/rclone-current-linux-amd64.zip
        python3 -c 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); n = next(n for n in z.namelist() if n.endswith("/rclone")); open(sys.argv[2], "wb").write(z.read(n))' \
            "$tmp/rclone.zip" "$HOME/.local/bin/rclone"
        chmod +x "$HOME/.local/bin/rclone"
        rm -rf "$tmp"
    fi
    if ! rclone listremotes 2>/dev/null | grep -qx 'gdrive:'; then
        cat >&2 <<'EOF'
ERROR: rclone has no remote named "gdrive". It needs access to the shared "Bags" Drive folder and the model
weights folder. Either copy a working config from your laptop into this checkout (it is gitignored):
    scp <laptop repo>/rclone.conf <server>:<this repo>/rclone.conf
or create one here with `rclone config` (new remote "gdrive", type "drive", answer "n" to auto config and run the
printed `rclone authorize` command on a machine with a browser). Then rerun this script.
EOF
        exit 1
    fi
    rclone lsf --max-depth 1 --drive-root-folder-id 127Kw3o7Org474rkK1wwMDRZKiYHaR0Db gdrive: >/dev/null \
        || { echo "ERROR: rclone remote gdrive cannot list the shared Bags folder" >&2; exit 1; }
    mkdir -p "$REPO/logs"
    JOB="$(sbatch --parsable --output="$REPO/logs/stage1-%j.out" --export="ALL,SEMVECNET_REPO=$REPO" "$@" "$REPO/run_stage1_slurm.sh")"
    JOB="${JOB%%;*}"
    LOG="$REPO/logs/stage1-$JOB.out"
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
REPO="${SEMVECNET_REPO:?submit with ./run_stage1_slurm.sh from the repo}"
[ -f "$REPO/rclone.conf" ] && export RCLONE_CONFIG="$REPO/rclone.conf"   # gitignored credentials in the checkout
WORK="${SEMVECNET_WORK:-$HOME/semvecnet-phase1}"
DATA="${SEMVECNET_DATA:-$REPO/data}"
CACHE=/tmp/semvecnet_bags   # job-local tmpdisk: the largest split is ~22 GB
BASE_IMAGE=naruarjun/tensorrt-nvidia-semseg
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
cd "$REPO"
step() { echo; echo "==== [$(date '+%F %T')] $*"; }

step "Job $SLURM_JOB_ID on $(hostname); repo $REPO ($(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "  work dir $WORK, dataset -> $DATA, bag cache $CACHE"
df -h /tmp "$HOME" | sed 's/^/  /'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU /' || true
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

step "Starting rootless Docker"
slurm-start-dockerd.sh
if ! docker info >/dev/null 2>&1; then
    for s in "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock" /tmp/run/docker.sock "/run/user/$(id -u)/docker.sock"; do
        if [ -S "$s" ] && DOCKER_HOST="unix://$s" docker info >/dev/null 2>&1; then export DOCKER_HOST="unix://$s"; break; fi
    done
fi
docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not reachable after slurm-start-dockerd.sh" >&2; exit 1; }
echo "  docker $(docker version --format '{{.Server.Version}}'), $(docker info --format '{{.SecurityOptions}}' | grep -o rootless || echo rootful)"

step "Pulling the HRNet image $BASE_IMAGE (~10 GB download, ~31 GB unpacked in /tmp)"
docker pull "$BASE_IMAGE"

step "Checking the GPU inside Docker"
SEEN="$(docker run --rm --gpus all --entrypoint nvidia-smi "$BASE_IMAGE" -L)"
echo "$SEEN" | sed 's/^/  /'
# If a container sees more GPUs than this job's own, pick the job's GPU by index (shards share GPUs between jobs).
if [ "$(echo "$SEEN" | grep -c '^GPU')" -gt 1 ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export SEMVECNET_DOCKER_GPUS="\"device=$CUDA_VISIBLE_DEVICES\""
    echo "  using --gpus $SEMVECNET_DOCKER_GPUS"
fi
docker run --rm --gpus "${SEMVECNET_DOCKER_GPUS:-all}" --entrypoint python "$BASE_IMAGE" -c \
    'import pycuda.driver as d; d.init(); print("  CUDA ok:", d.Device(0).name())'

# One engine per GPU model: a later (resumed) job may land on a different GPU type.
GPU_NAME="$(docker run --rm --gpus "${SEMVECNET_DOCKER_GPUS:-all}" --entrypoint nvidia-smi "$BASE_IMAGE" \
    --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_')"
export SEMVECNET_ENGINE="$WORK/assets/hrnet_${GPU_NAME}.engine"
if [ -f "$SEMVECNET_ENGINE" ]; then echo "  TensorRT engine for $GPU_NAME already built"
else echo "  TensorRT engine for $GPU_NAME will be built first (~15 min, once per GPU model)"; fi

step "Stage 1: per split download -> semantic maps -> alignment -> delete; then the train/test export"
python3 -u tools/run_phase1.py --work-dir "$WORK" --cache "$CACHE" --dataset-out "$DATA" --preview-port 0

step "Done. Dataset in $DATA"
ls "$DATA" | sed 's/^/  /'
