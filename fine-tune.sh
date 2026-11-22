#!/usr/bin/env bash
#SBATCH --job-name=semvecnet-finetune
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:16384,tmpdisk:102400
#SBATCH --time=12:00:00
set -euo pipefail

repo=${SEMVECNET_REPO:-$(dirname "$(realpath "$0")")}
dataset=$(realpath -m "${SEMVECNET_DATASET:-$HOME/ringroad_dataset_large}")
for split in train test; do
    [[ -r "$dataset/$split/$split.pkl" ]] || { echo "Missing $dataset/$split/$split.pkl" >&2; exit 1; }
done

local_run=false
if [[ ${1:-} == --local ]]; then
    local_run=true
    shift
elif [[ -z ${SLURM_JOB_ID:-} ]]; then
    command -v sbatch >/dev/null || { echo "sbatch not found - run on a WATcloud login node" >&2; exit 1; }
    mkdir -p "$repo/logs"
    job=$(sbatch --parsable --output="$repo/logs/finetune-%j.out" \
        --export="ALL,SEMVECNET_REPO=$repo" "$@" "$repo/fine-tune.sh")
    job=${job%%;*}
    log="$repo/logs/finetune-$job.out"
    echo "Submitted $job. Ctrl-C stops following $log and leaves the job running."
    while [[ ! -f $log && -n $(squeue -h -j "$job" 2>/dev/null) ]]; do
        squeue -h -j "$job" -o 'job %i: %T %R' || true
        sleep 30
    done
    [[ -f $log ]] || { echo "Job $job left the queue without a log" >&2; exit 1; }
    tail -n +1 -F "$log" &
    follower=$!
    trap 'kill "$follower" 2>/dev/null || true' EXIT
    while [[ -n $(squeue -h -j "$job" 2>/dev/null) ]]; do sleep 30; done
    sleep 5
    sacct -j "$job" --format=JobID,State,ExitCode,Elapsed -X
    exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
image=naruarjun/semvecnet:v2
output=$(realpath -m "${SEMVECNET_FT_OUT:-$HOME/semvecnet-finetune/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)-$$}}")
echo "Dataset $dataset; output $output"
nvidia-smi
if ! "$local_run"; then
    slurm-start-dockerd.sh
    if ! docker info >/dev/null 2>&1; then
        for s in "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/docker.sock" /tmp/run/docker.sock "/run/user/$(id -u)/docker.sock"; do
            if [[ -S $s ]] && DOCKER_HOST="unix://$s" docker info >/dev/null 2>&1; then
                export DOCKER_HOST="unix://$s"
                break
            fi
        done
    fi
    docker pull "$image"
else
    docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
fi
security=$(docker info --format '{{.SecurityOptions}}')
user_args=()
[[ $security == *rootless* ]] || user_args=(--user "$(id -u):$(id -g)")
seen=$(docker run --rm --gpus all --entrypoint nvidia-smi "$image" -L)
echo "$seen"
gpus=${SEMVECNET_DOCKER_GPUS:-all}
if [[ -z ${SEMVECNET_DOCKER_GPUS:-} && -n ${CUDA_VISIBLE_DEVICES:-} && $(grep -c '^GPU' <<< "$seen") -gt 1 ]]; then
    gpus="\"device=$CUDA_VISIBLE_DEVICES\""
fi
docker run --rm --gpus "$gpus" --entrypoint nvidia-smi "$image"
ckpt=$(realpath -m "${SEMVECNET_CKPT:-$repo/SemVecNet/ckpts/mAP_epoch_99.pth}")
if [[ ! -s $ckpt ]]; then
    if [[ -n ${SEMVECNET_CKPT:-} ]]; then echo "Missing checkpoint $ckpt" >&2; exit 1; fi
    [[ ! -f "$repo/rclone.conf" || -n ${RCLONE_CONFIG:-} ]] || export RCLONE_CONFIG="$repo/rclone.conf"
    mkdir -p "$(dirname "$ckpt")"
    tmp=$(mktemp "$ckpt.XXXXXX")
    trap 'rm -f "$tmp"' EXIT
    rclone copyto --drive-root-folder-id 1bV7mnjSOYg2M-LBbM59C_eFDIvqvZZ7i \
        'gdrive:SemVecNet checkpoints/nuscenes-semantic/mAP_epoch_99.pth' "$tmp"
    [[ -s $tmp ]] || { echo "Downloaded checkpoint is empty" >&2; exit 1; }
    mv "$tmp" "$ckpt"
fi
mkdir -p "$output"
docker run --rm --gpus "$gpus" --ipc=host "${user_args[@]}" -e HOME=/tmp \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -w /workspace/SemVecNet \
    -v "$repo/SemVecNet/projects:/workspace/SemVecNet/projects:ro" -v "$repo/tools:/repo/tools:ro" \
    -v "$ckpt:/weights.pth:ro" -v "$dataset:/dataset:ro" -v "$output:/out" --entrypoint python "$image" \
    -u /repo/tools/finetune_ringroad.py --data-root /dataset --output /out --checkpoint /weights.pth \
    --epochs "${EPOCHS:-40}" --batch "${BATCH:-4}" --lr "${LR:-0.0001}" --workers "${WORKERS:-4}" \
    --augment "${AUGMENT:-1}"
echo "Selected weights: $output/ringroad_finetuned.pth"
