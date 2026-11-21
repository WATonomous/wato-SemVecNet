#!/usr/bin/env bash
# One command: ROS 2 MCAP (cameras + LiDAR) -> SemVecNet vector map -> accuracy against a Lanelet2 HD map.
#
#   ./run_mcap_pipeline.sh BAG [options]
#   ./run_mcap_pipeline.sh --prepare-stage1     set up HRNet without a bag
#
#   BAG   a .mcap file or a rosbag2 directory (split files <name>_0.mcap, _1, ... are read in order)
#
# Stages (each in Docker, GPU required):
#   1  tools/mcap_semantic_mapping.py  HRNet (TensorRT) segments every camera, LiDAR points take the labels and
#                                      are accumulated into SemVecNet's own 60x30 m semantic BEV map (2 Hz keyframes)
#   2  tools/mcap_infer_semvecnet.py   pretrained SemVecNet (nuScenes checkpoint) -> polylines
#   3  tools/mcap_eval_lanelet2.py     Lanelet2 ground truth per keyframe -> MapTR chamfer AP, precision/recall,
#                                      overlays (skipped with --gt none)
#
# Options
#   -o, --out DIR        run directory        (default: $SEMVECNET_RUNS/<bag name>, SEMVECNET_RUNS=~/Desktop/data/semvecnet_runs)
#   --gt FILE|none       Lanelet2 .osm         (default: ./ring_road.osm)
#   --stages 123         stages to run, e.g. --stages 23 re-runs inference + evaluation on an existing run
#   --start S            skip the first S seconds of the bag
#   --duration S         process S seconds
#   --max-frames N       stop after N keyframes
#   --hz H               keyframe rate        (default 2, the nuScenes keyframe rate the model was trained on)
#   --min-move M         Minimum XY distance between written keyframes (default 2 m; 0 disables)
#   --min-sweep-move M   Minimum XY distance from the last processed sweep for map-only sweeps (default 0.25 m; 0 disables)
#   --map-hz H           LiDAR sweeps/s labelled into the semantic map (default 10; 2 = original nuScenes recipe)
#   --camera-ground 0|1  also paint camera labels onto the road plane (dense, sees thin paint)   (default 1)
#   --camera-classes C   which classes come from the camera: all | e.g. lane,crosswalk   (default all)
#   --camera-range M     max distance of camera ground labels (default 25 m)
#   --cameras A,B        camera names, e.g. camera_pano_nn,camera_pano_ne (names not in the bag are skipped; default:
#                        every camera with a camera_info - image_rect_compressed, else raw image_rect/image_raw)
#   --lidar-topic T      (default: auto, /lidar/all/points_merged - all LiDARs fused - else /lidar_cc/velodyne_points)
#   --odom-topic T       (default: auto, /novatel/oem7/odom - UTM, which is what the map is compared in)
#   --pose-source S      auto|odom|tf|inspva  (default auto: GNSS odometry, else NovAtel INSPVA converted to UTM,
#                        else /tf world frame, else any odometry)
#   --inspva-topic T     (default /novatel/oem7/inspva; used when the bag has no /novatel/oem7/odom)
#   --ego-frame F        map centre           (default: rear_axle if in TF, like nuScenes' ego frame)
#   --score-thr T        score threshold for drawn vectors and precision/recall (default 0.4; AP uses all)
#   --class-thr C=T,...  per-class overrides of --score-thr (default divider=0.1: SemVecNet finds dividers
#                        in the right place but with low confidence on this car's maps)
#   --pose-offset DX DY  shift the ego pose (m, UTM) before scoring
#   --near-map           stage 1 only maps while the ego is within 60 m (--near-map-margin M) of the --gt map's roads,
#                        so driving off the mapped roads costs no HRNet time (needs a UTM pose: NovAtel odom / INSPVA)
#   --divider-paint-check  drop predicted dividers with no lane paint under them in the semantic map (removes
#                        lanes SemVecNet invents from the road's width; predictions_unfiltered.json keeps all)
#
# One-time setup is automatic: the HRNet TensorRT engine is built for this GPU, the stage-1 image is built,
# and missing weights are fetched with rclone (remote "gdrive", see docs/prepare_dataset.md for the links).
set -euo pipefail

REPO="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
ASSETS="${SEMVECNET_ASSETS:-$HOME/Desktop/data/semvecnet_assets}"
RUNS="${SEMVECNET_RUNS:-$HOME/Desktop/data/semvecnet_runs}"
CKPT="${SEMVECNET_CKPT:-$REPO/SemVecNet/ckpts/mAP_epoch_99.pth}"
ENGINE="$ASSETS/hrnet.engine"
HRNET_IMAGE="${SEMVECNET_HRNET_IMAGE:-semvecnet-mcap-hrnet:latest}"
SVN_IMAGE=naruarjun/semvecnet:v2
ASSETS_DRIVE_ID=1bV7mnjSOYg2M-LBbM59C_eFDIvqvZZ7i   # docs/prepare_dataset.md + README.md weights folder

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
# drop the NVIDIA container banners, keep everything else (including errors)
quiet() { grep --line-buffered -v -E '^(=+|NVIDIA Release|NVIDIA Deep Learning|PyTorch Version|Container image|Copyright|All rights|Various files|This container|By pulling|https://|NOTE: |   insufficient|   docker run|   Use the NVIDIA|      Multi-node|WARNING: The NVIDIA|No CUDA runtime|.*\[TRT\] \[E\] 3: (Cannot find binding|\[executionContext)|\)$|$)' || true; }
# MapTR's eval_map prints per-threshold tables and timings; the script prints its own summary
quiet_eval() { grep --line-buffered -v -E '^(results path|Formatting|Cls data formatting|cls:.* done in|\+|\| )' || true; }

# ------------------------------------------------------------------ arguments
[ $# -ge 1 ] || { usage; exit 1; }
case "$1" in -h|--help) usage; exit 0 ;; esac
PREPARE=0
if [[ "$1" == --prepare-stage1 ]]; then
    PREPARE=1 BAG="" STAGES=1
else
    BAG="$(realpath -m "$1")"; STAGES=123
fi
shift
OUT="" GT="$REPO/ring_road.osm" S1=() S2=() S3=() INFER=() CLASS_THR="divider=0.1"
while [ $# -gt 0 ]; do
    case "$1" in
        -o|--out)      OUT="$(realpath -m "$2")"; shift 2 ;;
        --gt)          GT="$2"; shift 2 ;;
        --stages)      STAGES="$2"; shift 2 ;;
        --start|--duration|--max-frames|--hz|--map-hz|--min-move|--min-sweep-move|--cameras|--lidar-topic|--odom-topic|--pose-source|--inspva-topic|--near-map-margin|--ego-frame|\
        --camera-ground|--camera-classes|--camera-range|--expect-cameras)
                       S1+=("$1" "$2"); shift 2 ;;
        --no-inputs)   S1+=("$1"); shift ;;
        --score-thr)   S2+=("$1" "$2"); S3+=("$1" "$2"); shift 2 ;;
        --pose-offset) S3+=("$1" "$2" "$3"); shift 3 ;;
        --class-thr)   CLASS_THR="$2"; shift 2 ;;
        --divider-paint-check) INFER+=("$1"); shift ;;
        --near-map)    NEAR_MAP=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             die "unknown option $1 (see --help)" ;;
    esac
done
if [[ "$PREPARE" == 0 ]]; then
    [ -e "$BAG" ] || die "bag not found: $BAG"
    NAME="$(basename "${BAG%.mcap}")"
    OUT="${OUT:-$RUNS/$NAME}"
    if [ -d "$BAG" ]; then BAG_DIR="$BAG"; BAG_ARG=/bag; else BAG_DIR="$(dirname "$BAG")"; BAG_ARG="/bag/$(basename "$BAG")"; fi
    if [ "$GT" != none ]; then GT="$(realpath "$GT")"; [ -f "$GT" ] || die "ground truth map not found: $GT (use --gt none to skip evaluation)"; fi
    mkdir -p "$OUT"
fi
mkdir -p "$ASSETS"
DOCKER=(docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONWARNINGS=ignore)

# ------------------------------------------------------------------ one-time setup
fetch() {  # fetch <path in the weights Drive folder> <local dir>
    command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q '^gdrive:$' \
        || die "missing $(basename "$1"); configure rclone remote 'gdrive' or download it from https://drive.google.com/drive/folders/$ASSETS_DRIVE_ID into $2"
    echo ">> Downloading $1"
    rclone copy --drive-root-folder-id "$ASSETS_DRIVE_ID" "gdrive:$1" "$2" 2>&1 | grep -v NOTICE || true
}

if [[ "$STAGES" == *1* ]] && [ ! -f "$ENGINE" ]; then
    echo ">> One-time: building the HRNet TensorRT engine for this GPU (~15 min)"
    W="$ASSETS/seg_weights"
    [ -f "$W/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth" ] || fetch seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth "$W"
    [ -f "$W/hrnetv2_w48_imagenet_pretrained.pth" ] || fetch seg_weights/hrnetv2_w48_imagenet_pretrained.pth "$W"
    [ -f "$ASSETS/data/Mapillary/data/config.json" ] || { mkdir -p "$ASSETS/data/Mapillary/data"; cp "$REPO/semantic_mapping/config/config_65.json" "$ASSETS/data/Mapillary/data/config.json"; }
    [ -d "$ASSETS/TensorRT-tutorial" ] || git clone --depth 1 https://github.com/naruarjun/TensorRT-tutorial.git "$ASSETS/TensorRT-tutorial"
    mkdir -p "$ASSETS/TensorRT-tutorial/build"
    # docs/prepare_dataset.md step (a); autograd is switched off so the ONNX export fits in 8 GB of GPU memory
    "${DOCKER[@]}" -v "$ASSETS:/home/dcg-adlr-atao-data.cosmos277/assets" -v "$ASSETS/TensorRT-tutorial:/trt" -w /trt \
        naruarjun/tensorrt-nvidia-semseg bash -c '
        set -e
        python -c "import sys, runpy, torch; torch.set_grad_enabled(False); sys.argv=[\"convert-onnx.py\"]+sys.argv[1:]; runpy.run_path(\"convert-onnx.py\", run_name=\"__main__\")" \
            --dataset mapillary --cv 0 --bs_val 1 --eval folder --eval_folder ./imgs/test_imgs --dump_assets --dump_all_images \
            --n_scales 0.5,1.0 --snapshot ASSETS_PATH/seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth \
            --arch ocrnet.HRNet_Mscale --result_dir logs/onnx-export/mcap > build/convert.log 2>&1 || { tail -20 build/convert.log; exit 1; }
        trtexec --onnx=hrnet.onnx --fp16 --workspace=4096 --buildOnly --saveEngine=build/hrnet.engine > build/trtexec.log 2>&1 \
            || { tail -20 build/trtexec.log; exit 1; }' 2>&1 | quiet
    mv "$ASSETS/TensorRT-tutorial/build/hrnet.engine" "$ENGINE"
fi

if [[ "$STAGES" == *1* ]] && ! docker image inspect "$HRNET_IMAGE" >/dev/null 2>&1; then
    echo ">> One-time: building $HRNET_IMAGE"
    docker build -q -t "$HRNET_IMAGE" -f "$REPO/docker/mcap_hrnet.Dockerfile" "$REPO/docker" >/dev/null
fi
[[ "$PREPARE" == 0 ]] || exit 0

# ------------------------------------------------------------------ stages
SVN_MOUNTS=(-v "$REPO/SemVecNet/projects:/workspace/SemVecNet/projects:ro"
            -v "$REPO/tools:/repo/tools:ro" -v "$OUT:/out" -w /workspace/SemVecNet)
if [[ "$STAGES" == *2* || "$STAGES" == *3* ]]; then
    docker image inspect "$SVN_IMAGE" >/dev/null 2>&1 || docker pull "$SVN_IMAGE"
fi
if [[ "$STAGES" == *2* ]]; then
    [ -f "$CKPT" ] || fetch "SemVecNet checkpoints/nuscenes-semantic/$(basename "$CKPT")" "$(dirname "$CKPT")"
    [ -f "$CKPT" ] || die "missing SemVecNet checkpoint $CKPT"
    SVN_MOUNTS+=(-v "$CKPT:/workspace/SemVecNet/ckpts/mAP_epoch_99.pth:ro")
fi

if [[ "$STAGES" == *1* ]]; then
    echo ">> [1/3] Semantic BEV maps from $BAG"
    NM=()
    if [ "${NEAR_MAP:-0}" = 1 ]; then
        [ "$GT" != none ] || die "--near-map needs the Lanelet2 map (--gt)"
        NM=(-v "$GT:/gt/map.osm:ro")
        S1+=(--near-map /gt/map.osm)
    fi
    "${DOCKER[@]}" -v "$REPO:/repo:ro" -v "$BAG_DIR:/bag:ro" -v "$ENGINE:/assets/hrnet.engine:ro" -v "$OUT:/out" "${NM[@]}" \
        "$HRNET_IMAGE" python /repo/tools/mcap_semantic_mapping.py --bag "$BAG_ARG" --out /out \
        --engine /assets/hrnet.engine "${S1[@]}" 2>&1 | quiet
fi

if [[ "$STAGES" == *2* ]]; then
    echo ">> [2/3] SemVecNet inference"
    "${DOCKER[@]}" "${SVN_MOUNTS[@]}" "$SVN_IMAGE" python /repo/tools/mcap_infer_semvecnet.py --run /out --class-thr "$CLASS_THR" "${S2[@]}" "${INFER[@]}" 2>&1 | quiet
fi

if [[ "$STAGES" == *3* ]] && [ "$GT" != none ]; then
    echo ">> [3/3] Accuracy against $(basename "$GT")"
    "${DOCKER[@]}" "${SVN_MOUNTS[@]}" -v "$GT:/gt/map.osm:ro" "$SVN_IMAGE" \
        python /repo/tools/mcap_eval_lanelet2.py --run /out --osm /gt/map.osm --class-thr "$CLASS_THR" "${S3[@]}" 2>&1 | quiet | quiet_eval
fi

if [[ "$STAGES" == *2* || "$STAGES" == *3* ]]; then
    echo ">> Side-by-side view: cameras + LiDAR | semantic map | SemVecNet | ground truth | overlay"
    "${DOCKER[@]}" "${SVN_MOUNTS[@]}" "$SVN_IMAGE" python /repo/tools/mcap_compare_viz.py --run /out --class-thr "$CLASS_THR" "${S2[@]}" 2>&1 | quiet
fi

echo ">> Done. Outputs in $OUT"
if [[ "$STAGES" == 1 ]]; then
    echo "   maps/*.png, cams/*.jpg, frames.jsonl, meta.json"
    exit 0
fi
echo "   compare.mp4, compare/      cameras + LiDAR | semantic map | SemVecNet | ground truth | overlay"
echo "   semvecnet.mp4, viz/        camera | semantic map | predicted polylines"
echo "   vectors/<frame>.json       polylines in the ego frame (x fwd, y left, m)"
[ "$GT" != none ] && echo "   eval/metrics.json          AP / precision / recall vs $(basename "$GT")" \
                  && echo "   eval/eval.mp4, eval/global_overlay.png   predictions over the ground truth"
exit 0
