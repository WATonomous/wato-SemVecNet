#!/bin/bash
# Self-contained: download the HRNet-OCR Mapillary weights, export ONNX (dynamic batch, 1-8 cameras) and build the
# TensorRT engine on this machine's RTX 4090. Copy this one file to the car; the repo is not needed.
#
#   bash build_hrnet_4090.sh [work dir]                  # default ./hrnet_build
#   bash build_hrnet_4090.sh [work dir] --download-only
#
# Needs: git, nvidia-smi, Docker with GPU access, internet, ~50 GB free disk (the export image alone is ~31 GB).
# Output: <work dir>/hrnet_ocrnet_mapillary_b8_s1.plan
# Reruns skip finished downloads and exports. The export step matches tool/export_hrnet_batched.sh.
set -euo pipefail

WORK="$(realpath -m "${1:-hrnet_build}")"
NAME=hrnet_ocrnet_mapillary_b8_s1
ASSETS="$WORK/assets"
OUT="$WORK"
EXPORT_IMAGE=naruarjun/tensorrt-nvidia-semseg@sha256:e624bf74764d9377a2888c53135af7b034b094eb65c87515afd81bdaa75c70a2
TRT_IMAGE=semvecnet-trtexec:11.4.0
TRT_VERSION=11.4.0.106-1+cuda13.4   # must match the TensorRT that loads the .plan
TUTORIAL_COMMIT=dc274b3281026fef3a1cb395df70d552df784dfd
USER_FLAGS=(--user "$(id -u):$(id -g)" -e HOME=/tmp)

die() { echo "ERROR: $*" >&2; exit 1; }
for tool in git docker sha256sum realpath; do command -v "$tool" >/dev/null || die "$tool is not installed"; done
docker info >/dev/null 2>&1 || die "cannot reach the Docker daemon"
if [ "${2:-}" != "--download-only" ]; then
  CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0)" || die "nvidia-smi failed"
  [ "$CAP" = "8.9" ] || die "GPU 0 has compute capability $CAP, not 8.9 (RTX 4090)"
fi
mkdir -p "$ASSETS/seg_weights" "$ASSETS/data/Mapillary/data" "$OUT"

# Files from the assets folder linked in docs/prepare_dataset.md:
# https://drive.google.com/drive/folders/1bV7mnjSOYg2M-LBbM59C_eFDIvqvZZ7i
fetch() {  # <drive file id> <path under assets> <sha256>
  local dest="$ASSETS/$2"
  if [ -f "$dest" ] && echo "$3  $dest" | sha256sum -c --status; then echo "   have $2"; return; fi
  echo "   downloading $2"
  docker run --rm "${USER_FLAGS[@]}" -v "$ASSETS:/assets" python:3.12-slim \
    sh -c 'pip install -q --user --no-warn-script-location gdown==6.5.2 && python -m gdown "$1" -O "/assets/$2.part"' \
    sh "$1" "$2"
  echo "$3  $dest.part" | sha256sum -c --status || die "checksum mismatch for $2"
  mv "$dest.part" "$dest"
}
echo ">> [1/4] Weights and config"
fetch 13XjSF-FYLRizNBVp05Ehq3LDQWRViUHm seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth \
  9f693372bd3234d956e047e93925d8c92af80ede11e51243b59aeae7d5b8c9da
fetch 1aaopV32f3h29WADNdx5H_WNW4fbI_P7- seg_weights/hrnetv2_w48_imagenet_pretrained.pth \
  0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8
fetch 1fc-cT209Q2ZGzi1cyXjS8xQlnnoj-c38 data/Mapillary/data/config.json \
  3ea5474fb2273bf8e919e8b2ca0ba797b8d9578832647b6afd3b2633234de170

if [ ! -d "$WORK/TensorRT-tutorial/.git" ]; then
  git clone https://github.com/naruarjun/TensorRT-tutorial.git "$WORK/TensorRT-tutorial"
fi
git -C "$WORK/TensorRT-tutorial" checkout -q "$TUTORIAL_COMMIT"
[ "${2:-}" != "--download-only" ] || { echo "Downloaded to $WORK"; exit 0; }

echo ">> [2/4] Export ONNX"
if [ -f "$OUT/$NAME.onnx" ]; then
  echo "   have $NAME.onnx"
else
  docker image inspect "$EXPORT_IMAGE" >/dev/null 2>&1 || docker pull "$EXPORT_IMAGE"
  # convert-onnx.py calls torch.onnx.export with a fixed [1,3,720,960] input. Replace that call:
  # trace with batch 2 (so the batch size is not folded into constants) and mark it dynamic.
  docker run --rm -i --gpus all --ipc=host "${USER_FLAGS[@]}" \
    -v "$ASSETS:/home/dcg-adlr-atao-data.cosmos277/assets" \
    -v "$WORK/TensorRT-tutorial:/trt" -v "$OUT:/out" -w /trt \
    "$EXPORT_IMAGE" python - \
      --dataset mapillary --cv 0 --bs_val 1 --eval folder --eval_folder ./imgs/test_imgs \
      --n_scales 1.0 --snapshot ASSETS_PATH/seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth \
      --arch ocrnet.HRNet_Mscale --result_dir /tmp/onnx-export <<PY > "$OUT/export.log" 2>&1 \
    || { tail -30 "$OUT/export.log"; die "ONNX export failed, see $OUT/export.log"; }
import os, runpy, sys, torch
torch.set_grad_enabled(False)
export = torch.onnx.export

class Fp16(torch.nn.Module):
    # TensorRT 11 builds strongly typed networks only, so FP16 must be in the graph. Keep the FP32 input.
    def __init__(self, net):
        super().__init__()
        self.net = net.half()

    def forward(self, images):
        return self.net(images.half())

def batched_export(model, example, _path, **_):
    example = torch.randn(2, *example.shape[1:], device=example.device)
    export(Fp16(model).eval(), example, "/out/$NAME.tmp.onnx", opset_version=17, do_constant_folding=True,
           input_names=["images"], output_names=["mask"],
           dynamic_axes={"images": {0: "batch"}, "mask": {0: "batch"}})
    os.rename("/out/$NAME.tmp.onnx", "/out/$NAME.onnx")

torch.onnx.export = batched_export
sys.argv = ["convert-onnx.py"] + sys.argv[1:]
runpy.run_path("convert-onnx.py", run_name="__main__")
PY
  [ -f "$OUT/$NAME.onnx" ] || die "export finished without writing $NAME.onnx, see $OUT/export.log"
fi

echo ">> [3/4] TensorRT $TRT_VERSION image"
docker image inspect "$TRT_IMAGE" >/dev/null 2>&1 || docker build -t "$TRT_IMAGE" - <<DOCKERFILE
FROM nvcr.io/nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04
RUN printf 'Package: libnvinfer* libnvonnxparsers*\nPin: version %s\nPin-Priority: 1001\n' '$TRT_VERSION' \
        > /etc/apt/preferences.d/tensorrt && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        'libnvinfer-bin=$TRT_VERSION' 'libnvinfer-dev=$TRT_VERSION' \
        'libnvinfer-plugin-dev=$TRT_VERSION' 'libnvonnxparsers-dev=$TRT_VERSION' && \
    rm -rf /var/lib/apt/lists/*
DOCKERFILE

echo ">> [4/4] Build engine (batch 1-8, optimized for 8), then load it back"
trt() { docker run --rm --gpus all "${USER_FLAGS[@]}" -v "$OUT:/out" "$TRT_IMAGE" trtexec "$@"; }
{ trt --onnx="/out/$NAME.onnx" --saveEngine="/out/$NAME.plan.tmp" \
      --minShapes=images:1x3x720x960 --optShapes=images:8x3x720x960 --maxShapes=images:8x3x720x960 \
      --memPoolSize=workspace:4096 &&
  trt --loadEngine="/out/$NAME.plan.tmp" --shapes=images:8x3x720x960 --iterations=10 --duration=0
} > "$OUT/trtexec.log" 2>&1 || { tail -30 "$OUT/trtexec.log"; die "engine build failed, see $OUT/trtexec.log"; }
mv "$OUT/$NAME.plan.tmp" "$OUT/$NAME.plan"
grep -E 'Throughput|Latency: min' "$OUT/trtexec.log" | tail -2 || true
echo "Wrote $OUT/$NAME.plan"
