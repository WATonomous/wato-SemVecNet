#!/bin/bash
# Export HRNet-OCR (Mapillary, mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth) to ONNX with a dynamic
# batch dimension, then build a TensorRT engine that accepts 1-8 images per call (tuned for 8).
#
#   tool/export_hrnet_batched.sh [assets dir] [output dir]
#
# assets dir: holds seg_weights/, data/Mapillary/data/config.json and TensorRT-tutorial/ as set up by
#             run_mcap_pipeline.sh (default ~/Desktop/data/semvecnet_assets)
# output dir: default CUDA-SemVecNet/model (gitignored)
#
# Outputs: hrnet_ocrnet_mapillary_b8.onnx  input  "images" FP32 [N,3,720,960]
#                                          output "mask"   INT64 [N,720,960] class IDs
#          hrnet_ocrnet_mapillary_b8.plan  TensorRT engine for this GPU + the semvecnet:dev-ort TensorRT version
set -euo pipefail
cd "$(dirname "$0")/.."

ASSETS="$(realpath "${1:-$HOME/Desktop/data/semvecnet_assets}")"
OUT="$(realpath -m "${2:-model}")"
NAME=hrnet_ocrnet_mapillary_b8
EXPORT_IMAGE="${HRNET_EXPORT_IMAGE:-naruarjun/tensorrt-nvidia-semseg}"
TRT_IMAGE="${HRNET_TRT_IMAGE:-semvecnet:dev-ort}"
mkdir -p "$OUT"

if [ ! -f "$OUT/$NAME.onnx" ]; then
  echo ">> Exporting $NAME.onnx"
  # convert-onnx.py loads the checkpoint, then calls torch.onnx.export with a fixed [1,3,720,960] input.
  # Replace that call: trace with batch 2 (so the batch size is not folded into constants) and mark it dynamic.
  docker run --rm -i --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$ASSETS:/home/dcg-adlr-atao-data.cosmos277/assets" \
    -v "$ASSETS/TensorRT-tutorial:/trt" -v "$OUT:/out" -w /trt \
    "$EXPORT_IMAGE" python - \
      --dataset mapillary --cv 0 --bs_val 1 --eval folder --eval_folder ./imgs/test_imgs \
      --n_scales 0.5,1.0 --snapshot ASSETS_PATH/seg_weights/mapillary_ocrnet.HRNet_Mscale_fast-rattlesnake.pth \
      --arch ocrnet.HRNet_Mscale --result_dir /tmp/onnx-export <<PY > "$OUT/$NAME.export.log" 2>&1 \
    || { tail -30 "$OUT/$NAME.export.log"; exit 1; }
import runpy, sys, torch
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
    export(Fp16(model).eval(), example, "/out/$NAME.onnx", opset_version=17, do_constant_folding=True,
           input_names=["images"], output_names=["mask"],
           dynamic_axes={"images": {0: "batch"}, "mask": {0: "batch"}})

torch.onnx.export = batched_export
sys.argv = ["convert-onnx.py"] + sys.argv[1:]
runpy.run_path("convert-onnx.py", run_name="__main__")
PY
fi

echo ">> Building $NAME.plan (batch 1-8, optimized for 8)"
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$OUT:/out" "$TRT_IMAGE" \
  bash -c 'trtexec --onnx=/out/'"$NAME"'.onnx --saveEngine=/out/'"$NAME"'.plan \
    --minShapes=images:1x3x720x960 --optShapes=images:8x3x720x960 --maxShapes=images:8x3x720x960 \
    --memPoolSize=workspace:4096' > "$OUT/$NAME.trtexec.log" 2>&1 \
  || { tail -30 "$OUT/$NAME.trtexec.log"; exit 1; }
grep -E 'TensorRT version|Throughput|Latency: min' "$OUT/$NAME.trtexec.log" || true
echo "Wrote $OUT/$NAME.onnx and $OUT/$NAME.plan"
