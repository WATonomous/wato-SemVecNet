# SemVecNet inference on CUDA-BEVFusion maps.
# The base image already has torch 1.10 / CUDA 11.4, mmcv-full 1.4.0 (sm_52..sm_86),
# mmdet3d (develop install with compiled ops) and the GeometricKernelAttention op.
# We only replace the plugin code with this repo's copy and add the inference script;
# mmdetection3d/ is NOT mounted from the host so the image's compiled ops stay valid.
FROM naruarjun/semvecnet:v2

WORKDIR /workspace/SemVecNet
RUN rm -rf projects
COPY SemVecNet/projects projects
COPY tools/infer_bevfusion.py /opt/infer_bevfusion.py

ENTRYPOINT ["python", "/opt/infer_bevfusion.py"]
