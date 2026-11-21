# Stage 1 of run_mcap_pipeline.sh: MCAP -> HRNet (TensorRT) segmentation -> SemVecNet semantic BEV maps.
# The base image has TensorRT 8.6, pycuda, torch, OpenCV, scipy and yacs (it is the image docs/prepare_dataset.md
# uses to build the HRNet engine). Shapely lets the same image align and export
# the fine-tuning maps without pulling a second CUDA image. The repo is mounted at runtime.
FROM naruarjun/tensorrt-nvidia-semseg

RUN pip install --no-cache-dir "mcap>=1.1,<2" "mcap-ros2-support>=0.5,<0.6" yacs "shapely==1.8.5.post1"
ENV PYTHONDONTWRITEBYTECODE=1
