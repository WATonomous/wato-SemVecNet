# Development container with CUDA 11.8 + TensorRT 8.5.3
# Matches wato-cuda-bevfusion's environment EXACTLY
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1

# Install TensorRT 8.5.3 and development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git wget ca-certificates \
        libcudnn8=8.6.0.163-1+cuda11.8 \
        libcudnn8-dev=8.6.0.163-1+cuda11.8 \
        libnvinfer8=8.5.3-1+cuda11.8 \
        libnvinfer-dev=8.5.3-1+cuda11.8 \
        libnvinfer-plugin8=8.5.3-1+cuda11.8 \
        libnvinfer-plugin-dev=8.5.3-1+cuda11.8 \
        libnvonnxparsers8=8.5.3-1+cuda11.8 \
        libnvonnxparsers-dev=8.5.3-1+cuda11.8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up environment variables for CMake
ENV CUDA_HOME=/usr/local/cuda
ENV TensorRT_Inc=/usr/include/x86_64-linux-gnu
ENV TensorRT_Lib=/usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

WORKDIR /workspace
