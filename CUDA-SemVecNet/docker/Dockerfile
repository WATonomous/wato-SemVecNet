# Build context: CUDA-SemVecNet/
#   docker build -f docker/Dockerfile --target dev   -t semvecnet:dev .
#   docker build -f docker/Dockerfile --target build -t semvecnet:build .
#
# CUDA 13.4 + TensorRT 11.4 on Ubuntu 24.04, matching what bevfusion compiles against in
# wato_monorepo/docker/perception.Dockerfile (TRT11_VERSION=11.4.0.106-1+cuda13.4).
# Keep TRT_VERSION in sync with that file: a .plan only loads on the TensorRT version that built it.
ARG CUDA_VERSION=13.4.3
ARG UBUNTU_VERSION=ubuntu24.04

############################ dev: toolchain only ############################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-${UBUNTU_VERSION} AS dev

ARG DEBIAN_FRONTEND=noninteractive
ARG TRT_VERSION=11.4.0.106-1+cuda13.4
# GPU arch: 86 = RTX 30xx, 89 = RTX 40xx. Override with --build-arg CUDASM=89
ARG CUDASM=86

# Pin every TensorRT package (including transitive ones like libnvinfer-lean10 and
# libnvinfer-headers-dev) to one version; otherwise apt mixes in other TRT builds.
# The -dev packages pull in the matching runtime libs (libnvinfer11, ...).
RUN printf 'Package: libnvinfer* libnvonnxparsers*\nPin: version %s\nPin-Priority: 1001\n' \
        "${TRT_VERSION}" > /etc/apt/preferences.d/tensorrt && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git gdb pkg-config \
        ca-certificates curl less \
        libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev libnvinfer-bin && \
    rm -rf /var/lib/apt/lists/*

# Same variable names wato-cuda-bevfusion / the monorepo vendor package use.
ENV CUDASM=${CUDASM} \
    CUDA_HOME=/usr/local/cuda \
    CUDA_Inc=/usr/local/cuda/include \
    CUDA_Lib=/usr/local/cuda/lib64 \
    TensorRT_Inc=/usr/include/x86_64-linux-gnu \
    TensorRT_Lib=/usr/lib/x86_64-linux-gnu \
    TensorRT_Bin=/usr/src/tensorrt/bin \
    PATH=/usr/src/tensorrt/bin:${PATH}

WORKDIR /workspaces/wato-SemVecNet/CUDA-SemVecNet

############################ build: compile + install ########################
FROM dev AS build
COPY . /src/CUDA-SemVecNet
RUN cmake -S /src/CUDA-SemVecNet -B /src/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/semvecnet && \
    cmake --build /src/build && \
    cmake --install /src/build
ENV LD_LIBRARY_PATH=/opt/semvecnet/lib:${LD_LIBRARY_PATH}
