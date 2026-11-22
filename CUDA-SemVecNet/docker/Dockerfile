# Keep the CUDA/TensorRT versions used by the project's demo and dev image.
FROM nvcr.io/nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04 AS dev

ARG DEBIAN_FRONTEND=noninteractive
ARG TRT_VERSION=11.4.0.106-1+cuda13.4
ARG CUDASM=86

RUN printf 'Package: libnvinfer* libnvonnxparsers*\nPin: version %s\nPin-Priority: 1001\n' \
        "$TRT_VERSION" > /etc/apt/preferences.d/tensorrt && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git gdb pkg-config \
        ca-certificates curl less unzip libopencv-dev \
        "libnvinfer-dev=$TRT_VERSION" "libnvinfer-plugin-dev=$TRT_VERSION" \
        "libnvonnxparsers-dev=$TRT_VERSION" "libnvinfer-bin=$TRT_VERSION" && \
    rm -rf /var/lib/apt/lists/*

# This ONNX Runtime release provides the native x86-64 CUDA 13 / cuDNN 9 SDK.
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
RUN curl -fsSL --retry 3 -o /tmp/onnxruntime.nupkg \
        https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.gpu.linux/1.27.1/microsoft.ml.onnxruntime.gpu.linux.1.27.1.nupkg && \
    echo '71d5b6f5596c25e2b8a92fe488b98ea4b580dde2adfab079d5651f768a01a512  /tmp/onnxruntime.nupkg' | sha256sum -c - && \
    mkdir -p "$ONNXRUNTIME_ROOT/include" "$ONNXRUNTIME_ROOT/lib" && \
    unzip -jq /tmp/onnxruntime.nupkg 'buildTransitive/native/include/*.h' -d "$ONNXRUNTIME_ROOT/include" && \
    unzip -jq /tmp/onnxruntime.nupkg 'runtimes/linux-x64/native/*.so' -d "$ONNXRUNTIME_ROOT/lib" && \
    unzip -jq /tmp/onnxruntime.nupkg LICENSE ThirdPartyNotices.txt -d "$ONNXRUNTIME_ROOT" && \
    rm /tmp/onnxruntime.nupkg && \
    echo "$ONNXRUNTIME_ROOT/lib" > /etc/ld.so.conf.d/onnxruntime.conf && ldconfig

ENV CUDASM=$CUDASM \
    CUDA_HOME=/usr/local/cuda \
    CUDA_Inc=/usr/local/cuda/include \
    CUDA_Lib=/usr/local/cuda/lib64 \
    TensorRT_Inc=/usr/include/x86_64-linux-gnu \
    TensorRT_Lib=/usr/lib/x86_64-linux-gnu \
    TensorRT_Bin=/usr/src/tensorrt/bin
ENV PATH=$TensorRT_Bin:$PATH

WORKDIR /workspace
