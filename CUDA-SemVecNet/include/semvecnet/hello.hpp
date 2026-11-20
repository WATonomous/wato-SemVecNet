#pragma once

#include <string>

namespace semvecnet {

// TensorRT runtime version, e.g. "11.4.0" (hello.cpp, links libnvinfer).
std::string tensorrt_version();

// Name + compute capability of GPU 0, e.g. "NVIDIA GeForce RTX 3070 Ti (sm_86)" (hello.cu).
std::string gpu_info();

// Launches a kernel that adds 1 to n ints on the GPU and returns their sum
// (expected: n * (n + 1) / 2). Throws std::runtime_error on CUDA errors.
long long gpu_hello(int n);

}  // namespace semvecnet
