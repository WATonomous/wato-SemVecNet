#include "semvecnet/hello.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t err_ = (call);                                                      \
    if (err_ != cudaSuccess) {                                                      \
      throw std::runtime_error(std::string(#call " failed: ") + cudaGetErrorString(err_)); \
    }                                                                               \
  } while (0)

namespace semvecnet {
namespace {

__global__ void iota_plus_one(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) printf("Hello from the GPU (block %d, thread %d)\n", blockIdx.x, threadIdx.x);
  if (i < n) out[i] = i + 1;
}

}  // namespace

std::string gpu_info() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  return std::string(prop.name) + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) + ")";
}

long long gpu_hello(int n) {
  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));

  const int threads = 256;
  iota_plus_one<<<(n + threads - 1) / threads, threads>>>(d_out, n);
  CUDA_CHECK(cudaGetLastError());

  std::vector<int> h_out(n);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));

  return std::accumulate(h_out.begin(), h_out.end(), 0LL);
}

}  // namespace semvecnet
