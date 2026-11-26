#pragma once

#include <NvInfer.h>
#include <cstdint>
#include <memory>

namespace semvecnet::hrnet {

// Largest batch the engine's optimization profile accepts (tool/export/export_hrnet.sh).
inline constexpr int kMaxBatch = 8;

// Runs every camera through HRNet in one batched TensorRT call. The engine must take a dynamic
// batch [N,3,720,960] with N up to kMaxBatch, and must outlive this object. Calls to forward must not overlap.
class Inference {
public:
  explicit Inference(nvinfer1::ICudaEngine *engine);
  ~Inference();
  Inference(const Inference &) = delete;
  Inference &operator=(const Inference &) = delete;

  // images_d is device memory [count, 3, 720, 960]. Returns device memory holding one row-major
  // [720, 960] class-ID mask per camera, in input order. It is overwritten by the next call.
  const std::int64_t *forward(const float *images_d, int count);

private:
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  const char *input_name_ = nullptr;
  cudaStream_t stream_ = nullptr;
  std::int64_t *output_d_ = nullptr; // kMaxBatch masks
};

} // namespace semvecnet::hrnet
