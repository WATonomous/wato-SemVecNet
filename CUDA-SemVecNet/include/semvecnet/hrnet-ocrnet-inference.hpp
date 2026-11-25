#pragma once

#include <NvInfer.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace semvecnet::hrnet {

// Largest batch the engine's optimization profile accepts (tool/export_hrnet_batched.sh).
inline constexpr int kMaxBatch = 8;

// Runs every camera through HRNet in one batched TensorRT call. The engine must take a dynamic
// batch [N,3,720,960] with N up to kMaxBatch, and must outlive this object. Calls to forward must not overlap.
class Inference {
public:
  explicit Inference(nvinfer1::ICudaEngine *engine);
  ~Inference();
  Inference(const Inference &) = delete;
  Inference &operator=(const Inference &) = delete;

  // Returns one row-major [720, 960] class-ID mask per camera, in input order. Larger inputs run in chunks of kMaxBatch.
  std::vector<std::vector<std::int64_t>> forward(const std::vector<std::vector<float>> &images);

private:
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  const char *inputName_ = nullptr;
  cudaStream_t stream_ = nullptr;
  float *input_h_ = nullptr;          // pinned, kMaxBatch images
  std::int64_t *output_h_ = nullptr; // pinned, kMaxBatch masks
  void *input_d_ = nullptr;
  void *output_d_ = nullptr;
};

} // namespace semvecnet::hrnet
