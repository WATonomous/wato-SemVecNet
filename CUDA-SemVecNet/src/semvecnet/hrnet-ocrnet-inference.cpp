#include "semvecnet/hrnet-ocrnet-inference.hpp"
#include "semvecnet/base-config.hpp"

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>
#include <stdexcept>

namespace semvecnet::hrnet {

namespace {
constexpr std::size_t pixels = kHrnetHeight * kHrnetWidth;
} // namespace

Inference::Inference(nvinfer1::ICudaEngine *engine) {
  if (!engine)
    throw std::invalid_argument("HRNet engine is null");

  // Figure out the names the model expects
  const char *output_name = nullptr;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char *name = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
      input_name_ = name;
    else if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
      output_name = name;
  }
  if (!input_name_ || !output_name)
    throw std::runtime_error("HRNet input or output tensor not found");

  const nvinfer1::Dims max_shape = engine->getProfileShape(input_name_, 0, nvinfer1::OptProfileSelector::kMAX);
  if (engine->getTensorShape(input_name_).d[0] != -1 || max_shape.nbDims != 4 || max_shape.d[0] < kMaxBatch)
    throw std::runtime_error("HRNet engine must take a dynamic batch of up to 8 images; "
                             "build it with tool/export/export_hrnet.sh");

  context_.reset(engine->createExecutionContext());
  if (!context_)
    throw std::runtime_error("Failed to create HRNet execution context");
  cudaMalloc(reinterpret_cast<void **>(&output_d_), kMaxBatch * pixels * sizeof(std::int64_t));
  cudaStreamCreate(&stream_);
  if (!context_->setTensorAddress(output_name, output_d_))
    throw std::runtime_error("Failed to bind HRNet output address");
}

Inference::~Inference() {
  if (stream_)
    cudaStreamSynchronize(stream_);
  context_.reset();
  cudaFree(output_d_);
  if (stream_)
    cudaStreamDestroy(stream_);
}

const std::int64_t *Inference::forward(const float *images_d, int count) {
  const nvtx3::scoped_range range{"SemVecNet::hrnet_inference"};
  // One batch only: a second one would overwrite the first batch's masks
  if (!images_d || count <= 0 || count > kMaxBatch)
    throw std::invalid_argument("Expected 1 to 8 HRNet input images");
  if (!context_->setInputShape(input_name_, nvinfer1::Dims4{count, kHrnetChannels, kHrnetHeight, kHrnetWidth}) ||
      !context_->setTensorAddress(input_name_, const_cast<float *>(images_d)))
    throw std::runtime_error("Failed to set HRNet input");

  if (!context_->enqueueV3(stream_))
    throw std::runtime_error("Failed to enqueue HRNet inference");
  cudaStreamSynchronize(stream_);
  return output_d_;
}

} // namespace semvecnet::hrnet
