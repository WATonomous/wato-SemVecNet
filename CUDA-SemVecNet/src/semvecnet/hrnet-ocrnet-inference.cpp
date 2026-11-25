#include "semvecnet/hrnet-ocrnet-inference.hpp"
#include "semvecnet/base-config.hpp"

#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace semvecnet::hrnet {

namespace {
constexpr std::size_t pixels = kHrnetHeight * kHrnetWidth;
constexpr std::size_t imageFloats = kHrnetChannels * pixels;
constexpr std::size_t inputBytes = imageFloats * sizeof(float);
constexpr std::size_t outputBytes = pixels * sizeof(std::int64_t);

void checkCuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
} // namespace

Inference::Inference(nvinfer1::ICudaEngine *engine) {
  if (!engine)
    throw std::invalid_argument("HRNet engine is null");

  // Figure out the names the model expects
  const char *outputName = nullptr;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char *name = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
      inputName_ = name;
    else if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
      outputName = name;
  }
  if (!inputName_ || !outputName)
    throw std::runtime_error("HRNet input or output tensor not found");

  const nvinfer1::Dims maxShape = engine->getProfileShape(inputName_, 0, nvinfer1::OptProfileSelector::kMAX);
  if (engine->getTensorShape(inputName_).d[0] != -1 || maxShape.nbDims != 4 || maxShape.d[0] < kMaxBatch)
    throw std::runtime_error("HRNet engine must take a dynamic batch of up to 8 images; "
                             "build it with tool/export_hrnet_batched.sh");

  try {
    context_.reset(engine->createExecutionContext());
    if (!context_)
      throw std::runtime_error("Failed to create HRNet execution context");
    checkCuda(cudaMallocHost(reinterpret_cast<void **>(&input_h_), kMaxBatch * inputBytes),
              "Allocating pinned HRNet input");
    checkCuda(cudaMallocHost(reinterpret_cast<void **>(&output_h_), kMaxBatch * outputBytes),
              "Allocating pinned HRNet output");
    checkCuda(cudaMalloc(&input_d_, kMaxBatch * inputBytes), "Allocating HRNet input");
    checkCuda(cudaMalloc(&output_d_, kMaxBatch * outputBytes), "Allocating HRNet output");
    checkCuda(cudaStreamCreate(&stream_), "Creating HRNet stream");
    if (!context_->setTensorAddress(inputName_, input_d_) || !context_->setTensorAddress(outputName, output_d_))
      throw std::runtime_error("Failed to bind HRNet tensor addresses");
  } catch (...) {
    release();
    throw;
  }
}

Inference::~Inference() { release(); }

void Inference::release() noexcept {
  if (stream_)
    cudaStreamSynchronize(stream_);
  context_.reset();
  cudaFree(input_d_);
  cudaFree(output_d_);
  cudaFreeHost(input_h_);
  cudaFreeHost(output_h_);
  if (stream_)
    cudaStreamDestroy(stream_);
  input_d_ = output_d_ = nullptr;
  input_h_ = nullptr;
  output_h_ = nullptr;
  stream_ = nullptr;
}

std::vector<std::vector<std::int64_t>> Inference::forward(const std::vector<std::vector<float>> &images) {
  for (const auto &image : images) {
    if (image.size() != imageFloats)
      throw std::invalid_argument("Unexpected HRNet input image size");
  }

  std::vector<std::vector<std::int64_t>> masks;
  masks.reserve(images.size());
  for (std::size_t base = 0; base < images.size(); base += kMaxBatch) {
    const int count = static_cast<int>(std::min<std::size_t>(kMaxBatch, images.size() - base));
    // Pack the cameras back to back: the engine input is [count, 3, 720, 960]
    for (int i = 0; i < count; ++i)
      std::memcpy(input_h_ + i * imageFloats, images[base + i].data(), inputBytes);

    if (!context_->setInputShape(inputName_, nvinfer1::Dims4{count, kHrnetChannels, kHrnetHeight, kHrnetWidth}))
      throw std::runtime_error("Failed to set HRNet batch size");
    try {
      checkCuda(cudaMemcpyAsync(input_d_, input_h_, count * inputBytes, cudaMemcpyHostToDevice, stream_),
                "Copying HRNet input");
      if (!context_->enqueueV3(stream_))
        throw std::runtime_error("Failed to enqueue HRNet inference");
      checkCuda(cudaMemcpyAsync(output_h_, output_d_, count * outputBytes, cudaMemcpyDeviceToHost, stream_),
                "Copying HRNet output");
      checkCuda(cudaStreamSynchronize(stream_), "Synchronizing HRNet stream");
    } catch (...) {
      // The pinned buffers must not be reused while submitted work is still running.
      cudaStreamSynchronize(stream_);
      throw;
    }
    for (int i = 0; i < count; ++i)
      masks.emplace_back(output_h_ + i * pixels, output_h_ + (i + 1) * pixels);
  }
  return masks;
}

} // namespace semvecnet::hrnet
