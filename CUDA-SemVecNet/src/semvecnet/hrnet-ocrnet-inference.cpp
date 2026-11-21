#include "semvecnet/hrnet-ocrnet-inference.hpp"
#include "semvecnet/semvecnet.hpp"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace semvecnet::hrnet {
namespace {

// Unbind before freeing, including when an exception leaves forward().
class DeviceTensor {
public:
  DeviceTensor(nvinfer1::IExecutionContext &context, const char *name, std::size_t bytes) : context_(context), name_(name) {
    const auto error = cudaMalloc(&data_, bytes);
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("Failed to allocate tensor: ") + cudaGetErrorString(error));
    if (!context_.setTensorAddress(name_, data_)) {
      cudaFree(data_);
      throw std::runtime_error(std::string("Failed to bind tensor: ") + name_);
    }
  }
  DeviceTensor(const DeviceTensor &) = delete;
  DeviceTensor &operator=(const DeviceTensor &) = delete;
  ~DeviceTensor() {
    context_.setTensorAddress(name_, nullptr);
    cudaFree(data_);
  }
  void *data() const { return data_; }

private:
  nvinfer1::IExecutionContext &context_;
  const char *name_;
  void *data_ = nullptr;
};

class CudaStream {
public:
  CudaStream() {
    const auto error = cudaStreamCreate(&stream_);
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("Failed to create CUDA stream: ") + cudaGetErrorString(error));
  }
  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;
  ~CudaStream() {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};

} // namespace

std::vector<std::vector<std::int64_t>> forward(nvinfer1::IExecutionContext &context, const std::vector<std::vector<float>> &images) {
  const nvinfer1::ICudaEngine &engine = context.getEngine();

  const char *inputName = nullptr;
  const char *outputName = nullptr;
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char *name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      if (inputName)
        throw std::runtime_error("Expected one HRNet input");
      inputName = name;
    } else if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
      if (outputName)
        throw std::runtime_error("Expected one HRNet output");
      outputName = name;
    }
  }

  if (!inputName || !outputName)
    throw std::runtime_error("Could not find HRNet input and output tensors");

  const auto inputShape = engine.getTensorShape(inputName);
  const auto outputShape = engine.getTensorShape(outputName);
  if (engine.getTensorDataType(inputName) != nvinfer1::DataType::kFLOAT ||
      engine.getTensorLocation(inputName) != nvinfer1::TensorLocation::kDEVICE ||
      engine.getTensorFormat(inputName) != nvinfer1::TensorFormat::kLINEAR || inputShape.nbDims != 4 || inputShape.d[0] != 1 ||
      inputShape.d[1] != 3 || inputShape.d[2] != kHrnetHeight || inputShape.d[3] != kHrnetWidth)
    throw std::runtime_error("Expected linear device FP32 input [1,3,720,960]");
  if (engine.getTensorDataType(outputName) != nvinfer1::DataType::kINT64 ||
      engine.getTensorLocation(outputName) != nvinfer1::TensorLocation::kDEVICE ||
      engine.getTensorFormat(outputName) != nvinfer1::TensorFormat::kLINEAR || outputShape.nbDims != 3 || outputShape.d[0] != 1 ||
      outputShape.d[1] != kHrnetHeight || outputShape.d[2] != kHrnetWidth)
    throw std::runtime_error("Expected linear device INT64 output [1,720,960]");

  constexpr std::size_t pixels = kHrnetHeight * kHrnetWidth;
  const std::size_t inputBytes = 3 * pixels * sizeof(float);
  const std::size_t outputBytes = pixels * sizeof(std::int64_t);
  DeviceTensor input(context, inputName, inputBytes);
  DeviceTensor output(context, outputName, outputBytes);

  CudaStream stream;

  std::vector<std::vector<std::int64_t>> masks;
  masks.reserve(images.size());

  for (const auto &image : images) {
    if (image.size() != 3 * pixels)
      throw std::runtime_error("Incorrect HRNet input image size");

    // Copy the image data to the GPU
    const cudaError_t error = cudaMemcpy(input.data(), image.data(), inputBytes, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
      throw std::runtime_error(std::string("Failed to copy HRNet input: ") + cudaGetErrorString(error));

    std::vector<std::int64_t> mask(pixels);
    try {
      if (!context.enqueueV3(stream.get()))
        throw std::runtime_error("Failed to enqueue HRNet inference");

      // Copy output data back to the mask
      auto copyError = cudaMemcpyAsync(mask.data(), output.data(), outputBytes, cudaMemcpyDeviceToHost, stream.get());
      if (copyError != cudaSuccess)
        throw std::runtime_error(std::string("Failed to copy HRNet output: ") + cudaGetErrorString(copyError));
      auto syncError = cudaStreamSynchronize(stream.get());
      if (syncError != cudaSuccess)
        throw std::runtime_error(std::string("Failed to synchronize HRNet inference: ") + cudaGetErrorString(syncError));
    } catch (...) {
      // The mask and device buffers must outlive any queued GPU work.
      cudaStreamSynchronize(stream.get());
      throw;
    }

    masks.push_back(std::move(mask));
  }

  return masks;
}

} // namespace semvecnet::hrnet
