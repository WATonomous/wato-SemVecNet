#include "semvecnet/hrnet-ocrnet-inference.hpp"
#include "semvecnet/semvecnet.hpp"

#include <cuda_runtime_api.h>
#include <utility>

namespace semvecnet::hrnet {
namespace {

class CudaStream {
public:
  CudaStream() { cudaStreamCreate(&stream_); }
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
      inputName = name;
    } else if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
      outputName = name;
    }
  }

  constexpr std::size_t pixels = kHrnetHeight * kHrnetWidth;
  const std::size_t inputBytes = 3 * pixels * sizeof(float);
  const std::size_t outputBytes = pixels * sizeof(std::int64_t);
  CudaStream stream;
  void *input_d = nullptr;
  void *output_d = nullptr;
  cudaMalloc(&input_d, inputBytes);
  cudaMalloc(&output_d, outputBytes);
  context.setTensorAddress(inputName, input_d);
  context.setTensorAddress(outputName, output_d);
  std::vector<std::vector<std::int64_t>> masks;
  masks.reserve(images.size());

  for (const auto &image : images) {
    // Copy the image data to the GPU
    cudaMemcpy(input_d, image.data(), inputBytes, cudaMemcpyHostToDevice);

    std::vector<std::int64_t> mask(pixels);
    context.enqueueV3(stream.get());

    // Copy output data back to the mask
    cudaMemcpyAsync(mask.data(), output_d, outputBytes, cudaMemcpyDeviceToHost, stream.get());
    cudaStreamSynchronize(stream.get());

    masks.push_back(std::move(mask));
  }

  context.setTensorAddress(inputName, nullptr);
  context.setTensorAddress(outputName, nullptr);
  cudaFree(input_d);
  cudaFree(output_d);
  return masks;
}

} // namespace semvecnet::hrnet
