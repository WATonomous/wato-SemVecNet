#include "semvecnet/semvecnet-inference.hpp"

#include <onnxruntime_cxx_api.h>
#include <nvtx3/nvtx3.hpp>
#include <stdexcept>

namespace semvecnet::model {

Model::Model(const std::string &onnx_path) {
  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = 0;
  options_.AppendExecutionProvider_CUDA(cuda_options);
  session_ = Ort::Session(env_, onnx_path.c_str(), options_);
}

Model::~Model() = default;

DecoderOutput Model::forward(const SemanticMapInput &semantic_map) {
  const nvtx3::scoped_range range{"SemVecNet::semvecnet_model"};
  constexpr std::size_t input_count = 1 * 3 * 300 * 150;
  constexpr std::size_t logits_count = 1 * 70 * 4;
  constexpr std::size_t points_count = 1 * 70 * 20 * 2;

  if (semantic_map.values.size() != input_count)
    throw std::invalid_argument("Expected sem_map with shape [1, 3, 300, 150]");

  const std::array<int64_t, 4> input_shape = {1, 3, 300, 150};

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Tensor for the input data
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(semantic_map.values.data()),
                                                            semantic_map.values.size(), input_shape.data(), input_shape.size());

  const char *input_names[] = {"sem_map"};
  const char *output_names[] = {"class_logits", "normalized_points"};

  std::vector<Ort::Value> outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);

  // Validate the outputs
  const auto logits_info = outputs[0].GetTensorTypeAndShapeInfo();
  const auto points_info = outputs[1].GetTensorTypeAndShapeInfo();

  if (logits_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      points_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || logits_info.GetElementCount() != logits_count ||
      points_info.GetElementCount() != points_count) {
    throw std::runtime_error("Unexpected SemVecNet output type or size");
  }

  const float *logits = outputs[0].GetTensorData<float>();
  const float *points = outputs[1].GetTensorData<float>();

  return {std::vector<float>(logits, logits + logits_count), std::vector<float>(points, points + points_count)};
}

} // namespace semvecnet::model
