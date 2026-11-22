#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace semvecnet::model {

struct SemanticMapInput {
  // Normalized FP32 RGB, NCHW [1, 3, 300, 150].
  std::vector<float> values;
};

struct DecoderOutput {
  std::vector<float> class_logits;      // [1, 70, 4]
  std::vector<float> normalized_points; // [1, 70, 20, 2]
};

class Model {
public:
  explicit Model(const std::string &onnx_path);
  ~Model();

  DecoderOutput forward(const SemanticMapInput &semantic_map);

private:
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "semvecnet"};
  Ort::SessionOptions options_;
  Ort::Session session_{nullptr};
};

} // namespace semvecnet::model
