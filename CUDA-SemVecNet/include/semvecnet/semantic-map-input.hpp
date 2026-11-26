#pragma once

#include <vector>

namespace semvecnet::model {

struct SemanticMapInput {
  // Normalized FP32 RGB, NCHW [1, 3, 300, 150].
  std::vector<float> values;
};

} // namespace semvecnet::model
