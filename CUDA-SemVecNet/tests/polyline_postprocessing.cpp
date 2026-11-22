#include "semvecnet/polyline-postprocessing.hpp"
#include "semvecnet/semvecnet-inference.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

int main() {
  const semvecnet::postprocessing::Postprocessor postprocessor;
  semvecnet::model::DecoderOutput output;
  output.class_logits.resize(70 * 4);
  output.normalized_points.resize(70 * 20 * 2);

  auto expect_invalid = [&](const char *expected) {
    try {
      postprocessor.forward(output);
    } catch (const std::invalid_argument &error) {
      if (std::string(error.what()) == expected)
        return;
    }
    throw std::runtime_error("Expected validation error");
  };

  output.class_logits.pop_back();
  expect_invalid("Expected class logits with shape [1, 70, 4]");
  output.class_logits.resize(70 * 4 + 1);
  expect_invalid("Expected class logits with shape [1, 70, 4]");
  output.class_logits.resize(70 * 4);
  output.normalized_points.pop_back();
  expect_invalid("Expected normalized points with shape [1, 70, 20, 2]");
  output.normalized_points.resize(70 * 20 * 2 + 1);
  expect_invalid("Expected normalized points with shape [1, 70, 20, 2]");
  output.normalized_points.resize(70 * 20 * 2);

  output.class_logits.assign(70 * 4, -10.0f);
  output.class_logits[7 * 4 + 2] = 4.0f;
  output.class_logits[7 * 4] = 3.0f;
  output.normalized_points[(7 * 20) * 2] = 0.0f;
  output.normalized_points[(7 * 20) * 2 + 1] = 1.0f;
  output.normalized_points[(7 * 20 + 1) * 2] = 1.0f;
  output.normalized_points[(7 * 20 + 1) * 2 + 1] = 0.0f;
  const auto polylines = postprocessor.forward(output);
  if (polylines.size() != 2 || polylines[0].label != 2 || polylines[1].label != 0 ||
      std::abs(polylines[0].score - 1.0f / (1.0f + std::exp(-4.0f))) > 1e-6f ||
      polylines[0].points[0][0] != -15.0f || polylines[0].points[0][1] != 30.0f ||
      polylines[0].points[1][0] != 15.0f || polylines[0].points[1][1] != -30.0f ||
      polylines[0].points != polylines[1].points)
    throw std::runtime_error("Unexpected polylines");

  // A point exactly on the x lower bound passes; one beyond it is rejected.
  output.normalized_points[(7 * 20) * 2] = -1.0f / 6.0f;
  if (postprocessor.forward(output).size() != 2)
    throw std::runtime_error("Expected inclusive range boundary");
  output.normalized_points[(7 * 20) * 2] = -0.2f;
  output.class_logits[8 * 4 + 1] = 2.0f;
  const auto filtered = postprocessor.forward(output);
  if (filtered.size() != 1 || filtered[0].label != 1)
    throw std::runtime_error("Expected out-of-range candidates to be skipped");
}
