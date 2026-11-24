#include "semvecnet/polyline-postprocessing.hpp"

#include "semvecnet/base-config.hpp"
#include "semvecnet/semvecnet-inference.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace semvecnet::postprocessing {

std::vector<Polyline> Postprocessor::forward(const model::DecoderOutput &predictions) const {
  // TODO: match the Python full-map path. tools/infer_bevfusion.py runs 5x3 tiles,
  // clips each prediction to its tile core, drops vectors without semantic
  // support, stitches same-class lines across seams and de-duplicates them.
  // This port only thresholds and range-filters a single ego crop.
  constexpr std::size_t num_queries = 70;
  constexpr std::size_t num_classes = 4;
  constexpr std::size_t num_points = 20;
  constexpr std::size_t coordinates_per_point = 2;

  if (predictions.class_logits.size() != num_queries * num_classes)
    throw std::invalid_argument("Expected class logits with shape [1, 70, 4]");

  if (predictions.normalized_points.size() != num_queries * num_points * coordinates_per_point)
    throw std::invalid_argument("Expected normalized points with shape [1, 70, 20, 2]");

  // Logits -> Probabilities
  std::vector<Candidate> candidates;
  candidates.reserve(predictions.class_logits.size());

  for (std::size_t i = 0; i < predictions.class_logits.size(); ++i) {
    const float logit = predictions.class_logits[i];

    candidates.push_back({
        1.0f / (1.0f + std::exp(-logit)),
        i / 4, // query
        i % 4  // class
    });
  }

  // Take the top 50 and discard the rest
  std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
    return a.score > b.score; // Highest score first
  });
  candidates.resize(50);

  std::vector<Polyline> polylines;
  polylines.reserve(candidates.size());

  for (const Candidate &candidate : candidates) {
    // Candidates are sorted highest first, so all later scores also fail.
    if (candidate.score <= kSemvecnetConfidenceThreshold)
      break;

    Polyline polyline{};
    polyline.label = static_cast<int>(candidate.class_id);
    polyline.score = candidate.score;

    for (std::size_t p = 0; p < num_points; ++p) {
      const std::size_t offset = (candidate.query_id * num_points + p) * coordinates_per_point;

      // This converts the normalized points that are from [0, 1] to real-world coordinates from [-15, 15] and [-30, 30]
      const auto &range = kSemvecnetPointCloudRange;
      polyline.points[p][0] = predictions.normalized_points[offset] * (range[2] - range[0]) + range[0];
      polyline.points[p][1] = predictions.normalized_points[offset + 1] * (range[3] - range[1]) + range[1];
    }

    if (!isWithinPostCenterRange(polyline))
      continue;

    polylines.push_back(polyline);
  }

  return polylines;
}

bool Postprocessor::isWithinPostCenterRange(const Polyline &polyline) const {
  float xmin = polyline.points[0][0];
  float ymin = polyline.points[0][1];
  float xmax = xmin;
  float ymax = ymin;

  for (const auto &point : polyline.points) {
    xmin = std::min(xmin, point[0]);
    ymin = std::min(ymin, point[1]);
    xmax = std::max(xmax, point[0]);
    ymax = std::max(ymax, point[1]);
  }

  const std::array<float, 4> box = {xmin, ymin, xmax, ymax};
  for (std::size_t i = 0; i < box.size(); ++i)
    if (!(box[i] >= kSemvecnetPostCenterLower[i] && box[i] <= kSemvecnetPostCenterUpper[i]))
      return false;

  return true;
}

} // namespace semvecnet::postprocessing
