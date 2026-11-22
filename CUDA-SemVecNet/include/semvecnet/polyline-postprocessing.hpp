#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace semvecnet::model {
struct DecoderOutput;
}

namespace semvecnet::postprocessing {

struct Candidate {
  float score;
  std::size_t query_id;
  std::size_t class_id;
};

struct Polyline {
  int label; // 0: divider, 1: pedestrian crossing, 2: boundary, 3: centerline
  float score;
  std::array<std::array<float, 2>, 20> points; // Ordered (x, y), in meters.
};

class Postprocessor {
public:
  // NuScenes settings used by tool/export_semvecnet_onnx.py.
  std::vector<Polyline> forward(const semvecnet::model::DecoderOutput &predictions) const;

private:
  // Return whether the polyline's metric bounding box is within post_center_range.
  bool isWithinPostCenterRange(const Polyline &polyline) const;
};

} // namespace semvecnet::postprocessing
