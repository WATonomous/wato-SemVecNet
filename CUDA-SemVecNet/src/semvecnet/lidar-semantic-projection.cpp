#include "semvecnet/lidar-semantic-projection.hpp"

#include <stdexcept>

namespace semvecnet::semantic_projection {

std::vector<LabeledPoint>
forward(const std::vector<std::array<float, 5>> & /*lidar_points*/,
        const std::vector<std::vector<std::int64_t>> & /*masks*/,
        const std::vector<CameraCalibration> & /*calibrations*/) {
  // TODO: Transform points into each camera, project to mask pixels, and
  // return the class ID for each visible point-camera pair.
  throw std::logic_error("LiDAR semantic projection is not implemented");
}

} // namespace semvecnet::semantic_projection
