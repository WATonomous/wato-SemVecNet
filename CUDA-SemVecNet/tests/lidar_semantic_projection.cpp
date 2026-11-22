#include "semvecnet/lidar-semantic-projection.hpp"
#include "semvecnet/base-config.hpp"

#include <stdexcept>

int main() {
  using namespace semvecnet;
  using namespace semvecnet::semantic_projection;
  constexpr int width = 1600;
  constexpr int height = 1200;
  // Input-image intrinsics project the point to pixel (1.9, 0).
  const CameraCalibration calibration = {
      {1, 0, 1.9f, 0, 1, 0, 0, 0, 1},
      {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
  std::vector<std::int64_t> mask(kHrnetWidth * kHrnetHeight, 0);
  mask[0] = 8;
  mask[1] = 13;
  const auto labels = forward({{0, 0, 1, 0, 0}}, {mask}, {calibration}, width, height);
  if (labels.size() != 1 || labels[0].class_id != 8)
    throw std::runtime_error("Expected resized-mask lookup at pixel 0, not direct mask pixel 1");
}
