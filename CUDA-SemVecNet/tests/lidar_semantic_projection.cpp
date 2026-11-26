#include "semvecnet/lidar-semantic-projection.hpp"
#include "semvecnet/base-config.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

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
  std::int64_t *mask_d = nullptr;
  cudaMalloc(&mask_d, mask.size() * sizeof(std::int64_t));
  cudaMemcpy(mask_d, mask.data(), mask.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice);
  const auto labels = Projector{}.forward({{0, 0, 1, 0, 0}}, mask_d, 1, {calibration}, width, height);
  const auto legacy_labels = forward({{0, 0, 1, 0, 0}}, mask_d, 1, {calibration}, width, height);
  cudaFree(mask_d);
  if (labels.size() != 1 || labels[0].class_id != 8)
    throw std::runtime_error("Expected resized-mask lookup at pixel 0, not direct mask pixel 1");
  if (legacy_labels.size() != labels.size() || legacy_labels[0].point_index != labels[0].point_index ||
      legacy_labels[0].camera_index != labels[0].camera_index || legacy_labels[0].class_id != labels[0].class_id)
    throw std::runtime_error("Functional and object projection interfaces must agree");
}
