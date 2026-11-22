#include "semvecnet/lidar-semantic-projection.hpp"
#include "semvecnet/semvecnet.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace semvecnet::semantic_projection {
namespace {

// Plain arrays are accessible in CUDA device code.
struct DeviceCalibration {
  float intrinsics[9];
  float lidar_to_camera[16];
};

struct ProjectionResult {
  std::int64_t class_id;
  bool valid;
};

} // namespace

__global__ void projectPoints(const float *points_d, const std::int64_t *masks_d, const DeviceCalibration *calibrations_d,
                              ProjectionResult *results_d, std::size_t point_count, std::size_t camera_count, float max_depth) {

  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  // There is no [camera, point] pair associated with this index
  if (index > camera_count * point_count)
    return;

  const std::size_t camera_index = index / point_count;
  const std::size_t point_index = index % point_count;
}

std::vector<LabeledPoint> forward(const std::vector<std::array<float, 5>> &lidar_points,
                                  const std::vector<std::vector<std::int64_t>> &masks, const std::vector<CameraCalibration> &calibrations) {
  if (lidar_points.empty() || masks.empty())
    return {};

  constexpr std::size_t mask_size = kHrnetHeight * kHrnetWidth;
  const std::size_t point_count = lidar_points.size();
  const std::size_t camera_count = masks.size();
  const std::size_t point_values = point_count * 5;
  const std::size_t mask_values = camera_count * mask_size;
  const std::size_t result_count = camera_count * point_count;

  // Flatten host data explicitly: points [point][feature], masks [camera][pixel].
  std::vector<float> points;
  points.reserve(point_values);
  for (const auto &point : lidar_points)
    points.insert(points.end(), point.begin(), point.end());

  std::vector<DeviceCalibration> camera_data(camera_count);
  for (std::size_t i = 0; i < camera_count; ++i) {
    std::copy(calibrations[i].intrinsics.begin(), calibrations[i].intrinsics.end(), camera_data[i].intrinsics);
    std::copy(calibrations[i].lidar_to_camera.begin(), calibrations[i].lidar_to_camera.end(), camera_data[i].lidar_to_camera);
  }

  float *points_d = nullptr;
  std::int64_t *masks_d = nullptr;
  DeviceCalibration *calibrations_d = nullptr;
  ProjectionResult *results_d = nullptr;
  cudaMalloc(&points_d, point_values * sizeof(float));
  cudaMalloc(&masks_d, mask_values * sizeof(std::int64_t));
  cudaMalloc(&calibrations_d, camera_count * sizeof(DeviceCalibration));
  cudaMalloc(&results_d, result_count * sizeof(ProjectionResult));

  cudaMemcpy(points_d, points.data(), point_values * sizeof(float), cudaMemcpyHostToDevice);
  for (std::size_t i = 0; i < camera_count; ++i)
    cudaMemcpy(masks_d + i * mask_size, masks[i].data(), mask_size * sizeof(std::int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(calibrations_d, camera_data.data(), camera_count * sizeof(DeviceCalibration), cudaMemcpyHostToDevice);

  cudaFree(results_d);
  cudaFree(calibrations_d);
  cudaFree(masks_d);
  cudaFree(points_d);
  throw std::logic_error("LiDAR semantic projection is not implemented");
}

} // namespace semvecnet::semantic_projection
