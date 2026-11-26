#include "semvecnet/lidar-semantic-projection.hpp"

#include <nvtx3/nvtx3.hpp>
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
                              ProjectionResult *results_d, std::size_t point_count, std::size_t camera_count, float max_depth,
                              int image_width, int image_height) {

  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  // There is no [camera, point] pair associated with this index
  if (index >= camera_count * point_count)
    return;

  results_d[index] = {0, false};
  const std::size_t camera_index = index / point_count;
  const std::size_t point_index = index % point_count;

  const float lidar_x = points_d[point_index * 5 + 0];
  const float lidar_y = points_d[point_index * 5 + 1];
  const float lidar_z = points_d[point_index * 5 + 2];

  // Matrix for converting lidar points to points on the camera
  const float *T = calibrations_d[camera_index].lidar_to_camera;

  // These describe the lidar point's position relative to where the camera is mounted in 3d space
  const float camera_x = T[0] * lidar_x + T[1] * lidar_y + T[2] * lidar_z + T[3];
  const float camera_y = T[4] * lidar_x + T[5] * lidar_y + T[6] * lidar_z + T[7];
  const float camera_z = T[8] * lidar_x + T[9] * lidar_y + T[10] * lidar_z + T[11];

  if (!isfinite(camera_x) || !isfinite(camera_y) || !isfinite(camera_z) || camera_z <= 0.0f || camera_z >= max_depth)
    return;

  // Camera intrinsics matrix
  const float *K = calibrations_d[camera_index].intrinsics;

  // Turn the coordinates of where the point is in 3d to where it is on the flat 2d camera plane
  const float image_x = K[0] * camera_x + K[1] * camera_y + K[2] * camera_z;
  const float image_y = K[3] * camera_x + K[4] * camera_y + K[5] * camera_z;
  const float image_z = K[6] * camera_x + K[7] * camera_y + K[8] * camera_z;

  if (!isfinite(image_z) || image_z == 0.0f)
    return;

  // Intrinsics describe the input image, as in Python's projection.
  const float pixel_x = image_x / image_z;
  const float pixel_y = image_y / image_z;

  if (!isfinite(pixel_x) || !isfinite(pixel_y) || pixel_x <= -1.0f || pixel_x >= image_width || pixel_y <= -1.0f || pixel_y >= image_height)
    return;

  const int image_u = static_cast<int>(pixel_x);
  const int image_v = static_cast<int>(pixel_y);

  // Multiply it by the width and height of the Hrnet mask which is 960x720 vs the original 1920x1440 size
  // of the input camera image, and this gives us the corodinates on the Hrnet mask
  const int mask_u = static_cast<int>(static_cast<std::int64_t>(image_u) * kHrnetWidth / image_width);
  const int mask_v = static_cast<int>(static_cast<std::int64_t>(image_v) * kHrnetHeight / image_height);

  const std::size_t mask_size = kHrnetHeight * kHrnetWidth;
  const std::size_t mask_index = camera_index * mask_size + mask_v * kHrnetWidth + mask_u;

  // Update results with the mask that is in the camera array
  results_d[index].class_id = masks_d[mask_index];
  results_d[index].valid = true;
}

std::vector<LabeledPoint> Projector::forward(const std::vector<std::array<float, 5>> &lidar_points,
                                             const std::int64_t *masks_d, std::size_t camera_count,
                                             const std::vector<CameraCalibration> &calibrations, int image_width, int image_height) const {
  const nvtx3::scoped_range range{"SemVecNet::semantic_projection"};
  if (image_width <= 0 || image_height <= 0)
    throw std::invalid_argument("Expected positive input image dimensions");
  if (lidar_points.empty() || !masks_d || camera_count == 0)
    return {};

  const std::size_t point_count = lidar_points.size();
  const std::size_t point_values = point_count * 5;
  const std::size_t result_count = camera_count * point_count;

  // Flatten host data explicitly: points [point][feature].
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
  DeviceCalibration *calibrations_d = nullptr;
  ProjectionResult *results_d = nullptr;
  cudaMalloc(&points_d, point_values * sizeof(float));
  cudaMalloc(&calibrations_d, camera_count * sizeof(DeviceCalibration));
  cudaMalloc(&results_d, result_count * sizeof(ProjectionResult));

  cudaMemcpy(points_d, points.data(), point_values * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(calibrations_d, camera_data.data(), camera_count * sizeof(DeviceCalibration), cudaMemcpyHostToDevice);

  constexpr unsigned int threads_per_block = 256;
  const std::size_t blocks = (result_count + threads_per_block - 1) / threads_per_block;
  // TODO: kProjectionMaxDepth is 30 m here but Python's nuScenes
  // config uses MAPPING.PCD.RANGE_MAX = 100 m (see base-config.hpp).
  projectPoints<<<blocks, threads_per_block>>>(points_d, masks_d, calibrations_d, results_d, point_count, camera_count, kProjectionMaxDepth,
                                               image_width, image_height);
  cudaDeviceSynchronize();

  std::vector<ProjectionResult> results(result_count);
  cudaMemcpy(results.data(), results_d, result_count * sizeof(ProjectionResult), cudaMemcpyDeviceToHost);

  std::vector<LabeledPoint> labeled_points(result_count);
  std::size_t valid_count = 0;

  for (std::size_t i = 0; i < result_count; ++i) {
    if (!results[i].valid)
      continue;

    std::size_t camera_index = i / point_count;
    std::size_t point_index = i % point_count;
    labeled_points[valid_count] = {point_index, camera_index, results[i].class_id};
    ++valid_count;
  }
  labeled_points.resize(valid_count);

  cudaFree(results_d);
  cudaFree(calibrations_d);
  cudaFree(points_d);

  return labeled_points;
}

std::vector<LabeledPoint> forward(const std::vector<std::array<float, 5>> &lidar_points,
                                  const std::int64_t *masks_d, std::size_t camera_count,
                                  const std::vector<CameraCalibration> &calibrations, int image_width, int image_height) {
  return Projector{}.forward(lidar_points, masks_d, camera_count, calibrations, image_width, image_height);
}

} // namespace semvecnet::semantic_projection
