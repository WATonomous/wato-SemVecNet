#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace semvecnet::semantic_projection {

struct CameraCalibration {
  // Row-major 3x3 camera intrinsics for the input image dimensions.
  std::array<float, 9> intrinsics;
  // Row-major 4x4 homogeneous transform from LiDAR to camera coordinates.
  std::array<float, 16> lidar_to_camera;
};

struct LabeledPoint {
  std::size_t point_index;  // Index into the input LiDAR points.
  std::size_t camera_index; // Index into masks and calibrations.
  std::int64_t class_id;   // HRNet class ID at the projected pixel.
};

// Associate visible LiDAR points with HRNet mask labels, in camera order.
// Each point has x, y, z, intensity, time_lag; masks_d is device memory
// holding camera_count row-major 720x960 masks back to back.
// A point visible in multiple cameras can produce multiple associations.
std::vector<LabeledPoint>
forward(const std::vector<std::array<float, 5>> &lidar_points,
        const std::int64_t *masks_d, std::size_t camera_count,
        const std::vector<CameraCalibration> &calibrations, int image_width, int image_height);

} // namespace semvecnet::semantic_projection
