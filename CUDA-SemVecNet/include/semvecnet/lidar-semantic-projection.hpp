#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace semvecnet::semantic_projection {

struct CameraCalibration {
  // Row-major 3x3 camera intrinsics, scaled to the 960x720 mask.
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
// Each point has x, y, z, intensity, time_lag; each mask is row-major 720x960.
// A point visible in multiple cameras can produce multiple associations.
std::vector<LabeledPoint>
forward(const std::vector<std::array<float, 5>> &lidar_points,
        const std::vector<std::vector<std::int64_t>> &masks,
        const std::vector<CameraCalibration> &calibrations);

} // namespace semvecnet::semantic_projection
