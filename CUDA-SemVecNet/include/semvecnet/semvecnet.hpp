#pragma once

#include <common/dtype.hpp>

#include <array>
#include <vector>

namespace semvecnet {

inline constexpr int kHrnetWidth = 960;
inline constexpr int kHrnetHeight = 720;

struct PreparedFrame {
  // One contiguous FP32 [1, 3, 720, 960] RGB tensor per camera, in input order.
  std::vector<std::vector<float>> hrnet_images;

  // x, y, z, intensity, time_lag. For projection after HRNet
  std::vector<std::array<float, 5>> lidar_points;
};

// Same host buffer layout as BEVFusion: one tightly packed RGB8 image per
// pointer (all the same size), and five FP16 values per LiDAR point.
// Writes the intermediate frame for HRNet and LiDAR projection. No CUDA stream
// or calibration is needed. The final pipeline result is not defined yet.
void forward(const unsigned char *const *camera_images,
             const nvtype::half *lidar_points, int num_points, int num_cameras,
             int image_width, int image_height);

} // namespace semvecnet
