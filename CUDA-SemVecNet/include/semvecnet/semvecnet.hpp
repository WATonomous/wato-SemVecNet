#pragma once

#include <common/dtype.hpp>

#include <array>
#include <vector>
#include <NvInfer.h>

#include "common/common.hpp"
#include "semvecnet/base-config.hpp"
#include "semvecnet/lidar-semantic-projection.hpp"

namespace semvecnet {

struct PreparedFrame {
  // One contiguous FP32 [1, 3, 720, 960] RGB tensor per camera, in input order.
  std::vector<std::vector<float>> hrnet_images;

  // x, y, z, intensity, time_lag. For projection after HRNet
  std::vector<std::array<float, 5>> lidar_points;
};

// Same host buffer layout as BEVFusion: one tightly packed RGB8 image per
// pointer (all the same size), and five FP16 values per LiDAR point.
// Calibrations must match camera order and use intrinsics scaled to the HRNet
// mask dimensions. Returned point indices refer to the input LiDAR points.
std::vector<semantic_projection::LabeledPoint>
forward(const unsigned char *const *camera_images,
        const nvtype::half *lidar_points, int num_points, int num_cameras,
        int image_width, int image_height,
        const std::vector<semantic_projection::CameraCalibration> &calibrations);

} // namespace semvecnet
