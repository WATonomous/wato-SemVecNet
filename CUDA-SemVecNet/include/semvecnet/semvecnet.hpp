#pragma once

#include <common/dtype.hpp>

#include <array>
#include <string>
#include <vector>
#include <NvInfer.h>

#include "common/common.hpp"
#include "semvecnet/base-config.hpp"
#include "semvecnet/lidar-semantic-projection.hpp"
#include "semvecnet/semantic-bev-mapping.hpp"
#include "semvecnet/semvecnet-inference.hpp"

namespace semvecnet {

struct PreparedFrame {
  // One contiguous FP32 [1, 3, 720, 960] RGB tensor per camera, in input order.
  std::vector<std::vector<float>> hrnet_images;

  // x, y, z, intensity, time_lag. For projection after HRNet
  std::vector<std::array<float, 5>> lidar_points;
};

struct ForwardResult {
  std::vector<semantic_projection::LabeledPoint> labeled_points;
  model::DecoderOutput predictions;
};

// One instance owns the inference sessions and persistent mapper across frames.
class SemVecNet {
public:
  explicit SemVecNet(const std::string &hrnet_plan = "model/hrnet_ocrnet_mapillary.plan",
                     const std::string &semvecnet_onnx = "model/semvecnet.onnx");

  // Same host buffers as BEVFusion: tightly packed RGB8 camera images and
  // five FP16 values per LiDAR point. Calibrations follow camera order and
  // use mask-sized intrinsics; lidar_to_map is a row-major 4x4 transform.
  ForwardResult
  forward(const unsigned char *const *camera_images,
          const nvtype::half *lidar_points, int num_points, int num_cameras,
          int image_width, int image_height,
          const std::vector<semantic_projection::CameraCalibration> &calibrations,
          const std::array<float, 16> &lidar_to_map);

private:
  TensorRtEngine hrnet_engine_;
  bev_mapping::Mapper mapper_;
  model::Model model_;
};

} // namespace semvecnet
