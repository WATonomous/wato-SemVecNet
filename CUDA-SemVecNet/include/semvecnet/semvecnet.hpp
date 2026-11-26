#pragma once

#include <common/dtype.hpp>

#include <array>
#include <string>
#include <vector>
#include <NvInfer.h>

#include "common/common.hpp"
#include "semvecnet/base-config.hpp"
#include "semvecnet/camera-preprocessing.hpp"
#include "semvecnet/hrnet-ocrnet-inference.hpp"
#include "semvecnet/lidar-semantic-projection.hpp"
#include "semvecnet/polyline-postprocessing.hpp"
#include "semvecnet/semantic-bev-mapping.hpp"
#include "semvecnet/semantic-map-renderer.hpp"
#include "semvecnet/semvecnet-inference.hpp"

namespace semvecnet {

struct PreparedFrame {
  // x, y, z, intensity, time_lag. For projection after HRNet
  std::vector<std::array<float, 5>> lidar_points;
};

using Polylines = std::vector<postprocessing::Polyline>;

// One instance owns the inference sessions and persistent mapper across frames.
class SemVecNet {
public:
  explicit SemVecNet(const std::string &hrnet_plan = "model/hrnet_ocrnet_mapillary_b8_s1.plan",
                     const std::string &semvecnet_onnx = "model/semvecnet.onnx");

  // Same host buffers as BEVFusion: tightly packed RGB8 camera images and
  // five FP16 values per LiDAR point. Calibrations follow camera order and
  // use input-image intrinsics; poses are row-major 4x4 transforms into map coordinates.
  // semantic_rgb, if requested, receives the rendered model input before normalization.
  Polylines
  forward(const unsigned char *const *camera_images,
          const nvtype::half *lidar_points, int num_points, int num_cameras,
          int image_width, int image_height,
          const std::vector<semantic_projection::CameraCalibration> &calibrations,
          const std::array<float, 16> &lidar_to_map,
           const std::array<float, 16> &ego_to_map,
           std::vector<std::uint8_t> *semantic_rgb = nullptr);

private:
  TensorRtEngine hrnet_engine_;
  hrnet::Inference hrnet_inference_;
  camera::Preprocessor camera_preprocessor_;
  bev_mapping::Mapper mapper_;
  semantic_map::Renderer renderer_;
  model::Model model_;
  postprocessing::Postprocessor postprocessor_;
};

} // namespace semvecnet
