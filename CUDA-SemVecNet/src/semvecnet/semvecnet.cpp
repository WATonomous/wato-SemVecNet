#include "semvecnet/semvecnet.hpp"

#include "semvecnet/camera-preprocessing.hpp"
#include "semvecnet/hrnet-ocrnet-inference.hpp"
#include "semvecnet/lidar-preprocessing.hpp"

#include <stdexcept>

namespace semvecnet {

SemVecNet::SemVecNet(const std::string &hrnet_plan, const std::string &semvecnet_onnx)
    : hrnet_engine_(TensorRtEngine::load(hrnet_plan)), mapper_(), model_(semvecnet_onnx) {
  if (!hrnet_engine_)
    throw std::runtime_error("Failed to initialize HRNet engine and context");
}

Polylines SemVecNet::forward(const unsigned char *const *camera_images, const nvtype::half *lidar_points, int num_points, int num_cameras,
                             int image_width, int image_height, const std::vector<semantic_projection::CameraCalibration> &calibrations,
                              const std::array<float, 16> &lidar_to_map, const std::array<float, 16> &ego_to_map,
                              std::vector<std::uint8_t> *semantic_rgb) {
  if (!camera_images || num_cameras <= 0 || image_width <= 0 || image_height <= 0)
    throw std::invalid_argument("Expected RGB8 cameras with positive dimensions");

  if (num_points < 0 || (num_points > 0 && !lidar_points))
    throw std::invalid_argument("Expected five FP16 features per LiDAR point");

  if (calibrations.size() != static_cast<std::size_t>(num_cameras))
    throw std::invalid_argument("Expected one calibration per camera");

  // Prepare camera images and LiDAR points for inference and projection
  const camera::Preprocessor camera;
  const lidar::Preprocessor lidar;
  PreparedFrame prepared = {camera.forward(camera_images, num_cameras, image_width, image_height), lidar.forward(lidar_points, num_points)};

  // Run HRNet to get a per-pixel semantic class mask for each camera
  std::vector<std::vector<int64_t>> masks = hrnet::forward(*hrnet_engine_.context(), prepared.hrnet_images);

  // Associate LiDAR points with camera mask labels and update the BEV map
  auto labeled_points = semantic_projection::forward(prepared.lidar_points, masks, calibrations);
  mapper_.forward(prepared.lidar_points, labeled_points, lidar_to_map);

  // Render the ego-centric model input from the map.
  const model::SemanticMapInput semantic_map = renderer_.forward(mapper_.grid(), ego_to_map, semantic_rgb);

  // Predict normalized polyline points and class logits
  semvecnet::model::DecoderOutput predictions = model_.forward(semantic_map);

  // Score and filter predictions, converting their ordered points to meters
  std::vector<semvecnet::postprocessing::Polyline> polylines = postprocessor_.forward(predictions);

  return polylines;
}

} // namespace semvecnet
