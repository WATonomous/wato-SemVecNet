#include "semvecnet/semvecnet.hpp"

#include <nvtx3/nvtx3.hpp>

#include <stdexcept>

namespace semvecnet {
SemVecNet::SemVecNet(const std::string &hrnet_plan, const std::string &semvecnet_onnx)
    : hrnet_engine_(TensorRtEngine::load(hrnet_plan)), hrnet_inference_(hrnet_engine_.get()), mapper_(), model_(semvecnet_onnx) {
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

  const nvtx3::scoped_range total{"SemVecNet::forward"};

  // Prepare camera images
  const float *images_d = camera_preprocessor_.forward(camera_images, num_cameras, image_width, image_height);

  // Prepare Lidar points
  const auto decoded_points = lidar_preprocessor_.forward(lidar_points, num_points);

  // Run HRNet to get a per-pixel semantic class mask for each camera
  const std::int64_t *masks_d = hrnet_inference_.forward(images_d, num_cameras);

  // Associate LiDAR points with camera mask labels and update the BEV map
  const auto labeled_points = projector_.forward(decoded_points, masks_d, num_cameras, calibrations, image_width, image_height);

  // Update the semantic map
  mapper_.forward(decoded_points, labeled_points, lidar_to_map);

  // Render the ego-centric model input from the map.
  const model::SemanticMapInput semantic_map = renderer_.forward(mapper_.grid(), ego_to_map, semantic_rgb);

  // Predict normalized polyline points and class logits
  semvecnet::model::DecoderOutput predictions = model_.forward(semantic_map);

  // Score and filter predictions, converting their ordered points to meters
  std::vector<semvecnet::postprocessing::Polyline> polylines = postprocessor_.forward(predictions);

  return polylines;
}

} // namespace semvecnet
