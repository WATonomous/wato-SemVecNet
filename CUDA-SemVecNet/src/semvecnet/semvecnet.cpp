#include "semvecnet/semvecnet.hpp"

#include "semvecnet/camera-preprocessing.hpp"
#include "semvecnet/hrnet-ocrnet-inference.hpp"
#include "semvecnet/lidar-preprocessing.hpp"

#include <stdexcept>
#include <utility>

namespace semvecnet {

namespace {
model::SemanticMapInput renderSemanticMap(const bev_mapping::Grid & /*grid*/,
                                         const std::array<float, 16> & /*lidar_to_map*/) {
  throw std::logic_error("Rendering the ego-centric semantic map is not implemented");
}
} // namespace

SemVecNet::SemVecNet(const std::string &hrnet_plan, const std::string &semvecnet_onnx)
    : hrnet_engine_(TensorRtEngine::load(hrnet_plan)), mapper_(), model_(semvecnet_onnx) {
  if (!hrnet_engine_)
    throw std::runtime_error("Failed to initialize HRNet engine and context");
}

ForwardResult SemVecNet::forward(const unsigned char *const *camera_images, const nvtype::half *lidar_points,
                                                       int num_points, int num_cameras, int image_width, int image_height,
                                                       const std::vector<semantic_projection::CameraCalibration> &calibrations,
                                                       const std::array<float, 16> &lidar_to_map) {
  if (!camera_images || num_cameras <= 0 || image_width <= 0 || image_height <= 0)
    throw std::invalid_argument("Expected RGB8 cameras with positive dimensions");

  if (num_points < 0 || (num_points > 0 && !lidar_points))
    throw std::invalid_argument("Expected five FP16 features per LiDAR point");

  if (calibrations.size() != static_cast<std::size_t>(num_cameras))
    throw std::invalid_argument("Expected one calibration per camera");

  const camera::Preprocessor camera;
  const lidar::Preprocessor lidar;
  PreparedFrame prepared = {camera.forward(camera_images, num_cameras, image_width, image_height), lidar.forward(lidar_points, num_points)};

  std::vector<std::vector<int64_t>> masks = hrnet::forward(*hrnet_engine_.context(), prepared.hrnet_images);
  auto labeled_points = semantic_projection::forward(prepared.lidar_points, masks, calibrations);
  mapper_.forward(prepared.lidar_points, labeled_points, lidar_to_map);
  const model::SemanticMapInput semantic_map = renderSemanticMap(mapper_.grid(), lidar_to_map);
  auto predictions = model_.forward(semantic_map);
  return {std::move(labeled_points), std::move(predictions)};
}

} // namespace semvecnet
