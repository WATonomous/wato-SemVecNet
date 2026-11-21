#include "semvecnet/semvecnet.hpp"

#include "semvecnet/camera-preprocessing.hpp"
#include "semvecnet/hrnet-ocrnet-inference.hpp"
#include "semvecnet/lidar-preprocessing.hpp"

#include <stdexcept>

namespace semvecnet {

std::vector<semantic_projection::LabeledPoint>
forward(const unsigned char *const *camera_images,
        const nvtype::half *lidar_points, int num_points, int num_cameras,
        int image_width, int image_height,
        const std::vector<semantic_projection::CameraCalibration> &calibrations) {
  if (!camera_images || num_cameras <= 0 || image_width <= 0 ||
      image_height <= 0)
    throw std::invalid_argument(
        "Expected RGB8 cameras with positive dimensions");

  if (num_points < 0 || (num_points > 0 && !lidar_points))
    throw std::invalid_argument("Expected five FP16 features per LiDAR point");

  if (calibrations.size() != static_cast<std::size_t>(num_cameras))
    throw std::invalid_argument("Expected one calibration per camera");

  const camera::Preprocessor camera;
  const lidar::Preprocessor lidar;
  PreparedFrame prepared = {
      camera.forward(camera_images, num_cameras, image_width, image_height),
      lidar.forward(lidar_points, num_points)};

  auto engine = TensorRtEngine::load("model/hrnet_ocrnet_mapillary.plan");
  if (!engine)
    throw std::runtime_error("Failed to initialize HRNet engine and context");

  std::vector<std::vector<int64_t>> masks = hrnet::forward(*engine.context(), prepared.hrnet_images);
  return semantic_projection::forward(prepared.lidar_points, masks, calibrations);
}

} // namespace semvecnet
