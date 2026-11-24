#include "semvecnet/semvecnet.hpp"

#include "semvecnet/camera-preprocessing.hpp"
#include "semvecnet/lidar-preprocessing.hpp"

#include <stdexcept>

namespace semvecnet {

void forward(const unsigned char *const *camera_images,
             const nvtype::half *lidar_points, int num_points, int num_cameras,
             int image_width, int image_height) {
  if (!camera_images || num_cameras <= 0 || image_width <= 0 ||
      image_height <= 0)
    throw std::invalid_argument(
        "Expected RGB8 cameras with positive dimensions");

  if (num_points < 0 || (num_points > 0 && !lidar_points))
    throw std::invalid_argument("Expected five FP16 features per LiDAR point");

  const camera::Preprocessor camera;
  const lidar::Preprocessor lidar;
  PreparedFrame prepared = {
      camera.forward(camera_images, num_cameras, image_width, image_height),
      lidar.forward(lidar_points, num_points)};

  
}

} // namespace semvecnet
