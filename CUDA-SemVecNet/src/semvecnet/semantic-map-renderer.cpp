#include "semvecnet/semantic-map-renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace semvecnet::semantic_map {
namespace {
constexpr std::array<std::array<float, 3>, 4> kColors = {{{{128, 64, 128}}, {{140, 140, 200}}, {{255, 255, 255}}, {{244, 35, 232}}}};
constexpr std::array<float, 3> kMean = {123.675f, 116.28f, 103.53f};
constexpr std::array<float, 3> kStd = {58.395f, 57.12f, 57.375f};
} // namespace

model::SemanticMapInput Renderer::forward(const bev_mapping::Grid &grid, const std::array<float, 16> &ego_to_map,
                                          std::vector<std::uint8_t> *rgb) const {
  model::SemanticMapInput output;
  output.values.resize(3 * kHeight * kWidth);
  if (rgb)
    rgb->resize(output.values.size());

  // Locate the ego position in grid cells; map x indexes rows and map y columns.
  const float center_x = std::nearbyint((ego_to_map[3] - grid.origin_x) / grid.resolution);
  const float center_y = std::nearbyint((ego_to_map[7] - grid.origin_y) / grid.resolution);

  // Rotate the ego-aligned output view into the map's coordinate frame.
  // TODO: match Python DynamicMap.render_local_map. It applies a 3x3 mean filter
  // (renderer.apply_filter) when its `filter` argument is true. The nuScenes
  // mapping path passes filter=False, which is what this port matches; add the
  // filter here only if we need parity with the generic/single-camera path.
  const float yaw = std::atan2(ego_to_map[4], ego_to_map[5]);
  const float cosine = std::cos(yaw);
  const float sine = std::sin(yaw);

  // TODO: apply all of these operations to every pixel in parallel? (only have like 150x300 pixels)
  for (int row = 0; row < kHeight; ++row) {
    for (int col = 0; col < kWidth; ++col) {
      // Each output pixel covers 0.2 m; flip both axes so ego-forward is up.
      // Convert the pixel's offset to grid-cell units before rotating it.
      const float dx = (kHeight / 2 - 1 - row) * 0.2f / grid.resolution;
      const float dy = (kWidth / 2 - 1 - col) * 0.2f / grid.resolution;

      const float map_x = center_x + cosine * dx - sine * dy;
      const float map_y = center_y + sine * dx + cosine * dy;

      // Sample the nearest grid cell rather than blending semantic classes.
      const long x = std::lrint(map_x);
      const long y = std::lrint(map_y);

      // Unobserved cells and positions outside the grid stay black.
      std::array<float, 3> color = {0, 0, 0};
      if (x >= 0 && y >= 0 && static_cast<std::size_t>(x) < grid.cells_x && static_cast<std::size_t>(y) < grid.cells_y) {
        const auto offset = (static_cast<std::size_t>(x) * grid.cells_y + static_cast<std::size_t>(y)) * grid.class_count;
        const auto begin = grid.evidence.begin() + offset;
        const auto end = begin + grid.class_count;

        // The mapper accumulates weighted evidence per class; choose the largest
        // score, even when every score is negative.
        if (std::any_of(begin, end, [](float value) { return value != 0.0f; })) {
          const auto best = static_cast<std::size_t>(std::max_element(begin, end) - begin);
          color = kColors[best];
        }
      }
      // Normalize RGB and store each channel in its own contiguous image plane.
      const auto pixel = static_cast<std::size_t>(row * kWidth + col);
      for (std::size_t channel = 0; channel < color.size(); ++channel) {
        output.values[channel * kHeight * kWidth + pixel] = (color[channel] - kMean[channel]) / kStd[channel];
        if (rgb)
          (*rgb)[pixel * 3 + channel] = static_cast<std::uint8_t>(color[channel]);
      }
    }
  }
  return output;
}

} // namespace semvecnet::semantic_map
