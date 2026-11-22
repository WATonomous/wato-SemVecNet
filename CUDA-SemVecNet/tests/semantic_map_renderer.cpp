#include "semvecnet/semantic-map-renderer.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

int main() {
  using namespace semvecnet;
  bev_mapping::Grid grid;
  grid.origin_x = -40.0f;
  grid.origin_y = -40.0f;
  grid.origin_initialized = true;
  const auto set = [&](std::size_t x, std::size_t y, const std::array<float, 4> &evidence) {
    const auto index = (x * grid.cells_y + y) * grid.class_count;
    for (std::size_t c = 0; c < evidence.size(); ++c)
      grid.evidence[index + c] = evidence[c];
  };
  // All-negative evidence is still observed. The strongest class is crosswalk.
  set(210, 200, {-3, -1, -4, -5});
  // A zero-sum cell can still contain evidence; its strongest class is sidewalk.
  set(200, 210, {-1, -1, -1, 3});

  const semantic_map::Renderer renderer;
  std::array<float, 16> pose = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::vector<std::uint8_t> rgb;
  const auto image = renderer.forward(grid, pose, &rgb);
  const auto near = [](float a, float b) { return std::abs(a - b) < 1e-5f; };
  const auto pixel = [](int channel, int row, int col) { return channel * 300 * 150 + row * 150 + col; };
  if (image.values.size() != 3 * 300 * 150 ||
      !near(image.values[pixel(0, 139, 74)], (140.0f - 123.675f) / 58.395f) ||
      !near(image.values[pixel(2, 139, 74)], (200.0f - 103.53f) / 57.375f) ||
      !near(image.values[pixel(0, 149, 74)], -123.675f / 58.395f))
    throw std::runtime_error("Incorrect map colors, normalization, or layout");
  const auto crosswalk = 3 * (139 * 150 + 74);
  const auto unknown = 3 * (149 * 150 + 74);
  if (rgb.size() != image.values.size() ||
      rgb[crosswalk] != 140 || rgb[crosswalk + 1] != 140 || rgb[crosswalk + 2] != 200 ||
      rgb[unknown] != 0 || rgb[unknown + 1] != 0 || rgb[unknown + 2] != 0)
    throw std::runtime_error("Incorrect packed RGB preview of model input");

  // A 90-degree yaw points ego forward toward increasing map y.
  pose[0] = 0; pose[1] = -1;
  pose[4] = 1; pose[5] = 0;
  const auto rotated = renderer.forward(grid, pose);
  if (!near(rotated.values[pixel(0, 139, 74)], (244.0f - 123.675f) / 58.395f) ||
      !near(rotated.values[pixel(1, 139, 74)], (35.0f - 116.28f) / 57.12f))
    throw std::runtime_error("Incorrect map rotation");

  // An ego pose outside the active grid produces black, not an out-of-bounds read.
  pose[3] = 1000;
  const auto outside = renderer.forward(grid, pose);
  if (!near(outside.values[pixel(0, 149, 74)], -123.675f / 58.395f))
    throw std::runtime_error("Expected an unobserved pixel outside the grid");
}
