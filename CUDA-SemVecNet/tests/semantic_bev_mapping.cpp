#include "semvecnet/semantic-bev-mapping.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
  using namespace semvecnet::bev_mapping;
  using semvecnet::semantic_projection::LabeledPoint;
  Mapper mapper;
  const std::array<float, 16> transform = {0, -1, 0, 1, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1};
  const std::vector<std::array<float, 5>> points = {
      {1.1f, 2.1f, 0, 5, 0}, {1000, 0, 0, 0, 0}};
  const std::vector<LabeledPoint> labels = {{0, 0, 23}, {0, 1, 8}, {0, 0, 99}, {1, 0, 13}};
  mapper.forward(points, labels, transform);
  const auto &grid = mapper.grid();
  const auto near = [](float a, float b) { return std::abs(a - b) < 1e-4f; };
  if (!grid.origin_initialized || !near(grid.origin_x, -35.4f) || !near(grid.origin_y, -70.8f))
    throw std::runtime_error("Unexpected grid origin");
  const auto index_for = [&](float x, float y) {
    const auto ix = static_cast<std::size_t>((x - grid.origin_x) / grid.resolution);
    const auto iy = static_cast<std::size_t>((y - grid.origin_y) / grid.resolution);
    return (ix * grid.cells_y + iy) * grid.class_count;
  };
  const auto index = index_for(-1.1f, 0.1f);
  const float weight = std::log(std::sqrt(1.1f * 1.1f + 2.1f * 2.1f));
  if (!near(grid.evidence[index], -5.48753365f + weight) ||
      !near(grid.evidence[index + 1], -0.20017053f + weight))
    throw std::runtime_error("Unexpected remapping or class likelihood");
  mapper.forward(points, {{0, 0, 24}}, transform);
  if (!near(grid.evidence[index + 2], (-3.86334867f + weight) + (-0.50561930f + weight)))
    throw std::runtime_error("Evidence did not persist across frames");

  const std::array<float, 4> saved = {grid.evidence[index], grid.evidence[index + 1],
                                      grid.evidence[index + 2], grid.evidence[index + 3]};
  auto moved = transform;
  moved[3] = -1;
  mapper.forward(points, {}, moved);
  if (!near(grid.origin_x, -70.8f))
    throw std::runtime_error("Grid did not move");
  const auto shifted = index_for(-1.1f, 0.1f);
  for (std::size_t c = 0; c < saved.size(); ++c)
    if (!near(grid.evidence[shifted + c], saved[c]))
      throw std::runtime_error("Overlapping tile evidence was not preserved");
  mapper.forward(points, {}, transform);
  const auto overlap = index_for(-1.1f, 0.1f);
  for (std::size_t c = 0; c < saved.size(); ++c)
    if (!near(grid.evidence[overlap + c], saved[c]))
      throw std::runtime_error("Overlapping tile evidence was not preserved");
  moved[3] = 200;
  mapper.forward(points, {}, moved);
  mapper.forward(points, {}, transform);
  const auto restored = index_for(-1.1f, 0.1f);
  for (std::size_t c = 0; c < saved.size(); ++c)
    if (!near(grid.evidence[restored + c], saved[c]))
      throw std::runtime_error("Inactive tile evidence was not restored");

  Mapper intensity_mapper;
  const std::vector<std::array<float, 5>> intensity_points = {{1.1f, 2.1f, 0, 1, 0}};
  intensity_mapper.forward(intensity_points, {{0, 0, 24}}, transform);
  if (!near(intensity_mapper.grid().evidence[index + 2], -0.50561930f + weight + 2.0f))
    throw std::runtime_error("Lane intensity weighting was not applied");

  Mapper duplicates;
  const std::vector<std::array<float, 5>> repeated = {
      {1.11f, 2.11f, 0, 1, 0}, {1.12f, 2.12f, 0, 5, 0}};
  duplicates.forward(repeated, {{0, 0, 24}, {1, 1, 24}, {0, 0, 13}}, transform);
  const float last_weight = std::log(std::hypot(1.12f, 2.12f));
  const float road_weight = std::log(std::hypot(1.11f, 2.11f));
  if (!near(duplicates.grid().evidence[index + 2],
            (-1.01453005f + road_weight) + (-0.50561930f + last_weight) + 2.0f))
    throw std::runtime_error("Duplicate grouping, class order, or intensity boost is incorrect");
  duplicates.forward(repeated, {{0, 0, 24}, {1, 1, 24}}, transform);
  if (!near(duplicates.grid().evidence[index + 2],
            (-1.01453005f + road_weight) + 2 * ((-0.50561930f + last_weight) + 2.0f)))
    throw std::runtime_error("Observations across separate sweeps must accumulate");
}
