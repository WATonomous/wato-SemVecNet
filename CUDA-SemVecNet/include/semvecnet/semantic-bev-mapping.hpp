#pragma once

#include "semvecnet/base-config.hpp"
#include "semvecnet/lidar-semantic-projection.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace semvecnet::bev_mapping {

// Per-class evidence for each map cell. All-zero cells have no evidence.
struct Grid {
  std::size_t cells_x = kBevGridCells;
  std::size_t cells_y = kBevGridCells;
  std::size_t class_count = kBevClassIds.size();
  float resolution = kBevResolution; // metres per cell
  // Lower corner of the current active map, in map coordinates.
  float origin_x = 0.0f;
  float origin_y = 0.0f;
  bool origin_initialized = false;
  // Index: (x * cells_y + y) * class_count + class_index.
  std::vector<float> evidence = std::vector<float>(cells_x * cells_y * class_count, 0.0f);
};

// Keep one mapper alive across frames so its BEV grid can persist.
class Mapper {
public:
  const Grid &grid() const { return grid_; }
  // Replacing the active grid clears saved tiles; dimensions must match the configured tile layout.
  void setGrid(Grid grid) { grid_ = std::move(grid); tiles_.clear(); }

  // Updates the map using LiDAR points and their camera-label associations.
  // lidar_to_map is a row-major 4x4 transform from LiDAR to map coordinates.
  // Keeps evidence in active and inactive tiles as the vehicle moves.
  // Expects a valid grid, finite transform, and in-range association indices.
  void forward(const std::vector<std::array<float, 5>> &lidar_points, const std::vector<semantic_projection::LabeledPoint> &labeled_points,
               const std::array<float, 16> &lidar_to_map);

private:
  using Origin = std::array<float, 2>;
  using TileKey = std::pair<int, int>;

  // Initialize the map origin and calculate tile-aligned origins.
  void checkMapOrigin(const std::array<float, 16> &lidar_to_map);
  Origin getOrigin(const Origin &position) const;

  // Save active tiles and load the tiles around the new origin.
  void updateMapOrigin(const Origin &origin);

  // Map an HRNet class ID to a BEV class index, or -1 if it is not mapped.
  int remapClass(std::int64_t class_id) const;

  Grid grid_;
  std::map<TileKey, std::vector<float>> tiles_;
};

} // namespace semvecnet::bev_mapping
