#pragma once

#include "semvecnet/base-config.hpp"
#include "semvecnet/lidar-semantic-projection.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
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
  void setGrid(Grid grid) { grid_ = std::move(grid); }

  // Updates the map using LiDAR points and their camera-label associations.
  // lidar_to_map is a row-major 4x4 transform from LiDAR to map coordinates.
  void forward(const std::vector<std::array<float, 5>> &lidar_points,
               const std::vector<semantic_projection::LabeledPoint> &labeled_points,
               const std::array<float, 16> &lidar_to_map);

private:
  using Origin = std::array<float, 2>;
  // Half-open cell ranges: {{x_begin, x_end}, {y_begin, y_end}}.
  using TileBoundary = std::array<std::array<std::size_t, 2>, 2>;

  // Python DynamicMap::check_map_origin / get_origin.
  void checkMapOrigin(const std::array<float, 16> &lidar_to_map);
  Origin getOrigin(const Origin &position) const;

  // Python init_map / need_update / update_map_origin.
  void initMap(const Origin &origin);
  bool needUpdate(const Origin &origin) const;
  void updateMapOrigin(const Origin &origin);

  // Python tile-coordinate and tile-state helpers.
  Origin getTileOrigin(std::size_t x, std::size_t y, const Origin &map_origin) const;
  TileBoundary getTileBoundary(std::size_t x, std::size_t y) const;
  bool tileExists(const Origin &tile_origin) const;
  bool tileIsActive(const Origin &tile_origin, const Origin &map_origin) const;

  // Python merge_color, adapted to HRNet class IDs instead of RGB colors.
  // Returns a BEV class index, or -1 for a class that is not mapped.
  int remapClass(std::int64_t class_id) const;

  // Python update_map: transform points, find cells, and update evidence.
  void updateMap(const std::vector<std::array<float, 5>> &lidar_points,
                 const std::vector<semantic_projection::LabeledPoint> &labeled_points,
                 const std::array<float, 16> &lidar_to_map);

  Grid grid_;
};

} // namespace semvecnet::bev_mapping
