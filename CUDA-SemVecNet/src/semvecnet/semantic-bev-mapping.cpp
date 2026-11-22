#include "semvecnet/semantic-bev-mapping.hpp"

#include <algorithm>
#include <cmath>

namespace semvecnet::bev_mapping {

namespace {
// Log likelihoods for the four classes, indexed by true class then observed class.
// Derived from the bundled HRNet confusion matrix after class merging and reweighting.
constexpr float kLogLikelihood[4][4] = {
    {-0.02048131f, -5.48753365f, -4.91826495f, -4.73037683f},
    {-1.83409733f, -0.20017053f, -3.99618096f, -5.72434692f},
    {-1.01453005f, -3.86334867f, -0.50561930f, -4.32016333f},
    {-2.81124422f, -7.19043372f, -6.59377868f, -0.06427482f}};
}

void Mapper::forward(const std::vector<std::array<float, 5>> &lidar_points,
                     const std::vector<semantic_projection::LabeledPoint> &labeled_points, const std::array<float, 16> &lidar_to_map) {
  checkMapOrigin(lidar_to_map);

  for (const auto &association : labeled_points) {
    const int class_index = remapClass(association.class_id);
    if (class_index < 0)
      continue;
    const auto &point = lidar_points[association.point_index];

    // Transform the point into map coordinates before finding its grid cell.
    const float x = lidar_to_map[0] * point[0] + lidar_to_map[1] * point[1] + lidar_to_map[2] * point[2] + lidar_to_map[3];
    const float y = lidar_to_map[4] * point[0] + lidar_to_map[5] * point[1] + lidar_to_map[6] * point[2] + lidar_to_map[7];
    const float cell_x = (x - grid_.origin_x) / grid_.resolution;
    const float cell_y = (y - grid_.origin_y) / grid_.resolution;

    // Check before conversion: truncating a negative float would put it in cell 0.
    if (!(cell_x >= 0.0f && cell_y >= 0.0f && cell_x < grid_.cells_x && cell_y < grid_.cells_y))
      continue;
    const auto x_index = static_cast<std::size_t>(cell_x);
    const auto y_index = static_cast<std::size_t>(cell_y);
    const auto index = (x_index * grid_.cells_y + y_index) * grid_.class_count + static_cast<std::size_t>(class_index);

    // Update each class with the observation likelihood and range weighting.
    const float distance = std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
    const float distance_weight = std::log(std::max(distance, 1e-6f));
    const auto cell_start = index - static_cast<std::size_t>(class_index);
    for (std::size_t c = 0; c < grid_.class_count; ++c)
      grid_.evidence[cell_start + c] += kLogLikelihood[c][class_index] + distance_weight;
    if (class_index == 2 && (point[3] < 2.0f || point[3] > 14.0f))
      grid_.evidence[index] += 2.0f;
  }
}

void Mapper::checkMapOrigin(const std::array<float, 16> &lidar_to_map) {
  // Initialize the grid origin from the LiDAR position.
  const Origin origin = getOrigin({lidar_to_map[3], lidar_to_map[7]});
  if (!grid_.origin_initialized) {
    grid_.origin_x = origin[0];
    grid_.origin_y = origin[1];
    grid_.origin_initialized = true;
  } else if (grid_.origin_x != origin[0] || grid_.origin_y != origin[1]) {
    updateMapOrigin(origin);
  }
}

Mapper::Origin Mapper::getOrigin(const Origin &position) const {
  // Align the origin to the tile grid, including for negative coordinates.
  return {(std::floor(position[0] / kBevCellAlignedTileSizeMetres) - kBevTilePadding) * kBevCellAlignedTileSizeMetres,
          (std::floor(position[1] / kBevCellAlignedTileSizeMetres) - kBevTilePadding) * kBevCellAlignedTileSizeMetres};
}

void Mapper::updateMapOrigin(const Origin &origin) {
  constexpr std::size_t tile_count = 2 * kBevTilePadding + 1;
  const auto tile_key = [](float x, float y) -> TileKey {
    return {static_cast<int>(std::lround(x / kBevCellAlignedTileSizeMetres)),
            static_cast<int>(std::lround(y / kBevCellAlignedTileSizeMetres))};
  };
  const auto old_base = tile_key(grid_.origin_x, grid_.origin_y);
  const auto new_base = tile_key(origin[0], origin[1]);
  const std::size_t tile_row_size = kBevCellsPerTile * grid_.class_count;

  // Store every active tile, including those that will stay in the active region.
  for (std::size_t tx = 0; tx < tile_count; ++tx) {
    for (std::size_t ty = 0; ty < tile_count; ++ty) {
      auto &tile = tiles_[{old_base.first + static_cast<int>(tx), old_base.second + static_cast<int>(ty)}];
      tile.resize(kBevCellsPerTile * tile_row_size);
      for (std::size_t x = 0; x < kBevCellsPerTile; ++x) {
        const auto source = ((tx * kBevCellsPerTile + x) * grid_.cells_y + ty * kBevCellsPerTile) * grid_.class_count;
        std::copy_n(grid_.evidence.begin() + source, tile_row_size, tile.begin() + x * tile_row_size);
      }
    }
  }

  std::fill(grid_.evidence.begin(), grid_.evidence.end(), 0.0f);
  for (std::size_t tx = 0; tx < tile_count; ++tx) {
    for (std::size_t ty = 0; ty < tile_count; ++ty) {
      const auto it = tiles_.find({new_base.first + static_cast<int>(tx), new_base.second + static_cast<int>(ty)});
      if (it == tiles_.end())
        continue;
      for (std::size_t x = 0; x < kBevCellsPerTile; ++x) {
        const auto destination = ((tx * kBevCellsPerTile + x) * grid_.cells_y + ty * kBevCellsPerTile) * grid_.class_count;
        std::copy_n(it->second.begin() + x * tile_row_size, tile_row_size, grid_.evidence.begin() + destination);
      }
      tiles_.erase(it);
    }
  }
  grid_.origin_x = origin[0];
  grid_.origin_y = origin[1];
}

int Mapper::remapClass(std::int64_t class_id) const {
  // Merge zebra line into crosswalk and bike lane into road.
  if (class_id == 23)
    class_id = 8;
  else if (class_id == 7)
    class_id = 13;

  const auto it = std::find(kBevClassIds.begin(), kBevClassIds.end(), class_id);
  return it == kBevClassIds.end() ? -1 : static_cast<int>(it - kBevClassIds.begin());
}

} // namespace semvecnet::bev_mapping
