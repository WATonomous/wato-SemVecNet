#pragma once

#include <array>
#include <cstddef>

namespace semvecnet {

// Defaults for the HRNet model and semantic projection.
inline constexpr int kHrnetWidth = 960;
inline constexpr int kHrnetHeight = 720;
inline constexpr int kHrnetChannels = 3;
inline constexpr std::array<float, 3> kHrnetMean = {0.485f, 0.456f, 0.406f};
inline constexpr std::array<float, 3> kHrnetStd = {0.229f, 0.224f, 0.225f};
// Python base_cfg.py MAPPING.PCD.RANGE_MAX, in metres along camera Z.
inline constexpr float kProjectionMaxDepth = 30.0f;

// Python base_cfg.py mapping defaults (dataset configurations may override).
inline constexpr float kBevResolution = 0.2f;
inline constexpr float kBevTileSizeMetres = 35.5f;
inline constexpr std::size_t kBevTilePadding = 1;
inline constexpr std::size_t kBevCellsPerTile =
    static_cast<std::size_t>(kBevTileSizeMetres / kBevResolution);
inline constexpr std::size_t kBevGridCells =
    (2 * kBevTilePadding + 1) * kBevCellsPerTile;
// Python LABELS and LABELS_NAMES: road, crosswalk, lane, sidewalk.
inline constexpr std::array<int, 4> kBevClassIds = {13, 8, 24, 15};

} // namespace semvecnet
