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
// Maximum projection depth, in metres along camera Z.
inline constexpr float kProjectionMaxDepth = 30.0f;

// Exported SemVecNet model's point_cloud_range, in metres: [xmin, ymin, xmax, ymax].
// These must match the model's training configuration and semantic-map extent.
inline constexpr std::array<float, 4> kSemvecnetPointCloudRange = {-15.0f, -30.0f, 15.0f, 30.0f};
// Postprocessing bounds for [xmin, ymin, xmax, ymax], in metres.
inline constexpr std::array<float, 4> kSemvecnetPostCenterLower = {-20.0f, -35.0f, -20.0f, -35.0f};
inline constexpr std::array<float, 4> kSemvecnetPostCenterUpper = {20.0f, 35.0f, 20.0f, 35.0f};
// Minimum confidence score for predicted polylines.
inline constexpr float kSemvecnetConfidenceThreshold = 0.4f;

// Default BEV mapping resolution and tile dimensions.
inline constexpr float kBevResolution = 0.2f;
inline constexpr float kBevTileSizeMetres = 35.5f;
inline constexpr std::size_t kBevTilePadding = 1;
inline constexpr std::size_t kBevCellsPerTile =
    static_cast<std::size_t>(kBevTileSizeMetres / kBevResolution);
inline constexpr float kBevCellAlignedTileSizeMetres = kBevCellsPerTile * kBevResolution;
inline constexpr std::size_t kBevGridCells =
    (2 * kBevTilePadding + 1) * kBevCellsPerTile;
// HRNet class IDs for road, crosswalk, lane, and sidewalk.
inline constexpr std::array<int, 4> kBevClassIds = {13, 8, 24, 15};

} // namespace semvecnet
