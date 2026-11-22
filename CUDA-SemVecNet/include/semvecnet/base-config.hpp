#pragma once

#include <array>

namespace semvecnet {

// Defaults for the HRNet model and semantic projection.
inline constexpr int kHrnetWidth = 960;
inline constexpr int kHrnetHeight = 720;
inline constexpr int kHrnetChannels = 3;
inline constexpr std::array<float, 3> kHrnetMean = {0.485f, 0.456f, 0.406f};
inline constexpr std::array<float, 3> kHrnetStd = {0.229f, 0.224f, 0.225f};
// Python base_cfg.py MAPPING.PCD.RANGE_MAX, in metres along camera Z.
inline constexpr float kProjectionMaxDepth = 30.0f;

} // namespace semvecnet
