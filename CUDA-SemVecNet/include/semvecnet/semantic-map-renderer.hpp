#pragma once

#include "semvecnet/semantic-bev-mapping.hpp"
#include "semvecnet/semantic-map-input.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace semvecnet::semantic_map {

inline constexpr int kHeight = 300;
inline constexpr int kWidth = 150;

class Renderer {
public:
  // Render a 60 m by 30 m ego-centric map using a row-major ego-to-map pose.
  // Optionally retain the packed RGB8 pixels before normalization.
  model::SemanticMapInput forward(const bev_mapping::Grid &grid,
                                  const std::array<float, 16> &ego_to_map,
                                  std::vector<std::uint8_t> *rgb = nullptr) const;
};

} // namespace semvecnet::semantic_map
