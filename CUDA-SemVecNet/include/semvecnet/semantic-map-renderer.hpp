#pragma once

#include "semvecnet/semantic-bev-mapping.hpp"
#include "semvecnet/semantic-map-input.hpp"

#include <array>

namespace semvecnet::semantic_map {

class Renderer {
public:
  // Render a 60 m by 30 m ego-centric map using a row-major ego-to-map pose.
  model::SemanticMapInput forward(const bev_mapping::Grid &grid,
                                  const std::array<float, 16> &ego_to_map) const;
};

} // namespace semvecnet::semantic_map
