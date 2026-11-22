#include "semvecnet/semantic-bev-mapping.hpp"

#include <stdexcept>

namespace semvecnet::bev_mapping {

void Mapper::forward(const std::vector<std::array<float, 5>> & /*lidar_points*/,
                     const std::vector<semantic_projection::LabeledPoint> & /*labeled_points*/,
                     const std::array<float, 16> & /*lidar_to_map*/) {
  // TODO: Remap HRNet classes and update the persistent BEV grid.
  throw std::logic_error("Semantic BEV mapping is not implemented");
}

Mapper::Origin Mapper::getOrigin(const Origin &position) const {}

} // namespace semvecnet::bev_mapping
