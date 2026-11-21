#pragma once

#include <common/dtype.hpp>

#include <array>
#include <vector>

namespace semvecnet::lidar {

class Preprocessor {
public:
  // Requires a nonnegative count and a valid point buffer when count is positive.
  std::vector<std::array<float, 5>> forward(const nvtype::half *points,
                                            int count) const;
};

} // namespace semvecnet::lidar
