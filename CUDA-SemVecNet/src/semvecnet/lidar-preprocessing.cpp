#include "semvecnet/lidar-preprocessing.hpp"

#include <cuda_fp16.h>

#include <cstring>

namespace semvecnet::lidar {

std::vector<std::array<float, 5>>
Preprocessor::forward(const nvtype::half *points, int count) const {
  std::vector<std::array<float, 5>> decoded(count);

  // The ROS wrapper may pass a reinterpreted __half buffer. Copy the bits
  // instead of dereferencing a pointer to a different C++ type.
  static_assert(sizeof(nvtype::half) == sizeof(__half_raw));
  const auto *bits = reinterpret_cast<const unsigned char *>(points);

  for (auto &point : decoded) {
    for (float &feature : point) {
      __half_raw raw{};
      std::memcpy(&raw, bits, sizeof(raw));
      feature = __half2float(__half(raw));
      bits += sizeof(raw);
    }
  }
  return decoded;
}

} // namespace semvecnet::lidar
