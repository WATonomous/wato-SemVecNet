#pragma once

#include <vector>

namespace semvecnet::camera {

class Preprocessor {
public:
  // Requires a valid pointer array, positive camera count and positive dimensions.
  std::vector<std::vector<float>> forward(const unsigned char *const *images,
                                          int count, int width,
                                          int height) const;
};

} // namespace semvecnet::camera
