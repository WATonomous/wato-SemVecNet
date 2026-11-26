#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace semvecnet::camera {

// Owns the GPU buffers and stream used for preprocessing, so create it once and reuse it.
// Calls to forward must not overlap.
class Preprocessor {
public:
  Preprocessor();
  ~Preprocessor();
  Preprocessor(const Preprocessor &) = delete;
  Preprocessor &operator=(const Preprocessor &) = delete;

  // Requires a valid pointer array, 1 to 8 cameras, and RGB8 images of at least 960x720.
  // Returns device memory laid out [count, 3, 720, 960], valid until the next call.
  const float *forward(const unsigned char *const *images, int count, int width, int height);

private:
  cudaStream_t stream_ = nullptr;
  std::uint8_t *raw_d_ = nullptr; // camera images back to back, grown when a frame needs more
  float *dst_d_ = nullptr;        // normalized planes for hrnet::kMaxBatch cameras
  std::size_t raw_capacity_ = 0;   // bytes allocated at raw_d_
};

} // namespace semvecnet::camera
