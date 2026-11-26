#include "semvecnet/camera-preprocessing.hpp"

#include "semvecnet/base-config.hpp"
#include "semvecnet/hrnet-ocrnet-inference.hpp"

#include <nvtx3/nvtx3.hpp>
#include <stdexcept>

namespace semvecnet::camera {
namespace {

// These values are from the Python code
__constant__ float kMean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float kStd[3] = {0.229f, 0.224f, 0.225f};
constexpr std::size_t kImageFloats = std::size_t(kHrnetChannels) * kHrnetWidth * kHrnetHeight;

// src: N raw camera images back to back, [N, src_height, src_width, 3] RGB8
// dst: the HRNet input buffer,           [N, 3, 720, 960] FP32
__global__ void prepareImages(const uint8_t *__restrict__ src, float *__restrict__ dst, int src_width, int src_height) {
  std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t cam = blockIdx.z;

  if (x >= kHrnetWidth || y >= kHrnetHeight)
    return;

  // Scale the image to be 960x720, this is essentially what cv::resize with cv::INTER_AREA does
  const float sx = float(src_width) / kHrnetWidth, sy = float(src_height) / kHrnetHeight;
  const float fx1 = x * sx, fx2 = fx1 + sx;
  const float fy1 = y * sy, fy2 = fy1 + sy;
  const uint8_t *img = src + size_t(cam) * src_width * src_height * 3;

  // Go to the surrounding 4 or 6 pixels and get all their RGB values since we're scaling the image down
  float acc[3] = {0.f, 0.f, 0.f};
  for (int iy = int(fy1); iy < src_height && iy < fy2; ++iy) {
    const float wy = fminf(fy2, iy + 1.f) - fmaxf(fy1, float(iy)); // vertical overlap
    for (int ix = int(fx1); ix < src_width && ix < fx2; ++ix) {
      const float w = wy * (fminf(fx2, ix + 1.f) - fmaxf(fx1, float(ix)));
      const uint8_t *p = img + (size_t(iy) * src_width + ix) * 3;
      acc[0] += w * p[0];
      acc[1] += w * p[1];
      acc[2] += w * p[2];
    }
  }

  // Size of one plane which is 960 * 720
  const size_t plane = size_t(kHrnetWidth) * kHrnetHeight;
  float *out = dst + size_t(cam) * 3 * plane + size_t(y) * kHrnetWidth + x;
  const float area = sx * sy;
  for (int c = 0; c < 3; ++c) {
    // OpenCV resizes RGB8 → RGB8, so round to a byte before normalizing
    const float v = fminf(fmaxf(rintf(acc[c] / area), 0.f), 255.f);
    out[c * plane] = (v / 255.f - kMean[c]) / kStd[c]; // write straight into the R, G, B planes
  }
}

} // namespace

Preprocessor::Preprocessor() {
  cudaStreamCreate(&stream_);
  cudaMalloc(&dst_d_, hrnet::kMaxBatch * kImageFloats * sizeof(float));
}

Preprocessor::~Preprocessor() {
  if (stream_)
    cudaStreamSynchronize(stream_);
  cudaFree(raw_d_);
  cudaFree(dst_d_);
  if (stream_)
    cudaStreamDestroy(stream_);
}

const float *Preprocessor::forward(const unsigned char *const *images, int count, int width, int height) {
  const nvtx3::scoped_range range{"SemVecNet::camera_preprocess"};
  if (count <= 0 || count > hrnet::kMaxBatch)
    throw std::invalid_argument("Expected 1 to 8 camera images");
  // The kernel's area averaging only handles shrinking
  if (width < kHrnetWidth || height < kHrnetHeight)
    throw std::invalid_argument("Camera images must be at least 960x720");
  for (int i = 0; i < count; ++i)
    if (!images[i])
      throw std::invalid_argument("Missing RGB8 camera image");

  const std::size_t image_bytes = std::size_t(width) * height * 3;

  // Only allocates on the first frame, or if a later frame is bigger
  if (image_bytes * count > raw_capacity_) {
    cudaFree(raw_d_);
    raw_d_ = nullptr;
    raw_capacity_ = 0;
    cudaMalloc(&raw_d_, image_bytes * count);
    raw_capacity_ = image_bytes * count;
  }

  for (int i = 0; i < count; ++i)
    cudaMemcpyAsync(raw_d_ + i * image_bytes, images[i], image_bytes, cudaMemcpyHostToDevice, stream_);

  const dim3 block(32, 8);

  // Add 31 and 7 because C++ rounds down
  const dim3 grid((kHrnetWidth + 31) / 32, (kHrnetHeight + 7) / 8, count);

  prepareImages<<<grid, block, 0, stream_>>>(raw_d_, dst_d_, width, height);

  cudaStreamSynchronize(stream_);
  return dst_d_;
}

} // namespace semvecnet::camera
