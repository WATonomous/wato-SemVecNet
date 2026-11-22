#include "semvecnet/camera-preprocessing.hpp"

#include "semvecnet/semvecnet.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace semvecnet::camera {
namespace {

constexpr std::size_t kPlaneSize = kHrnetWidth * kHrnetHeight;

std::vector<float> prepareImage(const unsigned char *rgb, int width,
                                int height) {
  // Read-only view of the caller's packed RGB8 buffer; no BGR conversion.
  const cv::Mat image(height, width, CV_8UC3, const_cast<unsigned char *>(rgb));
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(kHrnetWidth, kHrnetHeight), 0, 0,
             cv::INTER_AREA);

  // Resize RGB8, then normalize each FP32 channel with (value / 255 - mean) / std.
  cv::Mat normalized;
  resized.convertTo(normalized, CV_32F);
  cv::divide(normalized, cv::Scalar::all(255.0), normalized);
  cv::subtract(normalized, cv::Scalar(kHrnetMean[0], kHrnetMean[1], kHrnetMean[2]), normalized);
  cv::divide(normalized, cv::Scalar(kHrnetStd[0], kHrnetStd[1], kHrnetStd[2]), normalized);

  // Like NumPy's moveaxis followed by ascontiguousarray: write R, G and B
  // directly into three consecutive planes of the [1, 3, 720, 960] tensor.
  std::vector<float> tensor(kHrnetChannels * kPlaneSize);
  cv::Mat planes[kHrnetChannels];
  for (int c = 0; c < kHrnetChannels; ++c)
    planes[c] = cv::Mat(kHrnetHeight, kHrnetWidth, CV_32FC1,
                        tensor.data() + c * kPlaneSize);
  cv::split(normalized, planes);
  return tensor;
}

} // namespace

std::vector<std::vector<float>>
Preprocessor::forward(const unsigned char *const *images, int count, int width,
                      int height) const {
  for (int i = 0; i < count; ++i)
    if (!images[i])
      throw std::invalid_argument("Missing RGB8 camera image");

  std::vector<std::vector<float>> tensors;
  tensors.reserve(count);
  for (int i = 0; i < count; ++i)
    tensors.push_back(prepareImage(images[i], width, height));
  return tensors;
}

} // namespace semvecnet::camera
