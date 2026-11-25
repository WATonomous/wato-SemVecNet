#include "semvecnet/camera-preprocessing.hpp"
#include "semvecnet/lidar-preprocessing.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  constexpr int width = 1280;
  constexpr int height = 1024;
  constexpr int cameras = 8;
  constexpr int plane = 960 * 720;
  std::array<std::vector<unsigned char>, cameras> pixels;
  std::array<const unsigned char *, cameras> images{};
  for (int i = 0; i < cameras; ++i) {
    auto &rgb = pixels[i];
    rgb.resize(width * height * 3);
    for (std::size_t p = 0; p < rgb.size(); p += 3) {
      rgb[p] = i * 20;
      rgb[p + 1] = 64;
      rgb[p + 2] = 255;
    }
    images[i] = rgb.data();
  }
  // The first output pixel covers 4/3 x 64/45 source pixels.
  // A red value of 160 at the first source pixel contributes 84.375 -> 84.
  pixels[0][0] = 160;
  const nvtype::half lidar[] = {{0x3c00}, {0xc000}, {0x3800}, {0x4800},
                                {0},      {0x4400}, {0x4000}, {0x4200},
                                {0},      {0x3400}};
  semvecnet::camera::Preprocessor camera_preprocessor;
  const semvecnet::lidar::Preprocessor lidar_preprocessor;
  const auto hrnet_images = camera_preprocessor.forward(images.data(), cameras, width, height);
  const auto lidar_points = lidar_preprocessor.forward(lidar, 2);
  const auto near = [](float a, float b) { return std::abs(a - b) < 1e-6f; };
  if (hrnet_images.size() != cameras) {
    std::cerr << "Incorrect camera count\n";
    return 1;
  }
  for (int i = 0; i < cameras; ++i) {
    const auto &tensor = hrnet_images[i];
    if (tensor.size() != 3 * plane ||
        !near(tensor[0], ((i == 0 ? 84 : i * 20) / 255.0f - 0.485f) / 0.229f) ||
        !near(tensor[plane - 1], (i * 20 / 255.0f - 0.485f) / 0.229f) ||
        !near(tensor[plane], (64 / 255.0f - 0.456f) / 0.224f) ||
        !near(tensor[2 * plane], (1.0f - 0.406f) / 0.225f)) {
      std::cerr << "Unexpected HRNet tensor for camera " << i << '\n';
      return 1;
    }
  }
  const std::vector<std::array<float, 5>> expected = {{1, -2, 0.5f, 8, 0},
                                                      {4, 2, 3, 0, 0.25f}};
  if (lidar_points != expected) {
    std::cerr << "Incorrect FP16 LiDAR decoding or point stride\n";
    return 1;
  }
  images.back() = nullptr;
  try {
    camera_preprocessor.forward(images.data(), cameras, width, height);
  } catch (const std::invalid_argument &) {
    return 0;
  }
  std::cerr << "Missing camera was accepted\n";
  return 1;
}
