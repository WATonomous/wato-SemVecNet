#include "semvecnet/semvecnet.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;

// Read the little-endian frame stream written by tool/eval/run_rosbag_smoke.py.
template <typename T> void read(std::istream &in, T *data, std::size_t count = 1) {
  if (!in.read(reinterpret_cast<char *>(data), count * sizeof(T)))
    throw std::runtime_error("Truncated sensor frame");
}

constexpr std::array<const char *, 4> kClasses = {"divider", "ped_crossing", "boundary", "centerline"};
const std::array<cv::Scalar, 4> kClassBgr = {{{0, 165, 255}, {255, 80, 0}, {0, 0, 255}, {0, 200, 0}}};
constexpr float kLidarForwardOffset = 0.943713f; // nuScenes model origin ahead of the rear axle.

std::array<float, 2> toEgo(const std::array<float, 2> &point) {
  return {point[1] + kLidarForwardOffset, -point[0]}; // Model right/forward -> ego forward/left.
}

cv::Mat combinedView(const cv::Mat &map, const semvecnet::Polylines &lines) {
  // TODO: match the Python visualization. tools/infer_bevfusion.py visualizes the
  // BEVFusion input, SemVecNet input and vectors/Voronoi panels and draws
  // centerline arrows. This C++ view only shows semantic map + polylines.
  constexpr double scale = 2.4;
  constexpr int header = 30;
  constexpr int footer = 30;
  const int w = cvRound(map.cols * scale);
  const int h = cvRound(map.rows * scale);
  cv::Mat combined(h + header + footer, 2 * w, CV_8UC3, cv::Scalar(24, 24, 24));
  cv::Mat semantic = combined(cv::Rect(0, header, w, h));
  cv::Mat vectors = combined(cv::Rect(w, header, w, h));
  cv::resize(map, semantic, semantic.size(), 0, 0, cv::INTER_NEAREST);
  const auto pixel = [w, h](float forward, float left) {
    return cv::Point(cvRound(w / 2.0 - left * scale / semvecnet::kBevResolution),
                     cvRound(h / 2.0 - forward * scale / semvecnet::kBevResolution));
  };
  const cv::Scalar grid(45, 45, 45), white(255, 255, 255);
  for (int m = -30; m <= 30; m += 10) {
    const int y = pixel(m, 0).y;
    cv::line(vectors, {0, y}, {w - 1, y}, grid);
  }
  for (int m : {-10, 0, 10}) {
    const int x = pixel(0, m).x;
    cv::line(vectors, {x, 0}, {x, h - 1}, grid);
  }
  for (const auto &line : lines) {
    std::vector<cv::Point> points;
    points.reserve(line.points.size());
    for (const auto &point : line.points) {
      const auto ego = toEgo(point);
      points.push_back(pixel(ego[0], ego[1]));
    }
    const auto color = kClassBgr.at(line.label);
    cv::polylines(vectors, points, false, color, 2, cv::LINE_AA);
    cv::circle(vectors, points.back(), 3, color, cv::FILLED, cv::LINE_AA);
  }
  cv::rectangle(vectors, pixel(3.6f, 0.95f), pixel(-0.9f, -0.95f), white);
  cv::circle(vectors, pixel(0, 0), 2, white, cv::FILLED);
  const auto title = [&](const std::string &text, int x, int y) {
    cv::putText(combined, text, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv::LINE_AA);
  };
  title("SemVecNet input: semantic map", 8, 20);
  title("SemVecNet output: " + std::to_string(lines.size()) + " polylines", w + 8, 20);
  for (std::size_t i = 0; i < kClasses.size(); ++i) {
    const int x = static_cast<int>(i) * combined.cols / kClasses.size() + 8;
    cv::line(combined, {x, h + header + 15}, {x + 20, h + header + 15}, kClassBgr[i], 2);
    title(kClasses[i], x + 26, h + header + 20);
  }
  return combined;
}

void savePng(const fs::path &path, const cv::Mat &image) {
  if (!cv::imwrite(path.string(), image))
    throw std::runtime_error("Could not write " + path.string());
}

void runFrame(semvecnet::SemVecNet &pipeline, std::istream &in, const fs::path &out, std::size_t index, bool profile) {
  std::uint64_t timestamp;
  std::uint32_t count, width, height, point_count;
  read(in, &timestamp);
  read(in, &count);
  read(in, &width);
  read(in, &height);
  read(in, &point_count);
  if (count != 8 || !width || !height || width > 8192 || height > 8192 || !point_count || point_count > 2'000'000)
    throw std::runtime_error("Invalid sensor frame dimensions");

  std::array<float, 16> lidar_to_map, ego_to_map;
  read(in, lidar_to_map.data(), lidar_to_map.size());
  read(in, ego_to_map.data(), ego_to_map.size());

  std::vector<semvecnet::semantic_projection::CameraCalibration> calibrations(count);
  std::vector<cv::Mat> cameras(count);
  std::vector<const unsigned char *> images(count);
  for (std::size_t c = 0; c < count; ++c) {
    auto &calibration = calibrations[c];
    read(in, calibration.intrinsics.data(), calibration.intrinsics.size());
    read(in, calibration.lidar_to_camera.data(), calibration.lidar_to_camera.size());
    std::uint32_t size;
    read(in, &size);
    if (!size || size > 16 * 1024 * 1024)
      throw std::runtime_error("Invalid camera payload size");
    std::vector<std::uint8_t> jpeg(size);
    read(in, jpeg.data(), jpeg.size());
    const cv::Mat bgr = cv::imdecode(jpeg, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (bgr.cols != static_cast<int>(width) || bgr.rows != static_cast<int>(height))
      throw std::runtime_error("Camera image does not match CameraInfo dimensions");
    cv::cvtColor(bgr, cameras[c], cv::COLOR_BGR2RGB);
    images[c] = cameras[c].data;
  }
  static_assert(sizeof(nvtype::half) == 2);
  std::vector<nvtype::half> points(5 * point_count);
  read(in, points.data(), points.size());

  std::vector<std::uint8_t> rgb;
  const auto polylines = pipeline.forward(images.data(), points.data(), point_count, count, width, height, calibrations, lidar_to_map,
                                          ego_to_map, profile ? nullptr : &rgb);
  std::ostringstream name;
  name << std::setfill('0') << std::setw(6) << index;
  const auto prefix = name.str();
  if (!profile) {
    const cv::Mat map(semvecnet::semantic_map::kHeight, semvecnet::semantic_map::kWidth, CV_8UC3, rgb.data());
    cv::Mat bgr_map;
    cv::cvtColor(map, bgr_map, cv::COLOR_RGB2BGR);
    std::ofstream json(out / "polylines" / (prefix + ".json"));
    json.exceptions(std::ios::failbit | std::ios::badbit);
    json << std::setprecision(std::numeric_limits<float>::max_digits10) << "{\n  \"stamp_ns\": " << timestamp << ",\n"
         << "  \"pose\": [" << ego_to_map[3] << ", " << ego_to_map[7] << ", " << std::atan2(ego_to_map[4], ego_to_map[0]) << "],\n"
         << "  \"frame\": \"rear_axle (x fwd, y left, m)\",\n"
         << "  \"vectors\": [";
    const auto number = [](float value) {
      if (!std::isfinite(value))
        throw std::runtime_error("Non-finite polyline output");
      return value;
    };
    for (std::size_t i = 0; i < polylines.size(); ++i) {
      const auto &line = polylines[i];
      json << (i ? ",\n" : "\n") << "    {\"cls\": \"" << kClasses.at(line.label) << "\", \"score\": " << number(line.score)
           << ", \"pts\": [";
      for (std::size_t j = 0; j < line.points.size(); ++j) {
        const auto ego = toEgo(line.points[j]);
        json << (j ? ", " : "") << '[' << number(ego[0]) << ", " << number(ego[1]) << ']';
      }
      json << "]}";
    }
    json << "\n  ]\n}\n";
    json.close();
    savePng(out / "semantic_maps" / (prefix + ".png"), bgr_map);
    savePng(out / "combined" / (prefix + ".png"), combinedView(bgr_map, polylines));
  }
  if (index < 2 || (index + 1) % 100 == 0) {
    std::cout << prefix << ": " << count << " cameras, " << point_count << " LiDAR points";
    if (!profile) {
      const cv::Mat map(semvecnet::semantic_map::kHeight, semvecnet::semantic_map::kWidth, CV_8UC3, rgb.data());
      const auto observed = cv::countNonZero(map.reshape(1)) / 3; // Palette colors have three nonzero channels.
      std::cout << ", " << observed << " semantic pixels";
    }
    std::cout << ", " << polylines.size() << " polylines" << std::endl;
  }
}
} // namespace

int main(int argc, char **argv) {
  const bool profile = argc == 5 && std::string(argv[4]) == "--profile";
  if ((argc != 4 && !profile) || (argc == 5 && !profile)) {
    std::cerr << "Usage: semvecnet HRNET.plan SEMVECNET.onnx OUTPUT_DIR [--profile] < frames.bin\n"
              << "Use tool/eval/run_rosbag_smoke.py to read the ROS bag and run this demo.\n";
    return 2;
  }
  try {
    const fs::path out = argv[3];
    if (!profile)
      for (const char *dir : {"polylines", "semantic_maps", "combined"})
        fs::create_directories(out / dir);
    semvecnet::SemVecNet pipeline(argv[1], argv[2]);
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::size_t index = 0;
    while (std::cin.peek() != std::char_traits<char>::eof())
      runFrame(pipeline, std::cin, out, index++, profile);
    if (!index || std::cin.bad())
      throw std::runtime_error("No frames or error reading the frame stream");
    std::cout << (profile ? "Profiled " : "Wrote ") << index << (profile ? " frames" : " frame triplets") << std::endl;
  } catch (const std::exception &error) {
    std::cerr << "SemVecNet: " << error.what() << '\n';
    return 1;
  }
}
