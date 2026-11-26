#include "common/common.hpp"

#include <fstream>
#include <iostream>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
};

} // namespace

TensorRtEngine TensorRtEngine::load(const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    std::cerr << "Failed to open " << file_path << std::endl;
    return {};
  }

  const auto size = file.tellg();
  if (size <= 0) {
    std::cerr << "Empty or unreadable TensorRT plan: " << file_path << '\n';
    return {};
  }
  std::vector<char> data(static_cast<std::size_t>(size));
  if (!file.seekg(0) ||
      !file.read(data.data(), static_cast<std::streamsize>(data.size()))) {
    std::cerr << "Failed to read TensorRT plan: " << file_path << '\n';
    return {};
  }

  static Logger logger;
  TensorRtEngine loaded;
  loaded.runtime_.reset(nvinfer1::createInferRuntime(logger));
  if (!loaded.runtime_) {
    std::cerr << "Failed to create runtime" << std::endl;
    return {};
  }

  loaded.engine_.reset(
      loaded.runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!loaded.engine_) {
    std::cerr << "Failed to deserialize TensorRT plan: " << file_path << '\n';
    return {};
  }

  loaded.context_.reset(loaded.engine_->createExecutionContext());
  if (!loaded) {
    std::cerr << "Failed to create TensorRT execution context: " << file_path
              << '\n';
    return {};
  }
  return loaded;
}
