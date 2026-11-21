#include "common/common.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
};

Logger gLogger;

} // namespace

nvinfer1::ICudaEngine *loadRuntime(const std::string &filePath) {
  std::ifstream file(filePath, std::ios::binary);
  if (!file.good()) {
    std::cerr << "Failed to open " << filePath << std::endl;
    return nullptr;
  }

  std::vector<char> data((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  file.close();

  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
  if (!runtime) {
    std::cerr << "Failed to create runtime" << std::endl;
    return nullptr;
  }

  nvinfer1::ICudaEngine *engine =
      runtime->deserializeCudaEngine(data.data(), data.size());
  if (!engine) {
    std::cerr << "Failed to deserialize engine" << std::endl;
    delete runtime;
    return nullptr;
  }

  // The runtime must outlive the engine. The current raw-pointer API does
  // not give the caller a way to release the runtime after destroying it.
  return engine;
}
