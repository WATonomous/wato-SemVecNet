#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <NvInfer.h>
#include <fstream>
#include <vector>

#include "semvecnet/hello.hpp"

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} gLogger;

nvinfer1::ICudaEngine *loadRuntime(const std::string &filePath)
{
  std::ifstream file(filePath, std::ios::binary);
  if (!file.good())
  {
    std::cerr << "Failed to open " << filePath << std::endl;
    return nullptr;
  }

  std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
  if (!runtime)
  {
    std::cerr << "Failed to create runtime" << std::endl;
    return nullptr;
  }

  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(data.data(), data.size());

  if (!engine)
  {
    std::cerr << "Failed to deserialize engine" << std::endl;
    delete runtime; // Clean up runtime on failure
    return nullptr;
  }

  // Note: Keep runtime alive or delete after engine creation depending on TRT version
  // For TensorRT 8.x and earlier, keep runtime alive
  // For TensorRT 10+, can delete runtime after engine is created
  // We'll keep it alive for compatibility
  return engine;
}

int main(int argc, char **argv)
{
  // std::cout << "🚀 SemVecNet is running!" << std::endl;
  // std::cout << "Tensor " << semvecnet::tensorrt_version() << std::endl;
  // std::cout << "GPU     " << semvecnet::gpu_info() << std::endl;

  // // Test GPU with a simple sum
  // const int n = 1000;
  // const long long sum = semvecnet::gpu_hello(n);
  // const long long expected = 1LL * n * (n + 1) / 2;
  // std::cout << "GPU sum  " << sum << " (expected " << expected << ")" << std::endl;

  // if (sum != expected)
  // {
  //   std::cerr << "❌ FAIL: wrong result from GPU" << std::endl;
  //   return 1;
  // }

  // std::cout << "✅ SUCCESS: Everything works!" << std::endl;

  // Test .plan loading with our simple test model
  std::string planPath = argc > 1 ? argv[1] : "models/test_simple.plan";
  std::cout << "Loading .plan file: " << planPath << std::endl;
  nvinfer1::ICudaEngine *engine = loadRuntime(planPath);
  if (!engine)
  {
    std::cerr << "❌ FAILED: Could not load engine" << std::endl;
    return 1;
  }
  std::cout << "✅ Engine loaded successfully!" << std::endl;
  std::cout << "   Device Memory: " << (engine->getDeviceMemorySizeV2() / 1024.0) << " KB" << std::endl;
  std::cout << "   I/O Tensors: " << engine->getNbIOTensors() << std::endl;
  
  // Show tensor info
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    nvinfer1::Dims dims = engine->getTensorShape(name);
    bool isInput = (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
    std::cout << "   [" << i << "] " << name << " (" << (isInput ? "INPUT" : "OUTPUT") << ") ";
    std::cout << "shape=(";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j];
      if (j < dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
  }
  
  delete engine;
  std::cout << "\n🎉 All tests passed!" << std::endl;

  return 0;
}
