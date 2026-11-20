// Creates a minimal TensorRT .plan file for testing
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <vector>

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << "[TensorRT] " << msg << std::endl;
  }
} gLogger;

int main() {
  std::cout << "Creating a simple test .plan file..." << std::endl;
  
  // Create builder
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
  if (!builder) {
    std::cerr << "Failed to create builder" << std::endl;
    return 1;
  }
  
  // Create network (TensorRT 11+ uses explicit batch by default)
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0U);
  if (!network) {
    std::cerr << "Failed to create network" << std::endl;
    return 1;
  }
  
  // Build a simple network: Input -> Identity -> Output
  // Input tensor: [1, 3, 224, 224] (batch, channels, height, width)
  nvinfer1::ITensor* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 3, 224, 224});
  if (!input) {
    std::cerr << "Failed to add input" << std::endl;
    return 1;
  }
  
  // Add a simple identity layer (just passes input through)
  nvinfer1::IIdentityLayer* identity = network->addIdentity(*input);
  if (!identity) {
    std::cerr << "Failed to add identity layer" << std::endl;
    return 1;
  }
  
  // Mark output
  nvinfer1::ITensor* output = identity->getOutput(0);
  output->setName("output");
  network->markOutput(*output);
  
  // Build configuration
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  if (!config) {
    std::cerr << "Failed to create builder config" << std::endl;
    return 1;
  }
  
  // Set max workspace size (512 MB)
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512ULL << 20);
  
  // Build engine
  std::cout << "Building engine (this may take a moment)..." << std::endl;
  nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
  if (!serializedModel) {
    std::cerr << "Failed to build engine" << std::endl;
    return 1;
  }
  
  // Save to file
  std::string outputPath = "models/test_simple.plan";
  std::ofstream outFile(outputPath, std::ios::binary);
  if (!outFile) {
    std::cerr << "Failed to open output file: " << outputPath << std::endl;
    return 1;
  }
  
  outFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
  outFile.close();
  
  std::cout << "✅ SUCCESS: Created " << outputPath << " (" << (serializedModel->size() / 1024.0) << " KB)" << std::endl;
  
  // Cleanup
  delete serializedModel;
  delete config;
  delete network;
  delete builder;
  
  return 0;
}
