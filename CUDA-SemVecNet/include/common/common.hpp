#pragma once

#include <NvInfer.h>
#include <memory>
#include <string>

class TensorRtEngine {
public:
  TensorRtEngine(TensorRtEngine &&) noexcept = default;
  TensorRtEngine &operator=(TensorRtEngine &&) = delete;

  // Returns an empty owner if loading or context creation fails.
  static TensorRtEngine load(const std::string &file_path);

  // Borrowed pointers. Keep this owner alive while using them.
  nvinfer1::ICudaEngine *get() const { return engine_.get(); }
  nvinfer1::IExecutionContext *context() const { return context_.get(); }
  explicit operator bool() const { return context() != nullptr; }

private:
  TensorRtEngine() = default;

  // Destruction order: context, engine, runtime.
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};
