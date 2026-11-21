#pragma once

#include <NvInfer.h>
#include <string>

nvinfer1::ICudaEngine *loadRuntime(const std::string &filePath);
