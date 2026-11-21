#pragma once

#include <cstdint>
#include <vector>
#include <NvInfer.h>

namespace nvinfer1 {
class IExecutionContext;
}

namespace semvecnet::hrnet {

// Runs one normalized FP32 [1, 3, 720, 960] RGB tensor at a time.
// Returns one row-major [720, 960] class-ID mask per camera, in input order.
std::vector<std::vector<std::int64_t>>
forward(nvinfer1::IExecutionContext &context,
        const std::vector<std::vector<float>> &images);

} // namespace semvecnet::hrnet
