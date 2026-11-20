#include "semvecnet/hello.hpp"

#include <NvInferRuntime.h>

namespace semvecnet {

std::string tensorrt_version() {
  // getInferLibVersion() comes from libnvinfer, so this proves TensorRT links and loads.
  const int32_t v = getInferLibVersion();  // e.g. 110400 -> 11.4.0
  return std::to_string(v / 10000) + "." + std::to_string((v / 100) % 100) + "." +
         std::to_string(v % 100);
}

}  // namespace semvecnet
