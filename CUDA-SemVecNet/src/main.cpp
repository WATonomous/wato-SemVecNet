// Hello-world smoke test: checks the C++ build, CUDA kernel launch and TensorRT linking.
//   tool/docker_run.sh build/semvecnet
#include <exception>
#include <iostream>

#include "semvecnet/hello.hpp"

int main()
{
  try
  {
    std::cout << "Tensor " << semvecnet::tensorrt_version() << "\n";
    std::cout << "GPU     " << semvecnet::gpu_info() << "\n";

    const int n = 1000;
    const long long sum = semvecnet::gpu_hello(n);
    const long long expected = 1LL * n * (n + 1) / 2;
    std::cout << "GPU sum  " << sum << " (expected " << expected << ")\n";

    if (sum != expected)
    {
      std::cerr << "FAIL: wrong result from GPU\n";
      return 1;
    }
    std::cout << "OK\n";
    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
