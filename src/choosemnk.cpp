#include <choosemnk.h>
#include <cmath>
#include <stdexcept>

// Return (m, n, k) that fit within mem_limit_bytes
std::tuple<int, int, int> choose_mnk(size_t mem_limit_bytes, size_t dtype_size) {
  // Let's try to balance m, n, k equally
  // i.e. assume m = n = k = x
  // Then memory usage ~ (x^2 + x^2 + x^2) * dtype_size = 3 * x^2 * dtype_size
  // Solve for x
  double x = std::sqrt(mem_limit_bytes / (3.0 * dtype_size));
  if (x < 1) {
    throw std::runtime_error("Memory limit too small for even 1x1x1 tensor.");
  }

  int m = static_cast<int>(x);
  int n = static_cast<int>(x);
  int k = static_cast<int>(x);

  return {m, n, k};
}
