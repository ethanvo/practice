#include <vector>
#include <saverandomtensor.hpp>

int main() {
  // Shape: 3 x 4 x 5 x 6
  std::vector<hsize_t> shape = {3, 4, 5, 6, 7};
  saveRandomTensorHDF5("tensor.h5", "random_tensor", shape);
  return 0;
}
