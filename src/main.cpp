#include <vector>
#include <saverandomtensor.hpp>
#include <readrandomtensor.hpp>
#include <iostream>
#include <gemmplan.h>

int main() {
  // Shape: 3 x 4 x 5 x 6
  std::vector<hsize_t> shape = {3, 4, 5, 6, 7};
  saveRandomTensorHDF5("tensor.h5", "random_tensor", shape);
  
  // Load tensor back
  std::vector<hsize_t> loaded_shape;
  double* data = readTensorHDF5("tensor.h5", "random_tensor", loaded_shape);

  // Print a few values
  hsize_t totalSize = 1;
  for (auto d : loaded_shape) totalSize *= d;
  std::cout << "Loaded tensor shape: ";
  for (auto d : loaded_shape) std::cout << d << " ";
  std::cout << "\nFirst element: " << data[0]
    << "\nLast element: " << data[totalSize-1] << std::endl;
  
  // Clean up
  delete[] data;
  
  EinsumParsed parsed;
  parsed.inputs = {"ij", "jk"};
  parsed.output = "ik";
  
  try {
    GemmPlan plan = make_gemm_plan(parsed);
    print_plan(plan);
  } catch (const std::exception &e) {
    std::cerr << "Error constructing GEMM plan: " << e.what() << "\n";
    return 1;
  }

  EinsumParsed parsed2;
  parsed2.inputs = {"abc", "cde"};
  parsed2.output = "ade";
  
  try {
    GemmPlan plan = make_gemm_plan(parsed2);
    print_plan(plan);
  } catch (const std::exception &e) {
    std::cerr << "Error constructing GEMM plan: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
