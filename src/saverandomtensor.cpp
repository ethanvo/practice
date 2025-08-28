#include <saverandomtensor.hpp>
#include <random>
#include <stdexcept>
#include <cstdlib>

using namespace H5;

// Function to create and save a random tensor
void saveRandomTensorHDF5(const std::string& filename,
                          const std::string &datasetName,
                          const std::vector<hsize_t> &shape) {
  if (shape.empty()) {
    throw std::invalid_argument("Shape cannot be empty");
  }

  // Compute total number of elements
  hsize_t totalSize = 1;
  for (auto d : shape) totalSize *= d;

  // Allocate raw array
  double* data = new double[totalSize];

  // Generate random data
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (hsize_t i = 0; i < totalSize; ++i) {
    data[i] = dist(gen);
  }

  // Create HDF5 file
  H5File file(filename, H5F_ACC_TRUNC);

  // Define dataspace with the given shape
  DataSpace dataspace(shape.size(), shape.data());

  // Create dataset of double
  DataSet dataset = file.createDataSet(datasetName, PredType::NATIVE_DOUBLE, dataspace);

  // Write data to dataset
  dataset.write(data, PredType::NATIVE_DOUBLE);

  // Free allocated memory
  delete[] data;
}
