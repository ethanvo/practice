#include <string>
#include <vector>
#include <H5Cpp.h>

void saveRandomTensorHDF5(const std::string &filename,
                          const std::string &datasetName,
                          const std::vector<hsize_t> &shape);
