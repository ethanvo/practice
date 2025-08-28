#include <string>
#include <vector>
#include <H5Cpp.h>

double* readTensorHDF5(const std::string& filename,
                       const std::string& datasetName,
                       std::vector<hsize_t>& shape_out);
