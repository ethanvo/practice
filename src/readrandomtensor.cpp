#include <readrandomtensor.hpp>
#include <stdexcept>

using namespace H5;

double* readTensorHDF5(const std::string& filename,
                       const std::string& datasetName,
                       std::vector<hsize_t>& shape_out) {
    // Open the file
    H5File file(filename, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet(datasetName);

    // Get dataspace
    DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    shape_out.resize(rank);
    dataspace.getSimpleExtentDims(shape_out.data());

    // Compute total size
    hsize_t total_size = 1;
    for (auto d : shape_out) total_size *= d;

    // Allocate raw array
    double* data = new double[total_size];
    
    // Read into array
    dataset.read(data, PredType::NATIVE_DOUBLE);

    return data; // caller must delete[] later
}
