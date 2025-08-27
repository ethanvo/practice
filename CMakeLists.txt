cmake_minimum_required(VERSION 3.12)
project(myproject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# Include path for our headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add sources
add_executable(myproject
    src/main.cpp
    src/saverandomtensor.cpp
)

# Link against HDF5
target_include_directories(myproject PRIVATE ${HDFT_INCLUDE_DIRS})
target_link_libraries(myproject PRIVATE ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
