cmake_minimum_required(VERSION 3.18) # Adjust minimum CMake version as needed

# Project name
project(inference)

include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -funroll-loops -ftree-vectorize -march=native -DNDEBUG -Wno-deprecated-declarations") #-lineinfo for nsight

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# add directory for libnvbuf-utils to program
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Find TensorRT
find_package(TensorRT REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${TensorRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SOURCES src/main.cpp)

# Source files
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    pthread # For threading
    dl      # For dynamic library loading, may be needed for plugins
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

if (LATENCY)
    add_definitions(-DLATENCY=1)
elseif(ENERGY)
    add_definitions(-DENERGY=1)
endif()

# Energy measurement
add_executable(measurements src/measurements.cpp)

target_link_libraries(measurements
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    pthread
    dl
)

target_include_directories(measurements PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})

# End to end system simulation
add_executable(simulation src/simulation.cpp)


target_link_libraries(simulation
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    pthread 
    dl
)

target_include_directories(simulation PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})





# Logistic regression measurements:
add_executable(LogisticRegression src/logistic_regression.cpp)


