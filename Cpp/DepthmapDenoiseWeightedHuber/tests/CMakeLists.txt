cmake_minimum_required (VERSION 2.6)
project(test_computeG)

set(CMAKE_BUILD_TYPE Debug) 
cmake_policy(SET CMP0012 NEW)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb ")

find_package(OpenCV 2.4.9 EXACT REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_30,code=compute_30; -gencode arch=compute_35,code=compute_35;--use_fast_math ;--restrict; -lineinfo; -O3)

cuda_add_executable(${PROJECT_NAME} test_computeG.cpp computeG.cu)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
