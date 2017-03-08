cmake_minimum_required (VERSION 2.6)
project(OpenDTAM)

set(CMAKE_BUILD_TYPE Debug) 
cmake_policy(SET CMP0012 NEW)

# ROS setup
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
rosbuild_init()
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb ")

find_package(OpenCV 2.4.9 EXACT REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system thread)

###############################################################################
macro (add_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND DTAM_SOURCES "${_relPath}/${_src}")
        else()
            list (APPEND DTAM_SOURCES "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (DTAM_SOURCES ${DTAM_SOURCES} PARENT_SCOPE)
    endif()
endmacro(add_sources)
###############################################################################

set(BASEPATH "${CMAKE_SOURCE_DIR}/Cpp")

add_subdirectory(Cpp/CostVolume)
add_subdirectory(Cpp/Track)
add_subdirectory(Cpp/Optimizer)
add_subdirectory(Cpp/DepthmapDenoiseWeightedHuber)
add_subdirectory(Cpp/utils)
add_subdirectory(Cpp/Scheduler)

include_directories("${BASEPATH}" ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Future proofing gpu code by forcing device independent PTX code to be generated, instead of gpu specific cubin code. See:
# https://devtalk.nvidia.com/default/topic/938165/when-will-cuda-8-be-released-1080-can-t-run-with-7-5/
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_30,code=compute_30; -gencode arch=compute_35,code=compute_35;--use_fast_math ;--restrict; -lineinfo; -O3)
cuda_add_library(OpenDTAM_L SHARED ${DTAM_SOURCES} Cpp/OpenDTAM.cpp)
target_link_libraries(OpenDTAM_L ${OpenCV_LIBS})

# rosbuild_add_executable(${PROJECT_NAME} Cpp/dense-mapping-orbslam-tracking.cpp Cpp/convertAhandaPovRayToStandard.cpp Cpp/graphics.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} OpenDTAM_L ${Boost_LIBRARIES})

add_executable(a.out Cpp/testmapper.cpp Cpp/convertAhandaPovRayToStandard.cpp Cpp/graphics.cpp)
target_link_libraries(a.out ${OpenCV_LIBS} OpenDTAM_L ${Boost_LIBRARIES})

