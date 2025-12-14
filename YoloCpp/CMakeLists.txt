cmake_minimum_required (VERSION 3.10)

project ("YoloCpp")

set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0074 NEW) 

find_package(OpenCV REQUIRED)

find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES onnxruntime onnxruntime/core/session
)

find_library(ONNXRUNTIME_LIB 
    NAMES onnxruntime onnxruntime_providers_cuda
    PATH_SUFFIXES lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could NOT find ONNX Runtime. Check vcpkg installation.")
else()
    message(STATUS "Found ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "Found ONNX Runtime Lib: ${ONNXRUNTIME_LIB}")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

add_executable (YoloCpp "src/main.cpp" "src/YoloDetector.h" "src/YoloDetector.cpp")

target_link_libraries(YoloCpp PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})