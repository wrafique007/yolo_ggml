cmake_minimum_required(VERSION 3.14)
project(yolov7-ggml C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Where ggml comes from, in order of preference:
#   -DGGML_DIR=/path/to/ggml   use a local ggml source checkout
#   (default)                  fetch ggml from GitHub at a tested commit
# GPU backends are the usual ggml options, e.g. -DGGML_CUDA=ON, -DGGML_METAL=ON, -DGGML_VULKAN=ON
set(GGML_DIR "" CACHE PATH "Path to a ggml source tree")
set(GGML_GIT_TAG "facdc44a67da17df4433adcc468b424f48e9ded2" CACHE STRING "ggml commit to fetch when GGML_DIR is not set")

if (GGML_DIR)
    add_subdirectory(${GGML_DIR} ${CMAKE_BINARY_DIR}/ggml)
else()
    include(FetchContent)
    FetchContent_Declare(ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG        ${GGML_GIT_TAG})
    FetchContent_MakeAvailable(ggml)
endif()

add_executable(yolov7 yolov7.cpp)
target_include_directories(yolov7 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(yolov7 PRIVATE ggml)
