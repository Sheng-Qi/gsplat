cmake_minimum_required(VERSION 3.10)
project(gsplat_viewer_bridge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH "/home/shengqi/miniconda3/lib/python3.13/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -O3 --forward-unknown-opts")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DAT_PARALLEL_OPENMP")

file(GLOB GSPLAT_SOURCES
    "../gsplat/cuda/csrc/*.cpp"
    "../gsplat/cuda/csrc/*.cu"
)

set(BRIDGE_SOURCES
    gsplat_bridge.cpp
)

add_library(CudaRasterizer STATIC ${GSPLAT_SOURCES} ${BRIDGE_SOURCES})
set_target_properties(CudaRasterizer PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20)

# Torch injects -std=c++17 through usage requirements in some environments.
# Append explicit standard flags here so nvcc/g++ end on C++20.
set_property(TARGET CudaRasterizer APPEND_STRING PROPERTY COMPILE_FLAGS " -std=c++20")
set_property(TARGET CudaRasterizer APPEND_STRING PROPERTY CUDA_NVCC_FLAGS " --std=c++20")

target_include_directories(CudaRasterizer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ../gsplat/cuda/csrc
    PRIVATE ../gsplat/cuda/include
    PRIVATE ../gsplat/cuda/csrc/third_party/glm
)

target_link_libraries(CudaRasterizer PUBLIC ${TORCH_LIBRARIES})
