cmake_minimum_required(VERSION 3.10)
project(gsplat_viewer_bridge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_PREFIX_PATH "/home/shengqi/miniconda3/lib/python3.13/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20 --expt-relaxed-constexpr -O3 --forward-unknown-opts")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DAT_PARALLEL_OPENMP")

file(GLOB GSPLAT_SOURCES
    "../gsplat/cuda/csrc/*.cpp"
    "../gsplat/cuda/csrc/*.cu"
)

set(BRIDGE_SOURCES
    gsplat_bridge.cpp
)

add_library(CudaRasterizer_gsplat STATIC ${GSPLAT_SOURCES} ${BRIDGE_SOURCES})
set_target_properties(CudaRasterizer_gsplat PROPERTIES CXX_STANDARD 20 CUDA_STANDARD 20)

target_include_directories(CudaRasterizer_gsplat
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ../gsplat/cuda/csrc
    PRIVATE ../gsplat/cuda/include
    PRIVATE ../gsplat/cuda/csrc/third_party/glm
)

target_link_libraries(CudaRasterizer_gsplat PUBLIC ${TORCH_LIBRARIES})
add_library(CudaRasterizer ALIAS CudaRasterizer_gsplat)
string(REPLACE "-std=c++17" "-std=c++20" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-std=c++14" "-std=c++20" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
