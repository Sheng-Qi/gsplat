cmake_minimum_required(VERSION 3.22)

enable_language(CUDA)

add_library(CudaRasterizer
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/backward.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/rasterizer_impl.cu
)

target_compile_features(CudaRasterizer PUBLIC cxx_std_17)

set_target_properties(CudaRasterizer PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET CudaRasterizer PROPERTY CUDA_ARCHITECTURES "75;86")
endif()

target_include_directories(CudaRasterizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
    PRIVATE
        ${CMAKE_SOURCE_DIR}/gsplat/cuda/csrc/third_party/glm
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
