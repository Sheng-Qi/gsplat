cmake_minimum_required(VERSION 3.22)

add_library(CudaRasterizer
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_stub.cpp
)

target_compile_features(CudaRasterizer PUBLIC cxx_std_17)

target_include_directories(CudaRasterizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
)

set_target_properties(CudaRasterizer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
