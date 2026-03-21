cmake_minimum_required(VERSION 3.22)

project(gsplat_viewer_bridge LANGUAGES CXX)

option(GSPLAT_BUILD_VIEWER_BRIDGE "Build ABI-compatible CudaRasterizer bridge for gsplat_viewer" ON)

if(GSPLAT_BUILD_VIEWER_BRIDGE)
    add_subdirectory(viewer_bridge)
endif()
