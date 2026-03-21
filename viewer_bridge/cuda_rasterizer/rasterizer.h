#ifndef GSPLAT_VIEWER_BRIDGE_RASTERIZER_H_INCLUDED
#define GSPLAT_VIEWER_BRIDGE_RASTERIZER_H_INCLUDED

#include <cstddef>
#include <functional>

namespace CudaRasterizer {

enum class SplatMode : int {
    COLOR = 0,
    MOTION_HEATMAP = 1,
    T_SCALE_HEATMAP = 2,
    OPACITY_HEATMAP = 3,
    SCALE_HEATMAP = 4,
    FLOW_MAP = 5,
};

class Rasterizer {
  public:
    static void markVisible(
        int P,
        float* means3D,
        float* viewmatrix,
        float* projmatrix,
        bool* present);

    static int forward(
        std::function<char*(size_t)> geometryBuffer,
        std::function<char*(size_t)> binningBuffer,
        std::function<char*(size_t)> imageBuffer,
        const int P,
        int D,
        int M,
        const float timestamp,
        const float* trbfcenter,
        const float* trbfscale,
        const float* motion,
        const float* means3D,
        const float* opacities,
        const float* background,
        const int width,
        int height,
        const float* shs,
        const float* colors_precomp,
        const float* scales,
        const float scale_modifier,
        const float* rotations,
        const float* cov3D_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const float* cam_pos,
        const float tan_fovx,
        float tan_fovy,
        const bool prefiltered,
        const SplatMode splat_mode,
        const float splat_mode_modifier,
        float* out_color,
        float* depth,
        float* flow,
        bool antialiasing,
        int* radii = nullptr,
        bool debug = false,
        int* rects = nullptr,
        float* boxmin = nullptr,
        float* boxmax = nullptr);

    static void backward(
        const int P,
        int D,
        int M,
        int R,
        const float timestamp,
        const float* trbfcenter,
        const float* trbfscale,
        const float* motion,
        const float* background,
        const int width,
        int height,
        const float* means3D,
        const float* shs,
        const float* colors_precomp,
        const float* opacities,
        const float* scales,
        const float scale_modifier,
        const float* rotations,
        const float* cov3D_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const float* campos,
        const float tan_fovx,
        float tan_fovy,
        const int* radii,
        char* geom_buffer,
        char* binning_buffer,
        char* image_buffer,
        const float* dL_dpix,
        const float* dL_invdepths,
        const float* dL_doutflows,
        float* dL_dtrbfcenter,
        float* dL_dtrbfscale,
        float* dL_dmotion,
        float* dL_dmean2D,
        float* dL_dconic,
        float* dL_dopacity,
        float* dL_dcolor,
        float* dL_dinvdepth,
        float* dL_dflow,
        float* dL_dmean3D,
        float* dL_dcov3D,
        float* dL_dsh,
        float* dL_dscale,
        float* dL_drot,
        bool antialiasing,
        bool debug);
};

} // namespace CudaRasterizer

#endif
