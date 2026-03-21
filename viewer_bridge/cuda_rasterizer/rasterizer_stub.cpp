#include "rasterizer.h"

namespace CudaRasterizer {

void Rasterizer::markVisible(int P, float* means3D, float* viewmatrix, float* projmatrix, bool* present) {
    (void)P;
    (void)means3D;
    (void)viewmatrix;
    (void)projmatrix;
    (void)present;
}

int Rasterizer::forward(
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
    int* radii,
    bool debug,
    int* rects,
    float* boxmin,
    float* boxmax) {
    (void)geometryBuffer;
    (void)binningBuffer;
    (void)imageBuffer;
    (void)P;
    (void)D;
    (void)M;
    (void)timestamp;
    (void)trbfcenter;
    (void)trbfscale;
    (void)motion;
    (void)means3D;
    (void)opacities;
    (void)background;
    (void)width;
    (void)height;
    (void)shs;
    (void)colors_precomp;
    (void)scales;
    (void)scale_modifier;
    (void)rotations;
    (void)cov3D_precomp;
    (void)viewmatrix;
    (void)projmatrix;
    (void)cam_pos;
    (void)tan_fovx;
    (void)tan_fovy;
    (void)prefiltered;
    (void)splat_mode;
    (void)splat_mode_modifier;
    (void)out_color;
    (void)depth;
    (void)flow;
    (void)antialiasing;
    (void)radii;
    (void)debug;
    (void)rects;
    (void)boxmin;
    (void)boxmax;
    return 0;
}

void Rasterizer::backward(
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
    bool debug) {
    (void)P;
    (void)D;
    (void)M;
    (void)R;
    (void)timestamp;
    (void)trbfcenter;
    (void)trbfscale;
    (void)motion;
    (void)background;
    (void)width;
    (void)height;
    (void)means3D;
    (void)shs;
    (void)colors_precomp;
    (void)opacities;
    (void)scales;
    (void)scale_modifier;
    (void)rotations;
    (void)cov3D_precomp;
    (void)viewmatrix;
    (void)projmatrix;
    (void)campos;
    (void)tan_fovx;
    (void)tan_fovy;
    (void)radii;
    (void)geom_buffer;
    (void)binning_buffer;
    (void)image_buffer;
    (void)dL_dpix;
    (void)dL_invdepths;
    (void)dL_doutflows;
    (void)dL_dtrbfcenter;
    (void)dL_dtrbfscale;
    (void)dL_dmotion;
    (void)dL_dmean2D;
    (void)dL_dconic;
    (void)dL_dopacity;
    (void)dL_dcolor;
    (void)dL_dinvdepth;
    (void)dL_dflow;
    (void)dL_dmean3D;
    (void)dL_dcov3D;
    (void)dL_dsh;
    (void)dL_dscale;
    (void)dL_drot;
    (void)antialiasing;
    (void)debug;
}

} // namespace CudaRasterizer
