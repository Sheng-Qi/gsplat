#pragma once
#include <cstdint>

// If we want a clean C/C++ interface, we should avoid C++ specific enum class inside extern "C"
// We'll just define the interface as pure C++ without Torch.

namespace gsplat_bridge {

enum class SplatMode {
    COLOR = 0,
    FLOW = 1,
    MOTION_HEATMAP = 2,
    T_SCALE_HEATMAP = 3,
    OPACITY_HEATMAP = 4,
    SCALE_HEATMAP = 5,
    FLOW_MAP = 6
};

// Gaussian Splatting scene data
struct GsplatScene {
    int num_gaussians;
    const float* means;       // [N, 3]
    const float* scales;      // [N, 3]
    const float* quats;       // [N, 4]
    const float* opacities;   // [N]
    const float* sh_coeffs;   // [N, K, 3] (where K = (sh_degree+1)^2)
    
    // For precomputed colors and covariances
    const float* colors_precomp; // [N, 3] 
    const float* cov3D_precomp;  // [N, 6] 

    // Timing and flow
    const float* motions;      // [N, 3]
    const float* trbf_centers; // [N]
    const float* trbf_scales;  // [N]
    float current_time;
    int sh_degree;
};

// Camera parameters
struct GsplatCamera {
    int width;
    int height;
    float fx;
    float fy;
    float cx;
    float cy;
    
    const float* view_matrix; // 4x4 column-major or row-major?
    const float* proj_matrix; // 4x4 column-major or row-major?
    const float* cam_pos;     // 3

    float near_plane;
    float far_plane;
};

// Outputs
struct GsplatOutputs {
    float* color; // [C, H, W] - gsplat naturally outputs this, but viewer expects [C, H, W] meaning C outer loop (like diff-gaussian)
};

struct GsplatRenderOptions {
    SplatMode splat_mode;
    float splat_mode_modifier;
    float scale_modifier;
    bool white_bg;
    bool packed;
};

// Main forward rasterization function
// Returns 0 on success, <0 on failure.
int gsplat_rasterize_forward(
    const GsplatScene* scene,
    const GsplatCamera* camera,
    const GsplatRenderOptions* options,
    GsplatOutputs* outputs
);

} // namespace gsplat_bridge
