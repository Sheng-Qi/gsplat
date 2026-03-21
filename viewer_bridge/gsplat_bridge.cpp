#include "gsplat_bridge.h"
#include <torch/torch.h>
#include <iostream>

// Include gsplat operations (native C++)

#include "Ops.h"

namespace gsplat_bridge {

int gsplat_rasterize_forward(
    const GsplatScene* scene,
    const GsplatCamera* camera,
    const GsplatRenderOptions* options,
    GsplatOutputs* outputs
) {
    try {
        int N = scene->num_gaussians;
        if (N <= 0) return 0;
        
        auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false);
        auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

        // Map pointers to LibTorch tensors without copying
        auto means = torch::from_blob((void*)scene->means, {N, 3}, tensor_opts);
        auto scales = scene->scales ? torch::from_blob((void*)scene->scales, {N, 3}, tensor_opts) : torch::Tensor();
        auto quats = scene->quats ? torch::from_blob((void*)scene->quats, {N, 4}, tensor_opts) : torch::Tensor();
        auto opacities = scene->opacities ? torch::from_blob((void*)scene->opacities, {N}, tensor_opts) : torch::Tensor();
        auto sh_coeffs = scene->sh_coeffs ? torch::from_blob((void*)scene->sh_coeffs, {N, scene->sh_degree > 0 ? (scene->sh_degree+1)*(scene->sh_degree+1) : 1, 3}, tensor_opts) : torch::Tensor();
        
        // Background color
        torch::Tensor bg_color = torch::zeros({3}, tensor_opts);
        if (options->white_bg) {
            bg_color.fill_(1.0f);
        }

        // Camera matrices
        auto viewmat = torch::from_blob((void*)camera->view_matrix, {4, 4}, tensor_opts).transpose(0, 1).contiguous();
        auto projmat = torch::from_blob((void*)camera->proj_matrix, {4, 4}, tensor_opts).transpose(0, 1).contiguous();
        auto cam_pos = torch::from_blob((void*)camera->cam_pos, {3}, tensor_opts);
        auto K = torch::tensor({
            {camera->fx, 0.0f, camera->cx},
            {0.0f, camera->fy, camera->cy},
            {0.0f, 0.0f, 1.0f}
        }, tensor_opts);

        // Optional TRBF/Motions processing could go here if implemented...
        
        // This is a minimal wrap for projection, waiting for further integration mappings matching gsplat/rendering.py
        
        return 0; // Success
    } catch (const std::exception& e) {
        std::cerr << "Gsplat rasterization failed: " << e.what() << std::endl;
        return -1;
    }
}

} // namespace gsplat_bridge
