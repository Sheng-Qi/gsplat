#include "gsplat_bridge.h"
#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>

#include "Ops.h"

#define CUDA_SAFE_CALL_ALWAYS(A) \
    { \
        cudaError_t error = (A); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error " << error << " at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
            exit(-1); \
        } \
    }

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
        
        // SIBR always allocates max SH coefficients (degree 3 -> 16 coefficients per channel)
        auto sh_coeffs = scene->sh_coeffs ? torch::from_blob((void*)scene->sh_coeffs, {N, 16, 3}, tensor_opts) : torch::Tensor();
        
        at::optional<at::Tensor> covars = torch::nullopt;
        if (scene->cov3D_precomp) {
            covars = torch::from_blob((void*)scene->cov3D_precomp, {N, 6}, tensor_opts);
        }

        at::optional<at::Tensor> opt_quats = quats.defined() ? at::optional<at::Tensor>(quats) : torch::nullopt;
        at::optional<at::Tensor> opt_scales = scales.defined() ? at::optional<at::Tensor>(scales) : torch::nullopt;

        // Apply scale modifier if needed
        if (options->scale_modifier != 1.0f && opt_scales.has_value()) {
            opt_scales = opt_scales.value() * options->scale_modifier;
        }

        // Apply 4D motion and opacity adjustments
        if (scene->motions && scene->trbf_centers && scene->trbf_scales) {
            auto motions = torch::from_blob((void*)scene->motions, {N, 3}, tensor_opts);
            auto trbf_centers = torch::from_blob((void*)scene->trbf_centers, {N}, tensor_opts);
            auto trbf_scales = torch::from_blob((void*)scene->trbf_scales, {N}, tensor_opts);
            
            auto dt = scene->current_time - trbf_centers;
            means = means + dt.unsqueeze(1) * motions;
            
            auto dt_scaled = dt / (trbf_scales + 1e-5f);
            auto gauss = torch::exp(-dt_scaled * dt_scaled);
            if (opacities.defined()) {
                opacities = opacities * gauss;
            }
        }
        
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

        int64_t image_width = camera->width;
        int64_t image_height = camera->height;
        int64_t tile_size = 16;
        int64_t tile_width = (image_width + tile_size - 1) / tile_size;
        int64_t tile_height = (image_height + tile_size - 1) / tile_size;

        viewmat = viewmat.unsqueeze(0); // [1, 4, 4]
        K = K.unsqueeze(0); // [1, 3, 3]

        // 1. Projection
        auto proj_res = gsplat::projection_ewa_3dgs_packed_fwd(
            means,
            torch::nullopt, // motions
            covars,
            opt_quats,
            opt_scales,
            opacities,
            viewmat,
            K,
            image_width,
            image_height,
            0.3, // eps2d
            0.01, // near_plane
            1e10, // far_plane
            0.0, // radius_clip
            false, // calc_compensations
            false, // calc_flows2d
            0 // camera_model (0 = pinhole)
        );

        auto radii = std::get<4>(proj_res);
        auto means2d = std::get<5>(proj_res);
        auto depths = std::get<6>(proj_res);
        auto conics = std::get<8>(proj_res);
        auto compensations = std::get<9>(proj_res);
        auto camera_ids = std::get<2>(proj_res);
        auto gaussian_ids = std::get<3>(proj_res);
        auto p_opacities = opacities.index({gaussian_ids});

        // Helpers for SplatMode
        auto apply_heatmap = [&](torch::Tensor value, float mod) -> torch::Tensor {
            auto norm = torch::clamp_max(value / std::max(mod, 1e-6f), 1.0f);
            auto h = torch::where(norm < 0.5f, 
                240.0f + (120.0f - 240.0f) * (norm * 2.0f), 
                120.0f + (60.0f - 120.0f) * ((norm - 0.5f) * 2.0f));

            auto c = torch::ones_like(h);
            auto x = c * (1.0f - torch::abs(torch::fmod(h / 60.0f, 2.0f) - 1.0f));

            auto r = torch::zeros_like(h);
            auto g = torch::zeros_like(h);
            auto b = torch::zeros_like(h);

            auto m1 = h < 60.0f;
            auto m2 = (h >= 60.0f) & (h < 120.0f);
            auto m3 = (h >= 120.0f) & (h < 180.0f);
            auto m4 = (h >= 180.0f) & (h < 240.0f);
            auto m5 = h >= 240.0f;

            r.masked_scatter_(m1, c.masked_select(m1)); g.masked_scatter_(m1, x.masked_select(m1));
            r.masked_scatter_(m2, x.masked_select(m2)); g.masked_scatter_(m2, c.masked_select(m2));
            g.masked_scatter_(m3, c.masked_select(m3)); b.masked_scatter_(m3, x.masked_select(m3));
            g.masked_scatter_(m4, x.masked_select(m4)); b.masked_scatter_(m4, c.masked_select(m4));
            r.masked_scatter_(m5, x.masked_select(m5)); b.masked_scatter_(m5, c.masked_select(m5));

            return torch::stack({r, g, b}, -1);
        };

        // Compute Colors via SplatMode
        torch::Tensor colors;
        if (options->splat_mode == SplatMode::COLOR) {
            if (scene->colors_precomp != nullptr) {
                auto colors_pre = torch::from_blob((void*)scene->colors_precomp, {N, 3}, tensor_opts);
                colors = colors_pre.index({gaussian_ids});
            } else if (scene->sh_degree >= 0 && scene->sh_coeffs != nullptr) {
                auto dirs = means.index({gaussian_ids}) - cam_pos.unsqueeze(0);
                dirs = torch::nn::functional::normalize(dirs, torch::nn::functional::NormalizeFuncOptions().dim(-1));
                auto shs = sh_coeffs.index({gaussian_ids}); // K, 3
                colors = gsplat::spherical_harmonics_fwd(scene->sh_degree, dirs, shs, torch::nullopt);
                colors = colors + 0.5f;
            } else {
                colors = torch::zeros({gaussian_ids.size(0), 3}, tensor_opts);
            }
        } else if (options->splat_mode == SplatMode::MOTION_HEATMAP) {
            if (scene->motions) {
                auto mot = torch::from_blob((void*)scene->motions, {N, 3}, tensor_opts).index({gaussian_ids});
                auto speed = torch::norm(mot, 2, -1);
                colors = apply_heatmap(speed, options->splat_mode_modifier);
            } else {
                colors = torch::zeros({gaussian_ids.size(0), 3}, tensor_opts);
            }
        } else if (options->splat_mode == SplatMode::T_SCALE_HEATMAP) {
            if (scene->trbf_scales) {
                auto ts = torch::from_blob((void*)scene->trbf_scales, {N}, tensor_opts).index({gaussian_ids});
                colors = apply_heatmap(ts, options->splat_mode_modifier);
            } else {
                colors = torch::zeros({gaussian_ids.size(0), 3}, tensor_opts);
            }
        } else if (options->splat_mode == SplatMode::OPACITY_HEATMAP) {
            colors = apply_heatmap(p_opacities, options->splat_mode_modifier);
        } else if (options->splat_mode == SplatMode::SCALE_HEATMAP) {
            if (scene->scales) {
                auto sc = torch::from_blob((void*)scene->scales, {N, 3}, tensor_opts).index({gaussian_ids});
                auto sl = torch::norm(sc, 2, -1);
                colors = apply_heatmap(sl, options->splat_mode_modifier);
            } else {
                colors = torch::zeros({gaussian_ids.size(0), 3}, tensor_opts);
            }
        } else if (options->splat_mode == SplatMode::FLOW_MAP) {
            if (scene->motions) {
                auto mv = torch::from_blob((void*)scene->motions, {N, 3}, tensor_opts).index({gaussian_ids});
                auto view_R = viewmat[0].slice(0, 0, 3).slice(1, 0, 3);
                auto m_view = torch::matmul(mv, view_R.t()); 
                
                auto p_time = means.index({gaussian_ids});
                auto p_time_hom = torch::cat({p_time, torch::ones({p_time.size(0), 1}, tensor_opts)}, -1);
                auto p_view = torch::matmul(p_time_hom, viewmat[0].t());
                
                auto z_inv = 1.0f / p_view.select(1, 2);
                auto z_inv2 = z_inv * z_inv;
                
                auto flow_x = camera->fx * z_inv * m_view.select(1, 0) - (camera->fx * p_view.select(1, 0) * z_inv2) * m_view.select(1, 2);
                auto flow_y = camera->fy * z_inv * m_view.select(1, 1) - (camera->fy * p_view.select(1, 1) * z_inv2) * m_view.select(1, 2);
                
                auto mag = torch::sqrt(flow_x*flow_x + flow_y*flow_y) * options->splat_mode_modifier;
                auto ang = torch::atan2(flow_y, flow_x);
                
                auto hue = ang / (2.0f * M_PI) + 0.5f;
                auto sat = torch::clamp_max(mag * options->splat_mode_modifier, 1.0f);
                
                auto h = hue * 6.0f;
                auto i = torch::floor(h).to(torch::kInt32).remainder(6);
                auto f = h - torch::floor(h);
                
                auto p_v = 1.0f - sat;
                auto q_v = 1.0f - sat * f;
                auto t_v = 1.0f - sat * (1.0f - f);
                
                auto c_z = torch::ones_like(hue);
                auto r = torch::zeros_like(h);
                auto g = torch::zeros_like(h);
                auto b = torch::zeros_like(h);
                
                auto m0 = i == 0; r.masked_scatter_(m0, c_z.masked_select(m0)); g.masked_scatter_(m0, t_v.masked_select(m0)); b.masked_scatter_(m0, p_v.masked_select(m0));
                auto mx1 = i == 1; r.masked_scatter_(mx1, q_v.masked_select(mx1)); g.masked_scatter_(mx1, c_z.masked_select(mx1)); b.masked_scatter_(mx1, p_v.masked_select(mx1));
                auto mx2 = i == 2; r.masked_scatter_(mx2, p_v.masked_select(mx2)); g.masked_scatter_(mx2, c_z.masked_select(mx2)); b.masked_scatter_(mx2, t_v.masked_select(mx2));
                auto mx3 = i == 3; r.masked_scatter_(mx3, p_v.masked_select(mx3)); g.masked_scatter_(mx3, q_v.masked_select(mx3)); b.masked_scatter_(mx3, c_z.masked_select(mx3));
                auto mx4 = i == 4; r.masked_scatter_(mx4, t_v.masked_select(mx4)); g.masked_scatter_(mx4, p_v.masked_select(mx4)); b.masked_scatter_(mx4, c_z.masked_select(mx4));
                auto mx5 = i == 5; r.masked_scatter_(mx5, c_z.masked_select(mx5)); g.masked_scatter_(mx5, p_v.masked_select(mx5)); b.masked_scatter_(mx5, q_v.masked_select(mx5));
                
                colors = torch::stack({r, g, b}, -1);
            } else {
                colors = torch::zeros({gaussian_ids.size(0), 3}, tensor_opts);
            }
        } else {
            colors = torch::zeros({gaussian_ids.size(0), 3}, tensor_opts);
        }

        // 2. Intersect tile
        auto isect_res = gsplat::intersect_tile(
            means2d,
            radii,
            depths,
            camera_ids,
            gaussian_ids,
            1, // I (num cameras)
            tile_size,
            tile_width,
            tile_height,
            true, // sort
            false // segmented
        );

        auto isect_ids = std::get<1>(isect_res);
        auto flatten_ids = std::get<2>(isect_res);

        // 3. Intersect offset
        auto isect_offsets = gsplat::intersect_offset(
            isect_ids,
            1, // I
            tile_width,
            tile_height
        );
        isect_offsets = isect_offsets.view({1, tile_height, tile_width});

        // 4. Rasterize
        auto rasterize_res = gsplat::rasterize_to_pixels_3dgs_fwd(
            means2d,
            conics,
            colors,
            p_opacities,
            bg_color,
            torch::nullopt, // masks
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids
        );

        // 5. Output
        auto render_colors = std::get<0>(rasterize_res);
        if (render_colors.dim() > 3) render_colors = render_colors.squeeze(0); // [H, W, 3]
        render_colors = render_colors.permute({2, 0, 1}).contiguous(); // [3, H, W]

        if (outputs->color) {
            CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(outputs->color, render_colors.data_ptr<float>(), 
                3 * image_height * image_width * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        
        return 0; // Success
    } catch (const std::exception& e) {
        std::cerr << "Gsplat rasterization failed: " << e.what() << std::endl;
        return -1;
    }
}

} // namespace gsplat_bridge
