/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

#include <cub/cub.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "ppisp_constants.h"
#include "ppisp_math.cuh"
#include "ppisp_math_bwd.cuh"

// ============================================================================
// Configuration
// ============================================================================

// Helper function to compute grid size
inline int divUp(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float ppisp_smooth_l1(float x, float beta) {
    float ax = fabsf(x);
    return ax < beta ? 0.5f * x * x / beta : ax - 0.5f * beta;
}

__device__ __forceinline__ float ppisp_smooth_l1_bwd(float x, float beta) {
    float ax = fabsf(x);
    if (ax < beta) {
        return x / beta;
    }
    return x < 0.0f ? -1.0f : 1.0f;
}

__device__ __forceinline__ float2 ppisp_apply_color_block(int block_idx,
                                                          const float2 &latent) {
    const float *m = COLOR_PINV_BLOCKS[block_idx];
    return make_float2(__fmaf_rn(m[0], latent.x, m[1] * latent.y),
                       __fmaf_rn(m[2], latent.x, m[3] * latent.y));
}

__device__ __forceinline__ void ppisp_color_offset_grad_to_latent(int block_idx,
                                                                  const float2 &grad_offset,
                                                                  float2 &grad_latent) {
    const float *m = COLOR_PINV_BLOCKS[block_idx];
    grad_latent.x = __fmaf_rn(m[0], grad_offset.x, m[2] * grad_offset.y);
    grad_latent.y = __fmaf_rn(m[1], grad_offset.x, m[3] * grad_offset.y);
}

// ============================================================================
// PPISP Forward Kernel
// ============================================================================

__global__ void ppisp_kernel(int batch_size, int num_cameras, int num_frames,
                             const float *__restrict__ exposure_params,
                             const VignettingChannelParams *__restrict__ vignetting_params,
                             const ColorPPISPParams *__restrict__ color_params,
                             const CRFPPISPChannelParams *__restrict__ crf_params,
                             const float3 *__restrict__ rgb_in, float3 *__restrict__ rgb_out,
                             const float2 *__restrict__ pixel_coords, int resolution_x,
                             int resolution_y, int camera_idx, int frame_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    // Load RGB input
    float3 rgb = rgb_in[tid];

    // ISP Pipeline - Full PPISP

    // 1. Exposure compensation
    if (frame_idx != -1) {
        apply_exposure(rgb, exposure_params[frame_idx], rgb);
    }

    // 2. Vignetting correction
    if (camera_idx != -1) {
        float2 pixel_coord;
        if (pixel_coords != nullptr) {
            pixel_coord = pixel_coords[tid];
        } else {
            pixel_coord =
                make_float2(float(tid % resolution_x) + 0.5f, float(tid / resolution_x) + 0.5f);
        }
        apply_vignetting(rgb, &vignetting_params[camera_idx * 3], pixel_coord, (float)resolution_x,
                         (float)resolution_y, rgb);
    }

    // 3. Color correction (homography)
    if (frame_idx != -1) {
        apply_color_correction_ppisp(rgb, &color_params[frame_idx], rgb);
    }

    // 4. Camera Response Function (CRF)
    if (camera_idx != -1) {
        apply_crf_ppisp(rgb, &crf_params[camera_idx * 3], rgb);
    }

    // Store output
    rgb_out[tid] = rgb;
}

// ============================================================================
// PPISP Backward Kernel
// ============================================================================

template <int BLOCK_SIZE>
__global__ void ppisp_bwd_kernel(
    int batch_size, int num_cameras, int num_frames, const float *__restrict__ exposure_params,
    const VignettingChannelParams *__restrict__ vignetting_params,
    const ColorPPISPParams *__restrict__ color_params,
    const CRFPPISPChannelParams *__restrict__ crf_params, const float3 *__restrict__ rgb_in,
    const float3 *__restrict__ rgb_out, const float3 *__restrict__ grad_rgb_out,
    float *__restrict__ grad_exposure_params,
    VignettingChannelParams *__restrict__ grad_vignetting_params,
    ColorPPISPParams *__restrict__ grad_color_params,
    CRFPPISPChannelParams *__restrict__ grad_crf_params, float3 *__restrict__ grad_rgb_in,
    const float2 *__restrict__ pixel_coords, int resolution_x, int resolution_y, int camera_idx,
    int frame_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Per-thread gradient accumulators
    float grad_exposure_local = 0.0f;
    VignettingChannelParams grad_vignetting_local[3] = {
        {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    ColorPPISPParams grad_color_local = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    CRFPPISPChannelParams grad_crf_local[3] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    if (tid < batch_size) {
        // Load input
        float3 rgb_input = rgb_in[tid];

        // Load or compute pixel coordinate if needed
        float2 pixel_coord = {0.f, 0.f};
        if (camera_idx != -1) {
            if (pixel_coords != nullptr) {
                pixel_coord = pixel_coords[tid];
            } else {
                pixel_coord =
                    make_float2(float(tid % resolution_x) + 0.5f, float(tid / resolution_x) + 0.5f);
            }
        }

        // Recompute forward pass using separate output variables to avoid aliasing
        float3 rgb = rgb_input;
        float3 rgb_after_exp = rgb;
        float3 rgb_after_vig = rgb;
        float3 rgb_after_color = rgb;

        // 1. Exposure
        if (frame_idx != -1) {
            apply_exposure(rgb, exposure_params[frame_idx], rgb_after_exp);
            rgb = rgb_after_exp;
        }

        // 2. Vignetting
        if (camera_idx != -1) {
            apply_vignetting(rgb, &vignetting_params[camera_idx * 3], pixel_coord,
                             (float)resolution_x, (float)resolution_y, rgb_after_vig);
            rgb = rgb_after_vig;
        } else {
            rgb_after_vig = rgb;
        }

        // 3. Color correction
        if (frame_idx != -1) {
            apply_color_correction_ppisp(rgb, &color_params[frame_idx], rgb_after_color);
            rgb = rgb_after_color;
        } else {
            rgb_after_color = rgb;
        }

        // Backward pass (reverse order)
        float3 grad_rgb = grad_rgb_out[tid];

        // 4. CRF backward
        if (camera_idx != -1) {
            apply_crf_ppisp_bwd(rgb_after_color, &crf_params[camera_idx * 3], grad_rgb, grad_rgb,
                                grad_crf_local);
        }

        // 3. Color correction backward
        if (frame_idx != -1) {
            apply_color_correction_ppisp_bwd(rgb_after_vig, &color_params[frame_idx], grad_rgb,
                                             grad_rgb, &grad_color_local);
        }

        // 2. Vignetting backward
        if (camera_idx != -1) {
            apply_vignetting_bwd(rgb_after_exp, &vignetting_params[camera_idx * 3], pixel_coord,
                                 (float)resolution_x, (float)resolution_y, grad_rgb, grad_rgb,
                                 grad_vignetting_local);
        }

        // 1. Exposure backward
        if (frame_idx != -1) {
            apply_exposure_bwd(rgb_input, exposure_params[frame_idx], grad_rgb, grad_rgb,
                               grad_exposure_local);
        }

        // Store RGB input gradient
        grad_rgb_in[tid] = grad_rgb;
    }  // END if (tid < batch_size)

    // Block-level reduction and atomic add for parameter gradients
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceFloat;
    typedef cub::BlockReduce<float2, BLOCK_SIZE> BlockReduceFloat2;

    if (frame_idx != -1) {
        // Exposure
        {
            __shared__ typename BlockReduceFloat::TempStorage temp;
            float val = BlockReduceFloat(temp).Sum(grad_exposure_local);
            if (threadIdx.x == 0)
                atomicAdd(&grad_exposure_params[frame_idx], val);
        }

        // Color params (4 x float2)
        {
            __shared__ typename BlockReduceFloat2::TempStorage temp;
            ColorPPISPParams *grad_color_out = &grad_color_params[frame_idx];

            float2 val_b = BlockReduceFloat2(temp).Sum(grad_color_local.b);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->b.x, val_b.x);
                atomicAdd(&grad_color_out->b.y, val_b.y);
            }

            float2 val_r = BlockReduceFloat2(temp).Sum(grad_color_local.r);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->r.x, val_r.x);
                atomicAdd(&grad_color_out->r.y, val_r.y);
            }

            float2 val_g = BlockReduceFloat2(temp).Sum(grad_color_local.g);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->g.x, val_g.x);
                atomicAdd(&grad_color_out->g.y, val_g.y);
            }

            float2 val_n = BlockReduceFloat2(temp).Sum(grad_color_local.n);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->n.x, val_n.x);
                atomicAdd(&grad_color_out->n.y, val_n.y);
            }
        }
    }

    if (camera_idx != -1) {
        // Vignetting params (3 channels x 5 params)
        {
            __shared__ typename BlockReduceFloat::TempStorage temp;
            VignettingChannelParams *grad_vig_out = &grad_vignetting_params[camera_idx * 3];

#pragma unroll
            for (int ch = 0; ch < 3; ch++) {
                float val_cx = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].cx);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].cx, val_cx);

                float val_cy = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].cy);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].cy, val_cy);

                float val_a0 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha0);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].alpha0, val_a0);

                float val_a1 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha1);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].alpha1, val_a1);

                float val_a2 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha2);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].alpha2, val_a2);
            }
        }

        // CRF params (3 channels x 4 params)
        {
            __shared__ typename BlockReduceFloat::TempStorage temp;
            CRFPPISPChannelParams *grad_crf_out = &grad_crf_params[camera_idx * 3];

#pragma unroll
            for (int ch = 0; ch < 3; ch++) {
                float val_toe = BlockReduceFloat(temp).Sum(grad_crf_local[ch].toe);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].toe, val_toe);

                float val_shoulder = BlockReduceFloat(temp).Sum(grad_crf_local[ch].shoulder);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].shoulder, val_shoulder);

                float val_gamma = BlockReduceFloat(temp).Sum(grad_crf_local[ch].gamma);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].gamma, val_gamma);

                float val_center = BlockReduceFloat(temp).Sum(grad_crf_local[ch].center);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].center, val_center);
            }
        }
    }
}

// ============================================================================
// Forward Pass Implementation
// ============================================================================

void ppisp_forward(const float *exposure_params, const float *vignetting_params,
                   const float *color_params, const float *crf_params, const float *rgb_in,
                   float *rgb_out, const float *pixel_coords, int num_pixels, int num_cameras,
                   int num_frames, int resolution_w, int resolution_h, int camera_idx,
                   int frame_idx) {
    if (num_pixels == 0) return;
    const int threads = PPISP_BLOCK_SIZE;
    const int blocks = divUp(num_pixels, threads);
    const auto stream = at::cuda::getCurrentCUDAStream();

    ppisp_kernel<<<blocks, threads, 0, stream>>>(
        num_pixels, num_cameras, num_frames, exposure_params,
        reinterpret_cast<const VignettingChannelParams *>(vignetting_params),
        reinterpret_cast<const ColorPPISPParams *>(color_params),
        reinterpret_cast<const CRFPPISPChannelParams *>(crf_params),
        reinterpret_cast<const float3 *>(rgb_in), reinterpret_cast<float3 *>(rgb_out),
        reinterpret_cast<const float2 *>(pixel_coords), resolution_w, resolution_h, camera_idx,
        frame_idx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Backward Pass Implementation
// ============================================================================

void ppisp_backward(const float *exposure_params, const float *vignetting_params,
                    const float *color_params, const float *crf_params, const float *rgb_in,
                    const float *rgb_out, const float *pixel_coords, const float *v_rgb_out,
                    float *v_exposure_params, float *v_vignetting_params, float *v_color_params,
                    float *v_crf_params, float *v_rgb_in, int num_pixels, int num_cameras,
                    int num_frames, int resolution_w, int resolution_h, int camera_idx,
                    int frame_idx) {
    if (num_pixels == 0) return;
    const int threads = PPISP_BLOCK_SIZE;
    const int blocks = divUp(num_pixels, threads);
    const auto stream = at::cuda::getCurrentCUDAStream();

    ppisp_bwd_kernel<PPISP_BLOCK_SIZE><<<blocks, threads, 0, stream>>>(
        num_pixels, num_cameras, num_frames, exposure_params,
        reinterpret_cast<const VignettingChannelParams *>(vignetting_params),
        reinterpret_cast<const ColorPPISPParams *>(color_params),
        reinterpret_cast<const CRFPPISPChannelParams *>(crf_params),
        reinterpret_cast<const float3 *>(rgb_in), reinterpret_cast<const float3 *>(rgb_out),
        reinterpret_cast<const float3 *>(v_rgb_out), v_exposure_params,
        reinterpret_cast<VignettingChannelParams *>(v_vignetting_params),
        reinterpret_cast<ColorPPISPParams *>(v_color_params),
        reinterpret_cast<CRFPPISPChannelParams *>(v_crf_params),
        reinterpret_cast<float3 *>(v_rgb_in), reinterpret_cast<const float2 *>(pixel_coords),
        resolution_w, resolution_h, camera_idx, frame_idx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// PPISP Regularization Loss Kernels
// ============================================================================

// Frame-mean loss group: exposure_mean and color_mean. These terms need
// cross-frame sums before the final mean loss can be computed.
// frame_mean_sums shape: [PPISP_FRAME_MEAN_SUMS_SIZE].
// frame_mean_sums[0] stores sum(exposure_params); frame_mean_sums[1 + i]
// stores the summed color offset component for i in [0, PPISP_COLOR_PARAMS).
template <int BLOCK_SIZE>
__global__ void ppisp_regularization_frame_mean_sums_kernel(
    const float *__restrict__ exposure_params,
    const ColorPPISPParams *__restrict__ color_params, float *__restrict__ frame_mean_sums,
    int num_frames, bool compute_exposure_stats, bool compute_color_stats) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float exposure_sum = 0.0f;
    float color_sums[PPISP_COLOR_PARAMS];

#pragma unroll
    for (int i = 0; i < PPISP_COLOR_PARAMS; i++) {
        color_sums[i] = 0.0f;
    }

    for (int frame = tid; frame < num_frames; frame += stride) {
        if (compute_exposure_stats) {
            exposure_sum += exposure_params[frame];
        }

        if (compute_color_stats) {
            const ColorPPISPParams &params = color_params[frame];
            float2 offsets[4] = {
                ppisp_apply_color_block(0, params.b),
                ppisp_apply_color_block(1, params.r),
                ppisp_apply_color_block(2, params.g),
                ppisp_apply_color_block(3, params.n),
            };

#pragma unroll
            for (int block = 0; block < 4; block++) {
                color_sums[block * 2] += offsets[block].x;
                color_sums[block * 2 + 1] += offsets[block].y;
            }
        }
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceFloat;
    __shared__ typename BlockReduceFloat::TempStorage temp;

    if (compute_exposure_stats) {
        float block_exposure_sum = BlockReduceFloat(temp).Sum(exposure_sum);
        if (threadIdx.x == 0) {
            atomicAdd(&frame_mean_sums[0], block_exposure_sum);
        }
    }

    __syncthreads();

    if (compute_color_stats) {
#pragma unroll
        for (int i = 0; i < PPISP_COLOR_PARAMS; i++) {
            float block_color_sum = BlockReduceFloat(temp).Sum(color_sums[i]);
            if (threadIdx.x == 0) {
                atomicAdd(&frame_mean_sums[1 + i], block_color_sum);
            }
            __syncthreads();
        }
    }
}

__global__ void ppisp_regularization_frame_mean_loss_kernel(float *__restrict__ loss,
                                                            const float *__restrict__ frame_mean_sums,
                                                            int num_frames,
                                                            float exposure_mean_weight,
                                                            float color_mean_weight) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || num_frames <= 0) {
        return;
    }

    float inv_frames = 1.0f / static_cast<float>(num_frames);

    // Exposure mean regularization (fix SH <-> exposure ambiguity)
    if (exposure_mean_weight > 0.0f) {
        float exposure_residual = frame_mean_sums[0] * inv_frames;
        atomicAdd(loss, exposure_mean_weight * ppisp_smooth_l1(exposure_residual, 0.1f));
    }

    // Color mean regularization using ZCA block-diagonal matrix
    if (color_mean_weight > 0.0f) {
        float color_loss = 0.0f;
#pragma unroll
        for (int i = 0; i < PPISP_COLOR_PARAMS; i++) {
            float color_residual = frame_mean_sums[1 + i] * inv_frames;
            color_loss += ppisp_smooth_l1(color_residual, 0.005f);
        }
        atomicAdd(loss, color_mean_weight * color_loss / static_cast<float>(PPISP_COLOR_PARAMS));
    }
}

// Camera-parameter loss group: vig_center, vig_non_pos, vig_channel, and
// crf_channel. These terms reduce directly over per-camera parameter tensors.
template <int BLOCK_SIZE>
__global__ void ppisp_regularization_camera_param_loss_kernel(
    const float *__restrict__ vignetting_params, const float *__restrict__ crf_params,
    float *__restrict__ loss, int num_cameras, float vig_center_weight,
    float vig_channel_weight, float vig_non_pos_weight, float crf_channel_weight) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_vig = num_cameras * 3 * PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
    int total_vig_channel = num_cameras * PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
    int total_crf_channel = num_cameras * PPISP_CRF_PARAMS_PER_CHANNEL;
    int total = total_vig > total_vig_channel ? total_vig : total_vig_channel;
    total = total > total_crf_channel ? total : total_crf_channel;

    float inv_vig_center_denom = 1.0f / static_cast<float>(num_cameras * 3);
    float inv_vig_non_pos_denom = 1.0f / static_cast<float>(num_cameras * 3 * 3);
    float inv_vig_channel_denom =
        1.0f / static_cast<float>(num_cameras * PPISP_VIGNETTING_PARAMS_PER_CHANNEL * 3);
    float inv_crf_channel_denom =
        1.0f / static_cast<float>(num_cameras * PPISP_CRF_PARAMS_PER_CHANNEL * 3);

    float vig_center_part = 0.0f;
    float vig_non_pos_part = 0.0f;
    float vig_channel_part = 0.0f;
    float crf_channel_part = 0.0f;

    for (int idx = tid; idx < total; idx += stride) {
        if (idx < total_vig) {
            int param_idx = idx % PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
            float val = vignetting_params[idx];

            // Vignetting center loss: optical center should be near image center (0, 0)
            if (vig_center_weight > 0.0f && param_idx < 2) {
                vig_center_part += vig_center_weight * val * val * inv_vig_center_denom;
            }

            // Vignetting non-positivity loss: alpha coefficients should be <= 0
            if (vig_non_pos_weight > 0.0f && param_idx >= 2 && val > 0.0f) {
                vig_non_pos_part += vig_non_pos_weight * val * inv_vig_non_pos_denom;
            }
        }

        // Vignetting channel variance
        if (idx < total_vig_channel && vig_channel_weight > 0.0f) {
            int param_idx = idx % PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
            int camera_idx = idx / PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
            int base = camera_idx * 3 * PPISP_VIGNETTING_PARAMS_PER_CHANNEL + param_idx;
            float v0 = vignetting_params[base];
            float v1 = vignetting_params[base + PPISP_VIGNETTING_PARAMS_PER_CHANNEL];
            float v2 = vignetting_params[base + 2 * PPISP_VIGNETTING_PARAMS_PER_CHANNEL];
            float mean = (v0 + v1 + v2) / 3.0f;
            float d0 = v0 - mean;
            float d1 = v1 - mean;
            float d2 = v2 - mean;
            vig_channel_part +=
                vig_channel_weight * (d0 * d0 + d1 * d1 + d2 * d2) * inv_vig_channel_denom;
        }

        // CRF channel variance
        if (idx < total_crf_channel && crf_channel_weight > 0.0f) {
            int param_idx = idx % PPISP_CRF_PARAMS_PER_CHANNEL;
            int camera_idx = idx / PPISP_CRF_PARAMS_PER_CHANNEL;
            int base = camera_idx * 3 * PPISP_CRF_PARAMS_PER_CHANNEL + param_idx;
            float v0 = crf_params[base];
            float v1 = crf_params[base + PPISP_CRF_PARAMS_PER_CHANNEL];
            float v2 = crf_params[base + 2 * PPISP_CRF_PARAMS_PER_CHANNEL];
            float mean = (v0 + v1 + v2) / 3.0f;
            float d0 = v0 - mean;
            float d1 = v1 - mean;
            float d2 = v2 - mean;
            crf_channel_part +=
                crf_channel_weight * (d0 * d0 + d1 * d1 + d2 * d2) * inv_crf_channel_denom;
        }
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceFloat;
    __shared__ typename BlockReduceFloat::TempStorage temp;

    if (vig_center_weight > 0.0f) {
        float block_part = BlockReduceFloat(temp).Sum(vig_center_part);
        if (threadIdx.x == 0) {
            atomicAdd(loss, block_part);
        }
    }

    __syncthreads();

    if (vig_non_pos_weight > 0.0f) {
        float block_part = BlockReduceFloat(temp).Sum(vig_non_pos_part);
        if (threadIdx.x == 0) {
            atomicAdd(loss, block_part);
        }
    }

    __syncthreads();

    if (vig_channel_weight > 0.0f) {
        float block_part = BlockReduceFloat(temp).Sum(vig_channel_part);
        if (threadIdx.x == 0) {
            atomicAdd(loss, block_part);
        }
    }

    __syncthreads();

    if (crf_channel_weight > 0.0f) {
        float block_part = BlockReduceFloat(temp).Sum(crf_channel_part);
        if (threadIdx.x == 0) {
            atomicAdd(loss, block_part);
        }
    }
}

__global__ void ppisp_regularization_frame_mean_backward_kernel(
    const float *__restrict__ frame_mean_sums, const float *__restrict__ grad_loss,
    float *__restrict__ grad_exposure_params, float *__restrict__ grad_color_params,
    int num_frames, float exposure_mean_weight, float color_mean_weight) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (num_frames <= 0) {
        return;
    }

    float inv_frames = 1.0f / static_cast<float>(num_frames);
    float upstream = grad_loss[0];
    float grad_exposure = 0.0f;
    float grad_offsets[PPISP_COLOR_PARAMS];

    if (exposure_mean_weight > 0.0f) {
        float exposure_residual = frame_mean_sums[0] * inv_frames;
        grad_exposure = upstream * exposure_mean_weight *
                        ppisp_smooth_l1_bwd(exposure_residual, 0.1f) * inv_frames;
    }

#pragma unroll
    for (int i = 0; i < PPISP_COLOR_PARAMS; i++) {
        grad_offsets[i] = 0.0f;
        if (color_mean_weight > 0.0f) {
            float color_residual = frame_mean_sums[1 + i] * inv_frames;
            grad_offsets[i] =
                upstream * color_mean_weight * ppisp_smooth_l1_bwd(color_residual, 0.005f) *
                inv_frames / static_cast<float>(PPISP_COLOR_PARAMS);
        }
    }

    for (int frame = tid; frame < num_frames; frame += stride) {
        if (exposure_mean_weight > 0.0f) {
            grad_exposure_params[frame] += grad_exposure;
        }

        if (color_mean_weight > 0.0f) {
            int base = frame * PPISP_COLOR_PARAMS;
#pragma unroll
            for (int block = 0; block < 4; block++) {
                float2 grad_offset =
                    make_float2(grad_offsets[block * 2], grad_offsets[block * 2 + 1]);
                float2 grad_latent;
                ppisp_color_offset_grad_to_latent(block, grad_offset, grad_latent);
                grad_color_params[base + block * 2] += grad_latent.x;
                grad_color_params[base + block * 2 + 1] += grad_latent.y;
            }
        }
    }
}

__global__ void ppisp_regularization_vignetting_backward_kernel(
    const float *__restrict__ vignetting_params, const float *__restrict__ grad_loss,
    float *__restrict__ grad_vignetting_params, int num_cameras, float vig_center_weight,
    float vig_channel_weight, float vig_non_pos_weight) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_vig = num_cameras * PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
    float upstream = grad_loss[0];
    float inv_vig_center_denom = 1.0f / static_cast<float>(num_cameras * 3);
    float inv_vig_non_pos_denom = 1.0f / static_cast<float>(num_cameras * 3 * 3);
    float inv_vig_channel_denom =
        1.0f / static_cast<float>(num_cameras * PPISP_VIGNETTING_PARAMS_PER_CHANNEL);

    for (int idx = tid; idx < total_vig; idx += stride) {
        int param_idx = idx % PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
        int camera_idx = idx / PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
        int base = camera_idx * 3 * PPISP_VIGNETTING_PARAMS_PER_CHANNEL + param_idx;
        float v0 = vignetting_params[base];
        float v1 = vignetting_params[base + PPISP_VIGNETTING_PARAMS_PER_CHANNEL];
        float v2 = vignetting_params[base + 2 * PPISP_VIGNETTING_PARAMS_PER_CHANNEL];
        float mean = (v0 + v1 + v2) / 3.0f;
        float values[3] = {v0, v1, v2};

#pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            float val = values[channel];
            float grad = 0.0f;

            if (vig_center_weight > 0.0f && param_idx < 2) {
                grad += vig_center_weight * 2.0f * val * inv_vig_center_denom;
            }

            if (vig_non_pos_weight > 0.0f && param_idx >= 2 && val > 0.0f) {
                grad += vig_non_pos_weight * inv_vig_non_pos_denom;
            }

            if (vig_channel_weight > 0.0f) {
                grad += vig_channel_weight * (2.0f / 3.0f) * (val - mean) *
                        inv_vig_channel_denom;
            }

            grad_vignetting_params[base + channel * PPISP_VIGNETTING_PARAMS_PER_CHANNEL] +=
                upstream * grad;
        }
    }
}

__global__ void ppisp_regularization_crf_backward_kernel(
    const float *__restrict__ crf_params, const float *__restrict__ grad_loss,
    float *__restrict__ grad_crf_params, int num_cameras, float crf_channel_weight) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_crf = num_cameras * PPISP_CRF_PARAMS_PER_CHANNEL;
    float upstream = grad_loss[0];
    float inv_crf_channel_denom =
        1.0f / static_cast<float>(num_cameras * PPISP_CRF_PARAMS_PER_CHANNEL);

    for (int idx = tid; idx < total_crf; idx += stride) {
        int param_idx = idx % PPISP_CRF_PARAMS_PER_CHANNEL;
        int camera_idx = idx / PPISP_CRF_PARAMS_PER_CHANNEL;
        int base = camera_idx * 3 * PPISP_CRF_PARAMS_PER_CHANNEL + param_idx;
        float v0 = crf_params[base];
        float v1 = crf_params[base + PPISP_CRF_PARAMS_PER_CHANNEL];
        float v2 = crf_params[base + 2 * PPISP_CRF_PARAMS_PER_CHANNEL];
        float mean = (v0 + v1 + v2) / 3.0f;
        float values[3] = {v0, v1, v2};

#pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            float grad = crf_channel_weight * (2.0f / 3.0f) * (values[channel] - mean) *
                         inv_crf_channel_denom;
            grad_crf_params[base + channel * PPISP_CRF_PARAMS_PER_CHANNEL] += upstream * grad;
        }
    }
}

// ============================================================================
// Regularization Loss Implementation
// ============================================================================

// Inputs:
// - exposure_params: [num_frames]
// - vignetting_params: [num_cameras, 3, PPISP_VIGNETTING_PARAMS_PER_CHANNEL]
// - color_params: [num_frames, PPISP_COLOR_PARAMS]
// - crf_params: [num_cameras, 3, PPISP_CRF_PARAMS_PER_CHANNEL]
// Outputs, expected zero-initialized:
// - loss_out: scalar total weighted regularization loss
// - frame_mean_sums: [PPISP_FRAME_MEAN_SUMS_SIZE], saved for backward
// frame_mean_sums layout:
// [sum(exposure_params), sum(color_offset_0), ..., sum(color_offset_7)].
void ppisp_regularization_forward(
    const float *exposure_params, const float *vignetting_params, const float *color_params,
    const float *crf_params, float *loss_out, float *frame_mean_sums, int num_cameras,
    int num_frames, float exposure_mean_weight, float vig_center_weight,
    float vig_channel_weight, float vig_non_pos_weight, float color_mean_weight,
    float crf_channel_weight) {
    const int threads = PPISP_BLOCK_SIZE;
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (num_frames > 0 && (exposure_mean_weight > 0.0f || color_mean_weight > 0.0f)) {
        int blocks = divUp(num_frames, threads);
        ppisp_regularization_frame_mean_sums_kernel<PPISP_BLOCK_SIZE><<<blocks, threads, 0, stream>>>(
            exposure_params, reinterpret_cast<const ColorPPISPParams *>(color_params),
            frame_mean_sums,
            num_frames, exposure_mean_weight > 0.0f, color_mean_weight > 0.0f);
        ppisp_regularization_frame_mean_loss_kernel<<<1, 1, 0, stream>>>(
            loss_out, frame_mean_sums, num_frames, exposure_mean_weight, color_mean_weight);
    }

    if (num_cameras > 0 &&
        (vig_center_weight > 0.0f || vig_channel_weight > 0.0f ||
         vig_non_pos_weight > 0.0f || crf_channel_weight > 0.0f)) {
        int total_vig = num_cameras * 3 * PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
        int total_vig_channel = num_cameras * PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
        int total_crf_channel = num_cameras * PPISP_CRF_PARAMS_PER_CHANNEL;
        int blocks = divUp(std::max(total_vig, std::max(total_vig_channel, total_crf_channel)),
                           threads);
        ppisp_regularization_camera_param_loss_kernel<PPISP_BLOCK_SIZE><<<blocks, threads, 0, stream>>>(
            vignetting_params, crf_params, loss_out, num_cameras, vig_center_weight,
            vig_channel_weight, vig_non_pos_weight, crf_channel_weight);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Inputs:
// - exposure_params: [num_frames]
// - vignetting_params: [num_cameras, 3, PPISP_VIGNETTING_PARAMS_PER_CHANNEL]
// - color_params: [num_frames, PPISP_COLOR_PARAMS]
// - crf_params: [num_cameras, 3, PPISP_CRF_PARAMS_PER_CHANNEL]
// - grad_loss: scalar upstream gradient
// - frame_mean_sums: [PPISP_FRAME_MEAN_SUMS_SIZE] output from forward
// Outputs, expected zero-initialized:
// - grad_exposure_params: [num_frames]
// - grad_vignetting_params: [num_cameras, 3, PPISP_VIGNETTING_PARAMS_PER_CHANNEL]
// - grad_color_params: [num_frames, PPISP_COLOR_PARAMS]
// - grad_crf_params: [num_cameras, 3, PPISP_CRF_PARAMS_PER_CHANNEL]
// frame_mean_sums layout:
// [sum(exposure_params), sum(color_offset_0), ..., sum(color_offset_7)].
void ppisp_regularization_backward(
    const float *exposure_params, const float *vignetting_params, const float *color_params,
    const float *crf_params, const float *grad_loss, float *grad_exposure_params,
    float *grad_vignetting_params, float *grad_color_params, float *grad_crf_params,
    float *frame_mean_sums, int num_cameras, int num_frames, float exposure_mean_weight,
    float vig_center_weight, float vig_channel_weight, float vig_non_pos_weight,
    float color_mean_weight, float crf_channel_weight) {
    const int threads = PPISP_BLOCK_SIZE;
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (num_frames > 0 && (exposure_mean_weight > 0.0f || color_mean_weight > 0.0f)) {
        int blocks = divUp(num_frames, threads);
        ppisp_regularization_frame_mean_backward_kernel<<<blocks, threads, 0, stream>>>(
            frame_mean_sums, grad_loss, grad_exposure_params, grad_color_params, num_frames,
            exposure_mean_weight, color_mean_weight);
    }

    if (num_cameras > 0 &&
        (vig_center_weight > 0.0f || vig_channel_weight > 0.0f || vig_non_pos_weight > 0.0f)) {
        int total_vig = num_cameras * PPISP_VIGNETTING_PARAMS_PER_CHANNEL;
        int blocks = divUp(total_vig, threads);
        ppisp_regularization_vignetting_backward_kernel<<<blocks, threads, 0, stream>>>(
            vignetting_params, grad_loss, grad_vignetting_params, num_cameras, vig_center_weight,
            vig_channel_weight, vig_non_pos_weight);
    }

    if (num_cameras > 0 && crf_channel_weight > 0.0f) {
        int total_crf = num_cameras * PPISP_CRF_PARAMS_PER_CHANNEL;
        int blocks = divUp(total_crf, threads);
        ppisp_regularization_crf_backward_kernel<<<blocks, threads, 0, stream>>>(
            crf_params, grad_loss, grad_crf_params, num_cameras, crf_channel_weight);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
