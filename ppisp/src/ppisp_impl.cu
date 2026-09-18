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
#include <array>
#include <atomic>

#include <cub/cub.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAMacros.h>

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

// Each thread accumulates parameter gradients over several pixels, then the
// block reduces all accumulators at once: warp shuffles, one shared-memory
// exchange, one barrier, and one global atomic per parameter per block.
// Register-cap request for __launch_bounds__: three resident blocks per SM
// measured 3% faster than the unconstrained build on sm_89 and sm_90. The grid
// cap is not derived from it; ppisp_bwd_resident_blocks asks the occupancy API.
constexpr int PPISP_BWD_MIN_BLOCKS_PER_SM = 3;
constexpr int PPISP_BWD_NUM_COLOR = PPISP_COLOR_PARAMS;                             // 8
constexpr int PPISP_BWD_NUM_VIG = 3 * PPISP_VIGNETTING_PARAMS_PER_CHANNEL;          // 15
constexpr int PPISP_BWD_NUM_CRF = 3 * PPISP_CRF_PARAMS_PER_CHANNEL;                 // 12
constexpr int PPISP_BWD_NUM_ACCUM = 1 + PPISP_BWD_NUM_COLOR + PPISP_BWD_NUM_VIG + PPISP_BWD_NUM_CRF;  // 36
// The block reduction views each gradient struct as a flat float array on both
// the accumulator and the output side, so the structs must be packed floats.
static_assert(sizeof(ColorPPISPParams) == PPISP_BWD_NUM_COLOR * sizeof(float), "layout");
static_assert(sizeof(VignettingChannelParams) == PPISP_VIGNETTING_PARAMS_PER_CHANNEL * sizeof(float), "layout");
static_assert(sizeof(CRFPPISPChannelParams) == PPISP_CRF_PARAMS_PER_CHANNEL * sizeof(float), "layout");

__device__ __forceinline__ float ppisp_warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// On sm_89 the register cap lands at 80 registers, which spills a little but
// measured faster than the unconstrained 109 at 960x540, 1920x1080, and 3840x2160.
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE, PPISP_BWD_MIN_BLOCKS_PER_SM) ppisp_bwd_kernel(
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
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    static_assert(BLOCK_SIZE % 32 == 0 && BLOCK_SIZE >= PPISP_BWD_NUM_ACCUM, "block size");

    // Per-thread gradient accumulators
    float grad_exposure_local = 0.0f;
    VignettingChannelParams grad_vignetting_local[3] = {
        {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    ColorPPISPParams grad_color_local = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    CRFPPISPChannelParams grad_crf_local[3] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < batch_size;
         tid += blockDim.x * gridDim.x) {
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
            float grad_exposure_pixel;
            apply_exposure_bwd(rgb_input, exposure_params[frame_idx], grad_rgb, grad_rgb,
                               grad_exposure_pixel);
            grad_exposure_local += grad_exposure_pixel;
        }

        // Store RGB input gradient
        grad_rgb_in[tid] = grad_rgb;
    }

    // Pack accumulators: [exposure | color(8) | vignetting(15) | crf(12)], viewing
    // the local structs as flat floats exactly as the output side does below.
    float acc[PPISP_BWD_NUM_ACCUM];
    acc[0] = grad_exposure_local;
    {
        const float *color = reinterpret_cast<const float *>(&grad_color_local);
        const float *vig = reinterpret_cast<const float *>(grad_vignetting_local);
        const float *crf = reinterpret_cast<const float *>(grad_crf_local);
#pragma unroll
        for (int k = 0; k < PPISP_BWD_NUM_COLOR; k++) acc[1 + k] = color[k];
#pragma unroll
        for (int k = 0; k < PPISP_BWD_NUM_VIG; k++) acc[1 + PPISP_BWD_NUM_COLOR + k] = vig[k];
#pragma unroll
        for (int k = 0; k < PPISP_BWD_NUM_CRF; k++)
            acc[1 + PPISP_BWD_NUM_COLOR + PPISP_BWD_NUM_VIG + k] = crf[k];
    }

    // Block reduction: warp shuffle, then one shared-memory exchange. The frame
    // slots are all zero when frame_idx == -1 and the camera slots when
    // camera_idx == -1, so those slots are skipped (uniform per kernel argument).
    const bool frame_active = frame_idx != -1;
    const bool camera_active = camera_idx != -1;
    __shared__ float warp_sums[NUM_WARPS][PPISP_BWD_NUM_ACCUM];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < PPISP_BWD_NUM_ACCUM; i++) {
        if (i < 1 + PPISP_BWD_NUM_COLOR ? frame_active : camera_active) {
            float v = ppisp_warp_sum(acc[i]);
            if (lane == 0) warp_sums[warp][i] = v;
        }
    }
    __syncthreads();

    // One thread per parameter: sum warps and add to the global gradient
    const int i = threadIdx.x;
    if (i < PPISP_BWD_NUM_ACCUM && (i < 1 + PPISP_BWD_NUM_COLOR ? frame_active : camera_active)) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) total += warp_sums[w][i];

        float *dst;
        if (i == 0) {
            dst = &grad_exposure_params[frame_idx];
        } else if (i < 1 + PPISP_BWD_NUM_COLOR) {
            dst = reinterpret_cast<float *>(&grad_color_params[frame_idx]) + (i - 1);
        } else if (i < 1 + PPISP_BWD_NUM_COLOR + PPISP_BWD_NUM_VIG) {
            dst = reinterpret_cast<float *>(&grad_vignetting_params[camera_idx * 3]) +
                  (i - 1 - PPISP_BWD_NUM_COLOR);
        } else {
            dst = reinterpret_cast<float *>(&grad_crf_params[camera_idx * 3]) +
                  (i - 1 - PPISP_BWD_NUM_COLOR - PPISP_BWD_NUM_VIG);
        }
        atomicAdd(dst, total);
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

// Blocks of the backward kernel that the current device holds at once, from
// the occupancy API for the compiled kernel, so the grid cap follows the real
// register and shared-memory footprint on every architecture. Cached per
// device; the launch path does one relaxed atomic load.
static int ppisp_bwd_resident_blocks() {
    static std::array<std::atomic<int>, C10_COMPILE_TIME_MAX_GPUS> cache{};
    const int device = c10::cuda::current_device();
    int blocks = cache[device].load(std::memory_order_relaxed);
    if (blocks == 0) {
        int blocks_per_sm = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, ppisp_bwd_kernel<PPISP_BLOCK_SIZE>, PPISP_BLOCK_SIZE, 0));
        // Plain runtime query rather than at::cuda::getCurrentDeviceProperties():
        // that lives in libtorch_cuda, which consumers linking only c10_cuda
        // (the NRE Bazel build) do not provide.
        int sm_count = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
        blocks = std::max(blocks_per_sm, 1) * std::max(sm_count, 1);
        cache[device].store(blocks, std::memory_order_relaxed);
    }
    return blocks;
}

void ppisp_backward(const float *exposure_params, const float *vignetting_params,
                    const float *color_params, const float *crf_params, const float *rgb_in,
                    const float *rgb_out, const float *pixel_coords, const float *v_rgb_out,
                    float *v_exposure_params, float *v_vignetting_params, float *v_color_params,
                    float *v_crf_params, float *v_rgb_in, int num_pixels, int num_cameras,
                    int num_frames, int resolution_w, int resolution_h, int camera_idx,
                    int frame_idx) {
    if (num_pixels == 0) return;
    const int threads = PPISP_BLOCK_SIZE;
    // No more blocks than can be resident at once: the grid-stride loop absorbs
    // the rest, so no partial wave of blocks trails the launch.
    const int blocks = std::min(divUp(num_pixels, threads), ppisp_bwd_resident_blocks());
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
