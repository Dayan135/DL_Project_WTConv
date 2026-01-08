#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_kernel.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

#define TILE_DIM 16

// ------------------------------------------------------------------
// 1. FUSED KERNEL: DWT + CONV (Downward Path)
// ------------------------------------------------------------------
template <int K_VAL>
__global__ void fused_wtconv_fwd_kernel(const float* __restrict__ input, 
                                        const float* __restrict__ weight, 
                                        float* __restrict__ output,
                                        float* __restrict__ next_ll,
                                        int N, int C, int H, int W, float scale) {
    
    extern __shared__ float s_raw[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int w_out = blockIdx.x * TILE_DIM + tx;
    int h_out = blockIdx.y * TILE_DIM + ty;
    int c = blockIdx.z % C;
    int n = blockIdx.z / C;
    int out_h = H / 2;
    int out_w = W / 2;
    
    int coeff_dim = TILE_DIM + K_VAL - 1;
    int raw_dim = coeff_dim * 2;
    
    int pad = K_VAL / 2;
    int h_dwt_start = blockIdx.y * TILE_DIM - pad;
    int w_dwt_start = blockIdx.x * TILE_DIM - pad;
    int h_raw_start = h_dwt_start * 2;
    int w_raw_start = w_dwt_start * 2;

    const float* in_base = input + (n * C + c) * (H * W);

    int tid = ty * TILE_DIM + tx;
    int num_threads = TILE_DIM * TILE_DIM;
    int pixels_to_load = raw_dim * raw_dim;

    for (int i = tid; i < pixels_to_load; i += num_threads) {
        int r = i / raw_dim;
        int c_loc = i % raw_dim;
        int global_h = h_raw_start + r;
        int global_w = w_raw_start + c_loc;
        float val = 0.0f;
        if (global_h >= 0 && global_h < H && global_w >= 0 && global_w < W) {
            val = in_base[global_h * W + global_w];
        }
        s_raw[i] = val;
    }

    __syncthreads();

    if (w_out >= out_w || h_out >= out_h) return;

    const float* w_base_c = weight + c * (4 * K_VAL * K_VAL);

    float sum_ll = 0.0f; float sum_lh = 0.0f;
    float sum_hl = 0.0f; float sum_hh = 0.0f;
    float center_ll_val = 0.0f; 

    for (int kh = 0; kh < K_VAL; ++kh) {
        int sm_r = (ty + kh) * 2;
        for (int kw = 0; kw < K_VAL; ++kw) {
            int sm_c = (tx + kw) * 2;
            float x00 = s_raw[sm_r * raw_dim + sm_c];
            float x01 = s_raw[sm_r * raw_dim + sm_c + 1];
            float x10 = s_raw[(sm_r + 1) * raw_dim + sm_c];
            float x11 = s_raw[(sm_r + 1) * raw_dim + sm_c + 1];

            float ll = (x00 + x01 + x10 + x11) * scale;
            float lh = (x00 + x01 - x10 - x11) * scale;
            float hl = (x00 - x01 + x10 - x11) * scale;
            float hh = (x00 - x01 - x10 + x11) * scale;

            if (kh == pad && kw == pad) center_ll_val = ll;

            int w_idx = kh * K_VAL + kw;
            int k_sq = K_VAL * K_VAL;
            sum_ll += ll * w_base_c[0 * k_sq + w_idx];
            sum_lh += lh * w_base_c[1 * k_sq + w_idx];
            sum_hl += hl * w_base_c[2 * k_sq + w_idx];
            sum_hh += hh * w_base_c[3 * k_sq + w_idx];
        }
    }

    int out_plane = out_h * out_w;
    int idx = h_out * out_w + w_out;
    int n_offset = n * (4 * C * out_plane);
    
    output[n_offset + (4*c + 0) * out_plane + idx] = sum_ll;
    output[n_offset + (4*c + 1) * out_plane + idx] = sum_lh;
    output[n_offset + (4*c + 2) * out_plane + idx] = sum_hl;
    output[n_offset + (4*c + 3) * out_plane + idx] = sum_hh;

    next_ll[(n * C + c) * out_plane + idx] = center_ll_val;
}

// ------------------------------------------------------------------
// 2. FUSED RECONSTRUCTION: Add + IDWT (Upward Path)
// ------------------------------------------------------------------
__global__ void fused_idwt_add_kernel(const float* __restrict__ coeffs, 
                                      const float* __restrict__ deep_recon,
                                      float* __restrict__ output,
                                      int N, int C, int H, int W, float scale) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % C;
    int n = blockIdx.z / C;

    if (w >= W || h >= H || n >= N) return;

    int plane_size = H * W;
    int idx = h * W + w;

    const float* coeff_ptr = coeffs + n * (4 * C * plane_size);
    float ll = coeff_ptr[(4*c + 0) * plane_size + idx];
    float lh = coeff_ptr[(4*c + 1) * plane_size + idx];
    float hl = coeff_ptr[(4*c + 2) * plane_size + idx];
    float hh = coeff_ptr[(4*c + 3) * plane_size + idx];

    if (deep_recon != nullptr) {
        const float* recon_ptr = deep_recon + (n * C + c) * plane_size;
        ll += recon_ptr[idx];
    }

    float x00 = (ll + lh + hl + hh) * scale;
    float x01 = (ll + lh - hl - hh) * scale;
    float x10 = (ll - lh + hl - hh) * scale;
    float x11 = (ll - lh - hl + hh) * scale;

    float* out_ptr = output + (n * C + c) * (4 * plane_size);
    int out_w_stride = 2 * W;
    int r0 = (2*h) * out_w_stride + (2*w);
    int r1 = (2*h+1) * out_w_stride + (2*w);

    out_ptr[r0] = x00;
    out_ptr[r0 + 1] = x01;
    out_ptr[r1] = x10;
    out_ptr[r1 + 1] = x11;
}

// ------------------------------------------------------------------
// Launchers
// ------------------------------------------------------------------

void launch_fused_wtconv_fwd(const float* input, const float* weight, 
                             float* output, float* next_ll,
                             int N, int C, int H, int W, 
                             int K, float scale) {
    int out_h = H / 2;
    int out_w = W / 2;
    dim3 block(TILE_DIM, TILE_DIM); 
    dim3 grid((out_w + TILE_DIM - 1) / TILE_DIM, (out_h + TILE_DIM - 1) / TILE_DIM, N * C);
    
    int coeff_dim = TILE_DIM + K - 1;
    int raw_dim = coeff_dim * 2;
    int smem_size = raw_dim * raw_dim * sizeof(float);
    
    if (K == 5) fused_wtconv_fwd_kernel<5><<<grid, block, smem_size>>>(input, weight, output, next_ll, N, C, H, W, scale);
    else if (K == 3) fused_wtconv_fwd_kernel<3><<<grid, block, smem_size>>>(input, weight, output, next_ll, N, C, H, W, scale);
    else if (K == 7) fused_wtconv_fwd_kernel<7><<<grid, block, smem_size>>>(input, weight, output, next_ll, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_idwt_add(const float* coeffs, const float* deep_recon, float* output,
                           int N, int C, int H, int W, float scale) {
    dim3 block(32, 8);
    dim3 grid((W + 31)/32, (H + 7)/8, N * C);
    fused_idwt_add_kernel<<<grid, block>>>(coeffs, deep_recon, output, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

// Placeholders for linking
void launch_dwt_forward(const float* input, float* output, int N, int C, int H, int W, float scale) {}
void launch_idwt_forward(const float* input, float* output, int N, int C, int H, int W, float scale) {}
void launch_dwt_backward(const float* grad_output, float* grad_input, int N, int C, int H, int W, float scale) {}
void launch_idwt_backward(const float* grad_output, float* grad_input, int N, int C, int H, int W, float scale) {}
void launch_conv_depthwise_fwd(const float* input, const float* weight, float* output, int N, int C, int H, int W, int K, int Stride, int Pad, int H_out, int W_out) {}
void launch_conv_depthwise_bwd_in(const float* grad_out, const float* weight, float* grad_in, int N, int C, int H_in, int W_in, int K, int Stride, int Pad, int H_out, int W_out) {}
void launch_conv_depthwise_bwd_w(const float* grad_out, const float* input, float* grad_weight, int N, int C, int H_out, int W_out, int K, int Stride, int Pad, int H_in, int W_in) {}