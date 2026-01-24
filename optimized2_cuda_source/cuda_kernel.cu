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

// =================================================================================================
// 1. SUPER OPTIMIZED FUSED FORWARD: Pre-computed DWT in Shared Memory
// =================================================================================================
template <int K_VAL>
__global__ void fused_wtconv_fwd_precalc_kernel(const float* __restrict__ input, 
                                                const float* __restrict__ weight, 
                                                float* __restrict__ output, 
                                                float* __restrict__ next_ll,
                                                int N, int C, int H, int W, float scale) {
    // Shared Memory now stores DWT Coefficients, not raw pixels.
    // Layout: [4 bands][Height][Width]
    // 4 bands * (TILE + K - 1)^2
    extern __shared__ float s_coeffs[];
    
    // Pointers to the 4 bands in shared memory
    int smem_dim = TILE_DIM + K_VAL - 1;
    int smem_plane = smem_dim * smem_dim;
    float* s_ll = s_coeffs;
    float* s_lh = s_coeffs + smem_plane;
    float* s_hl = s_coeffs + 2 * smem_plane;
    float* s_hh = s_coeffs + 3 * smem_plane;

    int tx = threadIdx.x; int ty = threadIdx.y;
    int w_out = blockIdx.x * TILE_DIM + tx;
    int h_out = blockIdx.y * TILE_DIM + ty;
    int c = blockIdx.z % C; int n = blockIdx.z / C;

    int pad = K_VAL / 2;
    
    // Calculate global offsets for DWT input (Raw Pixels)
    // We need to load enough raw pixels to generate the coefficients.
    // The "h_out" corresponds to the coefficient index.
    // Raw pixel index = coefficient index * 2.
    
    int h_start = blockIdx.y * TILE_DIM - pad;
    int w_start = blockIdx.x * TILE_DIM - pad;
    
    int tid = ty * TILE_DIM + tx;
    int num_threads = TILE_DIM * TILE_DIM;
    
    const float* in_base = input + (n * C + c) * (H * W);

    // --- STEP 1: LOAD RAW & COMPUTE DWT INTO SMEM ---
    for (int i = tid; i < smem_plane; i += num_threads) {
        int r = i / smem_dim;      // Row in shared memory (coefficient space)
        int c_loc = i % smem_dim;  // Col in shared memory (coefficient space)
        
        int global_h_coeff = h_start + r;
        int global_w_coeff = w_start + c_loc;
        
        // Map coefficient coord -> raw pixel top-left coord
        int gh_raw = global_h_coeff * 2;
        int gw_raw = global_w_coeff * 2;
        
        float x00 = 0.0f, x01 = 0.0f, x10 = 0.0f, x11 = 0.0f;

        // Boundary check (Raw Input)
        if (gh_raw >= 0 && gh_raw < H - 1 && gw_raw >= 0 && gw_raw < W - 1) {
            // Load 2x2 block
            int row_offset = gh_raw * W + gw_raw;
            x00 = in_base[row_offset];
            x01 = in_base[row_offset + 1];
            x10 = in_base[row_offset + W];
            x11 = in_base[row_offset + W + 1];
        } 
        else if (gh_raw >= 0 && gh_raw < H && gw_raw >= 0 && gw_raw < W) {
             // Precise boundary handling for edge cases
             x00 = in_base[gh_raw * W + gw_raw];
             if (gw_raw + 1 < W) x01 = in_base[gh_raw * W + gw_raw + 1];
             if (gh_raw + 1 < H) x10 = in_base[(gh_raw + 1) * W + gw_raw];
             if (gh_raw + 1 < H && gw_raw + 1 < W) x11 = in_base[(gh_raw + 1) * W + gw_raw + 1];
        }

        // Compute DWT immediately
        s_ll[i] = (x00 + x01 + x10 + x11) * scale;
        s_lh[i] = (x00 + x01 - x10 - x11) * scale;
        s_hl[i] = (x00 - x01 + x10 - x11) * scale;
        s_hh[i] = (x00 - x01 - x10 + x11) * scale;
    }
    __syncthreads();

    // --- STEP 2: CONVOLUTION (READ FROM SMEM) ---
    if (w_out >= W/2 || h_out >= H/2) return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float* w_ptr = weight + c * (4 * K_VAL * K_VAL); // (4, K, K)

    // Save the center LL for the skip connection
    // Corresponds to kh=pad, kw=pad in the loop
    // In shared memory, our thread's output is at (ty + pad, tx + pad) relative to smem start
    int center_idx = (ty + pad) * smem_dim + (tx + pad);
    float center_ll = s_ll[center_idx];

    #pragma unroll
    for (int kh = 0; kh < K_VAL; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < K_VAL; ++kw) {
            int sm_idx = (ty + kh) * smem_dim + (tx + kw);
            
            float val_ll = s_ll[sm_idx];
            float val_lh = s_lh[sm_idx];
            float val_hl = s_hl[sm_idx];
            float val_hh = s_hh[sm_idx];

            int w_idx = kh * K_VAL + kw;
            int k_sq = K_VAL * K_VAL;

            // Simple MAD (Multiply-Add), no DWT math here
            acc[0] += val_ll * w_ptr[0 * k_sq + w_idx];
            acc[1] += val_lh * w_ptr[1 * k_sq + w_idx];
            acc[2] += val_hl * w_ptr[2 * k_sq + w_idx];
            acc[3] += val_hh * w_ptr[3 * k_sq + w_idx];
        }
    }

    // Write Output
    int plane = (H/2)*(W/2);
    int out_idx = n*(4*C*plane) + h_out*(W/2) + w_out;
    
    output[out_idx + 0*plane + 4*c*plane] = acc[0];
    output[out_idx + 1*plane + 4*c*plane] = acc[1];
    output[out_idx + 2*plane + 4*c*plane] = acc[2];
    output[out_idx + 3*plane + 4*c*plane] = acc[3];
    
    next_ll[n*(C*plane) + c*plane + h_out*(W/2) + w_out] = center_ll;
}

// =================================================================================================
// 2. FUSED UPWARD (Unchanged)
// =================================================================================================
__global__ void fused_idwt_add_kernel(const float* __restrict__ coeffs, const float* __restrict__ deep_recon,
                                      float* __restrict__ output, int N, int C, int H, int W, float scale) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % C; int n = blockIdx.z / C;
    if (w >= W || h >= H) return;

    int plane = H*W;
    int idx = h*W + w;
    const float* c_ptr = coeffs + n*(4*C*plane) + 4*c*plane;
    
    float b[4];
    b[0] = c_ptr[0*plane + idx];
    b[1] = c_ptr[1*plane + idx];
    b[2] = c_ptr[2*plane + idx];
    b[3] = c_ptr[3*plane + idx];

    if (deep_recon) b[0] += deep_recon[n*(C*plane) + c*plane + idx];

    float x00 = (b[0] + b[1] + b[2] + b[3]) * scale;
    float x01 = (b[0] + b[1] - b[2] - b[3]) * scale;
    float x10 = (b[0] - b[1] + b[2] - b[3]) * scale;
    float x11 = (b[0] - b[1] - b[2] + b[3]) * scale;

    float* out = output + (n*C+c)*(4*plane);
    int stride = 2*W;
    out[(2*h)*stride + 2*w] = x00;
    out[(2*h)*stride + 2*w+1] = x01;
    out[(2*h+1)*stride + 2*w] = x10;
    out[(2*h+1)*stride + 2*w+1] = x11;
}

// =================================================================================================
// 3. BACKWARD KERNELS (Unchanged)
// =================================================================================================
__global__ void fused_dwt_split_kernel(const float* __restrict__ grad_recon, 
                                       float* __restrict__ grad_conv_out,
                                       float* __restrict__ grad_prev_ll,
                                       int N, int C, int H, int W, float scale) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % C; int n = blockIdx.z / C;
    if (w >= W || h >= H) return;

    int stride = 2*W;
    const float* in = grad_recon + (n*C+c)*(2*H*2*W);
    int r0 = (2*h)*stride + 2*w; 
    int r1 = (2*h+1)*stride + 2*w;

    float x00 = in[r0]; float x01 = in[r0+1];
    float x10 = in[r1]; float x11 = in[r1+1];

    float ll = (x00 + x01 + x10 + x11) * scale;
    float lh = (x00 + x01 - x10 - x11) * scale;
    float hl = (x00 - x01 + x10 - x11) * scale;
    float hh = (x00 - x01 - x10 + x11) * scale;

    int plane = H*W;
    int idx = h*W + w;
    float* out = grad_conv_out + n*(4*C*plane);
    out[(4*c+0)*plane + idx] = ll;
    out[(4*c+1)*plane + idx] = lh;
    out[(4*c+2)*plane + idx] = hl;
    out[(4*c+3)*plane + idx] = hh;

    if(grad_prev_ll) grad_prev_ll[(n*C+c)*plane + idx] = ll;
}

template <int K_VAL>
__global__ void fused_conv_bwd_idwt_kernel(const float* __restrict__ grad_conv_out, 
                                           const float* __restrict__ grad_next_ll, 
                                           const float* __restrict__ weight, 
                                           float* __restrict__ grad_input,
                                           int N, int C, int H, int W, float scale) {
    extern __shared__ float s_tiles[];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int w_out = blockIdx.x * TILE_DIM + tx;
    int h_out = blockIdx.y * TILE_DIM + ty;
    int c = blockIdx.z % C; int n = blockIdx.z / C;

    int pad = K_VAL / 2;
    int dim_tile = TILE_DIM + K_VAL - 1;
    int tile_area = dim_tile * dim_tile;
    
    float* s_ll = s_tiles;
    float* s_lh = s_tiles + tile_area;
    float* s_hl = s_tiles + 2 * tile_area;
    float* s_hh = s_tiles + 3 * tile_area;

    int plane_sz = H * W;
    const float* g_base_c = grad_conv_out + n * (4 * C * plane_sz) + 4*c * plane_sz;
    
    int h_start = blockIdx.y * TILE_DIM - pad;
    int w_start = blockIdx.x * TILE_DIM - pad;
    int tid = ty * TILE_DIM + tx;
    int num_threads = TILE_DIM * TILE_DIM;
    
    for (int i = tid; i < tile_area; i += num_threads) {
        int r = i / dim_tile; int c_loc = i % dim_tile;
        int h_g = h_start + r; int w_g = w_start + c_loc;
        bool valid = (h_g >= 0 && h_g < H && w_g >= 0 && w_g < W);
        int g_idx = valid ? (h_g * W + w_g) : 0;
        
        s_ll[i] = valid ? g_base_c[0 * plane_sz + g_idx] : 0.0f;
        s_lh[i] = valid ? g_base_c[1 * plane_sz + g_idx] : 0.0f;
        s_hl[i] = valid ? g_base_c[2 * plane_sz + g_idx] : 0.0f;
        s_hh[i] = valid ? g_base_c[3 * plane_sz + g_idx] : 0.0f;
    }
    __syncthreads(); 

    float val_ll = 0.0f; float val_lh = 0.0f;
    float val_hl = 0.0f; float val_hh = 0.0f;

    if (w_out < W && h_out < H) {
        const float* w_base_c = weight + c * (4 * K_VAL * K_VAL);
        for (int kh = 0; kh < K_VAL; ++kh) {
            for (int kw = 0; kw < K_VAL; ++kw) {
                int sm_idx = (ty + kh) * dim_tile + (tx + kw);
                float px_ll = s_ll[sm_idx]; float px_lh = s_lh[sm_idx];
                float px_hl = s_hl[sm_idx]; float px_hh = s_hh[sm_idx];
                
                int w_idx_flip = (K_VAL - 1 - kh) * K_VAL + (K_VAL - 1 - kw);
                int k_sq = K_VAL * K_VAL;
                
                val_ll += px_ll * w_base_c[0 * k_sq + w_idx_flip];
                val_lh += px_lh * w_base_c[1 * k_sq + w_idx_flip];
                val_hl += px_hl * w_base_c[2 * k_sq + w_idx_flip];
                val_hh += px_hh * w_base_c[3 * k_sq + w_idx_flip];
            }
        }

        if (grad_next_ll != nullptr) {
            val_ll += grad_next_ll[n * (C * plane_sz) + c * plane_sz + h_out * W + w_out];
        }
    }

    if (w_out >= W || h_out >= H) return;

    float x00 = (val_ll + val_lh + val_hl + val_hh) * scale;
    float x01 = (val_ll + val_lh - val_hl - val_hh) * scale;
    float x10 = (val_ll - val_lh + val_hl - val_hh) * scale;
    float x11 = (val_ll - val_lh - val_hl + val_hh) * scale;

    float* out_ptr = grad_input + (n * C + c) * (2 * H * 2 * W);
    int row_stride = 2 * W;
    int r0 = (2*h_out) * row_stride + (2*w_out);
    int r1 = (2*h_out+1) * row_stride + (2*w_out);

    out_ptr[r0] = x00; out_ptr[r0 + 1] = x01;
    out_ptr[r1] = x10; out_ptr[r1 + 1] = x11;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void conv2d_dw_bwd_w_reduce_kernel(const float* __restrict__ grad_out, 
                                              const float* __restrict__ input, 
                                              float* __restrict__ grad_weight,
                                              int N, int C, int H_out, int W_out, 
                                              int K, int Stride, int Pad, int H_in, int W_in) {
    int kw_idx = blockIdx.x; 
    int c = blockIdx.y;      
    int kh = kw_idx / K;
    int kw = kw_idx % K;
    
    int out_plane = H_out * W_out;
    int in_plane = H_in * W_in;
    int total = N * out_plane;
    float sum = 0.0f;

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int idx_in_plane = i % out_plane;
        int n = i / out_plane;
        int h = idx_in_plane / W_out;
        int w = idx_in_plane % W_out;
        
        int h_in = h * Stride - Pad + kh;
        int w_in = w * Stride - Pad + kw;
        
        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            int go_idx = (n * C + c) * out_plane + h * W_out + w;
            int in_idx = (n * C + c) * in_plane + h_in * W_in + w_in;
            sum += grad_out[go_idx] * input[in_idx];
        }
    }
    
    sum = warpReduceSum(sum);
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) shared[wid] = sum;
    __syncthreads();
    sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) sum = warpReduceSum(sum);
    
    if (threadIdx.x == 0) grad_weight[c * (K*K) + kw_idx] = sum;
}

// =================================================================================================
// Launchers
// =================================================================================================
void launch_fused_wtconv_fwd(const float* input, const float* weight, float* output, float* next_ll, int N, int C, int H, int W, int K, float scale) {
    dim3 block(TILE_DIM, TILE_DIM); dim3 grid((W + TILE_DIM - 1)/TILE_DIM, (H + TILE_DIM - 1)/TILE_DIM, N * C);
    // Updated SMEM size: 4 bands * (TILE + K - 1)^2
    int smem = 4 * (TILE_DIM+K-1)*(TILE_DIM+K-1) * sizeof(float);
    if(K==5) fused_wtconv_fwd_precalc_kernel<5><<<grid, block, smem>>>(input, weight, output, next_ll, N, C, H*2, W*2, scale);
    else if(K==3) fused_wtconv_fwd_precalc_kernel<3><<<grid, block, smem>>>(input, weight, output, next_ll, N, C, H*2, W*2, scale);
    else if(K==7) fused_wtconv_fwd_precalc_kernel<7><<<grid, block, smem>>>(input, weight, output, next_ll, N, C, H*2, W*2, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_idwt_add(const float* coeffs, const float* deep_recon, float* output, int N, int C, int H, int W, float scale) {
    dim3 block(32, 8); dim3 grid((W + 31)/32, (H + 7)/8, N * C);
    fused_idwt_add_kernel<<<grid, block>>>(coeffs, deep_recon, output, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_dwt_split(const float* grad_recon, float* grad_conv_out, float* grad_prev_ll, int N, int C, int H, int W, float scale) {
    dim3 block(32, 8); dim3 grid((W + 31)/32, (H + 7)/8, N * C);
    fused_dwt_split_kernel<<<grid, block>>>(grad_recon, grad_conv_out, grad_prev_ll, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_conv_bwd_idwt(const float* grad_conv_out, const float* grad_next_ll, const float* weight, float* grad_input, int N, int C, int H, int W, int K, float scale) {
    dim3 block(TILE_DIM, TILE_DIM); dim3 grid((W + TILE_DIM - 1)/TILE_DIM, (H + TILE_DIM - 1)/TILE_DIM, N * C);
    int smem = 4 * (TILE_DIM+K-1)*(TILE_DIM+K-1) * sizeof(float);
    if(K==5) fused_conv_bwd_idwt_kernel<5><<<grid, block, smem>>>(grad_conv_out, grad_next_ll, weight, grad_input, N, C, H, W, scale);
    else if(K==3) fused_conv_bwd_idwt_kernel<3><<<grid, block, smem>>>(grad_conv_out, grad_next_ll, weight, grad_input, N, C, H, W, scale);
    else if(K==7) fused_conv_bwd_idwt_kernel<7><<<grid, block, smem>>>(grad_conv_out, grad_next_ll, weight, grad_input, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv_depthwise_bwd_w(const float* grad_out, const float* input, float* grad_weight, int N, int C, int H_out, int W_out, int K, int Stride, int Pad, int H_in, int W_in) {
    dim3 grid(K*K, C); dim3 block(256);
    conv2d_dw_bwd_w_reduce_kernel<<<grid, block>>>(grad_out, input, grad_weight, N, C, H_out, W_out, K, Stride, Pad, H_in, W_in);
    CUDA_CHECK(cudaGetLastError());
}