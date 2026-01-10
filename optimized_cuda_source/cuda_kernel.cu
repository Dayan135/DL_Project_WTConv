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
// 1. FUSED FORWARD: DWT + Conv
// =================================================================================================
template <int K_VAL>
__global__ void fused_wtconv_fwd_kernel(const float* __restrict__ input, const float* __restrict__ weight, 
                                        float* __restrict__ output, float* __restrict__ next_ll,
                                        int N, int C, int H, int W, float scale) {
    extern __shared__ float s_raw[];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int w_out = blockIdx.x * TILE_DIM + tx;
    int h_out = blockIdx.y * TILE_DIM + ty;
    int c = blockIdx.z % C; int n = blockIdx.z / C;
    
    int coeff_dim = TILE_DIM + K_VAL - 1;
    int raw_dim = coeff_dim * 2;
    int pad = K_VAL/2;
    int h_dwt_start = blockIdx.y * TILE_DIM - pad;
    int w_dwt_start = blockIdx.x * TILE_DIM - pad;
    
    int tid = ty * TILE_DIM + tx;
    int num_threads = TILE_DIM * TILE_DIM;
    
    const float* in_base = input + (n * C + c) * (H * W);
    for (int i = tid; i < raw_dim * raw_dim; i += num_threads) {
        int r = i / raw_dim; int c_loc = i % raw_dim;
        int gh = h_dwt_start * 2 + r; int gw = w_dwt_start * 2 + c_loc;
        float val = 0.0f;
        if (gh >= 0 && gh < H && gw >= 0 && gw < W) val = in_base[gh * W + gw];
        s_raw[i] = val;
    }
    __syncthreads();

    if (w_out >= W/2 || h_out >= H/2) return;

    const float* w_ptr = weight + c * (4 * K_VAL * K_VAL);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float center_ll = 0.0f;

    for (int kh = 0; kh < K_VAL; ++kh) {
        int sm_r = (ty + kh) * 2;
        for (int kw = 0; kw < K_VAL; ++kw) {
            int sm_c = (tx + kw) * 2;
            float x00 = s_raw[sm_r * raw_dim + sm_c];
            float x01 = s_raw[sm_r * raw_dim + sm_c + 1];
            float x10 = s_raw[(sm_r + 1) * raw_dim + sm_c];
            float x11 = s_raw[(sm_r + 1) * raw_dim + sm_c + 1];

            float b[4];
            b[0] = (x00 + x01 + x10 + x11) * scale;
            b[1] = (x00 + x01 - x10 - x11) * scale;
            b[2] = (x00 - x01 + x10 - x11) * scale;
            b[3] = (x00 - x01 - x10 + x11) * scale;

            if (kh == pad && kw == pad) center_ll = b[0];

            int w_idx = kh * K_VAL + kw;
            int k_sq = K_VAL * K_VAL;
            for(int k=0; k<4; ++k) acc[k] += b[k] * w_ptr[k*k_sq + w_idx];
        }
    }

    int plane = (H/2)*(W/2);
    int out_idx = n*(4*C*plane) + h_out*(W/2) + w_out;
    output[out_idx + 0*plane + 4*c*plane] = acc[0];
    output[out_idx + 1*plane + 4*c*plane] = acc[1];
    output[out_idx + 2*plane + 4*c*plane] = acc[2];
    output[out_idx + 3*plane + 4*c*plane] = acc[3];
    next_ll[n*(C*plane) + c*plane + h_out*(W/2) + w_out] = center_ll;
}

// =================================================================================================
// 2. FUSED UPWARD: IDWT + Add
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
// 3. BACKWARD: DWT Splitter
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

// =================================================================================================
// 4. BACKWARD: Weights (Fast Parallel Reduction)
// =================================================================================================
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
            // Note: grad_out is (N, C, H, W)
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
// 5. BACKWARD: Input (Unrolled Fused Conv+IDWT)
// =================================================================================================
template <int K_VAL>
__global__ void fused_conv_bwd_idwt_kernel(const float* __restrict__ grad_conv_out, 
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

// =================================================================================================
// Launchers
// =================================================================================================
void launch_fused_wtconv_fwd(const float* input, const float* weight, float* output, float* next_ll, int N, int C, int H, int W, int K, float scale) {
    dim3 block(TILE_DIM, TILE_DIM); dim3 grid((W + TILE_DIM - 1)/TILE_DIM, (H + TILE_DIM - 1)/TILE_DIM, N * C);
    int smem = (TILE_DIM+K-1)*2 * (TILE_DIM+K-1)*2 * sizeof(float);
    if(K==5) fused_wtconv_fwd_kernel<5><<<grid, block, smem>>>(input, weight, output, next_ll, N, C, H*2, W*2, scale);
    else if(K==3) fused_wtconv_fwd_kernel<3><<<grid, block, smem>>>(input, weight, output, next_ll, N, C, H*2, W*2, scale);
    else if(K==7) fused_wtconv_fwd_kernel<7><<<grid, block, smem>>>(input, weight, output, next_ll, N, C, H*2, W*2, scale);
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
void launch_fused_conv_bwd_idwt(const float* grad_conv_out, const float* weight, float* grad_input, int N, int C, int H, int W, int K, float scale) {
    dim3 block(TILE_DIM, TILE_DIM); dim3 grid((W + TILE_DIM - 1)/TILE_DIM, (H + TILE_DIM - 1)/TILE_DIM, N * C);
    int smem = 4 * (TILE_DIM+K-1)*(TILE_DIM+K-1) * sizeof(float);
    if(K==5) fused_conv_bwd_idwt_kernel<5><<<grid, block, smem>>>(grad_conv_out, weight, grad_input, N, C, H, W, scale);
    else if(K==3) fused_conv_bwd_idwt_kernel<3><<<grid, block, smem>>>(grad_conv_out, weight, grad_input, N, C, H, W, scale);
    else if(K==7) fused_conv_bwd_idwt_kernel<7><<<grid, block, smem>>>(grad_conv_out, weight, grad_input, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}
void launch_conv_depthwise_bwd_w(const float* grad_out, const float* input, float* grad_weight, int N, int C, int H_out, int W_out, int K, int Stride, int Pad, int H_in, int W_in) {
    dim3 grid(K*K, C); dim3 block(256);
    conv2d_dw_bwd_w_reduce_kernel<<<grid, block>>>(grad_out, input, grad_weight, N, C, H_out, W_out, K, Stride, Pad, H_in, W_in);
    CUDA_CHECK(cudaGetLastError());
}