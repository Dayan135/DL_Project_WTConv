#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
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

// ==========================
// Forward-optimized tile size
// ==========================
#ifndef FWD_TILE_X
#define FWD_TILE_X 32
#endif

#ifndef FWD_TILE_Y
#define FWD_TILE_Y 8
#endif

// =================================================================================================
// 1. FUSED FORWARD: DWT + Depthwise Conv (Optimized Forward Only)
//    - 32x8 blocks for better coalescing
//    - Row-wise cooperative load into shared memory
//    - Cache 4*K*K weights in shared memory (broadcast-friendly)
//    - Unroll loops for fixed K
// =================================================================================================
template <int K_VAL>
__global__ void fused_wtconv_fwd_kernel_opt(const float* __restrict__ input,
                                            const float* __restrict__ weight,
                                            float* __restrict__ output,
                                            float* __restrict__ next_ll,
                                            int N, int C, int H, int W,
                                            float scale) {
    // Shared layout:
    //  [raw tile floats] [weights floats]
    extern __shared__ float smem[];
    float* s_raw = smem;

    const int tx = threadIdx.x; // [0..31]
    const int ty = threadIdx.y; // [0..7]

    const int w_out = blockIdx.x * FWD_TILE_X + tx; // coeff-domain x (W/2)
    const int h_out = blockIdx.y * FWD_TILE_Y + ty; // coeff-domain y (H/2)

    const int c = blockIdx.z % C;
    const int n = blockIdx.z / C;

    constexpr int pad = K_VAL / 2;

    // Tile geometry in coeff-domain (subsampled): (FWD_TILE + K - 1)
    constexpr int coeff_w = FWD_TILE_X + K_VAL - 1;
    constexpr int coeff_h = FWD_TILE_Y + K_VAL - 1;

    // Tile geometry in raw-domain (pre-DWT): *2
    constexpr int raw_w = coeff_w * 2;
    constexpr int raw_h = coeff_h * 2;

    // Start in coeff-domain, then *2 in raw-domain
    const int h_dwt_start = (int)blockIdx.y * FWD_TILE_Y - pad;
    const int w_dwt_start = (int)blockIdx.x * FWD_TILE_X - pad;

    const float* in_base = input + (n * C + c) * (H * W);

    // 1) Coalesced 2D cooperative load of raw tile
    //    Row-wise load gives warp-coalesced global reads.
    for (int r = ty; r < raw_h; r += blockDim.y) {
        const int gh = h_dwt_start * 2 + r;
        for (int cc = tx; cc < raw_w; cc += blockDim.x) {
            const int gw = w_dwt_start * 2 + cc;
            float val = 0.0f;
            if ((unsigned)gh < (unsigned)H && (unsigned)gw < (unsigned)W) {
                val = in_base[gh * W + gw];
            }
            s_raw[r * raw_w + cc] = val;
        }
    }

    // 2) Cache weights in shared memory
    float* s_w = s_raw + raw_h * raw_w;
    constexpr int w_elems = 4 * K_VAL * K_VAL;

    const int lin_tid = ty * blockDim.x + tx; // 0..255
    const float* w_ptr = weight + c * w_elems;

    for (int i = lin_tid; i < w_elems; i += blockDim.x * blockDim.y) {
        s_w[i] = w_ptr[i];
    }

    __syncthreads();

    // Bounds in coeff-domain
    if ((unsigned)w_out >= (unsigned)(W / 2) || (unsigned)h_out >= (unsigned)(H / 2)) return;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float center_ll = 0.0f;

    constexpr int k_sq = K_VAL * K_VAL;

    // 3) Compute DWT coeffs on-the-fly from shared raw tile and accumulate conv
    #pragma unroll
    for (int kh = 0; kh < K_VAL; ++kh) {
        const int sm_r = (ty + kh) * 2;
        #pragma unroll
        for (int kw = 0; kw < K_VAL; ++kw) {
            const int sm_c = (tx + kw) * 2;

            const float x00 = s_raw[sm_r * raw_w + sm_c];
            const float x01 = s_raw[sm_r * raw_w + (sm_c + 1)];
            const float x10 = s_raw[(sm_r + 1) * raw_w + sm_c];
            const float x11 = s_raw[(sm_r + 1) * raw_w + (sm_c + 1)];

            // Same Haar DWT math, fewer adds via reuse
            const float a  = x00 + x01;
            const float b  = x10 + x11;
            const float c1 = x00 - x01;
            const float d  = x10 - x11;

            const float ll = (a + b)  * scale;
            const float lh = (a - b)  * scale;
            const float hl = (c1 + d) * scale;
            const float hh = (c1 - d) * scale;

            if (kh == pad && kw == pad) center_ll = ll;

            const int w_idx = kh * K_VAL + kw;

            // s_w layout: [LL][LH][HL][HH]
            acc0 = fmaf(ll, s_w[0 * k_sq + w_idx], acc0);
            acc1 = fmaf(lh, s_w[1 * k_sq + w_idx], acc1);
            acc2 = fmaf(hl, s_w[2 * k_sq + w_idx], acc2);
            acc3 = fmaf(hh, s_w[3 * k_sq + w_idx], acc3);
        }
    }

    const int plane = (H / 2) * (W / 2);
    const int out_idx = n * (4 * C * plane) + h_out * (W / 2) + w_out;

    output[out_idx + 0 * plane + 4 * c * plane] = acc0;
    output[out_idx + 1 * plane + 4 * c * plane] = acc1;
    output[out_idx + 2 * plane + 4 * c * plane] = acc2;
    output[out_idx + 3 * plane + 4 * c * plane] = acc3;

    next_ll[n * (C * plane) + c * plane + h_out * (W / 2) + w_out] = center_ll;
}

// =================================================================================================
// 2. FUSED FORWARD: IDWT + Residual Add  (UNCHANGED)
// =================================================================================================
__global__ void fused_idwt_add_kernel(const float* __restrict__ coeffs, const float* __restrict__ deep_recon,
                                      float* __restrict__ output, int N, int C, int H, int W, float scale) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % C; int n = blockIdx.z / C;
    if (w >= W || h >= H) return;

    int plane = H * W;
    int idx = h * W + w;
    const float* c_ptr = coeffs + n * (4 * C * plane) + 4 * c * plane;

    float b[4];
    b[0] = c_ptr[0 * plane + idx];
    b[1] = c_ptr[1 * plane + idx];
    b[2] = c_ptr[2 * plane + idx];
    b[3] = c_ptr[3 * plane + idx];

    if (deep_recon) b[0] += deep_recon[n * (C * plane) + c * plane + idx];

    float x00 = (b[0] + b[1] + b[2] + b[3]) * scale;
    float x01 = (b[0] + b[1] - b[2] - b[3]) * scale;
    float x10 = (b[0] - b[1] + b[2] - b[3]) * scale;
    float x11 = (b[0] - b[1] - b[2] + b[3]) * scale;

    float* out = output + (n * C + c) * (4 * plane);
    int stride = 2 * W;
    out[(2 * h) * stride + 2 * w]     = x00;
    out[(2 * h) * stride + 2 * w + 1] = x01;
    out[(2 * h + 1) * stride + 2 * w] = x10;
    out[(2 * h + 1) * stride + 2 * w + 1] = x11;
}

// =================================================================================================
// 3. BACKWARD: Split Gradients (DWT Split)  (UNCHANGED)
// =================================================================================================
__global__ void fused_dwt_split_kernel(const float* __restrict__ grad_recon,
                                       float* __restrict__ grad_conv_out,
                                       float* __restrict__ grad_prev_ll,
                                       int N, int C, int H, int W, float scale) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % C; int n = blockIdx.z / C;
    if (w >= W || h >= H) return;

    int stride = 2 * W;
    const float* in = grad_recon + (n * C + c) * (2 * H * 2 * W);
    int r0 = (2 * h) * stride + 2 * w;
    int r1 = (2 * h + 1) * stride + 2 * w;

    float x00 = in[r0];     float x01 = in[r0 + 1];
    float x10 = in[r1];     float x11 = in[r1 + 1];

    float ll = (x00 + x01 + x10 + x11) * scale;
    float lh = (x00 + x01 - x10 - x11) * scale;
    float hl = (x00 - x01 + x10 - x11) * scale;
    float hh = (x00 - x01 - x10 + x11) * scale;

    int plane = H * W;
    int idx = h * W + w;
    float* out = grad_conv_out + n * (4 * C * plane);
    out[(4 * c + 0) * plane + idx] = ll;
    out[(4 * c + 1) * plane + idx] = lh;
    out[(4 * c + 2) * plane + idx] = hl;
    out[(4 * c + 3) * plane + idx] = hh;

    if (grad_prev_ll) grad_prev_ll[(n * C + c) * plane + idx] = ll;
}

// =================================================================================================
// 4. BACKWARD: Input Gradient (Fused Conv Bwd + IDWT)  (UNCHANGED)
// =================================================================================================
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
    const float* g_base_c = grad_conv_out + n * (4 * C * plane_sz) + 4 * c * plane_sz;

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

    float val_ll = 0.0f, val_lh = 0.0f, val_hl = 0.0f, val_hh = 0.0f;

    if (w_out < W && h_out < H) {
        const float* w_base_c = weight + c * (4 * K_VAL * K_VAL);
        for (int kh = 0; kh < K_VAL; ++kh) {
            for (int kw = 0; kw < K_VAL; ++kw) {
                int sm_idx = (ty + kh) * dim_tile + (tx + kw);
                float px_ll = s_ll[sm_idx];
                float px_lh = s_lh[sm_idx];
                float px_hl = s_hl[sm_idx];
                float px_hh = s_hh[sm_idx];

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
    int r0 = (2 * h_out) * row_stride + (2 * w_out);
    int r1 = (2 * h_out + 1) * row_stride + (2 * w_out);

    out_ptr[r0]     = x00; out_ptr[r0 + 1] = x01;
    out_ptr[r1]     = x10; out_ptr[r1 + 1] = x11;
}

// =================================================================================================
// 5. BACKWARD: Weight Gradient (FAST PARALLEL REDUCTION)  (UNCHANGED)
// =================================================================================================
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void conv2d_dw_bwd_w_parallel_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ input,
    float* __restrict__ grad_weight,
    int N, int C, int H_out, int W_out,
    int K, int Stride, int Pad,
    int H_in, int W_in
) {
    int c = blockIdx.x;
    int kw_idx = blockIdx.y;

    int kh = kw_idx / K;
    int kw = kw_idx % K;

    int total = N * H_out * W_out;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int tmp = idx;
        int w = tmp % W_out; tmp /= W_out;
        int h = tmp % H_out; tmp /= H_out;
        int n = tmp;

        int go = ((n * C + c) * H_out + h) * W_out + w;

        int ih = h * Stride + kh - Pad;
        int iw = w * Stride + kw - Pad;

        float in_val = 0.0f;
        if ((unsigned)ih < (unsigned)H_in && (unsigned)iw < (unsigned)W_in) {
            int in = ((n * C + c) * H_in + ih) * W_in + iw;
            in_val = input[in];
        }

        sum += grad_out[go] * in_val;
    }

    sum = warp_reduce_sum(sum);

    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) shared[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            grad_weight[c * (K * K) + kw_idx] = block_sum;
        }
    }
}

// =================================================================================================
// Launchers
// =================================================================================================
void launch_fused_wtconv_fwd(const float* input, const float* weight, float* output, float* next_ll,
                             int N, int C, int H, int W, int K, float scale) {
    // H,W here are subsampled spatial dims (H_sub, W_sub).
    // Kernel expects full-res: H_full = H*2, W_full = W*2.
    dim3 block(FWD_TILE_X, FWD_TILE_Y);
    dim3 grid((W + FWD_TILE_X - 1) / FWD_TILE_X,
              (H + FWD_TILE_Y - 1) / FWD_TILE_Y,
              N * C);

    // Shared memory = raw tile + cached weights
    const int raw_w = (FWD_TILE_X + K - 1) * 2;
    const int raw_h = (FWD_TILE_Y + K - 1) * 2;
    const int smem_floats = raw_w * raw_h + (4 * K * K);
    const int smem_bytes = smem_floats * (int)sizeof(float);

    if (K == 5) fused_wtconv_fwd_kernel_opt<5><<<grid, block, smem_bytes>>>(input, weight, output, next_ll, N, C, H * 2, W * 2, scale);
    else if (K == 3) fused_wtconv_fwd_kernel_opt<3><<<grid, block, smem_bytes>>>(input, weight, output, next_ll, N, C, H * 2, W * 2, scale);
    else if (K == 7) fused_wtconv_fwd_kernel_opt<7><<<grid, block, smem_bytes>>>(input, weight, output, next_ll, N, C, H * 2, W * 2, scale);

    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_idwt_add(const float* coeffs, const float* deep_recon, float* output,
                           int N, int C, int H, int W, float scale) {
    dim3 block(32, 8);
    dim3 grid((W + 31) / 32, (H + 7) / 8, N * C);
    fused_idwt_add_kernel<<<grid, block>>>(coeffs, deep_recon, output, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_dwt_split(const float* grad_recon, float* grad_conv_out, float* grad_prev_ll,
                            int N, int C, int H, int W, float scale) {
    dim3 block(32, 8);
    dim3 grid((W + 31) / 32, (H + 7) / 8, N * C);
    fused_dwt_split_kernel<<<grid, block>>>(grad_recon, grad_conv_out, grad_prev_ll, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_conv_bwd_idwt(const float* grad_conv_out, const float* grad_next_ll, const float* weight,
                                float* grad_input, int N, int C, int H, int W, int K, float scale) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((W + TILE_DIM - 1) / TILE_DIM, (H + TILE_DIM - 1) / TILE_DIM, N * C);
    int smem = 4 * (TILE_DIM + K - 1) * (TILE_DIM + K - 1) * sizeof(float);

    if (K == 5) fused_conv_bwd_idwt_kernel<5><<<grid, block, smem>>>(grad_conv_out, grad_next_ll, weight, grad_input, N, C, H, W, scale);
    else if (K == 3) fused_conv_bwd_idwt_kernel<3><<<grid, block, smem>>>(grad_conv_out, grad_next_ll, weight, grad_input, N, C, H, W, scale);
    else if (K == 7) fused_conv_bwd_idwt_kernel<7><<<grid, block, smem>>>(grad_conv_out, grad_next_ll, weight, grad_input, N, C, H, W, scale);

    CUDA_CHECK(cudaGetLastError());
}

void launch_conv_depthwise_bwd_w(const float* grad_out, const float* input, float* grad_weight,
                                 int N, int C, int H_out, int W_out,
                                 int K, int Stride, int Pad, int H_in, int W_in) {
    dim3 grid(C, K * K, 1);

    int total = N * H_out * W_out;
    int threads = 256;
    if (total < 256) threads = 128;
    if (total < 128) threads = 64;
    if (total < 64)  threads = 32;

    conv2d_dw_bwd_w_parallel_kernel<<<grid, threads>>>(
        grad_out, input, grad_weight,
        N, C, H_out, W_out,
        K, Stride, Pad, H_in, W_in
    );
    CUDA_CHECK(cudaGetLastError());
}
