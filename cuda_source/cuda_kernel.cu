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

// ------------------------------------------------------------------
// DWT
// ------------------------------------------------------------------
__global__ void haar_dwt_fwd_kernel(const float* __restrict__ input, float* __restrict__ output,
                                    int N, int C, int H, int W, float scale) {
    int out_h = H / 2;
    int out_w = W / 2;
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    
    int h = hw / out_w; 
    int w = hw % out_w;
    
    if (h >= out_h || w >= out_w || n >= N || c >= C) return;

    int in_idx = (n*C + c)*(H*W);
    int row_stride = W;
    
    float x00 = input[in_idx + (2*h)*row_stride + (2*w)];
    float x01 = input[in_idx + (2*h)*row_stride + (2*w+1)];
    float x10 = input[in_idx + (2*h+1)*row_stride + (2*w)];
    float x11 = input[in_idx + (2*h+1)*row_stride + (2*w+1)];

    float ll = (x00 + x01 + x10 + x11) * scale;
    float lh = (x00 + x01 - x10 - x11) * scale;
    float hl = (x00 - x01 + x10 - x11) * scale;
    float hh = (x00 - x01 - x10 + x11) * scale;

    int out_plane_sz = out_h * out_w;
    float* out_base = output + n*(4*C*out_plane_sz);
    
    out_base[(4*c+0)*out_plane_sz + hw] = ll;
    out_base[(4*c+1)*out_plane_sz + hw] = lh;
    out_base[(4*c+2)*out_plane_sz + hw] = hl;
    out_base[(4*c+3)*out_plane_sz + hw] = hh;
}

__global__ void haar_dwt_bwd_kernel(const float* __restrict__ grad_output, float* __restrict__ grad_input,
                                    int N, int C, int H, int W, float scale) {
    int in_h = H / 2;
    int in_w = W / 2;
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    
    int h = hw / in_w; 
    int w = hw % in_w;
    
    if (h >= in_h || w >= in_w || n >= N || c >= C) return;

    int in_plane_sz = in_h * in_w;
    const float* go_base = grad_output + n*(4*C*in_plane_sz);
    
    float dll = go_base[(4*c+0)*in_plane_sz + hw];
    float dlh = go_base[(4*c+1)*in_plane_sz + hw];
    float dhl = go_base[(4*c+2)*in_plane_sz + hw];
    float dhh = go_base[(4*c+3)*in_plane_sz + hw];

    float dx00 = (dll + dlh + dhl + dhh) * scale;
    float dx01 = (dll + dlh - dhl - dhh) * scale;
    float dx10 = (dll - dlh + dhl - dhh) * scale;
    float dx11 = (dll - dlh - dhl + dhh) * scale;

    int out_stride = W;
    float* gi_base = grad_input + (n*C + c)*(H*W);
    
    gi_base[(2*h)*out_stride + (2*w)]     = dx00;
    gi_base[(2*h)*out_stride + (2*w+1)]   = dx01;
    gi_base[(2*h+1)*out_stride + (2*w)]   = dx10;
    gi_base[(2*h+1)*out_stride + (2*w+1)] = dx11;
}

// ------------------------------------------------------------------
// IDWT
// ------------------------------------------------------------------
__global__ void haar_idwt_fwd_kernel(const float* __restrict__ input, float* __restrict__ output,
                                     int N, int C, int H, int W, float scale) {
    int in_h = H / 2;
    int in_w = W / 2;
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int h = hw / in_w; int w = hw % in_w;
    
    if (h >= in_h || w >= in_w || n >= N || c >= C) return;

    int in_plane_sz = in_h * in_w;
    const float* in_base = input + n*(4*C*in_plane_sz);
    
    float ll = in_base[(4*c+0)*in_plane_sz + hw];
    float lh = in_base[(4*c+1)*in_plane_sz + hw];
    float hl = in_base[(4*c+2)*in_plane_sz + hw];
    float hh = in_base[(4*c+3)*in_plane_sz + hw];

    float x00 = (ll + lh + hl + hh) * scale;
    float x01 = (ll + lh - hl - hh) * scale;
    float x10 = (ll - lh + hl - hh) * scale;
    float x11 = (ll - lh - hl + hh) * scale;

    float* out_base = output + (n*C+c)*(H*W);
    out_base[(2*h)*W + (2*w)]     = x00;
    out_base[(2*h)*W + (2*w+1)]   = x01;
    out_base[(2*h+1)*W + (2*w)]   = x10;
    out_base[(2*h+1)*W + (2*w+1)] = x11;
}

__global__ void haar_idwt_bwd_kernel(const float* __restrict__ grad_output, float* __restrict__ grad_input,
                                     int N, int C, int H, int W, float scale) {
    int out_h = H / 2;
    int out_w = W / 2;
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int h = hw / out_w; int w = hw % out_w;
    
    if (h >= out_h || w >= out_w || n >= N || c >= C) return;

    const float* go_base = grad_output + (n*C+c)*(H*W);
    int row_stride = W;

    float g00 = go_base[(2*h)*row_stride + (2*w)];
    float g01 = go_base[(2*h)*row_stride + (2*w+1)];
    float g10 = go_base[(2*h+1)*row_stride + (2*w)];
    float g11 = go_base[(2*h+1)*row_stride + (2*w+1)];

    float dll = (g00 + g01 + g10 + g11) * scale;
    float dlh = (g00 + g01 - g10 - g11) * scale;
    float dhl = (g00 - g01 + g10 - g11) * scale;
    float dhh = (g00 - g01 - g10 + g11) * scale;

    int out_plane_sz = out_h * out_w;
    float* gi_base = grad_input + n*(4*C*out_plane_sz);
    
    gi_base[(4*c+0)*out_plane_sz + hw] = dll;
    gi_base[(4*c+1)*out_plane_sz + hw] = dlh;
    gi_base[(4*c+2)*out_plane_sz + hw] = dhl;
    gi_base[(4*c+3)*out_plane_sz + hw] = dhh;
}

// ------------------------------------------------------------------
// Conv
// ------------------------------------------------------------------
__global__ void conv2d_dw_fwd_kernel(const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ output,
                                     int N, int C, int H, int W, int K, int Stride, int Pad, int H_out, int W_out) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = hw / W_out;
    int w_out = hw % W_out;
    if (h_out >= H_out || w_out >= W_out || n >= N || c >= C) return;

    float sum = 0.0f;
    const float* in_ptr = input + (n*C+c)*(H*W);
    const float* w_ptr = weight + c*(K*K);

    int h_start = h_out*Stride - Pad;
    int w_start = w_out*Stride - Pad;

    for(int kh=0; kh<K; ++kh) {
        int h_in = h_start + kh;
        if(h_in >= 0 && h_in < H) {
            for(int kw=0; kw<K; ++kw) {
                int w_in = w_start + kw;
                if(w_in >= 0 && w_in < W) 
                    sum += in_ptr[h_in*W + w_in] * w_ptr[kh*K+kw];
            }
        }
    }
    output[(n*C+c)*(H_out*W_out) + hw] = sum;
}

// BWD INPUT
__global__ void conv2d_dw_bwd_in_kernel(const float* __restrict__ grad_out, const float* __restrict__ weight, float* __restrict__ grad_in,
                                        int N, int C, int H_in, int W_in, int K, int Stride, int Pad, int H_out, int W_out) {
    int n = blockIdx.z;
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int h_in = hw / W_in;
    int w_in = hw % W_in;
    if (h_in >= H_in || w_in >= W_in || n >= N || c >= C) return;

    float sum = 0.0f;
    const float* go_ptr = grad_out + (n*C+c)*(H_out*W_out);
    const float* w_ptr = weight + c*(K*K);

    for(int kh=0; kh<K; ++kh) {
        int h_out_idx = h_in + Pad - kh;
        if (h_out_idx % Stride == 0) {
            int h_k = h_out_idx / Stride;
            if(h_k >= 0 && h_k < H_out) {
                for(int kw=0; kw<K; ++kw) {
                    int w_out_idx = w_in + Pad - kw;
                    if (w_out_idx % Stride == 0) {
                        int w_k = w_out_idx / Stride;
                        if(w_k >= 0 && w_k < W_out) {
                            sum += go_ptr[h_k*W_out + w_k] * w_ptr[kh*K+kw];
                        }
                    }
                }
            }
        }
    }
    grad_in[(n*C+c)*(H_in*W_in) + hw] = sum;
}

// BWD WEIGHT
__global__ void conv2d_dw_bwd_w_kernel(const float* __restrict__ grad_out, const float* __restrict__ input, float* __restrict__ grad_weight,
                                       int N, int C, int H_out, int W_out, int K, int Stride, int Pad, int H_in, int W_in) {
    // One thread per weight element per channel
    int c = blockIdx.y;
    int kw_idx = blockIdx.x * blockDim.x + threadIdx.x; // 0..K*K
    if (c >= C || kw_idx >= K*K) return;

    int kh = kw_idx / K;
    int kw = kw_idx % K;

    float sum = 0.0f;
    for(int n=0; n<N; ++n) {
        const float* go_ptr = grad_out + (n*C+c)*(H_out*W_out);
        const float* in_ptr = input + (n*C+c)*(H_in*W_in);
        
        for(int h=0; h<H_out; ++h) {
            for(int w=0; w<W_out; ++w) {
                int h_in = h*Stride - Pad + kh;
                int w_in = w*Stride - Pad + kw;
                if(h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    sum += go_ptr[h*W_out+w] * in_ptr[h_in*W_in + w_in];
                }
            }
        }
    }
    grad_weight[c*(K*K) + kw_idx] = sum;
}

// ------------------------------------------------------------------
// Launchers
// ------------------------------------------------------------------
void launch_dwt_forward(const float* input, float* output, int N, int C, int H, int W, float scale) {
    dim3 block(256);
    dim3 grid(((H/2)*(W/2) + 255)/256, C, N);
    haar_dwt_fwd_kernel<<<grid, block>>>(input, output, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_dwt_backward(const float* grad_output, float* grad_input, int N, int C, int H, int W, float scale) {
    dim3 block(256);
    dim3 grid(((H/2)*(W/2) + 255)/256, C, N);
    haar_dwt_bwd_kernel<<<grid, block>>>(grad_output, grad_input, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_idwt_forward(const float* input, float* output, int N, int C, int H, int W, float scale) {
    dim3 block(256);
    dim3 grid(((H/2)*(W/2) + 255)/256, C, N);
    haar_idwt_fwd_kernel<<<grid, block>>>(input, output, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_idwt_backward(const float* grad_output, float* grad_input, int N, int C, int H, int W, float scale) {
    dim3 block(256);
    dim3 grid(((H/2)*(W/2) + 255)/256, C, N);
    haar_idwt_bwd_kernel<<<grid, block>>>(grad_output, grad_input, N, C, H, W, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv_depthwise_fwd(const float* input, const float* weight, float* output,
                               int N, int C, int H, int W, int K, int Stride, int Pad, int H_out, int W_out) {
    dim3 block(256);
    dim3 grid((H_out*W_out + 255)/256, C, N);
    conv2d_dw_fwd_kernel<<<grid, block>>>(input, weight, output, N, C, H, W, K, Stride, Pad, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv_depthwise_bwd_in(const float* grad_out, const float* weight, float* grad_in,
                                  int N, int C, int H_in, int W_in, int K, int Stride, int Pad, int H_out, int W_out) {
    dim3 block(256);
    dim3 grid((H_in*W_in + 255)/256, C, N);
    conv2d_dw_bwd_in_kernel<<<grid, block>>>(grad_out, weight, grad_in, N, C, H_in, W_in, K, Stride, Pad, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

void launch_conv_depthwise_bwd_w(const float* grad_out, const float* input, float* grad_weight,
                                 int N, int C, int H_out, int W_out, int K, int Stride, int Pad, int H_in, int W_in) {
    dim3 block(K*K);
    dim3 grid(1, C);
    conv2d_dw_bwd_w_kernel<<<grid, block>>>(grad_out, input, grad_weight, N, C, H_out, W_out, K, Stride, Pad, H_in, W_in);
    CUDA_CHECK(cudaGetLastError());
}