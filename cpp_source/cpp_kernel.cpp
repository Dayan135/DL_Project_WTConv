#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "cpp_kernel.h"

// ==========================================
// Haar DWT 2D (CPU)
// ==========================================
void haar_dwt_2d(const float* input, float* output, int N, int C, int H, int W, float scale) {
    int out_h = H / 2;
    int out_w = W / 2;
    int in_stride_h = W;
    int out_plane_sz = out_h * out_w;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < out_h; ++h) {
                for (int w = 0; w < out_w; ++w) {
                    int idx_base = ((n * C + c) * H + (2 * h)) * in_stride_h + (2 * w);
                    float x00 = input[idx_base];
                    float x01 = input[idx_base + 1];
                    float x10 = input[idx_base + in_stride_h];
                    float x11 = input[idx_base + in_stride_h + 1];

                    float ll = (x00 + x01 + x10 + x11) * scale;
                    float lh = (x00 + x01 - x10 - x11) * scale;
                    float hl = (x00 - x01 + x10 - x11) * scale;
                    float hh = (x00 - x01 - x10 + x11) * scale;

                    int out_base_ptr = (n * (4 * C)) * out_plane_sz + (h * out_w + w);
                    
                    output[out_base_ptr + (4 * c + 0) * out_plane_sz] = ll;
                    output[out_base_ptr + (4 * c + 1) * out_plane_sz] = lh;
                    output[out_base_ptr + (4 * c + 2) * out_plane_sz] = hl;
                    output[out_base_ptr + (4 * c + 3) * out_plane_sz] = hh;
                }
            }
        }
    }
}

// ==========================================
// Inverse Haar DWT 2D (CPU)
// ==========================================
void haar_idwt_2d(const float* input, float* output, int N, int C, int H, int W, float scale) {
    int in_h = H / 2;
    int in_w = W / 2;
    int in_plane_sz = in_h * in_w;
    int out_stride_h = W;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < in_h; ++h) {
                for (int w = 0; w < in_w; ++w) {
                    int in_base = (n * (4 * C)) * in_plane_sz + (h * in_w + w);
                    
                    float ll = input[in_base + (4 * c + 0) * in_plane_sz];
                    float lh = input[in_base + (4 * c + 1) * in_plane_sz];
                    float hl = input[in_base + (4 * c + 2) * in_plane_sz];
                    float hh = input[in_base + (4 * c + 3) * in_plane_sz];

                    float x00 = (ll + lh + hl + hh) * scale;
                    float x01 = (ll + lh - hl - hh) * scale;
                    float x10 = (ll - lh + hl - hh) * scale;
                    float x11 = (ll - lh - hl + hh) * scale;

                    int out_idx = ((n * C + c) * H + (2 * h)) * out_stride_h + (2 * w);
                    output[out_idx]                 = x00;
                    output[out_idx + 1]             = x01;
                    output[out_idx + out_stride_h]     = x10;
                    output[out_idx + out_stride_h + 1] = x11;
                }
            }
        }
    }
}

// ==========================================
// Conv Forward (CPU)
// ==========================================
void conv2d_forward_impl(const float* input, const float* weight, float* output,
                         int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups) {
    int H_out = (H + 2 * Pad - K) / Stride + 1;
    int W_out = (W + 2 * Pad - K) / Stride + 1;
    int Cin_per_group = Cin / Groups;
    int Cout_per_group = Cout / Groups;

    std::fill(output, output + (N * Cout * H_out * W_out), 0.0f);

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < Groups; ++g) {
            for (int c_out_g = 0; c_out_g < Cout_per_group; ++c_out_g) {
                int cout = g * Cout_per_group + c_out_g;
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        float sum = 0.0f;
                        int h_in_start = h * Stride - Pad;
                        int w_in_start = w * Stride - Pad;
                        for (int c_in_g = 0; c_in_g < Cin_per_group; ++c_in_g) {
                            int cin = g * Cin_per_group + c_in_g;
                            for (int kh = 0; kh < K; ++kh) {
                                int h_in = h_in_start + kh;
                                if (h_in >= 0 && h_in < H) {
                                    for (int kw = 0; kw < K; ++kw) {
                                        int w_in = w_in_start + kw;
                                        if (w_in >= 0 && w_in < W) {
                                            int in_idx = ((n * Cin + cin) * H + h_in) * W + w_in;
                                            int w_idx = ((cout * Cin_per_group + c_in_g) * K + kh) * K + kw;
                                            sum += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                        int out_idx = ((n * Cout + cout) * H_out + h) * W_out + w;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }
}

// ==========================================
// Conv Backward (CPU)
// ==========================================
void conv2d_backward_impl(const float* grad_output, const float* input, const float* weight,
                          float* grad_input, float* grad_weight,
                          int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups) {
    int H_out = (H + 2 * Pad - K) / Stride + 1;
    int W_out = (W + 2 * Pad - K) / Stride + 1;
    int Cin_per_group = Cin / Groups;
    int Cout_per_group = Cout / Groups;

    // Initialize gradients to 0
    // (Assuming caller allocated memory)
    // Note: We use atomic adds in parallel regions, so zeroing is crucial.
    // However, memset/fill assumes sequential access.
    
    // For OMP Atomic, we just run the loops. The caller should clear buffers or we do it here.
    // We assume the caller handles allocation, but we must zero them out.
    // std::fill is safe here.
    
    #pragma omp parallel for
    for(int i=0; i < N*Cin*H*W; ++i) grad_input[i] = 0.0f;
    
    #pragma omp parallel for
    for(int i=0; i < Cout*Cin_per_group*K*K; ++i) grad_weight[i] = 0.0f;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < Groups; ++g) {
            for (int c_out_g = 0; c_out_g < Cout_per_group; ++c_out_g) {
                int cout = g * Cout_per_group + c_out_g;
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        int out_idx = ((n * Cout + cout) * H_out + h) * W_out + w;
                        float grad_val = grad_output[out_idx];
                        
                        int h_in_start = h * Stride - Pad;
                        int w_in_start = w * Stride - Pad;
                        
                        for (int c_in_g = 0; c_in_g < Cin_per_group; ++c_in_g) {
                            int cin = g * Cin_per_group + c_in_g;
                            for (int kh = 0; kh < K; ++kh) {
                                int h_in = h_in_start + kh;
                                if (h_in >= 0 && h_in < H) {
                                    for (int kw = 0; kw < K; ++kw) {
                                        int w_in = w_in_start + kw;
                                        if (w_in >= 0 && w_in < W) {
                                            int in_idx = ((n * Cin + cin) * H + h_in) * W + w_in;
                                            int w_idx = ((cout * Cin_per_group + c_in_g) * K + kh) * K + kw;
                                            
                                            // Atomic updates for gradients
                                            #pragma omp atomic
                                            grad_input[in_idx] += grad_val * weight[w_idx];
                                            
                                            #pragma omp atomic
                                            grad_weight[w_idx] += grad_val * input[in_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}