#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "cpp_kernel.h"

// ==========================================
// Helper: Haar DWT 2D (Corrected Order)
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

                    // --- CORRECTED DEFINITIONS ---
                    // LL: Avg
                    float ll = (x00 + x01 + x10 + x11) * scale;
                    
                    // LH (Band 1): Low X (Sum), High Y (Diff) -> Vertical Derivative (Top - Bottom)
                    float lh = (x00 + x01 - x10 - x11) * scale; 

                    // HL (Band 2): High X (Diff), Low Y (Sum) -> Horizontal Derivative (Left - Right)
                    float hl = (x00 - x01 + x10 - x11) * scale;

                    // HH: Diagonal
                    float hh = (x00 - x01 - x10 + x11) * scale;

                    int out_base = (n * (4 * C)) * out_plane_sz + (h * out_w + w);
                    output[out_base + (4 * c + 0) * out_plane_sz] = ll;
                    output[out_base + (4 * c + 1) * out_plane_sz] = lh; // Was hl before
                    output[out_base + (4 * c + 2) * out_plane_sz] = hl; // Was lh before
                    output[out_base + (4 * c + 3) * out_plane_sz] = hh;
                }
            }
        }
    }
}

// ==========================================
// Helper: Inverse Haar DWT 2D (Corrected Order)
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
                    
                    // Retrieve in same order: LL, LH, HL, HH
                    float ll = input[in_base + (4 * c + 0) * in_plane_sz];
                    float lh = input[in_base + (4 * c + 1) * in_plane_sz]; // Top-Bottom
                    float hl = input[in_base + (4 * c + 2) * in_plane_sz]; // Left-Right
                    float hh = input[in_base + (4 * c + 3) * in_plane_sz];

                    // Inverse Logic must match the Forward definitions:
                    // x00 (TL) = LL + LH + HL + HH 
                    // Let's verify:
                    // LL+LH = (TL+TR+BL+BR) + (TL+TR-BL-BR) = 2(TL+TR)
                    // HL+HH = (TL-TR+BL-BR) + (TL-TR-BL+BR) = 2(TL-TR)
                    // (LL+LH) + (HL+HH) = 2(TL+TR) + 2(TL-TR) = 4*TL.
                    // So we sum them all and multiply by scale (if scale is 0.25 effective).
                    // If we use scale=0.5 here (and 0.5 forward), 0.5*0.5*4 = 1.0. Correct.

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

// [Include the SAME conv2d_forward_impl and conv2d_backward_impl from previous step here]
// I will not repeat them to save space, but they MUST be in the file.
// ... (Paste conv2d_forward_impl and conv2d_backward_impl here) ...
// <START_PASTE_PLACEHOLDER>
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
void conv2d_backward_impl(const float* grad_output, const float* input, const float* weight,
                          float* grad_input, float* grad_weight,
                          int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups) {
    int H_out = (H + 2 * Pad - K) / Stride + 1;
    int W_out = (W + 2 * Pad - K) / Stride + 1;
    int Cin_per_group = Cin / Groups;
    int Cout_per_group = Cout / Groups;
    std::fill(grad_input, grad_input + (N * Cin * H * W), 0.0f);
    std::fill(grad_weight, grad_weight + (Cout * Cin_per_group * K * K), 0.0f);
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
                                            grad_input[in_idx] += grad_val * weight[w_idx];
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
// <END_PASTE_PLACEHOLDER>

// ==========================================
// WTConv Forward
// ==========================================
void wtconv_forward(const float* input, const float* weight, float* output,
                    int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups,
                    float dwt_scale, float idwt_scale) {
    int H_wt = H / 2;
    int W_wt = W / 2;
    int Cin_wt = 4 * Cin;
    int Cout_wt = 4 * Cout;

    std::vector<float> input_wt(N * Cin_wt * H_wt * W_wt);
    haar_dwt_2d(input, input_wt.data(), N, Cin, H, W, dwt_scale);

    int H_conv_out = (H_wt + 2 * Pad - K) / Stride + 1;
    int W_conv_out = (W_wt + 2 * Pad - K) / Stride + 1;
    
    std::vector<float> output_wt(N * Cout_wt * H_conv_out * W_conv_out);
    
    conv2d_forward_impl(input_wt.data(), weight, output_wt.data(), 
                        N, Cin_wt, Cout_wt, H_wt, W_wt, K, Stride, Pad, Groups);

    haar_idwt_2d(output_wt.data(), output, N, Cout, H_conv_out * 2, W_conv_out * 2, idwt_scale);
}

// ==========================================
// WTConv Backward
// ==========================================
void wtconv_backward(const float* grad_output, const float* input, const float* weight,
                     float* grad_input, float* grad_weight,
                     int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups,
                     float dwt_scale, float idwt_scale) {
    
    int H_wt = H / 2;
    int W_wt = W / 2;
    int Cin_wt = 4 * Cin;
    int Cout_wt = 4 * Cout;

    std::vector<float> grad_output_wt(N * Cout_wt * H_wt * W_wt);
    // DWT of Grad Output uses 'dwt_scale' (symmetric)
    haar_dwt_2d(grad_output, grad_output_wt.data(), N, Cout, H, W, dwt_scale); 

    std::vector<float> input_wt(N * Cin_wt * H_wt * W_wt);
    haar_dwt_2d(input, input_wt.data(), N, Cin, H, W, dwt_scale);

    std::vector<float> grad_input_wt(N * Cin_wt * H_wt * W_wt);
    
    conv2d_backward_impl(grad_output_wt.data(), input_wt.data(), weight,
                         grad_input_wt.data(), grad_weight,
                         N, Cin_wt, Cout_wt, H_wt, W_wt, K, Stride, Pad, Groups);

    // IDWT uses 'idwt_scale'
    haar_idwt_2d(grad_input_wt.data(), grad_input, N, Cin, H, W, idwt_scale);
}