#include <torch/extension.h>
#include <cstdio>
#include <cmath>
#include "cpp_kernel.h"

// 1. DWT
void haar_dwt_2d(const float* input, float* output, int N, int C, int H, int W, float scale) {
    int H_out = H / 2;
    int W_out = W / 2;
    int plane_in = H * W;
    int plane_out = H_out * W_out;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* in_ptr = input + n * (C * plane_in) + c * plane_in;
            float* out_ptr = output + n * (4 * C * plane_out) + (4 * c) * plane_out;

            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float x00 = in_ptr[(2*h) * W + (2*w)];
                    float x01 = in_ptr[(2*h) * W + (2*w+1)];
                    float x10 = in_ptr[(2*h+1) * W + (2*w)];
                    float x11 = in_ptr[(2*h+1) * W + (2*w+1)];

                    out_ptr[0 * plane_out + h * W_out + w] = (x00 + x01 + x10 + x11) * scale;
                    out_ptr[1 * plane_out + h * W_out + w] = (x00 + x01 - x10 - x11) * scale;
                    out_ptr[2 * plane_out + h * W_out + w] = (x00 - x01 + x10 - x11) * scale;
                    out_ptr[3 * plane_out + h * W_out + w] = (x00 - x01 - x10 + x11) * scale;
                }
            }
        }
    }
}

// 2. IDWT
void haar_idwt_2d(const float* input, float* output, int N, int C, int H_in, int W_in, float scale) {
    int H_out = H_in * 2;
    int W_out = W_in * 2;
    int plane_in = H_in * W_in;
    int plane_out = H_out * W_out;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float* in_ptr = input + n * (4 * C * plane_in) + (4 * c) * plane_in;
            float* out_ptr = output + n * (C * plane_out) + c * plane_out;

            for (int h = 0; h < H_in; ++h) {
                for (int w = 0; w < W_in; ++w) {
                    float ll = in_ptr[0 * plane_in + h * W_in + w];
                    float lh = in_ptr[1 * plane_in + h * W_in + w];
                    float hl = in_ptr[2 * plane_in + h * W_in + w];
                    float hh = in_ptr[3 * plane_in + h * W_in + w];

                    out_ptr[(2*h) * W_out + (2*w)]     = (ll + lh + hl + hh) * scale;
                    out_ptr[(2*h) * W_out + (2*w+1)]   = (ll + lh - hl - hh) * scale;
                    out_ptr[(2*h+1) * W_out + (2*w)]   = (ll - lh + hl - hh) * scale;
                    out_ptr[(2*h+1) * W_out + (2*w+1)] = (ll - lh - hl + hh) * scale;
                }
            }
        }
    }
}

// 3. Conv Forward (Depthwise)
void conv2d_forward_impl(const float* input, const float* weight, float* output,
                         int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups) {
    int H_out = (H + 2 * Pad - K) / Stride + 1;
    int W_out = (W + 2 * Pad - K) / Stride + 1;
    
    // Depthwise: Groups == Cin == Cout.
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < Cout; ++c) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                const float* in_plane = input + n * (Cin * H * W) + c * (H * W);
                const float* w_plane = weight + c * (K * K);
                int h_start = h_out * Stride - Pad;
                
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    int w_start = w_out * Stride - Pad;
                    float sum = 0.0f;
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            int h_in = h_start + kh;
                            int w_in = w_start + kw;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                sum += in_plane[h_in * W + w_in] * w_plane[kh * K + kw];
                            }
                        }
                    }
                    output[n*(Cout*H_out*W_out) + c*(H_out*W_out) + h_out*W_out + w_out] = sum;
                }
            }
        }
    }
}

// 4. Conv Backward (Depthwise)
void conv2d_backward_impl(const float* grad_output, const float* input, const float* weight,
                          float* grad_input, float* grad_weight,
                          int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups) {
    int H_out = (H + 2 * Pad - K) / Stride + 1;
    int W_out = (W + 2 * Pad - K) / Stride + 1;

    // --- Grad Input (dX) ---
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < Cin; ++c) {
            for (int h = 0; h < H; ++h) {
                const float* go_plane = grad_output + n*(Cout*H_out*W_out) + c*(H_out*W_out);
                const float* w_plane = weight + c*(K*K);
                
                for (int w = 0; w < W; ++w) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < K; ++kh) {
                        int h_num = h + Pad - kh;
                        if (h_num % Stride == 0) {
                            int h_out_idx = h_num / Stride;
                            if (h_out_idx >= 0 && h_out_idx < H_out) {
                                for (int kw = 0; kw < K; ++kw) {
                                    int w_num = w + Pad - kw;
                                    if (w_num % Stride == 0) {
                                        int w_out_idx = w_num / Stride;
                                        if (w_out_idx >= 0 && w_out_idx < W_out) {
                                            sum += go_plane[h_out_idx * W_out + w_out_idx] * w_plane[kh * K + kw];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    grad_input[n*(Cin*H*W) + c*(H*W) + h*W + w] = sum;
                }
            }
        }
    }

    // --- Grad Weight (dW) ---
    #pragma omp parallel for collapse(2)
    for (int c = 0; c < Cout; ++c) {
        for (int k_idx = 0; k_idx < K*K; ++k_idx) {
            int kh = k_idx / K;
            int kw = k_idx % K;
            float sum = 0.0f;
            
            for (int n = 0; n < N; ++n) {
                const float* go_plane = grad_output + n*(Cout*H_out*W_out) + c*(H_out*W_out);
                const float* in_plane = input + n*(Cin*H*W) + c*(H*W);
                
                for (int h_out = 0; h_out < H_out; ++h_out) {
                    int h_in = h_out * Stride - Pad + kh;
                    if (h_in >= 0 && h_in < H) {
                        for (int w_out = 0; w_out < W_out; ++w_out) {
                            int w_in = w_out * Stride - Pad + kw;
                            if (w_in >= 0 && w_in < W) {
                                sum += go_plane[h_out * W_out + w_out] * in_plane[h_in * W + w_in];
                            }
                        }
                    }
                }
            }
            grad_weight[c*(K*K) + k_idx] = sum;
        }
    }
}