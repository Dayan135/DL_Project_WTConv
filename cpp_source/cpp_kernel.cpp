#include <iostream>
#include "cpp_kernel.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
void hello_world() {
    std::cout << "Hello, World!\n";
}


// ==========================================
// Helper: Haar Discrete Wavelet Transform 2D
// ==========================================
// Transforms N x C x H x W -> N x (4*C) x (H/2) x (W/2)
// Channels are interleaved: LL, LH, HL, HH
//
// Inputs:
//   input:  Pointer to input data array of shape (N, C, H, W).
//   output: Pointer to output data array of shape (N, 4*C, H/2, W/2).
//   N:      Batch size.
//   C:      Number of input channels.
//   H:      Input height (must be even).
//   W:      Input width (must be even).
void haar_dwt_2d(const float* input, float* output, int N, int C, int H, int W) {
    int out_h = H / 2;
    int out_w = W / 2;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < out_h; ++h) {
                for (int w = 0; w < out_w; ++w) {
                    // Indices for 2x2 block
                    int idx_base = ((n * C + c) * H + (2 * h)) * W + (2 * w);
                    float x00 = input[idx_base];
                    float x01 = input[idx_base + 1];
                    float x10 = input[idx_base + W];
                    float x11 = input[idx_base + W + 1];

                    // Haar Transform (using 0.5 scaling)
                    float ll = (x00 + x01 + x10 + x11) * 0.5f;
                    float lh = (x00 - x01 + x10 - x11) * 0.5f;
                    float hl = (x00 + x01 - x10 - x11) * 0.5f;
                    float hh = (x00 - x01 - x10 + x11) * 0.5f;

                    // Output index base: N x (4C) x H/2 x W/2
                    int out_plane_size = out_h * out_w;
                    int out_base = (n * (4 * C)) * out_plane_size + (h * out_w + w);

                    output[out_base + (4 * c + 0) * out_plane_size] = ll;
                    output[out_base + (4 * c + 1) * out_plane_size] = lh;
                    output[out_base + (4 * c + 2) * out_plane_size] = hl;
                    output[out_base + (4 * c + 3) * out_plane_size] = hh;
                }
            }
        }
    }
}


// ==========================================
// Helper: Inverse Haar DWT 2D
// ==========================================
// Transforms N x (4*C) x (H/2) x (W/2) -> N x C x H x W
//
// Inputs:
//   input:  Pointer to input data array of shape (N, 4*C, H/2, W/2).
//   output: Pointer to output data array of shape (N, C, H, W).
//   N:      Batch size.
//   C:      Number of output channels (input has 4*C channels).
//   H:      Output height (must be even).
//   W:      Output width (must be even).
void haar_idwt_2d(const float* input, float* output, int N, int C, int H, int W) {
    int in_h = H / 2;
    int in_w = W / 2;
    int in_plane_size = in_h * in_w;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < in_h; ++h) {
                for (int w = 0; w < in_w; ++w) {
                    int in_base = (n * (4 * C)) * in_plane_size + (h * in_w + w);

                    float ll = input[in_base + (4 * c + 0) * in_plane_size];
                    float lh = input[in_base + (4 * c + 1) * in_plane_size];
                    float hl = input[in_base + (4 * c + 2) * in_plane_size];
                    float hh = input[in_base + (4 * c + 3) * in_plane_size];

                    // Inverse Haar
                    float x00 = (ll + lh + hl + hh) * 0.5f;
                    float x01 = (ll - lh + hl - hh) * 0.5f;
                    float x10 = (ll + lh - hl - hh) * 0.5f;
                    float x11 = (ll - lh - hl + hh) * 0.5f;

                    int out_idx_base = ((n * C + c) * H + (2 * h)) * W + (2 * w);
                    output[out_idx_base] = x00;
                    output[out_idx_base + 1] = x01;
                    output[out_idx_base + W] = x10;
                    output[out_idx_base + W + 1] = x11;
                }
            }
        }
    }
}

// ==========================================
// Helper: Standard Conv2d Forward
// ==========================================
// Inputs:
//   input:  Pointer to input data (N, Cin, H, W).
//   weight: Pointer to weight data (Cout, Cin, K, K).
//   output: Pointer to output data (N, Cout, H_out, W_out).
//   N:      Batch size.
//   Cin:    Number of input channels.
//   Cout:   Number of output channels.
//   H:      Input height.
//   W:      Input width.
//   K:      Kernel size (assumed square KxK).
//   Stride: Convolution stride.
//   Pad:    Padding added to both sides of input.
void conv2d_forward_impl(const float* input, const float* weight, float* output,
                         int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad) {
    int H_out = (H + 2 * Pad - K) / Stride + 1;
    int W_out = (W + 2 * Pad - K) / Stride + 1;

    // Initialize output to 0
    std::fill(output, output + (N * Cout * H_out * W_out), 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int cout = 0; cout < Cout; ++cout) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                    int h_in_start = h * Stride - Pad;
                    int w_in_start = w * Stride - Pad;

                    for (int cin = 0; cin < Cin; ++cin) {
                        for (int kh = 0; kh < K; ++kh) {
                            for (int kw = 0; kw < K; ++kw) {
                                int h_in = h_in_start + kh;
                                int w_in = w_in_start + kw;

                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    int in_idx = ((n * Cin + cin) * H + h_in) * W + w_in;
                                    int w_idx = ((cout * Cin + cin) * K + kh) * K + kw;
                                    sum += input[in_idx] * weight[w_idx];
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

// ==========================================
// Helper: Standard Conv2d Backward
// ==========================================
// Inputs:
//   grad_output: Gradient of loss w.r.t output (N, Cout, H_out, W_out).
//   input:       Original input data (N, Cin, H, W).
//   weight:      Original weight data (Cout, Cin, K, K).
//   grad_input:  Gradient of loss w.r.t input (N, Cin, H, W) - to be computed.
//   grad_weight: Gradient of loss w.r.t weight (Cout, Cin, K, K) - to be computed.
//   N, Cin, Cout, H, W, K, Stride, Pad: Same as forward pass.
void conv2d_backward_impl(const float* grad_output, const float* input, const float* weight,
                          float* grad_input, float* grad_weight,
                          int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad) {
    int H_out = (H + 2 * Pad - K) / Stride + 1;
    int W_out = (W + 2 * Pad - K) / Stride + 1;

    // Initialize gradients to 0
    std::fill(grad_input, grad_input + (N * Cin * H * W), 0.0f);
    std::fill(grad_weight, grad_weight + (Cout * Cin * K * K), 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int cout = 0; cout < Cout; ++cout) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    int out_idx = ((n * Cout + cout) * H_out + h) * W_out + w;
                    float grad_val = grad_output[out_idx];

                    int h_in_start = h * Stride - Pad;
                    int w_in_start = w * Stride - Pad;

                    for (int cin = 0; cin < Cin; ++cin) {
                        for (int kh = 0; kh < K; ++kh) {
                            for (int kw = 0; kw < K; ++kw) {
                                int h_in = h_in_start + kh;
                                int w_in = w_in_start + kw;

                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    int in_idx = ((n * Cin + cin) * H + h_in) * W + w_in;
                                    int w_idx = ((cout * Cin + cin) * K + kh) * K + kw;

                                    // Accumulate gradients
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

// ==========================================
// WTConv Forward
// ==========================================
// Performs Wavelet Transform Convolution: DWT -> Conv -> IDWT
//
// Inputs:
//   input:  Pointer to input data (N, Cin, H, W).
//   weight: Pointer to weight data in wavelet domain (4*Cout, 4*Cin, K, K).
//   output: Pointer to output data (N, Cout, H, W).
//   N:      Batch size.
//   Cin:    Number of input channels.
//   Cout:   Number of output channels.
//   H:      Input height.
//   W:      Input width.
//   K:      Kernel size for the convolution in wavelet domain.
//   Stride: Stride for the convolution in wavelet domain.
//   Pad:    Padding for the convolution in wavelet domain.
void wtconv_forward(const float* input, const float* weight, float* output,
                    int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad) {
    // 1. DWT: Input (N, Cin, H, W) -> Wavelet (N, 4*Cin, H/2, W/2)
    int H_wt = H / 2;
    int W_wt = W / 2;
    int Cin_wt = 4 * Cin;
    int Cout_wt = 4 * Cout;

    std::vector<float> input_wt(N * Cin_wt * H_wt * W_wt);
    haar_dwt_2d(input, input_wt.data(), N, Cin, H, W);

    // 2. Convolution in Wavelet Domain
    // Input: (N, 4*Cin, H/2, W/2)
    // Weight: (4*Cout, 4*Cin, K, K)
    // Output: (N, 4*Cout, H/2, W/2) (Assuming stride/pad preserves size relative to input for simplicity, or standard conv)
    // Note: To reconstruct perfectly, Conv output usually matches DWT output size.
    // We use the provided Stride/Pad on the wavelet domain dimensions.
    
    // Calculate Conv output size
    int H_conv_out = (H_wt + 2 * Pad - K) / Stride + 1;
    int W_conv_out = (W_wt + 2 * Pad - K) / Stride + 1;
    
    std::vector<float> output_wt(N * Cout_wt * H_conv_out * W_conv_out);
    conv2d_forward_impl(input_wt.data(), weight, output_wt.data(), 
                        N, Cin_wt, Cout_wt, H_wt, W_wt, K, Stride, Pad);

    // 3. IDWT: Wavelet Output -> Final Output
    // We assume the convolution parameters were set such that H_conv_out == H/2 for reconstruction,
    // or we just IDWT whatever we got.
    // IDWT expects (N, 4*Cout, H_sub, W_sub) -> (N, Cout, 2*H_sub, 2*W_sub)
    haar_idwt_2d(output_wt.data(), output, N, Cout, H_conv_out * 2, W_conv_out * 2);
}

// ==========================================
// WTConv Backward
// ==========================================
// Backward pass for WTConv.
//
// Inputs:
//   grad_output: Gradient of loss w.r.t output (N, Cout, H, W).
//   input:       Original input data (N, Cin, H, W).
//   weight:      Weights in wavelet domain (4*Cout, 4*Cin, K, K).
//   grad_input:  Gradient w.r.t input (N, Cin, H, W) - to be computed.
//   grad_weight: Gradient w.r.t weights (4*Cout, 4*Cin, K, K) - to be computed.
//   N, Cin, Cout, H, W, K, Stride, Pad: Same as forward pass.
void wtconv_backward(const float* grad_output, const float* input, const float* weight,
                     float* grad_input, float* grad_weight,
                     int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad) {
    
    // Dimensions in Wavelet Domain
    int H_wt = H / 2;
    int W_wt = W / 2;
    int Cin_wt = 4 * Cin;
    int Cout_wt = 4 * Cout;

    // 1. DWT on grad_output
    // grad_output is (N, Cout, H, W) -> (N, 4*Cout, H/2, W/2)
    std::vector<float> grad_output_wt(N * Cout_wt * H_wt * W_wt);
    haar_dwt_2d(grad_output, grad_output_wt.data(), N, Cout, H, W);

    // 2. DWT on input (needed for weight gradient)
    std::vector<float> input_wt(N * Cin_wt * H_wt * W_wt);
    haar_dwt_2d(input, input_wt.data(), N, Cin, H, W);

    // 3. Conv2d Backward in Wavelet Domain
    // We need buffers for grad_input_wt and grad_weight
    // grad_input_wt: (N, 4*Cin, H/2, W/2)
    // grad_weight: (4*Cout, 4*Cin, K, K) - same as passed in pointer
    std::vector<float> grad_input_wt(N * Cin_wt * H_wt * W_wt);

    // Note: grad_weight is passed directly from caller
    conv2d_backward_impl(grad_output_wt.data(), input_wt.data(), weight,
                         grad_input_wt.data(), grad_weight,
                         N, Cin_wt, Cout_wt, H_wt, W_wt, K, Stride, Pad);

    // 4. IDWT on grad_input_wt to get grad_input
    // (N, 4*Cin, H/2, W/2) -> (N, Cin, H, W)
    haar_idwt_2d(grad_input_wt.data(), grad_input, N, Cin, H, W);
}
