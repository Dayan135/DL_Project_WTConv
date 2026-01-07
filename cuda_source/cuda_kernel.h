#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <torch/extension.h>

// Atomic CUDA Launchers
void launch_dwt_forward(const float* input, float* output, int N, int C, int H, int W, float scale);
void launch_dwt_backward(const float* grad_output, float* grad_input, int N, int C, int H, int W, float scale);

void launch_idwt_forward(const float* input, float* output, int N, int C, int H, int W, float scale);
void launch_idwt_backward(const float* grad_output, float* grad_input, int N, int C, int H, int W, float scale);

void launch_conv_depthwise_fwd(const float* input, const float* weight, float* output,
                               int N, int C, int H, int W, int K, int Stride, int Pad, int H_out, int W_out);

void launch_conv_depthwise_bwd_in(const float* grad_out, const float* weight, float* grad_in,
                                  int N, int C, int H_in, int W_in, int K, int Stride, int Pad, int H_out, int W_out);

void launch_conv_depthwise_bwd_w(const float* grad_out, const float* input, float* grad_weight,
                                 int N, int C, int H_out, int W_out, int K, int Stride, int Pad, int H_in, int W_in);

#endif