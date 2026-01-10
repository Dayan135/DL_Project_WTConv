#pragma once

void haar_dwt_2d(const float* input, float* output, int N, int C, int H, int W, float scale);
void haar_idwt_2d(const float* input, float* output, int N, int C, int H, int W, float scale);

void conv2d_forward_impl(const float* input, const float* weight, float* output,
                         int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups);

void conv2d_backward_impl(const float* grad_output, const float* input, const float* weight,
                          float* grad_input, float* grad_weight,
                          int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups);