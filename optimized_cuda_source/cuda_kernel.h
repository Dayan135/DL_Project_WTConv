#pragma once

void launch_dwt_forward(const float* input, float* output, int N, int C, int H, int W, float scale);
void launch_idwt_forward(const float* input, float* output, int N, int C, int H, int W, float scale);
void launch_dwt_backward(const float* grad_output, float* grad_input, int N, int C, int H, int W, float scale);
void launch_idwt_backward(const float* grad_output, float* grad_input, int N, int C, int H, int W, float scale);
void launch_conv_depthwise_fwd(const float* input, const float* weight, float* output,
                               int N, int C, int H, int W, int K, int Stride, int Pad, int H_out, int W_out);

// --- NEW FUSED LAUNCHER ---
void launch_fused_wtconv_fwd(const float* input, const float* weight, 
                             float* output, float* next_ll,
                             int N, int C, int H, int W, 
                             int K, float scale);

void launch_fused_idwt_add(const float* coeffs, const float* deep_recon, float* output,
                           int N, int C, int H, int W, float scale);