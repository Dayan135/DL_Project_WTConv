#pragma once

// Forward
void launch_fused_wtconv_fwd(const float* input, const float* weight, float* output, float* next_ll, int N, int C, int H, int W, int K, float scale);
void launch_fused_idwt_add(const float* coeffs, const float* deep_recon, float* output, int N, int C, int H, int W, float scale);

// Backward
void launch_fused_dwt_split(const float* grad_recon, float* grad_conv_out, float* grad_prev_ll, int N, int C, int H, int W, float scale);

// UPDATED: Added grad_next_ll argument
void launch_fused_conv_bwd_idwt(const float* grad_conv_out, const float* grad_next_ll, const float* weight, float* grad_input, int N, int C, int H, int W, int K, float scale);

void launch_conv_depthwise_bwd_w(const float* grad_out, const float* input, float* grad_weight, int N, int C, int H_out, int W_out, int K, int Stride, int Pad, int H_in, int W_in);

