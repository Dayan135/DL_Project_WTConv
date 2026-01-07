#ifndef CPP_KERNEL_H
#define CPP_KERNEL_H

void wtconv_forward(const float* input, const float* weight, float* output,
                    int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups,
                    float dwt_scale, float idwt_scale);

void wtconv_backward(const float* grad_output, const float* input, const float* weight,
                     float* grad_input, float* grad_weight,
                     int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad, int Groups,
                     float dwt_scale, float idwt_scale);

#endif // CPP_KERNEL_H