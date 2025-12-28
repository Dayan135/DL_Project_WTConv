#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "cpp_kernel.h"

int main() {
    std::srand(std::time(nullptr));

    // Parameters
    int N = 2;      // Batch size
    int Cin = 3;    // Input channels
    int Cout = 3;   // Output channels
    int H = 16;     // Height (must be even for Haar)
    int W = 16;     // Width (must be even for Haar)
    int K = 3;      // Kernel size
    int Stride = 1;
    int Pad = 1;    // Padding to keep size same

    std::cout << "Initializing WTConv Test..." << std::endl;
    std::cout << "Input: " << N << "x" << Cin << "x" << H << "x" << W << std::endl;

    // Allocate memory
    // Input size
    int input_size = N * Cin * H * W;
    std::vector<float> input(input_size);
    std::vector<float> grad_input(input_size);

    // Weight size: In wavelet domain, channels are multiplied by 4
    // Shape: (4*Cout) x (4*Cin) x K x K
    int weight_size = (4 * Cout) * (4 * Cin) * K * K;
    std::vector<float> weight(weight_size);
    std::vector<float> grad_weight(weight_size);

    // Output size
    int output_size = N * Cout * H * W;
    std::vector<float> output(output_size);
    std::vector<float> grad_output(output_size);

    // Initialize data
    for (int i = 0; i < input_size; ++i) input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < weight_size; ++i) weight[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < output_size; ++i) grad_output[i] = static_cast<float>(rand()) / RAND_MAX;

    // ==========================================
    // Forward Pass
    // ==========================================
    std::cout << "Running Forward Pass..." << std::endl;
    wtconv_forward(input.data(), weight.data(), output.data(),
                   N, Cin, Cout, H, W, K, Stride, Pad);

    // Basic check: print first value
    std::cout << "Forward Output[0]: " << output[0] << std::endl;

    // ==========================================
    // Backward Pass
    // ==========================================
    std::cout << "Running Backward Pass..." << std::endl;
    wtconv_backward(grad_output.data(), input.data(), weight.data(),
                    grad_input.data(), grad_weight.data(),
                    N, Cin, Cout, H, W, K, Stride, Pad);

    std::cout << "Backward GradInput[0]: " << grad_input[0] << std::endl;
    std::cout << "Backward GradWeight[0]: " << grad_weight[0] << std::endl;

   std::cout << "Test Completed Successfully." << std::endl;

    return 0;
}
