#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cpp_kernel.h"

namespace py = pybind11;

// Wrapper for WTConv Forward
py::array_t<float> wtconv_forward_py(py::array_t<float> input, 
                                     py::array_t<float> weight, 
                                     int stride, int pad) {
    // 1. Request buffer info
    py::buffer_info buf_in = input.request();
    py::buffer_info buf_w = weight.request();

    // 2. Validate dimensions
    if (buf_in.ndim != 4 || buf_w.ndim != 4) {
        throw std::runtime_error("Input and Weight must be 4D tensors");
    }

    int N = buf_in.shape[0];
    int Cin = buf_in.shape[1];
    int H = buf_in.shape[2];
    int W = buf_in.shape[3];

    int Cout_4 = buf_w.shape[0]; // 4 * Cout
    int Cin_4 = buf_w.shape[1];  // 4 * Cin
    int K = buf_w.shape[2];

    int Cout = Cout_4 / 4;

    if (Cin_4 != 4 * Cin) {
        throw std::runtime_error("Weight input channels must be 4 * Input channels");
    }
    
    // 3. Allocate Output
    // Assume stride=1 logic for shape calculation in IDWT, or calculate normally
    // For standard WTConv, usually H_out = H.
    int H_wt = H / 2;
    int W_wt = W / 2;
    int H_conv = (H_wt + 2 * pad - K) / stride + 1;
    int W_conv = (W_wt + 2 * pad - K) / stride + 1;
    int H_out = H_conv * 2; 
    int W_out = W_conv * 2;

    auto result = py::array_t<float>({N, Cout, H_out, W_out});
    py::buffer_info buf_out = result.request();

    // 4. Call C++ Kernel
    wtconv_forward(static_cast<float*>(buf_in.ptr),
                   static_cast<float*>(buf_w.ptr),
                   static_cast<float*>(buf_out.ptr),
                   N, Cin, Cout, H, W, K, stride, pad);

    return result;
}

// Wrapper for WTConv Backward
// Returns a tuple (grad_input, grad_weight)
std::pair<py::array_t<float>, py::array_t<float>> wtconv_backward_py(
        py::array_t<float> grad_output,
        py::array_t<float> input,
        py::array_t<float> weight,
        int stride, int pad) {

    py::buffer_info buf_go = grad_output.request();
    py::buffer_info buf_in = input.request();
    py::buffer_info buf_w = weight.request();

    int N = buf_in.shape[0];
    int Cin = buf_in.shape[1];
    int H = buf_in.shape[2];
    int W = buf_in.shape[3];
    
    int Cout_4 = buf_w.shape[0];
    int K = buf_w.shape[2];
    int Cout = Cout_4 / 4;

    // Allocate Gradients
    auto grad_input = py::array_t<float>({N, Cin, H, W});
    auto grad_weight = py::array_t<float>({Cout_4, 4*Cin, K, K});

    py::buffer_info buf_gi = grad_input.request();
    py::buffer_info buf_gw = grad_weight.request();

    wtconv_backward(static_cast<float*>(buf_go.ptr),
                    static_cast<float*>(buf_in.ptr),
                    static_cast<float*>(buf_w.ptr),
                    static_cast<float*>(buf_gi.ptr),
                    static_cast<float*>(buf_gw.ptr),
                    N, Cin, Cout, H, W, K, stride, pad);

    return {grad_input, grad_weight};
}

void hello_world_wrapper() {
    std::cout << "Hello, World!\n";
}

PYBIND11_MODULE(cpp_module, m) {
    m.doc() = "WTConv C++ Kernel";
    m.def("hello_world", &hello_world_wrapper, "Print Hello World");
    m.def("wtconv_forward", &wtconv_forward_py, "WTConv Forward Pass");
    m.def("wtconv_backward", &wtconv_backward_py, "WTConv Backward Pass");
}