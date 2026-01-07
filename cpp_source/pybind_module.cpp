#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cpp_kernel.h"

namespace py = pybind11;

// Wrapper for WTConv Forward
py::array_t<float> wtconv_forward_py(py::array_t<float> input, 
                                     py::array_t<float> weight, 
                                     int stride, int pad, int groups) {
    // 1. Request buffer info
    py::buffer_info buf_in = input.request();
    py::buffer_info buf_w = weight.request();

    if (buf_in.ndim != 4 || buf_w.ndim != 4) {
        throw std::runtime_error("Input and Weight must be 4D tensors");
    }

    int N = buf_in.shape[0];
    int Cin = buf_in.shape[1];
    int H = buf_in.shape[2];
    int W = buf_in.shape[3];

    int Cout_4 = buf_w.shape[0]; 
    int K = buf_w.shape[2];
    int Cout = Cout_4 / 4;

    // Basic check for groups compatibility
    // In wavelet domain, channels are 4x. So groups must divide 4*Cin.
    if ((4 * Cin) % groups != 0) {
         throw std::runtime_error("Wavelet domain input channels must be divisible by groups");
    }

    // 2. Allocate Output
    int H_wt = H / 2;
    int W_wt = W / 2;
    int H_conv = (H_wt + 2 * pad - K) / stride + 1;
    int W_conv = (W_wt + 2 * pad - K) / stride + 1;
    int H_out = H_conv * 2; 
    int W_out = W_conv * 2;

    auto result = py::array_t<float>({N, Cout, H_out, W_out});
    py::buffer_info buf_out = result.request();

    // 3. Call C++ Kernel
    wtconv_forward(static_cast<float*>(buf_in.ptr),
                   static_cast<float*>(buf_w.ptr),
                   static_cast<float*>(buf_out.ptr),
                   N, Cin, Cout, H, W, K, stride, pad, groups);

    return result;
}

// Wrapper for WTConv Backward
std::pair<py::array_t<float>, py::array_t<float>> wtconv_backward_py(
        py::array_t<float> grad_output,
        py::array_t<float> input,
        py::array_t<float> weight,
        int stride, int pad, int groups) {

    py::buffer_info buf_go = grad_output.request();
    py::buffer_info buf_in = input.request();
    py::buffer_info buf_w = weight.request();

    int N = buf_in.shape[0];
    int Cin = buf_in.shape[1];
    int H = buf_in.shape[2];
    int W = buf_in.shape[3];
    
    int Cout_4 = buf_w.shape[0];
    int Cin_per_group_wt = buf_w.shape[1]; // 4*Cin / Groups
    int K = buf_w.shape[2];
    int Cout = Cout_4 / 4;

    // Allocate Gradients
    auto grad_input = py::array_t<float>({N, Cin, H, W});
    // Grad weight shape must match input weight shape
    auto grad_weight = py::array_t<float>({Cout_4, Cin_per_group_wt, K, K});

    py::buffer_info buf_gi = grad_input.request();
    py::buffer_info buf_gw = grad_weight.request();

    wtconv_backward(static_cast<float*>(buf_go.ptr),
                    static_cast<float*>(buf_in.ptr),
                    static_cast<float*>(buf_w.ptr),
                    static_cast<float*>(buf_gi.ptr),
                    static_cast<float*>(buf_gw.ptr),
                    N, Cin, Cout, H, W, K, stride, pad, groups);

    return {grad_input, grad_weight};
}


PYBIND11_MODULE(cpp_module, m) {
    m.doc() = "WTConv C++ Kernel";
    
    m.def("wtconv_forward", &wtconv_forward_py, "WTConv Forward Pass",
          py::arg("input"), py::arg("weight"), py::arg("stride"), py::arg("pad"), py::arg("groups")=1);
          
    m.def("wtconv_backward", &wtconv_backward_py, "WTConv Backward Pass",
          py::arg("grad_output"), py::arg("input"), py::arg("weight"), py::arg("stride"), py::arg("pad"), py::arg("groups")=1);
}