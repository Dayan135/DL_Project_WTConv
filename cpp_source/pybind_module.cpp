#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cpp_kernel.h"

namespace py = pybind11;

py::array_t<float> wtconv_forward_py(py::array_t<float> input, 
                                     py::array_t<float> weight, 
                                     int stride, int pad, int groups,
                                     float dwt_scale, float idwt_scale) {
    py::buffer_info buf_in = input.request();
    py::buffer_info buf_w = weight.request();

    if (buf_in.ndim != 4 || buf_w.ndim != 4) throw std::runtime_error("Inputs must be 4D");

    int N = buf_in.shape[0];
    int Cin = buf_in.shape[1];
    int H = buf_in.shape[2];
    int W = buf_in.shape[3];
    int Cout_4 = buf_w.shape[0]; 
    int K = buf_w.shape[2];
    int Cout = Cout_4 / 4;

    int H_wt = H / 2; int W_wt = W / 2;
    int H_conv = (H_wt + 2 * pad - K) / stride + 1;
    int W_conv = (W_wt + 2 * pad - K) / stride + 1;

    auto result = py::array_t<float>({N, Cout, H_conv*2, W_conv*2});
    py::buffer_info buf_out = result.request();

    wtconv_forward(static_cast<float*>(buf_in.ptr),
                   static_cast<float*>(buf_w.ptr),
                   static_cast<float*>(buf_out.ptr),
                   N, Cin, Cout, H, W, K, stride, pad, groups, dwt_scale, idwt_scale);

    return result;
}

std::pair<py::array_t<float>, py::array_t<float>> wtconv_backward_py(
        py::array_t<float> grad_output,
        py::array_t<float> input,
        py::array_t<float> weight,
        int stride, int pad, int groups,
        float dwt_scale, float idwt_scale) {

    py::buffer_info buf_go = grad_output.request();
    py::buffer_info buf_in = input.request();
    py::buffer_info buf_w = weight.request();

    int N = buf_in.shape[0];
    int Cin = buf_in.shape[1];
    int H = buf_in.shape[2];
    int W = buf_in.shape[3];
    int Cout_4 = buf_w.shape[0];
    int Cin_wt_g = buf_w.shape[1];
    int K = buf_w.shape[2];
    
    auto grad_input = py::array_t<float>({N, Cin, H, W});
    auto grad_weight = py::array_t<float>({Cout_4, Cin_wt_g, K, K});

    py::buffer_info buf_gi = grad_input.request();
    py::buffer_info buf_gw = grad_weight.request();

    wtconv_backward(static_cast<float*>(buf_go.ptr),
                    static_cast<float*>(buf_in.ptr),
                    static_cast<float*>(buf_w.ptr),
                    static_cast<float*>(buf_gi.ptr),
                    static_cast<float*>(buf_gw.ptr),
                    N, Cin, Cout_4/4, H, W, K, stride, pad, groups, dwt_scale, idwt_scale);

    return {grad_input, grad_weight};
}

PYBIND11_MODULE(cpp_module, m) {
    m.doc() = "WTConv C++ Kernel with Scaling";
    
    // Default scales set to 0.5f to match previous behavior
    m.def("wtconv_forward", &wtconv_forward_py, "Forward",
          py::arg("input"), py::arg("weight"), py::arg("stride"), py::arg("pad"), py::arg("groups")=1,
          py::arg("dwt_scale")=0.5f, py::arg("idwt_scale")=0.5f);
          
    m.def("wtconv_backward", &wtconv_backward_py, "Backward",
          py::arg("grad_output"), py::arg("input"), py::arg("weight"), py::arg("stride"), py::arg("pad"), py::arg("groups")=1,
          py::arg("dwt_scale")=0.5f, py::arg("idwt_scale")=0.5f);
}