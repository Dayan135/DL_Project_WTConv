#include <pybind11/pybind11.h>

namespace py = pybind11;

// forward declaration from cpp_kernel.cpp
void hello_world();

PYBIND11_MODULE(cpp_module, m) {
    m.doc() = "Simple bindings for cpp_kernel";
    m.def("hello_world", &hello_world, "Print Hello World from C++");
}
