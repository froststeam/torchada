/*
 * Python bindings for mixed source extension.
 * This .cpp file references CUDA symbols that need porting.
 */

#include <torch/extension.h>

// Forward declarations
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor mul_musa(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Mixed source extension test";
    m.def("add", &add_cuda, "Element-wise addition");
    m.def("mul", &mul_musa, "Element-wise multiplication");
}

