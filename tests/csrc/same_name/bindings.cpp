// Python bindings for the same_name test.
// This test verifies that when both .cu and .mu files exist with the same
// base name, the .mu file takes precedence.

#include <torch/extension.h>

// Declaration of the function from kernel.cu or kernel.mu
int get_magic_number();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_magic_number", &get_magic_number, "Get the magic number to verify which file was used");
}
