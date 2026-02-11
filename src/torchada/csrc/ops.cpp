// torchada C++ operator overrides - Main source file
//
// This file contains the operator registration infrastructure and example overrides.
// Custom operator implementations can be added here or in separate files.
//
// To add a new operator override:
//   1. Write the implementation function
//   2. Register it using TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
//
// Note: Operators registered here will override torch_musa's implementations.
// Use with caution and ensure correctness.

#include "ops.h"

namespace torchada {

// ============================================================================
// Example: Operator override template (commented out - for reference)
// ============================================================================
//
// To override an ATen operator, follow this pattern:
//
// static at::Tensor custom_add_impl(
//     const at::Tensor& self,
//     const at::Tensor& other,
//     const at::Scalar& alpha) {
//
//     log_op_call("add.Tensor");
//
//     // Your custom implementation here
//     // IMPORTANT: Avoid calling the same operator to prevent infinite recursion
//     // Use in-place operations or lower-level primitives instead
//     auto result = at::empty_like(self);
//     result.copy_(self);
//     result.add_(other, alpha);
//     return result;
// }
//
// Then register it:
// TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
//     m.impl("add.Tensor", custom_add_impl);
// }

// ============================================================================
// Utility functions exposed to Python
// ============================================================================

static bool cpp_ops_loaded = false;

bool is_loaded() {
    return cpp_ops_loaded;
}

const char* get_version() {
    return VERSION;
}

void mark_loaded() {
    cpp_ops_loaded = true;
}

}  // namespace torchada

// ============================================================================
// Python bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchada C++ operator overrides";

    m.def("is_loaded", &torchada::is_loaded,
          "Check if C++ ops extension is loaded");
    m.def("get_version", &torchada::get_version,
          "Get the C++ ops extension version");
    m.def("_mark_loaded", &torchada::mark_loaded,
          "Mark the extension as loaded (internal use)");
}
