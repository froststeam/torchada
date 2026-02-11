/*
 * Device type test CUDA kernel for testing torchada mapping rules.
 *
 * This file specifically tests the device type mappings:
 * - at::kCUDA -> at::kPrivateUse1
 * - at::DeviceType::CUDA -> at::DeviceType::PrivateUse1
 * - c10::DeviceType::CUDA -> c10::DeviceType::PrivateUse1
 * - torch::kCUDA -> torch::kPrivateUse1
 *
 * The test verifies that after porting, the code compiles and runs correctly.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>

// Test function that checks device types using various CUDA symbols
torch::Tensor check_device_type(torch::Tensor input) {
    // Test at::DeviceType::CUDA - should be ported to at::DeviceType::PrivateUse1
    TORCH_CHECK(
        input.device().type() == at::DeviceType::CUDA,
        "Input tensor must be on CUDA device"
    );

    // Test c10::DeviceType::CUDA - should be ported to c10::DeviceType::PrivateUse1
    c10::DeviceType device_type = input.device().type();
    bool is_cuda_c10 = (device_type == c10::DeviceType::CUDA);
    TORCH_CHECK(is_cuda_c10, "Device type check with c10:: failed");

    // Create output tensor on same device as input
    // Use c10::Device with DeviceType::CUDA which becomes PrivateUse1 after porting
    auto output = at::empty({1}, input.options());

    // Copy input value to output (just a simple operation to verify it works)
    output.copy_(input.slice(0, 0, 1));

    return output;
}

// Test function that creates a tensor on CUDA device
torch::Tensor create_cuda_tensor(int size) {
    // Test torch::kCUDA - should be ported to c10::DeviceType::PrivateUse1
    // Use c10::Device constructor which accepts DeviceType
    auto device = c10::Device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
    auto tensor = torch::ones({size}, options);

    // Verify device type
    TORCH_CHECK(
        tensor.device().type() == at::DeviceType::CUDA,
        "Created tensor should be on CUDA device"
    );

    return tensor;
}

// Test function that returns device type information
std::tuple<bool, bool, bool> get_device_info(torch::Tensor input) {
    // Check if tensor is on CUDA using different methods
    bool check1 = input.device().type() == at::DeviceType::CUDA;
    bool check2 = input.device().type() == c10::DeviceType::CUDA;
    bool check3 = input.is_cuda();

    return std::make_tuple(check1, check2, check3);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("check_device_type", &check_device_type,
          "Check device type using various CUDA symbols");
    m.def("create_cuda_tensor", &create_cuda_tensor,
          "Create a tensor on CUDA device");
    m.def("get_device_info", &get_device_info,
          "Get device type info using different check methods");
}
