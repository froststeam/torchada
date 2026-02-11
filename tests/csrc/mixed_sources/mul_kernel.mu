/*
 * MUSA kernel for multiplication - tests .mu file handling.
 * This file is already in MUSA format (no porting needed).
 */

#include <torch/extension.h>
#include <musa.h>
#include <musa_runtime.h>
#include "torch_musa/csrc/core/MUSAStream.h"
#include "utils.muh"

__global__ void mul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = test_utils::mul_elements(a[idx], b[idx]);
    }
}

torch::Tensor mul_musa(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_privateuseone(), "a must be a MUSA tensor");
    TORCH_CHECK(b.device().is_privateuseone(), "b must be a MUSA tensor");

    auto c = torch::empty_like(a);
    int n = a.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Use c10::musa namespace for MUSA stream
    musaStream_t stream = c10::musa::getCurrentMUSAStream();
    mul_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    musaError_t err = musaGetLastError();
    TORCH_CHECK(err == musaSuccess, "MUSA kernel error: ", musaGetErrorString(err));

    return c;
}
