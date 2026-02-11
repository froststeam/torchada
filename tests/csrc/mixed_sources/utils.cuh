/*
 * CUDA header file for testing mixed source extension building.
 * This file uses CUDA syntax and should be ported to MUSA.
 */

#pragma once

#include <cuda_runtime.h>

namespace test_utils {

// Simple device function for element-wise operations
template <typename T>
__device__ __forceinline__ T add_elements(T a, T b) {
    return a + b;
}

template <typename T>
__device__ __forceinline__ T mul_elements(T a, T b) {
    return a * b;
}

// Check CUDA errors
inline void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace test_utils
