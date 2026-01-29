// This is the CUDA version of the kernel.
// When both kernel.cu and kernel.mu exist, the .mu file should take precedence.
// This file should NOT be used - if it is, the test will fail because
// the magic number returned is different.

#include <torch/extension.h>
#include <cuda_runtime.h>

// Return 42 (wrong value - .mu version returns 123)
int get_magic_number() {
    return 42;  // CUDA version returns 42
}

