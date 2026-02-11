// This is the MUSA version of the kernel (hand-written).
// When both kernel.cu and kernel.mu exist, this .mu file should take precedence.
// This file should be used - the test verifies by checking the magic number.

#include <torch/extension.h>
#include <musa_runtime.h>

// Return 123 (correct value - this is the hand-written MUSA version)
int get_magic_number() {
    return 123;  // MUSA version returns 123
}
