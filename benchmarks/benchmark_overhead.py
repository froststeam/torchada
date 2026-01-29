#!/usr/bin/env python
"""
Benchmark to measure torchada patching overhead.

This script measures the overhead of torchada's runtime patching for common
torch.cuda.* API calls that are frequently used in sglang and similar projects.
"""

import time
import statistics


def benchmark_function(func, name, iterations=100000, warmup=1000):
    """Benchmark a function and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append(end - start)

    return {
        "name": name,
        "iterations": iterations,
        "mean_ns": statistics.mean(times),
        "median_ns": statistics.median(times),
        "stdev_ns": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ns": min(times),
        "max_ns": max(times),
    }


def run_comprehensive_benchmarks():
    """Run comprehensive benchmarks for all wrapper classes."""
    import torch
    import torchada

    results = []

    print("=" * 80)
    print("COMPREHENSIVE TORCHADA OVERHEAD ANALYSIS")
    print("=" * 80)
    print(f"Platform: {'MUSA' if torchada.is_musa_platform() else 'CUDA'}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    # === _CudaModuleWrapper (torch.cuda.*) ===
    print("1. _CudaModuleWrapper (torch.cuda.* access)")
    print("-" * 60)

    results.append(benchmark_function(
        lambda: torch.cuda.device_count(),
        "torch.cuda.device_count()"
    ))

    if torch.cuda.device_count() > 0:
        results.append(benchmark_function(
            lambda: torch.cuda.current_device(),
            "torch.cuda.current_device()"
        ))

    results.append(benchmark_function(
        lambda: torch.cuda.is_available(),
        "torch.cuda.is_available() [NOT redirected]"
    ))

    results.append(benchmark_function(
        lambda: torch.cuda.Stream,
        "torch.cuda.Stream (attr)"
    ))

    results.append(benchmark_function(
        lambda: torch.cuda.Event,
        "torch.cuda.Event (attr)"
    ))

    # === _CudartWrapper (torch.cuda.cudart()) ===
    print("\n2. _CudartWrapper (torch.cuda.cudart())")
    print("-" * 60)

    try:
        cudart = torch.cuda.cudart()
        # First access (uncached)
        results.append(benchmark_function(
            lambda: cudart.cudaHostRegister,
            "cudart.cudaHostRegister (attr)"
        ))
    except Exception as e:
        print(f"  Skipping cudart benchmarks: {e}")

    # === DeviceFactoryWrapper (torch.device) ===
    print("\n3. DeviceFactoryWrapper (torch.device)")
    print("-" * 60)

    results.append(benchmark_function(
        lambda: torch.device("cuda"),
        "torch.device('cuda')"
    ))

    results.append(benchmark_function(
        lambda: torch.device("cuda:0"),
        "torch.device('cuda:0')"
    ))

    results.append(benchmark_function(
        lambda: torch.device("cuda", 0),
        "torch.device('cuda', 0)"
    ))

    # === tensor.is_cuda property ===
    print("\n4. Tensor.is_cuda property")
    print("-" * 60)

    t_cpu = torch.zeros(1)
    results.append(benchmark_function(
        lambda: t_cpu.is_cuda,
        "cpu_tensor.is_cuda (property)"
    ))

    if torch.cuda.device_count() > 0:
        try:
            t_gpu = torch.zeros(1, device="cuda")
            results.append(benchmark_function(
                lambda: t_gpu.is_cuda,
                "gpu_tensor.is_cuda (property)"
            ))
        except RuntimeError as e:
            print(f"  Skipping GPU tensor: {e}")

    # === _translate_device function (internal) ===
    print("\n5. _translate_device (internal)")
    print("-" * 60)

    from torchada._patch import _translate_device
    results.append(benchmark_function(
        lambda: _translate_device("cuda"),
        "_translate_device('cuda')"
    ))

    results.append(benchmark_function(
        lambda: _translate_device("cuda:0"),
        "_translate_device('cuda:0')"
    ))

    # === torch.backends.cuda ===
    print("\n6. torch.backends.cuda")
    print("-" * 60)

    results.append(benchmark_function(
        lambda: torch.backends.cuda.is_built(),
        "torch.backends.cuda.is_built()"
    ))

    # === Print Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Operation':<45} {'Mean (ns)':<12} {'Median (ns)':<12} {'Min (ns)':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<45} {r['mean_ns']:<12.1f} {r['median_ns']:<12.1f} {r['min_ns']:<10}")

    print()
    print("Analysis:")
    print("-" * 40)

    # Categorize results
    fast = [r for r in results if r['mean_ns'] < 200]
    medium = [r for r in results if 200 <= r['mean_ns'] < 800]
    slow = [r for r in results if r['mean_ns'] >= 800]

    if fast:
        print(f"✅ Fast (<200ns): {len(fast)} operations - OPTIMIZED")
        for r in fast:
            print(f"   - {r['name']}: {r['mean_ns']:.0f}ns")

    if medium:
        print(f"⚠️  Medium (200-800ns): {len(medium)} operations")
        for r in medium:
            print(f"   - {r['name']}: {r['mean_ns']:.0f}ns")

    if slow:
        print(f"❌ Slow (>800ns): {len(slow)} operations - NEEDS OPTIMIZATION?")
        for r in slow:
            print(f"   - {r['name']}: {r['mean_ns']:.0f}ns")

    print()
    print("Note: 1 microsecond = 1000 nanoseconds")
    print("      Typical GPU kernel launch: 5,000-20,000 ns")
    print()

    return results


if __name__ == "__main__":
    run_comprehensive_benchmarks()

