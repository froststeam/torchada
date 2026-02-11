"""
Simple demo showing torch.profiler.ProfilerActivity.CUDA working on MUSA via torchada.

Usage:
    python examples/profiler_demo.py
"""

import torch


def main():
    print("=== torchada Profiler Demo ===\n")

    # Show that ProfilerActivity.CUDA is available
    print(f"ProfilerActivity.CUDA: {torch.profiler.ProfilerActivity.CUDA}")
    print(f"ProfilerActivity.PrivateUse1: {torch.profiler.ProfilerActivity.PrivateUse1}")

    # Use standard CUDA activity - torchada translates to PrivateUse1 on MUSA
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,  # Works on MUSA!
    ]

    print(f"\nUsing activities: {activities}")

    # Create some tensors and run operations
    device = "cuda" if torch.cuda.is_available() else "musa" if torch.musa.is_available() else "cpu"
    print(f"Device: {device}")

    x = torch.randn(1000, 1000, device=device)

    # Profile with CUDA activity
    print("\nRunning profiler...")
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(10):
            y = torch.matmul(x, x)
            z = y.sum()

    # Print profiler results
    print("\n=== Profiler Results ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
