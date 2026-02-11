#!/usr/bin/env python3
"""
Model training example using torchada.

This example demonstrates training a simple neural network
that works on any supported GPU platform (CUDA or MUSA) transparently.

Usage:
    Just import torchada at the top of your script, then use
    torch.cuda.* APIs as you normally would.

Platform Detection:
    - CUDA: torch.version.cuda is not None
    - MUSA: hasattr(torch.version, 'musa') and torch.version.musa is not None
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Use standard torch.cuda imports - they work on any supported GPU
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

import torchada  # noqa: F401 - Import first to apply patches (must be before torch.cuda usage)


def _is_cuda():
    """Check if running on CUDA platform."""
    return torch.version.cuda is not None


def _is_musa():
    """Check if running on MUSA platform."""
    return hasattr(torch.version, "musa") and torch.version.musa is not None


def is_gpu_available():
    """Check if any GPU (CUDA or MUSA) is available."""
    return _is_cuda() or _is_musa()


def get_platform_name():
    """Get the platform name."""
    if _is_cuda():
        return "CUDA"
    elif _is_musa():
        return "MUSA"
    return "CPU"


class SimpleModel(nn.Module):
    """A simple MLP model."""

    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, scaler, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    batches_processed = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        # Move to GPU
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        try:
            if use_amp:
                # Mixed precision training
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            batches_processed += 1
        except RuntimeError as e:
            # Handle driver/hardware issues gracefully
            if batch_idx == 0:
                print(f"  Training skipped (driver/hardware issue): {type(e).__name__}")
            return 0.0

    return total_loss / max(batches_processed, 1)


def main():
    # Check GPU availability (works on both CUDA and MUSA)
    gpu_available = is_gpu_available()
    device = "cuda" if gpu_available else "cpu"
    print(f"Using device: {device}")
    print(f"Platform: {get_platform_name()}")

    if gpu_available:
        print(f"Device name: {torch.cuda.get_device_name()}")

    # Create dummy dataset
    num_samples = 10000
    input_size = 784
    num_classes = 10

    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create model, move to GPU (works on both CUDA and MUSA)
    model = SimpleModel(input_size=input_size, output_size=num_classes)
    if gpu_available:
        model = model.cuda()

    print(f"Model device: {next(model.parameters()).device}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # Train for a few epochs
    num_epochs = 3
    use_amp = gpu_available  # Use AMP only on GPU

    print(f"\nTraining for {num_epochs} epochs (AMP: {use_amp})...")

    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, criterion, optimizer, scaler, use_amp)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        if gpu_available:
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print("\nTraining complete!")

    # Cleanup
    if gpu_available:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
