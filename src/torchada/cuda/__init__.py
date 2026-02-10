"""
torchada.cuda - CUDA-compatible API that works on both CUDA and MUSA platforms.

This module provides the same interface as torch.cuda but automatically
routes to torch.musa on Moore Threads hardware.

Note: After importing torchada, you can use standard torch.cuda APIs directly.
This module is provided for internal use and backwards compatibility.

Usage (preferred):
    import torchada  # Apply patches
    import torch

    # torch.cuda APIs work on MUSA after importing torchada
    torch.cuda.set_device(0)
    tensor = tensor.cuda()
"""

from typing import Optional, Union

from .._platform import Platform, detect_platform


def _get_backend():
    """Get the appropriate backend module (torch.cuda or torch.musa)."""
    platform = detect_platform()

    if platform == Platform.MUSA:
        import torch
        import torch_musa

        return torch.musa
    elif platform == Platform.CUDA:
        import torch

        return torch.cuda
    else:
        # Return torch.cuda for API compatibility, even if not available
        import torch

        return torch.cuda


# Core device functions
def is_available() -> bool:
    """Check if CUDA/MUSA is available."""
    return _get_backend().is_available()


def device_count() -> int:
    """Return the number of GPUs available."""
    backend = _get_backend()
    if hasattr(backend, "device_count"):
        return backend.device_count()
    return 0


def current_device() -> int:
    """Return the index of the currently selected device."""
    return _get_backend().current_device()


def set_device(device: Union[int, str, "torch.device"]) -> None:
    """Set the current device."""
    _get_backend().set_device(device)


def get_device_name(device: Optional[Union[int, str]] = None) -> str:
    """Get the name of a device."""
    return _get_backend().get_device_name(device)


def get_device_capability(device: Optional[Union[int, str]] = None) -> tuple:
    """Get the CUDA/MUSA compute capability of a device."""
    return _get_backend().get_device_capability(device)


def get_device_properties(device: Optional[Union[int, str]] = None):
    """Get the properties of a device."""
    return _get_backend().get_device_properties(device)


# Memory management
def memory_allocated(device: Optional[Union[int, str]] = None) -> int:
    """Return the current GPU memory occupied by tensors in bytes."""
    return _get_backend().memory_allocated(device)


def max_memory_allocated(device: Optional[Union[int, str]] = None) -> int:
    """Return the maximum GPU memory occupied by tensors in bytes."""
    return _get_backend().max_memory_allocated(device)


def memory_reserved(device: Optional[Union[int, str]] = None) -> int:
    """Return the current GPU memory managed by the caching allocator in bytes."""
    return _get_backend().memory_reserved(device)


def max_memory_reserved(device: Optional[Union[int, str]] = None) -> int:
    """Return the maximum GPU memory managed by the caching allocator in bytes."""
    return _get_backend().max_memory_reserved(device)


def memory_cached(device: Optional[Union[int, str]] = None) -> int:
    """Deprecated: Use memory_reserved instead."""
    return _get_backend().memory_cached(device)


def max_memory_cached(device: Optional[Union[int, str]] = None) -> int:
    """Deprecated: Use max_memory_reserved instead."""
    return _get_backend().max_memory_cached(device)


def empty_cache() -> None:
    """Release all unoccupied cached memory."""
    _get_backend().empty_cache()


def reset_peak_memory_stats(device: Optional[Union[int, str]] = None) -> None:
    """Reset the peak memory stats."""
    _get_backend().reset_peak_memory_stats(device)


def reset_max_memory_allocated(device: Optional[Union[int, str]] = None) -> None:
    """Reset the starting point in tracking maximum GPU memory occupied."""
    _get_backend().reset_max_memory_allocated(device)


def reset_max_memory_cached(device: Optional[Union[int, str]] = None) -> None:
    """Reset the starting point in tracking maximum GPU memory managed."""
    _get_backend().reset_max_memory_cached(device)


# Synchronization
def synchronize(device: Optional[Union[int, str]] = None) -> None:
    """Wait for all kernels in all streams on a device to complete."""
    _get_backend().synchronize(device)


# Stream and Event classes - will be set up dynamically
def _setup_stream_event_classes():
    """Set up Stream and Event classes from the backend."""
    backend = _get_backend()

    # These will be the actual classes from the backend
    global Stream, Event, current_stream, default_stream, stream

    Stream = backend.Stream if hasattr(backend, "Stream") else None
    Event = backend.Event if hasattr(backend, "Event") else None

    if hasattr(backend, "current_stream"):
        current_stream = backend.current_stream
    if hasattr(backend, "default_stream"):
        default_stream = backend.default_stream
    if hasattr(backend, "stream"):
        stream = backend.stream


# Initialize stream/event classes
try:
    _setup_stream_event_classes()
except:
    Stream = None
    Event = None
    current_stream = None
    default_stream = None
    stream = None
