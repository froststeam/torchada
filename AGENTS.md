# AGENTS.md

## Project Overview

**torchada** is a CUDA-to-MUSA compatibility adapter for PyTorch on Moore Threads GPUs. It enables PyTorch code written for NVIDIA CUDA GPUs to run on Moore Threads MUSA GPUs without code changes.

**Key Design Principle**: Developers import `torchada` once, then use standard `torch.cuda.*` APIs normally. All MUSA handling is invisible to users.

## Architecture

- **Runtime patching system**: Uses `@patch_function` decorator registry in `src/torchada/_patch.py`
- **Platform detection**: `is_musa_platform()`, `is_cuda_platform()` in `src/torchada/_platform.py`
- **Patches applied automatically** on `import torchada`
- **MUSA tensors dispatch to `PrivateUse1`** backend (not CUDA)

## File Structure

```
src/torchada/
├── __init__.py          # Entry point, auto-applies patches
├── _patch.py            # All patching logic (~1100 lines)
├── _platform.py         # Platform detection utilities
├── _mapping.py          # CUDA→MUSA symbol mappings for C++ extensions
├── cuda/                # CUDA module compatibility
└── utils/cpp_extension.py  # CUDAExtension wrapper
tests/
├── conftest.py          # Pytest fixtures and markers
├── test_cuda_patching.py # Main test file (~1270 lines)
└── ...
```

## Build and Install

```bash
# Development install
pip install -e .

# With dev dependencies (pytest, black, isort, mypy)
pip install -e .[dev]
```

## Code Style

- **Formatter**: black with line-length 100
- **Import sorting**: isort with black profile
- **Python version**: >=3.8
- **Pre-commit hooks**: Configured in `.pre-commit-config.yaml`

```bash
# Set up pre-commit hooks (one-time)
pre-commit install

# Run all hooks on all files
pre-commit run --all-files

# Format code manually
isort src/ tests/ && black src/ tests/

# In Docker containers, preserve file ownership
docker exec -w /ws <container> bash -c "isort src/ tests/ && black src/ tests/ && chown -R 1000:1000 src/ tests/"
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test class
pytest tests/test_cuda_patching.py::TestLibraryImpl -v

# Run with short traceback
pytest tests/ --tb=short
```

**Test Markers** (defined in `conftest.py`):
- `@pytest.mark.musa` - Requires MUSA platform
- `@pytest.mark.cuda` - Requires CUDA platform
- `@pytest.mark.gpu` - Requires any GPU
- `@pytest.mark.slow` - Slow tests

**Docker Containers for Testing**:
- `yeahdongcn` - torch_musa 2.7.0
- `yeahdongcn1` - torch_musa 2.7.1

```bash
# Run tests in Docker
docker exec -w /ws yeahdongcn1 python -m pytest tests/ --tb=short
```

## Adding New Patches

1. Add patch function in `src/torchada/_patch.py`:

```python
@patch_function
@requires_import("torch_musa")
def _patch_feature_name():
    """Docstring explaining what this patch does."""
    # Patch implementation
    original_func = torch.module.func
    def patched_func(*args, **kwargs):
        # Translation logic
        return original_func(*args, **kwargs)
    torch.module.func = patched_func
```

2. Add tests in `tests/test_cuda_patching.py`:

```python
class TestFeatureName:
    def test_feature_works(self):
        import torchada
        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")
        # Test implementation
```

3. Update documentation in `README.md` and `README_CN.md`

## Critical Constraints

1. **Never patch** `torch.cuda.is_available()` or `torch.version.cuda` - downstream projects use these for platform detection
2. **Import order matters**: `import torchada` must come before other torch imports in user code
3. **MUSA tensors use `PrivateUse1` dispatch key**, not `CUDA` - always translate dispatch keys
4. **Keep file ownership 1000:1000** when running formatters in Docker

## Common Patterns

**Translating dispatch keys**:
```python
if dispatch_key == "CUDA":
    dispatch_key = "PrivateUse1"
```

**Platform-specific tests**:
```python
if not torchada.is_musa_platform():
    pytest.skip("Only applicable on MUSA platform")
```

**Unique library names in tests** (avoid conflicts):
```python
import uuid
lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
```

## Performance Benchmarking

torchada uses aggressive caching to minimize runtime overhead. Performance is tracked across versions.

**Benchmark files**:
- `benchmarks/benchmark_overhead.py` - Benchmark script
- `benchmarks/benchmark_history.json` - Historical results

**Running benchmarks**:
```bash
# Run benchmarks (print only)
docker exec -w /ws yeahdongcn1 python benchmarks/benchmark_overhead.py

# Run and save results to history (do this before releasing new versions)
docker exec -w /ws yeahdongcn1 python benchmarks/benchmark_overhead.py --save
```

**Performance targets**:
- Fast operations (<200ns): `torch.cuda.device_count()`, `torch.cuda.Stream`, `torch.cuda.Event`, `_translate_device()`, `torch.backends.cuda.is_built()`
- Medium operations (200-800ns): Operations with inherent costs (runtime calls, object creation) that cannot be optimized further

**When to run benchmarks**:
1. After adding new patches that affect hot paths
2. Before releasing a new version (use `--save` to record results)
3. When optimizing existing patches

**Optimization techniques used**:
- Attribute caching in `__dict__` to bypass `__getattr__` on subsequent accesses
- Platform check caching (global variable `_is_musa_platform_cached`)
- String translation caching (`_device_str_cache`)
- Closure variable caching for wrapper functions

## Security Considerations

- All patches are applied at import time via `apply_patches()`
- Patches only affect torch APIs, not system resources
- No network access or file system modifications
- C++ extension building uses standard torch/setuptools mechanisms
