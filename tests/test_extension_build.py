"""
Tests for building CUDA extensions with torchada.

These tests verify that CUDAExtension and BuildExtension work correctly
on MUSA platforms, including source code porting.

The key point is that after importing torchada, the standard torch imports
should work transparently:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

# Import torchada first to apply patches
import torchada  # noqa: F401

# Get the path to the test CUDA source file
CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")
VECTOR_ADD_CU = os.path.join(CSRC_DIR, "vector_add.cu")


class TestExtensionBuildSetup:
    """Test extension build setup and configuration."""

    def test_vector_add_cu_exists(self):
        """Test that vector_add.cu test file exists."""
        assert os.path.exists(VECTOR_ADD_CU), f"Test file not found: {VECTOR_ADD_CU}"

    def test_can_create_setup_py(self):
        """Test that we can create a setup.py for the extension."""
        # Use standard torch imports - torchada patches make them work on MUSA

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the source file
            shutil.copy(VECTOR_ADD_CU, tmpdir)

            # Create setup.py content - uses standard torch imports
            setup_content = f"""
import torchada  # noqa: F401 - Apply MUSA patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_vector_add",
    ext_modules=[
        CUDAExtension(
            name="test_vector_add",
            sources=["vector_add.cu"],
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            assert os.path.exists(setup_path)

            # Verify the setup.py is valid Python
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", setup_path],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Setup.py syntax error: {result.stderr}"


def _is_gpu_available():
    """Check if CUDA or MUSA GPU is available."""
    import torch

    if torch.cuda.is_available():
        return True
    if hasattr(torch, "musa") and torch.musa.is_available():
        return True
    return False


@pytest.mark.skipif(
    not os.environ.get("TORCHADA_TEST_BUILD", "0") == "1",
    reason="Extension build tests are slow; set TORCHADA_TEST_BUILD=1 to run",
)
class TestExtensionBuild:
    """Test actual extension building (slow, opt-in)."""

    def test_build_vector_add_extension(self):
        """Test building the vector_add extension."""
        if not _is_gpu_available():
            pytest.skip("CUDA/MUSA not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the source file
            shutil.copy(VECTOR_ADD_CU, tmpdir)

            # Create setup.py using standard torch imports
            setup_content = """
import torchada  # noqa: F401 - Apply MUSA patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_vector_add",
    ext_modules=[
        CUDAExtension(
            name="test_vector_add",
            sources=["vector_add.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build the extension
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

            assert result.returncode == 0, f"Build failed: {result.stderr}"

            # Check that the extension was built
            ext_files = [f for f in os.listdir(tmpdir) if f.endswith(".so") or f.endswith(".pyd")]
            assert len(ext_files) > 0, "No extension file was built"

    def test_run_vector_add_extension(self):
        """Test running the vector_add extension after building."""
        import torch

        if not _is_gpu_available():
            pytest.skip("CUDA/MUSA not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the source file
            shutil.copy(VECTOR_ADD_CU, tmpdir)

            # Create setup.py using standard torch imports
            setup_content = """
import torchada  # noqa: F401 - Apply MUSA patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_vector_add",
    ext_modules=[
        CUDAExtension(
            name="test_vector_add",
            sources=["vector_add.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build the extension
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert result.returncode == 0, f"Build failed: {result.stderr}"

            # Add tmpdir to Python path and import the extension
            sys.path.insert(0, tmpdir)
            try:
                import test_vector_add

                try:
                    # Create test tensors
                    a = torch.randn(1000, device="cuda")
                    b = torch.randn(1000, device="cuda")

                    # Run vector add
                    c = test_vector_add.vector_add(a, b)

                    # Verify result
                    expected = a + b
                    assert torch.allclose(c, expected), "Vector add result incorrect"
                except RuntimeError as e:
                    # Skip if GPU is not working or kernel was compiled for different architecture
                    if "invalid device function" in str(e):
                        pytest.skip(
                            "GPU not available or kernel compiled for different architecture"
                        )
                    raise
            finally:
                sys.path.remove(tmpdir)


# Path to mixed sources test directory
MIXED_SOURCES_DIR = os.path.join(CSRC_DIR, "mixed_sources")


@pytest.mark.skipif(
    not os.environ.get("TORCHADA_TEST_BUILD", "0") == "1",
    reason="Extension build tests are slow; set TORCHADA_TEST_BUILD=1 to run",
)
class TestMixedSourcesBuild:
    """
    Test building extensions with mixed source types (.cu, .cuh, .mu, .muh, .cpp).

    This tests the fix for the issue where .mu files required manually specifying
    the ported path (e.g., csrc_musa/foo.mu instead of csrc/foo.mu).
    """

    def test_mixed_sources_dir_exists(self):
        """Test that mixed_sources test directory exists."""
        assert os.path.isdir(MIXED_SOURCES_DIR), f"Test dir not found: {MIXED_SOURCES_DIR}"

    def test_all_source_files_exist(self):
        """Test that all mixed source files exist."""
        expected_files = [
            "utils.cuh",  # CUDA header
            "add_kernel.cu",  # CUDA kernel
            "utils.muh",  # MUSA header (already ported)
            "mul_kernel.mu",  # MUSA kernel (already ported)
            "bindings.cpp",  # C++ bindings
        ]
        for f in expected_files:
            path = os.path.join(MIXED_SOURCES_DIR, f)
            assert os.path.exists(path), f"Source file not found: {path}"

    def test_build_mixed_sources_extension(self):
        """
        Test building an extension with mixed .cu/.cuh/.mu/.muh/.cpp sources.

        This is the key e2e test that verifies:
        1. .cu files are ported to .mu in the _musa directory
        2. .cuh files are ported to .muh in the _musa directory
        3. .mu files that don't exist at original path are found in _musa directory
        4. .muh files are handled correctly
        5. .cpp files are ported for CUDA symbol translation
        """
        if not _is_gpu_available():
            pytest.skip("CUDA/MUSA not available")

        if not torchada.is_musa_platform():
            pytest.skip("Mixed sources test only applicable on MUSA platform")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy all source files to temp directory, preserving structure
            src_dir = os.path.join(tmpdir, "csrc")
            shutil.copytree(MIXED_SOURCES_DIR, src_dir)

            # Create setup.py that uses CUDA-style paths for .mu files
            # This tests the fix: users can specify csrc/mul_kernel.mu
            # and torchada will find it in csrc_musa/mul_kernel.mu after porting
            setup_content = """
import torchada  # noqa: F401 - Apply MUSA patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_mixed_sources",
    ext_modules=[
        CUDAExtension(
            name="test_mixed_sources",
            sources=[
                "csrc/bindings.cpp",      # C++ file with CUDA symbols
                "csrc/add_kernel.cu",     # CUDA kernel -> ported to csrc_musa/add_kernel.mu
                "csrc/mul_kernel.mu",     # MUSA kernel -> found in csrc_musa/mul_kernel.mu
            ],
            include_dirs=["csrc"],        # Include dir for headers
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build the extension
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

            assert result.returncode == 0, f"Build failed: {result.stderr}"

            # Check that the extension was built
            ext_files = [f for f in os.listdir(tmpdir) if f.endswith(".so") or f.endswith(".pyd")]
            assert len(ext_files) > 0, "No extension file was built"

            # Verify ported directory was created
            ported_dir = os.path.join(tmpdir, "csrc_musa")
            assert os.path.isdir(ported_dir), "Ported directory csrc_musa was not created"

            # Verify ported files exist
            assert os.path.exists(
                os.path.join(ported_dir, "add_kernel.mu")
            ), "add_kernel.cu was not ported to add_kernel.mu"
            assert os.path.exists(
                os.path.join(ported_dir, "utils.muh")
            ), "utils.cuh was not ported to utils.muh"

    def test_run_mixed_sources_extension(self):
        """Test running the mixed sources extension after building."""
        import torch

        if not _is_gpu_available():
            pytest.skip("CUDA/MUSA not available")

        if not torchada.is_musa_platform():
            pytest.skip("Mixed sources test only applicable on MUSA platform")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy all source files
            src_dir = os.path.join(tmpdir, "csrc")
            shutil.copytree(MIXED_SOURCES_DIR, src_dir)

            # Create setup.py
            setup_content = """
import torchada  # noqa: F401
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_mixed_sources",
    ext_modules=[
        CUDAExtension(
            name="test_mixed_sources",
            sources=[
                "csrc/bindings.cpp",
                "csrc/add_kernel.cu",
                "csrc/mul_kernel.mu",
            ],
            include_dirs=["csrc"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert result.returncode == 0, f"Build failed: {result.stderr}"

            # Import and test
            sys.path.insert(0, tmpdir)
            try:
                import test_mixed_sources

                try:
                    # Test add function (from .cu file)
                    a = torch.randn(1000, device="cuda")
                    b = torch.randn(1000, device="cuda")
                    c = test_mixed_sources.add(a, b)
                    expected = a + b
                    assert torch.allclose(c, expected), "Add result incorrect"

                    # Test mul function (from .mu file)
                    d = test_mixed_sources.mul(a, b)
                    expected_mul = a * b
                    assert torch.allclose(d, expected_mul), "Mul result incorrect"

                except RuntimeError as e:
                    if "invalid device function" in str(e):
                        pytest.skip("Kernel compiled for different architecture")
                    raise
            finally:
                sys.path.remove(tmpdir)


# Path to same_name test directory (tests .cu and .mu with same base name)
SAME_NAME_DIR = os.path.join(CSRC_DIR, "same_name")


@pytest.mark.skipif(
    not os.environ.get("TORCHADA_TEST_BUILD", "0") == "1",
    reason="Extension build tests are slow; set TORCHADA_TEST_BUILD=1 to run",
)
class TestSameNameFilePrecedence:
    """
    Test that .mu files take precedence over .cu files when both exist.

    This tests the fix for the issue where SimplePorting's file processing
    order is non-deterministic, causing either the ported .cu or original .mu
    to end up in the _musa directory depending on which is processed last.

    The fix ensures that original .mu/.muh files are always copied after
    porting, so they take precedence over auto-ported .cu/.cuh files.
    """

    def test_same_name_dir_exists(self):
        """Test that same_name test directory exists."""
        assert os.path.isdir(SAME_NAME_DIR), f"Test dir not found: {SAME_NAME_DIR}"

    def test_both_cu_and_mu_exist(self):
        """Test that both kernel.cu and kernel.mu exist."""
        cu_path = os.path.join(SAME_NAME_DIR, "kernel.cu")
        mu_path = os.path.join(SAME_NAME_DIR, "kernel.mu")
        assert os.path.exists(cu_path), f"kernel.cu not found: {cu_path}"
        assert os.path.exists(mu_path), f"kernel.mu not found: {mu_path}"

    def test_mu_file_takes_precedence(self):
        """
        Test that .mu file takes precedence when both .cu and .mu exist.

        This is the key test: after porting, the csrc_musa/kernel.mu file
        should contain the content from the original kernel.mu (magic number 123),
        not the ported kernel.cu (magic number 42).
        """
        if not _is_gpu_available():
            pytest.skip("CUDA/MUSA not available")

        if not torchada.is_musa_platform():
            pytest.skip("Same name precedence test only applicable on MUSA platform")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create csrc directory and copy source files
            csrc_dir = os.path.join(tmpdir, "csrc")
            shutil.copytree(SAME_NAME_DIR, csrc_dir)

            # Create setup.py that only specifies the .cu file
            # (simulating user who wants to build CUDA code)
            setup_content = f"""
import torchada  # noqa: F401 - Apply MUSA patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_same_name",
    ext_modules=[
        CUDAExtension(
            name="test_same_name",
            sources=[
                "csrc/bindings.cpp",
                "csrc/kernel.cu",  # .mu file exists too - should take precedence
            ],
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert result.returncode == 0, f"Build failed: {result.stderr}"

            # Check that csrc_musa/kernel.mu contains the MUSA version (magic 123)
            ported_mu_path = os.path.join(tmpdir, "csrc_musa", "kernel.mu")
            assert os.path.exists(ported_mu_path), f"Ported file not found: {ported_mu_path}"

            with open(ported_mu_path, "r") as f:
                content = f.read()
                # The MUSA version returns 123, CUDA version returns 42
                assert "return 123" in content, (
                    f".mu file should take precedence but found ported .cu content. "
                    f"Expected 'return 123' but got:\n{content}"
                )
                assert "return 42" not in content, (
                    f"Found ported .cu content instead of original .mu. " f"Content:\n{content}"
                )
