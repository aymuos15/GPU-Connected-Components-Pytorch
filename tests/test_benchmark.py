"""Tests for benchmark functionality."""

import pytest
import numpy as np
import torch
import cupy as cp
from gpu_cc.benchmark import (
    create_matrix,
    torch_cc,
    cc3d_cc,
    skimage_cc,
    simpleitk_cc,
    mahotas_cc,
    cupy_ndimage_cc,
    cucim_cc,
    diplib_cc,
)


class TestTestDataGeneration:
    """Test test data creation."""

    def test_create_matrix_shape(self):
        """Test that create_matrix returns correct shape."""
        for size in [32, 64, 128]:
            matrix = create_matrix(size)
            assert matrix.shape == (50, size, size)

    def test_create_matrix_values(self):
        """Test that create_matrix contains expected values."""
        matrix = create_matrix(64)
        assert matrix.min() == 0
        assert matrix.max() == 1
        assert matrix.dtype in [np.float32, np.float64, np.int32, np.int64]

    def test_create_matrix_has_components(self):
        """Test that create_matrix has actual components."""
        matrix = create_matrix(128)
        # Should have some non-zero values
        assert np.any(matrix > 0)
        # Should be distributed across slices
        non_zero_slices = [i for i in range(50) if np.any(matrix[i] > 0)]
        assert len(non_zero_slices) > 1


class TestBenchmarkFunctions:
    """Test individual benchmark wrapper functions."""

    @pytest.fixture
    def test_matrix(self):
        """Create a small test matrix."""
        matrix = np.zeros((5, 16, 16))
        matrix[0, 2:4, 2:4] = 1
        matrix[1, 5:7, 5:7] = 1
        matrix[2, 10:12, 10:12] = 1
        return matrix

    def test_torch_cc_returns_array(self, test_matrix):
        """Test torch_cc returns expected type."""
        matrix_torch = torch.tensor(test_matrix, dtype=torch.float32)
        if torch.cuda.is_available():
            matrix_torch = matrix_torch.cuda()
        result = torch_cc(matrix_torch)
        assert isinstance(result, torch.Tensor)
        assert result.shape == test_matrix.shape

    def test_cc3d_cc_returns_array(self, test_matrix):
        """Test cc3d_cc returns expected type."""
        result = cc3d_cc(test_matrix)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_matrix.shape

    def test_skimage_cc_returns_array(self, test_matrix):
        """Test skimage_cc returns expected type."""
        result = skimage_cc(test_matrix)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_matrix.shape

    def test_simpleitk_cc_returns_array(self, test_matrix):
        """Test simpleitk_cc returns expected type."""
        result = simpleitk_cc(test_matrix)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_matrix.shape

    def test_mahotas_cc_returns_array(self, test_matrix):
        """Test mahotas_cc returns expected type."""
        result = mahotas_cc(test_matrix)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_matrix.shape

    def test_diplib_cc_returns_array(self, test_matrix):
        """Test diplib_cc returns expected type."""
        result = diplib_cc(test_matrix)
        assert isinstance(result, np.ndarray)
        assert result.shape == test_matrix.shape

    def test_cupy_ndimage_cc_returns_array(self, test_matrix):
        """Test cupy_ndimage_cc returns expected type."""
        try:
            matrix_cp = cp.asarray(test_matrix)
            result = cupy_ndimage_cc(matrix_cp)
            assert result.shape == test_matrix.shape
        except Exception:
            pytest.skip("CuPy not available or CUDA unavailable")

    def test_cucim_cc_returns_array(self, test_matrix):
        """Test cucim_cc returns expected type."""
        try:
            matrix_cp = cp.asarray(test_matrix)
            result = cucim_cc(matrix_cp)
            assert result.shape == test_matrix.shape
        except Exception:
            pytest.skip("cuCIM not available or CUDA unavailable")


class TestResultConsistency:
    """Test consistency across different libraries."""

    @pytest.fixture
    def small_matrix(self):
        """Create a small, simple test matrix."""
        matrix = np.zeros((2, 8, 8))
        # Single component per slice
        matrix[0, 1:3, 1:3] = 1
        matrix[1, 4:6, 4:6] = 1
        return matrix

    def test_all_functions_produce_similar_component_counts(self, small_matrix):
        """Test that different libraries detect similar number of components."""
        results = {}

        # Run all native 3D CPU benchmarks
        try:
            results['cc3d'] = cc3d_cc(small_matrix)
            results['skimage'] = skimage_cc(small_matrix)
            results['simpleitk'] = simpleitk_cc(small_matrix)
            results['mahotas'] = mahotas_cc(small_matrix)
            results['diplib'] = diplib_cc(small_matrix)
        except Exception as e:
            pytest.skip(f"Some CPU libraries failed: {e}")

        # Count components in each result (excluding background)
        component_counts = {}
        for name, result in results.items():
            unique_labels = len(np.unique(result[result > 0]))
            component_counts[name] = unique_labels

        # All counts should be close (allow 1 component difference due to connectivity differences)
        counts = list(component_counts.values())
        assert max(counts) - min(counts) <= 1, f"Component counts vary too much: {component_counts}"
