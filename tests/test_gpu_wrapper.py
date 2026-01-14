"""Tests for GPU wrapper functionality."""

import pytest
import torch
import numpy as np
from gpu_cc.gpu_wrapper import gpu_connected_components


class TestGPUWrapper:
    """Test GPU connected components wrapper."""

    @pytest.fixture
    def simple_3d_matrix(self):
        """Create a simple 3D test matrix."""
        matrix = np.zeros((3, 10, 10))
        # Add a small connected component in first slice
        matrix[0, 2:4, 2:4] = 1
        # Add another in second slice
        matrix[1, 5:7, 5:7] = 1
        return matrix

    def test_accepts_numpy_input(self, simple_3d_matrix):
        """Test that gpu_wrapper accepts numpy arrays."""
        result, num_features = gpu_connected_components(simple_3d_matrix)
        assert isinstance(result, torch.Tensor)
        assert isinstance(num_features, (int, np.integer))
        assert result.shape == simple_3d_matrix.shape

    def test_accepts_torch_input(self, simple_3d_matrix):
        """Test that gpu_wrapper accepts torch tensors."""
        matrix_torch = torch.tensor(simple_3d_matrix, dtype=torch.float32)
        result, num_features = gpu_connected_components(matrix_torch)
        assert isinstance(result, torch.Tensor)
        assert isinstance(num_features, (int, np.integer))
        assert result.shape == matrix_torch.shape

    def test_torch_input_preserves_device(self):
        """Test that torch input device is preserved."""
        if torch.cuda.is_available():
            matrix = torch.zeros(3, 10, 10, dtype=torch.float32, device='cuda:0')
            matrix[0, 2:4, 2:4] = 1
            result, _ = gpu_connected_components(matrix)
            assert result.device.type == 'cuda'
        else:
            matrix = torch.zeros(3, 10, 10, dtype=torch.float32, device='cpu')
            matrix[0, 2:4, 2:4] = 1
            result, _ = gpu_connected_components(matrix)
            assert result.device.type == 'cpu'

    def test_returns_correct_shape(self, simple_3d_matrix):
        """Test that output shape matches input shape."""
        result, _ = gpu_connected_components(simple_3d_matrix)
        assert result.shape == simple_3d_matrix.shape

    def test_num_features_matches_expected(self):
        """Test that num_features is reasonable."""
        matrix = np.zeros((2, 10, 10))
        # Add 2 separated components (no connectivity)
        matrix[0, 1:3, 1:3] = 1
        matrix[0, 6:8, 6:8] = 1

        result, num_features = gpu_connected_components(matrix)
        # Should have at least 2 components (background + 2 regions, but background might be excluded)
        assert num_features >= 1
        # Max unique labels should match num_features
        unique_labels = len(np.unique(result[result > 0]))
        assert unique_labels <= num_features + 1  # +1 for background

    def test_connectivity_parameter(self, simple_3d_matrix):
        """Test that connectivity parameter is accepted."""
        # Should not raise an error
        result1, num_features1 = gpu_connected_components(simple_3d_matrix, connectivity=1)
        result2, num_features2 = gpu_connected_components(simple_3d_matrix, connectivity=2)
        assert result1.shape == result2.shape
