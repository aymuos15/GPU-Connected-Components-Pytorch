"""GPU-accelerated connected components using cuCIM."""

import torch
import cupy as cp
from cucim.skimage import measure as cucim_measure


def gpu_connected_components(img, connectivity=None):
    """
    Compute connected components on GPU using cuCIM.

    Args:
        img: Input tensor/array (numpy or torch)
        connectivity: Connectivity for labeling (default: None)

    Returns:
        Tuple of (labeled_tensor, num_features)
    """
    # Determine device based on input type
    if isinstance(img, torch.Tensor):
        device = img.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    img_cupy = cp.asarray(img)
    labeled_img, num_features = cucim_measure.label(
        img_cupy, connectivity=connectivity, return_num=True
    )
    labeled_img_torch = torch.as_tensor(labeled_img, device=device)

    return labeled_img_torch, num_features
