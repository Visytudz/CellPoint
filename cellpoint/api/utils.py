"""Utility functions for data preparation"""

import torch
import numpy as np
from typing import Union, List

from cellpoint.utils.io import load_ply
from cellpoint.utils.misc import batch_normalize_to_unit_sphere_torch


def prepare_input(
    data: Union[str, np.ndarray, torch.Tensor],
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Prepare input data for inference.

    Parameters
    ----------
    data : Union[str, np.ndarray, torch.Tensor]
        Input point cloud (file path, numpy array, or tensor)
    device : torch.device
        Target device
    normalize : bool
        Whether to normalize to unit sphere

    Returns
    -------
    torch.Tensor
        Preprocessed point cloud (1, N, 3)
    """
    # Load from file if path is provided
    if isinstance(data, str):
        data = load_ply(data)

    # Convert to tensor
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    # Ensure 3D tensor (add batch dimension if needed)
    if data.dim() == 2:
        data = data.unsqueeze(0)  # (N, 3) -> (1, N, 3)

    # Move to device
    data = data.to(device)

    # Normalize to unit sphere
    if normalize:
        data, _ = batch_normalize_to_unit_sphere_torch(data)

    return data


def prepare_batch_input(
    data_list: List, device: torch.device, normalize: bool = True
) -> torch.Tensor:
    """
    Prepare batch of inputs for inference.

    Parameters
    ----------
    data_list : List
        List of point clouds
    device : torch.device
        Target device
    normalize : bool
        Whether to normalize to unit sphere

    Returns
    -------
    torch.Tensor
        Batch of preprocessed point clouds (B, N, 3)
    """
    batch_data = [prepare_input(d, device, normalize) for d in data_list]
    return torch.cat(batch_data, dim=0)
