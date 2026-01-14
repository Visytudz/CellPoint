"""Point cloud reconstruction functionality"""

import torch
import numpy as np
from typing import Union, List, Dict

from .utils import prepare_input, prepare_batch_input


class ReconstructionEngine:
    """Point cloud reconstruction engine"""

    def __init__(self, model):
        """
        Initialize reconstruction engine.

        Parameters
        ----------
        model : InferenceModel
            Loaded inference model
        """
        self.model = model
        self.extractor = model.extractor
        self.decoder = model.decoder
        self.global_decoder = model.global_decoder
        self.view_generator = model.view_generator
        self.device = model.device

    @torch.no_grad()
    def self_reconstruct(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Perform self-reconstruction from cls features.

        Parameters
        ----------
        data : Union[str, np.ndarray, torch.Tensor, List]
            Input point cloud(s)
        normalize : bool
            Normalize input point clouds
        return_numpy : bool
            Return numpy array or torch tensor

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Reconstructed point cloud(s)
            - Single input: (N, 3)
            - Batch input: (B, N, 3)
        """
        # Handle batch input
        if isinstance(data, list):
            data = prepare_batch_input(data, self.device, normalize)
        else:
            data = prepare_input(data, self.device, normalize)

        # Extract cls features
        cls_features, _, _, _ = self.extractor(data)

        # Reconstruct from cls features
        reconstructed = self.model.model.self_reconstruction(cls_features)  # (B, N, 3)

        if return_numpy:
            reconstructed = reconstructed.cpu().numpy()
            # Remove batch dimension if single input
            if reconstructed.shape[0] == 1:
                reconstructed = reconstructed[0]

        return reconstructed

    @torch.no_grad()
    def cross_reconstruct(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
        return_numpy: bool = True,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Perform cross-reconstruction using view generator.

        Parameters
        ----------
        data : Union[str, np.ndarray, torch.Tensor, List]
            Input point cloud(s)
        normalize : bool
            Normalize input point clouds
        return_numpy : bool
            Return numpy arrays or torch tensors

        Returns
        -------
        Dict[str, Union[np.ndarray, torch.Tensor]]
            Dictionary containing:
            - "view1": First view point cloud
            - "view2": Second view point cloud
            - "view1_rot": Rotated first view
            - "view2_rot": Rotated second view
            - "cross_recon1": Cross reconstruction of view1 from view2
            - "cross_recon2": Cross reconstruction of view2 from view1
            - "group1": Grouped patches from view1
            - "group2": Grouped patches from view2
        """
        # Handle batch input
        if isinstance(data, list):
            data = prepare_batch_input(data, self.device, normalize)
        else:
            data = prepare_input(data, self.device, normalize)

        # Generate view pairs
        relative_center_1_2, (view1_rot, view1, scale1), (view2_rot, view2, scale2) = (
            self.view_generator(data)
        )

        # Extract features from both views
        cls_features1, patch_features1, centers1, group1 = self.extractor(view1_rot)
        cls_features2, patch_features2, centers2, group2 = self.extractor(view2_rot)

        # Cross reconstruction
        cross_recon1, cross_recon2 = self.model.model.cross_reconstruction(
            patch_features1, patch_features2, centers1, centers2, relative_center_1_2
        )

        # Add centers to align patches in global space
        group1_with_centers = (group1 + centers1.unsqueeze(2)).flatten(
            1, 2
        )  # (B, P*K, 3)
        group2_with_centers = (group2 + centers2.unsqueeze(2)).flatten(1, 2)
        cross_recon1_with_centers = (cross_recon1 + centers1.unsqueeze(2)).flatten(1, 2)
        cross_recon2_with_centers = (cross_recon2 + centers2.unsqueeze(2)).flatten(1, 2)

        results = {
            "view1": view1,
            "view2": view2,
            "view1_rot": view1_rot,
            "view2_rot": view2_rot,
            "cross_recon1": cross_recon1_with_centers,
            "cross_recon2": cross_recon2_with_centers,
            "group1": group1_with_centers,
            "group2": group2_with_centers,
        }

        if return_numpy:
            results = {k: v.cpu().numpy() for k, v in results.items()}
            # Remove batch dimension if single input
            if list(results.values())[0].shape[0] == 1:
                results = {k: v[0] for k, v in results.items()}

        return results
