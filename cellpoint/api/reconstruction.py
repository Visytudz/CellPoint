"""Point cloud reconstruction functionality"""

import torch
import numpy as np
from typing import Union, List, Dict, Optional

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
    def reconstruct_from_features(
        self,
        cls_features: Union[torch.Tensor, np.ndarray],
        patch_features: Union[torch.Tensor, np.ndarray, None] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Reconstruct point cloud directly from extracted features.

        Parameters
        ----------
        cls_features : Union[torch.Tensor, np.ndarray]
            Global features of shape (C,) or (B, C) or (B, 1, C)
        patch_features : Union[torch.Tensor, np.ndarray, None], optional
            Patch features of shape (C,), (B, C), or (B, P, C).
            - If (B, P, C): will be max-pooled then fused with cls_features.
            - If (C,) or (B, C): directly fused without pooling (e.g., pre-pooled features).
        return_numpy : bool
            Return numpy array or torch tensor

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Reconstructed point cloud(s)
            - Single input: (N, 3)
            - Batch input: (B, N, 3)
        """
        # Convert numpy arrays to torch tensors if needed
        if isinstance(cls_features, np.ndarray):
            cls_features = torch.from_numpy(cls_features).float()
        if patch_features is not None and isinstance(patch_features, np.ndarray):
            patch_features = torch.from_numpy(patch_features).float()

        # Track if input was single sample (no batch dimension)
        single_input = cls_features.ndim == 1

        # Add batch dimension if needed
        if single_input:
            cls_features = cls_features.unsqueeze(0)  # (C,) -> (1, C)
            if patch_features is not None:
                patch_features = patch_features.unsqueeze(0)  # (C,) -> (1, C)

        # Ensure features are on the correct device
        if cls_features.device != self.device:
            cls_features = cls_features.to(self.device)
        if patch_features is not None and patch_features.device != self.device:
            patch_features = patch_features.to(self.device)

        # Reconstruct from features
        reconstructed = self.model.model.self_reconstruction(
            cls_features, patch_features
        )  # (B, N, 3)

        if return_numpy:
            reconstructed = reconstructed.cpu().numpy()
            # Remove batch dimension if single input
            if single_input or reconstructed.shape[0] == 1:
                reconstructed = reconstructed[0]

        return reconstructed

    @torch.no_grad()
    def self_reconstruct(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
        return_numpy: bool = True,
        use_patch_fusion: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Perform self-reconstruction from point cloud input.

        Parameters
        ----------
        data : Union[str, np.ndarray, torch.Tensor, List]
            Input point cloud(s)
        normalize : bool
            Normalize input point clouds
        return_numpy : bool
            Return numpy array or torch tensor
        use_patch_fusion : bool
            If True, fuse max-pooled patch features with cls features for reconstruction.
            This typically improves reconstruction quality by incorporating local details.

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

        # Extract features
        cls_features, patch_features, _, _ = self.extractor(data)

        # Use reconstruct_from_features for actual reconstruction
        patch_feat_to_use = patch_features if use_patch_fusion else None
        return self.reconstruct_from_features(
            cls_features, patch_feat_to_use, return_numpy
        )

    @torch.no_grad()
    def fusion_reconstruct(
        self,
        data_list: List[Union[str, np.ndarray, torch.Tensor]],
        weights: Union[List[float], np.ndarray, torch.Tensor],
        normalize: bool = True,
        normalize_weights: bool = True,
        use_patch_fusion: bool = True,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Reconstruct point cloud from weighted fusion of multiple point clouds' features.

        Parameters
        ----------
        data_list : List[Union[str, np.ndarray, torch.Tensor]]
            List of input point clouds to fuse
        weights : Union[List[float], np.ndarray, torch.Tensor]
            Weights for each point cloud. Must have same length as data_list.
        normalize : bool
            Normalize input point clouds
        normalize_weights : bool
            If True, normalize weights to sum to 1.0
        use_patch_fusion : bool
            If True, also fuse patch features for enhanced reconstruction
        return_numpy : bool
            Return numpy array or torch tensor

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Reconstructed point cloud from fused features, shape (N, 3)

        Raises
        ------
        ValueError
            If length of data_list and weights don't match

        Examples
        --------
        >>> # Interpolate between two cells
        >>> recon = engine.fusion_reconstruct([cell1, cell2], [0.7, 0.3])
        >>>
        >>> # Average multiple cells
        >>> recon = engine.fusion_reconstruct([c1, c2, c3], [1/3, 1/3, 1/3])
        """
        # Validate inputs
        if len(data_list) != len(weights):
            raise ValueError(
                f"Length mismatch: data_list has {len(data_list)} items, "
                f"weights has {len(weights)} items"
            )

        if len(data_list) == 0:
            raise ValueError("data_list cannot be empty")

        # Convert weights to tensor
        if isinstance(weights, list):
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float().to(self.device)
        else:
            weights = weights.float().to(self.device)

        # Normalize weights if requested
        if normalize_weights:
            weights = weights / weights.sum()

        # Extract features from all point clouds
        cls_features_list = []
        patch_features_list = [] if use_patch_fusion else None

        for data in data_list:
            # Prepare single input
            pts = prepare_input(data, self.device, normalize)
            # Extract features
            cls_feat, patch_feat, _, _ = self.extractor(pts)
            cls_features_list.append(cls_feat)
            if use_patch_fusion:
                patch_features_list.append(patch_feat)

        # Weighted fusion of cls features: Σ(weight_i × cls_i)
        cls_fused = sum(w * cls for w, cls in zip(weights, cls_features_list))

        # Weighted fusion of patch features (if enabled)
        patch_fused = None
        if use_patch_fusion:
            patch_fused = sum(
                w * patch for w, patch in zip(weights, patch_features_list)
            )

        # Reconstruct from fused features
        return self.reconstruct_from_features(cls_fused, patch_fused, return_numpy)

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
