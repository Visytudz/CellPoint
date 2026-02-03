"""Gradient saliency analysis functionality"""

import torch
import numpy as np
from typing import Union, Callable, Tuple, List


class SaliencyAnalyzer:
    """Gradient-based saliency analysis for point clouds"""

    def __init__(self, model):
        """
        Initialize saliency analyzer.

        Parameters
        ----------
        model : InferenceModel
            Loaded inference model
        """
        self.model = model
        self.extractor = model.extractor
        self.device = model.device

    def compute_pca_saliency(
        self,
        pts: Union[torch.Tensor, np.ndarray],
        pca_components: np.ndarray,
        aggregate: str = "norm",
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute gradient saliency for PCA principal components.

        For each principal component PC_k, computes ∂(features·PC_k)/∂(points).

        Parameters
        ----------
        pts : Union[torch.Tensor, np.ndarray]
            Input point clouds, shape (N, 3) or (B, N, 3)
            If 2D input, will be treated as single sample with batch size 1
        pca_components : np.ndarray
            PCA principal component vectors, shape (n_components, feature_dim)
            where feature_dim = 2*C (concatenated cls and patch features)
        aggregate : str
            How to aggregate 3D gradients into saliency scores:
            - 'norm' (default): L2 norm of gradient vectors
            - 'abs': L1 norm (sum of absolute values)
            - 'raw': Return raw gradients without aggregation

        Returns
        -------
        saliency_per_pc : np.ndarray
            Saliency maps for each PC, shape (n_components, B, P*K) or (n_components, P*K)
        group_pts : np.ndarray
            Point cloud coordinates (absolute) for visualization, shape (B, P*K, 3) or (P*K, 3)
        """
        # Convert PCA components to tensor
        pca_components_tensor = torch.from_numpy(pca_components).float().to(self.device)
        n_components = len(pca_components)

        saliency_list = []
        group_pts = None

        # Compute gradient for each principal component
        for pc_idx in range(n_components):
            pc_vector = pca_components_tensor[pc_idx]  # (feature_dim,)

            # Define target function for this PC
            def target_fn(cls_feat, patch_feat):
                cls_squeezed = cls_feat.squeeze(1)  # (B, C)
                patch_max = patch_feat.max(dim=1)[0]  # (B, C)
                combined = torch.cat([cls_squeezed, patch_max], dim=-1)  # (B, 2C)
                return (combined * pc_vector).sum()  # Scalar

            # Use generic gradient saliency method
            saliency, pts_coords = self.compute_gradient_saliency(
                pts, target_fn, aggregate=aggregate
            )
            saliency_list.append(saliency)

            # Store coordinates (same for all PCs)
            if group_pts is None:
                group_pts = pts_coords

        # convert saliency_list to ndarray
        saliency_array = np.array(
            saliency_list
        )  # (n_components, B, P*K) or (n_components, P*K)

        return saliency_array, group_pts

    def compute_gradient_saliency(
        self,
        pts: Union[torch.Tensor, np.ndarray],
        target_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        aggregate: str = "norm",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generic gradient saliency computation with custom target function.

        Computes ∂(target)/∂(points) where target is defined by user-provided function.
        This provides maximum flexibility for computing saliency maps for any
        scalar target derived from model features.

        Parameters
        ----------
        pts : Union[torch.Tensor, np.ndarray]
            Input point clouds, shape (N, 3) or (B, N, 3)
            If 2D input, will be treated as single sample with batch size 1
        target_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            Function that takes (cls_features, patch_features) and returns a scalar.
            - cls_features: shape (B, 1, C)
            - patch_features: shape (B, P, C)
            - Must return a scalar tensor for gradient computation
        aggregate : str
            How to aggregate 3D gradients into saliency scores:
            - 'norm' (default): L2 norm of gradient vectors
            - 'abs': L1 norm (sum of absolute values)
            - 'raw': Return raw gradients without aggregation

        Returns
        -------
        saliency : np.ndarray
            Saliency scores, shape (B, P*K) or (P*K,) for single sample
            For 'raw' aggregate: (B, P*K, 3) or (P*K, 3)
        group_pts : np.ndarray
            Point cloud coordinates (absolute) for visualization, shape (B, P*K, 3) or (P*K, 3)
        """
        # Convert input to tensor
        if isinstance(pts, np.ndarray):
            pts = torch.from_numpy(pts).float()
        pts = pts.to(self.device)

        # Handle non-batch input: (N, 3) -> (1, N, 3)
        squeeze_output = False
        if pts.ndim == 2:
            pts = pts.unsqueeze(0)
            squeeze_output = True

        # Get grouping results
        with torch.no_grad():
            neighborhood, centers = self.extractor.grouping(pts)

        # Set requires_grad on neighborhood
        neighborhood_grad = neighborhood.clone().requires_grad_(True)

        # Forward pass
        tokens = self.extractor.patch_embed(neighborhood_grad)
        cls_feat, patch_feat = self.extractor.encoder(tokens, centers)

        # Compute target using user-provided function
        target = target_fn(cls_feat, patch_feat)

        # Validate target is scalar
        if target.numel() != 1:
            raise ValueError(
                f"target_fn must return a scalar tensor, got shape {target.shape}. "
                "Try adding .sum() to reduce to scalar."
            )

        # Backward pass
        target.backward()

        # Extract gradient
        grad = neighborhood_grad.grad  # (B, P, K, 3)

        # Aggregate gradients based on method
        B, P, K = grad.shape[:3]

        if aggregate == "norm":
            saliency = grad.norm(dim=-1)  # (B, P, K) - L2 norm
            saliency_flat = saliency.reshape(B, P * K).detach().cpu().numpy()
        elif aggregate == "abs":
            saliency = grad.abs().sum(dim=-1)  # (B, P, K) - L1 norm
            saliency_flat = saliency.reshape(B, P * K).detach().cpu().numpy()
        elif aggregate == "raw":
            # Return raw gradients without aggregation
            saliency_flat = grad.reshape(B, P * K, 3).detach().cpu().numpy()
        else:
            raise ValueError(
                f"Unknown aggregate method: {aggregate}. Use 'norm', 'abs', or 'raw'."
            )

        # Get absolute coordinates for visualization
        neighborhood_absolute = neighborhood + centers.unsqueeze(2)
        group_pts_flat = (
            neighborhood_absolute.reshape(B, P * K, 3).detach().cpu().numpy()
        )

        # Squeeze output if input was non-batch
        if squeeze_output:
            saliency_flat = saliency_flat.squeeze(0)
            group_pts_flat = group_pts_flat.squeeze(0)

        return saliency_flat, group_pts_flat
