import torch
import torch.nn as nn

from .parts.patch_generator import PatchGenerator
from .parts.center_regressor import CenterRegressor


class ClsToPatch(nn.Module):
    """
    Integrates patch generation and center regression.

    This module takes a global cls token and:
    1. Generates patch-level features
    2. Predicts center coordinates for each patch
    """

    def __init__(self, embed_dim: int, num_patches: int):
        super().__init__()
        self.patch_generator = PatchGenerator(embed_dim, num_patches)
        self.center_regressor = CenterRegressor(embed_dim)

    def forward(self, cls_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate patch features and predict centers.

        Parameters
        ----------
        cls_token : torch.Tensor
            Global cls token. Shape: (B, 1, C) or (B, C).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - patch_features: Shape (B, P, C)
            - pred_centers: Shape (B, P, 3)
        """
        # Generate patch features
        patch_features = self.patch_generator(cls_token)  # (B, P, C)

        # Predict centers from patch features
        pred_centers = self.center_regressor(patch_features)  # (B, P, 3)

        return patch_features, pred_centers
