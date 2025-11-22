import torch
import torch.nn as nn


class CenterRegressor(nn.Module):
    """Predicts the center coordinates from patch features."""

    def __init__(self, embed_dim: int):
        super().__init__()
        # Regress center (3D coordinates) from each patch feature
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Predict centers from patch features.
        
        Parameters
        ----------
        patch_features : torch.Tensor
            Patch features. Shape: (B, P, C).
        
        Returns
        -------
        torch.Tensor
            Predicted centers. Shape: (B, P, 3).
        """
        # Apply MLP to each patch feature independently
        centers = self.mlp(patch_features)  # (B, P, 3)
        
        return centers
