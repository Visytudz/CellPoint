import torch
import torch.nn as nn


class PatchGenerator(nn.Module):
    """Generates patch features from the global cls token."""

    def __init__(self, embed_dim: int, num_patches: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Two-layer MLP to generate patch features
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim * num_patches),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Generate patch features from cls token.
        
        Parameters
        ----------
        cls_token : torch.Tensor
            Global cls token. Shape: (B, 1, C) or (B, C).
        
        Returns
        -------
        torch.Tensor
            Patch features. Shape: (B, P, C).
        """
        B = cls_token.shape[0]
        
        # Handle both (B, 1, C) and (B, C) inputs
        if cls_token.dim() == 3:
            cls_token = cls_token.squeeze(1)  # (B, C)
        
        # Generate flattened patch features
        patch_features_flat = self.mlp(cls_token)  # (B, C*P)
        
        # Reshape to (B, P, C)
        patch_features = patch_features_flat.view(B, self.num_patches, self.embed_dim)
        
        return patch_features
