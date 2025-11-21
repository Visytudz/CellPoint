import torch.nn as nn
from omegaconf import DictConfig

from .parts.position_query import PositionalQuery
from .parts.recon_head import ReconstructionHead


class PointDecoder(nn.Module):
    """
    The decoder module for the PQAE model.
    This module is responsible for reconstructing the point cloud patches
    from the encoded features using positional queries.
    It comprises two main components:
    1. Positional Query Module: Performs cross-attention to query features
       based on positional information.
    2. Reconstruction Head: Decodes the queried features into point cloud patches.
    """

    def __init__(
        self, positional_query_cfg: DictConfig, reconstruction_head_cfg: DictConfig
    ):
        super().__init__()
        self.query_module = PositionalQuery(**positional_query_cfg)
        self.recon_head = ReconstructionHead(**reconstruction_head_cfg)

    def forward(self, source_tokens, target_centers, relative_center):
        # query features
        queried_features = self.query_module(
            source_tokens=source_tokens,
            target_centers=target_centers,
            relative_center=relative_center,
        )
        # reconstruct patches
        recon_patches = self.recon_head(queried_features)

        return recon_patches
