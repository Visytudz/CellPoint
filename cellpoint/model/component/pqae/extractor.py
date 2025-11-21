import torch.nn as nn
from omegaconf import DictConfig

from .parts.tokenizer import Group, PatchEmbed
from .parts.encoder import Encoder


class FeatureExtractor(nn.Module):
    """
    The feature extractor module for the PQAE model.
    This module is responsible for processing the input point cloud to
    extract both global and local features.

    It comprises three main components:
    1. Grouping: Divides the point cloud into local neighborhoods.
    2. Patch Embedding: Converts local neighborhoods into feature tokens.
    3. Encoder: Captures contextual information from the feature tokens.
    """

    def __init__(
        self,
        grouping_cfg: DictConfig,
        patch_embed_cfg: DictConfig,
        encoder_cfg: DictConfig,
    ):
        super().__init__()

        self.grouping = Group(**grouping_cfg)
        self.patch_embed = PatchEmbed(**patch_embed_cfg)
        self.encoder = Encoder(**encoder_cfg)

    def forward(self, pts):
        """
        Forward pass of the feature extractor.

        Parameters
        ----------
        pts : torch.Tensor
            The input point cloud. Shape: (B, N, 3).

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            A tuple containing:
            - cls_feature: The global feature vector. Shape: (B, C).
            - patch_features: The local patch features. Shape: (B, P, C).
            - centers: The centers of the local neighborhoods. Shape: (B, P, 3).
            - neighborhood: The grouped local neighborhoods. Shape: (B, P, K, 3).
        """
        neighborhood, centers = self.grouping(pts)
        tokens = self.patch_embed(neighborhood)
        cls_feature, patch_features = self.encoder(tokens, centers)

        return cls_feature, patch_features, centers, neighborhood
