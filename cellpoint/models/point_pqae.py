import torch
import torch.nn as nn

from .modules.tokenize import Group, PatchEmbed
from .modules.view_generator import PointViewGenerator
from .modules.transformer import EncoderWrapper
from .modules.reconstruction import PositionalQuery, ReconstructionHead


class PointPQAE(nn.Module):
    """
    The main Point-PQAE model for self-supervised pre-training.

    This model orchestrates the entire cross-reconstruction pipeline, including:
    1. Generating two decoupled views from an input point cloud.
    2. Tokenizing and embedding each view into a sequence of feature vectors.
    3. Encoding these sequences using a shared deep Transformer encoder.
    4. Performing a symmetric cross-attention using a positional query mechanism.
    5. Decoding the queried features to reconstruct the target views.
    """

    def __init__(self, config):
        """Initializes the Point-PQAE model and all its sub-modules."""
        super().__init__()
        self.config = config

        self.view_generator = PointViewGenerator(min_crop_rate=config.min_crop_rate)
        self.grouping = Group(num_group=config.num_group, group_size=config.group_size)
        self.patch_embed = PatchEmbed(embed_dim=config.embed_dim)
        self.encoder = EncoderWrapper(
            embed_dim=config.embed_dim,
            trans_dim=config.trans_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_num_heads,
            drop_path_rate=config.drop_path_rate,
        )
        self.positional_query = PositionalQuery(
            embed_dim=config.trans_dim, num_heads=config.decoder_num_heads
        )
        self.reconstruction_head = ReconstructionHead(
            trans_dim=config.trans_dim,
            depth=config.decoder_depth,
            num_heads=config.decoder_num_heads,
            group_size=config.group_size,
        )

    def _get_tokens(
        self, view: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function to perform the tokenization pipeline for a single view."""
        neighborhood, centers = self.grouping(view)
        tokens = self.patch_embed(neighborhood)
        return neighborhood, centers, tokens

    def forward(self, pts: torch.Tensor) -> dict:
        """
        The main forward pass for the pre-training task.

        Parameters
        ----------
        pts : torch.Tensor
            The input batch of point clouds. Shape: (B, N, 3).

        Returns
        -------
        dict
            A dictionary containing the calculated losses, e.g., {'loss': total_loss}.
        """
        # 1. Generate two decoupled views and their relative position
        relative_center_2_to_1, view1, view2 = self.view_generator(pts)
        relative_center_1_to_2 = -relative_center_2_to_1
        # 2. Tokenize and get initial embeddings for both views
        neighborhood1, centers1, tokens1 = self._get_tokens(view1)
        neighborhood2, centers2, tokens2 = self._get_tokens(view2)
        # 3. Encode both token sequences using the shared encoder
        _, encoded_tokens1 = self.encoder(tokens1, centers1)
        _, encoded_tokens2 = self.encoder(tokens2, centers2)
        # --- 4. Symmetric Cross-Reconstruction ---
        # 4a. Reconstruct View 1 from View 2
        queried_features_1 = self.positional_query(
            source_tokens=encoded_tokens2,
            target_centers=centers1,
            relative_center=relative_center_1_to_2,
        )
        reconstructed_patches_1 = self.reconstruction_head(queried_features_1)
        # 4b. Reconstruct View 2 from View 1
        queried_features_2 = self.positional_query(
            source_tokens=encoded_tokens1,
            target_centers=centers2,
            relative_center=relative_center_2_to_1,
        )
        reconstructed_patches_2 = self.reconstruction_head(queried_features_2)

        B, G, K, C_in = neighborhood1.shape
        return {
            "reconstructed_view1": reconstructed_patches_1.view(B, G, K, C_in),
            "reconstructed_view2": reconstructed_patches_2.view(B, G, K, C_in),
            "target_view1": neighborhood1,  # (B, G, K, C_in)
            "target_view2": neighborhood2,  # (B, G, K, C_in)
        }
