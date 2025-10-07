import torch
import torch.nn as nn

from .transformer import Attention, TransformerDecoder
from cellpoint.utils.misc import get_pos_embed


class PositionalQuery(nn.Module):
    """
    Performs the core cross-attention mechanism for Point-PQAE.

    This module takes the encoded features of two views and their geometric
    information, generates the View-Relative Positional Embedding (VRPE), and
    then uses it as a query to perform cross-attention, effectively "querying"
    the features needed to reconstruct a target view from a source view.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        # The cross-attention layer is our unified Attention module
        self.cross_attention = Attention(dim=embed_dim, num_heads=num_heads)

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_centers: torch.Tensor,
        relative_center: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the positional query.

        Parameters
        ----------
        source_tokens : torch.Tensor
            The encoded patch tokens of the source view. Shape: (B, G, C).
            This will be used as the Key and Value.
        target_centers : torch.Tensor
            The center coordinates of the target view's patches. Shape: (B, G, 3).
        relative_center : torch.Tensor
            The vector difference between the target view's center and the
            source view's center. Shape: (B, 3).

        Returns
        -------
        torch.Tensor
            The queried features, ready for decoding. Shape: (B, G, C).
        """
        B, G, C = source_tokens.shape

        # 1. Construct the 6D relative position vector for the query
        # Expand the global relative center to match the number of patches
        expanded_relative_center = relative_center.unsqueeze(1).expand(
            -1, G, -1
        )  # Shape: (B, G, 3)
        # Concatenate local patch centers with the global view offset
        six_dim_pos = torch.cat(
            [target_centers, expanded_relative_center], dim=-1
        )  # Shape: (B, G, 6)

        # 2. Generate the high-dimensional VRPE from the 6D vector
        vrpe = get_pos_embed(self.embed_dim, six_dim_pos)  # Shape: (B, G, C)

        # 3. Perform cross-attention
        # Query: The positional embedding of the target view (vrpe)
        # Key/Value: The content features of the source view (source_tokens)
        queried_features = self.cross_attention(
            query_source=vrpe, kv_source=source_tokens
        )

        return queried_features


class ReconstructionHead(nn.Module):
    """
    The final layers of the model that reconstruct the point cloud patches.

    This module consists of a lightweight Transformer decoder to refine the
    queried features, followed by a linear layer to project them back to
    the 3D coordinate space.
    """

    def __init__(
        self,
        trans_dim: int,
        depth: int,
        num_heads: int,
        group_size: int,
        C_out: int = 3,
    ):
        super().__init__()

        # A shallow Transformer decoder for feature refinement
        self.decoder = TransformerDecoder(
            embed_dim=trans_dim,
            depth=depth,
            num_heads=num_heads,
        )

        # A linear layer to project features to 3D point coordinates
        self.to_points = nn.Linear(trans_dim, C_out * group_size)

    def forward(self, queried_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        queried_features : torch.Tensor
            The features obtained from the PositionalQueryModule. Shape: (B, G, C).

        Returns
        -------
        torch.Tensor
            The reconstructed point cloud patches. Shape: (B, G, C_out * group_size).
        """
        # 1. Refine features with the decoder
        decoded_features = self.decoder(queried_features)
        # 2. Project back to point coordinates
        reconstructed_patches = self.to_points(
            decoded_features
        )  # (B, G, C_out * group_size)

        return reconstructed_patches
