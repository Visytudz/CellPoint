import torch
import torch.nn as nn

from ..layers.attention import Attention
from cellpoint.utils.ops import get_pos_embed


class PositionalQuery(nn.Module):
    """
    Performs the core cross-attention mechanism for Point-PQAE.

    This module takes the encoded features of two views and their geometric
    information, generates the View-Relative Positional Embedding (VRPE), and
    then uses it as a query to perform cross-attention, effectively "querying"
    the features needed to reconstruct a target view from a source view.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # The cross-attention layer is our unified Attention module
        self.cross_attention = Attention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
        )

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
            The encoded patch tokens of the source view. Shape: (B, P, C).
            This will be used as the Key and Value.
        target_centers : torch.Tensor
            The center coordinates of the target view's patches. Shape: (B, P, 3).
        relative_center : torch.Tensor
            The vector difference between the target view's center and the
            source view's center. Shape: (B, 3).

        Returns
        -------
        torch.Tensor
            The queried features, ready for decoding. Shape: (B, P, C).
        """
        B, P, C = source_tokens.shape

        # 1. Construct the 6D relative position vector for the query
        # Expand the global relative center to match the number of patches
        expanded_relative_center = relative_center.unsqueeze(1).expand(
            -1, P, -1
        )  # Shape: (B, P, 3)
        # Concatenate local patch centers with the global view offset
        six_dim_pos = torch.cat(
            [target_centers, expanded_relative_center], dim=-1
        )  # Shape: (B, P, 6)

        # 2. Generate the high-dimensional VRPE from the 6D vector
        vrpe = get_pos_embed(self.embed_dim, six_dim_pos)  # Shape: (B, P, C)

        # 3. Perform cross-attention
        # Query: The positional embedding of the target view (vrpe)
        # Key/Value: The content features of the source view (source_tokens)
        queried_features = self.cross_attention(
            query_source=vrpe, kv_source=source_tokens
        )

        return queried_features
