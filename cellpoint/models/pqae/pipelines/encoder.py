import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from ..layers.transformer import TransformerEncoder


class EncoderWrapper(nn.Module):
    """
    A complete Transformer encoder for token sequences from point clouds.

    This module takes token embeddings and their corresponding center coordinates
    as input. It adds a learnable CLS token, computes positional embeddings for
    the patch tokens, and then processes the full sequence through a deep
    Transformer encoder.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        """
        Initializes the EncoderWrapper module.

        Parameters
        ----------
        embed_dim : int
            The dimensionality of the transformer layers.
        depth : int
            The number of transformer blocks.
        num_heads : int
            The number of attention heads.
        mlp_ratio : float, optional
            The ratio to expand the hidden dimension in the MLP blocks, by default 4.0.
        qkv_bias : bool, optional
            Whether to include bias terms in the QKV projections, by default True.
        proj_drop : float, optional
            The dropout rate after the projection layer, by default 0.0.
        attn_drop : float, optional
            The dropout rate within the attention mechanism, by default 0.0.
        drop_path_rate : float, optional
            The stochastic depth rate, by default 0.0.
        """
        super().__init__()

        # 1. Learnable CLS Token and its Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

        # 2. Positional Embedding generator for patch centers
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim)
        )

        # 3. Core Transformer Encoder architecture
        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path_rate,
        )

    def forward(
        self, patch_tokens: torch.Tensor, centers: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the EncoderWrapper.

        Parameters
        ----------
        patch_tokens : torch.Tensor
            The token embeddings for the point cloud patches. Shape: (B, G, C).
        centers : torch.Tensor
            The center coordinates of each patch. Shape: (B, G, 3).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - Final CLS token features. Shape: (B, 1, C).
            - Final patch token features. Shape: (B, G, C).
        """
        # Generate positional embeddings from patch centers
        pos_embed = self.pos_embed(centers)  # (B, G, C)

        # Prepare CLS token and its position for the batch
        cls_token = self.cls_token.expand(patch_tokens.size(0), -1, -1)  # (B, 1, C)
        cls_pos = self.cls_pos.expand(patch_tokens.size(0), -1, -1)  # (B, 1, C)

        # Concatenate CLS token with patch tokens and their positions
        full_tokens = torch.cat((cls_token, patch_tokens), dim=1)  # (B, G+1, C)
        full_pos = torch.cat((cls_pos, pos_embed), dim=1)  # (B, G+1, C)

        # Pass the full sequence through the Transformer encoder
        encoded_features = self.transformer_encoder(full_tokens, full_pos)

        # Separate the final CLS token feature from the patch token features
        cls_feature = encoded_features[:, :1]  # (B, 1, C)
        patch_features = encoded_features[:, 1:]  # (B, G, C)

        return cls_feature, patch_features
