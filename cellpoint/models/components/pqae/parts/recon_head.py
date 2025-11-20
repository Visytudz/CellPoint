import torch
import torch.nn as nn

from ..layers.transformer import TransformerDecoder


class ReconstructionHead(nn.Module):
    """
    The final layers of the model that reconstruct the point cloud patches.

    This module consists of a lightweight Transformer decoder to refine the
    queried features, followed by a linear layer to project them back to
    the 3D coordinate space.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        group_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        C_out: int = 3,
    ):
        super().__init__()

        # A shallow Transformer decoder for feature refinement
        self.decoder = TransformerDecoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
        )

        # A linear layer to project features to 3D point coordinates
        self.to_points = nn.Linear(embed_dim, C_out * group_size)
        self.C_out = C_out
        self.group_size = group_size

    def forward(self, queried_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        queried_features : torch.Tensor
            The features obtained from the PositionalQueryModule. Shape: (B, P, C).

        Returns
        -------
        torch.Tensor
            The reconstructed point cloud patches. Shape: (B, P, K, C_out).
        """
        # 1. Refine features with the decoder
        decoded_features = self.decoder(queried_features)
        # 2. Project back to point coordinates
        reconstructed_patches = self.to_points(
            decoded_features
        )  # (B, P, C_out * group_size)
        reconstructed_patches = reconstructed_patches.reshape(
            -1, -1, self.group_size, self.C_out
        )  # (B, P, group_size, C_out)

        return reconstructed_patches
