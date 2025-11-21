import torch
import torch.nn as nn


class CenterRegressor(nn.Module):
    """Predicts the center coordinates of point cloud patches from their global features."""

    def __init__(self, embed_dim: int, num_group: int):
        super().__init__()
        self.num_group = num_group
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_group * 3),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        if cls_token.dim() == 3:
            cls_token = cls_token.squeeze()  # (B, C)
        centers = self.mlp(cls_token).reshape(
            -1, self.num_group, 3
        )  # (B, num_group, 3)

        return centers
