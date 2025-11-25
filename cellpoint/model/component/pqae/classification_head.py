import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()

        in_channels = embed_dim * 2
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C) -> (B, num_classes)
        return self.head(x)
