import torch
import torch.nn as nn
from timm.layers import DropPath

from .attention import Attention


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.activate = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, D_in).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, D_out).
        """
        x = self.linear1(x)
        x = self.activate(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, D).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, D).
        """
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x
