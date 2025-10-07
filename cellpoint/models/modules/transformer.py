import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_


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

    def forward(self, x):
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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0  # dim must be divisible by num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, query_source: torch.Tensor, kv_source: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Performs the attention mechanism.

        Parameters
        ----------
        query_source : torch.Tensor
            The source for generating queries. Shape: (B, N_q, D).
        kv_source : torch.Tensor, optional
            The source for generating keys and values. Shape: (B, N_kv, D).
            If None, self-attention is performed.

        Returns
        -------
        torch.Tensor
            The result of the attention mechanism. Shape: (B, N_q, D).
        """
        # Perform self-attention if kv_source is not provided
        if kv_source is None:
            kv_source = query_source
        # B_q equals B_kv in general
        B_q, N_q, D = query_source.shape
        B_kv, N_kv, _ = kv_source.shape

        # Generate Q, K, V matrices
        q = (
            self.q(query_source)
            .reshape(B_q, N_q, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (B, num_heads, N_q, head_dim)
        kv = (
            self.kv(kv_source)
            .reshape(B_kv, N_kv, 2, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # (2, B, num_heads, N_kv, head_dim)
        k, v = kv[0], kv[1]  # Each is (B, num_heads, N_kv, head_dim)

        # Calculate attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N_q, N_kv)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Apply attention to V
        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, D)  # (B, N_q, D)
        # Project output
        x = self.proj(x)  # (B, N_q, D)
        x = self.proj_drop(x)

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

    def forward(self, x):
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


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        # Create a list of drop path rates for stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, pos):
        """
        Forward pass of the Transformer encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input embedding tensor of shape (B, N, D).
        pos : torch.Tensor
            Positional embedding tensor of shape (B, N, D).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, D).
        """
        for block in self.blocks:
            x = block(x + pos)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        # Create a list of drop path rates for stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


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
        trans_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
    ):
        """
        Initializes the EncoderWrapper module.

        Parameters
        ----------
        trans_dim : int
            The dimensionality of the transformer layers.
        depth : int
            The number of transformer blocks.
        num_heads : int
            The number of attention heads.
        mlp_ratio : float, optional
            The ratio to expand the hidden dimension in the MLP blocks, by default 4.0.
        drop_path_rate : float, optional
            The stochastic depth rate, by default 0.0.
        """
        super().__init__()

        # 1. Learnable CLS Token and its Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, trans_dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

        # 2. Positional Embedding generator for patch centers
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim)
        )

        # 3. Core Transformer Encoder architecture
        self.transformer_encoder = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
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
        cls_feature = encoded_features[:, :1]
        patch_features = encoded_features[:, 1:]

        return cls_feature, patch_features


if __name__ == "__main__":
    # Example usage and simple test
    B, N, D = 2, 16, 64
    x = torch.randn(B, N, D)
    pos = torch.randn(B, N, D)

    encoder = TransformerEncoder(embed_dim=D, depth=2, num_heads=4)
    out = encoder(x, pos)
    print("TransformerEncoder output shape:", out.shape)  # Expected: (B, N, D)

    decoder = TransformerDecoder(embed_dim=D, depth=2, num_heads=4)
    out_dec = decoder(out)
    print("TransformerDecoder output shape:", out_dec.shape)  # Expected: (B, N, D)
