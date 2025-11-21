import torch
import torch.nn as nn


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
