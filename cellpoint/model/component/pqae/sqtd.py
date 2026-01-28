import math
import torch
import torch.nn as nn


class SphericalFourierEmbedding(nn.Module):
    """
    Spherical Fourier Features for positional encoding.

    Maps low-dimensional spherical coordinates (x, y, z) to high-dimensional space,
    enabling the Transformer to perceive high-frequency geometric details.

    Parameters
    ----------
    in_dim : int, optional
        Input dimension (default is 3 for 3D coordinates).
    embed_dim : int, optional
        Output embedding dimension (default is 384).
    scale : float, optional
        Scaling factor for frequency sampling (default is 10.0).
    """

    def __init__(self, in_dim=3, embed_dim=384, scale=10.0):
        super().__init__()
        self.pi = math.pi
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0, "Embedding dimension must be even."

        # Randomly sample projection matrix B
        B = torch.randn(embed_dim // 2, in_dim) * scale  # (embed_dim/2, 3)
        self.register_buffer("B", B)

    def forward(self, v):
        """
        Forward pass for spherical Fourier embedding.

        Parameters
        ----------
        v : torch.Tensor
            Input tensor of shape (Batch, N_points, 3).

        Returns
        -------
        torch.Tensor
            Embedded tensor of shape (Batch, N_points, embed_dim).
        """
        # v: (Batch, N_points, 3)
        # projected: (Batch, N_points, embed_dim // 2)
        projected = 2 * self.pi * (v @ self.B.T)
        # Concatenate sin and cos -> (Batch, N_points, embed_dim)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class SphericalQueryTransformerDecoder(nn.Module):
    """
    Spherical Query Transformer Decoder (SQTD).

    A Transformer decoder based on spherical queries, specialized for reconstructing
    closed cell surfaces from global features (cls_feat).

    Parameters
    ----------
    embed_dim : int, optional
        Dimension of input cls_feat (default is 384).
    num_queries : int, optional
        Number of reconstruction points (spherical anchor points) (default is 2048).
    num_layers : int, optional
        Number of Transformer layers (default is 4).
    nhead : int, optional
        Number of attention heads (default is 6).
    dim_feedforward : int, optional
        Dimension of feedforward network (default is 1024).
    dropout : float, optional
        Dropout rate (default is 0.1).
    """

    def __init__(
        self,
        embed_dim=384,
        num_queries=2048,
        num_layers=4,
        nhead=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        # Generate standard unit sphere (Fibonacci Lattice) as anchor template
        initial_template = self._generate_fibonacci_sphere(num_queries)  # (1, N, 3)
        self.register_buffer("template", initial_template)
        # self.template = nn.Parameter(initial_template, requires_grad=True)

        # Feature fusion layer for combining cls_feat and pooled patch_features
        self.feature_fusion = nn.Linear(embed_dim * 2, embed_dim)

        self.pos_encoding = SphericalFourierEmbedding(in_dim=3, embed_dim=embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 3)
        )  # Output 3D offsets

        # Initialize output head to output close to 0 initially (keep initial shape as sphere)
        nn.init.constant_(self.head[-1].weight, 0)
        nn.init.constant_(self.head[-1].bias, 0)

    def _generate_fibonacci_sphere(self, samples):
        """
        Generate uniformly distributed Fibonacci sphere points.

        Parameters
        ----------
        samples : int
            Number of points to generate.

        Returns
        -------
        torch.Tensor
            Tensor of shape (1, samples, 3) containing the sphere points.
        """
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append([x, y, z])
        return torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)

    def forward(self, cls_feat, patch_features=None):
        """
        Forward pass for reconstructing point cloud from global features.

        Parameters
        ----------
        cls_feat : torch.Tensor
            Global features of shape (Batch, C) or (Batch, 1, C).
        patch_features : torch.Tensor, optional
            Patch features of shape (Batch, P, C) or (Batch, C).
            - If (Batch, P, C): will be max-pooled to (Batch, C) then fused.
            - If (Batch, C): directly fused without pooling.

        Returns
        -------
        torch.Tensor
            Reconstructed point cloud of shape (Batch, num_queries, 3).
        """
        B = cls_feat.shape[0]

        # 1. Prepare Memory (global features from Encoder)
        if cls_feat.dim() == 2:
            cls_feat = cls_feat.unsqueeze(1)  # (B, 1, C)

        # 2. Optionally fuse with patch features
        if patch_features is not None:
            # Handle different patch_features shapes
            if patch_features.dim() == 3:
                # (B, P, C) -> Max pool to (B, C)
                pooled_patch, _ = torch.max(patch_features, dim=1)
            elif patch_features.dim() == 2:
                # Already (B, C), use directly
                pooled_patch = patch_features
            else:
                raise ValueError(
                    f"patch_features must be 2D (B, C) or 3D (B, P, C), "
                    f"got {patch_features.dim()}D"
                )

            # Concat: (B, C) + (B, C) -> (B, 2C)
            cls_feat_squeezed = cls_feat.squeeze(1)
            combined = torch.cat([cls_feat_squeezed, pooled_patch], dim=-1)

            # Fuse back to embed_dim: (B, 2C) -> (B, C) -> (B, 1, C)
            memory = self.feature_fusion(combined).unsqueeze(1)
        else:
            memory = cls_feat  # (B, 1, C)

        # 3. Prepare Query (spherical anchors)
        template = self.template.expand(B, -1, -1)  # (B, N, 3)
        tgt = self.pos_encoding(template)  # (B, N, C)

        # 4. Transformer decoding
        # tgt: Query (containing spherical geometric info)
        # memory: Key/Value (containing cell semantic info)
        # Points on sphere query memory for their desired shape
        out = self.decoder(tgt=tgt, memory=memory)  # (B, N, C)

        # 5. Predict deformation and apply
        delta = self.head(out)  # (B, N, 3)
        recon_pc = template + delta  # Add offsets

        return recon_pc
