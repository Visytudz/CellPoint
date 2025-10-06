class EncoderWrapper(nn.Module):
    """
    A complete encoder module for point clouds.

    It encapsulates the entire process from receiving local point cloud patches
    to outputting final deep features, including tokenization, positional
    embedding, and deep contextual encoding via a Transformer.
    """

    def __init__(
        self,
        patch_embed_dim,
        trans_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        # 1. Patch to Token Embedding Module
        self.patch_embed = Encoder(encoder_channel=patch_embed_dim)

        # 2. CLS Token and its Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, trans_dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

        # 3. Positional Embedding for Patches
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim)
        )

        # 4. Core Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, neighborhood, center):
        """
        Parameters
        ----------
        neighborhood : torch.Tensor
            The local point cloud patches. Shape: (B, G, K, 3).
        center : torch.Tensor
            The center coordinates of each patch. Shape: (B, G, 3).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - Final CLS token features. Shape: (B, 1, C).
            - Final patch token features. Shape: (B, G, C).
        """
        # Generate token embeddings from patches
        patch_tokens = self.patch_embed(neighborhood)

        # Generate positional embeddings from patch centers
        pos_embed = self.pos_embed(center)

        # Prepare CLS token and its position
        cls_token = self.cls_token.expand(patch_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(patch_tokens.size(0), -1, -1)

        # Concatenate CLS token with patch tokens
        full_tokens = torch.cat((cls_token, patch_tokens), dim=1)
        full_pos = torch.cat((cls_pos, pos_embed), dim=1)

        # Pass through the Transformer encoder
        encoded_features = self.transformer_encoder(full_tokens, full_pos)

        # Separate CLS token from patch tokens and return
        return encoded_features[:, :1], encoded_features[:, 1:]
