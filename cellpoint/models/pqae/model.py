import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from .pipelines.encoder import EncoderWrapper
from .pipelines.tokenizer import Group, PatchEmbed
from .pipelines.view_generator import PointViewGenerator
from .pipelines.decoder import PositionalQuery, ReconstructionHead


class PointPQAE(nn.Module):
    """
    The main Point-PQAE model for self-supervised pre-training.

    This model orchestrates the entire cross-reconstruction pipeline, including:
    1. Generating two decoupled views from an input point cloud.
    2. Tokenizing and embedding each view into a sequence of feature vectors.
    3. Encoding these sequences using a shared deep Transformer encoder.
    4. Performing a symmetric cross-attention using a positional query mechanism.
    5. Decoding the queried features to reconstruct the target views.
    """

    def __init__(self, params):
        """Initializes the Point-PQAE model and all its sub-modules."""
        super().__init__()
        self.params = params

        self.view_generator = PointViewGenerator(**params.view_generator)
        self.grouping = Group(**params.grouping)
        self.patch_embed = PatchEmbed(**params.patch_embed)
        self.encoder = EncoderWrapper(**params.encoder)
        self.positional_query = PositionalQuery(**params.positional_query)
        self.reconstruction_head = ReconstructionHead(**params.reconstruction_head)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_tokens(
        self, view: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function to perform the tokenization pipeline for a single view."""
        neighborhood, centers = self.grouping(view)
        tokens = self.patch_embed(neighborhood)
        return neighborhood, centers, tokens

    def forward(self, pts: torch.Tensor, viz: bool = False) -> dict:
        """
        The main forward pass for the pre-training task.

        Parameters
        ----------
        pts : torch.Tensor
            The input batch of point clouds. Shape: (B, N, 3).
        viz : bool, optional
            If True, returns visualization data. Default is False.

        Returns
        -------
        dict
            A dictionary containing the calculated losses, e.g., {'loss': total_loss}.
        """
        # 1. Generate two decoupled views and their relative position
        relative_center_2_to_1, (view1_rotated, view1), (view2_rotated, view2) = (
            self.view_generator(pts)
        )
        relative_center_1_to_2 = -relative_center_2_to_1

        # 2. Tokenize and get initial embeddings for both views
        neighborhood1, centers1, tokens1 = self._get_tokens(view1_rotated)
        neighborhood2, centers2, tokens2 = self._get_tokens(view2_rotated)

        # 3. Encode both token sequences using the shared encoder
        _, encoded_tokens1 = self.encoder(tokens1, centers1)
        _, encoded_tokens2 = self.encoder(tokens2, centers2)

        # --- 4. Symmetric Cross-Reconstruction ---
        # 4a. Reconstruct View 1 from View 2
        queried_features_1 = self.positional_query(
            source_tokens=encoded_tokens2,
            target_centers=centers1,
            relative_center=relative_center_1_to_2,
        )
        reconstructed_patches_1 = self.reconstruction_head(queried_features_1)
        # 4b. Reconstruct View 2 from View 1
        queried_features_2 = self.positional_query(
            source_tokens=encoded_tokens1,
            target_centers=centers2,
            relative_center=relative_center_2_to_1,
        )
        reconstructed_patches_2 = self.reconstruction_head(queried_features_2)

        B, G, K, C_in = neighborhood1.shape
        reconstructed_patches_1 = reconstructed_patches_1.reshape(B, G, K, C_in)
        reconstructed_patches_2 = reconstructed_patches_2.reshape(B, G, K, C_in)
        if viz:
            return {
                "input": pts,  # (B, N, 3)
                "view1": view1,  # (B, M1, 3)
                "view2": view2,  # (B, M2, 3)
                "view1_rotated": view1_rotated,  # (B, M1, 3)
                "view2_rotated": view2_rotated,  # (B, M2, 3)
                "group1": neighborhood1 + centers1.unsqueeze(2),  # (B, G, K, C_in)
                "group2": neighborhood2 + centers2.unsqueeze(2),  # (B, G, K, C_in)
                "recon1": reconstructed_patches_1
                + centers1.unsqueeze(2),  # (B, G, K, C_in)
                "recon2": reconstructed_patches_2
                + centers2.unsqueeze(2),  # (B, G, K, C_in)
                "relative_center_1_to_2": relative_center_1_to_2,  # (B, 3)
            }
        else:
            return {
                "recon1": reconstructed_patches_1,
                "recon2": reconstructed_patches_2,
                "group1": neighborhood1,
                "group2": neighborhood2,
            }

    @staticmethod
    def get_loss(loss_fn: callable, outputs: dict, *args) -> torch.Tensor:
        """Calculates the total loss from the model outputs."""
        # Unpack outputs
        recon_v1 = outputs["recon1"]  # Shape: [B, G, K, C]
        target_v1 = outputs["group1"]  # Shape: [B, G, K, C]
        recon_v2 = outputs["recon2"]  # Shape: [B, G, K, C]
        target_v2 = outputs["group2"]  # Shape: [B, G, K, C]

        B, G, K, C = recon_v1.shape

        # Reshape for per-patch loss calculation by merging Batch and Group dimensions
        recon_v1_flat = recon_v1.reshape(B * G, K, C)
        target_v1_flat = target_v1.reshape(B * G, K, C)
        recon_v2_flat = recon_v2.reshape(B * G, K, C)
        target_v2_flat = target_v2.reshape(B * G, K, C)

        # Calculate loss for each view
        loss1 = loss_fn(target_v1_flat, recon_v1_flat)
        loss2 = loss_fn(target_v2_flat, recon_v2_flat)

        return loss1 + loss2


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = {
        "min_crop_rate": 0.6,
        "num_group": 64,
        "group_size": 32,
        "embed_dim": 384,
        "in_channels": 3,
        "encoder_depth": 12,
        "encoder_num_heads": 6,
        "decoder_depth": 4,
        "decoder_num_heads": 6,
        "drop_path_rate": 0.1,
        "mlp_ratio": 4.0,
        "qkv_bias": False,
        "proj_drop": 0.0,
        "attn_drop": 0.0,
    }
    cfg = OmegaConf.create(cfg)

    model = PointPQAE(cfg).cuda()
    print(model)

    pts = torch.randn(2, 1024, 3).cuda()
    out = model(pts)

    for k, v in out.items():
        print(k, v.shape)
    print("Total number of parameters:", sum(p.numel() for p in model.parameters()))
    print(
        "Total number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
