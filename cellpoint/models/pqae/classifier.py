import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig

from .pipelines.encoder import EncoderWrapper
from .pipelines.tokenizer import Group, PatchEmbed

log = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()

        in_dim = embed_dim * 2
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PQAEClassifier(nn.Module):
    """
    Downstream classification model for Point-PQAE.
    This model reuses the exact encoder architecture from the
    PointPQAE pre-training model and attaches a classification head.
    """

    def __init__(self, params: DictConfig):
        super().__init__()
        self.params = params

        self.grouping = Group(**params.grouping)
        self.patch_embed = PatchEmbed(**params.patch_embed)
        self.encoder = EncoderWrapper(**params.encoder)
        self.classification_head = ClassificationHead(**params.classifier_head)

    def load_pretrain(self, ckpt_path: str = None, only_encoder: bool = True):
        """Loads pre-trained weights from a checkpoint."""
        if ckpt_path is None:
            log.warning(
                "No pre-trained checkpoint path provided. Training from scratch."
            )
            return
        log.info(f"Loading pre-trained weights from: {ckpt_path}")

        # read pre-trained checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        pretrain_state_dict = checkpoint["model_state_dict"]
        state_dict = {}
        for k, v in pretrain_state_dict.items():
            if only_encoder and not (
                k.startswith("grouping.")
                or k.startswith("patch_embed.")
                or k.startswith("encoder.")
            ):
                continue
            state_dict[k] = v

        # load weights into encoder
        incompatible_keys = self.load_state_dict(state_dict, strict=False)
        if incompatible_keys.missing_keys:
            log.warning(
                f"Missing keys when loading pretrain: {incompatible_keys.missing_keys}"
            )
        if incompatible_keys.unexpected_keys:
            log.warning(
                f"Unexpected keys when loading pretrain: {incompatible_keys.unexpected_keys}"
            )

        log.info("Successfully loaded pre-trained encoder weights.")

    @property
    def encoder_parameters(self) -> list[torch.nn.Parameter]:
        """Returns the encoder parameters."""
        params = []
        params += list(self.grouping.parameters())
        params += list(self.patch_embed.parameters())
        params += list(self.encoder.parameters())
        return params

    def toggle_encoder(self, freeze: bool = True):
        """Freezes or unfreezes the encoder parameters."""
        for param in self.encoder_parameters:
            param.requires_grad = not freeze

    def log_parameters(self):
        """Logs the total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"Total parameters: {total_params:,}")
        log.info(f"Trainable parameters: {trainable_params:,}")
        if total_params > trainable_params:
            log.info(
                f"Note: {total_params - trainable_params:,} parameters are frozen."
            )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Parameters
        ----------
        pts : torch.Tensor
            Input point cloud. Shape: (B, N, 3).

        Returns
        -------
        torch.Tensor
            Classification logits. Shape: (B, num_classes).
        """
        # 1. Tokenize (Grouping + Patch Embedding)
        neighborhood, centers = self.grouping(pts)
        tokens = self.patch_embed(neighborhood)  # (B, G, C)

        # 2. Encode (EncoderWrapper)
        # cls_feature: (B, 1, C), patch_features: (B, G, C)
        cls_feature, patch_features = self.encoder(tokens, centers)

        # 3. Aggregate features
        # (B, G, C) -> (B, C)
        max_pool_feature = patch_features.max(dim=1)[0]
        # (B, C) & (B, C) -> (B, 2*C)
        global_feature = torch.cat((cls_feature.squeeze(1), max_pool_feature), dim=1)

        # 4. Classify
        logits = self.classification_head(global_feature)  # (B, num_classes)

        return logits
