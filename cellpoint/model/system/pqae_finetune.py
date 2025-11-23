import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from logging import getLogger

logger = getLogger(__name__)


class PQAEFinetune(pl.LightningModule):
    def __init__(
        self,
        extractor,
        classification_head,
        transform,
        metrics,
        optimizer_cfg,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["extractor", "classification_head", "transform", "metrics"]
        )

        # prepare model components
        self.extractor = extractor
        self.classification_head = classification_head

        # data augmentation, optimizer config, loss and metrics
        self.transform = transform
        self.optimizer_cfg = optimizer_cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = metrics.train_acc
        self.val_acc = metrics.val_acc

        # Encoder Freeze Control
        self.encoder_frozen = False

    def configure_optimizers(self):
        params = [
            {
                "params": self.extractor.parameters(),
                "lr": self.optimizer_cfg.extractor_lr,
            },
            {
                "params": self.classification_head.parameters(),
                "lr": self.optimizer_cfg.head_lr,
            },
        ]
        optimizer = AdamW(params, weight_decay=self.optimizer_cfg.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.optimizer_cfg.epochs,
            eta_min=self.optimizer_cfg.min_lr,
        )
        return [optimizer], [scheduler]

    def load_pretrained_encoder(self, checkpoint_path):
        """load pretrained encoder weights from PQAE pretraining checkpoint"""
        if not checkpoint_path:
            logger.info("No checkpoint path provided for pretrained encoder.")
            return

        logger.info(f"Loading pretrained encoder from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("extractor."):
                new_state_dict[k] = v
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded keys: {len(new_state_dict)}")
        logger.info(f"Missing keys (mostly head): {len(missing)}")

    def on_train_epoch_start(self):
        """control encoder freeze/unfreeze"""
        current_epoch = self.current_epoch
        unfreeze_epoch = self.optimizer_cfg.unfreeze_extractor_epoch

        if current_epoch < unfreeze_epoch and not self.encoder_frozen:
            logger.info(f"Epoch {current_epoch}: Freezing encoder.")
            for param in self.extractor.parameters():
                param.requires_grad = False
            self.encoder_frozen = True

        elif current_epoch >= unfreeze_epoch and self.encoder_frozen:
            logger.info(f"Epoch {current_epoch}: Unfreezing encoder.")
            for param in self.extractor.parameters():
                param.requires_grad = True
            self.encoder_frozen = False

    def forward(self, pts):
        # 1. extract features from backbone
        # cls_feat: (B, 1, C), patch_feat: (B, G, C)
        cls_feat, patch_feat, _, _ = self.extractor(pts)

        # 2. feature aggregation (Global Pooling)
        cls_feat = cls_feat.squeeze(1)  # (B, C)
        max_pool_feat = patch_feat.max(dim=1)[0]  # (B, C)
        global_feat = torch.cat((cls_feat, max_pool_feat), dim=1)  # (B, 2C)

        # 3. classification head
        logits = self.classification_head(global_feat)

        return logits

    def training_step(self, batch, batch_idx):
        pts, labels = batch["points"], batch["label"].squeeze()
        pts = self.transform(pts)
        logits = self(pts)
        loss = self.loss_fn(logits, labels)

        self.train_acc(logits, labels)
        self.log_dict(
            {
                "train/loss": loss,
                "train/acc": self.train_acc,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        pts, labels = batch["points"], batch["label"].squeeze()
        logits = self(pts)
        loss = self.loss_fn(logits, labels)

        self.val_acc(logits, labels)
        self.log_dict(
            {
                "val/loss": loss,
                "val/acc": self.val_acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        pts, labels = batch["points"], batch["label"].squeeze()
        logits = self(pts)
        loss = self.loss_fn(logits, labels)

        self.val_acc(logits, labels)
        self.log_dict(
            {
                "test/loss": loss,
                "test/acc": self.val_acc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
