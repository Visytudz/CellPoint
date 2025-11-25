import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

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

        # Get warmup settings (with defaults)
        warmup_epochs = self.optimizer_cfg.warmup_epochs
        warmup_lr_init = self.optimizer_cfg.warmup_lr_init
        warmup_start_factor = warmup_lr_init / self.optimizer_cfg.head_lr

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # Cosine annealing scheduler
        cosine_epochs = self.optimizer_cfg.epochs - warmup_epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.optimizer_cfg.min_lr,
        )

        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained weights (extractor and/or classification_head) from checkpoint"""
        if not checkpoint_path:
            logger.info("No checkpoint path provided, using random initialization.")
            return

        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(
            f"✓ Loaded {len(state_dict) - len(unexpected)} keys from checkpoint"
        )

        if missing:
            logger.info(
                f"Missing {len(missing)} keys (will use random initialization):"
            )
            for key in missing:
                logger.info(f"  - {key}")

        if unexpected:
            logger.info(f"Unexpected {len(unexpected)} keys (ignored):")
            for key in unexpected:
                logger.info(f"  - {key}")

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
            batch_size=pts.size(0),
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
            batch_size=pts.size(0),
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
            batch_size=pts.size(0),
        )

    def on_train_epoch_end(self):
        """Log epoch summary"""
        epoch = self.current_epoch
        # Get epoch metrics from trainer's logged metrics
        metrics = self.trainer.callback_metrics
        loss_epoch = metrics.get("train/loss_epoch", 0)
        acc_epoch = metrics.get("train/acc_epoch", 0)

        logger.info(
            f"Epoch {epoch} finished | "
            f"Loss: {loss_epoch:.4f} | "
            f"Acc: {acc_epoch:.4f}"
        )

    def on_validation_epoch_end(self):
        """Log validation epoch summary"""
        epoch = self.current_epoch
        # Get epoch metrics from trainer's logged metrics
        metrics = self.trainer.callback_metrics
        loss_epoch = metrics.get("val/loss", 0)
        acc_epoch = metrics.get("val/acc", 0)

        logger.info(
            f"Epoch {epoch} validation finished | "
            f"Loss: {loss_epoch:.4f} | "
            f"Acc: {acc_epoch:.4f}"
        )
