import copy
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
        acc_metrics,
        train_transform=torch.nn.Identity(),
        val_transform=torch.nn.Identity(),
        optimizer_cfg=None,
        vote_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "extractor",
                "classification_head",
                "acc_metrics",
                "train_transform",
                "val_transform",
            ]
        )

        # prepare model components
        self.extractor = extractor
        self.classification_head = classification_head

        # data augmentation, optimizer config, loss and metrics
        self.train_transform = train_transform
        self.val_transform = val_transform  # only for vote inference
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = acc_metrics
        self.val_acc = copy.deepcopy(self.train_acc)
        self.vote_acc = copy.deepcopy(self.train_acc)
        self.optimizer_cfg = optimizer_cfg
        self.vote_cfg = vote_cfg

        # Encoder Freeze Control
        self.encoder_frozen = False
        self.best_val_acc = 0.0
        self.should_vote_this_epoch = False

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
        pts = self.train_transform(pts)
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

    def on_train_epoch_end(self):
        """Log epoch summary"""
        # Get epoch metrics from trainer's logged metrics
        metrics = self.trainer.callback_metrics
        loss_epoch = metrics.get("train/loss_epoch", 0)
        acc_epoch = metrics.get("train/acc_epoch", 0)

        logger.info(
            f"Epoch {self.current_epoch} finished | "
            f"Loss: {loss_epoch:.4f} | "
            f"Acc: {acc_epoch:.4f}"
        )

    def on_validation_epoch_start(self):
        self.should_vote_this_epoch = False
        if self.trainer.sanity_checking:
            return
        if (
            self.vote_cfg is not None
            and self.vote_cfg["enabled"]
            and self.best_val_acc >= self.vote_cfg["acc_threshold"]
        ):
            self.should_vote_this_epoch = True

    def _vote_inference(self, pts):
        num_votes = self.vote_cfg["num_votes"]
        vote_logits = 0
        for _ in range(num_votes):
            pts_aug = self.val_transform(pts)
            logits = self(pts_aug)
            vote_logits += logits

        return vote_logits / num_votes

    def validation_step(self, batch, batch_idx):
        pts, labels = batch["points"], batch["label"].squeeze()

        # 1. Standard inference
        logits = self(pts)
        loss = self.loss_fn(logits, labels)
        self.val_acc(logits, labels)

        # 2. Vote inference
        if self.should_vote_this_epoch:
            logits_vote = self._vote_inference(pts)
            self.vote_acc(logits_vote, labels)

        # 3. Log
        self.log_dict(
            {
                "val/loss": loss,
                "val/acc": self.val_acc,
                "val/acc_vote": self.vote_acc if self.should_vote_this_epoch else 0.0,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=pts.size(0),
        )

    def on_validation_epoch_end(self):
        """Log validation epoch summary"""
        metrics = self.trainer.callback_metrics
        current_acc = metrics.get("val/acc", 0)
        vote_acc = metrics.get("val/acc_vote", 0)
        loss_epoch = metrics.get("val/loss", 0)

        if current_acc is not None and current_acc > self.best_val_acc:
            self.best_val_acc = current_acc.item()

        logger.info(
            f"Validation Epoch {self.current_epoch} finished | "
            f"Loss: {loss_epoch:.4f} | "
            f"Acc: {current_acc:.4f} | "
            f"Vote Acc: {vote_acc:.4f} | "
            f"Best Standard Acc: {self.best_val_acc:.4f}"
        )

    def test_step(self, batch, batch_idx):
        pts, labels = batch["points"], batch["label"].squeeze()

        if self.vote_cfg is not None and self.vote_cfg["enabled"]:
            logits = self._vote_inference(pts)
        else:
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
