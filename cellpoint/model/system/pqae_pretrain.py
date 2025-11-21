import torch
from torch.optim import AdamW
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

from cellpoint.loss import ChamferLoss

log = logging.getLogger(__name__)


class PQAEPretrain(pl.LightningModule):
    def __init__(
        self,
        extractor,
        view_generator,
        decoder,
        center_regressor,
        transform,
        optimizer_cfg,
        loss_weights,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "extractor",
                "view_generator",
                "decoder",
                "center_regressor",
                "transform",
            ]
        )

        # prepare model components
        self.view_generator = view_generator
        self.extractor = extractor
        self.center_regressor = center_regressor
        self.decoder = decoder

        # loss function and data augmentation
        self.transform = transform
        self.loss_fn = ChamferLoss()

        # optimizer config and loss weights
        self.optimizer_cfg = optimizer_cfg
        self.loss_weights = loss_weights

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.optimizer_cfg.epochs,
            eta_min=self.optimizer_cfg.min_lr,
        )
        return [optimizer], [scheduler]

    def self_reconstruction(
        self, cls_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Self reconstruction from cls token to patch.

        Parameters
        ----------
        cls_feature : torch.Tensor
            The global feature of the point cloud. Shape: (B, 1, C).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The reconstructed point cloud patches and the predicted centers.
            Shape: (B, P, K, 3), (B, P, 3).
        """
        # prepare input
        B = cls_feature.shape[0]
        pred_centers = self.center_regressor(cls_feature)  # (B, P, 3)
        num_patches = pred_centers.shape[1]
        source_tokens = cls_feature.expand(-1, num_patches, -1)  # (B, P, C)
        relative_center = torch.zeros(B, 3, device=self.device)  # (B, 3)

        # reconstruct from cls token to patch
        self_recon = self.decoder(
            source_tokens=source_tokens,
            target_centers=pred_centers,
            relative_center=relative_center,
        )  # (B, P, K, 3)

        return self_recon, pred_centers

    def cross_reconstruction(
        self,
        patch_features1: torch.Tensor,
        patch_features2: torch.Tensor,
        centers1: torch.Tensor,
        centers2: torch.Tensor,
        relative_center_1_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cross reconstruction from view1 to view2 and vice versa.

        Parameters
        ----------
        patch_features1 : torch.Tensor
            The patch features of the first view. Shape: (B, P, C).
        patch_features2 : torch.Tensor
            The patch features of the second view. Shape: (B, P, C).
        centers1 : torch.Tensor
            The centers of the first view. Shape: (B, P, 3).
        centers2 : torch.Tensor
            The centers of the second view. Shape: (B, P, 3).
        relative_center_1_2 : torch.Tensor
            The relative center of the first view to the second view. Shape: (B, 3).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The reconstructed point cloud patches and the predicted centers.
            Shape: (B, P, K, 3), (B, P, K, 3).
        """
        # use view2 to reconstruct view1
        cross_recon1 = self.decoder(
            source_tokens=patch_features2,
            target_centers=centers1,
            relative_center=-relative_center_1_2,
        )  # (B, P, K, 3)

        # use view1 to reconstruct view2
        cross_recon2 = self.decoder(
            source_tokens=patch_features1,
            target_centers=centers2,
            relative_center=relative_center_1_2,
        )  # (B, P, K, 3)

        return cross_recon1, cross_recon2

    def training_step(self, batch, batch_idx):
        # 1. get points and data agumentation
        pts = batch["points"]  # (B, N, 3)
        pts = self.transform(pts)

        # 2. generate view pairs and their relative position
        relative_center_1_2, (view1_rot, view1), (view2_rot, view2) = (
            self.view_generator(pts)
        )

        # 3. extract features
        # cls: (B, 1, C), patch: (B, P, C), centers: (B, P, 3), group: (B, P, K, 3)
        cls_features1, patch_features1, centers1, group1 = self.extractor(view1_rot)
        cls_features2, patch_features2, centers2, group2 = self.extractor(view2_rot)

        # 4. cross reconstruction
        cross_recon1, cross_recon2 = self.cross_reconstruction(
            patch_features1, patch_features2, centers1, centers2, relative_center_1_2
        )  # (B, P, K, 3)

        # 5. self reconstruction
        # recon: (B, P, K, 3), pred_centers: (B, P, 3)
        self_recon1, pred_centers1 = self.self_reconstruction(cls_features1)
        self_recon2, pred_centers2 = self.self_reconstruction(cls_features2)

        # 6. calculate loss
        loss_cross = self.loss_fn(
            group1.flatten(0, 1), cross_recon1.flatten(0, 1)
        ) + self.loss_fn(group2.flatten(0, 1), cross_recon2.flatten(0, 1))
        loss_self = self.loss_fn(
            group1.flatten(0, 1), self_recon1.flatten(0, 1)
        ) + self.loss_fn(group2.flatten(0, 1), self_recon2.flatten(0, 1))
        loss_center = self.loss_fn(centers1, pred_centers1) + self.loss_fn(
            centers2, pred_centers2
        )

        # 7. combine losses with weights
        w_cross = self.loss_weights.cross
        w_self = self.loss_weights.self
        w_center = self.loss_weights.center
        warmup_epochs = self.loss_weights.warmup_epochs
        if self.current_epoch < warmup_epochs:
            # warm-up period: focus on geometric and feature learning, pause self-reconstruction
            w_self = 0.0
            w_center = 1.0
        total_loss = w_cross * loss_cross + w_self * loss_self + w_center * loss_center

        # Logging
        self.log_dict(
            {
                "train/loss": total_loss * 1000,
                "train/loss_cross": loss_cross * 1000,
                "train/loss_self": loss_self * 1000,
                "train/loss_center": loss_center * 1000,
                "train/w_cross": w_cross,
                "train/w_self": w_self,
                "train/w_center": w_center,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return total_loss

    def on_train_epoch_end(self):
        """Log epoch summary"""
        epoch = self.current_epoch
        # Get epoch metrics from trainer's logged metrics
        metrics = self.trainer.callback_metrics
        loss_epoch = metrics.get("train/loss_epoch", 0)
        loss_cross = metrics.get("train/loss_cross_epoch", 0)
        loss_self = metrics.get("train/loss_self_epoch", 0)
        loss_center = metrics.get("train/loss_center_epoch", 0)

        log.info(
            f"Epoch {epoch} finished | "
            f"Loss: {loss_epoch:.4f} | "
            f"Cross: {loss_cross:.4f} | "
            f"Self: {loss_self:.4f} | "
            f"Center: {loss_center:.4f}"
        )
