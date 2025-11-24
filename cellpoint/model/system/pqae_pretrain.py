import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import pytorch_lightning as pl

import json
import logging
from pathlib import Path

from cellpoint.loss import ChamferLoss
from cellpoint.utils.io import save_ply

logger = logging.getLogger(__name__)


class PQAEPretrain(pl.LightningModule):
    def __init__(
        self,
        extractor,
        view_generator,
        decoder,
        cls_to_patch,
        transform=torch.nn.Identity(),
        optimizer_cfg=None,
        loss_weights=None,
        save_dir=None,
        enable_self_reconstruction: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "extractor",
                "view_generator",
                "decoder",
                "cls_to_patch",
                "transform",
            ]
        )

        # prepare model components
        self.view_generator = view_generator
        self.extractor = extractor
        self.cls_to_patch = cls_to_patch
        self.decoder = decoder

        # loss function and data augmentation
        self.transform = transform
        self.loss_fn = ChamferLoss()

        # optimizer config and loss weights
        self.optimizer_cfg = optimizer_cfg
        self.loss_weights = loss_weights

        # other attributes
        self.save_dir = save_dir
        self.enable_self_reconstruction = enable_self_reconstruction
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay,
        )

        # Warmup scheduler: linear warmup from warmup_lr_init to lr
        warmup_epochs = self.optimizer_cfg.warmup_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.optimizer_cfg.warmup_lr_init / self.optimizer_cfg.lr,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # Main scheduler: cosine annealing from lr to min_lr
        cosine_epochs = self.optimizer_cfg.epochs - warmup_epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.optimizer_cfg.min_lr,
        )

        # Combine warmup + cosine annealing
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
        # generate patch features and predict centers
        B = cls_feature.shape[0]
        patch_features, pred_centers = self.cls_to_patch(
            cls_feature
        )  # (B, P, C), (B, P, 3)
        relative_center = torch.zeros(B, 3, device=self.device)  # (B, 3)

        # reconstruct from patch features
        self_recon = self.decoder(
            source_tokens=patch_features,
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

        # 5. self reconstruction (optional)
        if self.enable_self_reconstruction:
            # recon: (B, P, K, 3), pred_centers: (B, P, 3)
            self_recon1, pred_centers1 = self.self_reconstruction(cls_features1)
            self_recon2, pred_centers2 = self.self_reconstruction(cls_features2)

        # 6. calculate loss
        loss_cross = self.loss_fn(
            group1.flatten(0, 1), cross_recon1.flatten(0, 1)
        ) + self.loss_fn(group2.flatten(0, 1), cross_recon2.flatten(0, 1))

        if self.enable_self_reconstruction:
            loss_self = self.loss_fn(
                group1.flatten(0, 1), self_recon1.flatten(0, 1)
            ) + self.loss_fn(group2.flatten(0, 1), self_recon2.flatten(0, 1))
            loss_center = self.loss_fn(centers1, pred_centers1) + self.loss_fn(
                centers2, pred_centers2
            )
        else:
            # If disabled, set auxiliary losses to zero
            loss_self = torch.tensor(0.0, device=self.device)
            loss_center = torch.tensor(0.0, device=self.device)

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
        log_dict = {
            "train/loss": total_loss * 1000,
            "train/loss_cross": loss_cross * 1000,
            "train/w_cross": w_cross,
            "train/w_self": w_self,
            "train/w_center": w_center,
        }
        # include auxiliary losses only if enabled (keeps logs consistent)
        log_dict["train/loss_self"] = loss_self * 1000
        log_dict["train/loss_center"] = loss_center * 1000

        self.log_dict(
            log_dict,
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

        logger.info(
            f"Epoch {epoch} finished | "
            f"Loss: {loss_epoch:.4f} | "
            f"Cross: {loss_cross:.4f} | "
            f"Self: {loss_self:.4f} | "
            f"Center: {loss_center:.4f}"
        )

    def test_step(self, batch, batch_idx):
        """
        Test step to return point clouds at each stage of the inference process.

        Returns a dictionary containing:
        - input_points: Original input point cloud (B, N, 3)
        - view1: First view point cloud (B, N, 3)
        - view2: Second view point cloud (B, N, 3)
        - view1_rot: Rotated first view (B, N, 3)
        - view2_rot: Rotated second view (B, N, 3)
        - group1: Grouped patches from view1 (B, P, K, 3)
        - group2: Grouped patches from view2 (B, P, K, 3)
        - centers1: Centers of patches from view1 (B, P, 3)
        - centers2: Centers of patches from view2 (B, P, 3)
        - cross_recon1: Cross reconstruction view1 (B, P, K, 3)
        - cross_recon2: Cross reconstruction view2 (B, P, K, 3)
        - self_recon1: Self reconstruction view1 (B, P, K, 3)
        - self_recon2: Self reconstruction view2 (B, P, K, 3)
        - pred_centers1: Predicted centers from view1 (B, P, 3)
        - pred_centers2: Predicted centers from view2 (B, P, 3)
        - label: Label from batch if available
        """
        # 1. Get input points
        pts = batch["points"]  # (B, N, 3)
        input_points = pts.clone()
        label = batch.get("label", None)
        id = batch.get("id", None)

        # 2. Generate view pairs and their relative position
        relative_center_1_2, (view1_rot, view1), (view2_rot, view2) = (
            self.view_generator(pts)
        )

        # 3. Extract features
        cls_features1, patch_features1, centers1, group1 = self.extractor(view1_rot)
        cls_features2, patch_features2, centers2, group2 = self.extractor(view2_rot)

        # 4. Cross reconstruction
        cross_recon1, cross_recon2 = self.cross_reconstruction(
            patch_features1, patch_features2, centers1, centers2, relative_center_1_2
        )

        # 5. Self reconstruction (optional)
        if self.enable_self_reconstruction:
            self_recon1, pred_centers1 = self.self_reconstruction(cls_features1)
            self_recon2, pred_centers2 = self.self_reconstruction(cls_features2)
        else:
            # placeholders for downstream code; keep shapes compatible where possible
            B, P, K = group1.shape[0], group1.shape[1], group1.shape[2]
            self_recon1 = torch.zeros(B, P, K, 3, device=self.device)
            self_recon2 = torch.zeros(B, P, K, 3, device=self.device)
            pred_centers1 = torch.zeros(B, centers1.shape[1], 3, device=self.device)
            pred_centers2 = torch.zeros(B, centers2.shape[1], 3, device=self.device)

        # 6. Add centers to reconstructions to align patches in global space
        # Add real centers: (B, P, K, 3) + (B, P, 1, 3) -> (B, P, K, 3)
        group1_with_centers = group1 + centers1.unsqueeze(2)
        group2_with_centers = group2 + centers2.unsqueeze(2)
        cross_recon1_with_centers = cross_recon1 + centers1.unsqueeze(2)
        cross_recon2_with_centers = cross_recon2 + centers2.unsqueeze(2)

        # Add both real and predicted centers to self_recon
        self_recon1_with_real_centers = self_recon1 + centers1.unsqueeze(2)
        self_recon2_with_real_centers = self_recon2 + centers2.unsqueeze(2)
        self_recon1_with_pred_centers = self_recon1 + pred_centers1.unsqueeze(2)
        self_recon2_with_pred_centers = self_recon2 + pred_centers2.unsqueeze(2)

        # Return all intermediate point clouds
        output = {
            "input_points": input_points,
            "view1": view1,
            "view2": view2,
            "view1_rot": view1_rot,
            "view2_rot": view2_rot,
            "group1": group1_with_centers,
            "group2": group2_with_centers,
            "centers1": centers1,
            "centers2": centers2,
            "cross_recon1": cross_recon1_with_centers,
            "cross_recon2": cross_recon2_with_centers,
            "self_recon1_real": self_recon1_with_real_centers,
            "self_recon2_real": self_recon2_with_real_centers,
            "self_recon1_pred": self_recon1_with_pred_centers,
            "self_recon2_pred": self_recon2_with_pred_centers,
            "pred_centers1": pred_centers1,
            "pred_centers2": pred_centers2,
            "relative_center_1_2": relative_center_1_2,
            "label": label,
            "id": id,
        }

        # Collect outputs for epoch end processing
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        """
        Save all test results to disk as PLY files.
        Each sample is saved in save_dir/id/ folder.
        """
        if self.save_dir is None:
            raise ValueError("save_dir must be specified for test mode")

        save_dir = Path(self.save_dir)

        for batch_output in self.test_step_outputs:
            # Get batch data
            batch_size = batch_output["input_points"].shape[0]
            ids = batch_output["id"]

            # Process each sample in the batch
            for i in range(batch_size):
                # Get sample id
                sample_id = ids[i] if ids is not None else f"sample_{i}"
                if isinstance(sample_id, torch.Tensor):
                    sample_id = sample_id.item()

                # Create directory for this sample
                sample_dir = save_dir / str(sample_id)
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Save all point clouds as PLY files
                # Single point clouds (N, 3)
                save_ply(
                    batch_output["input_points"][i].cpu().numpy(),
                    str(sample_dir / "input_points.ply"),
                )
                save_ply(
                    batch_output["view1"][i].cpu().numpy(),
                    str(sample_dir / "view1.ply"),
                )
                save_ply(
                    batch_output["view2"][i].cpu().numpy(),
                    str(sample_dir / "view2.ply"),
                )
                save_ply(
                    batch_output["view1_rot"][i].cpu().numpy(),
                    str(sample_dir / "view1_rot.ply"),
                )
                save_ply(
                    batch_output["view2_rot"][i].cpu().numpy(),
                    str(sample_dir / "view2_rot.ply"),
                )

                # Patch-based point clouds (P, K, 3) - flatten to (P*K, 3)
                save_ply(
                    batch_output["group1"][i].reshape(-1, 3).cpu().numpy(),
                    str(sample_dir / "group1.ply"),
                )
                save_ply(
                    batch_output["group2"][i].reshape(-1, 3).cpu().numpy(),
                    str(sample_dir / "group2.ply"),
                )
                save_ply(
                    batch_output["cross_recon1"][i].reshape(-1, 3).cpu().numpy(),
                    str(sample_dir / "cross_recon1.ply"),
                )
                save_ply(
                    batch_output["cross_recon2"][i].reshape(-1, 3).cpu().numpy(),
                    str(sample_dir / "cross_recon2.ply"),
                )
                # Save self-reconstruction outputs only if enabled
                if self.enable_self_reconstruction:
                    save_ply(
                        batch_output["self_recon1_real"][i]
                        .reshape(-1, 3)
                        .cpu()
                        .numpy(),
                        str(sample_dir / "self_recon1_real.ply"),
                    )
                    save_ply(
                        batch_output["self_recon2_real"][i]
                        .reshape(-1, 3)
                        .cpu()
                        .numpy(),
                        str(sample_dir / "self_recon2_real.ply"),
                    )
                    save_ply(
                        batch_output["self_recon1_pred"][i]
                        .reshape(-1, 3)
                        .cpu()
                        .numpy(),
                        str(sample_dir / "self_recon1_pred.ply"),
                    )
                    save_ply(
                        batch_output["self_recon2_pred"][i]
                        .reshape(-1, 3)
                        .cpu()
                        .numpy(),
                        str(sample_dir / "self_recon2_pred.ply"),
                    )

                # Save metadata (centers, pred_centers, label, relative center) as json
                metadata = {
                    "id": sample_id,
                    "self_reconstruction_enabled": self.enable_self_reconstruction,
                    "centers1": batch_output["centers1"][i].cpu().numpy().tolist(),
                    "centers2": batch_output["centers2"][i].cpu().numpy().tolist(),
                    "pred_centers1": batch_output["pred_centers1"][i]
                    .cpu()
                    .numpy()
                    .tolist(),
                    "pred_centers2": batch_output["pred_centers2"][i]
                    .cpu()
                    .numpy()
                    .tolist(),
                    "relative_center_1_2": batch_output["relative_center_1_2"][i]
                    .cpu()
                    .numpy()
                    .tolist(),
                }
                if batch_output["label"] is not None:
                    label = batch_output["label"][i]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    metadata["label"] = label

                with open(sample_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

        logger.info(f"Test results saved to {save_dir}")

        # Clear outputs for next test run
        self.test_step_outputs.clear()
