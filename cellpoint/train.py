import os
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig
import wandb

from dataset import HDF5Dataset
from model import Reconstructor
from loss import ChamferLoss

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig, output_dir: str):
        """
        Initializes the Trainer.

        Parameters
        ----------
        cfg : DictConfig
            The Hydra configuration object.
        output_dir : str
            The directory to save checkpoints and other artifacts.
        """
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_reproducibility()

        # Instantiate components
        self.train_loader = self._create_dataloader("train")
        self.val_loader = self._create_dataloader("val")
        self.model = self._build_model().to(self.device)
        self.loss_fn = ChamferLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.training.epochs,
            eta_min=self.cfg.training.min_lr,
        )

        # State attributes
        self.epoch = 0
        self.best_val_loss = float("inf")
        self._load_checkpoint()
        self.visualization_batch = self._prepare_visualization_batch()

    def _setup_reproducibility(self):
        """Sets random seeds for reproducibility."""
        log.info(f"Setting random seed to {self.cfg.seed}")
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

    def _create_dataloader(self, split: str) -> DataLoader:
        """Creates a DataLoader for the specified data split."""
        log.info(f"Creating {split} dataloader...")
        dataset = HDF5Dataset(
            root=self.cfg.dataset.root,
            dataset_name=self.cfg.dataset.name,
            split=[split],
            num_points=self.cfg.dataset.num_points,
            normalize=self.cfg.dataset.normalize,
            random_jitter=self.cfg.dataset.random_jitter,
            random_rotate=self.cfg.dataset.random_rotate,
            random_translate=self.cfg.dataset.random_translate,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=(split == "train"),
            num_workers=self.cfg.training.num_workers,
        )

    def _build_model(self) -> torch.nn.Module:
        """Builds the FoldingNet model from the configuration."""
        return Reconstructor(**self.cfg.model)

    def _load_checkpoint(self):
        """Loads a checkpoint to resume training or for fine-tuning."""
        # check checkpoint path from config
        checkpoint_path = self.cfg.training.get("checkpoint_path")
        if checkpoint_path is None:
            log.info("No checkpoint specified. Starting training from scratch.")
            return
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            log.warning(
                f"Checkpoint path {checkpoint_path} does not exist. Starting from scratch."
            )
            return

        # Load checkpoint
        log.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # If resuming training, load optimizer, scheduler, and epoch state
        if self.cfg.training.get("resume_training", False):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.epoch = checkpoint["epoch"]
            # Load best_val_loss if it exists in the checkpoint
            if "best_val_loss" in checkpoint:
                self.best_val_loss = checkpoint["best_val_loss"]
            log.info(f"Resuming training from epoch {self.epoch + 1}.")
        else:
            log.info("Loaded model weights only for fine-tuning.")

    def _prepare_visualization_batch(self):
        """Prepares a fixed batch of data for visualization."""
        indices = self.cfg.training.get("visualization_indices")
        if not indices:
            log.info(
                "No visualization indices specified. Visualization will be disabled."
            )
            return None

        log.info(f"Preparing visualization samples for indices: {indices}")

        # Ensure indices are valid
        val_dataset = self.val_loader.dataset
        valid_indices = [i for i in indices if i < len(val_dataset)]
        if len(valid_indices) != len(indices):
            log.warning(
                f"Some visualization indices were out of bounds. Using valid indices: {valid_indices}"
            )
        if not valid_indices:
            return None

        # Retrieve the specific samples and stack them into a single batch
        points_list = [val_dataset[i]["points"] for i in valid_indices]
        batch_tensor = torch.stack(points_list).to(self.device)  # (B, N, 3)
        return batch_tensor

    def _train_epoch(self) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.epoch} Training", leave=False
        )

        for batch in progress_bar:
            points = batch["points"].to(self.device)
            reconstructed_points = self.model(points)
            loss = self.loss_fn(points, reconstructed_points)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def _evaluate_epoch(self) -> float:
        """Runs a single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                points = batch["points"].to(self.device)
                reconstructed_points = self.model(points)
                loss = self.loss_fn(points, reconstructed_points)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _visualize_reconstructions(self):
        """Uses the pre-selected fixed batch to log reconstructions to W&B."""
        if self.visualization_batch is None:
            return
        log.info(f"--- Visualizing reconstructions @ Epoch {self.epoch} ---")
        self.model.eval()
        with torch.no_grad():
            reconstructed_points = self.model(self.visualization_batch)

        point_clouds_to_log = []
        captions = []
        # Iterate through the samples in the fixed batch
        for i in range(self.visualization_batch.shape[0]):
            ground_truth_pc = self.visualization_batch[i].cpu().numpy()
            reconstructed_pc = reconstructed_points[i].cpu().numpy()
            # Add ground truth and reconstruction for each sample
            point_clouds_to_log.append(
                wandb.Object3D({"type": "lidar/beta", "points": ground_truth_pc})
            )
            point_clouds_to_log.append(
                wandb.Object3D({"type": "lidar/beta", "points": reconstructed_pc})
            )
            # Create corresponding captions
            original_index = self.cfg.training.visualization_indices[i]
            captions.append(f"Sample {original_index} - Ground Truth")
            captions.append(f"Sample {original_index} - Reconstruction")

        # Log the combined list to W&B
        wandb.log(
            {
                "Point_Cloud_Reconstruction": point_clouds_to_log,
                "captions": captions,
                "epoch": self.epoch,
            }
        )

    def _save_best_checkpoint(self, val_loss: float):
        """Saves the model checkpoint if the validation loss improves."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_path = self.output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "loss": self.best_val_loss,
                },
                checkpoint_path,
            )
            log.info(
                f"New best model saved to {os.path.join(os.getcwd(), checkpoint_path)}"
            )

            if self.cfg.training.wandb.log:
                artifact = wandb.Artifact(f"{wandb.run.name}-best-model", type="model")
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)

    def _save_last_checkpoint(self):
        """Saves the latest model state for resuming training."""
        checkpoint_path = self.output_dir / "last_model.pth"
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,  # Save best loss for correct resumption
            },
            checkpoint_path,
        )
        log.debug(f"Saved last checkpoint to {checkpoint_path}")

    def fit(self):
        """The main training and validation loop."""
        log.info(f"Using device: {self.device}")
        log.info(
            f"Train dataset size: {len(self.train_loader.dataset)}, Val dataset size: {len(self.val_loader.dataset)}"
        )
        log.info(f"Starting training from epoch {self.epoch + 1}.")

        if self.cfg.training.wandb.log:
            wandb.watch(
                self.model, log="gradients", log_freq=self.cfg.training.wandb.log_freq
            )

        for epoch in range(self.epoch + 1, self.cfg.training.epochs + 1):
            self.epoch = epoch

            # training step
            train_loss = self._train_epoch()
            log.info(
                f"Epoch {self.epoch}/{self.cfg.training.epochs} | Train Loss: {train_loss:.4f}"
            )
            if self.cfg.training.wandb.log:
                wandb.log(
                    {
                        "epoch": self.epoch,
                        "train_loss": train_loss,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                    }
                )

            # validation step
            should_validate = (self.cfg.training.val_every_n_epochs > 0) and (
                (self.epoch % self.cfg.training.val_every_n_epochs == 0)
                or (self.epoch == self.cfg.training.epochs)
            )
            if should_validate:
                val_loss = self._evaluate_epoch()
                log.info(
                    f"--- Validation @ Epoch {self.epoch} --- | Val Loss: {val_loss:.4f}"
                )
                if self.cfg.training.wandb.log:
                    wandb.log({"epoch": self.epoch, "val_loss": val_loss})
                self._save_best_checkpoint(val_loss)

            # visualization step
            if self.cfg.training.wandb.log:
                should_visualize = (
                    self.cfg.training.visualize_every_n_epochs > 0
                ) and (
                    (self.epoch % self.cfg.training.visualize_every_n_epochs == 0)
                    or (self.epoch == self.cfg.training.epochs)
                )
                if should_visualize:
                    self._visualize_reconstructions()

            # scheduler step
            self.scheduler.step()
            self._save_last_checkpoint()

        log.info("Training finished.")
