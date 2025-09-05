import os
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from dataset import Dataset
from model import FoldingNet
from loss import ChamferLoss

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig):
        """
        Initializes the Trainer.

        Parameters
        ----------
        cfg : DictConfig
            The Hydra configuration object.
        """
        self.cfg = cfg
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

    def _setup_reproducibility(self):
        """Sets random seeds for reproducibility."""
        log.info(f"Setting random seed to {self.cfg.seed}")
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

    def _create_dataloader(self, split: str) -> DataLoader:
        """Creates a DataLoader for the specified data split."""
        log.info(f"Creating {split} dataloader...")
        dataset = Dataset(
            root=self.cfg.dataset.root,
            dataset_name=self.cfg.dataset.name,
            split=[split],
            num_points=self.cfg.dataset.num_points,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=(split == "train"),
            num_workers=self.cfg.training.num_workers,
        )

    def _build_model(self) -> torch.nn.Module:
        """Builds the FoldingNet model from the configuration."""
        return FoldingNet(
            feat_dims=self.cfg.model.feat_dims,
            k=self.cfg.model.k,
            grid_size=self.cfg.model.grid_size,
        )

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

    def _train_epoch(self) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.epoch} Training", leave=False
        )

        for points, _ in progress_bar:
            points = points.to(self.device)
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
            for points, _ in self.val_loader:
                points = points.to(self.device)
                reconstructed_points = self.model(points)
                loss = self.loss_fn(points, reconstructed_points)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _save_best_checkpoint(self, val_loss: float):
        """Saves the model checkpoint if the validation loss improves."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_path = "best_model.pth"
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
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
        checkpoint_path = "last_model.pth"
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

            self.scheduler.step()
            self._save_last_checkpoint()

        log.info("Training finished.")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.training.wandb.log:
        wandb.init(
            project=cfg.training.wandb.project,
            entity=cfg.training.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    trainer = Trainer(cfg)
    trainer.fit()

    if cfg.training.wandb.log:
        wandb.finish()
    log.info("Process finished.")


if __name__ == "__main__":
    main()
