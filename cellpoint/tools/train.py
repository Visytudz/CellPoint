import hydra
import wandb
import logging
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from timm.scheduler import CosineLRScheduler

from cellpoint.loss import ChamferLoss
from cellpoint.utils.io import save_ply
from cellpoint.utils.misc import get_pqae_loss
from cellpoint.utils.transforms import (
    Compose,
    PointcloudRotate,
    PointcloudScaleAndTranslate,
    PointcloudJitter,
)


log = logging.getLogger(__name__)


class PretrainTrainer:
    def __init__(self, cfg: DictConfig, output_dir: str):
        """Initializes the Trainer for pre-training without validation.

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
        self._setup_random_seed()

        # Instantiate components
        self.train_transform = self._build_transforms()
        self.train_loader = self._build_dataloader(cfg.dataset.splits)
        self.model = self._build_model().to(self.device)
        self.loss_fn = ChamferLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.cfg.training.epochs,
            lr_min=self.cfg.training.min_lr,
            warmup_t=self.cfg.training.warmup_epochs,
            warmup_lr_init=1e-6,
            t_in_epochs=True,
        )

        # State attributes
        self.epoch = 0
        self.best_train_loss = float("inf")
        self._load_checkpoint()
        self.vis_root_dir = self.output_dir / "visualizations"

    def _setup_random_seed(self):
        """Sets random seeds for reproducibility."""
        log.info(f"Setting random seed to {self.cfg.seed}")
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

    def _build_transforms(self):
        """Builds a composition of transforms from the configuration."""
        log.info("Building data augmentations...")
        cfg_aug = self.cfg.training.augmentations
        transforms = []

        if cfg_aug.get("rotate"):
            transforms.append(PointcloudRotate())
            log.info("  - Rotate added.")

        if cfg_aug.get("scale_and_translate"):
            transforms.append(
                PointcloudScaleAndTranslate(
                    scale_low=cfg_aug.scale_low,
                    scale_high=cfg_aug.scale_high,
                    translate_range=cfg_aug.translate_range,
                )
            )
            log.info("  - Scale and Translate added.")

        if cfg_aug.get("jitter"):
            transforms.append(
                PointcloudJitter(clip=cfg_aug.jitter_clip, sigma=cfg_aug.jitter_sigma)
            )
            log.info("  - Jitter added.")

        return Compose(transforms) if transforms else None

    def _build_dataloader(self, splits: list[str]) -> DataLoader:
        """Creates a DataLoader for the specified data split."""
        log.info(f"Creating {splits} dataloader...")

        datasets_to_concat = []
        for dataset_key in self.cfg.dataset.selected:
            ds_config = self.cfg.dataset.available[dataset_key]
            log.info(f"Loading dataset: '{dataset_key}' by {ds_config._target_}")

            dataset = hydra.utils.instantiate(
                ds_config, splits=splits, transform=self.train_transform
            )
            datasets_to_concat.append(dataset)

        final_dataset = (
            ConcatDataset(datasets_to_concat)
            if len(datasets_to_concat) > 1
            else datasets_to_concat[0]
        )

        return DataLoader(
            final_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
        )

    def _build_model(self) -> torch.nn.Module:
        """Builds the model from the configuration."""
        log.info(f"Building model: {self.cfg.model._target_}")
        model = hydra.utils.instantiate(self.cfg.model)
        return model

    def _load_checkpoint(self):
        """Loads a checkpoint to resume training."""
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

        log.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.cfg.training.get("resume_training", False):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.best_train_loss = checkpoint.get("train_loss")
            log.info(f"Resuming training from epoch {self.epoch + 1}.")
        else:
            log.info("Loaded model weights only.")

    def _save_checkpoint(self, file_name: str):
        """Saves the model state to a file."""
        checkpoint_path = self.output_dir / file_name
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_loss": self.best_train_loss,
            },
            checkpoint_path,
        )
        log.info(f"Saved checkpoint to {checkpoint_path} at epoch {self.epoch}")

    def _train_epoch(self) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.epoch} Training", leave=False
        )

        for batch in progress_bar:
            points = batch["points"].to(self.device)
            self.optimizer.zero_grad()

            if self.cfg.model.config.name == "foldingnet":
                reconstructed_points = self.model(points)
                loss = self.loss_fn(points, reconstructed_points)
            elif self.cfg.model.config.name == "pqae":
                outputs = self.model(points)
                loss = get_pqae_loss(outputs, self.loss_fn)
            else:
                raise ValueError(f"Unknown model name: {self.cfg.model.name}")

            loss.backward()
            self.optimizer.step()

            loss = loss * 1000  # Scale loss for better logging visibility
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def fit(self):
        log.info(f"Using device: {self.device}")
        log.info(f"Train dataset size: {len(self.train_loader.dataset)}")
        log.info(f"Starting pre-training from epoch {self.epoch + 1}.")

        for epoch in range(self.epoch + 1, self.cfg.training.epochs + 1):
            self.epoch = epoch

            # Train step
            train_loss = self._train_epoch()
            log.info(
                f"Epoch {self.epoch}/{self.cfg.training.epochs} | Train Loss: {train_loss:.4f}"
            )
            if self.cfg.wandb.log:
                wandb.log(
                    {
                        "epoch": self.epoch,
                        "train_loss": train_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Other housekeeping
            self.scheduler.step(self.epoch)
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self._save_checkpoint("best_model.pth")
            save_interval = self.cfg.training.get("save_interval", 50)
            if (
                self.epoch % save_interval == 0
                or self.epoch == self.cfg.training.epochs
            ):
                self._save_checkpoint("last_model.pth")

        log.info("Pre-training finished.")
