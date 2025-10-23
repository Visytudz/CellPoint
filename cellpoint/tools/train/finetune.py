import hydra
import wandb
import logging
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from timm.scheduler import CosineLRScheduler


log = logging.getLogger(__name__)


class FinetuneTrainer:
    def __init__(self, cfg: DictConfig, output_dir: str):
        """
        Initializes the Trainer for fine-tuning and validation.

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
        self.train_loader = self._build_dataloader(["train"])
        self.val_loader = self._build_dataloader(["val"])
        self.model = self._build_model().to(self.device)
        self.metrics = self._build_metrics()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.classification_head.parameters(),
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
        self.best_val_accuracy = 0.0
        self._load_checkpoint()

    def _setup_random_seed(self):
        """Sets random seeds for reproducibility."""
        seed = self.cfg.training.seed
        log.info(f"Setting random seed to {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_dataloader(self, splits: list[str]) -> DataLoader:
        """Creates a DataLoader for the specified data split."""
        log.info(f"Creating {splits} dataloader...")

        datasets_to_concat = []
        for dataset_key in self.cfg.dataset.selected:
            ds_config = self.cfg.dataset.available[dataset_key]
            log.info(f"Loading dataset: '{dataset_key}' by {ds_config._target_}")

            dataset = hydra.utils.instantiate(ds_config, splits=splits, transform=None)
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
        """Builds the classifier model and loads pre-trained encoder weights."""
        log.info(f"Building model: {self.cfg.model._target_}")
        model = hydra.utils.instantiate(self.cfg.model)
        if self.cfg.training.get("freeze_encoder", True):
            model.freeze_encoder()

        # Log model parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Total parameters: {total_params:,}")
        log.info(f"Trainable parameters: {trainable_params:,}")
        if total_params > trainable_params:
            log.info(
                f"Note: {total_params - trainable_params:,} parameters are frozen."
            )

        return model

    def _build_metrics(self):
        """Builds TorchMetrics trackers."""
        log.info("Building metrics (Accuracy).")
        num_classes = self.cfg.model.params.classifier_head.num_classes
        metrics = {
            "train_acc": torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(self.device),
            "val_acc": torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(self.device),
        }
        return metrics

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
            self.best_val_accuracy = checkpoint.get("best_val_accuracy")
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
                "best_val_accuracy": self.best_val_accuracy,
            },
            checkpoint_path,
        )
        log.info(f"Saved checkpoint to {checkpoint_path} at epoch {self.epoch}")

    def _train_epoch(self) -> tuple[float, float]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        self.metrics["train_acc"].reset()

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.epoch} Training", leave=False
        )
        for batch in progress_bar:
            points = batch["points"].to(self.device)
            labels = batch["label"].to(self.device).squeeze()
            self.optimizer.zero_grad()

            # forward
            logits = self.model(points)  # (B, num_classes)
            loss = self.loss_fn(logits, labels)

            # backward
            loss.backward()
            self.optimizer.step()

            # Update metrics
            self.metrics["train_acc"].update(logits, labels)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        train_loss = total_loss / len(self.train_loader)
        train_accuracy = self.metrics["train_acc"].compute()
        return train_loss, train_accuracy

    @torch.no_grad()
    def _validate_epoch(self) -> float:
        """Runs validation and returns accuracy."""
        self.model.eval()
        self.metrics["val_acc"].reset()

        for batch in self.val_loader:
            points = batch["points"].to(self.device)
            labels = batch["label"].to(self.device).squeeze()

            logits = self.model(points)  # (B, num_classes)
            self.metrics["val_acc"].update(logits, labels)

        accuracy = self.metrics["val_acc"].compute()
        return accuracy

    def fit(self):
        log.info(f"Using device: {self.device}")
        log.info(f"Train dataset size: {len(self.train_loader.dataset)}")
        log.info(f"Validation dataset size: {len(self.val_loader.dataset)}")
        log.info(f"Starting fine-tuning from epoch {self.epoch + 1}.")

        for epoch in range(self.epoch + 1, self.cfg.training.epochs + 1):
            self.epoch = epoch

            # Train step
            train_loss, train_accuracy = self._train_epoch()
            self._save_checkpoint("last_model.pth")
            log.info(
                f"Epoch {self.epoch}/{self.cfg.training.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            )
            if self.cfg.wandb.log:
                wandb.log(
                    {
                        "epoch": self.epoch,
                        "train_loss": train_loss,
                        "train_accuracy": train_accuracy,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Validation step
            val_interval = self.cfg.training.get("val_interval")
            if epoch % val_interval == 0 or epoch == self.cfg.training.epochs:
                log.info(f"--- Performing validation at epoch {self.epoch} ---")
                val_accuracy = self._validate_epoch()
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self._save_checkpoint("best_model.pth")
                log.info(f"Validation Accuracy: {val_accuracy:.4f}")
                if self.cfg.wandb.log:
                    wandb.log(
                        {
                            "epoch": self.epoch,
                            "val_accuracy": val_accuracy,
                        }
                    )

            # Housekeeping
            self.scheduler.step(self.epoch)

        log.info("Fine-tuning finished.")
