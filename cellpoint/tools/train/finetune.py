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

from cellpoint.utils.transforms import Compose
from cellpoint.utils.misc import decompose_confusion_matrix, plot_confusion_matrix


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
        self.train_transform = Compose.from_cfg(self.cfg.training.augmentations)
        self.model = self._build_model().to(self.device)
        self.metrics = self._build_metrics()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._build_optimizer_and_scheduler()

        # State attributes
        self.epoch = 0
        self.best_val_accuracy = 0.0
        self.class_names = self.train_loader.dataset.class_names
        self.cm_dir = self.output_dir / "confusion_matrices"
        self.cm_dir.mkdir(parents=True, exist_ok=True)
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

            dataset = hydra.utils.instantiate(ds_config, splits=splits)
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
        return model

    def _build_optimizer_and_scheduler(self):
        """Builds the optimizer and learning rate scheduler for training."""
        log.info("Building optimizer and learning rate scheduler.")
        param_groups = [
            {
                "params": self.model.classification_head.parameters(),
                "lr": self.cfg.training.lr,
                "name": "classifier",
            },
            {
                "params": self.model.encoder_parameters,
                "lr": self.cfg.training.encoder_lr,
                "name": "encoder",
            },
        ]
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.cfg.training.epochs,
            lr_min=self.cfg.training.min_lr,
            warmup_t=self.cfg.training.warmup_epochs,
            warmup_lr_init=1e-6,
            t_in_epochs=True,
        )
        return optimizer, scheduler

    def _build_metrics(self):
        """Builds TorchMetrics trackers."""
        log.info("Building metrics (Accuracy, ConfusionMatrix).")
        num_classes = self.cfg.model.params.classifier_head.num_classes
        metrics = {
            "train_acc": torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(self.device),
            "val_acc": torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(self.device),
            "train_cm": torchmetrics.ConfusionMatrix(
                task="multiclass", num_classes=num_classes
            ).to(self.device),
            "val_cm": torchmetrics.ConfusionMatrix(
                task="multiclass", num_classes=num_classes
            ).to(self.device),
        }
        return metrics

    def _load_checkpoint(self):
        """Loads a checkpoint to resume training."""
        checkpoint_path = self.cfg.training.get("checkpoint_path")
        self.model.load_pretrain(
            ckpt_path=checkpoint_path, only_encoder=self.cfg.training.only_encoder
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if self.cfg.training.get("resume_training", False):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.best_val_accuracy = checkpoint.get("best_val_accuracy")
            log.info(f"Resuming training from epoch {self.epoch + 1}.")
        else:
            log.info("Loaded model weights only.")

    def _save_checkpoint(self, file_name: str, verbose: bool = True):
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
        if verbose:
            log.info(f"Saved checkpoint to {checkpoint_path} at epoch {self.epoch}")

    def _train_epoch(self) -> tuple[float, float, torch.Tensor]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        self.metrics["train_acc"].reset()
        self.metrics["train_cm"].reset()

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.epoch} Training", leave=False
        )
        for batch in progress_bar:
            points = batch["points"].to(self.device)
            points = self.train_transform(points)
            labels = batch["label"].to(self.device).squeeze()
            self.optimizer.zero_grad()

            # forward
            logits = self.model(points)  # (B, num_classes)
            loss = self.loss_fn(logits, labels)

            # backward
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update metrics
            self.metrics["train_acc"].update(logits, labels)
            self.metrics["train_cm"].update(logits, labels)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        train_loss = total_loss / len(self.train_loader)
        train_accuracy = self.metrics["train_acc"].compute()
        train_cm = self.metrics["train_cm"].compute()
        return train_loss, train_accuracy, train_cm

    @torch.no_grad()
    def _validate_epoch(self) -> tuple[float, torch.Tensor]:
        """Runs validation and returns accuracy."""
        self.model.eval()
        self.metrics["val_acc"].reset()
        self.metrics["val_cm"].reset()

        for batch in self.val_loader:
            points = batch["points"].to(self.device)
            labels = batch["label"].to(self.device).squeeze()

            logits = self.model(points)  # (B, num_classes)
            self.metrics["val_acc"].update(logits, labels)
            self.metrics["val_cm"].update(logits, labels)

        accuracy = self.metrics["val_acc"].compute()
        val_cm = self.metrics["val_cm"].compute()
        return accuracy, val_cm

    def fit(self):
        log.info(f"Using device: {self.device}")
        log.info(f"Train dataset size: {len(self.train_loader.dataset)}")
        log.info(f"Validation dataset size: {len(self.val_loader.dataset)}")
        log.info(f"Starting fine-tuning from epoch {self.epoch + 1}.")

        # Handle encoder freezing
        unfreeze_epoch = self.cfg.training.get("unfreeze_encoder_epoch")
        is_encoder_frozen = unfreeze_epoch > self.epoch
        if is_encoder_frozen:
            self.model.toggle_encoder(freeze=True)
            log.info("Encoder is initially frozen.")
        self.model.log_parameters()

        # Training loop
        for epoch in range(self.epoch + 1, self.cfg.training.epochs + 1):
            self.epoch = epoch

            # Unfreeze step
            if is_encoder_frozen and self.epoch == unfreeze_epoch:
                log.info("Unfreezing encoder parameters.")
                self.model.toggle_encoder(freeze=False)
                is_encoder_frozen = False

            # Train step
            train_loss, train_accuracy, train_cm = self._train_epoch()
            self._save_checkpoint("last_model.pth", verbose=False)
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
                        "head_lr": self.optimizer.param_groups[0]["lr"],
                        "encoder_lr": self.optimizer.param_groups[1]["lr"],
                    }
                )

            # Validation step
            val_interval = self.cfg.training.get("val_interval")
            if epoch % val_interval == 0 or epoch == self.cfg.training.epochs:
                log.info(f"--- Performing validation at epoch {self.epoch} ---")
                val_accuracy, val_cm = self._validate_epoch()
                log.info(f"Validation Accuracy: {val_accuracy:.4f}")
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self._save_checkpoint("best_model.pth")
                # Plot and save confusion matrices
                plot_confusion_matrix(
                    train_cm.cpu().numpy(),
                    class_names=self.class_names,
                    save_path=self.cm_dir / f"train_epoch{self.epoch}.png",
                    title=f"Training Confusion Matrix - Epoch {self.epoch}",
                )
                plot_confusion_matrix(
                    val_cm.cpu().numpy(),
                    class_names=self.class_names,
                    save_path=self.cm_dir / f"val_epoch{self.epoch}.png",
                    title=f"Validation Confusion Matrix - Epoch {self.epoch}",
                )
                # Log to WandB
                if self.cfg.wandb.log:
                    train_targets, train_preds = decompose_confusion_matrix(train_cm)
                    val_targets, val_preds = decompose_confusion_matrix(val_cm)
                    wandb.log(
                        {
                            "epoch": self.epoch,
                            "val_accuracy": val_accuracy,
                            "train_cm": wandb.plot.confusion_matrix(
                                y_true=train_targets,
                                preds=train_preds,
                                class_names=self.class_names,
                                title="Training Confusion Matrix",
                            ),
                            "val_cm": wandb.plot.confusion_matrix(
                                y_true=val_targets,
                                preds=val_preds,
                                class_names=self.class_names,
                                title="Validation Confusion Matrix",
                            ),
                        }
                    )

            # Housekeeping
            self.scheduler.step(self.epoch)

        log.info("Fine-tuning finished.")
