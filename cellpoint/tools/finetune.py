import hydra
import wandb
import logging
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import accuracy_score

from cellpoint.models.pqae.classifier import PointPQAE_Classifier

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
        # 注意：微调时通常不需要复杂的增强，可以简化或移除
        self.train_loader = self._build_dataloader(cfg.dataset.train_splits)
        self.val_loader = self._build_dataloader(cfg.dataset.val_splits)

        self.model = self._build_model().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            # 关键：只优化分类头的参数
            self.model.classifier_head.parameters(),
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

    def _setup_random_seed(self):
        """Sets random seeds for reproducibility."""
        log.info(f"Setting random seed to {self.cfg.seed}")
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

    def _build_dataloader(self, splits: list[str]) -> DataLoader:
        """Creates a DataLoader for the specified data split."""
        log.info(f"Creating {splits} dataloader...")
        # 假设您的下游任务数据集也通过 hydra 配置
        dataset = hydra.utils.instantiate(
            self.cfg.dataset.target, splits=splits, transform=None  # 微调通常不需要增强
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True if "train" in splits else False,
            num_workers=self.cfg.training.num_workers,
        )

    def _build_model(self) -> torch.nn.Module:
        """Builds the classifier model and loads pre-trained encoder weights."""
        log.info(f"Building model: {self.cfg.model._target_}")

        # 1. 实例化包含预训练编码器和分类头的完整模型
        # 我们假设 PointPQAE_Classifier 接受预训练模型参数和分类头参数
        model = hydra.utils.instantiate(self.cfg.model)

        # 2. 加载预训练模型的 checkpoint
        pretrain_ckpt_path = self.cfg.training.get("pretrain_checkpoint_path")
        if not pretrain_ckpt_path:
            log.warning("No pre-train checkpoint specified. Training from scratch.")
            return model

        log.info(f"Loading pre-trained encoder weights from: {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location=self.device)

        # 从 checkpoint 中提取 PointPQAE (encoder) 的 state_dict
        # 注意：这里的 'model_state_dict' 键需要和您预训练保存时的一致
        encoder_state_dict = ckpt["model_state_dict"]

        # 加载权重到模型的 encoder 部分
        model.encoder.load_state_dict(encoder_state_dict)
        log.info("Successfully loaded pre-trained encoder weights.")

        # 3. 冻结编码器的权重
        if self.cfg.training.get("freeze_encoder", True):
            for param in model.encoder.parameters():
                param.requires_grad = False
            log.info(
                "Froze the encoder parameters. Only the classifier head will be trained."
            )

        return model

    def _save_checkpoint(self, file_name: str):
        """Saves the model state to a file."""
        checkpoint_path = self.output_dir / file_name
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_accuracy": self.best_val_accuracy,
            },
            checkpoint_path,
        )
        log.info(f"Saved checkpoint to {checkpoint_path} at epoch {self.epoch}")

    def _train_epoch(self) -> float:
        """Runs a single training epoch."""
        self.model.train()
        # 确保编码器在评估模式（如果它有 dropout 或 BN 层）
        if self.cfg.training.get("freeze_encoder", True):
            self.model.encoder.eval()

        total_loss = 0.0
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.epoch} Training", leave=False
        )

        for batch in progress_bar:
            points = batch["points"].to(self.device)
            labels = batch["label"].to(self.device)  # 假设数据加载器提供 'label'

            self.optimizer.zero_grad()

            # forward
            logits = self.model(points)
            loss = self.loss_fn(logits, labels)

            # backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def _validate_epoch(self) -> float:
        """Runs validation and returns accuracy."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                points = batch["points"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(points)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy

    def fit(self):
        log.info(f"Using device: {self.device}")
        log.info(f"Train dataset size: {len(self.train_loader.dataset)}")
        log.info(f"Validation dataset size: {len(self.val_loader.dataset)}")
        log.info(f"Starting fine-tuning from epoch {self.epoch + 1}.")

        for epoch in range(self.epoch + 1, self.cfg.training.epochs + 1):
            self.epoch = epoch

            # Train step
            train_loss = self._train_epoch()

            # Validation step
            val_accuracy = self._validate_epoch()

            log.info(
                f"Epoch {self.epoch}/{self.cfg.training.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
            )
            if self.cfg.wandb.log:
                wandb.log(
                    {
                        "epoch": self.epoch,
                        "train_loss": train_loss,
                        "val_accuracy": val_accuracy,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Housekeeping
            self.scheduler.step(self.epoch)
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self._save_checkpoint("best_model_finetuned.pth")

            self._save_checkpoint("last_model_finetuned.pth")

        log.info("Fine-tuning finished.")
