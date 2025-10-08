import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from cellpoint.models.pqae.model import PointPQAE
from cellpoint.loss.chamfer_loss import ChamferLoss
from cellpoint.datasets.shapenet_dataset import ShapeNetDataset


class PQAEPretrainTrainer:
    """A trainer class to handle the pre-training loop for Point-PQAE."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)

        # 1. Dataset and DataLoader
        dataset = ShapeNetDataset(
            pc_path=cfg.dataset.pc_path,
            data_path=cfg.dataset.path,
            subset="train",
            n_points=cfg.dataset.n_points,
        )
        self.train_loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )

        # 2. Model
        self.model = PointPQAE(cfg.model).to(self.device)

        # 3. Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=cfg.training.optimizer.betas,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs
        )

        # 4. Loss Function (decoupled from the model)
        self.loss_fn = ChamferLoss().to(self.device)

        # Checkpoint directory
        os.makedirs(cfg.training.checkpoint.save_dir, exist_ok=True)

    def _train_epoch(self, epoch):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch}/{self.cfg.training.epochs}"
        )

        for i, (_, _, points) in enumerate(progress_bar):
            points = points.to(self.device)

            # --- Core Training Step ---
            self.optimizer.zero_grad()

            # 1. Get model outputs (predictions, not loss)
            outputs = self.model(points)

            # 2. Calculate loss in the trainer
            loss1 = self.loss_fn(
                outputs["reconstructed_view1"], outputs["target_view1"]
            )
            loss2 = self.loss_fn(
                outputs["reconstructed_view2"], outputs["target_view2"]
            )
            loss = loss1 + loss2

            # 3. Backpropagation
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if i % self.cfg.training.logging.log_interval == 0:
                progress_bar.set_postfix(
                    loss=loss.item(), lr=self.scheduler.get_last_lr()[0]
                )

        return total_loss / len(self.train_loader)

    def train(self):
        """
        The main training loop.
        """
        print("Starting pre-training...")
        for epoch in range(1, self.cfg.training.epochs + 1):
            avg_loss = self._train_epoch(epoch)
            self.scheduler.step()

            print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            if epoch % self.cfg.training.checkpoint.save_interval == 0:
                ckpt_path = os.path.join(
                    self.cfg.training.checkpoint.save_dir, f"epoch_{epoch}.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")
