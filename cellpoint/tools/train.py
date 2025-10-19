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
from cellpoint.datasets import HDF5Dataset, ShapeNetDataset
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
        self.train_loader = self._create_dataloader(cfg.dataset.splits)
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
        self.visualization_data = self._prepare_visualization_batch()

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

    def _create_dataloader(self, splits: list[str]) -> DataLoader:
        """Creates a DataLoader for the specified data split.

        This method supports loading and concatenating multiple datasets of
        different types (e.g., HDF5, ShapeNet) as defined in the config.

        Parameters
        ----------
        splits : list[str]
            The dataset splits to load (e.g., ['train']).

        Returns
        -------
        DataLoader
            The configured DataLoader for the specified splits.
        """
        log.info(f"Creating {splits} dataloader...")
        datasets_to_concat = []

        for dataset_key in self.cfg.dataset.selected:
            if dataset_key not in self.cfg.dataset.available:
                log.warning(
                    f"Dataset key '{dataset_key}' from 'selected' list not found in 'available' datasets. Skipping."
                )
                continue

            ds_config = self.cfg.dataset.available[dataset_key]
            log.info(f"Loading dataset: '{dataset_key}' of type '{ds_config.type}'")

            dataset = None
            if ds_config.type == "hdf5":
                dataset = HDF5Dataset(
                    root=ds_config.root,
                    dataset_name=ds_config.name,
                    splits=splits,
                    num_points=ds_config.num_points,
                    normalize=ds_config.get("normalize"),
                    transform=self.train_transform,
                )
            elif ds_config.type == "shapenet":
                dataset = ShapeNetDataset(
                    pc_path=ds_config.pc_path,
                    split_path=ds_config.split_path,
                    splits=splits,
                    num_points=ds_config.num_points,
                    transform=self.train_transform,
                )
            else:
                log.warning(
                    f"Unknown dataset type '{ds_config.type}' for key '{dataset_key}'. Skipping."
                )

            if dataset:
                datasets_to_concat.append(dataset)

        if not datasets_to_concat:
            raise ValueError("No valid datasets were loaded. Check your configuration.")

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

    def _prepare_visualization_batch(self):
        """Prepares a fixed batch for visualization, creates directories, and saves the ground truth point clouds."""
        # valdate visualization indices
        indices = self.cfg.training.get("visualization_indices")
        if not indices:
            log.info(
                "No visualization indices specified. Visualization will be disabled."
            )
            return None
        log.info(f"Preparing visualization samples for indices: {indices}")
        train_dataset = self.train_loader.dataset
        valid_indices = [i for i in indices if i < len(train_dataset)]
        if len(valid_indices) != len(indices):
            log.warning(
                f"Some visualization indices were out of bounds. Using valid indices: {valid_indices}"
            )
        if not valid_indices:
            return None

        # Fetch both points and their IDs
        points_list = [train_dataset[i]["points"] for i in valid_indices]
        batch_tensor = torch.stack(points_list).to(self.device)
        path_list = [
            self.vis_root_dir / f"{i}_{train_dataset[i].get('id', f'index_{i}')}"
            for i in valid_indices
        ]

        # Create directories and save ground truths
        log.info(f"Saving ground truth visualization files to: {self.vis_root_dir}")
        for i, sample_dir in enumerate(path_list):
            sample_dir.mkdir(parents=True, exist_ok=True)
            gt_path = sample_dir / "input.ply"
            gt_points_np = points_list[i].cpu().numpy()
            save_ply(gt_points_np, str(gt_path))

        return {"points": batch_tensor, "paths": path_list}

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

    def _visualize_reconstructions(self):
        """Runs model inference on the visualization batch and saves the
        reconstructed point clouds locally.

        For PQAE, this function saves both reconstructed views and their
        corresponding target views (ground truth).
        """
        if self.visualization_data is None:
            return

        log.info(f"--- Visualizing reconstructions @ Epoch {self.epoch} ---")
        self.model.eval()

        points_batch = self.visualization_data["points"]
        paths_batch = self.visualization_data["paths"]

        with torch.no_grad():
            if self.cfg.model.config.name == "foldingnet":
                reconstructed_points = self.model(points_batch)
                for i, sample_dir in enumerate(paths_batch):
                    recon_path = sample_dir / f"recon_epoch_{self.epoch}.ply"
                    recon_pc_np = reconstructed_points[i].cpu().numpy()
                    save_ply(recon_pc_np, str(recon_path))

            elif self.cfg.model.config.name == "pqae":
                outputs = self.model(points_batch, viz=True)
                B, G, K, C = outputs["reconstructed_view1"].shape
                for key, value in outputs.items():
                    outputs[key] = value.reshape(B, G * K, C)

                save_tasks = [
                    ("group1", f"group1_epoch_{self.epoch}.ply"),
                    ("group2", f"group2_epoch_{self.epoch}.ply"),
                    (
                        "recon1",
                        f"recon1_epoch_{self.epoch}.ply",
                    ),
                    (
                        "recon2",
                        f"recon2_epoch_{self.epoch}.ply",
                    ),
                ]

                # Iterate through each sample in the batch
                for i, sample_dir in enumerate(paths_batch):
                    for key, filename in save_tasks:
                        filepath = sample_dir / filename
                        point_cloud_np = outputs[key][i].cpu().numpy()
                        save_ply(point_cloud_np, str(filepath))

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

    def fit(self):
        log.info(f"Using device: {self.device}")
        log.info(f"Train dataset size: {len(self.train_loader.dataset)}")
        log.info(f"Starting pre-training from epoch {self.epoch + 1}.")

        if self.cfg.wandb.log:
            wandb.watch(self.model, log="gradients", log_freq=self.cfg.wandb.log_freq)

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

            # Visualization step
            should_visualize = (self.cfg.training.visualize_every_n_epochs > 0) and (
                (self.epoch % self.cfg.training.visualize_every_n_epochs == 0)
                or (self.epoch == self.cfg.training.epochs)
            )
            if should_visualize:
                self._visualize_reconstructions()

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
