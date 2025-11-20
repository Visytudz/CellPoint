import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, ConcatDataset


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers
        self.task_type = cfg.task

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _build_dataset(self, splits: str = None):
        """Builds and returns a dataset for the given splits."""
        datasets = []
        for key in self.cfg.dataset.selected:
            if key not in self.cfg.dataset.available:
                raise ValueError(
                    f"Dataset '{key}' selected but not defined in available datasets."
                )

            ds_cfg = self.cfg.dataset.available[key]
            splits = splits if splits is not None else self.cfg.get("splits")
            dataset = hydra.utils.instantiate(ds_cfg, splits=splits)
            datasets.append(dataset)

        if len(datasets) == 0:
            raise ValueError("No datasets selected in config.")

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    def setup(self, stage: str = None):
        """Sets up datasets for different stages: 'fit' and 'test'."""
        if stage == "fit" or stage is None:
            if self.task_type == "pretrain":
                self.train_dataset = self._build_dataset()
            elif self.task_type == "finetune":
                self.train_dataset = self._build_dataset(["train"])
                self.val_dataset = self._build_dataset(["val"])
        if stage == "test":
            self.test_dataset = self._build_dataset(["test"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return None

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return None
