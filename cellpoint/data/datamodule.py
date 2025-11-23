from typing import Dict, Union
from omegaconf import DictConfig, ListConfig

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from logging import getLogger

logger = getLogger(__name__)


class PointCloudDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_datasets: Union[Dict[str, Dataset], list[Dataset]] = None,
        val_datasets: Union[Dict[str, Dataset], list[Dataset]] = None,
        test_datasets: Union[Dict[str, Dataset], list[Dataset]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["train_datasets", "val_datasets", "test_datasets"]
        )

        self.train_ds_list = self._to_list(train_datasets)
        self.val_ds_list = self._to_list(val_datasets)
        self.test_ds_list = self._to_list(test_datasets)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _to_list(self, datasets) -> list[Dataset]:
        if not datasets:
            return []
        if isinstance(datasets, (dict, DictConfig)):
            return list(datasets.values())
        if isinstance(datasets, (list, ListConfig)):
            return list(datasets)
        return [datasets]

    def _concat(self, datasets: list[Dataset]):
        if not datasets:
            return None
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            total_train = sum(len(ds) for ds in self.train_ds_list)
            logger.info(
                f"🔥 [DataModule] Train datasets loaded: {len(self.train_ds_list)} sets, Total samples: {total_train}"
            )
            if self.val_ds_list:
                total_val = sum(len(ds) for ds in self.val_ds_list)
                logger.info(
                    f"📊 [DataModule] Val datasets loaded: {len(self.val_ds_list)} sets, Total samples: {total_val}"
                )

    def train_dataloader(self):
        dataset = self._concat(self.train_ds_list)
        if not dataset:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        dataset = self._concat(self.val_ds_list)
        if not dataset:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        dataset = self._concat(self.test_ds_list)
        if not dataset:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
