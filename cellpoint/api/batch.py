"""Batch processing for datasets"""

import torch
import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from .features import FeatureExtractor
from .reconstruction import ReconstructionEngine
from cellpoint.utils.io import save_ply
from cellpoint.utils.misc import batch_normalize_to_unit_sphere_torch

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processing for HDF5 and PyTorch datasets"""

    def __init__(
        self,
        model,
        batch_size: int = 32,
    ):
        """
        Initialize batch processor.

        Parameters
        ----------
        model : InferenceModel
            Loaded inference model
        batch_size : int
            Batch size for processing
        """
        self.model = model
        self.batch_size = batch_size
        self.device = model.device

        # Initialize sub-modules
        self.feature_extractor = FeatureExtractor(model)
        self.reconstruction_engine = ReconstructionEngine(model)

    @torch.no_grad()
    def process_hdf5(
        self,
        hdf5_path: str,
        ids: Optional[List[int]] = None,
        output_features: bool = True,
        output_reconstruction: bool = False,
        save_dir: Optional[str] = None,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Process point clouds from HDF5 file.

        Parameters
        ----------
        hdf5_path : str
            Path to HDF5 file
        ids : Optional[List[int]]
            List of specific IDs to process (None = all)
        output_features : bool
            Extract and return features
        output_reconstruction : bool
            Perform reconstruction
        save_dir : Optional[str]
            Directory to save results (None = memory only)
        normalize : bool
            Normalize point clouds

        Returns
        -------
        Dict[str, Any]
            Results for all processed samples:
            - "features": Dict[id -> feature dict]
            - "reconstructions": Dict[id -> reconstruction dict]
        """
        results = {
            "features": {} if output_features else None,
            "reconstructions": {} if output_reconstruction else None,
        }

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Open HDF5 file
        with h5py.File(hdf5_path, "r") as f:
            # Get all IDs if not specified
            if ids is None:
                ids = list(range(len(f["points"])))

            # Process in batches
            for i in range(0, len(ids), self.batch_size):
                batch_ids = ids[i : i + self.batch_size]

                # Load batch data
                batch_points = []
                for idx in batch_ids:
                    points = f["points"][idx]
                    batch_points.append(torch.from_numpy(points).float())

                # Stack into batch
                batch_data = torch.stack(batch_points, dim=0)
                batch_data = batch_data.to(self.device)
                if normalize:
                    batch_data, _ = batch_normalize_to_unit_sphere_torch(batch_data)

                # Extract features
                if output_features:
                    features = self.feature_extractor.extract_features(
                        batch_data,
                        return_cls=True,
                        return_patch=True,
                        return_concat=True,
                        normalize=False,  # Already normalized
                    )

                    # Store per-sample features
                    for j, idx in enumerate(batch_ids):
                        sample_features = {
                            k: v[j].cpu().numpy() for k, v in features.items()
                        }
                        results["features"][idx] = sample_features

                        # Save to disk if requested
                        if save_dir:
                            feature_dir = save_dir / f"sample_{idx}"
                            feature_dir.mkdir(parents=True, exist_ok=True)
                            for k, v in sample_features.items():
                                np.save(feature_dir / f"{k}.npy", v)

                # Perform reconstruction
                if output_reconstruction:
                    # Self reconstruction
                    self_recon = self.reconstruction_engine.self_reconstruct(
                        batch_data, normalize=False, return_numpy=True
                    )

                    # Cross reconstruction
                    cross_recon = self.reconstruction_engine.cross_reconstruct(
                        batch_data, normalize=False, return_numpy=True
                    )

                    # Store per-sample reconstructions
                    for j, idx in enumerate(batch_ids):
                        sample_recon = {
                            "self_reconstruction": (
                                self_recon[j] if self_recon.ndim == 3 else self_recon
                            ),
                            "cross_recon1": (
                                cross_recon["cross_recon1"][j]
                                if cross_recon["cross_recon1"].ndim == 3
                                else cross_recon["cross_recon1"]
                            ),
                            "cross_recon2": (
                                cross_recon["cross_recon2"][j]
                                if cross_recon["cross_recon2"].ndim == 3
                                else cross_recon["cross_recon2"]
                            ),
                        }
                        results["reconstructions"][idx] = sample_recon

                        # Save to disk if requested
                        if save_dir:
                            recon_dir = save_dir / f"sample_{idx}"
                            recon_dir.mkdir(parents=True, exist_ok=True)
                            save_ply(
                                sample_recon["self_reconstruction"],
                                str(recon_dir / "self_recon.ply"),
                            )
                            save_ply(
                                sample_recon["cross_recon1"],
                                str(recon_dir / "cross_recon1.ply"),
                            )
                            save_ply(
                                sample_recon["cross_recon2"],
                                str(recon_dir / "cross_recon2.ply"),
                            )

                logger.info(f"Processed samples {i} to {i + len(batch_ids)}")

        return results

    @torch.no_grad()
    def process_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        output_features: bool = True,
        output_reconstruction: bool = False,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a PyTorch dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            PyTorch dataset to process
        output_features : bool
            Extract and return features
        output_reconstruction : bool
            Perform reconstruction
        save_dir : Optional[str]
            Directory to save results

        Returns
        -------
        Dict[str, Any]
            Results for all samples
        """
        from torch.utils.data import DataLoader

        results = {
            "features": {} if output_features else None,
            "reconstructions": {} if output_reconstruction else None,
        }

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        sample_idx = 0
        for batch in dataloader:
            # Get points from batch
            if isinstance(batch, dict):
                points = batch["points"]
                batch_ids = batch.get(
                    "id", list(range(sample_idx, sample_idx + points.shape[0]))
                )
            else:
                points = batch
                batch_ids = list(range(sample_idx, sample_idx + points.shape[0]))

            points = points.to(self.device)

            # Extract features
            if output_features:
                features = self.feature_extractor.extract_features(
                    points,
                    return_cls=True,
                    return_patch=True,
                    return_concat=True,
                    normalize=False,
                )

                for j, idx in enumerate(batch_ids):
                    sample_features = {
                        k: v[j].cpu().numpy() for k, v in features.items()
                    }
                    results["features"][idx] = sample_features

                    if save_dir:
                        feature_dir = save_dir / f"sample_{idx}"
                        feature_dir.mkdir(parents=True, exist_ok=True)
                        for k, v in sample_features.items():
                            np.save(feature_dir / f"{k}.npy", v)

            # Perform reconstruction
            if output_reconstruction:
                self_recon = self.reconstruction_engine.self_reconstruct(
                    points, normalize=False, return_numpy=True
                )
                cross_recon = self.reconstruction_engine.cross_reconstruct(
                    points, normalize=False, return_numpy=True
                )

                for j, idx in enumerate(batch_ids):
                    sample_recon = {
                        "self_reconstruction": (
                            self_recon[j] if self_recon.ndim == 3 else self_recon
                        ),
                        "cross_recon1": (
                            cross_recon["cross_recon1"][j]
                            if cross_recon["cross_recon1"].ndim == 3
                            else cross_recon["cross_recon1"]
                        ),
                        "cross_recon2": (
                            cross_recon["cross_recon2"][j]
                            if cross_recon["cross_recon2"].ndim == 3
                            else cross_recon["cross_recon2"]
                        ),
                    }
                    results["reconstructions"][idx] = sample_recon

                    if save_dir:
                        recon_dir = save_dir / f"sample_{idx}"
                        recon_dir.mkdir(parents=True, exist_ok=True)
                        save_ply(
                            sample_recon["self_reconstruction"],
                            str(recon_dir / "self_recon.ply"),
                        )
                        save_ply(
                            sample_recon["cross_recon1"],
                            str(recon_dir / "cross_recon1.ply"),
                        )
                        save_ply(
                            sample_recon["cross_recon2"],
                            str(recon_dir / "cross_recon2.ply"),
                        )

            sample_idx += points.shape[0]
            logger.info(f"Processed {sample_idx} samples")

        return results
