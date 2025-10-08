import os
import logging
from pathlib import Path
from typing import Tuple

import h5py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from cellpoint.utils.io import load_ply
from cellpoint.models.foldingnet import Reconstructor
from cellpoint.datasets.hdf5_dataset import HDF5Dataset, normalize_to_unit_sphere

log = logging.getLogger(__name__)


class Inferencer:
    """A tool for point cloud reconstruction and feature extraction."""

    def __init__(self, checkpoint_path: str, model_config: dict, device: str = "auto"):
        """
        Initializes the Inferencer.

        Parameters
        ----------
        checkpoint_path : str
            Path to the trained model checkpoint (.pth file).
        model_config : dict
            A dictionary containing the model's architecture parameters.
        device : str, optional
            Device to run the model on ('cpu', 'cuda', or 'auto'). Default is 'auto'.
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        log.info(f"Inferencer initialized on device: {self.device}")

        self.model = self._load_model(checkpoint_path, model_config)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, checkpoint_path: str, model_config: dict) -> torch.nn.Module:
        """Loads the model and its trained weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        log.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = Reconstructor(**model_config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info("Model loaded successfully.")
        return model

    @torch.no_grad()
    def predict_file(
        self, file_path: str, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on a single .ply file and returns the results.

        Parameters
        ----------
        file_path : str
            Path to the input .ply file.
        normalize : bool, optional
            Whether to normalize the point cloud to a unit sphere before inference.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - reconstruction_np (np.ndarray): The reconstructed point cloud, shape (M, 3).
            - codeword_np (np.ndarray): The feature vector, shape (feat_dims,).
        """
        log.info(f"Processing file: {Path(file_path).name}")
        points_np = load_ply(file_path)  # (N, 3)
        if normalize:
            points_np = normalize_to_unit_sphere(points_np)  # (N, 3)
        points_tensor = (
            torch.from_numpy(points_np).unsqueeze(0).to(self.device)
        )  # (1, N, 3)

        codeword = self.model.encoder(points_tensor)  # (1, feat_dims, 1)
        reconstruction = self.model.decoder(codeword)  # (1, M, 3)
        reconstruction_np = reconstruction.squeeze(0).cpu().numpy()  # (M, 3)
        codeword_np = codeword.squeeze().cpu().numpy()  # (feat_dims,)

        return reconstruction_np, codeword_np

    @torch.no_grad()
    def predict_dataset(
        self,
        dataset: HDF5Dataset,
        output_hdf5_path: str,
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> np.ndarray:
        """
        Runs inference on a dataset, saves all results to a single HDF5 file,
        and returns all codewords.

        Parameters
        ----------
        dataset : HDF5Dataset
            An initialized Dataset object to process.
        output_hdf5_path : str
            Path to save the output HDF5 file.
        batch_size : int, optional
            The batch size for inference.
        num_workers : int, optional
            Number of workers for the DataLoader.

        Returns
        -------
        np.ndarray
            A numpy array containing all codewords from the dataset, shape (num_samples, feat_dims).
        """
        log.info(
            f"Running inference on dataset '{getattr(dataset, 'dataset_name', 'N/A')}'..."
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_recons, all_codewords, all_labels, all_ids = [], [], [], []
        for batch in tqdm(dataloader, desc="Processing dataset"):
            points = batch["points"].to(self.device)  # (B, N, 3)
            codeword = self.model.encoder(points)  # (B, feat_dims, 1)
            reconstruction = self.model.decoder(codeword)  # (B, M, 3)

            all_recons.append(reconstruction.cpu().numpy())
            all_codewords.append(codeword.squeeze(-1).cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_ids.extend(batch["id"])  # .extend for list of strings

        # Concatenate results from all batches
        final_recons = np.concatenate(all_recons, axis=0)  # (num_samples, M, 3)
        final_codewords = np.concatenate(all_codewords, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)  # (num_samples,)

        # Save aggregated results to a single HDF5 file
        self._save_results_to_hdf5(
            output_hdf5_path, final_recons, final_codewords, final_labels, all_ids
        )

        return final_codewords

    def _save_results_to_hdf5(self, path, recons, codewords, labels, ids):
        """Saves aggregated inference results into a single HDF5 file."""
        log.info(f"Saving aggregated results to {path}...")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=recons)
            f.create_dataset("codeword", data=codewords)
            f.create_dataset("label", data=labels)
            # HDF5 requires special handling for variable-length strings
            f.create_dataset(
                "id", data=np.array(ids, dtype=h5py.special_dtype(vlen=str))
            )
        log.info("HDF5 file saved successfully.")
