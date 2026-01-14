"""Unified inference API"""

import torch
import numpy as np
from typing import Union, List, Dict, Optional, Any

from cellpoint.api.model import InferenceModel
from cellpoint.api.features import FeatureExtractor
from cellpoint.api.reconstruction import ReconstructionEngine
from cellpoint.api.batch import BatchProcessor


class CellPointInference:
    """
    Unified inference interface for CellPoint models.

    Provides a single entry point for all inference operations including
    feature extraction, reconstruction, and batch processing.

    Example
    -------
    >>> model = CellPointInference(
    ...     config_path="cellpoint/config/system/pretrain.yaml",
    ...     checkpoint_path="outputs/pretrain/model.ckpt",
    ...     device="auto"
    ... )
    >>>
    >>> # Feature extraction
    >>> features = model.extract_features("pointcloud.ply")
    >>>
    >>> # Reconstruction
    >>> reconstructed = model.self_reconstruct("pointcloud.ply")
    >>>
    >>> # Batch processing
    >>> results = model.process_hdf5("dataset.h5", ids=[0, 1, 2])
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "auto",
        batch_size: int = 32,
    ):
        """
        Initialize unified inference model.

        Parameters
        ----------
        config_path : str
            Path to config YAML file
        checkpoint_path : str
            Path to checkpoint file
        device : str
            Device to use ('auto', 'cuda', 'cpu')
        batch_size : int
            Default batch size for batch processing
        """
        # Initialize base model
        self._model = InferenceModel(config_path, checkpoint_path, device)

        # Initialize functional modules
        self._feature_extractor = FeatureExtractor(self._model)
        self._reconstruction_engine = ReconstructionEngine(self._model)
        self._batch_processor = BatchProcessor(self._model, batch_size)

        self.device = self._model.device
        self.batch_size = batch_size

    # ==================== Feature Extraction ====================

    @torch.no_grad()
    def extract_features(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        return_cls: bool = True,
        return_patch: bool = True,
        return_concat: bool = True,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from point cloud(s).

        Parameters
        ----------
        data : Union[str, np.ndarray, torch.Tensor, List]
            Input point cloud(s)
        return_cls : bool
            Return cls (global) features
        return_patch : bool
            Return patch (local) features
        return_concat : bool
            Return max-pooled concatenated features
        normalize : bool
            Normalize input point clouds

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with requested features
        """
        return self._feature_extractor.extract_features(
            data, return_cls, return_patch, return_concat, normalize
        )

    @torch.no_grad()
    def extract_cls_features(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract only cls (global) features."""
        return self._feature_extractor.extract_cls_features(data, normalize)

    @torch.no_grad()
    def extract_patch_features(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract only patch (local) features."""
        return self._feature_extractor.extract_patch_features(data, normalize)

    @torch.no_grad()
    def extract_concat_features(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract only concatenated features (cls + max-pooled patch)."""
        return self._feature_extractor.extract_concat_features(data, normalize)

    # ==================== Reconstruction ====================

    @torch.no_grad()
    def self_reconstruct(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Perform self-reconstruction from cls features.

        Parameters
        ----------
        data : Union[str, np.ndarray, torch.Tensor, List]
            Input point cloud(s)
        normalize : bool
            Normalize input point clouds
        return_numpy : bool
            Return numpy array or torch tensor

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Reconstructed point cloud(s)
        """
        return self._reconstruction_engine.self_reconstruct(
            data, normalize, return_numpy
        )

    @torch.no_grad()
    def cross_reconstruct(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
        return_numpy: bool = True,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Perform cross-reconstruction using view generator.

        Parameters
        ----------
        data : Union[str, np.ndarray, torch.Tensor, List]
            Input point cloud(s)
        normalize : bool
            Normalize input point clouds
        return_numpy : bool
            Return numpy arrays or torch tensors

        Returns
        -------
        Dict[str, Union[np.ndarray, torch.Tensor]]
            Dictionary with reconstruction results
        """
        return self._reconstruction_engine.cross_reconstruct(
            data, normalize, return_numpy
        )

    # ==================== Batch Processing ====================

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
            Results for all processed samples
        """
        return self._batch_processor.process_hdf5(
            hdf5_path, ids, output_features, output_reconstruction, save_dir, normalize
        )

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
        return self._batch_processor.process_dataset(
            dataset, output_features, output_reconstruction, save_dir
        )

    # ==================== Utility Methods ====================

    def get_device(self) -> torch.device:
        """Get current device."""
        return self.device

    def get_model(self):
        """Get underlying model (for advanced usage)."""
        return self._model.get_model()

    def get_extractor(self):
        """Get feature extractor module (for advanced usage)."""
        return self._model.extractor

    def get_decoder(self):
        """Get decoder module (for advanced usage)."""
        return self._model.decoder

    def get_global_decoder(self):
        """Get global decoder module (for advanced usage)."""
        return self._model.global_decoder

    def get_view_generator(self):
        """Get view generator module (for advanced usage)."""
        return self._model.view_generator
