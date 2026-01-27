"""Unified inference API"""

import torch
import numpy as np
from typing import Union, List, Dict, Optional, Any

from cellpoint.api.model import InferenceModel
from cellpoint.api.features import FeatureExtractor
from cellpoint.api.reconstruction import ReconstructionEngine


class CellPointInference:
    """
    Unified inference interface for CellPoint models.

    Provides a single entry point for all inference operations including
    feature extraction, reconstruction, and batch processing.

    Example
    -------
    >>> model = CellPointInference(
    ...     config_dir="cellpoint/config",
    ...     config_name="system/pretrain",
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
        config_dir: str,
        config_name: str,
        checkpoint_path: str,
        device: str = "auto",
        batch_size: int = 32,
    ):
        """
        Initialize unified inference model.

        Parameters
        ----------
        config_dir : str
            Path to config directory (e.g., "cellpoint/config")
        config_name : str
            Config name without .yaml (e.g., "system/pretrain")
        checkpoint_path : str
            Path to checkpoint file
        device : str
            Device to use ('auto', 'cuda', 'cpu')
        batch_size : int
            Default batch size for batch processing
        """
        # Initialize base model
        self._model = InferenceModel(config_dir, config_name, checkpoint_path, device)
        self._feature_extractor = FeatureExtractor(self._model)
        self._reconstruction_engine = ReconstructionEngine(self._model)

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
        use_patch_fusion: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Perform self-reconstruction from cls features (optionally fused with patch features).

        Parameters
        ----------
        data : Union[str, np.ndarray, torch.Tensor, List]
            Input point cloud(s)
        normalize : bool
            Normalize input point clouds
        return_numpy : bool
            Return numpy array or torch tensor
        use_patch_fusion : bool
            If True, fuse max-pooled patch features with cls features for reconstruction.
            This typically improves reconstruction quality by incorporating local details.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Reconstructed point cloud(s)
        """
        return self._reconstruction_engine.self_reconstruct(
            data, normalize, return_numpy, use_patch_fusion
        )

    @torch.no_grad()
    def reconstruct_from_features(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Reconstruct point cloud directly from extracted features.

        Parameters
        ----------
        cls_features : torch.Tensor
            Global features of shape (B, C) or (B, 1, C)
        patch_features : torch.Tensor, optional
            Patch features of shape (B, P, C). If provided, will be fused with
            cls_features for enhanced reconstruction.
        return_numpy : bool
            Return numpy array or torch tensor

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Reconstructed point cloud(s)

        Examples
        --------
        >>> # Extract features once
        >>> features = model.extract_features(data)
        >>>
        >>> # Reconstruct with fusion
        >>> recon1 = model.reconstruct_from_features(
        ...     features['cls'], features['patch']
        ... )
        >>>
        >>> # Reconstruct without fusion
        >>> recon2 = model.reconstruct_from_features(features['cls'])
        >>>
        >>> # Feature interpolation
        >>> feat_interp = 0.5 * feat1['cls'] + 0.5 * feat2['cls']
        >>> recon_interp = model.reconstruct_from_features(feat_interp)
        """
        return self._reconstruction_engine.reconstruct_from_features(
            cls_features, patch_features, return_numpy
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
