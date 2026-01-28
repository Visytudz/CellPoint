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
        cls_features: Union[torch.Tensor, np.ndarray],
        patch_features: Union[torch.Tensor, np.ndarray, None] = None,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Reconstruct point cloud directly from extracted features.

        This is useful when you want to:
        - Avoid re-extracting features for multiple reconstructions
        - Perform feature interpolation or manipulation before reconstruction
        - Batch process features separately from reconstruction
        - Use pre-pooled patch features to skip pooling step

        Parameters
        ----------
        cls_features : Union[torch.Tensor, np.ndarray]
            Global features of shape (C,), (B, C), or (B, 1, C)
            Automatically adds batch dimension if single sample (C,) is provided
        patch_features : Union[torch.Tensor, np.ndarray, None], optional
            Patch features of shape (C,), (B, C), or (B, P, C).
            - If (B, P, C): will be max-pooled then fused with cls_features.
            - If (C,) or (B, C): directly fused without pooling (e.g., pre-pooled features).
        return_numpy : bool
            Return numpy array or torch tensor

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Reconstructed point cloud(s)
            - Single input: (N, 3)
            - Batch input: (B, N, 3)

        Examples
        --------
        >>> # Method 1: Use original patch features (auto-pooled)
        >>> features = model.extract_features(data)
        >>> recon = model.reconstruct_from_features(
        ...     features['cls'], features['patch']
        ... )
        >>>
        >>> # Method 2: Use pre-pooled patch features (skip pooling)
        >>> pooled_patch = torch.max(features['patch'], dim=1)[0]  # (B, C)
        >>> recon = model.reconstruct_from_features(
        ...     features['cls'], pooled_patch
        ... )
        >>>
        >>> # Method 3: Feature interpolation
        >>> feat_interp = 0.5 * feat1['cls'] + 0.5 * feat2['cls']
        >>> recon_interp = model.reconstruct_from_features(feat_interp)
        """
        return self._reconstruction_engine.reconstruct_from_features(
            cls_features, patch_features, return_numpy
        )

    @torch.no_grad()
    def fusion_reconstruct(
        self,
        data_list: List[Union[str, np.ndarray, torch.Tensor]],
        weights: Union[List[float], np.ndarray, torch.Tensor],
        normalize: bool = True,
        normalize_weights: bool = True,
        use_patch_fusion: bool = True,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Reconstruct point cloud from weighted fusion of multiple point clouds.

        This enables morphology interpolation, averaging, and blending by fusing
        features from multiple point clouds before reconstruction.

        Parameters
        ----------
        data_list : List[Union[str, np.ndarray, torch.Tensor]]
            List of input point clouds to fuse
        weights : Union[List[float], np.ndarray, torch.Tensor]
            Weights for each point cloud. Must have same length as data_list.
            Will be automatically normalized to sum to 1.0 if normalize_weights=True.
        normalize : bool
            Normalize input point clouds
        normalize_weights : bool
            If True, normalize weights to sum to 1.0
        use_patch_fusion : bool
            If True, also fuse patch features for enhanced reconstruction
        return_numpy : bool
            Return numpy array or torch tensor

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Reconstructed point cloud from fused features

        Examples
        --------
        >>> # Interpolate between two cell morphologies
        >>> cell_interp = model.fusion_reconstruct(
        ...     [cell_normal, cell_abnormal],
        ...     weights=[0.7, 0.3]
        ... )
        >>>
        >>> # Average multiple cells
        >>> cell_avg = model.fusion_reconstruct(
        ...     [cell1, cell2, cell3],
        ...     weights=[1/3, 1/3, 1/3]  # Auto-normalized if sum != 1
        ... )
        >>>
        >>> # Weighted blending with different importance
        >>> cell_blend = model.fusion_reconstruct(
        ...     [template, variant1, variant2],
        ...     weights=[0.6, 0.3, 0.1]
        ... )
        """
        return self._reconstruction_engine.fusion_reconstruct(
            data_list,
            weights,
            normalize,
            normalize_weights,
            use_patch_fusion,
            return_numpy,
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
