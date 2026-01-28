"""Feature extraction functionality"""

import torch
import numpy as np
from typing import Union, List, Dict

from .utils import prepare_input, prepare_batch_input


class FeatureExtractor:
    """Feature extraction from point clouds"""

    def __init__(self, model):
        """
        Initialize feature extractor.

        Parameters
        ----------
        model : InferenceModel
            Loaded inference model
        """
        self.model = model
        self.extractor = model.extractor
        self.device = model.device

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
            Dictionary with requested features:
            - "cls": (B, C)
            - "patch": (B, P, C)
            - "concat": (B, C_cls + C_patch)
        """
        # Handle batch input
        if isinstance(data, list):
            data = prepare_batch_input(data, self.device, normalize)
        else:
            data = prepare_input(data, self.device, normalize)

        results = {}

        # Extract features
        cls_features, patch_features, centers, group = self.extractor(data)
        # cls_features: (B, 1, C), patch_features: (B, P, C)

        if return_cls:
            results["cls"] = cls_features.squeeze(1)  # (B, C)

        if return_patch:
            results["patch"] = patch_features  # (B, P, C)

        if return_concat:
            # Max pool patch features and concatenate with cls features
            patch_max = patch_features.max(dim=1)[0]  # (B, C)
            cls_squeezed = cls_features.squeeze(1)  # (B, C)
            concat_features = torch.cat([cls_squeezed, patch_max], dim=-1)  # (B, 2C)
            results["concat"] = concat_features

        return results

    @torch.no_grad()
    def extract_cls_features(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract only cls features"""
        features = self.extract_features(
            data,
            return_cls=True,
            return_patch=False,
            return_concat=False,
            normalize=normalize,
        )
        return features["cls"]

    @torch.no_grad()
    def extract_patch_features(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract only patch features"""
        features = self.extract_features(
            data,
            return_cls=False,
            return_patch=True,
            return_concat=False,
            normalize=normalize,
        )
        return features["patch"]

    @torch.no_grad()
    def extract_concat_features(
        self,
        data: Union[str, np.ndarray, torch.Tensor, List],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract only concatenated features"""
        features = self.extract_features(
            data,
            return_cls=False,
            return_patch=False,
            return_concat=True,
            normalize=normalize,
        )
        return features["concat"]
