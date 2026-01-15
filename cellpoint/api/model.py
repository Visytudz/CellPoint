"""Model loading and initialization"""

import torch
import hydra
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class InferenceModel:
    """Model loader and manager for inference"""

    def __init__(
        self,
        config_dir: str,
        config_name: str,
        checkpoint_path: str,
        device: str = "auto",
    ):
        """
        Initialize inference model from config and checkpoint.

        Parameters
        ----------
        config_dir : str
            Path to config directory (e.g., "cellpoint/config")
        config_name : str
            Config name without .yaml extension (e.g., "system/pretrain")
        checkpoint_path : str
            Path to checkpoint file
        device : str
            Device to use ('auto', 'cuda', 'cpu')
        """
        self.config_dir = Path(config_dir).absolute()
        self.config_name = config_name
        self.checkpoint_path = Path(checkpoint_path)

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self._load_model()
        logger.info("Model loaded successfully")

    def _load_model(self):
        """Load model from config and checkpoint"""
        logger.info(f"Config directory: {self.config_dir}")
        logger.info(f"Config name: {self.config_name}")

        # Load config using hydra
        with hydra.initialize_config_dir(
            config_dir=str(self.config_dir), version_base=None
        ):
            cfg = hydra.compose(config_name=self.config_name)

        # Extract the actual model config
        # For "system/pretrain", cfg will be wrapped as cfg.system
        config_key = self.config_name.split("/")[0] if "/" in self.config_name else None
        model_cfg = cfg[config_key] if config_key and config_key in cfg else cfg

        # Instantiate model from config
        logger.info(f"Instantiating model from config")
        self.model = hydra.utils.instantiate(model_cfg)

        # Load checkpoint weights using model's built-in method
        self.model.load_pretrained_weights(str(self.checkpoint_path))

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Store model components for easy access
        self.extractor = self.model.extractor
        self.decoder = self.model.decoder
        self.global_decoder = self.model.global_decoder
        self.view_generator = self.model.view_generator

    def get_device(self) -> torch.device:
        """Get current device"""
        return self.device

    def get_model(self):
        """Get the underlying model"""
        return self.model
