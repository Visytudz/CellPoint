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
        config_path: str,
        checkpoint_path: str,
        device: str = "auto",
    ):
        """
        Initialize inference model from config and checkpoint.

        Parameters
        ----------
        config_path : str
            Path to config YAML file
        checkpoint_path : str
            Path to checkpoint file
        device : str
            Device to use ('auto', 'cuda', 'cpu')
        """
        self.config_path = Path(config_path)
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
        # Load config using hydra
        with hydra.initialize_config_dir(
            config_dir=str(self.config_path.parent.absolute()), version_base=None
        ):
            cfg = hydra.compose(config_name=self.config_path.stem)

        # Instantiate model from config
        logger.info(f"Instantiating model from config: {self.config_path}")
        self.model = hydra.utils.instantiate(cfg)

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
