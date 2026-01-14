__version__ = "0.1.0"
__author__ = "verve"

# Import inference API for easy access
from cellpoint.api import CellPointInference

# Also expose individual components for advanced usage
from cellpoint.api import (
    InferenceModel,
    FeatureExtractor,
    ReconstructionEngine,
    BatchProcessor,
)

__all__ = [
    "CellPointInference",
    "InferenceModel",
    "FeatureExtractor",
    "ReconstructionEngine",
    "BatchProcessor",
]
