import os
import torch
import logging
import numpy as np
from glob import glob
from pathlib import Path
import torch.utils.data as data

from cellpoint.utils.io import load_ply
from cellpoint.utils.misc import normalize_to_unit_sphere, pca_align

logger = logging.getLogger(__name__)


class PLYDataset(data.Dataset):

    def __init__(
        self,
        ply_dir: str = None,
        ply_list: list[str] = None,
        num_points: int = None,
        normalize: bool = True,
        normalize_scale: float = None,
        load_to_ram: bool = False,
        pca_align: bool = False,
    ) -> None:
        """
        Initialize the PLY dataset with lazy loading.

        Parameters
        ----------
        ply_dir : str, optional
            The directory path containing PLY files. All .ply files in this directory will be loaded.
        ply_list : list[str], optional
            A list of PLY file paths.
        num_points : int, optional
            The number of points to load. If None, use all points. Note that
            we use random sampling, so it's recommended not to set this.
        normalize : bool, optional
            Whether to normalize the point cloud. The default is True.
        normalize_scale : float, optional
            The scale to normalize the point cloud to. If None, normalize to unit sphere. The default is None.
        load_to_ram : bool, optional
            Whether to load the entire dataset into RAM. The default is False.
        pca_align : bool, optional
            Whether to align the point cloud using PCA. The default is False.
        """
        if ply_dir is None and ply_list is None:
            raise ValueError("Either ply_dir or ply_list must be provided.")

        if ply_dir is not None and ply_list is not None:
            raise ValueError("Only one of ply_dir or ply_list should be provided.")

        self.ply_dir = ply_dir
        self.ply_list = ply_list
        self.num_points = num_points
        self.normalize = normalize
        self.normalize_scale = normalize_scale
        self.load_to_ram = load_to_ram
        self.pca_align = pca_align

        self._get_paths()
        self._load_ply()

    def _get_paths(self) -> None:
        """Get the paths of PLY files."""
        self.ply_paths: list[str] = []

        if self.ply_dir is not None:
            # Get all .ply files from directory
            ply_dir = os.path.abspath(self.ply_dir)
            if not os.path.isdir(ply_dir):
                raise ValueError(f"Directory {ply_dir} does not exist.")

            ply_pattern = os.path.join(ply_dir, "*.ply")
            self.ply_paths = sorted(glob(ply_pattern))

            if len(self.ply_paths) == 0:
                logger.warning(f"No .ply files found in {ply_dir}")

        elif self.ply_list is not None:
            # Use provided list of file paths
            for ply_path in self.ply_list:
                ply_path = os.path.abspath(ply_path)
                if os.path.exists(ply_path):
                    self.ply_paths.append(ply_path)
                else:
                    logger.warning(f"PLY file not found: {ply_path}")

    def _load_ply(self) -> None:
        """Loads PLY files and builds the data index."""
        name_list = []
        data_list = []  # Only used if load_to_ram is True

        for ply_path in self.ply_paths:
            # Extract filename without extension
            file_name = Path(ply_path).stem
            name_list.append(file_name)

            if self.load_to_ram:
                logger.info(f"Loading data from {ply_path} into RAM...")
                pcl = load_ply(ply_path)
                data_list.append(pcl)

        self.name = np.array(name_list)  # (B, )

        if self.load_to_ram:
            self.data = data_list  # List of arrays (due to varying point counts)

    def __getitem__(self, item: int) -> dict[str, any]:
        """Retrieves a single data point from the dataset using lazy loading."""
        # Get point cloud
        if self.load_to_ram:
            pcl = self.data[item].copy().astype(np.float32)
        else:
            ply_path = self.ply_paths[item]
            pcl = load_ply(ply_path)

        # Randomly sample, normalize and convert to tensor
        if self.num_points is not None and self.num_points < pcl.shape[0]:
            choice = np.random.choice(
                pcl.shape[0], self.num_points, replace=False
            )  # (num_points, )
            pcl = pcl[choice, :]
        scale_factor = 1.0
        if self.normalize:
            pcl, scale_factor = normalize_to_unit_sphere(
                pcl, scale=self.normalize_scale
            )
        if self.pca_align:
            pcl = pca_align(pcl)
        pcl_tensor = torch.from_numpy(pcl)

        sample = {
            "points": pcl_tensor,
            "name": str(self.name[item]),
            "scale_factor": scale_factor,
        }

        return sample

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.ply_paths)
