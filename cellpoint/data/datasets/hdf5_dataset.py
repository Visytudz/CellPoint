import os
import json
import h5py
import torch
import logging
import numpy as np
from glob import glob
import torch.utils.data as data

from cellpoint.utils.misc import normalize_to_unit_sphere, pca_align

logger = logging.getLogger(__name__)


class HDF5Dataset(data.Dataset):

    def __init__(
        self,
        path: str,
        splits: list[str] = ["train"],
        num_points: int = None,
        normalize: bool = True,
        normalize_scale: float = None,
        class_choice: list[str] = None,
        load_to_ram: bool = False,
        pca_align: bool = False,
    ) -> None:
        """
        Initialize the dataset with lazy loading.

        Parameters
        ----------
        path : str
            The root path of the dataset.
        splits : list[str], optional
            The splits of the dataset. The default is ["train"].
        num_points : int, optional
            The number of points to load. If None, use all points. Note that
            we use random sampling, so it's recommended not to set this.
        normalize : bool, optional
            Whether to normalize the point cloud. The default is True.
        normalize_scale : float, optional
            The scale to normalize the point cloud to. If None, normalize to unit sphere. The default is None.
        class_choice : list[str], optional
            The name of the class to load.
        load_to_ram : bool, optional
            Whether to load the entire dataset into RAM. The default is False.
        pca_align : bool, optional
            Whether to align the point cloud using PCA. The default is False.
        """
        self.path = os.path.abspath(path)
        self.splits = splits
        self.num_points = num_points
        self.normalize = normalize
        self.normalize_scale = normalize_scale
        self.class_choice = class_choice
        self.load_to_ram = load_to_ram
        self.pca_align = pca_align

        self._load_metadata()
        self._get_paths()
        self._load_h5()

        if self.class_choice is not None:
            indices_to_keep = np.isin(self.name, self.class_choice)
            self.points = [p for i, p in enumerate(self.points) if indices_to_keep[i]]
            self.label = self.label[indices_to_keep]
            self.id = self.id[indices_to_keep]
            self.name = self.name[indices_to_keep]

            if load_to_ram:
                self.data = self.data[indices_to_keep]

    def _load_metadata(self) -> None:
        """Loads metadata from metadata.json located in the dataset's root directory."""
        metadata_path = os.path.join(self.path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {self.path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        label2name: dict[str, str] = metadata["label2name"]
        self.label2name: dict[int, str] = {int(k): v for k, v in label2name.items()}
        self.label2name[-1] = "unlabeled"  # For unlabeled data
        self.name2label: dict[str, int] = {
            name: i for i, name in self.label2name.items()
        }
        self.sorted_names = [
            value for key, value in sorted(self.label2name.items()) if key != -1
        ]

    def _get_paths(self) -> None:
        """Get the paths of h5 files for a given data split."""
        self.h5_paths: list[str] = []
        for split in self.splits:
            split_dir = os.path.join(self.path, split)
            if not os.path.isdir(split_dir):
                logger.warning(f"Split directory {split_dir} does not exist.")
                continue
            # .h5 or .hdf5
            h5_pattern = os.path.join(split_dir, "*.h5")
            hdf5_pattern = os.path.join(split_dir, "*.hdf5")
            self.h5_paths.extend(glob(h5_pattern))
            self.h5_paths.extend(glob(hdf5_pattern))

    def _load_h5(self) -> None:
        """Loads only labels and IDs from HDF5 files and builds the data index."""
        points_list, label_list, id_list = [], [], []
        data_list = []  # Only used if load_to_ram is True
        for h5_path in self.h5_paths:
            with h5py.File(h5_path, "r") as f:
                samples_num = f["data"].shape[0]
                # Build the index for lazy loading
                for i in range(samples_num):
                    points_list.append((h5_path, i))
                # Load labels and IDs
                if "label" in f:
                    label_list.append(f["label"][:].astype("int64"))
                else:
                    label_list.append(np.array([-1] * samples_num))
                if "id" in f:
                    id_list.append(f["id"][:].astype("str"))
                else:
                    id_list.append(np.array([""] * samples_num))

                if self.load_to_ram:
                    logger.info(f"Loading data from {h5_path} into RAM...")
                    chunk_data = f["data"][:].astype(np.float32)
                    data_list.append(chunk_data)

        self.points = points_list  # List of (h5_path, index_in_file)
        self.label = np.concatenate(label_list, axis=0)  # (B, 1)
        self.id = np.concatenate(id_list, axis=0)  # (B, )
        self.name = np.array(
            [self.label2name[label_idx] for label_idx in self.label.squeeze()]
        )  # (B, )

        if self.load_to_ram:
            self.data = np.concatenate(data_list, axis=0)  # (B, N, 3)

    def __getitem__(self, item: int) -> dict[str, any]:
        """Retrieves a single data point from the dataset using lazy loading."""
        # Get point cloud & other data
        other_data = {}  # only used if lazy loading
        if self.load_to_ram:
            pcl = self.data[item].copy().astype(np.float32)
        else:
            h5_path, index_in_file = self.points[item]
            with h5py.File(h5_path, "r") as f:
                pcl = f["data"][index_in_file].astype(np.float32)
                # get all keys except 'data', 'label', 'id'
                other_data = {
                    key: f[key][index_in_file]
                    for key in f.keys()
                    if key not in ["data", "label", "id"]
                }

        # randomly sample, normalize and convert to tensor
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

        # Get label
        label = self.label[item]
        label_tensor = torch.from_numpy(label)

        sample = {
            "points": pcl_tensor,
            "label": label_tensor,
            "name": str(self.name[item]),
            "id": str(self.id[item]),
            "scale_factor": scale_factor,
        }

        return {**sample, **other_data}

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.points)
