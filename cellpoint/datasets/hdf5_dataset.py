import os
import json
import h5py
import torch
import numpy as np
from glob import glob
import torch.utils.data as data
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Optional, Any, Union

from cellpoint.utils.io import save_ply
from cellpoint.utils.process import normalize_to_unit_sphere


class HDF5Dataset(data.Dataset):
    def __init__(
        self,
        root: str,
        name: str,
        class_choice: Optional[Union[str, list[str]]] = None,
        num_points: int = None,
        splits: list[str] = ["train"],
        normalize: bool = True,
    ) -> None:
        """
        Initialize the dataset with lazy loading.

        Parameters
        ----------
        root : str
            The root directory of the dataset.
        name : str
            The name of the dataset.
        class_choice : Optional[str], optional
            The name of the class to load.
        num_points : int, optional
            The number of points to load. We use random sampling, so it's recommended not to use this.
        splits : list[str], optional
            The splits of the dataset.
        normalize : bool, optional
            Whether to normalize the point cloud.
        """
        self.root: str = os.path.join(root, name)
        self.dataset_name: str = name
        self.class_choice: Optional[Union[str, list[str]]] = class_choice
        if isinstance(class_choice, str):
            self.class_choice = [class_choice]
        self.num_points: int = num_points
        self.split: list[str] = splits
        self.normalize: bool = normalize

        # Load metadata from the JSON file
        self._load_metadata()
        # Get the paths of h5 files for all splits
        self.path_h5py_all: List[str] = []
        for s in self.split:
            self._get_path(s)
        if not self.path_h5py_all:
            raise FileNotFoundError(
                f"No HDF5 files found for split '{self.split}' in '{self.root}'. "
                f"Please check the directory structure."
            )

        # self.datapoints will store tuples of (h5_file_path, index_within_file)
        self.points: List[Tuple[str, int]] = []
        # Labels and IDs are small, so we can load them for easier filtering.
        label_list, id_list = self._load_h5(self.path_h5py_all)
        self.label: NDArray[np.int64] = np.concatenate(label_list, axis=0)  # (B, 1)
        self.id: NDArray[np.str_] = np.concatenate(id_list, axis=0)  # (B, )
        self.name: NDArray[np.str_] = np.array(
            [self.label2name[label_idx] for label_idx in self.label.squeeze()]
        )  # (B, )

        # Filter the data by class choice
        if self.class_choice is not None:
            indices_to_keep = np.isin(self.name, self.class_choice)
            # Filter the index, labels, and ids
            self.points = [dp for i, dp in enumerate(self.points) if indices_to_keep[i]]
            self.label = self.label[indices_to_keep]
            self.id = self.id[indices_to_keep]
            self.name = self.name[indices_to_keep]

    def _load_metadata(self) -> None:
        """Loads metadata from metadata.json located in the dataset's root directory."""
        metadata_path = os.path.join(self.root, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {self.root}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        label2name: Dict[str, str] = metadata["label2name"]
        self.label2name: Dict[int, str] = {int(k): v for k, v in label2name.items()}
        self.label2name[-1] = "unlabeled"  # For unlabeled data
        self.name2label: Dict[str, int] = {
            name: i for i, name in self.label2name.items()
        }

    def _get_path(self, split: str) -> None:
        """Get the paths of h5 files for a given data split."""
        split_dir = os.path.join(self.root, split)
        if not os.path.isdir(split_dir):
            return
        # .h5 or .hdf5
        h5_pattern = os.path.join(split_dir, "*.h5")
        hdf5_pattern = os.path.join(split_dir, "*.hdf5")
        self.path_h5py_all.extend(sorted(glob(h5_pattern)))
        self.path_h5py_all.extend(sorted(glob(hdf5_pattern)))
        return

    def _load_h5(
        self, paths: List[str]
    ) -> Tuple[List[NDArray[np.int64]], List[NDArray[np.str_]]]:
        """
        Loads only labels and IDs from HDF5 files and builds the data index.
        The actual point cloud data is NOT loaded here.
        """
        all_label: List[NDArray[np.int64]] = []
        all_id: List[NDArray[np.str_]] = []
        for h5_path in paths:
            with h5py.File(h5_path, "r") as f:
                samples_num = f["data"].shape[0]
                # Build the index for lazy loading
                for i in range(samples_num):
                    self.points.append((h5_path, i))
                # Load labels and IDs
                if "label" in f:
                    all_label.append(f["label"][:].astype("int64"))
                else:
                    all_label.append(np.array([-1] * samples_num))
                if "id" in f:
                    all_id.append(f["id"][:].astype("str"))
                else:
                    all_id.append(np.array([""] * samples_num))
        return all_label, all_id

    def to_ply(self, item: int, filename: str, normalize: bool) -> None:
        """
        Saves a point cloud to a PLY file in ASCII format.

        Parameters
        ----------
        item : int
            The index of the data point to save.
        filename : str
            The path to save the PLY file.
        normalize : bool
            Whether to normalize the point cloud before saving.
        """
        h5_path, index_in_file = self.points[item]
        with h5py.File(h5_path, "r") as f:
            point_cloud = f["data"][index_in_file]  # (N, 3)
        if normalize:
            point_cloud = normalize_to_unit_sphere(point_cloud)
        save_ply(point_cloud, filename)
        print(f"Point cloud saved to {filename}")

    @property
    def class_names(self) -> List[str]:
        """Returns the sorted list of class names in the dataset."""
        names = [value for key, value in sorted(self.label2name.items()) if key != -1]
        return names

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Retrieves a single data point from the dataset using lazy loading."""
        h5_path, index_in_file = self.points[item]

        # Get point cloud
        with h5py.File(h5_path, "r") as f:
            pcl = f["data"][index_in_file].astype(np.float32)
        if self.num_points is not None and self.num_points < pcl.shape[0]:
            choice = np.random.choice(
                pcl.shape[0], self.num_points, replace=False
            )  # (num_points, )
            pcl = pcl[choice, :]
        if self.normalize:
            pcl = normalize_to_unit_sphere(pcl)
        pcl_tensor = torch.from_numpy(pcl)

        # Get label
        label = self.label[item]
        label_tensor = torch.from_numpy(label)

        sample = {
            "points": pcl_tensor,
            "label": label_tensor,
            "name": str(self.name[item]),
            "id": str(self.id[item]),
        }

        return sample

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.points)
