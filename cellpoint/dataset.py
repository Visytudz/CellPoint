import os
import json
import h5py
import torch
import numpy as np
from glob import glob
import torch.utils.data as data
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Optional, Any

from utils import save_ply


def normalize_to_unit_sphere(
    pointcloud: np.ndarray[np.float32],
) -> np.ndarray[np.float32]:
    """Normalizes a point cloud to fit within a unit sphere."""
    # Center the point cloud
    centroid = np.mean(pointcloud, axis=0)  # (3,)
    pointcloud_centered = pointcloud - centroid  # (N, 3)

    # Scale to fit within unit sphere
    max_dist = np.max(np.sqrt(np.sum(pointcloud_centered**2, axis=1)))  # (N,)
    scale_factor = 1.0 / max_dist if max_dist > 1e-6 else 1.0

    normalized_pointcloud = pointcloud_centered * scale_factor  # (N, 3)

    return normalized_pointcloud


def translate_pointcloud(pointcloud: NDArray[np.float32]) -> NDArray[np.float32]:
    """Applies random scaling and translation to a point cloud."""
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = ((pointcloud * xyz1) + xyz2).astype("float32")
    return translated_pointcloud


def jitter_pointcloud(
    pointcloud: NDArray[np.float32], sigma: float = 0.01, clip: float = 0.02
) -> NDArray[np.float32]:
    """Applies random jitter to a point cloud."""
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud: NDArray[np.float32]) -> NDArray[np.float32]:
    """Applies a random rotation around the Y-axis to a point cloud."""
    theta = np.pi * 2 * np.random.rand()
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pointcloud[:, [0, 2]] = (
        pointcloud[:, [0, 2]] @ rotation_matrix
    )  # random rotation (x,z)
    return pointcloud


class Dataset(data.Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        class_choice: Optional[str] = None,
        num_points: int = 2048,
        split: list[str] = ["train"],
        load_name: bool = False,
        load_id: bool = False,
        normalize: bool = True,
        random_rotate: bool = False,
        random_jitter: bool = False,
        random_translate: bool = False,
    ) -> None:
        """
        Initialize the dataset with lazy loading.

        Parameters
        ----------
        root : str
            The root directory of the dataset.
        dataset_name : str
            The name of the dataset.
        class_choice : Optional[str], optional
            The name of the class to load.
        num_points : int, optional
            The number of points to load.
        split : list[str], optional
            The split of the dataset.
        load_name : bool, optional
            Whether to load the label name of the dataset.
        load_id : bool, optional
            Whether to load the id of the dataset.
        normalize : bool, optional
            Whether to normalize the point cloud.
        random_rotate : bool, optional
            Whether to apply random rotation to the dataset.
        random_jitter : bool, optional
            Whether to apply random jitter to the dataset.
        random_translate : bool, optional
            Whether to apply random translation to the dataset.
        """
        self.root: str = os.path.join(root, dataset_name)
        self.dataset_name: str = dataset_name
        self.class_choice: Optional[str] = class_choice
        self.num_points: int = num_points
        self.split: list[str] = split
        self.load_name: bool = load_name
        self.load_id: bool = load_id
        self.normalize: bool = normalize
        self.random_rotate: bool = random_rotate
        self.random_jitter: bool = random_jitter
        self.random_translate: bool = random_translate

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
        self.datapoints: List[Tuple[str, int]] = []
        # Labels and IDs are small, so we can load them for easier filtering.
        label_list, id_list = self._load_metadata_from_h5(self.path_h5py_all)
        self.label: NDArray[np.int64] = np.concatenate(label_list, axis=0)  # (B, 1)
        self.id: NDArray[np.str_] = np.concatenate(id_list, axis=0)  # (B, )
        self.name: NDArray[np.str_] = np.array(
            [self.label2name[label_idx] for label_idx in self.label.squeeze()]
        )  # (B, )

        # Filter the data by class choice
        if self.class_choice is not None:
            indices_to_keep = self.name == self.class_choice
            # Filter the index, labels, and ids
            self.datapoints = [
                dp for i, dp in enumerate(self.datapoints) if indices_to_keep[i]
            ]
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

        self.label2name: List[str] = metadata["label2name"]
        self.name2label: Dict[str, int] = {
            name: i for i, name in enumerate(self.label2name)
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

    def _load_metadata_from_h5(
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
                num_points_in_file = f["data"].shape[0]

                # Build the index for lazy loading
                for i in range(num_points_in_file):
                    self.datapoints.append((h5_path, i))

                # Load labels and IDs
                all_label.append(f["label"][:].astype("int64"))
                if "id" in f:
                    all_id.append(f["id"][:].astype("str"))
                else:
                    all_id.append(np.array([""] * num_points_in_file))
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
        h5_path, index_in_file = self.datapoints[item]
        with h5py.File(h5_path, "r") as f:
            point_cloud = f["data"][index_in_file]  # (N, 3)
        if normalize:
            point_cloud = normalize_to_unit_sphere(point_cloud)
        save_ply(point_cloud, filename)
        print(f"Point cloud saved to {filename}")

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Retrieves a single data point from the dataset using lazy loading."""
        h5_path, index_in_file = self.datapoints[item]
        with h5py.File(h5_path, "r") as f:
            point_set = f["data"][index_in_file][: self.num_points].astype(np.float32)
        if self.normalize:
            point_set = normalize_to_unit_sphere(point_set)
        label = self.label[item]  # Get pre-loaded label

        # Data augmentation
        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        point_set_tensor = torch.from_numpy(point_set)
        label_tensor = torch.from_numpy(label)

        sample = {"points": point_set_tensor, "label": label_tensor}
        if self.load_name:
            sample["name"] = str(self.name[item])
        if self.load_id:
            sample["id"] = str(self.id[item])

        return sample

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.datapoints)


if __name__ == "__main__":
    root = "datasets"
    dataset_name = "3dcell"
    split = ["train"]
    dataset = Dataset(
        root=root,
        dataset_name=dataset_name,
        num_points=20480,
        split=split,
        load_name=True,
        load_id=True,
        random_rotate=True,
        random_jitter=True,
        random_translate=True,
        # class_choice="cancerous"
    )
    print(f"Dataset size: {len(dataset)}")
    print(dataset[15])
