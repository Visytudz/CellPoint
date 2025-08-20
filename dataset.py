import os
import json
import h5py
import torch
import numpy as np
from glob import glob
import torch.utils.data as data
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Optional


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
        random_rotate: bool = False,
        random_jitter: bool = False,
        random_translate: bool = False,
    ) -> None:
        """
        Initialize the dataset.

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
        self.random_rotate: bool = random_rotate
        self.random_jitter: bool = random_jitter
        self.random_translate: bool = random_translate

        self.path_h5py_all: List[str] = []
        # Load metadata from the JSON file
        self._load_metadata()
        # Get the paths of h5 files for all splits
        for split in self.split:
            self._get_path(split)
        if not self.path_h5py_all:
            raise FileNotFoundError(
                f"No HDF5 files found for split '{self.split}' in '{self.root}'. "
                f"Please check the directory structure."
            )

        # Load the data from h5 files and json files
        data_list, label_list, id_list = self._load_h5py(self.path_h5py_all)
        self.data: NDArray[np.float32] = np.concatenate(data_list, axis=0)  # (B, N, 3)
        self.label: NDArray[np.int64] = np.concatenate(label_list, axis=0)  # (B, 1)
        self.id: NDArray[np.str_] = np.concatenate(id_list, axis=0)  # (B, )
        self.name: NDArray[np.str_] = np.array(
            [self.label2name[label_idx] for label_idx in self.label.squeeze()]
        )  # (B, )

        # Filter the data by class choice
        if self.class_choice is not None:
            indices = self.name == self.class_choice  # (B, )
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.id = self.id[indices]
            self.name = self.name[indices]

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
        h5_pattern = os.path.join(split_dir, "*.h5")
        self.path_h5py_all.extend(sorted(glob(h5_pattern)))
        return

    def _load_h5py(self, path: List[str]) -> Tuple[
        List[NDArray[np.float32]],
        List[NDArray[np.int64]],
        List[NDArray[np.str_]],
    ]:
        """Loads data from a list of HDF5 files."""
        all_data: List[NDArray[np.float32]] = []
        all_label: List[NDArray[np.int64]] = []
        all_id: List[NDArray[np.str_]] = []
        for h5_name in path:
            with h5py.File(h5_name, "r") as f:
                data = f["data"][:].astype("float32")  # (B, N, 3)
                label = f["label"][:].astype("int64")  # (B, 1)
                if "id" in f:
                    id = f["id"][:].astype("str")  # (B, )
                else:
                    id = np.array([""] * data.shape[0])  # (B, )
                all_data.append(data)
                all_label.append(label)
                all_id.append(id)
        return all_data, all_label, all_id

    def to_ply(self, item: int, filename: str) -> None:
        """
        Saves a point cloud to a PLY file in ASCII format.

        Parameters
        ----------
        item : int
            The index of the data point to save.
        filename : str
            The path to save the PLY file.
        """
        num_points = self.data[item].shape[0]

        # Create the PLY header
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {num_points}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
        ]

        # Ensure the output directory exists
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the header and data to the file
        with open(filename, "w") as f:
            f.write("\n".join(header) + "\n")
            np.savetxt(f, self.data[item], fmt="%.6f")

        print(f"Point cloud saved to {filename}")

    def __getitem__(
        self, item: int
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, str]
        | Tuple[torch.Tensor, torch.Tensor, str, str]
    ):
        """Retrieves a single data point from the dataset."""
        point_set = self.data[item][: self.num_points]  # (N, 3)
        label = self.label[item]  # (1, )

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set_tensor = torch.from_numpy(point_set)  # (N, 3)
        label_tensor = torch.from_numpy(label)  # (1, )

        # Prepare return values based on load flags
        result = [point_set_tensor, label_tensor]
        if self.load_name:
            result.append(str(self.name[item]))
        if self.load_id:
            result.append(str(self.id[item]))

        return tuple(result)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.data.shape[0]


if __name__ == "__main__":
    root = "datasets"
    dataset_name = "shapenetcorev2"
    split = ["train"]
    dataset = Dataset(
        root=root,
        dataset_name=dataset_name,
        num_points=2048,
        split=split,
        load_name=True,
        load_id=True,
        random_rotate=True,
        random_jitter=True,
        random_translate=True,
    )
    print(dataset[15])
    dataset.to_ply(15, "test.ply")
