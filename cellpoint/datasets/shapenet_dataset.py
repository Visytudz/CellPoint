import os
import torch
import numpy as np
import torch.utils.data as data
from typing import Dict, Optional, Any, List

from cellpoint.utils.io import save_ply
from cellpoint.utils.process import normalize_to_unit_sphere


class ShapeNetDataset(data.Dataset):
    def __init__(
        self,
        pc_path: str,
        split_path: str,
        splits: List[str] = ["train"],
        num_points: int = 2048,
    ):
        """
        Initializes the ShapeNetPointCloud dataset.

        Parameters
        ----------
        pc_path : str
            The directory where point cloud .npy files are stored.
        split_path : str
            The root directory of the list split files.
        splits : List[str], optional
            The dataset split(s) to use. Can be a list containing any combination of
            "train", "val", and "test". Default is ["train"].
        num_points : int, optional
            The number of points to sample from each point cloud. Default is 2048.
        """
        self.pc_path = pc_path
        self.num_points = num_points
        self.file_list: List[Dict[str, str]] = []
        for split in splits:
            self.file_list.extend(self._load_file_list(split_path, split))

    def _load_file_list(self, split_path: str, split: str) -> list[Dict[str, str]]:
        """Loads the list of files for the specified dataset split."""
        list_file_path = os.path.join(split_path, f"{split}.txt")
        if not os.path.exists(list_file_path):
            raise FileNotFoundError(f"List file not found: {list_file_path}")

        with open(list_file_path, "r") as f:
            lines = f.readlines()

        file_list = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            label = line.split("-")[0]
            id = line.split("-")[1].split(".")[0]
            file_list.append({"label": label, "id": id, "path": line})
        return file_list

    def to_ply(self, item: int, filename: str) -> None:
        """Saves the point cloud to a PLY file."""
        sample = self.__getitem__(item)
        points = sample["points"].numpy()
        save_ply(points, filename)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Retrieves a single data sample from the dataset."""
        # Get sample metadata
        sample_info = self.file_list[item]

        # Load the point cloud file from disk
        path = os.path.join(self.pc_path, sample_info["path"])
        point_cloud = np.load(path).astype(np.float32)  # (N, 3)

        # Randomly sample, normalize, and convert to tensor
        if len(point_cloud) > self.num_points:
            indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
            point_cloud = point_cloud[indices]
        point_cloud = normalize_to_unit_sphere(point_cloud)
        point_cloud_tensor = torch.from_numpy(point_cloud)

        sample = {
            "points": point_cloud_tensor,
            "label": sample_info["label"],
            "id": sample_info["id"],
        }

        return sample

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_list)


if __name__ == "__main__":
    dataset = ShapeNetDataset(
        pc_path="/home/verve/Project/research/CellPoint/datasets/shapenet55-34/pcl",
        split_path="/home/verve/Project/research/CellPoint/datasets/shapenet55-34/splits",
        splits=["train","test"],
        num_points=2048,
    )
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample ID: {sample['id']}")
    print(f"Sample Label: {sample['label']}")
    print(f"Sample Name: {sample['points']}")
    print(f"Point cloud shape: {sample['points'].shape}")
