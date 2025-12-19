import torch
import torch.nn as nn
import numpy as np

from cellpoint.utils.misc import (
    batch_rotate_torch,
    batch_normalize_to_unit_sphere_torch,
)


class PointViewGenerator(nn.Module):
    """
    A PyTorch module that generates two decoupled views from a single point cloud.

    Parameters
    ----------
    min_crop_rate : float, optional
        The minimum percentage of points to keep in a crop, by default 0.6.
    max_crop_rate : float, optional
        The maximum percentage of points to keep in a crop, by default 1.0.
    random_rotate : bool, optional
        Whether to apply random rotation to the cropped views, by default True.
    """

    def __init__(
        self,
        min_crop_rate: float = 0.6,
        max_crop_rate: float = 1.0,
        random_rotate: bool = True,
    ):
        super().__init__()
        self.min_crop_rate = min_crop_rate
        self.max_crop_rate = max_crop_rate
        self.random_rotate = random_rotate
        assert 0.0 < self.min_crop_rate <= self.max_crop_rate <= 1.0, (
            "min_crop_rate and max_crop_rate must satisfy 0.0 < min_crop_rate <= "
            "max_crop_rate <= 1.0"
        )

    def _crop_and_normalize(
        self, pts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single random crop and normalization on a batch of point clouds.
        """
        # Determine the crop rate for this operation
        crop_rate = np.random.uniform(self.min_crop_rate, self.max_crop_rate)
        batch_size, n_points, _ = pts.shape
        num_cropped_points = int(n_points * crop_rate)

        # 1. Randomly select center points for each item in the batch
        center_indices = torch.randint(0, n_points, (batch_size,), device=pts.device)
        center_points = torch.gather(
            pts, 1, center_indices.view(-1, 1, 1).expand(-1, -1, 3)
        )  # (B, 1, 3)

        # 2. Calculate distances and find the nearest neighbors (top-k)
        # distances shape: (B, N)
        distances = torch.norm(pts - center_points, dim=2)
        # indices shape: (B, num_cropped_points)
        _, indices = torch.topk(
            distances, num_cropped_points, largest=False, sorted=False
        )

        # 3. Gather the cropped point clouds
        # indices must be expanded to (B, num_cropped_points, 3) for gather
        indices = indices.unsqueeze(-1).expand(-1, -1, 3)
        selected_pts = torch.gather(pts, 1, indices)

        # 4. Record the geometric center in the original coordinate system
        original_centers = torch.mean(selected_pts, dim=1)  # (B, 3)

        # 5. Normalize the cropped view to fit within a unit sphere
        scaled_pts = batch_normalize_to_unit_sphere_torch(selected_pts)

        return scaled_pts, original_centers  # (B, num_cropped_points, 3), (B, 3)

    def forward(
        self, pts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates two decoupled views and their relative center position.

        Parameters
        ----------
        pts : torch.Tensor
            The input batch of point clouds. Shape: (B, N, 3).

        Returns
        -------
        tuple[torch.Tensor,(torch.Tensor, torch.Tensor),(torch.Tensor, torch.Tensor)]
            A tuple containing:
            - relative_center: The relative position of the centers of the two views. Shape: (B, 3).
            - (view1_rotated, view1): The first view and its unrotated version. Each of shape (B, M1, 3).
            - (view2_rotated, view2): The second view and its unrotated version. Each of shape (B, M2, 3).
        """
        # Generate the first view
        view1, center1 = self._crop_and_normalize(pts)
        # Generate the second view
        view2, center2 = self._crop_and_normalize(pts)

        # Apply independent random rotations to each view
        if self.random_rotate:
            view1_rotated = batch_rotate_torch(view1)
            view2_rotated = batch_rotate_torch(view2)
        else:
            view1_rotated = view1
            view2_rotated = view2

        # Calculate the relative position of the original centers
        relative_center = center2 - center1

        return relative_center, (view1_rotated, view1), (view2_rotated, view2)
