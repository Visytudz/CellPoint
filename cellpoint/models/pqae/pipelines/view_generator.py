import torch
import torch.nn as nn
import numpy as np

from cellpoint.utils.process import (
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
    """

    def __init__(self, min_crop_rate: float = 0.6):
        super().__init__()
        self.min_crop_rate = min_crop_rate

    def _crop_and_normalize(
        self, pts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single random crop and normalization on a batch of point clouds.
        """
        # Determine the crop rate for this operation
        if self.min_crop_rate >= 1.0:
            crop_rate = 1.0
        else:
            crop_rate = np.random.uniform(self.min_crop_rate, 1.0)

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
        pts_min = selected_pts.min(dim=1).values
        pts_max = selected_pts.max(dim=1).values
        original_centers = (pts_min + pts_max) / 2.0

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
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - relative_center: The vector difference between the original geometric
              centers of view 2 and view 1. Shape: (B, 3).
            - view1: The first processed (cropped, normalized, rotated) view.
            - view2: The second processed view.
        """
        # Generate the first view
        view1, center1 = self._crop_and_normalize(pts)
        # Generate the second view
        view2, center2 = self._crop_and_normalize(pts)

        # Apply independent random rotations to each view
        view1 = batch_rotate_torch(view1)
        view2 = batch_rotate_torch(view2)

        # Calculate the relative position of the original centers
        relative_center = center2 - center1

        return relative_center, view1, view2
