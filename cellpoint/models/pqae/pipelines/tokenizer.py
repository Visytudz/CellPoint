import torch
import torch.nn as nn

from cellpoint.utils.misc import fps, get_neighbors
from cellpoint.utils.knn import knn_vanilla as knn


class Group(nn.Module):
    """
    Groups a point cloud into overlapping local regions (patches).

    This module first selects a set of centroids using Farthest Point Sampling (FPS),
    and then for each centroid, it gathers its K-Nearest Neighbors (KNN) to form a patch.
    """

    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        points : torch.Tensor
            The input point cloud. Shape: (B, N, C).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - neighborhood: The local regions (patches). Shape: (B, G, K, C).
              Each patch is centered at its own origin.
            - center: The centroids of each patch in the original coordinate system.
              Shape: (B, G, C).
        """
        # 1. Use Farthest Point Sampling to select centroids.
        center = fps(points, self.num_group)  # Shape: (B, G, 3)
        # 2. Use K-Nearest Neighbors to find points for each patch.
        idx = knn(
            points.transpose(1, 2), center.transpose(1, 2), self.group_size
        )  # Shape: (B, G, K)
        # 3. Gather the neighborhood points using the indices.
        neighborhood = get_neighbors(points.transpose(1, 2), idx)  # Shape: (B, G, K, 3)
        # 4. Center each patch at its own origin.
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class PatchEmbed(nn.Module):
    """
    Encodes each point cloud patch into a feature vector (token embedding).

    This module uses a mini-PointNet architecture to process each local patch,
    aggregating the geometric information into a single high-dimensional vector.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.embed_dim, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        point_groups : torch.Tensor
            The batch of point cloud patches from the Group module. Shape: (B, G, K, C_in).

        Returns
        -------
        torch.Tensor
            The token embeddings for each patch. Shape: (B, G, C_out).
        """
        B, G, K, C_in = point_groups.shape
        # Reshape for batch processing of patches: (B, G, K, 3) -> (B*G, K, 3)
        point_groups = point_groups.reshape(B * G, K, C_in)
        # Transpose for Conv1d, which expects (B*G, C_in, K)
        patches = point_groups.transpose(2, 1)

        # 1. Extract point-wise features
        point_features = self.first_conv(patches)  # Shape: (B*G, 256, K)
        # 2. Aggregate features with max-pooling to get a global patch feature
        global_feature = torch.max(point_features, dim=2, keepdim=True)[
            0
        ]  # Shape: (B*G, 256, 1)
        # 3. Concatenate global feature with point-wise features
        combined_features = torch.cat(
            [global_feature.expand(-1, -1, K), point_features], dim=1
        )  # Shape: (B*G, 512, K)
        # 4. Extract more complex features from the combined representation
        final_point_features = self.second_conv(
            combined_features
        )  # Shape: (B*G, embed_dim, K)
        # 5. Final aggregation to get the definitive token for the patch
        patch_embedding = torch.max(final_point_features, dim=2, keepdim=False)[
            0
        ]  # Shape: (B*G, embed_dim)

        # Reshape back to (Batch, Num_Groups, Embed_dim)
        return patch_embedding.reshape(B, G, self.embed_dim)
