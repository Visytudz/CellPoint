import torch
import numpy as np
from numpy.typing import NDArray


def normalize_to_unit_sphere(
    pointcloud: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Normalizes a point cloud to fit within a unit sphere."""
    # Center the point cloud
    centroid = np.mean(pointcloud, axis=0)  # (3,)
    pointcloud_centered = pointcloud - centroid  # (N, 3)

    # Scale to fit within unit sphere
    max_dist = np.max(np.sqrt(np.sum(pointcloud_centered**2, axis=1)))  # (N,)
    scale_factor = 1.0 / max_dist if max_dist > 1e-6 else 1.0

    normalized_pointcloud = pointcloud_centered * scale_factor  # (N, 3)

    return normalized_pointcloud


def batch_normalize_to_unit_sphere_torch(
    pointcloud: torch.Tensor,
) -> torch.Tensor:
    """Normalizes a batch of point clouds to fit within a unit sphere."""
    # Center the point cloud
    centroid = torch.mean(pointcloud, dim=1, keepdim=True)  # (B, 1, 3)
    pointcloud_centered = pointcloud - centroid  # (B, N, 3)

    # Scale to fit within unit sphere
    max_dist = torch.max(
        torch.sqrt(torch.sum(pointcloud_centered**2, dim=2)), dim=1, keepdim=True
    ).values  # (B, 1)
    scale_factor = torch.where(
        max_dist > 1e-6, 1.0 / max_dist, torch.ones_like(max_dist)
    )  # (B, 1)

    normalized_pointcloud = pointcloud_centered * scale_factor.unsqueeze(2)  # (B, N, 3)

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
    """Applies a random 3D rotation to a point cloud."""
    # Generate a random rotation axis (a unit vector)
    axis = np.random.rand(3) - 0.5
    axis /= np.linalg.norm(axis)
    # Generate a random rotation angle
    theta = np.pi * 2 * np.random.rand()
    # Rodrigues' rotation formula
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    I = np.identity(3)
    rotation_matrix = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return pointcloud @ rotation_matrix.T


def batch_rotate_torch(pc: torch.Tensor) -> torch.Tensor:
    """Applies a random rotation to a batch of point clouds around the Z-axis."""
    # Generate random angles for each item in the batch
    batch_size = pc.shape[0]
    angles = torch.rand(batch_size) * 2 * np.pi  # (B,)
    cosval, sinval = torch.cos(angles), torch.sin(angles)  # (B,)
    # Create rotation matrices for the batch
    zeros = torch.zeros_like(cosval)  # (B,)
    ones = torch.ones_like(cosval)  # (B,)

    # Stack to create rotation matrices: shape (B, 3, 3)
    rotation_matrices = (
        torch.stack(
            [cosval, sinval, zeros, -sinval, cosval, zeros, zeros, zeros, ones],
            dim=1,
        )
        .reshape(batch_size, 3, 3)
        .to(pc.device, pc.dtype)
    )

    # Apply the rotation via batch matrix multiplication
    rotated_pc = torch.bmm(pc, rotation_matrices)
    return rotated_pc
