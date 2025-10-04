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
