import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def pca_align(point_cloud):
    """Aligns a 3D point cloud using PCA."""
    # 1. center the point cloud
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid

    # 2. PCA fit
    pca = PCA(n_components=3)
    pca.fit(point_cloud)

    # 3. rotate point cloud, PCA1 -> X, PCA2 -> Y, PCA3 -> Z
    rotation_matrix = pca.components_.T
    aligned_point_cloud = np.dot(point_cloud, rotation_matrix)

    # 4. Sign correction
    for i in range(3):
        # Make sure the max along each axis is positive
        if np.max(aligned_point_cloud[:, i]) < np.abs(
            np.min(aligned_point_cloud[:, i])
        ):
            aligned_point_cloud[:, i] *= -1

    return aligned_point_cloud


def normalize_to_unit_sphere(
    pointcloud: NDArray[np.float32], scale=None
) -> NDArray[np.float32]:
    """Normalizes a point cloud to fit within a unit sphere."""
    # Center the point cloud
    centroid = np.mean(pointcloud, axis=0)  # (3,)
    pointcloud_centered = pointcloud - centroid  # (N, 3)

    # Scale to fit within unit sphere
    if scale is not None:
        scale_factor = scale
    else:
        max_dist = np.max(np.sqrt(np.sum(pointcloud_centered**2, axis=1)))  # (N,)
        scale_factor = 1.0 / max_dist if max_dist > 1e-6 else 1.0

    normalized_pointcloud = pointcloud_centered * scale_factor  # (N, 3)

    return normalized_pointcloud


def batch_normalize_to_unit_sphere_torch(
    pointcloud: torch.Tensor, scale: float = None
) -> torch.Tensor:
    """Normalizes a batch of point clouds to fit within a unit sphere."""
    # Center the point cloud
    centroid = torch.mean(pointcloud, dim=1, keepdim=True)  # (B, 1, 3)
    pointcloud_centered = pointcloud - centroid  # (B, N, 3)

    # Scale to fit within unit sphere
    if scale is not None:
        scale_factor = torch.full(
            (pointcloud.shape[0], 1),
            scale,
            device=pointcloud.device,
            dtype=pointcloud.dtype,
        )  # (B, 1)
    else:
        max_dist = torch.max(
            torch.sqrt(torch.sum(pointcloud_centered**2, dim=2)), dim=1, keepdim=True
        ).values  # (B, 1)
        scale_factor = torch.where(
            max_dist > 1e-6, 1.0 / max_dist, torch.ones_like(max_dist)
        )  # (B, 1)

    normalized_pointcloud = pointcloud_centered * scale_factor.unsqueeze(2)  # (B, N, 3)

    return normalized_pointcloud


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


def decompose_confusion_matrix(cm) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct y_true and y_pred sequences from a confusion matrix.\n
    **Note: these returned sequences may not match the original order of samples.**
    """
    y_true, y_pred = [], []

    # Iterate over all cells in the confusion matrix
    for i in range(cm.shape[0]):  # true label index
        for j in range(cm.shape[1]):  # predicted label index
            count = int(cm[i, j])
            y_true += [i] * count
            y_pred += [j] * count

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(
    cm_numpy: np.ndarray,
    class_names: list[str],
    save_path: str,
    title=str,
) -> None:
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_numpy,
        annot=True,  # show data values in each cell
        fmt="d",  # integer format
        cmap="Blues",  # color theme
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,  # draw on the axes we created
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
