import os
import torch
import numpy as np


def knn_vanilla(points: torch.Tensor, k: int) -> torch.Tensor:
    """Finds the k-Nearest Neighbors for each point. Points shape (B, C, N)."""
    inner = -2 * (points.transpose(2, 1) @ points)  # (B, N, N)
    norm = torch.sum(points**2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -norm.transpose(2, 1) - inner - norm  # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def knn_torch3d(points: torch.Tensor, k: int) -> torch.Tensor:
    """Finds the k-Nearest Neighbors for each point using PyTorch3D. Points shape (B, C, N)."""
    try:
        from pytorch3d.ops import knn_points
    except ImportError:
        raise ImportError("Please install PyTorch3D to use this function.")

    # Note: PyTorch3D expects (B, N, C) input
    _, idx, _ = knn_points(
        points.transpose(2, 1), points.transpose(2, 1), K=k, return_nn=False
    )
    return idx  # (B, N, k)


def knn_block(points: torch.Tensor, k: int, block_size: int = 512) -> torch.Tensor:
    """
    Finds the k-Nearest Neighbors for each point in a memory-efficient manner.

    This implementation computes the pairwise distance matrix in blocks to avoid
    O(N^2) memory consumption.

    Parameters
    ----------
    points : torch.Tensor
        The source points, shape (B, C, N).
    k : int
        The number of nearest neighbors to find.
    block_size : int, optional
        The number of points to process in each block. A smaller block size
        reduces memory usage but may slightly decrease performance.

    Returns
    -------
    torch.Tensor
        The indices of the k-nearest neighbors for each point, shape (B, N, k).
    """
    B, C, N = points.shape

    # If the number of points is small, the original method is fine and faster.
    if N <= block_size:
        inner = -2 * (points.transpose(2, 1) @ points)  # (B, N, N)
        norm = torch.sum(points**2, dim=1, keepdim=True)  # (B, 1, N)
        pairwise_distance = -norm.transpose(2, 1) - inner - norm  # (B, N, N)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
        return idx

    # --- Memory-efficient block processing ---
    all_indices = []
    norm = torch.sum(points**2, dim=1, keepdim=True)  # (B, 1, N)

    # Process points in blocks
    for i in range(0, N, block_size):
        start = i
        end = min(i + block_size, N)
        query_block = points[:, :, start:end]  # (B, C, block_size)

        inner = -2 * (query_block.transpose(2, 1) @ points)  # (B, block_size, N)
        norm_block = torch.sum(
            query_block**2, dim=1, keepdim=True
        )  # (B, 1, block_size)
        pairwise_distance = (
            -norm_block.transpose(2, 1) - inner - norm
        )  # (B, block_size, N)

        _, idx_block = pairwise_distance.topk(k=k, dim=-1)  # (B, block_size, k)
        all_indices.append(idx_block)

    # Concatenate results from all blocks
    idx = torch.cat(all_indices, dim=1)  # (B, N, k)
    return idx


def get_neighbors(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gathers neighbor features based on indices from KNN.

    Parameters
    ----------
    points : torch.Tensor
        The source points, shape (B, C, N).
    idx : torch.Tensor
        The indices of neighbors for each point, shape (B, N, k).

    Returns
    -------
    torch.Tensor
        The features of the neighbor points, shape (B, N, k, C).
    """
    B, C, N = points.size()
    k = idx.size(-1)

    points_t = points.transpose(2, 1).contiguous()  # (B, N, C)
    idx = idx.reshape(B, -1, 1).expand(-1, -1, C)  # (B, N*k, C)
    neighbors = torch.gather(points_t, 1, idx)  # (B, N*k, C)
    neighbors = neighbors.view(B, N, k, C)  # (B, N, k, C)

    return neighbors


def local_cov(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Computes local covariance features for each point.
    Concatenates original points (C dims) with covariance matrix features (C*C dims).

    Parameters
    ----------
    points : torch.Tensor
        The input points, shape (B, C, N).
    idx : torch.Tensor
        The indices of neighbors for each point, shape (B, N, k).

    Returns
    -------
    torch.Tensor
        The local covariance features, shape (B, C + C*C, N).
    """
    B, C, N = points.size()
    k = idx.size(-1)

    neighbors = get_neighbors(points, idx)  # (B, N, k, C)
    mean = torch.mean(neighbors, dim=2, keepdim=True)  # (B, N, 1, C)
    neighbors_centered = neighbors - mean  # (B, N, k, C)

    cov = (neighbors_centered.transpose(3, 2) @ neighbors_centered) / k  # (B, N, C, C)
    cov_flat = cov.view(B, N, -1).transpose(1, 2)  # (B, C*C, N)

    return torch.cat((points, cov_flat), dim=1)  # (B, C+C*C, N)


def local_maxpool(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Performs local max pooling using neighbor indices.

    Parameters
    ----------
    points : torch.Tensor
        The input points, shape (B, C, N).
    idx : torch.Tensor
        The indices of neighbors for each point, shape (B, N, k).

    Returns
    -------
    torch.Tensor
        Pooled points with shape (B, C, N).
    """
    neighbors = get_neighbors(points, idx)  # (B, N, k, C)
    pooled_points, _ = torch.max(neighbors, dim=2)  # (B, N, C)
    return pooled_points.transpose(2, 1)  # (B, C, N)


def save_ply(points: np.ndarray, filename: str) -> None:
    """Saves a point cloud to a PLY file. Points shape (N, 3)."""
    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_points = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with open(filename, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, points, fmt="%.6f")
