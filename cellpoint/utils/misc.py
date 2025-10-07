import torch
from pointnet2_ops import pointnet2_utils


def fps(points: torch.Tensor, number: int) -> torch.Tensor:
    """
    Performs Farthest Point Sampling (FPS) on the input point cloud.

    Parameters
    ----------
    points : torch.Tensor
        The input point cloud. Shape: (B, N, 3).
    number : int
        The number of points to sample.

    Returns
    -------
    torch.Tensor
        The sampled points. Shape: (B, number, 3).
    """
    fps_idx = pointnet2_utils.furthest_point_sample(points, number)
    fps_data = (
        pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)
        .contiguous()
    )
    return fps_data


def get_neighbors(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gathers neighbor features based on indices from KNN.

    Parameters
    ----------
    points : torch.Tensor
        The source points, shape (B, C, N).
    idx : torch.Tensor
        The indices of neighbors for each query point, shape (B, M, k).

    Returns
    -------
    torch.Tensor
        The features of the neighbor points, shape (B, M, k, C).
    """
    B, C, _ = points.size()
    _, M, k = idx.size()

    points_t = points.transpose(2, 1).contiguous()  # (B, N, C)
    idx = idx.reshape(B, -1, 1).expand(-1, -1, C)  # (B, M*k, C)
    neighbors = torch.gather(points_t, 1, idx)  # (B, M*k, C)
    neighbors = neighbors.view(B, M, k, C)  # (B, M, k, C)

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
