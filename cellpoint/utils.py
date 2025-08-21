import torch


def knn(points: torch.Tensor, k: int) -> torch.Tensor:
    """Finds the k-Nearest Neighbors for each point. Points shape (B, C, N)."""
    inner = -2 * (points.transpose(2, 1) @ points)  # (B, N, N)
    norm = torch.sum(points**2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -norm - inner - norm.transpose(2, 1)  # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
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
