import torch


def knn_vanilla(
    points: torch.Tensor,
    query: torch.Tensor = None,
    k: int = 16,
) -> torch.Tensor:
    """
    Finds the k-Nearest Neighbors for each point.

    Parameters
    ----------
    points : torch.Tensor
        The source points, shape (B, C, N).
    query : torch.Tensor, optional
        The query points, shape (B, C, M). If None, use points as the query.
    k : int
        The number of nearest neighbors to find.

    Returns
    -------
    torch.Tensor
        The indices of the k-nearest neighbors for each query point, shape (B, M, k).

    """
    if query is None:
        query = points
    inner = -2 * (query.transpose(2, 1) @ points)  # (B, M, N)
    query_norm = torch.sum(query**2, dim=1, keepdim=True)  # (B, 1, M)
    point_norm = torch.sum(points**2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -query_norm.transpose(2, 1) - inner - point_norm  # (B, M, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, M, k)
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
        return knn_vanilla(points, k=k)

    # --- Memory-efficient block processing ---
    all_indices = []
    # Process points in blocks
    for i in range(0, N, block_size):
        start = i
        end = min(i + block_size, N)
        query_block = points[:, :, start:end]  # (B, C, block_size)
        idx_block = knn_vanilla(points, query_block, k)  # (B, block_size, k)
        all_indices.append(idx_block)

    # Concatenate results from all blocks
    idx = torch.cat(all_indices, dim=1)  # (B, N, k)
    return idx
