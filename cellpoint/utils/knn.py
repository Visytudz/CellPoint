import torch


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
