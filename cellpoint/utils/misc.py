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


def get_pos_embed(embed_dim: int, pos_vector: torch.Tensor) -> torch.Tensor:
    """
    Generates a high-dimensional positional embedding from a position vector.

    Parameters
    ----------
    embed_dim : int
        The target dimensionality of the output embedding. It must be divisible by
        (2 * C), where C is the number of channels in pos_vector.
    pos_vector : torch.Tensor
        The input position vectors. Shape: (B, G, C).

    Returns
    -------
    torch.Tensor
        The resulting high-dimensional positional embedding. Shape: (B, G, embed_dim).
    """
    B, G, C = pos_vector.shape

    # embded_dim = 2 * C * num_freqs
    if embed_dim % (2 * C) != 0:
        raise ValueError(
            f"embed_dim ({embed_dim}) must be divisible by 2 * input_channels ({2 * C})."
        )

    # 1. Create the frequency basis (omega)
    num_freqs = embed_dim // (2 * C)
    # Create a tensor of frequencies from 0 to num_freqs-1
    freqs = torch.arange(num_freqs, dtype=torch.float32, device=pos_vector.device)
    # Scale the frequencies logarithmically
    omega = 1.0 / (10000.0 ** (freqs / num_freqs))

    # 2. Process all input dimensions at once using broadcasting
    # pos_vector shape: (B, G, C) -> unsqueeze to (B, G, C, 1)
    # omega shape: (num_freqs,) -> view as (1, 1, 1, num_freqs)
    # Resulting 'out' shape: (B, G, C, num_freqs)
    out = pos_vector.unsqueeze(-1) * omega.view(1, 1, 1, -1)

    # 3. Apply sin and cos functions
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    # 4. Concatenate sin and cos embeddings
    # Shape of emb_sin and emb_cos is (B, G, C, num_freqs)
    # Concatenate them along the last dimension to get (B, G, C, num_freqs * 2)
    embedding = torch.cat([emb_sin, emb_cos], dim=-1)

    # 5. Reshape to the final embedding dimension
    # Reshape (B, G, C, num_freqs * 2) -> (B, G, C * num_freqs * 2)
    # which is (B, G, embed_dim)
    final_embedding = embedding.view(B, G, -1)

    return final_embedding


def get_pqae_loss(outputs: dict, loss_fn: callable) -> torch.Tensor:
    """
    Computes the combined loss for a PointPQAE model.

    Parameters
    ----------
    outputs : dict
        The model outputs containing reconstructions and targets.
    loss_fn : callable
        The loss function to use for computing the loss.

    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    # Unpack outputs
    recon_v1 = outputs["recon1"]  # Shape: [B, G, K, C]
    target_v1 = outputs["group1"]  # Shape: [B, G, K, C]
    recon_v2 = outputs["recon2"]  # Shape: [B, G, K, C]
    target_v2 = outputs["group2"]  # Shape: [B, G, K, C]

    B, G, K, C = recon_v1.shape

    # Reshape for per-patch loss calculation by merging Batch and Group dimensions
    recon_v1_flat = recon_v1.reshape(B * G, K, C)
    target_v1_flat = target_v1.reshape(B * G, K, C)
    recon_v2_flat = recon_v2.reshape(B * G, K, C)
    target_v2_flat = target_v2.reshape(B * G, K, C)

    # Calculate loss for each view
    loss1 = loss_fn(recon_v1_flat, target_v1_flat)
    loss2 = loss_fn(recon_v2_flat, target_v2_flat)

    return loss1 + loss2
