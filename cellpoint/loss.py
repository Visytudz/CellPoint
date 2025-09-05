import torch
import torch.nn as nn


class ChamferLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, gts: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """
        The returned loss is a sum over all points and all batches.

        Parameters
        ----------
        gts : torch.Tensor
            The ground truth point clouds, with shape (B, N, D)
        preds : torch.Tensor
            The predicted point clouds, with shape (B, M, D)

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the total Chamfer loss for the batch.
        """
        pairwise_dist_sq = torch.cdist(gts, preds, p=2.0).pow(2)  # (B, N, M)
        min_dist_gts_to_preds, _ = torch.min(pairwise_dist_sq, dim=2)  # (B, N)
        min_dist_preds_to_gts, _ = torch.min(pairwise_dist_sq, dim=1)  # (B, M)
        loss_1 = torch.mean(min_dist_gts_to_preds)  # scalar
        loss_2 = torch.mean(min_dist_preds_to_gts)  # scalar

        return loss_1 + loss_2  # scalar
