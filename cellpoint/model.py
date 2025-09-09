import torch
import torch.nn as nn

from utils import knn_block as knn
from utils import local_cov, local_maxpool


class GraphLayer(nn.Module):
    """A single graph layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the graph layer.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (B, C_in, N).
        idx : torch.Tensor
            Neighbor indices from KNN, shape (B, N, k).

        Returns
        -------
        torch.Tensor
            Output features, shape (B, C_out, N).
        """
        pooled_features = local_maxpool(x, idx)  # (B, C_in, N)
        output_features = self.mlp(pooled_features)  # (B, C_out, N)

        return output_features


class FoldingNetEncoder(nn.Module):
    """The Graph-based Encoder for FoldingNet, refactored for clarity."""

    def __init__(self, feat_dims: int = 512, k: int = 16) -> None:
        super().__init__()
        self.k = k

        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )

        self.graph_layer1 = GraphLayer(64, 128)
        self.graph_layer2 = GraphLayer(128, 1024)

        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Encodes a point cloud into a global codeword.

        Parameters
        ----------
        pts : torch.Tensor
            Input point cloud, shape (B, N, 3).

        Returns
        -------
        torch.Tensor
            The global codeword, shape (B, feat_dims, 1).
        """
        pts_t = pts.transpose(2, 1)  # (B, 3, N)
        idx = knn(pts_t, k=self.k)  # (B, N, k)
        features_cov = local_cov(pts_t, idx)  # (B, 12, N)
        features_mlp1 = self.mlp1(features_cov)  # (B, 64, N)

        idx = knn(features_mlp1, k=self.k)  # (B, N, k)
        graph_features1 = self.graph_layer1(features_mlp1, idx)  # (B, 128, N)
        idx = knn(graph_features1, k=self.k)
        graph_features2 = self.graph_layer2(graph_features1, idx)  # (B, 1024, N)

        features_global, _ = torch.max(graph_features2, 2, keepdim=True)  # (B, 1024, 1)
        codeword = self.mlp2(features_global)  # (B, feat_dims, 1)

        return codeword


class FoldingBlock(nn.Module):
    """A single folding operation block."""

    def __init__(self, in_channels: int, feat_dims: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels + feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, 3, 1),
        )

    def forward(
        self, codeword_expanded: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the folding operation.

        Parameters
        ----------
        codeword_expanded : torch.Tensor
            Expanded codeword, shape (B, feat_dims, M).
        points : torch.Tensor
            2D grid or intermediate 3D points, shape (B, C, M).

        Returns
        -------
        torch.Tensor
            Folded points, shape (B, 3, M).
        """
        concat = torch.cat((codeword_expanded, points), dim=1)  # (B, feat_dims+C, M)
        return self.mlp(concat)  # (B, 3, M)


class FoldingNetDecoder(nn.Module):
    """The folding-based decoder for FoldingNet"""

    def __init__(self, feat_dims: int = 512, grid_size: int = 45) -> None:
        super().__init__()
        self.M = grid_size**2
        self.feat_dims = feat_dims

        self.folding1 = FoldingBlock(in_channels=2, feat_dims=self.feat_dims)
        self.folding2 = FoldingBlock(in_channels=3, feat_dims=self.feat_dims)

        self._create_and_register_grid(grid_size)

    def _create_and_register_grid(self, grid_size: int) -> None:
        """Creates the 2D grid points and registers them as a buffer."""
        x_coords = torch.linspace(-0.3, 0.3, grid_size)  # (grid_size,)
        y_coords = torch.linspace(-0.3, 0.3, grid_size)  # (grid_size,)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0)  # (2, M)
        self.register_buffer("grid", grid_points.unsqueeze(0))  # (1, 2, M)

    def forward(self, codeword: torch.Tensor) -> torch.Tensor:
        """
        Decodes a codeword into a 3D point cloud.

        Parameters
        ----------
        codeword : torch.Tensor
            The global feature vector, shape (B, feat_dims, 1).

        Returns
        -------
        torch.Tensor
            The reconstructed point cloud, shape (B, M, 3).
        """
        B = codeword.shape[0]
        codeword_expanded = codeword.expand(-1, -1, self.M)  # (B, feat_dims, M)
        grid_points = self.grid.expand(B, -1, -1).to(codeword.device)  # (B, 2, M)

        intermediate_cloud = self.folding1(codeword_expanded, grid_points)  # (B, 3, M)
        reconstruction = self.folding2(
            codeword_expanded, intermediate_cloud
        )  # (B, 3, M)

        return reconstruction.transpose(2, 1)  # (B, M, 3)


class FoldingNet(nn.Module):
    def __init__(self, feat_dims: int = 512, k: int = 16, grid_size: int = 45) -> None:
        super().__init__()
        self.encoder = FoldingNetEncoder(feat_dims=feat_dims, k=k)
        self.decoder = FoldingNetDecoder(feat_dims=feat_dims, grid_size=grid_size)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Performs a full auto-encoding pass: encode and then decode.

        Parameters
        ----------
        point_cloud : torch.Tensor
            Input point cloud, shape (B, N, 3).

        Returns
        -------
        torch.Tensor
            Reconstructed point cloud, shape (B, M, 3).
        """
        codeword = self.encoder(point_cloud)  # (B, feat_dims, 1)
        reconstruction = self.decoder(codeword)  # (B, M, 3)
        return reconstruction


if __name__ == "__main__":
    model = FoldingNet()
    point_cloud = torch.randn(1, 1024, 3)
    reconstruction = model(point_cloud)
    print(reconstruction.shape)
