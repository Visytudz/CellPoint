import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import knn_block as knn
from utils import local_cov, local_maxpool, get_neighbors


class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EdgeConv block.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (B, C_in, N).

        Returns
        -------
        torch.Tensor
            Output features after EdgeConv and aggregation, shape (B, C_out, N).
        """
        B, C, N = x.shape

        # get edge features
        idx = knn(x, k=self.k)  # (B, N, k)
        neighbors = get_neighbors(x, idx)  # (B, N, k, C)
        central_points = (
            x.transpose(1, 2).view(B, N, 1, C).expand(-1, -1, self.k, -1)
        )  # (B, N, k, C)
        edge_features = torch.cat(
            [central_points, neighbors - central_points], dim=3
        )  # (B, N, k, 2*C)

        # apply MLP and aggregate
        edge_features = edge_features.permute(0, 3, 1, 2).contiguous()  # (B, 2*C, N, k)
        conv_out = self.mlp(edge_features)  # (B, C_out, N, k)
        aggregated_features, _ = torch.max(conv_out, dim=3)  # (B, C_out, N)

        return aggregated_features


class DGCNNEncoder(nn.Module):
    """The DGCNN-based Encoder, adapted for reconstruction tasks."""

    def __init__(self, feat_dims: int = 512, k: int = 20) -> None:
        super().__init__()
        self.k = k
        self.feat_dims = feat_dims

        self.conv1 = EdgeConv(in_channels=3, out_channels=64, k=self.k)
        self.conv2 = EdgeConv(in_channels=64, out_channels=64, k=self.k)
        self.conv3 = EdgeConv(in_channels=64, out_channels=128, k=self.k)
        self.conv4 = EdgeConv(in_channels=128, out_channels=256, k=self.k)

        # Final MLP to process concatenated features
        self.mlp_global = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, self.feat_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.feat_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Encodes a point cloud into a global codeword using DGCNN.

        Parameters
        ----------
        pts : torch.Tensor
            Input point cloud, shape (B, N, 3).

        Returns
        -------
        torch.Tensor
            The global codeword, shape (B, feat_dims, 1).
        """
        pts_t = pts.transpose(2, 1).contiguous()  # (B, 3, N)

        # Pass through EdgeConv blocks
        x1 = self.conv1(pts_t)  # (B, 64, N)
        x2 = self.conv2(x1)  # (B, 64, N)
        x3 = self.conv3(x2)  # (B, 128, N)
        x4 = self.conv4(x3)  # (B, 256, N)

        # Concatenate features from all layers (skip-connections)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        # Process with final MLP
        features_per_point = self.mlp_global(x_cat)  # (B, feat_dims, N)

        # Global max pooling to get the final codeword
        codeword, _ = torch.max(
            features_per_point, 2, keepdim=True
        )  # (B, feat_dims, 1)

        return codeword


class GraphLayer(nn.Module):
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

    def __init__(
        self, feat_dims: int = 512, grid_size: int = 45, grid_type: str = "plane"
    ) -> None:
        super().__init__()
        self.M = grid_size**2
        self.feat_dims = feat_dims
        self.grid_type = grid_type

        if self.grid_type == "plane":
            first_folding_in_channels = 2
        elif self.grid_type == "sphere":
            first_folding_in_channels = 3
        else:
            raise ValueError(
                f"Unknown grid_type: {grid_type}. Choose 'plane' or 'sphere'."
            )

        self.folding1 = FoldingBlock(
            in_channels=first_folding_in_channels, feat_dims=self.feat_dims
        )
        self.folding2 = FoldingBlock(in_channels=3, feat_dims=self.feat_dims)

        self._create_and_register_grid(grid_size)

    def _create_and_register_grid(self, grid_size: int) -> None:
        """Creates the 2D or 3D grid points and registers them as a buffer."""
        if self.grid_type == "plane":
            x_coords = torch.linspace(-0.3, 0.3, grid_size)  # (grid_size,)
            y_coords = torch.linspace(-0.3, 0.3, grid_size)  # (grid_size,)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
            grid_points = torch.stack(
                [grid_x.flatten(), grid_y.flatten()], dim=0
            )  # (2, M)
            self.register_buffer("grid", grid_points.unsqueeze(0))  # (1, 2, M)
        elif self.grid_type == "sphere":
            points = torch.randn(self.M, 3)  # (M, 3)
            points = F.normalize(points, p=2, dim=1)  # (M, 3)
            grid_points = points.transpose(0, 1)  # (3, M)
            self.register_buffer("grid", grid_points.unsqueeze(0))  # (1, 3, M)

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


class Reconstructer(nn.Module):
    def __init__(
        self,
        feat_dims: int = 512,
        k: int = 16,
        grid_size: int = 45,
        grid_type: str = "plane",
        encoder_type: str = "foldingnet",
    ) -> None:
        """
        Initializes the Autoencoder model.

        Parameters
        ----------
        feat_dims : int, optional
            The dimensionality of the global feature vector (codeword).
        k : int, optional
            The number of nearest neighbors, used by both encoders.
        grid_size : int, optional
            The grid size for the FoldingNet decoder.
        grid_type : str, optional
            The type of initial grid to use: 'plane' or 'sphere'.
        encoder_type : str, optional
            The type of encoder to use. Can be 'foldingnet' or 'dgcnn'.
        """
        super().__init__()

        if encoder_type.lower() == "foldingnet":
            self.encoder = FoldingNetEncoder(feat_dims=feat_dims, k=k)
        elif encoder_type.lower() == "dgcnn":
            self.encoder = DGCNNEncoder(feat_dims=feat_dims, k=k)
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. Choose 'foldingnet' or 'dgcnn'."
            )

        self.decoder = FoldingNetDecoder(
            feat_dims=feat_dims, grid_size=grid_size, grid_type=grid_type
        )

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
    model = Reconstructer(encoder_type="foldingnet", grid_type="sphere")
    point_cloud = torch.randn(1, 1024, 3)
    reconstruction = model(point_cloud)
    print(reconstruction.shape)
