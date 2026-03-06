import os
import torch
import numpy as np
import plotly.graph_objects as go


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
        np.savetxt(f, points, fmt="%.10f")


def load_ply(file_path: str) -> np.ndarray:
    """Loads a .ply file into a numpy array."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Please install open3d to load .ply files: pip install open3d"
        )
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points).astype(np.float32)


def plot_ply(pc, title="Point Cloud", size=2):
    """
    Interactive 3D point cloud visualization

    Parameters
    ----------
    pc : np.ndarray or torch.Tensor
        Point cloud (N, 3) or (B, N, 3)
    title : str
        Plot title
    size : int
        Marker size
    """
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()

    # Remove batch dimension
    if pc.ndim == 3:
        pc = pc[0]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pc[:, 0],
                y=pc[:, 1],
                z=pc[:, 2],
                mode="markers",
                marker=dict(
                    size=size,
                    color=pc[:, 2],
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=True,
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        width=900,
        height=900,
    )

    fig.show()
