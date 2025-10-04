import os
import numpy as np


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
