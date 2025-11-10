import cv2
import numpy as np
import open3d as o3d


class PointCloudProjector:
    """A class for projecting 3D point clouds into 2D images."""

    def __init__(
        self,
        render_width=224,
        render_height=224,
        point_size=4,
        output_glyph_size=(64, 64),
    ):
        self.render_width = render_width
        self.render_height = render_height
        self.point_size = point_size
        self.output_glyph_size = output_glyph_size
        self.render, self.material = self._setup_renderer_and_camera()

    @staticmethod
    def normalize_point_cloud(point_cloud):
        centroid = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - centroid

        furthest_distance = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
        if furthest_distance > 1e-6:
            point_cloud = point_cloud / furthest_distance
        return point_cloud

    def _setup_renderer_and_camera(self):
        # set material (black points)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = self.point_size
        mat.base_color = [0.0, 0.0, 0.0, 1.0]  # RGBA

        # set background and camera
        render = o3d.visualization.rendering.OffscreenRenderer(
            self.render_width, self.render_height
        )
        render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # White BG
        render.scene.camera.set_projection(
            o3d.visualization.rendering.Camera.Projection.Ortho,
            -1.1,
            1.1,
            -1.1,
            1.1,
            0.1,
            10,
        )
        center = [0, 0, 0]
        eye = [0, 0, 2]
        up = [0, 1, 0]
        render.scene.camera.look_at(center, eye, up)

        return render, mat

    def _post_process_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Crop the white borders and resize to a uniform glyph size.

        Parameters
        ----------
        image_rgb: np.ndarray
            The original rendered image of shape (H, W, 3).

        Returns
        -------
        np.ndarray
            The cropped and resized RGB image of shape (glyph_size, glyph_size, 3).
        """
        # 1. Convert to grayscale to find non-white pixels
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 2. Find coords of non-white pixels
        coords = cv2.findNonZero(255 - image_gray)
        if coords is None:
            # Image is all white
            return np.ones((*self.output_glyph_size, 3), dtype=np.uint8) * 255

        # 3. Crop the non-white region from the original RGB image
        x, y, w, h = cv2.boundingRect(coords)
        cropped_rgb = image_rgb[y : y + h, x : x + w, :]

        # 4. Place cropped image into a white square (maintain aspect ratio)
        max_dim = max(w, h)
        square_bg = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255

        paste_x = (max_dim - w) // 2
        paste_y = (max_dim - h) // 2
        square_bg[paste_y : paste_y + h, paste_x : paste_x + w, :] = cropped_rgb

        # 5. Scale to target size
        glyph = cv2.resize(
            square_bg, self.output_glyph_size, interpolation=cv2.INTER_AREA
        )

        return glyph

    def project(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Project a 3D point cloud into a 2D image.

        Parameters
        ----------
        point_cloud: np.ndarray
            Input point cloud of shape (N, 3).

        Returns
        -------
        np.ndarray
            The processed main view RGBA image of shape (glyph_size).
        """
        normed_pc = self.normalize_point_cloud(point_cloud)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(normed_pc)

        self.render.scene.clear_geometry()
        self.render.scene.add_geometry("pcd", pcd, self.material)

        img_np = np.asarray(self.render.render_to_image())
        glyph_image = self._post_process_image(img_np)

        return glyph_image


if __name__ == "__main__":
    from cellpoint.datasets import HDF5Dataset

    dataset = HDF5Dataset(
        root="datasets",
        name="intestine",
        splits=["train"],
        num_points=2048,
        normalize=True,
    )
    pcl = dataset[0]["points"].cpu().numpy()  # (N, 3)
    projector = PointCloudProjector()
    glyph = projector.project(pcl)
    cv2.imwrite("projected_glyph.png", glyph)
