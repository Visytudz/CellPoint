import torch
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points):
        for t in self.transforms:
            points = t(points)
        return points


class PointcloudRotate:
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        # Generate random angles for each item in the batch
        batch_size = points.shape[0]
        angles = torch.rand(batch_size) * 2 * np.pi  # (B,)
        cosval, sinval = torch.cos(angles), torch.sin(angles)  # (B,)
        # Create rotation matrices for the batch
        zeros = torch.zeros_like(cosval)  # (B,)
        ones = torch.ones_like(cosval)  # (B,)

        # Stack to create rotation matrices: shape (B, 3, 3)
        rotation_matrices = (
            torch.stack(
                [cosval, sinval, zeros, -sinval, cosval, zeros, zeros, zeros, ones],
                dim=1,
            )
            .reshape(batch_size, 3, 3)
            .to(points.device, points.dtype)
        )

        # Apply the rotation via batch matrix multiplication
        rotated_pc = torch.bmm(points, rotation_matrices)
        return rotated_pc


class PointcloudScaleAndTranslate:
    def __init__(self, scale_low=0.8, scale_high=1.2, translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, points):
        B = points.shape[0]
        scale = (
            torch.rand(B, 1, 3, device=points.device, dtype=points.dtype)
            * (self.scale_high - self.scale_low)
            + self.scale_low
        )
        translation = (
            (torch.rand(B, 1, 3, device=points.device, dtype=points.dtype) - 0.5)
            * 2
            * self.translate_range
        )

        points = points * scale + translation
        return points


class PointcloudJitter:
    def __init__(self, clip=0.05, sigma=0.01):
        self.clip = clip
        self.sigma = sigma

    def __call__(self, points):
        B, N, C = points.shape
        jitter = (
            torch.randn(B, N, C, device=points.device, dtype=points.dtype) * self.sigma
        )
        jitter = torch.clamp(jitter, -self.clip, self.clip)
        points = points + jitter
        return points
