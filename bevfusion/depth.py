"""LiDAR-conditioned LSS view transform compatible with BEVFusion detection."""
from __future__ import annotations

import torch
from torch import nn

from .camera import LSSTransform


__all__ = ["DepthLSSTransform", "project_lidar_depth"]


def project_lidar_depth(
    points, camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix,
    image_size=(256, 704),
):
    """Project augmented LiDAR points to a scalar depth image for each camera.

    ``points`` is an unvoxelized P,5 tensor for one batch item. Camera matrices
    have leading dimensions 1,N; LiDAR augmentation is 1,4,4. The returned float32
    tensor has shape 1,N,1,H,W. Inputs are never modified.

    Match BaseDepthTransform's clamped Z, continuous image bounds check, and
    truncation to integer pixels. In particular, negative camera Z is clamped
    rather than filtered. For repeated pixels, the last input point wins, with
    stable sorting making that overwrite order explicit on both CPU and CUDA.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected points with shape P,5 (at least XYZ columns)")
    if camera2lidar.ndim != 4 or camera2lidar.shape[0] != 1:
        raise ValueError("LiDAR depth projection supports batch size 1; expected 1,N,4,4 camera2lidar")
    height, width = image_size
    n = camera2lidar.shape[1]
    depth = torch.zeros((1, n, 1, height, width), dtype=torch.float32, device=points.device)
    if points.shape[0] == 0:
        return depth

    # Undo augmentation functionally: the original reference subtracts from a
    # slice of points in place, which corrupts subsequent uses of the point cloud.
    lidar_aug = lidar_aug_matrix[0].float()
    xyz = points[:, :3].float() - lidar_aug[:3, 3]
    xyz = torch.linalg.inv(lidar_aug[:3, :3]) @ xyz.T
    lidar2camera = torch.linalg.inv(camera2lidar[0].float())
    lidar2image = camera_intrinsics[0, :, :3, :3].float() @ lidar2camera[:, :3, :]
    projected = lidar2image[:, :3, :3] @ xyz + lidar2image[:, :3, 3, None]
    distance = projected[:, 2, :].clamp(1e-5, 1e5)
    # Keep clamped Z in the third component for the full 3x3 image augmentation,
    # just as the reference does after dividing only its X and Y components.
    normalized = torch.cat((projected[:, :2, :] / distance[:, None, :], distance[:, None, :]), dim=1)
    img_aug = img_aug_matrix[0].float()
    projected = img_aug[:, :3, :3] @ normalized + img_aug[:, :3, 3, None]
    u, v = projected[:, 0, :], projected[:, 1, :]
    on_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    for camera in range(n):
        valid = on_image[camera]
        pixels = v[camera, valid].long() * width + u[camera, valid].long()
        values = distance[camera, valid]
        order = pixels.argsort(stable=True)
        pixels, values = pixels[order], values[order]
        # A unique destination per assignment avoids CUDA's undefined duplicate
        # advanced-index writes while preserving input order, not nearest depth.
        last = torch.ones_like(pixels, dtype=torch.bool)
        last[:-1] = pixels[:-1] != pixels[1:]
        depth[0, camera, 0].view(-1)[pixels[last]] = values[last]
    return depth


class DepthLSSTransform(LSSTransform):
    """Detection's depth-conditioned camera lift, with chunked BEV pooling."""

    def __init__(self, image_size=(256, 704), pool_depth_chunk=16, pool_backend="reference",
                 precompute=True, pool_num_points=10):
        if pool_backend not in ("reference", "cuda", "auto", "compatible", "horizon", "hat", "hat_plugin"):
            raise ValueError(f"Unknown BEV pooling backend: {pool_backend!r}")
        if not isinstance(pool_depth_chunk, int) or pool_depth_chunk <= 0:
            raise ValueError("pool_depth_chunk must be a positive integer")
        super().__init__(image_size=image_size, pool_depth_chunk=pool_depth_chunk,
                         pool_backend=pool_backend, precompute=precompute, pool_num_points=pool_num_points)
        self.pool_backend = pool_backend
        self.dx = nn.Parameter(torch.tensor([0.3, 0.3, 20.0]), requires_grad=False)
        self.bx = nn.Parameter(torch.tensor([-53.85, -53.85, 0.0]), requires_grad=False)
        self.nx = nn.Parameter(torch.tensor([360, 360, 1]), requires_grad=False)
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1), nn.BatchNorm2d(8), nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(256 + 64, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, self.D + self.C, 1),
        )

    def _predict(self, x, depth):
        b, n, c, h, w = x.shape
        depth_features = self.dtransform(depth.reshape(b * n, 1, *self.image_size).float())
        return self.depthnet(torch.cat((depth_features, x.reshape(b * n, c, h, w).float()), dim=1))

    def get_cam_feats(self, x, depth):
        """Reference full-lift API; forward uses chunks to bound peak memory."""
        b, n, _, h, w = x.shape
        prediction = self._predict(x, depth)
        probabilities = prediction[:, :self.D].softmax(1)
        lifted = prediction[:, self.D:, None] * probabilities[:, None]
        return lifted.reshape(b, n, self.C, self.D, h, w).permute(0, 1, 3, 4, 5, 2)

    def forward(self, x, points, camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix):
        """B,N,256,H/8,W/8 and batch-one P,5 points -> B,80,180,180."""
        b, n, c, h, w = x.shape
        if b != 1:
            raise ValueError("DepthLSSTransform with a P,5 point tensor supports batch size 1")
        if c != 256 or (h, w) != self.feature_size:
            raise ValueError(f"Expected depth LSS features B,N,256,{self.feature_size[0]},{self.feature_size[1]}, "
                             f"got {tuple(x.shape)}")
        depth_image = project_lidar_depth(points, camera2lidar, camera_intrinsics,
                                          img_aug_matrix, lidar_aug_matrix, self.image_size)
        prediction = self._predict(x, depth_image).reshape(b, n, self.D + self.C, h, w)
        depth = prediction[:, :, :self.D].softmax(2)
        features = prediction[:, :, self.D:]
        out = self._pool_predictions(depth, features, camera2lidar, camera_intrinsics,
                                     img_aug_matrix, lidar_aug_matrix)
        return self.downsample(out.contiguous())
