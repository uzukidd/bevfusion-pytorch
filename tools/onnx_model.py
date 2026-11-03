"""ONNX-compatible camera geometry and full BEVFusion inference graph.

中文：使用标准 ONNX 算子表达矩阵逆、BEV 聚合与动态稀疏点云分支。
Inputs are already normalized/cropped RGB, raw points and per-frame calibration.
"""
import torch
from torch import nn

from .onnx_lidar import OnnxLidarEncoder, voxelize


def inverse3(matrix):
    """Differentiable 3x3 adjugate inverse, avoiding unsupported linalg_inv."""
    a, b, c = matrix.unbind(-2)
    def cross(x, y):
        return torch.stack((x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1],
                            x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2],
                            x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]), -1)
    first, second, third = cross(b, c), cross(c, a), cross(a, b)
    determinant = (a * first).sum(-1)
    return torch.stack((first, second, third), -1) / determinant[..., None, None]


class OnnxCamera(nn.Module):
    def __init__(self, camera):
        super().__init__()
        self.backbone = camera["backbone"]
        self.neck = camera["neck"]
        self.vtransform = camera["vtransform"]

    def forward(self, images, camera_intrinsics, camera2lidar, img_aug_matrix):
        # Six cameras are one static batch for Swin; point count stays dynamic.
        feature = self.neck(self.backbone(images.reshape(6, 3, 256, 704)))[0]
        feature = feature.reshape(1, 6, 256, 32, 88)
        vt = self.vtransform
        frustum = vt.frustum.float()[None, None]
        post_rot, post_trans = img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3]
        geometry = frustum - post_trans.reshape(1, 6, 1, 1, 1, 3)
        geometry = inverse3(post_rot).reshape(1, 6, 1, 1, 1, 3, 3) @ geometry.unsqueeze(-1)
        geometry = torch.cat((geometry[..., :2, :] * geometry[..., 2:3, :], geometry[..., 2:3, :]), -2)
        combine = camera2lidar[..., :3, :3] @ inverse3(camera_intrinsics)
        geometry = (combine.reshape(1, 6, 1, 1, 1, 3, 3) @ geometry).squeeze(-1)
        geometry = geometry + camera2lidar[..., :3, 3].reshape(1, 6, 1, 1, 1, 3)
        coords = ((geometry - (vt.bx - vt.dx / 2)) / vt.dx).long()
        prediction = vt.depthnet(feature.reshape(6, 256, 32, 88)).reshape(1, 6, 198, 32, 88)
        depth, channels = prediction[:, :, :118].softmax(2), prediction[:, :, 118:]
        out = feature.new_zeros((256 * 256, 80))
        bounds = torch.tensor([256, 256, 1], device=images.device)
        for camera_index in range(6):
            for start in range(0, 118, vt.pool_depth_chunk):
                end = min(118, start + vt.pool_depth_chunk)
                lifted = (depth[0, camera_index, start:end, :, :, None]
                          * channels[0, camera_index].permute(1, 2, 0)[None])
                co = coords[0, camera_index, start:end].reshape(-1, 3)
                valid = ((co >= 0) & (co < bounds)).all(-1)
                co = co[valid]
                ranks = co[:, 0] * 256 + co[:, 1]
                values = lifted.reshape(-1, 80)[valid]
                out = out.scatter_add(0, ranks[:, None].expand(-1, 80), values)
        out = out.reshape(1, 256, 256, 80).permute(0, 3, 1, 2).contiguous()
        return vt.downsample(out)


class OnnxBEVFusion(nn.Module):
    """Full weights, standard ONNX ops. Only the leading point count is dynamic."""
    def __init__(self, model):
        super().__init__()
        self.camera = OnnxCamera(model.encoders["camera"])
        self.lidar = OnnxLidarEncoder(model.encoders["lidar"]["backbone"])
        self.fuser = model.fuser
        self.decoder = model.decoder
        self.head = model.heads["map"]
        self.quantize_decoder = model.quantize_decoder

    def forward(self, images, points, camera_intrinsics, camera2lidar, img_aug_matrix):
        features, coords = voxelize(points)
        lidar_bev = self.lidar(features, coords)
        camera_bev = self.camera(images, camera_intrinsics, camera2lidar, img_aug_matrix)
        fused = self.fuser(torch.cat((camera_bev, lidar_bev), dim=1))
        decoded = self.decoder["neck"](self.decoder["backbone"](fused))[0]
        if self.quantize_decoder:
            decoded = decoded.half().float()
        return self.head(decoded)
