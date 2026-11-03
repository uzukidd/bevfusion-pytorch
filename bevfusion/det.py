"""Standalone six-camera + LiDAR 3D object detection model."""
from __future__ import annotations

import torch
from torch import nn

from .base import BEVFusionBase
from .camera import NativeCameraEncoder
from .configuration import load_detection_config
from .depth import DepthLSSTransform
from .detection_head import OBJECT_CLASSES, TransFusionHead
from .lidar import ReferenceSparseEncoder

FORMAT = "bevfusion-det-v1"


class BEVFusionDet(BEVFusionBase):
    """NuScenes TransFusion detector, tensor inputs and bottom-centred LiDAR boxes.

    Raw RGB inputs and calibration follow BEVFusionSeg's interface. Point inputs
    may contain current-frame points or sweeps already transformed to the current
    LiDAR frame; the fifth feature is time lag in seconds, not the LiDAR ring ID.
    ``auto`` uses installed CUDA operators on CUDA and the reference path on CPU.
    """
    task = "det"

    def __init__(self, *, camera_intrinsics=None, camera2lidar=None,
                 img_aug_matrix=None, camera_batch_size=1, pool_depth_chunk=16,
                 config=None, backend="auto"):
        if camera_batch_size < 0 or pool_depth_chunk < 1:
            raise ValueError("camera_batch_size must be >=0 and pool_depth_chunk must be >0")
        resolved = load_detection_config(config)
        lidar_cfg = resolved["model"]["encoders"]["lidar"]
        super().__init__(
            NativeCameraEncoder(camera_batch_size, pool_depth_chunk,
                                vtransform=DepthLSSTransform(pool_depth_chunk=pool_depth_chunk,
                                                            pool_backend=backend)),
            ReferenceSparseEncoder(lidar_cfg["backbone"]["sparse_shape"], backend=backend),
            backend=backend, voxel_size=resolved["voxel_size"],
            point_cloud_range=resolved["point_cloud_range"],
            max_voxels=lidar_cfg["voxelize"]["max_voxels"][1],
            camera_intrinsics=camera_intrinsics, camera2lidar=camera2lidar,
            img_aug_matrix=img_aug_matrix)
        self.config = resolved
        self.classes = OBJECT_CLASSES
        head_cfg = resolved["model"]["heads"]["object"]
        factor = head_cfg["test_cfg"]["out_size_factor"]
        grid = head_cfg["test_cfg"]["grid_size"]
        head = TransFusionHead(
            in_channels=head_cfg["in_channels"], hidden_channel=head_cfg["hidden_channel"],
            num_classes=head_cfg["num_classes"], num_proposals=head_cfg["num_proposals"],
            grid_size=(grid[0] // factor, grid[1] // factor),
            voxel_size=self.voxel_size[:2], pc_range=self.point_cloud_range[:2],
            out_size_factor=factor, post_center_range=head_cfg["bbox_coder"]["post_center_range"],
            decoded_z_origin=head_cfg["bbox_coder"]["decoded_z_origin"])
        self.heads = nn.ModuleDict({"object": head})
        self.eval()

    def _camera_features(self, images, points, intrinsic, extrinsic, augmentation):
        # Match PointsRangeFilter before the depth projection. Retain all selected
        # points, rather than only the first ten points kept in each hard voxel.
        lower = points.new_tensor(self.point_cloud_range[:3])
        upper = points.new_tensor(self.point_cloud_range[3:])
        selected = points[((points[:, :3] >= lower) & (points[:, :3] < upper)).all(1)]
        camera = self.encoders["camera"]
        return camera["vtransform"](
            camera.encode_images(images), selected, extrinsic, intrinsic, augmentation,
            torch.eye(4, device=images.device)[None])

    @classmethod
    def from_pretrained(cls, checkpoint, *, device="cpu", **kwargs):
        return cls._load_pretrained(checkpoint, FORMAT, device, kwargs)

    def save_pretrained(self, path):
        self._save_pretrained(path, FORMAT, {
            "config": self.config,
            "camera_batch_size": self.encoders["camera"].camera_batch_size,
            "pool_depth_chunk": self.encoders["camera"]["vtransform"].pool_depth_chunk,
            "backend": self.backend,
        })

    @torch.inference_mode()
    def forward(self, images, points, camera_intrinsics=None, camera2lidar=None,
                img_aug_matrix=None, *, score_threshold=None, return_raw=False):
        """Return boxes_3d[N,9], scores[N], labels[N], or the raw head dict.

        Boxes: x,y,z_bottom,size_x,size_y,size_z,yaw,vx,vy in reference LiDAR.
        No box NMS is applied, matching the original checkpoint configuration.
        """
        threshold = (self.config["model"]["heads"]["object"]["bbox_coder"]["score_threshold"]
                     if score_threshold is None else score_threshold)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("score_threshold must be in [0,1]")
        features = self.extract_features(images, points, camera_intrinsics, camera2lidar, img_aug_matrix)
        predictions = self.heads["object"](features)
        return predictions if return_raw else self.heads["object"].decode(predictions, threshold)


BEVFusionDetector = BEVFusionDet
