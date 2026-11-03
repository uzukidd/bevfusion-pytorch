"""Shared tensor input, calibration and feature extraction for det and seg."""
from __future__ import annotations

import torch
from torch import nn

from .camera import build_decoder, build_fuser
from .preprocessing import prepare_points, preprocess_images, voxelize


def _camera_matrix(value, name, sizes):
    if value is None:
        return None
    value = torch.as_tensor(value, dtype=torch.float32).detach().clone()
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if (value.ndim != 4 or value.shape[:2] != (1, 6)
            or tuple(value.shape[-2:]) not in sizes or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite [6,M,M] or [1,6,M,M], M in {sizes}")
    if torch.any(torch.linalg.det(value[..., :3, :3]).abs() < 1e-8):
        raise ValueError(f"{name} must contain invertible 3x3 matrices")
    if value.shape[-1] == 4:
        expected = value.new_tensor([0, 0, 0, 1]).expand_as(value[..., 3, :])
        if not torch.allclose(value[..., 3, :], expected, atol=1e-6):
            raise ValueError(f"{name} must contain homogeneous matrices with last row [0,0,0,1]")
    return value


class BEVFusionBase(nn.Module):
    """Common six-camera + LiDAR feature extractor; task heads live in subclasses.

    Images are uint8 RGB [6,3,H,W] or ImageNet-normalized [1,6,3,256,704].
    Points are [P,4/5] XYZ/intensity/optional time lag in the reference LiDAR frame.
    Each model owns its parameters even when its module implementations are shared.
    """
    def __init__(self, camera_encoder, lidar_encoder, *, backend="reference",
                 voxel_size=(0.1, 0.1, 0.2),
                 point_cloud_range=(-51.2, -51.2, -5., 51.2, 51.2, 3.),
                 max_voxels=120000, camera_intrinsics=None, camera2lidar=None,
                 img_aug_matrix=None):
        super().__init__()
        if backend not in ("reference", "auto", "cuda"):
            raise ValueError("backend must be reference, cuda or auto")
        self.backend = backend
        self.voxel_size = tuple(voxel_size)
        self.point_cloud_range = tuple(point_cloud_range)
        self.max_voxels = max_voxels
        self.encoders = nn.ModuleDict({
            "camera": camera_encoder,
            "lidar": nn.ModuleDict({"backbone": lidar_encoder}),
        })
        self.fuser = build_fuser()
        self.decoder = build_decoder()
        self.register_buffer("_camera_intrinsics", None, persistent=False)
        self.register_buffer("_camera2lidar", None, persistent=False)
        self.register_buffer("_img_aug_matrix", None, persistent=False)
        self.set_calibration(camera_intrinsics, camera2lidar, img_aug_matrix)
        self.load_info = {}

    def set_calibration(self, camera_intrinsics, camera2lidar, img_aug_matrix=None):
        """Bind original-image intrinsics and camera-to-reference-LiDAR transforms."""
        device = next(self.parameters()).device
        for attribute, value, sizes in (
            ("_camera_intrinsics", camera_intrinsics, ((3, 3), (4, 4))),
            ("_camera2lidar", camera2lidar, ((4, 4),)),
            ("_img_aug_matrix", img_aug_matrix, ((4, 4),)),
        ):
            matrix = _camera_matrix(value, attribute[1:], sizes)
            setattr(self, attribute, matrix.to(device) if matrix is not None else None)
        return self

    def _prepare_inputs(self, images, points, camera_intrinsics, camera2lidar, img_aug_matrix):
        if self.training:
            raise RuntimeError("This wrapper is for inference; call model.eval()")
        device = next(self.parameters()).device
        if not isinstance(images, torch.Tensor):
            raise ValueError("images must be a Tensor")
        images = images.to(device)
        augmentation = self._img_aug_matrix if img_aug_matrix is None else img_aug_matrix
        if images.dtype == torch.uint8:
            if img_aug_matrix is not None:
                raise ValueError("uint8 RGB inputs generate their own augmentation; omit img_aug_matrix")
            images, augmentation = preprocess_images(images)
        else:
            if images.ndim == 4:
                images = images.unsqueeze(0)
            if (not images.is_floating_point() or tuple(images.shape) != (1, 6, 3, 256, 704)
                    or not torch.isfinite(images).all()):
                raise ValueError("Normalized images must be finite float [1,6,3,256,704] or [6,3,256,704]")
            images = images.float()
        if tuple(images.shape) != (1, 6, 3, 256, 704):
            raise ValueError("This checkpoint requires exactly six cameras, batch=1")
        intrinsic = self._camera_intrinsics if camera_intrinsics is None else camera_intrinsics
        extrinsic = self._camera2lidar if camera2lidar is None else camera2lidar
        if intrinsic is None or extrinsic is None:
            raise ValueError("Camera calibration is required: bind camera_intrinsics and camera2lidar, or pass them to forward")
        intrinsic = _camera_matrix(intrinsic, "camera_intrinsics", ((3, 3), (4, 4))).to(device)
        extrinsic = _camera_matrix(extrinsic, "camera2lidar", ((4, 4),)).to(device)
        if augmentation is None:
            raise ValueError("Normalized float images require img_aug_matrix, bound or passed; "
                             "use explicit identity only if K already describes the processed images")
        augmentation = _camera_matrix(augmentation, "img_aug_matrix", ((4, 4),)).to(device)
        if not isinstance(points, torch.Tensor):
            raise ValueError("points must be a Tensor")
        points = prepare_points(points.to(device))
        return images, points, intrinsic, extrinsic, augmentation

    def _camera_features(self, images, points, intrinsic, extrinsic, augmentation):
        return self.encoders["camera"](images, extrinsic, intrinsic, augmentation,
                                       torch.eye(4, device=images.device)[None])

    @torch.inference_mode()
    def extract_features(self, images, points, camera_intrinsics=None, camera2lidar=None,
                         img_aug_matrix=None):
        """Return [1,512,X,Y] features: seg 128x128, det 180x180."""
        images, points, intrinsic, extrinsic, augmentation = self._prepare_inputs(
            images, points, camera_intrinsics, camera2lidar, img_aug_matrix)
        features, coords = voxelize(
            points, max_voxels=self.max_voxels, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range, backend=self.backend)
        lidar_bev = self.encoders["lidar"]["backbone"](features, coords, verbose=False)
        camera_bev = self._camera_features(images, points, intrinsic, extrinsic, augmentation)
        if camera_bev.shape[-2:] != lidar_bev.shape[-2:]:
            raise RuntimeError(f"Camera/LiDAR BEV shapes differ: {camera_bev.shape} / {lidar_bev.shape}")
        fused = self.fuser(torch.cat((camera_bev, lidar_bev), dim=1))
        return self.decoder["neck"](self.decoder["backbone"](fused))[0]

    def _save_pretrained(self, path, format_name, config):
        from pathlib import Path
        payload = {
            "format": format_name,
            "task": self.task,
            "config": config,
            "calibration": {name: getattr(self, "_" + name).detach().cpu()
                            for name in ("camera_intrinsics", "camera2lidar", "img_aug_matrix")
                            if getattr(self, "_" + name) is not None},
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
        }
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def _load_pretrained(cls, checkpoint, format_name, device, options, *, payload=None):
        if payload is None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        packed = isinstance(payload, dict) and payload.get("format") == format_name
        if isinstance(payload, dict) and payload.get("format") is not None and not packed:
            raise ValueError(f"Checkpoint format {payload['format']!r} does not match {cls.__name__}")
        if packed:
            options = {**payload["config"], **payload.get("calibration", {}), **options}
        state = payload.get("state_dict", payload)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        foreign_head = "heads.object." if cls.task == "seg" else "heads.map."
        if any(key.startswith(foreign_head) for key in state):
            raise ValueError(f"Checkpoint task differs from {cls.task}; use the matching det/seg model")
        model = cls(**options)
        model.load_state_dict(state, strict=True)
        model.load_info = {"base_tensors": len(state), "packed": packed}
        return model.to(device=device, dtype=torch.float32).eval()
