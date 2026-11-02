"""Standalone tensor-in/tensor-out BEVFusion segmentation model.

中文：模型本身不读取 nuScenes、导出标签、缓存或本机路径配置。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .camera import (MAP_CLASSES, NativeCameraEncoder, build_decoder, build_fuser,
                     BEVSegmentationHead)
from .lidar import ReferenceSparseEncoder
from .preprocessing import preprocess_images, voxelize

SURFACE_CLASSES = ("sidewalk", "road", "vegetation", "crosswalk")
FORMAT = "minimal-bevfusion-v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


class BEVFusion(nn.Module):
    """Single-frame six-camera + LiDAR segmentation, using reference PyTorch ops.

    Images: uint8 RGB [6,3,H,W] (automatic preprocessing), OR already normalized
    floating [6,3,256,704]; a leading batch dimension of 1 is also accepted.
    Points: float [P,4/5], XYZ metres / intensity / optional time lag in seconds,
    already in the reference LiDAR frame. Missing time lag is set to zero.
    Output: sigmoid probabilities [1,C,200,200], axes [batch,class,X,Y].
    Bind K / camera2lidar once, or override them per forward for moving sensors.
    中文：标定可初始化时绑定；每帧外参变化时应在 forward 中传入对应矩阵。
    """
    def __init__(self, *, camera_intrinsics=None, camera2lidar=None,
                 img_aug_matrix=None, camera_batch_size=1, pool_depth_chunk=16,
                 classes=MAP_CLASSES, quantize_decoder=False):
        super().__init__()
        if camera_batch_size < 0 or pool_depth_chunk < 1:
            raise ValueError("camera_batch_size must be >=0 and pool_depth_chunk must be >0")
        self.classes = tuple(classes)
        if self.classes not in (tuple(MAP_CLASSES), SURFACE_CLASSES):
            raise ValueError("Supported classes are the original six map classes or four surface classes")
        self.quantize_decoder = bool(quantize_decoder)
        self.encoders = nn.ModuleDict({
            "camera": NativeCameraEncoder(camera_batch_size, pool_depth_chunk),
            "lidar": nn.ModuleDict({"backbone": ReferenceSparseEncoder()}),
        })
        self.fuser = build_fuser()
        self.decoder = build_decoder()
        head = BEVSegmentationHead()
        if len(self.classes) != len(MAP_CLASSES):
            head.classifier[-1] = nn.Conv2d(512, len(self.classes), 1)
        head.classes = self.classes
        self.heads = nn.ModuleDict({"map": head})
        # Calibration moves with .to(), but does not change legacy weight keys.
        self.register_buffer("_camera_intrinsics", None, persistent=False)
        self.register_buffer("_camera2lidar", None, persistent=False)
        self.register_buffer("_img_aug_matrix", None, persistent=False)
        self.set_calibration(camera_intrinsics, camera2lidar, img_aug_matrix)
        self.load_info = {}
        self.eval()

    def set_calibration(self, camera_intrinsics, camera2lidar, img_aug_matrix=None):
        """Bind original-image K, camera-to-reference-LiDAR transforms and augmentation.

        Raw uint8 inputs compute augmentation automatically. Normalized floating
        inputs require explicit augmentation (bound or passed); use identity only
        when K already refers to the resized/cropped image coordinate system.
        """
        device = next(self.parameters()).device
        for attribute, value, sizes in (
            ("_camera_intrinsics", camera_intrinsics, ((3, 3), (4, 4))),
            ("_camera2lidar", camera2lidar, ((4, 4),)),
            ("_img_aug_matrix", img_aug_matrix, ((4, 4),)),
        ):
            matrix = _camera_matrix(value, attribute[1:], sizes)
            setattr(self, attribute, matrix.to(device) if matrix is not None else None)
        return self

    @classmethod
    def from_pretrained(cls, checkpoint, *, head_checkpoint=None, device="cpu", **kwargs):
        """Strictly load an original checkpoint or a self-contained save_pretrained file.

        Optional historical four-class head requires its matching original base
        checkpoint; its recorded SHA256 is verified before loading.
        """
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        packed = isinstance(payload, dict) and payload.get("format") == FORMAT
        if packed:
            if head_checkpoint is not None:
                raise ValueError("A packed model already contains its head; omit head_checkpoint")
            options = {**payload["config"], **payload.get("calibration", {}), **kwargs}
        else:
            options = kwargs
        model = cls(**options)
        state = payload.get("state_dict", payload)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.load_info = {"base_tensors": len(state), "packed": packed}
        if head_checkpoint is not None:
            adapted = torch.load(head_checkpoint, map_location="cpu", weights_only=True)
            if (adapted.get("format") != "bevfusion-surface-head-v1"
                    or tuple(adapted.get("classes", ())) != SURFACE_CLASSES):
                raise ValueError("Unsupported four-class surface head checkpoint")
            grid = adapted.get("model_grid", {})
            if (grid.get("shape") != [4, 200, 200] or grid.get("resolution_m") != 0.5
                    or grid.get("frame") != "reference LiDAR"
                    or grid.get("axis_order") != ["class", "increasing_x", "increasing_y"]
                    or grid.get("bounds_m") != {"x": [-50, 50], "y": [-50, 50]}):
                raise ValueError("Surface head uses an incompatible native grid")
            if _sha256(checkpoint) != adapted.get("frozen_checkpoint_sha256"):
                raise ValueError("Base checkpoint SHA256 differs from the surface head's frozen backbone")
            model.heads["map"].classifier[-1] = nn.Conv2d(512, 4, 1)
            model.heads["map"].load_state_dict(adapted["head_state_dict"], strict=True)
            model.classes = model.heads["map"].classes = SURFACE_CLASSES
            model.quantize_decoder = True
            model.load_info["head_tensors"] = len(adapted["head_state_dict"])
        return model.to(device=device, dtype=torch.float32).eval()

    def save_pretrained(self, path):
        """Save all branches, head and bound calibration together; refuse overwrites."""
        payload = {
            "format": FORMAT,
            "config": {"classes": self.classes, "quantize_decoder": self.quantize_decoder,
                       "camera_batch_size": self.encoders["camera"].camera_batch_size,
                       "pool_depth_chunk": self.encoders["camera"]["vtransform"].pool_depth_chunk},
            "calibration": {name: getattr(self, "_" + name).detach().cpu()
                            for name in ("camera_intrinsics", "camera2lidar", "img_aug_matrix")
                            if getattr(self, "_" + name) is not None},
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
        }
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @torch.inference_mode()
    def extract_features(self, images, points, camera_intrinsics=None, camera2lidar=None,
                         img_aug_matrix=None):
        """Return decoder features [1,512,128,128] without reading any files."""
        if self.training:
            raise RuntimeError("This reference wrapper is for inference; call model.eval()")
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
        features, coords = voxelize(points.to(device))
        lidar_bev = self.encoders["lidar"]["backbone"](features, coords, verbose=False)
        camera_bev = self.encoders["camera"](images, extrinsic, intrinsic, augmentation,
                                                 torch.eye(4, device=device)[None])
        fused = self.fuser(torch.cat((camera_bev, lidar_bev), dim=1))
        return self.decoder["neck"](self.decoder["backbone"](fused))[0]

    @torch.inference_mode()
    def forward(self, images, points, camera_intrinsics=None, camera2lidar=None,
                img_aug_matrix=None, *, return_logits=False):
        features = self.extract_features(images, points, camera_intrinsics, camera2lidar, img_aug_matrix)
        if self.quantize_decoder:
            features = features.to(torch.float16).to(torch.float32)
        return self.heads["map"](features, return_logits=return_logits)
