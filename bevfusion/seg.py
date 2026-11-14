"""Standalone six-camera + LiDAR BEV segmentation model."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .base import BEVFusionBase
from .camera import MAP_CLASSES, BEVSegmentationHead, NativeCameraEncoder
from .configuration import load_operator_config, resolve_pool_backend, resolve_pool_num_points
from .operators.sparse_conv.reference import ReferenceSparseEncoder

SURFACE_CLASSES = ("sidewalk", "road", "vegetation", "crosswalk")
FORMAT = "minimal-bevfusion-v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BEVFusionSeg(BEVFusionBase):
    """Segmentation probabilities [1,C,200,200], spatial axes [X,Y].

    Original six map classes and the historical four-class surface head are
    supported. The default reference backend preserves the previous inference.
    """
    task = "seg"

    def __init__(self, *, camera_intrinsics=None, camera2lidar=None,
                 img_aug_matrix=None, camera_batch_size=1, pool_depth_chunk=16,
                 classes=MAP_CLASSES, quantize_decoder=False, backend="reference", precompute=True,
                 pool_backend=None, lidar_backend=None, config=None, pool_num_points=None):
        if camera_batch_size < 0 or pool_depth_chunk < 1:
            raise ValueError("camera_batch_size must be >=0 and pool_depth_chunk must be >0")
        resolved_lidar_backend = backend if lidar_backend is None else lidar_backend
        if resolved_lidar_backend not in ("reference", "auto", "cuda", "compatible", "hybrid"):
            raise ValueError("lidar_backend must be reference, auto, cuda, compatible or hybrid")
        resolved = load_operator_config(config)
        resolved_pool_backend = resolve_pool_backend(pool_backend, resolved["operators"]["bev_pool"], backend)
        resolved_pool_num_points = resolve_pool_num_points(
            pool_num_points, resolved["operators"].get("bev_pool_num_points", 10))
        pool_options = {} if resolved_pool_num_points == 10 else {"pool_num_points": resolved_pool_num_points}
        classes = tuple(classes)
        if classes not in (tuple(MAP_CLASSES), SURFACE_CLASSES):
            raise ValueError("Supported classes are the original six map classes or four surface classes")
        super().__init__(
            NativeCameraEncoder(camera_batch_size, pool_depth_chunk,
                                 pool_backend=resolved_pool_backend,
                                precompute=precompute, **pool_options),
            ReferenceSparseEncoder(backend="reference" if resolved_lidar_backend == "compatible" else resolved_lidar_backend),
            backend=backend, lidar_backend=resolved_lidar_backend,
            camera_intrinsics=camera_intrinsics, camera2lidar=camera2lidar,
            img_aug_matrix=img_aug_matrix)
        self.config = resolved
        self.classes = classes
        self.quantize_decoder = bool(quantize_decoder)
        head = BEVSegmentationHead()
        if len(classes) != len(MAP_CLASSES):
            head.classifier[-1] = nn.Conv2d(512, len(classes), 1)
        head.classes = classes
        self.heads = nn.ModuleDict({"map": head})
        self.eval()

    @classmethod
    def from_pretrained(cls, checkpoint, *, head_checkpoint=None, device="cpu", **kwargs):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        packed = isinstance(payload, dict) and payload.get("format") == FORMAT
        if packed and head_checkpoint is not None:
            raise ValueError("A packed model already contains its head; omit head_checkpoint")
        if "config" in kwargs and "pool_num_points" not in kwargs:
            kwargs["pool_num_points"] = None
        if "config" in kwargs and "pool_backend" not in kwargs:
            kwargs["pool_backend"] = None
        model = cls._load_pretrained(checkpoint, FORMAT, device, kwargs, payload=payload)
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
            model.heads["map"].classifier[-1] = nn.Conv2d(512, 4, 1).to(device)
            model.heads["map"].load_state_dict(adapted["head_state_dict"], strict=True)
            model.classes = model.heads["map"].classes = SURFACE_CLASSES
            model.quantize_decoder = True
            model.load_info["head_tensors"] = len(adapted["head_state_dict"])
        return model.eval()

    def save_pretrained(self, path):
        self._save_pretrained(path, FORMAT, {
            "config": self.config,
            "classes": self.classes, "quantize_decoder": self.quantize_decoder,
            "camera_batch_size": self.encoders["camera"].camera_batch_size,
            "pool_depth_chunk": self.encoders["camera"]["vtransform"].pool_depth_chunk,
            "backend": self.backend,
            "lidar_backend": self.lidar_backend,
            "pool_backend": self.encoders["camera"]["vtransform"].pool_backend,
            "pool_num_points": getattr(self.encoders["camera"]["vtransform"], "pool_num_points", 10),
            "precompute": getattr(self.encoders["camera"]["vtransform"], "precompute_enabled", True),
        })

    @torch.inference_mode()
    def forward(self, images, points, camera_intrinsics=None, camera2lidar=None,
                img_aug_matrix=None, *, return_logits=False):
        features = self.extract_features(images, points, camera_intrinsics, camera2lidar, img_aug_matrix)
        if self.quantize_decoder:
            features = features.to(torch.float16).to(torch.float32)
        return self.heads["map"](features, return_logits=return_logits)


# Backward-compatible name for existing segmentation callers.
BEVFusion = BEVFusionSeg
