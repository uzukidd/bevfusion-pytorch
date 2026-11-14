"""Shared tensor preprocessing for BEVFusion detection and segmentation.

中文：RGB 张量预处理与保持点顺序的硬体素化，可选已安装的 CUDA 算子。
"""
from __future__ import annotations

import sys

import torch
from torch.nn import functional as F

from .operators.voxelization.reference import prepare_points, voxelize


def _resize_images(images, size):
    # The optional Horizon plugin replaces F.interpolate with an implementation
    # rejecting antialias=True. It retains the native function in this module;
    # use that function for raw-image preprocessing, independent of import order.
    interpolate = F.interpolate
    if getattr(interpolate, "__module__", "").startswith("horizon_plugin_pytorch"):
        module = sys.modules.get("horizon_plugin_pytorch.nn.interpolate")
        native = getattr(module, "interpolate", None)
        if not callable(native) or getattr(native, "__module__", "").startswith("horizon_plugin_pytorch"):
            raise RuntimeError("Horizon plugin interpolation override does not expose native antialiased resizing")
        interpolate = native
    return interpolate(images, size=size, mode="bicubic", align_corners=False, antialias=True)


def preprocess_images(images: torch.Tensor):
    """Convert uint8 RGB [N,3,H,W] or [1,N,3,H,W] to normalized images.

    Returns float32 [1,N,3,256,704] and image augmentation [1,N,4,4].
    Resize by 0.48, centre crop horizontally and crop the bottom 256 rows.
    Tensor bicubic resize may differ slightly from the historical PIL resize.
    中文：返回标准化图像和图像增强矩阵；输入通道顺序必须为 RGB。
    """
    if not isinstance(images, torch.Tensor) or images.dtype != torch.uint8:
        raise ValueError("Raw RGB images must be a uint8 Tensor")
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.ndim != 5 or images.shape[0] != 1 or images.shape[2] != 3 or images.shape[1] < 1:
        raise ValueError("Expected uint8 RGB [N,3,H,W] or [1,N,3,H,W], batch=1")
    _, n, _, height, width = images.shape
    resized_h, resized_w = int(height * 0.48), int(width * 0.48)
    if min(resized_h, resized_w) < 1:
        raise ValueError("Images are too small for resize=0.48")
    left, top = int(max(0, resized_w - 704) / 2), resized_h - 256
    flat = images[0].float()
    flat = _resize_images(flat, size=(resized_h, resized_w))
    # PIL RGB resize rounds/clamps to uint8 before normalization.
    flat = flat.round().clamp(0, 255)
    pad_top, pad_right = max(0, -top), max(0, left + 704 - resized_w)
    flat = F.pad(flat, (0, pad_right, pad_top, 0))
    flat = flat[:, :, top + pad_top:top + pad_top + 256, left:left + 704] / 255.0
    mean = flat.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = flat.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    normalized = ((flat - mean) / std).unsqueeze(0)
    augmentation = torch.eye(4, device=images.device, dtype=torch.float32).repeat(1, n, 1, 1)
    augmentation[..., 0, 0] = augmentation[..., 1, 1] = 0.48
    augmentation[..., 0, 3], augmentation[..., 1, 3] = -left, -top
    return normalized, augmentation
