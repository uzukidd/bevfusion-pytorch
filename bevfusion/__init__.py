"""Minimal, standalone PyTorch BEVFusion segmentation.

中文：独立的六相机与 LiDAR BEV 分割实现。
"""
from .model import BEVFusion, MAP_CLASSES, SURFACE_CLASSES
from .preprocessing import preprocess_images, voxelize

__all__ = ["BEVFusion", "MAP_CLASSES", "SURFACE_CLASSES", "preprocess_images", "voxelize"]
