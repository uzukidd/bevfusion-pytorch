"""Compatibility imports for the original segmentation API.

Use bevfusion.seg.BEVFusionSeg or bevfusion.det.BEVFusionDet for explicit tasks.
"""
from .base import _camera_matrix
from .seg import BEVFusion, BEVFusionSeg, FORMAT, MAP_CLASSES, SURFACE_CLASSES, _sha256

__all__ = ["BEVFusion", "BEVFusionSeg", "FORMAT", "MAP_CLASSES", "SURFACE_CLASSES"]
