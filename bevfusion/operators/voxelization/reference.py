"""Reference point validation and hard voxelization with optional CUDA dispatch."""
from __future__ import annotations

import torch


def prepare_points(points: torch.Tensor):
    """Validate XYZ/intensity/time-lag input and append zero lag to four columns."""
    if not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] not in (4, 5):
        raise ValueError("Expected points Tensor[P,4 or 5]")
    if not points.is_floating_point() or not torch.isfinite(points).all():
        raise ValueError("Point coordinates/features must be finite floating point values")
    points = points.float()
    if points.shape[1] == 4:
        points = torch.cat((points, points.new_zeros((len(points), 1))), dim=1)
    return points


def voxelize(points: torch.Tensor, max_points: int = 10, max_voxels: int = 120000,
             *, voxel_size=(0.1, 0.1, 0.2),
             point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
             backend="reference"):
    """Hard voxel mean, preserving first-seen voxel and point order.

    Input [P,4] = XYZ/intensity or [P,5] = XYZ/intensity/time lag (seconds).
    Four-column inputs get zero time lag. Output: float32 [M,5], int64 [M,3]
    XYZ coordinates, on the input device. The reference path uses only Torch;
    auto selects the installed voxelization extension for CUDA tensors.
    Defaults preserve segmentation geometry; detection passes its own bounds.
    中文：默认体素 0.1/0.1/0.2 米、XY±51.2；检测模型显式传入相应参数。
    """
    points = prepare_points(points)
    if max_points < 1 or max_voxels < 1:
        raise ValueError("max_points and max_voxels must be positive")
    if backend not in ("reference", "cuda", "auto"):
        raise ValueError("backend must be reference, cuda or auto")
    if len(voxel_size) != 3 or len(point_cloud_range) != 6:
        raise ValueError("voxel_size / point_cloud_range must have 3 / 6 elements")
    size = points.new_tensor(voxel_size)
    lower = points.new_tensor(point_cloud_range[:3])
    upper = points.new_tensor(point_cloud_range[3:])
    if not torch.isfinite(size).all() or not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (size <= 0).any() or (upper <= lower).any():
        raise ValueError("Voxel sizes must be positive and point cloud bounds finite and ordered")
    if backend == "cuda" and not points.is_cuda:
        raise ValueError("The cuda backend requires CUDA tensors")
    if points.is_cuda and backend in ("auto", "cuda"):
        try:
            from ops.voxel import Voxelization
        except (ImportError, OSError) as exc:
            raise RuntimeError("Install bevfusion-voxel-ops with scripts/install_ops.bat") from exc
        if not len(points):
            return points.new_empty((0, 5)), torch.empty((0, 3), device=points.device, dtype=torch.long)
        op = Voxelization(list(voxel_size), list(point_cloud_range), max_points,
                          (max_voxels, max_voxels)).eval()
        values, coordinates, counts = op(points.contiguous())
        return values.sum(1) / counts[:, None].clamp_min(1), coordinates.long()
    shape = torch.round((upper - lower) / size).long()
    coords = torch.floor((points[:, :3] - lower) / size).long()
    keep = ((coords >= 0) & (coords < shape)).all(1)
    points, coords = points[keep], coords[keep]
    if not len(points):
        return points.new_empty((0, 5)), coords.new_empty((0, 3))
    # Use the same extra Z cell as the sparse encoder; this also preserves
    # the original sorted voxel order for the segmentation profile.
    keys = (coords[:, 0] * shape[1] + coords[:, 1]) * (shape[2] + 1) + coords[:, 2]
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    first = torch.full((len(unique),), len(points), dtype=torch.long, device=points.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(points), device=points.device), reduce="amin", include_self=True)
    voxel_order = torch.argsort(first, stable=True)[:max_voxels]
    remap = torch.full_like(first, -1)
    remap[voxel_order] = torch.arange(len(voxel_order), device=points.device)
    voxel_id = remap[inverse]
    keep = voxel_id >= 0
    selected_points, voxel_id = points[keep], voxel_id[keep]
    order = torch.argsort(voxel_id, stable=True)
    sorted_id = voxel_id[order]
    starts_mask = torch.ones(len(order), dtype=torch.bool, device=points.device)
    starts_mask[1:] = sorted_id[1:] != sorted_id[:-1]
    positions = torch.arange(len(order), device=points.device)
    starts = torch.where(starts_mask, positions, 0).cummax(0).values
    selected = order[(positions - starts) < max_points]
    sums = points.new_zeros((len(voxel_order), 5))
    counts = points.new_zeros((len(voxel_order), 1))
    sums.index_add_(0, voxel_id[selected], selected_points[selected])
    counts.index_add_(0, voxel_id[selected], points.new_ones((len(selected), 1)))
    return sums / counts, coords[first[voxel_order]]
