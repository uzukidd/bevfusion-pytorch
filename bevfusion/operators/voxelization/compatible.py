"""Per-frame CPU voxel grouping and a standard-ONNX voxel-mean subgraph.

Only grouping runs on CPU: geometry and point order are inspected anew on every
build_voxel_plan call. The graph takes fresh points, INT32 indices and reciprocal
counts as runtime inputs; it never embeds sample-dependent grouping constants.
This is a Nash-P compilation candidate, not a statement of BPU compilation.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np
import torch
from torch import nn

from .reference import prepare_points


def _positive_integer(value, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _capacities(point_capacity, voxel_capacity, max_points):
    point_capacity = _positive_integer(point_capacity, "point_capacity")
    voxel_capacity = _positive_integer(voxel_capacity, "voxel_capacity")
    max_points = _positive_integer(max_points, "max_points")
    if point_capacity >= 131072:
        raise ValueError("point_capacity must be <=131071 for the Nash-P tensor dimension limit")
    if voxel_capacity >= 131072:
        raise ValueError("voxel_capacity must be <131072 for the Nash-P tensor dimension limit")
    if max_points > 65535:
        raise ValueError("max_points must be <=65535 for the Nash-P ReduceSum axis limit")
    return point_capacity, voxel_capacity, max_points


@dataclass(frozen=True)
class VoxelizationPlan:
    """One frame's CPU grouping; the mean result is padded to voxel_capacity.

    padded_points: FLOAT32 [point_capacity,5], XYZ/intensity/time_lag_seconds.
    point_indices: INT32 [voxel_capacity,max_points]; point_capacity denotes a
        zero sentinel which CompatibleVoxelMean appends internally.
    inverse_counts: FLOAT32 [voxel_capacity,1]; empty rows contain zero.
    coordinates: INT32 [voxel_count,3] in XYZ order, first-seen voxel order.

    All tensors are CPU-owned and detached. Pass tensor_feed(device) to the mean
    module, then take output[:voxel_count] together with coordinates. Coordinates
    and voxel_count are per-frame application metadata, not frozen ONNX output.
    """
    padded_points: torch.Tensor
    point_indices: torch.Tensor
    inverse_counts: torch.Tensor
    coordinates: torch.Tensor
    point_count: int
    voxel_count: int
    point_capacity: int
    voxel_capacity: int
    max_points: int
    max_voxels: int
    in_bounds_points: int
    retained_points: int
    unique_voxels: int
    source_device: str
    source_dtype: str
    cpu_transfer_bytes: int

    def tensor_feed(self, device="cpu"):
        """Return (points, point_indices, inverse_counts) on the chosen device."""
        return tuple(value.to(device=device) for value in (
            self.padded_points, self.point_indices, self.inverse_counts))

    def statistics(self):
        return {
            "point_count": self.point_count,
            "voxel_count": self.voxel_count,
            "point_capacity": self.point_capacity,
            "voxel_capacity": self.voxel_capacity,
            "max_points": self.max_points,
            "max_voxels": self.max_voxels,
            "in_bounds_points": self.in_bounds_points,
            "retained_points": self.retained_points,
            "unique_voxels": self.unique_voxels,
            "discarded_voxels_by_max_voxels": self.unique_voxels - self.voxel_count,
            "source_device": self.source_device,
            "source_dtype": self.source_dtype,
            "cpu_transfer_bytes": self.cpu_transfer_bytes,
            "plan_tensor_bytes": sum(value.numel() * value.element_size() for value in (
                self.padded_points, self.point_indices, self.inverse_counts, self.coordinates)),
            "point_columns": ["x", "y", "z", "intensity", "time_lag_seconds"],
            "coordinate_order": "XYZ",
            "grouping_device": "cpu",
            "point_indices_dtype": "int32",
            "sentinel_index": self.point_capacity,
            "source_page_count": (self.point_capacity + 32767) // 32768,
            "source_page_axis_max": min(self.point_capacity, 32768) + 1,
        }


def build_voxel_plan(points, *, voxel_size, point_cloud_range, max_points=10,
                     max_voxels=120000, point_capacity=None, voxel_capacity=None):
    """Regroup one cloud on CPU, matching preprocessing.voxelize's semantics.

    Quantization uses separate FLOAT32 subtract/divide/floor and round-to-even
    grid extents, as the reference path does. Keep first-seen max_voxels and the
    first max_points within each retained voxel, regardless of later voxels.
    Exceeding either explicit capacity raises; raw points are never truncated.
    A GPU input is copied to CPU as detached five-column FP32 data, reported by
    cpu_transfer_bytes. Plans always return CPU tensors, including coordinates.
    """
    max_points = _positive_integer(max_points, "max_points")
    max_voxels = _positive_integer(max_voxels, "max_voxels")
    normalized = prepare_points(points)
    point_count = len(normalized)
    if point_capacity is None:
        point_capacity = max(1, point_count)
    point_capacity = _positive_integer(point_capacity, "point_capacity")
    if point_capacity >= 131072:
        raise ValueError("point_capacity must be <=131071; split the cloud outside this mean subgraph")
    if point_count > point_capacity:
        raise ValueError(f"Cloud contains {point_count} points, exceeding point_capacity={point_capacity}; no truncation")
    if max_points > 65535:
        raise ValueError("max_points must be <=65535 for the Nash-P ReduceSum axis limit")
    if voxel_capacity is not None:
        voxel_capacity = _positive_integer(voxel_capacity, "voxel_capacity")
        if voxel_capacity >= 131072:
            raise ValueError("voxel_capacity must be <131072 for the Nash-P tensor dimension limit")
    try:
        size = np.asarray(voxel_size, dtype=np.float32)
        bounds = np.asarray(point_cloud_range, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("voxel_size / point_cloud_range must contain finite numbers") from exc
    if size.shape != (3,) or bounds.shape != (6,):
        raise ValueError("voxel_size / point_cloud_range must have 3 / 6 elements")
    lower, upper = bounds[:3], bounds[3:]
    if (not np.isfinite(size).all() or not np.isfinite(bounds).all()
            or np.any(size <= 0) or np.any(upper <= lower)):
        raise ValueError("Voxel sizes must be positive and point cloud bounds finite and ordered")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        grid_float = np.rint((upper - lower) / size)
    # Coordinates are INT32 application metadata; detect impossible grids early.
    if not np.isfinite(grid_float).all() or np.any(grid_float >= 2**31):
        raise ValueError("Voxel grid dimensions exceed INT32 coordinate capacity")
    grid_shape = grid_float.astype(np.int64)
    host = normalized.detach().to(device="cpu", dtype=torch.float32).contiguous()
    values = host.numpy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        quantized = np.floor((values[:, :3] - lower) / size)
    valid = ((quantized >= 0) & (quantized < grid_shape)).all(axis=1)
    valid_ids = np.flatnonzero(valid)
    quantized_valid = quantized[valid].astype(np.int32)

    # Python dict insertion order is first-seen order. Store original row IDs so
    # the exact same grouping can be used with fresh runtime point features.
    grouped = {}
    selected_coordinates = []
    selected_ids = []
    seen_voxels = set()
    for row_id, coordinate in zip(valid_ids, quantized_valid):
        key = tuple(int(value) for value in coordinate)
        seen_voxels.add(key)
        voxel = grouped.get(key)
        if voxel is None:
            if len(selected_coordinates) >= max_voxels:
                continue
            voxel = len(selected_coordinates)
            grouped[key] = voxel
            selected_coordinates.append(coordinate.copy())
            selected_ids.append([])
        if len(selected_ids[voxel]) < max_points:
            selected_ids[voxel].append(int(row_id))
    voxel_count = len(selected_coordinates)
    if voxel_capacity is None:
        voxel_capacity = max(1, voxel_count)
    point_capacity, voxel_capacity, max_points = _capacities(point_capacity, voxel_capacity, max_points)
    if voxel_count > voxel_capacity:
        raise ValueError(f"Retained cloud contains {voxel_count} voxels, exceeding voxel_capacity={voxel_capacity}; "
                         "increase capacity instead of silently truncating voxels")
    padded = torch.zeros((point_capacity, 5), dtype=torch.float32)
    padded[:point_count].copy_(host)
    indices = torch.full((voxel_capacity, max_points), point_capacity, dtype=torch.int32)
    inverse_counts = torch.zeros((voxel_capacity, 1), dtype=torch.float32)
    for index, row_ids in enumerate(selected_ids):
        indices[index, :len(row_ids)] = torch.tensor(row_ids, dtype=torch.int32)
        inverse_counts[index, 0] = 1.0 / len(row_ids)
    coordinates = torch.from_numpy(np.asarray(selected_coordinates, dtype=np.int32).reshape(-1, 3).copy())
    return VoxelizationPlan(
        padded, indices, inverse_counts, coordinates, point_count, voxel_count,
        point_capacity, voxel_capacity, max_points, max_voxels, len(valid_ids),
        sum(len(ids) for ids in selected_ids), len(seen_voxels), str(points.device), str(points.dtype),
        point_count * 5 * 4 if points.device.type != "cpu" else 0)


class _GatherInt32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, point_indices):
        return points.index_select(0, point_indices.long().reshape(-1)).reshape(*point_indices.shape, 5)

    @staticmethod
    def symbolic(g, points, point_indices):
        result = g.op("Gather", points, point_indices, axis_i=0)
        sizes = point_indices.type().sizes()
        if sizes is not None:
            result.setType(points.type().with_sizes(sizes + [5]))
        return result


class CompatibleVoxelMean(nn.Module):
    """Mean-only static subgraph with per-frame runtime grouping inputs.

    forward(points[point_capacity,5], point_indices[voxel_capacity,max_points],
    inverse_counts[voxel_capacity,1]) returns FP32[voxel_capacity,5]. Each <=32768
    point source page appends a zero sentinel. Runtime range masks select the
    matching page; point_capacity is the global padding sentinel and contributes
    zero in every page. Regroup on CPU whenever point XYZ/order changes.
    """
    def __init__(self, point_capacity, voxel_capacity, max_points=10):
        super().__init__()
        self.point_capacity, self.voxel_capacity, self.max_points = _capacities(
            point_capacity, voxel_capacity, max_points)

    def forward(self, points, point_indices, inverse_counts):
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            if not all(isinstance(value, torch.Tensor) for value in (points, point_indices, inverse_counts)):
                raise ValueError("points, point_indices and inverse_counts must be tensors")
            if tuple(points.shape) != (self.point_capacity, 5):
                raise ValueError(f"points must have shape [{self.point_capacity},5]")
            if tuple(point_indices.shape) != (self.voxel_capacity, self.max_points):
                raise ValueError(f"point_indices must have shape [{self.voxel_capacity},{self.max_points}]")
            if tuple(inverse_counts.shape) != (self.voxel_capacity, 1):
                raise ValueError(f"inverse_counts must have shape [{self.voxel_capacity},1]")
            if points.dtype != torch.float32 or inverse_counts.dtype != torch.float32 or point_indices.dtype != torch.int32:
                raise ValueError("points/inverse_counts must be float32 and point_indices int32")
            if points.device != point_indices.device or points.device != inverse_counts.device:
                raise ValueError("points, point_indices and inverse_counts must share a device")
            if not torch.isfinite(points).all() or not torch.isfinite(inverse_counts).all():
                raise ValueError("points/inverse_counts must be finite")
            if (point_indices < 0).any() or (point_indices > self.point_capacity).any():
                raise ValueError("point_indices are outside the point buffer and zero sentinel")
            if (inverse_counts < 0).any() or (inverse_counts > 1).any():
                raise ValueError("inverse_counts must be in [0,1]")
        selected = None
        for start in range(0, self.point_capacity, 32768):
            end = min(start + 32768, self.point_capacity)
            width = end - start
            source = torch.cat((points[start:end], points.new_zeros((1, 5))), dim=0)
            in_page = (point_indices >= start) & (point_indices < end)
            local_indices = torch.where(in_page, point_indices - start,
                                        torch.full_like(point_indices, width))
            gathered = _GatherInt32.apply(source, local_indices)
            selected = gathered if selected is None else selected + gathered
        return selected.sum(dim=1) * inverse_counts
