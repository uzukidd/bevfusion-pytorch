"""Fixed-calibration BEV pooling using standard INT32 Gather/Mul/ReduceSum.

Geometry is converted into bounded source pages offline. There is no runtime
sorting, coordinate filtering or scatter. This is a Nash-P compilation candidate;
actual BPU placement and performance require the matching target SDK.
"""
from __future__ import annotations

import hashlib
import math
import operator

import numpy as np
import torch
from torch import nn


class _GatherInt32(torch.autograd.Function):
    """Standard Gather accepts INT32; GatherND's ONNX schema does not."""
    @staticmethod
    def forward(ctx, values, indices):
        return values.index_select(0, indices.long().reshape(-1)).reshape(
            *indices.shape, *values.shape[1:])

    @staticmethod
    def symbolic(g, values, indices):
        result = g.op("Gather", values, indices, axis_i=0)
        vs, ids = values.type().sizes(), indices.type().sizes()
        if vs is not None and ids is not None:
            result.setType(values.type().with_sizes(ids + vs[1:]))
        return result


class _BoundedConcat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *values):
        return torch.cat(values, dim=0)

    @staticmethod
    def symbolic(g, *values):
        # Emit after tracing so the JIT cannot flatten a nested torch.cat tree.
        values = list(values)
        while len(values) > 64:
            values = [g.op("Concat", *values[i:i + 64], axis_i=0)
                      for i in range(0, len(values), 64)]
        return g.op("Concat", *values, axis_i=0)


def _concat(values):
    return values[0] if len(values) == 1 else _BoundedConcat.apply(*values)

class _ReductionPart(nn.Module):
    def __init__(self, depth_indices, feature_indices):
        super().__init__()
        self.register_buffer("depth_indices", torch.from_numpy(depth_indices.copy()).int())
        self.register_buffer("feature_indices", torch.from_numpy(feature_indices.copy()).int())

    def forward(self, depth, pixels):
        # Gather before promotion, multiply after promotion: FP16 IO/FP32 math.
        probability = _GatherInt32.apply(depth, self.depth_indices).float()
        feature = _GatherInt32.apply(pixels, self.feature_indices).float()
        return (probability.unsqueeze(-1) * feature).sum(dim=1)


class _RowReduction(nn.Module):
    def __init__(self, parts, rows):
        super().__init__()
        self.parts = nn.ModuleList(parts)
        self.rows = rows

    def forward(self, depth, pixels):
        result = self.parts[0](depth, pixels)
        for part in self.parts[1:]:
            result = result + part(depth, pixels)
        return result


class _SourcePage(nn.Module):
    def __init__(self, page, camera, reductions, lookup, depth_width, pixel_count, channels):
        super().__init__()
        self.page, self.camera = page, camera
        self.depth_width, self.pixel_count, self.channels = depth_width, pixel_count, channels
        self.reductions = nn.ModuleList(reductions)
        self.register_buffer("lookup", torch.from_numpy(lookup.copy()).int())

    def forward(self, pages, pixels, zero):
        depth = pages[self.page:self.page + 1].reshape(self.depth_width)
        features = pixels[self.camera:self.camera + 1].reshape(self.pixel_count, self.channels)
        rows = [zero] + [reduction(depth, features) for reduction in self.reductions]
        return _GatherInt32.apply(_concat(rows), self.lookup)


class _ReductionGroup(nn.Module):
    def __init__(self, pages, rows):
        super().__init__()
        self.pages = nn.ModuleList(pages)
        self.rows = rows

    def forward(self, depth, pixels, zero):
        result = self.pages[0](depth, pixels, zero)
        for page in self.pages[1:]:
            result = result + page(depth, pixels, zero)
        return result


class _DenseGather(nn.Module):
    def __init__(self, page, indices):
        super().__init__()
        self.page = page
        self.register_buffer("indices", torch.from_numpy(indices.copy()).int())

    def forward(self, pages):
        return _GatherInt32.apply(pages[self.page], self.indices)


class _DensePart(nn.Module):
    def __init__(self, gathers):
        super().__init__()
        self.gathers = nn.ModuleList(gathers)

    def forward(self, pages):
        result = self.gathers[0](pages)
        for gather in self.gathers[1:]:
            result = result + gather(pages)
        return result


def _positive_integer(value, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


class CompatibleBEVPool(nn.Module):
    """Inference-only fixed geometry/shapes; fresh inputs; output FP32 [B,Z*C,X,Y].

    Source depth pages contain <=16 depth planes, followed by a zero sentinel.
    Each cell's contributions are split by source page, bucketed by count and
    reduced. Page-local gathers restore cell order and Add combines partials.
    Compact result pages each have a zero row; Gather/Add reconstructs dense BEV.
    Every standard Gather uses INT32 indices and a source axis <=65536.

    Power-of-two buckets have <2x valid pairs; splitting never drops a point.
    max_pairs_per_chunk limits one weighted tile, NOT total executor peak memory:
    a compiler may schedule independent branches concurrently. FP32 reduction
    order differs from reference/CUDA; verify floating-point tolerances.
    """
    _ROW_PAGE = 32768

    def __init__(self, coordinates, grid_shape, channels, *, max_pairs_per_chunk=16384):
        super().__init__()
        if not isinstance(coordinates, torch.Tensor):
            raise ValueError("coordinates must be an integer tensor [B,N,D,H,W,3]")
        if coordinates.ndim != 6 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [B,N,D,H,W,3]")
        if coordinates.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("coordinates must have integer dtype")
        if len(grid_shape) != 3:
            raise ValueError("grid_shape must contain X,Y,Z")
        self.grid_shape = tuple(_positive_integer(v, "grid dimension") for v in grid_shape)
        self.channels = _positive_integer(channels, "channels")
        self.max_pairs_per_chunk = _positive_integer(max_pairs_per_chunk, "max_pairs_per_chunk")
        if self.max_pairs_per_chunk > 65535:
            raise ValueError("max_pairs_per_chunk must be <=65535 for Nash-P reduction limits")
        self.depth_shape = tuple(int(v) for v in coordinates.shape[:-1])
        if any(v < 1 for v in self.depth_shape):
            raise ValueError("B,N,D,H,W must be positive; empty geometry uses out-of-range coordinates")
        batch, cameras, depths, height, width = self.depth_shape
        self.feature_shape = (batch, cameras, self.channels, height, width)
        nx, ny, nz = self.grid_shape
        self.output_shape = (batch, nz * self.channels, nx, ny)
        total_depth, volume, pixels = math.prod(self.depth_shape), nx * ny * nz, height * width
        total_cells = batch * volume
        if max(total_depth, total_cells, batch * cameras * pixels) >= 2**31 - 1:
            raise ValueError("Pooling tensor indices exceed INT32 capacity")
        if total_cells >= 131072 or pixels >= 65535 or batch * cameras > 65535:
            raise ValueError("Static pooling requires B*Z*X*Y<131072, H*W<65535 and B*N<=65535")
        if any(v >= 131072 for v in (*self.depth_shape, *self.feature_shape, *self.output_shape)):
            raise ValueError("Tensor dimensions must be <131072 for the Nash-P candidate")
        self.depth_page_planes = min(depths, 16, 65534 // pixels)
        self.depth_pages_per_camera = math.ceil(depths / self.depth_page_planes)
        self.depth_padding = self.depth_pages_per_camera * self.depth_page_planes - depths
        self.depth_page_width = self.depth_page_planes * pixels
        source_pages = batch * cameras * self.depth_pages_per_camera
        if source_pages > 65536:
            raise ValueError("Too many depth source pages for the static pooling candidate")

        co = coordinates.detach().cpu().to(torch.int64).contiguous().numpy().reshape(-1, 3)
        self.geometry_sha256 = hashlib.sha256(
            np.asarray((*self.depth_shape, *self.grid_shape), dtype="<i8").tobytes()
            + co.astype("<i8", copy=False).tobytes()).hexdigest()
        valid = ((co >= 0) & (co < np.array(self.grid_shape))).all(axis=1)
        depth_ids = np.flatnonzero(valid)
        selected = co[valid]
        batch_ids = depth_ids // (cameras * depths * pixels)
        ranks = ((batch_ids * nz + selected[:, 2]) * nx + selected[:, 0]) * ny + selected[:, 1]
        permutation = np.argsort(ranks, kind="stable")
        ranks, depth_ids = ranks[permutation], depth_ids[permutation]
        cells, starts, counts = np.unique(ranks, return_index=True, return_counts=True)
        self.groups = nn.ModuleList()
        dense_lookup = np.zeros(total_cells, dtype=np.int32)
        result_offset, padded_pairs, max_tile_pairs = 1, 0, 0
        partial_count, bucket_statistics = 0, []
        global_buckets = np.array([1 << (int(c) - 1).bit_length() for c in counts], dtype=np.int64)
        for global_bucket in np.unique(global_buckets):
            members = np.flatnonzero(global_buckets == global_bucket)
            rows_per_group = min(32767, max(1, self.max_pairs_per_chunk // min(int(global_bucket), self.max_pairs_per_chunk)))
            for first in range(0, len(members), rows_per_group):
                group_ids = members[first:first + rows_per_group]
                row_count = len(group_ids)
                lengths = counts[group_ids]
                di = np.concatenate([depth_ids[start:start + count] for start, count in zip(starts[group_ids], lengths)])
                row_ids = np.repeat(np.arange(row_count), lengths)
                camera_ids = di // (depths * pixels)
                local = di % (depths * pixels)
                page_ids = camera_ids * self.depth_pages_per_camera + local // self.depth_page_width
                source_modules = []
                for page in np.unique(page_ids):
                    page_mask = page_ids == page
                    page_rows, page_di = row_ids[page_mask], di[page_mask]
                    active_rows, local_starts, local_counts = np.unique(page_rows, return_index=True, return_counts=True)
                    local_buckets = np.array([1 << (int(c) - 1).bit_length() for c in local_counts], dtype=np.int64)
                    lookup = np.zeros(row_count, dtype=np.int32)
                    offset, reductions = 1, []
                    for bucket in np.unique(local_buckets):
                        local_members = np.flatnonzero(local_buckets == bucket)
                        part_width = min(int(bucket), self.max_pairs_per_chunk)
                        local_rows_per_group = max(1, self.max_pairs_per_chunk // part_width)
                        bucket_statistics.append({"width": int(bucket), "cells": len(local_members)})
                        for begin in range(0, len(local_members), local_rows_per_group):
                            ids = local_members[begin:begin + local_rows_per_group]
                            parts = []
                            padded_ids = np.append(page_di, total_depth)
                            for start in range(0, int(bucket), part_width):
                                positions = np.arange(start, min(start + part_width, int(bucket)))
                                present = positions[None, :] < local_counts[ids, None]
                                pick = np.where(present, local_starts[ids, None] + positions[None, :], len(page_di))
                                contribution = padded_ids[pick]
                                depth_index = np.where(present, contribution % (depths * pixels) % self.depth_page_width,
                                                       self.depth_page_width).astype(np.int32)
                                feature_index = np.where(present, contribution % pixels, 0).astype(np.int32)
                                parts.append(_ReductionPart(depth_index, feature_index))
                                padded_pairs += depth_index.size
                                max_tile_pairs = max(max_tile_pairs, depth_index.size)
                            reductions.append(_RowReduction(parts, len(ids)))
                            lookup[active_rows[ids]] = np.arange(offset, offset + len(ids), dtype=np.int32)
                            offset += len(ids)
                    source_modules.append(_SourcePage(int(page), int(page) // self.depth_pages_per_camera,
                        reductions, lookup, self.depth_page_width + 1, pixels, self.channels))
                    partial_count += 1
                self.groups.append(_ReductionGroup(source_modules, row_count))
                dense_lookup[cells[group_ids]] = np.arange(result_offset, result_offset + row_count, dtype=np.int32)
                result_offset += row_count

        self.row_page_width = min(self._ROW_PAGE, result_offset)
        capacity = self.row_page_width - 1
        self.row_page_recipe = []
        pending, filled = [], 0
        for index, group in enumerate(self.groups, start=1):
            start = 0
            while start < group.rows:
                length = min(group.rows - start, capacity - filled)
                pending.append((index, start, start + length))
                start += length
                filled += length
                if filled == capacity:
                    self.row_page_recipe.append((tuple(pending), 0))
                    pending, filled = [], 0
        if pending or not self.row_page_recipe:
            self.row_page_recipe.append((tuple(pending), capacity - filled))
        self.dense_parts = nn.ModuleList()
        for start in range(0, total_cells, self._ROW_PAGE):
            lookup = dense_lookup[start:start + self._ROW_PAGE]
            if capacity:
                page_ids = np.where(lookup > 0, (lookup - 1) // capacity, 0)
                local_ids = np.where(lookup > 0, (lookup - 1) % capacity + 1, 0)
            else:
                page_ids, local_ids = np.zeros_like(lookup), np.zeros_like(lookup)
            gathers = []
            for page in np.unique(page_ids):
                ids = np.where(page_ids == page, local_ids, 0).astype(np.int32)
                gathers.append(_DenseGather(int(page), ids))
            self.dense_parts.append(_DensePart(gathers))
        percentile = np.percentile(counts, [50, 95, 99]).tolist() if counts.size else [0., 0., 0.]
        bucket_totals = {}
        for value in bucket_statistics:
            bucket_totals[value["width"]] = bucket_totals.get(value["width"], 0) + value["cells"]
        self._statistics = {
            "valid_points": len(depth_ids), "active_cells": len(cells),
            "max_contributors": int(counts.max()) if counts.size else 0,
            "contributor_percentiles": dict(zip(("p50", "p95", "p99"), percentile)),
            "padded_pairs": padded_pairs, "padding_ratio": padded_pairs / len(depth_ids) if len(depth_ids) else 0.,
            "buckets": [{"width": width, "cell_page_pairs": count} for width, count in sorted(bucket_totals.items())],
            "groups": len(self.groups), "source_page_partials": partial_count,
            "reduction_parts": sum(len(reduction.parts) for group in self.groups for page in group.pages for reduction in page.reductions),
            "dense_partial_count": sum(len(part.gathers) for part in self.dense_parts),
            "max_pairs_per_chunk": self.max_pairs_per_chunk, "max_tile_pairs": max_tile_pairs,
            "depth_source_pages": source_pages, "depth_source_page_width": self.depth_page_width + 1,
            "result_source_pages": len(self.row_page_recipe), "result_source_page_width": self.row_page_width,
            "largest_weighted_tile_bytes": max_tile_pairs * self.channels * 4,
        }
        self.eval()

    @classmethod
    def from_transform(cls, transform, camera2lidar, camera_intrinsics, img_aug_matrix,
                       lidar_aug_matrix=None, *, max_pairs_per_chunk=16384):
        """Use exact source geometry/truncation; rebuild when calibration changes."""
        device = transform.frustum.device
        def matrix(value, name, allowed):
            value = torch.as_tensor(value, device=device, dtype=torch.float32)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4 or tuple(value.shape[-2:]) not in allowed or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name} camera matrices")
            return value
        camera2lidar = matrix(camera2lidar, "camera2lidar", ((4, 4),))
        camera_intrinsics = matrix(camera_intrinsics, "camera_intrinsics", ((3, 3), (4, 4)))
        img_aug_matrix = matrix(img_aug_matrix, "img_aug_matrix", ((4, 4),))
        batch, cameras = camera2lidar.shape[:2]
        if camera_intrinsics.shape[:2] != (batch, cameras) or img_aug_matrix.shape[:2] != (batch, cameras):
            raise ValueError("Camera matrices must have matching batch/camera dimensions")
        if lidar_aug_matrix is None:
            lidar_aug_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(batch, 1, 1)
        else:
            lidar_aug_matrix = torch.as_tensor(lidar_aug_matrix, device=device, dtype=torch.float32)
            if lidar_aug_matrix.shape != (batch, 4, 4) or not torch.isfinite(lidar_aug_matrix).all():
                raise ValueError("lidar_aug_matrix must be finite [B,4,4]")
        with torch.no_grad():
            coordinates = transform._voxel_indices(transform.get_geometry(
                camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix))
        return cls(coordinates, tuple(int(v) for v in transform.nx), transform.C,
                   max_pairs_per_chunk=max_pairs_per_chunk)

    def statistics(self):
        return {**self._statistics, "grid_shape": list(self.grid_shape),
                "depth_shape": list(self.depth_shape), "feature_shape": list(self.feature_shape),
                "output_shape": list(self.output_shape), "geometry_sha256": self.geometry_sha256,
                "index_dtype": "int32", "accumulation_dtype": "float32", "output_dtype": "float32",
                "index_bytes": sum(buffer.numel() * buffer.element_size() for buffer in self.buffers())}

    def forward(self, depth, features):
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            if tuple(depth.shape) != self.depth_shape or tuple(features.shape) != self.feature_shape:
                raise ValueError(f"Expected depth {self.depth_shape}, features {self.feature_shape}")
            if depth.dtype not in (torch.float16, torch.float32) or features.dtype != depth.dtype:
                raise ValueError("depth/features must share float16 or float32 dtype")
            if depth.device != features.device or depth.device != next(self.buffers()).device:
                raise ValueError("depth/features and pooling indices must share a device")
            if not torch.isfinite(depth).all() or not torch.isfinite(features).all():
                raise ValueError("depth/features must be finite")
        batch, cameras, _, height, width = self.depth_shape
        if self.depth_padding:
            depth = torch.cat((depth, depth.new_zeros((batch, cameras, self.depth_padding, height, width))), dim=2)
        pages = depth.reshape(batch * cameras * self.depth_pages_per_camera, self.depth_page_width)
        pages = torch.cat((pages, pages.new_zeros((batch * cameras * self.depth_pages_per_camera, 1))), dim=1)
        pixels = features.permute(0, 1, 3, 4, 2).reshape(batch * cameras, height * width, self.channels)
        zero = pixels[0:1, :1].reshape(1, self.channels).float() * 0.0 + pages[0:1, :1].reshape(1, 1).float() * 0.0
        rows = [zero] + [group(pages, pixels, zero) for group in self.groups]
        result_pages = []
        for recipe, padding in self.row_page_recipe:
            pieces = [zero] + [rows[index][start:end] for index, start, end in recipe]
            if padding:
                pieces.append(zero.new_zeros((padding, self.channels)))
            result_pages.append(_concat(pieces))
        dense = _concat([part(result_pages) for part in self.dense_parts])
        nx, ny, nz = self.grid_shape
        return dense.reshape(batch, nz, nx, ny, self.channels).permute(0, 1, 4, 2, 3).reshape(
            batch, nz * self.channels, nx, ny)
