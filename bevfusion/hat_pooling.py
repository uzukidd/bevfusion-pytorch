"""Fixed-slot LSS pooling expressed as standard GridSample, Mul and Add.

This reproduces the *structure* described by the HAT FAQ, not undocumented HAT
reference-point selection. We choose first_depth_rank (ascending B,N,D,H,W),
bilinear interpolation, pixel-centre grids and align_corners=False explicitly.
Keeping only K contributions changes full pooling whenever a cell overflows;
no normalization, rescaling or compensation is applied. This is an inference
ONNX candidate, not a claim of successful BPU compilation or equivalent accuracy.
"""
from __future__ import annotations

import hashlib
import math
import operator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _positive(value, name, maximum=None):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, (bool, np.bool_)) or result < 1 or (maximum is not None and result > maximum):
        raise ValueError(f"{name} must be in [1,{maximum}]" if maximum else f"{name} must be positive")
    return result


class HATLSSPool(nn.Module):
    """CPU-planned fixed slots; fresh FP32 depth/features produce [B,Z*C,X,Y].

    coordinates: integer [B,N,D,H,W,3], XYZ; out-of-grid coordinates are ignored.
    num_points: first K valid contributions per cell, K <= 1024. This selection
        policy is our implementation choice, not HAT's unpublished exact sort key.
    source_page_shape: maximum (height,width) of each flattened source page,
        power-of-two height <=512 and width <=1024; small shapes also facilitate
        boundary tests. Pages use a power-of-two width dividing the source size
        and pad tail heights to powers of two, so pixel-centre coordinates are
        exactly representable before quantization.
    Geometry is fixed for this module; rebuild it after calibration changes.
    No Gather, TopK or coordinate-dependent operation is used in forward.
    """

    def __init__(self, coordinates, grid_shape, channels, *, num_points=10,
                 source_page_shape=(512, 1024)):
        super().__init__()
        if (not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 6
                or coordinates.shape[-1] != 3 or coordinates.dtype not in
                (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("coordinates must be integer [B,N,D,H,W,3] XYZ")
        self.depth_shape = tuple(int(v) for v in coordinates.shape[:-1])
        if min(self.depth_shape) < 1:
            raise ValueError("B,N,D,H,W must be positive; represent empty geometry with invalid coordinates")
        try:
            grid_shape, source_page_shape = tuple(grid_shape), tuple(source_page_shape)
        except TypeError as exc:
            raise ValueError("grid_shape/source_page_shape must be sequences") from exc
        if len(grid_shape) != 3 or len(source_page_shape) != 2:
            raise ValueError("grid_shape must contain XYZ, source_page_shape must contain height,width")
        self.grid_shape = tuple(_positive(v, "grid dimension") for v in grid_shape)
        self.channels = _positive(channels, "channels")
        self.num_points = _positive(num_points, "num_points", 1024)
        self.source_page_shape = (_positive(source_page_shape[0], "source page height", 512),
                                  _positive(source_page_shape[1], "source page width", 1024))
        if any(value & (value - 1) for value in self.source_page_shape):
            raise ValueError("source_page_shape dimensions must be powers of two for exact pixel-centre sampling")
        batch, cameras, depths, height, width = self.depth_shape
        nx, ny, nz = self.grid_shape
        self.feature_shape = (batch, cameras, self.channels, height, width)
        self.output_shape = (batch, nz * self.channels, nx, ny)
        self.sample_shape = (batch, nz * nx, ny)
        if nz * nx > 1024 or ny > 1024 or nz * nx * ny > 720 * 1024:
            raise ValueError("Packed output must have height,width <=1024 and area <=720*1024")
        self._feature_size = cameras * height * width
        self._depth_size = cameras * depths * height * width
        self._feature_pages = self._pages(self._feature_size)
        self._depth_pages = self._pages(self._depth_size)

        xyz = coordinates.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
        flat = xyz.reshape(-1, 3)
        valid = ((flat >= 0) & (flat < np.asarray(self.grid_shape))).all(axis=1)
        global_depth = np.flatnonzero(valid)
        per_batch_depth = self._depth_size
        batches = global_depth // per_batch_depth
        selected_xyz = flat[valid]
        cells = ((batches * nz + selected_xyz[:, 2]) * nx + selected_xyz[:, 0]) * ny + selected_xyz[:, 1]
        order = np.argsort(cells, kind="stable")
        cells, global_depth = cells[order], global_depth[order]
        if len(cells):
            starts = np.r_[0, np.flatnonzero(cells[1:] != cells[:-1]) + 1]
            counts = np.diff(np.r_[starts, len(cells)])
            positions = np.arange(len(cells)) - np.repeat(starts, counts)
        else:
            counts = np.empty(0, np.int64)
            positions = np.empty(0, np.int64)
        kept = positions < self.num_points
        slots = np.full((self.num_points, batch * nz * nx * ny), -1, np.int64)
        slots[positions[kept], cells[kept]] = global_depth[kept] % per_batch_depth
        slots = slots.reshape(self.num_points, *self.sample_shape)
        self.valid_points = int(valid.sum())
        self.selected_points = int(kept.sum())
        self.dropped_points = self.valid_points - self.selected_points
        self.max_contributors = int(counts.max()) if counts.size else 0
        self.overflow_cells = int((counts > self.num_points).sum())
        self.active_cells = len(counts)
        self.geometry_sha256 = hashlib.sha256(
            str((self.depth_shape, self.grid_shape)).encode("ascii")
            + xyz.astype("<i8", copy=False).tobytes()).hexdigest()
        for k in range(self.num_points):
            depth_indices = slots[k]
            pixel = depth_indices % (height * width)
            camera = depth_indices // (depths * height * width)
            feature_indices = camera * height * width + pixel
            feature_indices[depth_indices < 0] = -1
            mask = (depth_indices >= 0).astype(np.float32)[:, None]
            self.register_buffer(f"slot_mask_{k}", torch.from_numpy(mask).to(coordinates.device))
            for kind, indices, pages in (("feature", feature_indices, self._feature_pages),
                                         ("depth", depth_indices, self._depth_pages)):
                for page, (start, end, ph, pw) in enumerate(pages):
                    present = (indices >= start) & (indices < end)
                    local = np.where(present, indices - start, 0)
                    grid = np.full((*self.sample_shape, 2), 2.0, np.float32)
                    gx = (2.0 * ((local % pw).astype(np.float64) + .5) / pw - 1).astype(np.float32)
                    gy = (2.0 * ((local // pw).astype(np.float64) + .5) / ph - 1).astype(np.float32)
                    grid[..., 0][present] = gx[present]
                    grid[..., 1][present] = gy[present]
                    self.register_buffer(f"{kind}_grid_{k}_{page}", torch.from_numpy(grid).to(coordinates.device))
                    self.register_buffer(f"{kind}_mask_{k}_{page}",
                                         torch.from_numpy(present.astype(np.float32)[:, None]).to(coordinates.device))
        self.eval()

    def _pages(self, count):
        height, width = self.source_page_shape
        width = min(width, count & -count)
        if count // width >= 131072:
            raise ValueError("Packed source height must be <131072; choose a compatible source layout")
        capacity = height * width
        return tuple((start, min(start + capacity, count),
                      1 << (math.ceil(min(capacity, count - start) / width) - 1).bit_length(),
                      width) for start in range(0, count, capacity))

    @classmethod
    def from_transform(cls, transform, camera2lidar, camera_intrinsics, img_aug_matrix,
                       lidar_aug_matrix=None, *, num_points=10, source_page_shape=(512, 1024)):
        """Use the source transform's exact geometry and voxel quantization."""
        device = transform.frustum.device
        matrices = []
        for value, name, sizes in ((camera2lidar, "camera2lidar", (4,)),
                                   (camera_intrinsics, "camera_intrinsics", (3, 4)),
                                   (img_aug_matrix, "img_aug_matrix", (4,))):
            value = torch.as_tensor(value, dtype=torch.float32, device=device)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if (value.ndim != 4 or value.shape[-1] not in sizes
                    or value.shape[-2] != value.shape[-1] or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid {name} matrices")
            matrices.append(value)
        batch, cameras = matrices[0].shape[:2]
        if any(value.shape[:2] != (batch, cameras) for value in matrices):
            raise ValueError("Camera matrices must have matching B,N dimensions")
        if lidar_aug_matrix is None:
            lidar_aug_matrix = torch.eye(4, device=device)[None].repeat(batch, 1, 1)
        else:
            lidar_aug_matrix = torch.as_tensor(lidar_aug_matrix, dtype=torch.float32, device=device)
            if tuple(lidar_aug_matrix.shape) != (batch, 4, 4) or not torch.isfinite(lidar_aug_matrix).all():
                raise ValueError("lidar_aug_matrix must be finite [B,4,4]")
        with torch.no_grad():
            coordinates = transform._voxel_indices(transform.get_geometry(*matrices, lidar_aug_matrix))
        return cls(coordinates, tuple(int(v) for v in transform.nx), transform.C,
                   num_points=num_points, source_page_shape=source_page_shape)

    def statistics(self):
        return {"backend": "hat", "operator": "GridSample/Mul/Add", "num_points": self.num_points,
                "valid_points": self.valid_points, "selected_points": self.selected_points,
                "dropped_points": self.dropped_points,
                "retained_ratio": self.selected_points / self.valid_points if self.valid_points else 1.0,
                "max_contributors": self.max_contributors, "overflow_cells": self.overflow_cells,
                "active_cells": self.active_cells, "selection_policy": "first_depth_rank",
                "approximate": bool(self.dropped_points), "normalization": "none",
                "interpolation": "bilinear", "align_corners": False,
                "pixel_centres_exact_before_quantization": True,
                "official_exact_sort_key": False, "groups": self.num_points,
                "index_bytes": sum(v.numel() * v.element_size() for v in self.buffers()),
                "geometry_sha256": self.geometry_sha256, "grid_shape": list(self.grid_shape),
                "depth_shape": list(self.depth_shape), "feature_shape": list(self.feature_shape),
                "output_shape": list(self.output_shape), "source_page_shape": list(self.source_page_shape),
                "feature_pages": len(self._feature_pages), "depth_pages": len(self._depth_pages),
                "source_shapes": {"features": [[self.depth_shape[0], self.channels, p[2], p[3]] for p in self._feature_pages],
                                  "depth": [[self.depth_shape[0], 1, p[2], p[3]] for p in self._depth_pages]},
                "grid_sample_count": self.num_points * (len(self._feature_pages) + len(self._depth_pages)),
                "bpu_compiled": False}

    def _pack(self, values, pages):
        batch, channels = self.depth_shape[0], int(values.shape[1])
        width = pages[0][3]
        packed = values.reshape(batch, channels, -1, width)
        result = []
        for start, end, ph, pw in pages:
            page = packed[:, :, start // width:end // width, :]
            padding = ph - (end - start) // width
            if padding:
                page = torch.cat((page, values.new_zeros((batch, channels, padding, pw))), dim=2)
            result.append(page)
        return tuple(result)

    def _sample(self, pages, kind, slot):
        result = None
        for page, values in enumerate(pages):
            sampled = F.grid_sample(values, getattr(self, f"{kind}_grid_{slot}_{page}"),
                                    mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled * getattr(self, f"{kind}_mask_{slot}_{page}")
            result = sampled if result is None else result + sampled
        return result

    def forward(self, depth, features):
        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("HATLSSPool is inference-only; use eval() and inference_mode()")
            if not isinstance(depth, torch.Tensor) or not isinstance(features, torch.Tensor):
                raise ValueError("depth/features must be tensors")
            if tuple(depth.shape) != self.depth_shape or tuple(features.shape) != self.feature_shape:
                raise ValueError(f"Expected depth {self.depth_shape} and features {self.feature_shape}")
            if depth.dtype != torch.float32 or features.dtype != torch.float32:
                raise ValueError("HATLSSPool requires float32 inputs")
            if depth.device != features.device or depth.device != self.slot_mask_0.device:
                raise ValueError("Inputs and geometry buffers must share a device")
            if not torch.isfinite(depth).all() or not torch.isfinite(features).all():
                raise ValueError("Inputs must be finite")
        batch = self.depth_shape[0]
        packed_features = self._pack(features.permute(0, 2, 1, 3, 4), self._feature_pages)
        packed_depth = self._pack(depth.unsqueeze(1), self._depth_pages)
        output = None
        for k in range(self.num_points):
            contribution = (self._sample(packed_features, "feature", k)
                            * self._sample(packed_depth, "depth", k)) * getattr(self, f"slot_mask_{k}")
            output = contribution if output is None else output + contribution
        nx, ny, nz = self.grid_shape
        return output.reshape(batch, self.channels, nz, nx, ny).permute(0, 2, 1, 3, 4).reshape(self.output_shape)
