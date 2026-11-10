"""Fixed-geometry adapter for Horizon's native BevPoolV2 operator.

The optional horizon_plugin_pytorch package is imported only on the first
nonempty forward. Geometry buffers are fixed; depth and image features remain
fresh runtime inputs. Use the vendor HBIR export route, not standard ONNX export.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import operator

import torch
from torch import nn


def _positive_integer(value, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


class HorizonBEVPool(nn.Module):
    """Static XYZ associations and native BevPoolV2, output [B,Z*C,X,Y].

    ``operator_factory`` is a dependency-injection hook for tests. It must accept
    ``bev_feat_shape=(B,Z,X,Y,C)`` and return an operator with the vendor forward
    signature. Production callers should omit it. No standard-op fallback is
    performed when the optional vendor package is missing or incompatible.
    """

    def __init__(self, coordinates, grid_shape, channels, operator_factory=None):
        super().__init__()
        if not isinstance(coordinates, torch.Tensor):
            raise ValueError("coordinates must be an integer tensor [B,N,D,H,W,3]")
        if coordinates.ndim != 6 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [B,N,D,H,W,3]")
        if coordinates.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("coordinates must have integer dtype")
        try:
            grid_shape = tuple(grid_shape)
        except TypeError as exc:
            raise ValueError("grid_shape must contain X,Y,Z") from exc
        if len(grid_shape) != 3:
            raise ValueError("grid_shape must contain X,Y,Z")
        self.grid_shape = tuple(_positive_integer(value, "grid dimension") for value in grid_shape)
        self.channels = _positive_integer(channels, "channels")
        self.depth_shape = tuple(int(value) for value in coordinates.shape[:-1])
        if any(value < 1 for value in self.depth_shape):
            raise ValueError("B,N,D,H,W must all be positive; empty geometry uses invalid coordinates")
        batch, cameras, depths, height, width = self.depth_shape
        nx, ny, nz = self.grid_shape
        self.feature_shape = (batch, cameras, self.channels, height, width)
        self.output_shape = (batch, nz * self.channels, nx, ny)
        self.bev_feat_shape = (batch, nz, nx, ny, self.channels)
        int32_max = torch.iinfo(torch.int32).max
        if max(math.prod(self.depth_shape), batch * cameras * height * width,
               batch * nz * nx * ny) > int32_max:
            raise ValueError("Depth, feature or BEV ranks exceed INT32 capacity")
        if operator_factory is not None and not callable(operator_factory):
            raise ValueError("operator_factory must be callable")
        self._operator_factory = operator_factory
        # Keep the optional execution object separate from geometry/state_dict.
        # BevPoolV2 has no learned weights; no eager plugin import is needed.
        object.__setattr__(self, "_operator", None)

        device = coordinates.device
        coords = coordinates.detach().to(device="cpu", dtype=torch.int64).contiguous()
        flattened = coords.reshape(-1, 3)
        bounds = flattened.new_tensor(self.grid_shape)
        valid = ((flattened >= 0) & (flattened < bounds)).all(dim=1)
        ranks_depth = torch.arange(math.prod(self.depth_shape), dtype=torch.int64)[valid]
        xyz = flattened[valid]
        pixels = height * width
        camera_ids = torch.div(ranks_depth, depths * pixels, rounding_mode="floor")
        ranks_feat = camera_ids * pixels + ranks_depth.remainder(pixels)
        batch_ids = torch.div(camera_ids, cameras, rounding_mode="floor")
        ranks_bev = ((batch_ids * nz + xyz[:, 2]) * nx + xyz[:, 0]) * ny + xyz[:, 1]
        order = torch.argsort(ranks_bev, stable=True)
        ranks_depth, ranks_feat, ranks_bev = ranks_depth[order], ranks_feat[order], ranks_bev[order]
        if ranks_bev.numel():
            changed = torch.ones_like(ranks_bev, dtype=torch.bool)
            changed[1:] = ranks_bev[1:] != ranks_bev[:-1]
            interval_starts = torch.nonzero(changed, as_tuple=False).flatten()
            interval_lengths = torch.diff(torch.cat((interval_starts,
                                                     interval_starts.new_tensor([ranks_bev.numel()]))))
        else:
            interval_starts = ranks_bev.new_empty((0,))
            interval_lengths = ranks_bev.new_empty((0,))
        for name, value in (("ranks_depth", ranks_depth), ("ranks_feat", ranks_feat),
                            ("ranks_bev", ranks_bev), ("interval_starts", interval_starts),
                            ("interval_lengths", interval_lengths)):
            self.register_buffer(name, value.to(device=device, dtype=torch.int32).contiguous())
        self.valid_points = int(ranks_bev.numel())
        self.interval_count = int(interval_starts.numel())
        self.max_rank = int(ranks_bev.max()) if self.valid_points else -1
        self.geometry_sha256 = hashlib.sha256(
            str((self.depth_shape, self.grid_shape)).encode("ascii")
            + coords.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
        self.eval()

    @classmethod
    def from_transform(cls, transform, camera2lidar, camera_intrinsics, img_aug_matrix,
                       lidar_aug_matrix=None, *, operator_factory=None):
        """Build associations using the original transform's exact voxel indices."""
        device = transform.frustum.device

        def camera_matrix(value, name, allowed):
            value = torch.as_tensor(value, device=device, dtype=torch.float32)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if (value.ndim != 4 or tuple(value.shape[-2:]) not in allowed
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid {name} camera matrices")
            return value

        camera2lidar = camera_matrix(camera2lidar, "camera2lidar", ((4, 4),))
        camera_intrinsics = camera_matrix(camera_intrinsics, "camera_intrinsics", ((3, 3), (4, 4)))
        img_aug_matrix = camera_matrix(img_aug_matrix, "img_aug_matrix", ((4, 4),))
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
        return cls(coordinates, tuple(int(value) for value in transform.nx), transform.C,
                   operator_factory=operator_factory)

    def statistics(self):
        return {"backend": "horizon", "operator": "BevPoolV2", "groups": 1, "chunks": 1,
                "valid_points": self.valid_points, "all_points": math.prod(self.depth_shape),
                "intervals": self.interval_count, "active_cells": self.interval_count,
                "max_rank": self.max_rank, "grid_shape": list(self.grid_shape),
                "depth_shape": list(self.depth_shape), "feature_shape": list(self.feature_shape),
                "output_shape": list(self.output_shape), "vendor_output_shape": [self.depth_shape[0],
                    self.channels, self.grid_shape[2], self.grid_shape[0], self.grid_shape[1]],
                "bev_feat_shape": list(self.bev_feat_shape), "geometry_sha256": self.geometry_sha256,
                "index_dtype": "int32", "input_dtype": "float32", "output_dtype": "float32",
                "fixed_geometry": True,
                "index_bytes": sum(buffer.numel() * buffer.element_size() for buffer in self.buffers())}

    def _get_operator(self):
        operation = self._operator
        if operation is None:
            try:
                factory = self._operator_factory
                if factory is None:
                    factory = importlib.import_module("horizon_plugin_pytorch.nn").BevPoolV2
                operation = factory(bev_feat_shape=self.bev_feat_shape)
                if not callable(operation):
                    raise TypeError("BevPoolV2 factory returned a non-callable object")
                if isinstance(operation, nn.Module):
                    operation.eval()
            except Exception as exc:
                # Optional plugin imports may themselves assert ABI/version
                # compatibility; retain that original cause rather than fallback.
                raise RuntimeError(
                    "horizon_plugin_pytorch missing/incompatible: native BevPoolV2 is required; "
                    "run this backend with a compatible installed Horizon wheel and PyTorch version. "
                    "No fallback pooling was executed.") from exc
            object.__setattr__(self, "_operator", operation)
        return operation

    def forward(self, depth, features):
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError("HorizonBEVPool cannot use standard ONNX export; use the vendor HBIR export route")
        tracing = torch.jit.is_tracing()
        if not tracing:
            if not isinstance(depth, torch.Tensor) or not isinstance(features, torch.Tensor):
                raise ValueError("depth/features must be tensors")
            if tuple(depth.shape) != self.depth_shape or tuple(features.shape) != self.feature_shape:
                raise ValueError(f"Expected depth {self.depth_shape}, features {self.feature_shape}")
            if depth.dtype != torch.float32 or features.dtype != torch.float32:
                raise ValueError("HorizonBEVPool requires float32 depth and features")
            if depth.device != features.device or depth.device != self.ranks_depth.device:
                raise ValueError("depth/features and geometry buffers must share a device")
            if not torch.isfinite(depth).all() or not torch.isfinite(features).all():
                raise ValueError("depth/features must be finite")
        if not self.valid_points:
            # The vendor kernel cannot process an empty interval table. Preserve
            # both input dependencies for tracing while avoiding a full reduction.
            zero = depth.reshape(-1)[:1].sum() * 0.0 + features.reshape(-1)[:1].sum() * 0.0
            return depth.new_zeros(self.output_shape) + zero
        result = self._get_operator()(
            depth.contiguous(), features.contiguous(), self.ranks_depth, self.ranks_feat,
            self.ranks_bev, self.interval_starts, self.interval_lengths,
            bev_feat_shape=self.bev_feat_shape)
        if not tracing:
            expected = (self.depth_shape[0], self.channels, self.grid_shape[2],
                        self.grid_shape[0], self.grid_shape[1])
            if (not isinstance(result, torch.Tensor) or tuple(result.shape) != expected
                    or result.dtype != torch.float32 or result.device != depth.device):
                raise RuntimeError("Incompatible BevPoolV2 output; expected FP32 [B,C,Z,X,Y] on the input device")
        return result.permute(0, 2, 1, 3, 4).reshape(self.output_shape)
