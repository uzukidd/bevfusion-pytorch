"""Legacy-exportable dynamic LiDAR operations using standard ONNX operators.

The public entry points are ``voxelize(points)`` and
``OnnxLidarEncoder(original_backbone)(features, coords)``. Export with
``torch.onnx.export(..., dynamo=False, opset_version=18)`` and mark the
leading point/voxel dimensions dynamic. No global symbolic registrations or
changes to the source model are required.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from itertools import product

import torch
from torch import nn

from bevfusion.lidar import ReferenceSparseConv, SparseState, keys


_SENTINEL = 1 << 62


class SearchSorted(torch.autograd.Function):
    """One-dimensional int64 lower_bound with an ONNX binary-search Loop."""

    @staticmethod
    def forward(ctx, sorted_keys, queries):
        return torch.searchsorted(sorted_keys, queries)

    @staticmethod
    def symbolic(g, sorted_keys, queries):
        # PyTorch moved the legacy exporter helpers in recent releases.
        try:
            from torch.onnx._internal.torchscript_exporter import jit_utils
        except ImportError:
            from torch.onnx._internal import jit_utils

        def const(value, dtype=torch.int64):
            return g.op("Constant", value_t=torch.tensor(value, dtype=dtype))

        zero, one, two = const(0), const(1), const(2)
        n = g.op("Gather", g.op("Shape", sorted_keys), zero, axis_i=0)
        padded = g.op("Concat", sorted_keys, const([_SENTINEL]), axis_i=0)
        lower = g.op("ConstantOfShape", g.op("Shape", queries),
                     value_t=torch.tensor([0], dtype=torch.int64))
        upper = g.op("Expand", n, g.op("Shape", queries))

        def any_active(context, lo, hi):
            active = context.op("Cast", context.op("Less", lo, hi), to_i=7)
            total = context.op("ReduceSum", active, keepdims_i=0)
            return context.op("Greater", total, zero)

        # 64 iterations suffice for every possible int64 tensor length. The
        # scalar condition usually terminates after ceil(log2(N + 1)) steps;
        # ReduceSum is also defined for an empty set of queries.
        outputs, (body,), _ = jit_utils.add_op_with_blocks(
            g, "Loop", const(64), any_active(g, lower, upper), lower, upper,
            outputs=2, n_blocks=1,
        )
        scalar_long = torch._C.TensorType.get().with_dtype(torch.int64).with_sizes([])
        scalar_bool = torch._C.TensorType.get().with_dtype(torch.bool).with_sizes([])
        body.block.addInputToBlock().setType(scalar_long)  # iteration
        body.block.addInputToBlock().setType(scalar_bool)  # previous condition
        lo = body.block.addInputToBlock().setType(queries.type())
        hi = body.block.addInputToBlock().setType(queries.type())
        mid = body.op("Add", lo, body.op("Div", body.op("Sub", hi, lo), two))
        value = body.op("Gather", padded, mid, axis_i=0)
        active = body.op("Less", lo, hi)
        less = body.op("Less", value, queries)
        next_lo = body.op("Where", body.op("And", active, less),
                          body.op("Add", mid, one), lo)
        next_hi = body.op("Where", body.op("And", active, body.op("Not", less)), mid, hi)
        body.block.registerOutput(any_active(body, next_lo, next_hi))
        body.block.registerOutput(next_lo)
        body.block.registerOutput(next_hi)
        outputs[0].setType(queries.type())
        return outputs[0]


def searchsorted(sorted_keys: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    return SearchSorted.apply(sorted_keys, queries)


class _DynamicSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        return torch.sort(values, dim=0)

    @staticmethod
    def symbolic(g, values):
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        axes = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
        count = g.op("Gather", g.op("Shape", values), zero, axis_i=0)
        count = g.op("Unsqueeze", count, axes)
        sorted_values, permutation = g.op(
            "TopK", values, count, axis_i=0, largest_i=0, sorted_i=1, outputs=2,
        )
        sorted_values.setType(values.type())
        permutation.setType(values.type().with_dtype(torch.int64))
        return sorted_values, permutation


def _sort_keys(values: torch.Tensor):
    # All callers use nonnegative unique keys below the sentinel. Padding
    # guarantees TopK's K is positive even for empty sparse states. There are
    # no ties, so TopK's unspecified tie order cannot change point selection.
    padded = torch.cat((values, values.new_full((1,), _SENTINEL)))
    sorted_values, permutation = _DynamicSort.apply(padded)
    return sorted_values[:-1], permutation[:-1]


def voxelize(points: torch.Tensor, max_points: int = 10, max_voxels: int = 120000,
             *, voxel_size=(0.1, 0.1, 0.2),
             point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)):
    """Mean of the first ten points in each of the first 120000 voxels.

    Input is floating point [P,5]; output is float32 [M,5] and int64 XYZ
    coordinates [M,3]. Point order, limits and quantization match
    ``bevfusion.preprocessing.voxelize``. P and M may be zero.
    """
    points = points.float()
    # Keep the traced input path alive at P=0. This point is guaranteed to be
    # rejected by the ordinary coordinate bounds mask before grouping.
    grid_shape = tuple(round((high - low) / size) for low, high, size in zip(
        point_cloud_range[:3], point_cloud_range[3:], voxel_size))
    outside = [high + size for high, size in zip(point_cloud_range[3:], voxel_size)]
    dummy = points.new_tensor([[*outside, 0.0, 0.0]])
    points = torch.cat((points, dummy), dim=0)
    size = points.new_tensor(voxel_size)
    lower = points.new_tensor(point_cloud_range[:3])
    bounds = torch.tensor(grid_shape, device=points.device, dtype=torch.int64)
    coords = torch.floor((points[:, :3] - lower) / size).long()
    keep = ((coords >= 0) & (coords < bounds)).all(dim=1)
    points, coords = points[keep], coords[keep]
    point_keys = (coords[:, 0] * grid_shape[1] + coords[:, 1]) * (grid_shape[2] + 1) + coords[:, 2]
    unique, inverse = torch.unique(point_keys, sorted=True, return_inverse=True)
    positions = torch.arange(points.shape[0], device=points.device, dtype=torch.int64)
    first = torch.zeros_like(unique) + points.shape[0]
    first = first.scatter_reduce(0, inverse, positions, reduce="amin", include_self=True)
    _, voxel_order = _sort_keys(first)
    voxel_order = voxel_order[:max_voxels]
    remap = torch.full_like(first, -1).scatter(
        0, voxel_order,
        torch.arange(voxel_order.shape[0], device=points.device, dtype=torch.int64),
    )
    voxel_id = remap[inverse]
    keep = voxel_id >= 0
    selected_points, voxel_id = points[keep], voxel_id[keep]
    positions = torch.arange(voxel_id.shape[0], device=points.device, dtype=torch.int64)
    # Unique composite keys implement a stable sort without asking the legacy
    # exporter to handle aten::sort(stable=True).
    composite = voxel_id * (points.shape[0] + 1) + positions
    _, order = _sort_keys(composite)
    sorted_id = voxel_id[order]
    starts_mask = torch.cat((torch.ones_like(sorted_id[:1], dtype=torch.bool),
                             sorted_id[1:] != sorted_id[:-1]))
    group_id = torch.cumsum(starts_mask.long(), dim=0) - 1
    starts = torch.nonzero(starts_mask, as_tuple=False).flatten()
    sorted_positions = torch.arange(order.shape[0], device=points.device, dtype=torch.int64)
    selected = order[(sorted_positions - starts[group_id]) < max_points]
    selected_id = voxel_id[selected]
    sums = points.new_zeros((voxel_order.shape[0], 5)).scatter_add(
        0, selected_id[:, None].expand(-1, 5), selected_points[selected],
    )
    counts = points.new_zeros((voxel_order.shape[0], 1)).scatter_add(
        0, selected_id[:, None], points.new_ones((selected.shape[0], 1)),
    )
    return sums / counts, coords[first[voxel_order]]


class OnnxSparseConv(ReferenceSparseConv):
    """Stateless replacement accepted by the original SparseSequence."""

    def __init__(self, original: ReferenceSparseConv):
        # Retain the original Parameter rather than allocating or reinitializing
        # a second kernel. The wrapper owns a separate module hierarchy.
        nn.Module.__init__(self)
        self.kernel = original.kernel
        self.stride = original.stride
        self.padding = original.padding
        self.subm = original.subm
        self.weight = original.weight
        self.training = original.training

    def forward(self, state):
        device = state.features.device
        stride = torch.tensor(self.stride, device=device, dtype=torch.int64)
        pad_tuple = tuple(k // 2 for k in self.kernel) if self.subm else self.padding
        padding = torch.tensor(pad_tuple, device=device, dtype=torch.int64)
        offsets = tuple(product(*(range(k) for k in self.kernel)))
        in_shape = state.shape
        if self.subm:  # Architecture metadata, never a tensor-dependent branch.
            out_coords, out_shape = state.coords, in_shape
        else:
            out_shape = tuple((s + 2 * p - k) // st + 1
                              for s, p, k, st in zip(in_shape, self.padding, self.kernel, self.stride))
            candidates = []
            bounds = torch.tensor(out_shape, device=device, dtype=torch.int64)
            for offset in offsets:
                delta = torch.tensor(offset, device=device, dtype=torch.int64)
                numerator = state.coords + padding - delta
                co = torch.div(numerator, stride, rounding_mode="floor")
                valid = (numerator.remainder(stride) == 0).all(dim=1)
                valid = valid & ((co >= 0) & (co < bounds)).all(dim=1)
                candidates.append(co[valid])
            joined = torch.cat(candidates, dim=0)
            unique_keys = torch.unique(keys(joined, out_shape), sorted=True)
            x = unique_keys // (out_shape[1] * out_shape[2])
            y = unique_keys // out_shape[2] % out_shape[1]
            z = unique_keys % out_shape[2]
            out_coords = torch.stack((x, y, z), dim=1)

        sorted_keys, permutation = _sort_keys(keys(state.coords, in_shape))
        # A lower_bound is allowed to return N. The extra slot makes every
        # gather safe, including when the input contains no active voxels.
        padded_keys = torch.cat((sorted_keys, sorted_keys.new_full((1,), _SENTINEL)))
        bounds = torch.tensor(in_shape, device=device, dtype=torch.int64)
        result = state.features.new_zeros((out_coords.shape[0], self.weight.shape[-1]))
        for offset in offsets:
            delta = torch.tensor(offset, device=device, dtype=torch.int64)
            query = out_coords * stride - padding + delta
            valid = ((query >= 0) & (query < bounds)).all(dim=1)
            query_keys = keys(query, in_shape)
            index = searchsorted(sorted_keys, query_keys)
            valid = valid & (padded_keys[index] == query_keys)
            out_index = torch.nonzero(valid, as_tuple=False).flatten()
            in_index = permutation[index[valid]]
            values = state.features[in_index] @ self.weight[offset]
            # Each out_index appears at most once for this kernel offset.
            # Preserve the reference's offset-by-offset floating point order.
            result = result.index_copy(0, out_index, result[out_index] + values)
        return SparseState(result, out_coords, out_shape)


def _adapt_module(module: nn.Module) -> nn.Module:
    if isinstance(module, ReferenceSparseConv):
        return OnnxSparseConv(module)
    adapted = copy.copy(module)
    adapted._modules = OrderedDict((name, _adapt_module(child))
                                   for name, child in module._modules.items())
    # Separate dictionaries prevent mode changes / module registration on the
    # adapter from mutating the source. Parameters and BN statistics are reused.
    adapted._parameters = module._parameters.copy()
    adapted._buffers = module._buffers.copy()
    return adapted


class OnnxLidarEncoder(nn.Module):
    """Dynamic sparse backbone adapter reusing every original weight and BN."""

    def __init__(self, original_backbone: nn.Module):
        super().__init__()
        self.conv_input = _adapt_module(original_backbone.conv_input)
        self.encoder_layers = _adapt_module(original_backbone.encoder_layers)
        self.conv_out = _adapt_module(original_backbone.conv_out)
        self.training = original_backbone.training
        self.sparse_shape = tuple(original_backbone.sparse_shape)

    def forward(self, features: torch.Tensor, coords: torch.Tensor):
        state = SparseState(features, coords.long(), self.sparse_shape)
        state = self.conv_input(state)
        state = self.encoder_layers(state)
        state = self.conv_out(state)
        flat = keys(state.coords, state.shape)
        dense = state.features.new_zeros((state.shape[0] * state.shape[1] * state.shape[2],
                                          state.features.shape[1]))
        dense = dense.index_copy(0, flat, state.features)
        dense = dense.reshape(*state.shape, state.features.shape[1]).permute(3, 2, 0, 1).contiguous()
        return dense.reshape(1, state.features.shape[1] * state.shape[2], state.shape[0], state.shape[1])


__all__ = ["SearchSorted", "searchsorted", "voxelize", "OnnxSparseConv", "OnnxLidarEncoder"]
