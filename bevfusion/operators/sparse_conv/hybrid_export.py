"""Feature-only export of one explicitly fixed hybrid sparse topology.

Construct on the host for each coordinate topology. INT32 sentinel lookups,
submanifold neighbours, masks and tile extraction maps become graph constants;
forward contains only tensor arithmetic, Gather and dense Conv3D. This adapter
is a compilation probe, not a graph supporting new coordinates at runtime.
"""
from __future__ import annotations

import copy
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .compatible import CompatibleSparseConv, _exporting, _validate_tensor
from .conv3d import dense_conv3d_patch, ordinary_output_coordinates
from .plan import EncoderPlan, _coordinates
from .reference import ReferenceSparseConv, SparseBlock, SparseSequence


def _gather_rows(features, indices, capacity, page_size):
    """Sentinel Gather using native INT32 index_select, without autograd ops."""
    result = None
    for start in range(0, capacity, page_size):
        stop = min(start + page_size, capacity)
        source = torch.cat((features.new_zeros((1, features.shape[1])), features[start:stop]), dim=0)
        selected = (indices > start) & (indices <= stop)
        local = torch.where(selected, indices - start, indices.new_zeros(()))
        part = torch.index_select(source, 0, local)
        result = part if result is None else result + part
    return result


class _FixedSubmanifold(CompatibleSparseConv):
    """CompatibleSparseConv math with native Gather and constant CPU topology."""

    def __init__(self, original, plan, neighbors, mask, page_size):
        super().__init__(original, plan.input_capacity, plan.output_capacity, page_size)
        self.register_buffer("neighbor_indices", neighbors, persistent=False)
        self.register_buffer("output_mask", mask, persistent=False)

    def _compute(self, features, neighbor_indices, output_mask):
        # Same kernel/row/page order as CompatibleSparseConv._compute. Native
        # index_select preserves INT32 indices in the HBIR trace, unlike the
        # ONNX-specific _GatherInt32 autograd function's eager INT64 conversion.
        tiles = []
        for begin in range(0, self.output_capacity, 32768):
            end = min(begin + 32768, self.output_capacity)
            result = None
            for index, offset in enumerate(self.offsets):
                indices = neighbor_indices[index:index + 1, begin:end].reshape(end - begin)
                gathered = _gather_rows(features, indices, self.input_capacity, self.page_size)
                contribution = gathered @ self.weight[offset]
                result = contribution if result is None else result + contribution
            tiles.append(result * output_mask[begin:end])
        return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)

    def forward(self, features):
        return self._compute(features, self.neighbor_indices, self.output_mask)


class _DenseTile(nn.Module):
    def __init__(self, patch_lookup, active_lookup, patch_shape, output_shape, device):
        super().__init__()
        self.patch_shape, self.output_shape = tuple(patch_shape), tuple(output_shape)
        self.active_count = len(active_lookup)
        # HBDK 4.7.5 gather2d codegen fails for short outputs such as
        # [45056,64] -> [8,64], even padded to 32 rows. Gathering at least
        # 64 rows then slicing compiles. Added row-zero duplicates are discarded
        # before BatchNorm or sparse concatenation.
        if self.active_count < 64:
            active_lookup = np.pad(active_lookup, (0, 64 - self.active_count), constant_values=0)
        self.register_buffer("patch_lookup", torch.from_numpy(patch_lookup).to(device=device), persistent=False)
        self.register_buffer("active_lookup", torch.from_numpy(active_lookup).to(device=device), persistent=False)

    def forward(self, features, weight, stride, padding, capacity, page_size):
        rows = _gather_rows(features, self.patch_lookup, capacity, page_size)
        patch = rows.reshape(*self.patch_shape, weight.shape[-2]).permute(3, 0, 1, 2).unsqueeze(0)
        dense = dense_conv3d_patch(patch, weight, stride, padding)
        rows = dense.permute(0, 2, 3, 4, 1).reshape(math.prod(self.output_shape), weight.shape[-1])
        return torch.index_select(rows, 0, self.active_lookup)[:self.active_count]


class _FixedConv3d(nn.Module):
    def __init__(self, original, plan, input_coords, input_shape, tile_size, page_size, mask):
        super().__init__()
        self.weight = original.weight
        self.kernel, self.stride, self.padding = original.kernel, original.stride, original.padding
        self.input_capacity, self.output_capacity = int(plan.input_capacity), int(plan.output_capacity)
        self.in_channels, self.out_channels = original.weight.shape[-2:]
        self.page_size, self.output_count = page_size, int(plan.output_count)
        self.register_buffer("output_mask", mask, persistent=False)
        self.tiles = nn.ModuleList()
        coords = input_coords
        output = plan.output_coords.detach().cpu().numpy().astype(np.int64, copy=False)
        output_shape = tuple(plan.output_shape)
        for tile_id in np.unique(output[:, 0] // tile_size):
            start = int(tile_id) * tile_size
            stop = min(start + tile_size, output_shape[0])
            origin = start * self.stride[0] - self.padding[0]
            width = (stop - start - 1) * self.stride[0] + self.kernel[0]
            patch_shape = (width, input_shape[1], input_shape[2])
            patch_lookup = np.zeros(math.prod(patch_shape), dtype=np.int32)
            selected = np.flatnonzero((coords[:, 0] >= origin) & (coords[:, 0] < origin + width))
            local = coords[selected]
            linear = ((local[:, 0] - origin) * input_shape[1] + local[:, 1]) * input_shape[2] + local[:, 2]
            patch_lookup[linear] = (selected + 1).astype(np.int32)
            active = output[(output[:, 0] >= start) & (output[:, 0] < stop)]
            local_shape = (stop - start, output_shape[1], output_shape[2])
            if math.prod(local_shape) > np.iinfo(np.int32).max:
                raise ValueError("Conv3D tile extraction exceeds INT32 lookup capacity")
            active_lookup = (((active[:, 0] - start) * output_shape[1] + active[:, 1])
                             * output_shape[2] + active[:, 2]).astype(np.int32)
            self.tiles.append(_DenseTile(patch_lookup, active_lookup, patch_shape, local_shape, self.weight.device))

    def forward(self, features):
        pieces = [tile(features, self.weight, self.stride, (0, self.padding[1], self.padding[2]),
                       self.input_capacity, self.page_size) for tile in self.tiles]
        if pieces:
            result = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        else:
            result = features.new_zeros((0, self.out_channels))
        if self.output_capacity > self.output_count:
            result = torch.cat((result, features.new_zeros((self.output_capacity - self.output_count,
                                                          self.out_channels))), dim=0)
        return result * self.output_mask


class HybridExportEncoder(nn.Module):
    """Share source weights/BN while exporting one frame's fixed hybrid topology.

    ``features`` is padded FP32 [encoder_plan.input_capacity, Cin], with active
    rows in exactly ``input_coords`` order. Padding values are ignored. Rebuild
    this wrapper and re-export for different coordinates or row order. Call
    ``validate_topology(coords)`` on the host when reusing a compiled artifact.
    No coordinate tensor, neighbour discovery or sparse CUDA call is traced.
    Source parameter/checkpoint names are retained; topology buffers are omitted
    from state_dict because a checkpoint alone does not describe this graph.
    """

    fixed_topology = True

    def __init__(self, original, encoder_plan: EncoderPlan, input_coords, tile_size=16, page_size=32768):
        super().__init__()
        if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 65535:
            raise ValueError("page_size must be in [1,65535]")
        if not isinstance(encoder_plan, EncoderPlan):
            raise TypeError("encoder_plan must be EncoderPlan")
        self.sparse_shape = tuple(original.sparse_shape)
        initial_coords = _coordinates(input_coords, self.sparse_shape)
        if len(initial_coords) != encoder_plan.input_count:
            raise ValueError("input_coords count differs from encoder_plan fixed topology")
        self._input_coords = torch.from_numpy(initial_coords.copy()).int()
        self.input_capacity, self.input_count = int(encoder_plan.input_capacity), len(initial_coords)
        if not 1 <= self.input_capacity < 131072 or self.input_count > self.input_capacity:
            raise ValueError("input_capacity must cover active rows and be in [1,131071]")
        self.tile_size, self.page_size = tile_size, page_size
        self.input_names = ["features"]
        self.final_shape = tuple(int(v) for v in encoder_plan.final_shape)
        if any(v <= 0 for v in self.final_shape):
            raise ValueError("Final dense shape must have positive dimensions")
        self.volume = math.prod(self.final_shape)
        convs = {name: module for name, module in original.named_modules() if isinstance(module, ReferenceSparseConv)}
        if set(convs) != set(encoder_plan.plans):
            raise ValueError("Encoder plan must provide every original convolution by name")
        self.stats = {"subm": 0, "conv3d": 0, "tile_count": 0, "patch_lookup_elements": 0,
                      "input_count": self.input_count, "input_capacity": self.input_capacity,
                      "fixed_topology": True}
        shared = {}

        def adapt(module, name, coords, shape, capacity):
            if isinstance(module, ReferenceSparseConv):
                plan = encoder_plan.plans[name]
                if plan.input_count != len(coords) or plan.input_capacity != capacity:
                    raise ValueError(f"Encoder plan input topology/capacity differs at {name}")
                if not 1 <= plan.output_capacity < 131072 or not 0 <= plan.output_count <= plan.output_capacity:
                    raise ValueError(f"Invalid output capacity/count at {name}")
                expected_shape = shape if module.subm else tuple(
                    (s + 2*p - k)//st + 1 for s, p, k, st in zip(shape, module.padding, module.kernel, module.stride))
                if tuple(plan.output_shape) != expected_shape:
                    raise ValueError(f"Encoder plan output shape differs at {name}")
                next_coords = _coordinates(plan.output_coords, expected_shape)
                if len(next_coords) != plan.output_count:
                    raise ValueError(f"Encoder plan output count differs at {name}")
                if module.subm:
                    if not np.array_equal(coords, next_coords):
                        raise ValueError(f"input_coords differs from the encoder_plan fixed topology at {name}")
                else:
                    expected_coords, _ = ordinary_output_coordinates(torch.from_numpy(coords), shape,
                        module.kernel, module.stride, module.padding)
                    if not np.array_equal(expected_coords.numpy(), next_coords):
                        raise ValueError(f"Encoder plan ordinary topology differs at {name}")
                _validate_tensor(plan.output_mask, (plan.output_capacity, 1), torch.float32, f"{name}.output_mask")
                if (not torch.equal(plan.output_mask[:plan.output_count], torch.ones_like(plan.output_mask[:plan.output_count]))
                        or torch.count_nonzero(plan.output_mask[plan.output_count:]).item()):
                    raise ValueError(f"Encoder plan mask differs from active rows at {name}")
                identity = (id(plan), module.weight.device)
                if identity not in shared:
                    shared[identity] = (plan.neighbor_indices.detach().clone().to(module.weight.device),
                                        plan.output_mask.detach().clone().to(module.weight.device))
                neighbors, mask = shared[identity]
                if module.subm:
                    _validate_tensor(neighbors, (math.prod(module.kernel), plan.output_capacity), torch.int32,
                                     f"{name}.neighbor_indices")
                    if ((neighbors < 0) | (neighbors > plan.input_count)).any().item():
                        raise ValueError(f"Encoder plan neighbors exceed active input rows at {name}")
                    result = _FixedSubmanifold(module, plan, neighbors, mask, page_size)
                    self.stats["subm"] += 1
                else:
                    result = _FixedConv3d(module, plan, coords, shape, tile_size, page_size, mask)
                    self.stats["conv3d"] += 1
                    self.stats["tile_count"] += len(result.tiles)
                    self.stats["patch_lookup_elements"] += sum(tile.patch_lookup.numel() for tile in result.tiles)
                return result, next_coords, expected_shape, int(plan.output_capacity)
            if not isinstance(module, (SparseBlock, SparseSequence, nn.BatchNorm1d, nn.ReLU, nn.Identity)):
                raise ValueError(f"Unsupported export module {name}: {type(module).__name__}")
            result = copy.copy(module)
            result._parameters, result._buffers = module._parameters.copy(), module._buffers.copy()
            result._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()
            result._modules = OrderedDict()
            residual = (coords, shape, capacity)
            for child_name, child in module.named_children():
                adapted, coords, shape, capacity = adapt(child, f"{name}.{child_name}", coords, shape, capacity)
                result.add_module(child_name, adapted)
            if isinstance(module, SparseBlock) and (shape != residual[1] or capacity != residual[2]
                                                   or not np.array_equal(coords, residual[0])):
                raise ValueError(f"Residual block {name} must preserve topology and capacity")
            return result, coords, shape, capacity

        coords, shape, capacity = initial_coords, self.sparse_shape, self.input_capacity
        for name in ("conv_input", "encoder_layers", "conv_out"):
            module, coords, shape, capacity = adapt(getattr(original, name), name, coords, shape, capacity)
            setattr(self, name, module)
        if shape != self.final_shape or not np.array_equal(coords, encoder_plan.final_coords.cpu().numpy()):
            raise ValueError("Encoder plan final topology differs")
        self.final_capacity = capacity
        final_conv = next(reversed(convs.values()))
        self.in_channels = next(iter(convs.values())).weight.shape[-2]
        self.out_channels = final_conv.weight.shape[-1]
        _validate_tensor(encoder_plan.dense_lookup, (self.volume,), torch.int32, "dense_lookup")
        expected_lookup = np.zeros(self.volume, dtype=np.int32)
        if len(coords):
            flat = (coords[:, 0] * shape[1] + coords[:, 1]) * shape[2] + coords[:, 2]
            expected_lookup[flat] = np.arange(1, len(coords) + 1, dtype=np.int32)
        if not np.array_equal(encoder_plan.dense_lookup.cpu().numpy(), expected_lookup):
            raise ValueError("Encoder plan dense_lookup differs from fixed final topology")
        self.register_buffer("dense_lookup", encoder_plan.dense_lookup.detach().clone().to(final_conv.weight.device),
                             persistent=False)
        self.eval()

    def validate_topology(self, coords):
        """Host-only exact row-order check; never called by the export graph."""
        actual = torch.from_numpy(_coordinates(coords, self.sparse_shape)).int()
        if not torch.equal(actual, self._input_coords):
            raise ValueError("This artifact has fixed topology; rebuild and re-export for changed coordinates/row order")

    def _execute(self, module, features, mask):
        if isinstance(module, (_FixedSubmanifold, _FixedConv3d)):
            return module(features), module.output_mask
        if isinstance(module, SparseBlock):
            residual = features
            x, current = self._execute(module.conv1, features, mask)
            x = F.relu(module.bn1(x)) * current
            x, current = self._execute(module.conv2, x, current)
            return F.relu(module.bn2(x) + residual) * current, current
        if isinstance(module, SparseSequence):
            for child in module:
                features, mask = self._execute(child, features, mask)
            return features, mask
        return module(features) * mask, mask

    def forward(self, features):
        if not _exporting():
            if self.training:
                raise RuntimeError("HybridExportEncoder is inference-only; call eval()")
            _validate_tensor(features, (self.input_capacity, self.in_channels), torch.float32, "features")
            if features.device != self.dense_lookup.device or self.conv_input[0].weight.dtype != torch.float32:
                raise ValueError("Features, mapping buffers and FP32 weights must share device")
        x, mask = self._execute(self.conv_input, features, None)
        x, mask = self._execute(self.encoder_layers, x, mask)
        x, mask = self._execute(self.conv_out, x, mask)
        dense = _gather_rows(x, self.dense_lookup, self.final_capacity, self.page_size)
        sx, sy, sz = self.final_shape
        return dense.reshape(sx, sy, sz, self.out_channels).permute(3, 2, 0, 1).reshape(
            1, self.out_channels * sz, sx, sy)


__all__ = ["HybridExportEncoder"]
