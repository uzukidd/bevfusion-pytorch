"""Inference sparse convolution with CPU-planned, runtime INT32 neighbours.

The graph contains bounded Gather, MatMul and Add, not coordinate discovery.
CPU plans must be rebuilt for each changed point cloud and supplied as inputs.
This graph is a Nash-P compilation candidate, not verified BPU deployment.
"""
from __future__ import annotations

import copy
import math
import operator
from collections import OrderedDict
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from .compatible_pooling import _GatherInt32, _concat
from .lidar import ReferenceSparseConv, SparseBlock, SparseSequence


def _capacity(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not 1 <= value < 131072:
        raise ValueError(f"{name} must be in [1,131071]; capacity is never truncated")
    return value


def _validate_tensor(value, shape, dtype, name, device=None):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise ValueError(f"{name} must have shape {tuple(shape)} and dtype {dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}")


def _exporting():
    return torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()


def _paged_gather(features, indices, input_capacity, page_size):
    """Global sentinel=0, real row i encoded by i+1, page-local sentinel=0."""
    gathered = None
    for start in range(0, input_capacity, page_size):
        end = min(start + page_size, input_capacity)
        page = torch.cat((features.new_zeros((1, features.shape[1])), features[start:end]), dim=0)
        selected = (indices > start) & (indices <= end)
        local = torch.where(selected, indices - start, indices.new_zeros(()))
        part = _GatherInt32.apply(page, local)
        gathered = part if gathered is None else gathered + part
    return gathered


class CompatibleSparseConv(nn.Module):
    """Reuse [kx,ky,kz,in,out] parameters with runtime CPU-planned neighbours.

    Input features are padded FP32 [input_capacity,Cin]. Neighbours are INT32
    [kernel_volume,output_capacity], sentinel zero / input row+1. The FP32
    [output_capacity,1] mask zeros padding after the convolution. At most 32768
    output rows are gathered per tile, and each source page includes a zero row.
    """
    def __init__(self, original: ReferenceSparseConv, input_capacity, output_capacity, page_size=32768):
        super().__init__()
        if not isinstance(original, ReferenceSparseConv):
            raise TypeError("original must be ReferenceSparseConv")
        self.input_capacity = _capacity(input_capacity, "input_capacity")
        self.output_capacity = _capacity(output_capacity, "output_capacity")
        self.page_size = _capacity(page_size, "page_size")
        if self.page_size > 65535:
            raise ValueError("page_size must be <=65535 including an extra sentinel source row")
        self.kernel, self.stride, self.padding, self.subm = original.kernel, original.stride, original.padding, original.subm
        self.weight = original.weight
        self.offsets = tuple(product(*(range(k) for k in self.kernel)))
        self.in_channels, self.out_channels = original.weight.shape[-2:]
        if any(dimension >= 131072 for dimension in original.weight.shape):
            raise ValueError("Sparse convolution weight dimensions must be <131072")
        self.eval()

    def forward(self, features, neighbor_indices, output_mask):
        if not _exporting():
            if self.training:
                raise RuntimeError("CompatibleSparseConv is inference-only; call eval()")
            _validate_tensor(features, (self.input_capacity, self.in_channels), torch.float32, "features")
            _validate_tensor(neighbor_indices, (len(self.offsets), self.output_capacity), torch.int32,
                             "neighbor_indices", features.device)
            _validate_tensor(output_mask, (self.output_capacity, 1), torch.float32, "output_mask", features.device)
            if self.weight.device != features.device or self.weight.dtype != torch.float32:
                raise ValueError("weights and features must share FP32 dtype/device")
            if ((neighbor_indices < 0) | (neighbor_indices > self.input_capacity)).any():
                raise ValueError("neighbor indices must be sentinel 0 or input row+1 within capacity")
        return self._compute(features, neighbor_indices, output_mask)

    def _compute(self, features, neighbor_indices, output_mask):
        # Encoder validates all inputs together, avoiding a device synchronization
        # for each of its 21 convolutions. Standalone forward still checks bounds.
        tiles = []
        for begin in range(0, self.output_capacity, 32768):
            end = min(begin + 32768, self.output_capacity)
            result = None
            for index, offset in enumerate(self.offsets):
                indices = neighbor_indices[index:index + 1, begin:end].reshape(end - begin)
                gathered = _paged_gather(features, indices, self.input_capacity, self.page_size)
                contribution = gathered @ self.weight[offset]
                result = contribution if result is None else result + contribution
            tiles.append(result * output_mask[begin:end])
        return _concat(tiles)


class CompatibleSparseEncoder(nn.Module):
    """Reference sparse encoder with fixed capacities and dynamic neighbour inputs.

    Build once from a capacity-compatible CPU EncoderPlan. The construction plan
    provides shapes/capacities/alias layout ONLY: no coordinates or neighbours
    are retained as buffers. ``plan_inputs(new_plan, device)`` packs each frame's
    CPU adjacency and mask tensors, followed by final dense lookup. The same
    object-alias layout and capacities must be maintained across frames.
    """
    def __init__(self, original, encoder_plan, page_size=32768):
        super().__init__()
        self.page_size = _capacity(page_size, "page_size")
        if self.page_size > 65535:
            raise ValueError("page_size must be <=65535")
        self.input_capacity = _capacity(encoder_plan.input_capacity, "input_capacity")
        self.sparse_shape = tuple(original.sparse_shape)
        self.final_shape = tuple(int(v) for v in encoder_plan.final_shape)
        self.volume = math.prod(self.final_shape)
        if any(v < 1 for v in self.final_shape) or self.volume >= 131072:
            raise ValueError("Final dense volume must be positive and <131072")
        self._slot_by_name, self._slot_metadata, aliases = {}, [], {}
        convs = dict((name, module) for name, module in original.named_modules()
                     if isinstance(module, ReferenceSparseConv))
        if set(convs) != set(encoder_plan.plans):
            raise ValueError("Encoder plan must provide every original convolution by name")
        for name in convs:
            plan = encoder_plan.plans[name]
            identity = id(plan)
            if identity not in aliases:
                aliases[identity] = len(self._slot_metadata)
                self._slot_metadata.append((int(plan.input_capacity), int(plan.output_capacity),
                                            tuple(plan.output_shape), math.prod(convs[name].kernel)))
            slot = aliases[identity]
            metadata = (int(plan.input_capacity), int(plan.output_capacity), tuple(plan.output_shape), math.prod(convs[name].kernel))
            if metadata != self._slot_metadata[slot]:
                raise ValueError("Aliased convolution plans have different metadata")
            self._slot_by_name[name] = slot
        def adapt(module, prefix):
            if isinstance(module, ReferenceSparseConv):
                plan = encoder_plan.plans[prefix]
                return CompatibleSparseConv(module, plan.input_capacity, plan.output_capacity, self.page_size)
            result = copy.copy(module)
            result._parameters, result._buffers = module._parameters.copy(), module._buffers.copy()
            result._modules = OrderedDict((name, adapt(child, f"{prefix}.{name}" if prefix else name))
                                          for name, child in module._modules.items())
            return result
        self.conv_input = adapt(original.conv_input, "conv_input")
        self.encoder_layers = adapt(original.encoder_layers, "encoder_layers")
        self.conv_out = adapt(original.conv_out, "conv_out")
        self.final_capacity = int(encoder_plan.plans["conv_out.0"].output_capacity)
        self.input_names = ["features"]
        for slot in range(len(self._slot_metadata)):
            self.input_names.extend((f"neighbors_{slot}", f"mask_{slot}"))
        self.input_names.append("dense_lookup")
        if len(self.input_names) >= 256:
            raise ValueError("Runtime sparse-plan inputs must remain below 256")
        self.eval()

    def plan_inputs(self, plan, device=None):
        if int(plan.input_capacity) != self.input_capacity or tuple(plan.final_shape) != self.final_shape:
            raise ValueError("Runtime encoder plan capacity/final shape differs from the constructed graph")
        if set(plan.plans) != set(self._slot_by_name):
            raise ValueError("Runtime encoder plan convolution names differ")
        packed, representatives = [None] * len(self._slot_metadata), {}
        for name, slot in self._slot_by_name.items():
            item = plan.plans[name]
            meta = (int(item.input_capacity), int(item.output_capacity), tuple(item.output_shape), item.neighbor_indices.shape[0])
            if meta != self._slot_metadata[slot]:
                raise ValueError(f"Runtime plan metadata changed for {name}; rebuild the encoder wrapper")
            if slot in representatives:
                previous = representatives[slot]
                if item is not previous and (not torch.equal(item.neighbor_indices, previous.neighbor_indices)
                                             or not torch.equal(item.output_mask, previous.output_mask)):
                    raise ValueError(f"Runtime plans previously sharing inputs diverged: {name}")
            else:
                representatives[slot] = item
                packed[slot] = (item.neighbor_indices.to(device=device), item.output_mask.to(device=device))
        lookup = plan.dense_lookup
        _validate_tensor(lookup, (self.volume,), torch.int32, "dense_lookup")
        return tuple(tensor for pair in packed for tensor in pair) + (lookup.to(device=device),)

    def _execute(self, module, name, features, mask, tensors):
        if isinstance(module, CompatibleSparseConv):
            slot = self._slot_by_name[name]
            neighbors, mask = tensors[2 * slot:2 * slot + 2]
            return module._compute(features, neighbors, mask), mask
        if isinstance(module, SparseBlock):
            residual = features
            x, current = self._execute(module.conv1, name + ".conv1", features, mask, tensors)
            x = F.relu(module.bn1(x)) * current
            x, current = self._execute(module.conv2, name + ".conv2", x, current, tensors)
            if x.shape != residual.shape:
                raise ValueError("Residual sparse block capacities must match")
            return F.relu(module.bn2(x) + residual) * current, current
        if isinstance(module, (SparseSequence, nn.Sequential)):
            for child_name, child in module.named_children():
                features, mask = self._execute(child, name + "." + child_name, features, mask, tensors)
            return features, mask
        return module(features) * mask, mask

    def forward(self, features, *runtime_plan_tensors):
        if not _exporting():
            if self.training:
                raise RuntimeError("CompatibleSparseEncoder is inference-only; call eval()")
            _validate_tensor(features, (self.input_capacity, self.conv_input[0].in_channels), torch.float32, "features")
            if len(runtime_plan_tensors) != len(self.input_names) - 1:
                raise ValueError("Incorrect number of runtime sparse-plan tensors")
            invalid = torch.zeros((), dtype=torch.bool, device=features.device)
            for slot, (input_capacity, output_capacity, _, kernel_volume) in enumerate(self._slot_metadata):
                neighbors, mask = runtime_plan_tensors[2 * slot:2 * slot + 2]
                _validate_tensor(neighbors, (kernel_volume, output_capacity), torch.int32,
                                 f"neighbors_{slot}", features.device)
                _validate_tensor(mask, (output_capacity, 1), torch.float32, f"mask_{slot}", features.device)
                invalid = invalid | ((neighbors < 0) | (neighbors > input_capacity)).any()
            lookup = runtime_plan_tensors[-1]
            _validate_tensor(lookup, (self.volume,), torch.int32, "dense_lookup", features.device)
            invalid = invalid | ((lookup < 0) | (lookup > self.final_capacity)).any()
            if invalid.item():
                raise ValueError("Sparse neighbours/dense_lookup exceed their input capacities")
            if self.conv_input[0].weight.device != features.device or self.conv_input[0].weight.dtype != torch.float32:
                raise ValueError("Encoder weights and features must share FP32 dtype/device")
        x, mask = self._execute(self.conv_input, "conv_input", features, None, runtime_plan_tensors)
        x, mask = self._execute(self.encoder_layers, "encoder_layers", x, mask, runtime_plan_tensors)
        x, mask = self._execute(self.conv_out, "conv_out", x, mask, runtime_plan_tensors)
        lookup = runtime_plan_tensors[-1]
        tiles = [_paged_gather(x, lookup[start:start + 32768], self.final_capacity, self.page_size)
                 for start in range(0, self.volume, 32768)]
        dense = _concat(tiles)
        sx, sy, sz = self.final_shape
        return dense.reshape(sx, sy, sz, self.conv_out[0].out_channels).permute(3, 2, 0, 1).reshape(
            1, self.conv_out[0].out_channels * sz, sx, sy)


__all__ = ["CompatibleSparseConv", "CompatibleSparseEncoder"]
