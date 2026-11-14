"""Per-frame CPU topology planning for tensor-only sparse convolution inference.

Only geometry is planned here: no features, parameters or device kernels run.
Zero is the missing-neighbor sentinel; actual input rows are addressed by row+1.
Plans never truncate active sites or neighbors to fit a caller's fixed capacity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import math
import operator

import numpy as np
import torch

_INT32_MAX = np.iinfo(np.int32).max
_INT64_MAX = np.iinfo(np.int64).max


@dataclass(frozen=True)
class SparseConvPlan:
    input_count: int
    output_count: int
    input_capacity: int
    output_capacity: int
    output_shape: tuple[int, int, int]
    output_coords: torch.Tensor
    neighbor_indices: torch.Tensor
    output_mask: torch.Tensor


@dataclass(frozen=True)
class EncoderPlan:
    """Plans in execution order, keyed by original backbone named_modules names.

    dense_lookup flattens XYZ with Z varying fastest. Gathering from a zero row
    followed by final sparse features gives [X*Y*Z,C], then reshape [X,Y,Z,C],
    permute to [C,Z,X,Y], and combine C*Z for the source encoder's dense layout.
    """
    plans: dict[str, SparseConvPlan]
    input_count: int
    input_capacity: int
    final_coords: torch.Tensor
    final_shape: tuple[int, int, int]
    dense_lookup: torch.Tensor

    @property
    def output_count(self):
        return len(self.final_coords)

    @property
    def final_capacity(self):
        return next(reversed(self.plans.values())).output_capacity if self.plans else self.input_capacity


# Alias makes the return type discoverable without imposing an adapter class name.
SparseEncoderPlan = EncoderPlan


def _integer(value, name, minimum=0):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if isinstance(value, (bool, np.bool_)) or result < minimum or result > _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    return result


def _triple(value, name, minimum=0):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        value = (value,) * 3
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer or three integers") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain three integers")
    return tuple(_integer(v, name, minimum) for v in values)


def _shape(value):
    shape = _triple(value, "sparse_shape")
    if math.prod(shape) > _INT64_MAX:
        raise ValueError("sparse_shape product exceeds INT64 coordinate key capacity")
    return shape


def _coordinates(coords, shape):
    if isinstance(coords, torch.Tensor):
        if coords.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("coords must have integer dtype; floating, non-finite and bool coordinates are unsupported")
        array = coords.detach().cpu().numpy()
    elif isinstance(coords, np.ndarray):
        if coords.dtype.kind not in "iu":
            raise ValueError("coords must have integer dtype")
        array = coords
    else:
        raise ValueError("coords must be a Tensor or NumPy integer array [N,3]")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("coords must have shape [N,3] in XYZ order")
    if array.size and ((array < 0).any() or (array >= np.asarray(shape, dtype=np.int64)).any()):
        raise ValueError("coords contain sites outside sparse_shape")
    array = np.array(array, dtype=np.int64, order="C", copy=True)
    if len(array) > _INT32_MAX:
        raise ValueError("Input row+1 indices exceed INT32 capacity")
    encoded = _keys(array, shape)
    if len(np.unique(encoded)) != len(encoded):
        raise ValueError("coords must be unique; duplicate active sites are unsupported")
    return array


def _keys(coords, shape):
    return (coords[:, 0] * shape[1] + coords[:, 1]) * shape[2] + coords[:, 2]


def _capacity(value, count, name):
    capacity = max(1, count) if value is None else _integer(value, name, 1)
    if capacity < count:
        raise ValueError(f"{name}={capacity} is smaller than required active count {count}; refusing to truncate")
    return capacity


def plan_sparse_convolution(coords, sparse_shape, *, kernel, stride, padding, subm,
                            input_capacity=None, output_capacity=None):
    """Build exact CPU neighbors for ReferenceSparseConv's XYZ conventions.

    Submanifold output retains the input row order and shape, uses kernel//2
    padding regardless of declared padding, and follows query=out*stride-pad+off.
    Strided output is the sorted unique union of all in-bounds candidates with
    integral (input+padding-offset)/stride. Kernel order is product(X,Y,Z).

    neighbor_indices is INT32 [K,output_capacity]. Entries are 0 for missing
    neighbors/padded outputs, or original input row+1. Output coordinates are
    unpadded INT32 [output_count,3], mask is FP32 [output_capacity,1].
    """
    shape = _shape(sparse_shape)
    kernel = _triple(kernel, "kernel", 1)
    stride = _triple(stride, "stride", 1)
    padding = _triple(padding, "padding")
    if not isinstance(subm, (bool, np.bool_)):
        raise ValueError("subm must be bool")
    coordinates = _coordinates(coords, shape)
    input_count = len(coordinates)
    input_capacity = _capacity(input_capacity, input_count, "input_capacity")
    pad = tuple(k // 2 for k in kernel) if subm else padding
    output_shape = shape if subm else tuple((s + 2*p - k)//st + 1 for s,p,k,st in zip(shape,padding,kernel,stride))
    if any(v < 0 or v > _INT32_MAX for v in output_shape):
        raise ValueError(f"Convolution output shape is outside INT32 nonnegative dimensions: {output_shape}")
    if math.prod(output_shape) > _INT64_MAX:
        raise ValueError("Output shape product exceeds INT64 coordinate key capacity")
    offsets = tuple(product(*(range(k) for k in kernel)))
    stride_array, pad_array = np.asarray(stride, dtype=np.int64), np.asarray(pad, dtype=np.int64)
    if subm:
        output = coordinates.copy()
    elif not input_count or not all(output_shape):
        output = np.empty((0, 3), dtype=np.int64)
    else:
        candidates = []
        bounds = np.asarray(output_shape, dtype=np.int64)
        for offset in offsets:
            numerator = coordinates + pad_array - np.asarray(offset, dtype=np.int64)
            candidate = numerator // stride_array
            valid = ((numerator % stride_array == 0).all(axis=1)
                     & (candidate >= 0).all(axis=1) & (candidate < bounds).all(axis=1))
            candidates.append(_keys(candidate[valid], output_shape))
        encoded = np.unique(np.concatenate(candidates))
        yz = output_shape[1] * output_shape[2]
        output = np.column_stack((encoded // yz, encoded // output_shape[2] % output_shape[1],
                                  encoded % output_shape[2]))
    output_count = len(output)
    output_capacity = _capacity(output_capacity, output_count, "output_capacity")
    neighbors = np.zeros((len(offsets), output_capacity), dtype=np.int32)
    if input_count and output_count:
        encoded = _keys(coordinates, shape)
        permutation = np.argsort(encoded, kind="stable")
        sorted_keys = encoded[permutation]
        bounds = np.asarray(shape, dtype=np.int64)
        for index, offset in enumerate(offsets):
            query = output * stride_array - pad_array + np.asarray(offset, dtype=np.int64)
            valid = (query >= 0).all(axis=1) & (query < bounds).all(axis=1)
            rows = np.flatnonzero(valid)
            query_keys = _keys(query[rows], shape)
            positions = np.searchsorted(sorted_keys, query_keys)
            positions_safe = np.minimum(positions, input_count - 1)
            found = (positions < input_count) & (sorted_keys[positions_safe] == query_keys)
            neighbors[index, rows[found]] = (permutation[positions_safe[found]] + 1).astype(np.int32)
    mask = np.zeros((output_capacity, 1), dtype=np.float32)
    mask[:output_count] = 1
    return SparseConvPlan(input_count, output_count, input_capacity, output_capacity, tuple(output_shape),
                          torch.from_numpy(output.astype(np.int32)), torch.from_numpy(neighbors),
                          torch.from_numpy(mask))


def plan_sparse_encoder(original_backbone, coords, capacities=None, input_capacity=None):
    """Walk the source SparseSequence/SparseBlock without evaluating any weights.

    capacities optionally maps each ReferenceSparseConv named_modules name to
    its output capacity. Unspecified strided layers use max(1,output_count);
    unspecified subm layers preserve input_capacity for residual tensor shapes.
    Explicit capacities in residual blocks must preserve residual capacity.
    Identical subm topology with equal capacities shares a plan object inside
    this call only. Every invocation derives its plans from the current coords.
    """
    from .reference import ReferenceSparseConv, SparseBlock, SparseSequence

    if not hasattr(original_backbone, "sparse_shape"):
        raise ValueError("original_backbone must expose sparse_shape")
    shape = _shape(original_backbone.sparse_shape)
    coordinates = _coordinates(coords, shape)
    capacity = _capacity(input_capacity, len(coordinates), "input_capacity")
    initial_count, initial_capacity = len(coordinates), capacity
    if capacities is None:
        capacities = {}
    if not isinstance(capacities, Mapping):
        raise ValueError("capacities must map convolution names to output capacities")
    conv_names = {name for name, module in original_backbone.named_modules()
                  if isinstance(module, ReferenceSparseConv)}
    unknown = set(capacities) - conv_names
    if unknown:
        raise ValueError(f"Unknown convolution capacity names: {sorted(unknown)}")
    plans, topology_cache = {}, {}

    def walk(module, name, current, current_shape, current_capacity, token):
        if isinstance(module, ReferenceSparseConv):
            requested = capacities.get(name, current_capacity if module.subm else None)
            requested = None if requested is None else _capacity(requested, 0, f"capacities[{name!r}]")
            pad = tuple(k // 2 for k in module.kernel) if module.subm else tuple(module.padding)
            signature = (token, current_shape, tuple(module.kernel), tuple(module.stride), pad,
                         current_capacity, requested)
            plan = topology_cache.get(signature) if module.subm else None
            if plan is None:
                plan = plan_sparse_convolution(current, current_shape, kernel=module.kernel,
                    stride=module.stride, padding=module.padding, subm=module.subm,
                    input_capacity=current_capacity, output_capacity=requested)
                if module.subm:
                    topology_cache[signature] = plan
            plans[name] = plan
            next_token = token if module.subm else object()
            return plan.output_coords.numpy().astype(np.int64, copy=False), plan.output_shape, plan.output_capacity, next_token
        if isinstance(module, SparseBlock):
            original_coords, original_shape, original_capacity = current, current_shape, current_capacity
            # The residual sum has no remapping or padding adaptation in source.
            state = (current, current_shape, current_capacity, token)
            for child_name in ("conv1", "conv2"):
                state = walk(getattr(module, child_name), f"{name}.{child_name}", *state)
            if state[1] != original_shape or state[2] != original_capacity or not np.array_equal(state[0], original_coords):
                raise ValueError(f"Residual block {name!r} must preserve coordinates, shape and capacity")
            return state
        if isinstance(module, SparseSequence):
            state = (current, current_shape, current_capacity, token)
            for child_name, child in module.named_children():
                child_name = f"{name}.{child_name}" if name else child_name
                if isinstance(child, (ReferenceSparseConv, SparseSequence, SparseBlock)):
                    state = walk(child, child_name, *state)
                elif not isinstance(child, (torch.nn.BatchNorm1d, torch.nn.ReLU, torch.nn.Identity)):
                    raise ValueError(f"Unsupported sparse sequence module {child_name!r}: {type(child).__name__}")
            return state
        raise ValueError(f"Unsupported sparse planning module {name!r}: {type(module).__name__}")

    state = (coordinates, shape, capacity, object())
    for name in ("conv_input", "encoder_layers", "conv_out"):
        if not hasattr(original_backbone, name):
            raise ValueError(f"original_backbone is missing {name}")
        state = walk(getattr(original_backbone, name), name, *state)
    final, final_shape, _, _ = state
    if set(plans) != conv_names:
        raise ValueError("Backbone contains convolutions outside the supported encoder execution path")
    lookup = np.zeros(math.prod(final_shape), dtype=np.int32)
    if len(final):
        lookup[_keys(final, final_shape)] = np.arange(1, len(final) + 1, dtype=np.int32)
    return EncoderPlan(plans, initial_count, initial_capacity,
                       torch.from_numpy(final.astype(np.int32, copy=True)), final_shape, torch.from_numpy(lookup))
