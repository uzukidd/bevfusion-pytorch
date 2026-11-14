"""Exact ordinary sparse convolution via bounded dense Conv3D patches.

Spatial axes remain X/Y/Z (the D/H/W axes of torch Conv3D). Topology is
coordinate-only: zero features still activate every geometrically reached site.
"""
from contextlib import nullcontext
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F


def _triple(value):
    return (value,) * 3 if isinstance(value, int) else tuple(value)


def dense_conv3d_patch(patch, weight, stride=1, padding=0):
    """Convolve [1,Cin,X,Y,Z] using the original [kx,ky,kz,Cin,Cout] weight.

    This fixed-shape, purely numeric helper is suitable for a float/QAT export
    probe. Pass the source Parameter directly; the permutation neither replaces
    nor copies it into another Parameter. Patch construction/topology stay eager.
    """
    return F.conv3d(patch, weight.permute(4, 3, 0, 1, 2),
                    stride=_triple(stride), padding=_triple(padding))


class DenseConv3dPatch(nn.Module):
    """Stateless exportable tile operation: forward(patch, original_weight)."""

    def __init__(self, stride=1, padding=0):
        super().__init__()
        self.stride, self.padding = _triple(stride), _triple(padding)

    def forward(self, patch, weight):
        return dense_conv3d_patch(patch, weight, self.stride, self.padding)


def ordinary_output_coordinates(coords, spatial_shape, kernel, stride=1, padding=0):
    """Return sorted active XYZ coordinates and exact ordinary-conv geometry.

    An input i reaches output o for offset k exactly when i+p-k=o*stride.
    No feature values or numeric sparse convolution participate in this step.
    """
    kernel, stride, padding = _triple(kernel), _triple(stride), _triple(padding)
    out_shape = tuple((s + 2 * p - k) // st + 1
                      for s, p, k, st in zip(spatial_shape, padding, kernel, stride))
    coords = coords.long()
    if not len(coords) or any(s <= 0 for s in out_shape):
        return coords.new_empty((0, 3)), out_shape
    step, pad, bounds = (coords.new_tensor(v) for v in (stride, padding, out_shape))
    candidates = []
    for offset in product(*(range(k) for k in kernel)):
        numerator = coords + pad - coords.new_tensor(offset)
        output = torch.div(numerator, step, rounding_mode="floor")
        valid = (numerator.remainder(step) == 0).all(1)
        valid &= ((output >= 0) & (output < bounds)).all(1)
        candidates.append(output[valid])
    joined = torch.cat(candidates)
    linear = (joined[:, 0] * out_shape[1] + joined[:, 1]) * out_shape[2] + joined[:, 2]
    linear = torch.unique(linear, sorted=True)
    output = torch.stack((linear // (out_shape[1] * out_shape[2]),
                          linear // out_shape[2] % out_shape[1],
                          linear % out_shape[2]), dim=1)
    return output, out_shape


def tiled_sparse_conv3d(state, weight, stride=1, padding=0, tile_size=16):
    """Ordinary sparse Conv3D, tiling output X without a dense full input grid.

    Each patch spans at most (tile_size-1)*stride_x+kernel_x input X
    cells, including explicit X halo/padding, and the original input Y/Z.
    Conv3D retains the original Y/Z padding. Only active output rows are
    gathered before any downstream BatchNorm. There are no capacity limits.
    """
    from .reference import SparseState

    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    stride, padding = _triple(stride), _triple(padding)
    kernel = tuple(weight.shape[:3])
    output_coords, output_shape = ordinary_output_coordinates(
        state.coords, state.shape, kernel, stride, padding)
    if not len(output_coords):
        return SparseState(state.features.new_zeros((0, weight.shape[-1])),
                           output_coords, output_shape)

    coords = state.coords.long()
    pieces = []
    # Sorted topology makes tile order also sparse-output row order. Skip empty
    # X slabs, but process every active slab and its complete (possibly short) tail.
    tile_ids = torch.unique(output_coords[:, 0] // tile_size, sorted=True).tolist()
    for tile_id in tile_ids:
        start = tile_id * tile_size
        stop = min(start + tile_size, output_shape[0])
        origin = start * stride[0] - padding[0]
        width = (stop - start - 1) * stride[0] + kernel[0]
        selected = (coords[:, 0] >= origin) & (coords[:, 0] < origin + width)
        local = coords[selected]
        patch = state.features.new_zeros((1, weight.shape[-2], width,
                                          state.shape[1], state.shape[2]))
        patch[0, :, local[:, 0] - origin, local[:, 1], local[:, 2]] = state.features[selected].T
        # Sparse FP32 GEMM does not reduce inputs to TF32. Apply the matching
        # precision only in eager tiling; the export helper stays pure Conv3D.
        precision = (torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic, allow_tf32=False)
            if patch.is_cuda and patch.dtype == torch.float32 else nullcontext())
        with precision:
            dense = dense_conv3d_patch(patch, weight, stride, (0, padding[1], padding[2]))
        active = output_coords[(output_coords[:, 0] >= start) & (output_coords[:, 0] < stop)]
        pieces.append(dense[0, :, active[:, 0] - start, active[:, 1], active[:, 2]].T)
    return SparseState(torch.cat(pieces, dim=0), output_coords, output_shape)
