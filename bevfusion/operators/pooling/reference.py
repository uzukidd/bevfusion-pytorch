"""Inference-only BEV grid association and interval-reduction plans.

Plans contain geometry-derived indices only. Per-frame image features and depth
probabilities are always recomputed; no prediction tensor is retained here.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PoolChunk:
    batch: int
    camera: int
    start: int
    end: int
    # Original valid-point order for reference index_add; sorted order for CUDA.
    indices: torch.Tensor
    ranks: torch.Tensor
    sorted_indices: torch.Tensor
    coordinates: torch.Tensor
    interval_starts: torch.Tensor
    interval_lengths: torch.Tensor


@dataclass(frozen=True)
class BEVPoolPlan:
    shape: tuple
    batch_size: int
    pixels: int
    chunks: tuple

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for chunk in self.chunks
                   for t in (chunk.indices, chunk.ranks, chunk.sorted_indices,
                             chunk.coordinates, chunk.interval_starts, chunk.interval_lengths))


def build_pool_plan(coords, shape, depth_chunk):
    """Precompute valid points, their permutation, and contiguous BEV intervals.

    coords uses [B,N,D,H,W,XYZ]. CUDA coordinates use [X,Y,Z,local_batch],
    preserving the installed extension's B,Z,X,Y,C output memory layout.
    The same per-camera/depth chunks retain the previous reduction ordering.
    """
    if depth_chunk < 1:
        raise ValueError("pool_depth_chunk must be positive")
    batch_size, cameras, depths, height, width, _ = coords.shape
    nx, ny, nz = shape
    bounds = coords.new_tensor(shape)
    chunks = []
    for batch in range(batch_size):
        for camera in range(cameras):
            for start in range(0, depths, depth_chunk):
                end = min(depths, start + depth_chunk)
                co = coords[batch, camera, start:end].reshape(-1, 3)
                indices = ((co >= 0) & (co < bounds)).all(-1).nonzero().flatten()
                if not indices.numel():
                    continue  # Avoid the legacy CUDA kernel's zero-block launch.
                co = co[indices]
                ranks = (co[:, 2] * nx + co[:, 0]) * ny + co[:, 1]
                # Match ops.bev_pool's X,Y,Z,batch sort exactly. Sorting by the
                # output's Z,X,Y memory order would alter equal-grid summation.
                sort_ranks = co[:, 0] * (ny * nz) + co[:, 1] * nz + co[:, 2]
                permutation = sort_ranks.argsort()
                sorted_ranks = sort_ranks[permutation]
                coordinates = F.pad(co[permutation], (0, 1)).int().contiguous()
                kept = torch.ones_like(sorted_ranks, dtype=torch.bool)
                kept[1:] = sorted_ranks[1:] != sorted_ranks[:-1]
                starts = kept.nonzero().flatten().int().contiguous()
                lengths = torch.diff(torch.cat((starts, starts.new_tensor([len(indices)])))).contiguous()
                chunks.append(PoolChunk(
                    batch, camera, start, end, indices, ranks,
                    indices[permutation].contiguous(), coordinates, starts, lengths))
    return BEVPoolPlan(tuple(shape), batch_size, height * width, tuple(chunks))


def pool_with_plan(plan, depth, features, backend):
    """Apply fresh depth/features to a plan, with no sorting or geometry work."""
    if backend not in ("reference", "auto", "cuda"):
        raise ValueError("pool_backend must be reference, cuda or auto")
    use_cuda = depth.is_cuda and backend in ("auto", "cuda")
    if backend == "cuda" and not depth.is_cuda:
        raise ValueError("The cuda pooling backend requires CUDA tensors")
    kernel = None
    if use_cuda:
        try:
            from ops.bev_pool import bev_pool_ext
        except (ImportError, OSError) as exc:
            raise RuntimeError("Install bevfusion-bev-pool-ops with scripts/install_ops.bat") from exc
        kernel = bev_pool_ext.bev_pool_forward
        # The installed legacy kernel launches on the default CUDA stream.
        # Explicitly bridge a caller's non-default stream, including allocations
        # and gathers, rather than racing its features or association indices.
        caller = torch.cuda.current_stream(depth.device)
        default = torch.cuda.default_stream(depth.device)
        if caller != default:
            default.wait_stream(caller)
            for tensor in (depth, features):
                tensor.record_stream(default)
            for chunk in plan.chunks:
                for tensor in (chunk.sorted_indices, chunk.coordinates,
                               chunk.interval_starts, chunk.interval_lengths):
                    tensor.record_stream(default)
            with torch.cuda.stream(default):
                result = _execute_plan(plan, depth, features, kernel)
            caller.wait_stream(default)
            result.record_stream(caller)
            return result
    return _execute_plan(plan, depth, features, kernel)


def _execute_plan(plan, depth, features, kernel):
    use_cuda = kernel is not None
    nx, ny, nz = plan.shape
    channels = features.shape[2]
    volume = nz * nx * ny
    out = features.new_zeros((plan.batch_size * volume, channels), dtype=torch.float32)
    image_features = features.flatten(3).permute(0, 1, 3, 2)
    for chunk in plan.chunks:
        index = chunk.sorted_indices if use_cuda else chunk.indices
        probability = depth[chunk.batch, chunk.camera, chunk.start:chunk.end].reshape(-1)[index]
        values = probability[:, None] * image_features[chunk.batch, chunk.camera][index % plan.pixels]
        if use_cuda:
            # Native extension accepts already sorted coordinates/intervals.
            # Bypass its Python wrapper's per-frame argsort and interval building.
            pooled = kernel(values.float().contiguous(), chunk.coordinates,
                            chunk.interval_lengths, chunk.interval_starts, 1, nz, nx, ny)
            offset = chunk.batch * volume
            out[offset:offset + volume] += pooled.reshape(volume, channels)
        else:
            out.index_add_(0, chunk.ranks + chunk.batch * volume, values.float())
    return out.reshape(plan.batch_size, nz, nx, ny, channels).permute(0, 1, 4, 2, 3).reshape(
        plan.batch_size, nz * channels, nx, ny)
