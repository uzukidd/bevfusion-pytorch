"""Full-model inference bridge for per-frame CPU topology and tensor sparse ops.

These are runtime plans, never checkpoint buffers. Plans are recomputed for every
cloud; only tensor graph wrappers with identical capacities are reused. This does
not imply an entire point-cloud pipeline runs on BPU: grouping and rulebooks are
explicit CPU preprocessing. Learned weights are the original encoder Parameters.
"""
from __future__ import annotations

import time

import torch

from .operators.sparse_conv.compatible import CompatibleSparseEncoder
from .operators.voxelization.compatible import CompatibleVoxelMean, build_voxel_plan
from .operators.sparse_conv.plan import plan_sparse_encoder


def compatible_lidar_forward(model, points):
    """Raw points -> dense BEV, preserving weights and all retained active sites.

    No additional point or voxel truncation is introduced. Current candidate
    limits (<131072 raw points and active-row capacities) raise explicitly when
    exceeded. For fixed-shape hardware deployment use the exporter capacity
    profile; this PyTorch bridge sizes each frame to its actual activity.
    """
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("Compatible LiDAR is inference-only; use eval() and inference_mode()")
    started = time.perf_counter()
    voxel = build_voxel_plan(points, voxel_size=model.voxel_size,
                             point_cloud_range=model.point_cloud_range,
                             max_voxels=model.max_voxels)
    original = model.encoders["lidar"]["backbone"]
    topology = plan_sparse_encoder(original, voxel.coordinates, input_capacity=voxel.voxel_capacity)
    planned = time.perf_counter()
    signature = (id(original), voxel.point_capacity, voxel.voxel_capacity,
                 tuple((name, item.input_capacity, item.output_capacity, tuple(item.output_shape))
                       for name, item in topology.plans.items()),
                 tuple((id(parameter), parameter.device, parameter.dtype) for parameter in original.parameters()),
                 tuple((id(buffer), buffer.device, buffer.dtype) for buffer in original.buffers()))
    cached = getattr(model, "_compatible_lidar_runtime", None)
    if cached is None or cached[0] != signature:
        mean = CompatibleVoxelMean(voxel.point_capacity, voxel.voxel_capacity, voxel.max_points)
        sparse = CompatibleSparseEncoder(original, topology)
        # Tuple prevents registering weight-sharing wrappers in the model tree.
        model._compatible_lidar_runtime = (signature, mean, sparse)
    else:
        _, mean, sparse = cached
    device = next(original.parameters()).device
    features = mean(*voxel.tensor_feed(device))
    output = sparse(features, *sparse.plan_inputs(topology, device))
    model.compatible_lidar_info = {
        "grouping_device": "cpu", "topology_rebuilt": True,
        "cpu_planning_seconds": planned - started,
        "voxel": voxel.statistics(),
        "active_counts": {name: item.output_count for name, item in topology.plans.items()},
        "runtime_inputs": len(sparse.input_names), "graph_reused": cached is not None and cached[0] == signature,
        "bpu_compiled": False,
    }
    return output
