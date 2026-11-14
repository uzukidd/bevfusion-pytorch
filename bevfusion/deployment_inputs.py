"""Per-frame application inputs for fixed-capacity BEVFusion deployment graphs.

Grouping, sparse topology and detection depth projection run outside the graph.
Every call rebuilds its plans; sample coordinates are never treated as constants.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import operator
import time

import torch

from .base import _camera_matrix
from .operators.sparse_conv.compatible import CompatibleSparseEncoder
from .operators.voxelization.compatible import CompatibleVoxelMean, VoxelizationPlan, build_voxel_plan
from .depth import project_lidar_depth
from .operators.sparse_conv.reference import ReferenceSparseConv
from .preprocessing import prepare_points, preprocess_images, voxelize
from .operators.sparse_conv.plan import EncoderPlan, plan_sparse_encoder


@dataclass(frozen=True)
class DeploymentProfile:
    point_capacity: int = 65535
    voxel_capacity: int = 32768
    sparse_capacity: int = 32768
    page_size: int = 32768

    def __post_init__(self):
        for name in ("point_capacity", "voxel_capacity", "sparse_capacity", "page_size"):
            value = getattr(self, name)
            try:
                integer = operator.index(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be an integer") from exc
            maximum = 65535 if name == "page_size" else 131071
            if isinstance(value, bool) or not 1 <= integer <= maximum:
                raise ValueError(f"{name} must be in [1,{maximum}]")
            object.__setattr__(self, name, integer)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("profile must contain only point_capacity, voxel_capacity, sparse_capacity, page_size")
        return cls(**value)


@dataclass(frozen=True)
class DeploymentFrame:
    feed: dict[str, torch.Tensor]
    voxel_plan: VoxelizationPlan
    encoder_plan: EncoderPlan | None
    calibration: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    statistics: dict

    def tensor_feed(self, device=None):
        return tuple(value.to(device=device) if device is not None else value for value in self.feed.values())

    def feed_np(self):
        return {name: value.detach().cpu().contiguous().numpy() for name, value in self.feed.items()}


def _calibration(value):
    names = ("camera_intrinsics", "camera2lidar", "img_aug_matrix")
    if isinstance(value, dict):
        if any(name not in value for name in names):
            raise ValueError("calibration requires camera_intrinsics, camera2lidar and img_aug_matrix")
        value = tuple(value[name] for name in names)
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("calibration must be a (K, camera2lidar, img_aug_matrix) triple or mapping")
    result = []
    for matrix, name, sizes in zip(value, names, (((3, 3), (4, 4)), ((4, 4),), ((4, 4),))):
        matrix = _camera_matrix(matrix, name, sizes)
        if matrix is None:
            raise ValueError(f"{name} is required")
        if name == "camera_intrinsics":
            matrix = matrix[..., :3, :3]
        result.append(matrix.detach().cpu().float().contiguous().clone())
    return tuple(result)


def _prepare_sample(model, sample, bound):
    if not isinstance(sample, dict) or not {"images", "points"}.issubset(sample):
        raise ValueError("sample must contain images and points")
    images = sample["images"]
    if not isinstance(images, torch.Tensor):
        raise ValueError("images must be a tensor")
    images = images.detach().cpu()
    if images.dtype == torch.uint8:
        images, augmentation = preprocess_images(images)
    else:
        if not images.is_floating_point():
            raise ValueError("images must be uint8 RGB or normalized floating point")
        images = images.float()
        if images.ndim == 4:
            images = images.unsqueeze(0)
        augmentation = sample.get("img_aug_matrix")
        if augmentation is None:
            raise ValueError("Normalized images require explicit img_aug_matrix")
    if tuple(images.shape) != (1, 6, 3, 256, 704) or not torch.isfinite(images).all():
        raise ValueError("images must normalize to finite [1,6,3,256,704]")
    calibration = _calibration((sample.get("camera_intrinsics", getattr(model, "_camera_intrinsics", None)),
                                sample.get("camera2lidar", getattr(model, "_camera2lidar", None)), augmentation))
    if bound is not None:
        for name, actual, expected in zip(("camera_intrinsics", "camera2lidar", "img_aug_matrix"),
                                          calibration, _calibration(bound)):
            if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
                raise ValueError(f"Sample {name} differs from fixed calibration; re-export for changed calibration")
    points = prepare_points(sample["points"]).detach().cpu().contiguous()
    return images.contiguous(), points, calibration


def build_deployment_frame(model, sample, *, profile, voxel_backend="compatible",
                           spconv_backend="compatible", device="cpu", calibration=None):
    """Prepare one frame without mutating model weights, backend or device.

    ``cpu`` spconv means the entire LiDAR BEV is application preprocessing and
    requires an eval CPU reference/auto backbone. Other choices keep mean and/or
    sparse feature computation in the deployment graph. Capacities never cause
    extra truncation; the source max_voxels and first-ten-points rules still apply.
    """
    if not isinstance(profile, DeploymentProfile):
        raise ValueError("profile must be DeploymentProfile")
    if voxel_backend not in ("compatible", "cpu") or spconv_backend not in ("compatible", "cpu"):
        raise ValueError("voxel_backend and spconv_backend must be compatible or cpu")
    if model.training:
        raise ValueError("Deployment input preparation requires model.eval()")
    started = time.perf_counter()
    images, points, matrices = _prepare_sample(model, sample, calibration)
    backbone = model.encoders["lidar"]["backbone"]
    with torch.inference_mode():
        voxel = build_voxel_plan(points, voxel_size=model.voxel_size,
            point_cloud_range=model.point_cloud_range, max_voxels=model.max_voxels,
            max_points=10, point_capacity=profile.point_capacity, voxel_capacity=profile.voxel_capacity)
        feed = OrderedDict(images=images)
        cpu_features = None
        if voxel_backend == "cpu":
            source_features, source_coords = voxelize(points, voxel_size=model.voxel_size,
                point_cloud_range=model.point_cloud_range, max_voxels=model.max_voxels,
                max_points=10, backend="reference")
            if len(source_features) != voxel.voxel_count or not torch.equal(source_coords, voxel.coordinates.long()):
                raise AssertionError("CPU voxelization differs from the per-frame grouping plan")
            cpu_features = source_features.new_zeros((profile.voxel_capacity, 5))
            cpu_features[:voxel.voxel_count] = source_features
        topology = None
        if spconv_backend == "compatible":
            capacities = {name: profile.sparse_capacity for name, layer in backbone.named_modules()
                          if isinstance(layer, ReferenceSparseConv)}
            topology = plan_sparse_encoder(backbone, voxel.coordinates, capacities=capacities,
                                           input_capacity=profile.voxel_capacity)
            if voxel_backend == "compatible":
                feed.update(zip(("points", "point_indices", "inverse_counts"), voxel.tensor_feed("cpu")))
            else:
                feed["voxel_features"] = cpu_features
            adapter = CompatibleSparseEncoder(backbone, topology, page_size=profile.page_size)
            feed.update(zip(adapter.input_names[1:], adapter.plan_inputs(topology, "cpu")))
        else:
            if backbone.training or any(value.device.type != "cpu" for value in
                                        (*backbone.parameters(), *backbone.buffers())):
                raise ValueError("spconv_backend=cpu requires an eval CPU model; no implicit device migration")
            if any(getattr(layer, "backend", "reference") == "cuda" for layer in backbone.modules()
                   if isinstance(layer, ReferenceSparseConv)):
                raise ValueError("spconv_backend=cpu requires a reference/auto backbone, not cuda")
            if cpu_features is None:
                mean = CompatibleVoxelMean(profile.point_capacity, profile.voxel_capacity, 10).eval()
                cpu_features = mean(*voxel.tensor_feed("cpu"))
            features = cpu_features[:voxel.voxel_count]
            feed["lidar_bev"] = backbone(features, voxel.coordinates.long(), verbose=False).detach().contiguous()
        task = getattr(model, "task", None)
        if task not in ("det", "seg"):
            raise ValueError("Deployment model task must be det or seg")
        if task == "det":
            lower, upper = points.new_tensor(model.point_cloud_range[:3]), points.new_tensor(model.point_cloud_range[3:])
            raw = points[((points[:, :3] >= lower) & (points[:, :3] < upper)).all(1)]
            intrinsic, extrinsic, augmentation = matrices
            feed["lidar_depth"] = project_lidar_depth(raw, extrinsic, intrinsic, augmentation,
                                                       torch.eye(4)[None], (256, 704)).contiguous()
    statistics = {"task": task, "profile": profile.to_dict(), "voxel_backend": voxel_backend,
                  "spconv_backend": spconv_backend,
                  "voxel_compute_location": "graph" if voxel_backend == "compatible" and spconv_backend == "compatible" else "cpu_preprocess",
                  "spconv_location": "graph" if spconv_backend == "compatible" else "cpu_preprocess",
                  "cpu_planning_seconds": time.perf_counter() - started,
                  "topology_rebuilt": topology is not None, "grouping_device": "cpu",
                  "voxel": voxel.statistics(), "input_names": list(feed),
                  "input_bytes": sum(value.numel() * value.element_size() for value in feed.values()),
                  "detection_depth_uses_unvoxelized_points": task == "det"}
    if topology is not None:
        statistics["sparse_active_counts"] = {name: item.output_count for name, item in topology.plans.items()}
        statistics["final_shape_xyz"] = list(topology.final_shape)
    feed = OrderedDict((name, value.to(device=device).contiguous()) for name, value in feed.items())
    return DeploymentFrame(feed, voxel, topology, matrices, statistics)


__all__ = ["DeploymentProfile", "DeploymentFrame", "build_deployment_frame"]
