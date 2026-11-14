"""Compatible LiDAR ONNX adapter and weight-free per-frame input planning.

Geometry-derived neighbours are application inputs, never frozen sample buffers.
The ordinary ONNX detection output/filtering contract remains unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import json

import numpy as np
import torch
from torch import nn

from bevfusion.operators.sparse_conv.compatible import CompatibleSparseEncoder
from bevfusion.operators.voxelization.compatible import CompatibleVoxelMean, build_voxel_plan
from bevfusion.deployment_inputs import DeploymentProfile
from bevfusion.operators.sparse_conv.reference import ReferenceSparseConv, ReferenceSparseEncoder
from bevfusion.operators.sparse_conv.plan import plan_sparse_encoder
from tools.onnx_model import OnnxCamera, OnnxTransFusionHead, _fixed_calibration

SENSOR_NAMES = ("images", "points", "camera_intrinsics", "camera2lidar", "img_aug_matrix")


def _layout(backbone):
    return [{"name": name, "kernel": list(layer.kernel), "stride": list(layer.stride),
             "padding": list(layer.padding), "subm": bool(layer.subm)}
            for name, layer in backbone.named_modules() if isinstance(layer, ReferenceSparseConv)]


@lru_cache(maxsize=8)
def _planning_backbone(shape):
    # The planner reads convolution metadata only. Initializing this template
    # must neither load learned parameters nor alter the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        return ReferenceSparseEncoder(shape, backend="reference").eval()


def _topology(backbone, voxel, profile):
    capacities = {name: profile.sparse_capacity for name, layer in backbone.named_modules()
                  if isinstance(layer, ReferenceSparseConv)}
    return plan_sparse_encoder(backbone, voxel.coordinates, capacities=capacities,
                               input_capacity=profile.voxel_capacity)


def _plan_tensors(backbone, topology):
    """Pack the same execution-order topology aliases as CompatibleSparseEncoder."""
    result, aliases = OrderedDict(), {}
    for name, layer in backbone.named_modules():
        if not isinstance(layer, ReferenceSparseConv):
            continue
        plan = topology.plans[name]
        if id(plan) in aliases:
            continue
        slot = len(aliases)
        aliases[id(plan)] = slot
        result[f"neighbors_{slot}"] = plan.neighbor_indices
        result[f"mask_{slot}"] = plan.output_mask
    result["dense_lookup"] = topology.dense_lookup
    return result


def _json_record(metadata, key):
    try:
        value = json.loads(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Compatible ONNX requires valid {key} JSON metadata") from exc
    return value


def _sensor_tensors(feed, image_shape):
    if not isinstance(feed, dict) or not set(SENSOR_NAMES).issubset(feed):
        raise ValueError("Compatible ONNX planning requires images, raw points and all three camera matrices")
    values = []
    for name in SENSOR_NAMES:
        value = feed[name]
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(np.ascontiguousarray(value))
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a normalized FP32 tensor/NumPy array")
        value = value.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        values.append(value)
    images, points, *matrices = values
    if list(images.shape) != list(image_shape):
        raise ValueError(f"images must have shape {image_shape}")
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError("points must have shape [P,5], including time_lag_seconds")
    matrices = _fixed_calibration(tuple(matrices), "cpu")
    return images, points, *matrices


def _validate_contract(feed, contract):
    if not isinstance(contract, list) or not contract or any(not isinstance(item, dict) for item in contract):
        raise ValueError("lidar_input_contract must be a nonempty list of input records")
    if list(feed) != [item.get("name") for item in contract]:
        raise ValueError("Prepared compatible input names/order differ from the ONNX contract")
    for item in contract:
        value = feed[item["name"]]
        shape = item.get("shape")
        if (not isinstance(shape, list) or len(value.shape) != len(shape)
                or any(expected is not None and actual != expected for actual, expected in zip(value.shape, shape))):
            raise ValueError(f"Prepared {item['name']} shape differs from the ONNX contract")
        if str(value.dtype) != item.get("dtype"):
            raise ValueError(f"Prepared {item['name']} dtype differs from the ONNX contract")


def validate_compatible_metadata(metadata):
    """Validate the public planning contract without creating model weights."""
    if not isinstance(metadata, dict) or metadata.get("lidar_backend") != "compatible":
        raise ValueError("Compatible planning requires lidar_backend='compatible'")
    if metadata.get("cpu_planning_required") != "true" or metadata.get("plan_tensors") != "runtime_inputs":
        raise ValueError("Compatible ONNX must declare per-frame CPU planning and runtime plan inputs")
    raw_profile = _json_record(metadata, "lidar_profile")
    if not isinstance(raw_profile, dict) or set(raw_profile) != set(DeploymentProfile.__dataclass_fields__):
        raise ValueError("lidar_profile must explicitly contain all four capacity/page fields")
    profile = DeploymentProfile.from_dict(raw_profile)
    geometry = _json_record(metadata, "lidar_geometry")
    contract = _json_record(metadata, "lidar_input_contract")
    fields = {"voxel_size", "point_cloud_range", "max_voxels", "sparse_shape", "convolutions"}
    if not isinstance(geometry, dict) or set(geometry) != fields:
        raise ValueError("Compatible LiDAR geometry must contain voxel_size, point_cloud_range, max_voxels, sparse_shape and convolutions")
    size, bounds = np.asarray(geometry["voxel_size"]), np.asarray(geometry["point_cloud_range"])
    try:
        valid_bounds = (size.shape == (3,) and bounds.shape == (6,) and np.isfinite(size).all()
                        and np.isfinite(bounds).all() and (size > 0).all() and (bounds[3:] > bounds[:3]).all())
    except (TypeError, ValueError):
        valid_bounds = False
    if not valid_bounds:
        raise ValueError("voxel_size and point_cloud_range require finite positive sizes and ordered bounds")
    shape = geometry["sparse_shape"]
    if (not isinstance(shape, list) or len(shape) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape)):
        raise ValueError("sparse_shape requires three positive integers")
    maximum = geometry["max_voxels"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("max_voxels must be a positive integer")
    convolutions = geometry["convolutions"]
    if not isinstance(convolutions, list) or not convolutions:
        raise ValueError("Compatible metadata requires the convolution topology layout")
    for conv in convolutions:
        if (not isinstance(conv, dict) or set(conv) != {"name", "kernel", "stride", "padding", "subm"}
                or not isinstance(conv["name"], str) or not conv["name"] or not isinstance(conv["subm"], bool)):
            raise ValueError("Invalid sparse convolution topology record")
        for key in ("kernel", "stride", "padding"):
            if (not isinstance(conv[key], list) or len(conv[key]) != 3
                    or any(isinstance(v, bool) or not isinstance(v, int) or v < (0 if key == "padding" else 1) for v in conv[key])):
                raise ValueError("Invalid sparse convolution dimensions")
    if len({conv["name"] for conv in convolutions}) != len(convolutions):
        raise ValueError("Sparse convolution names must be unique")
    if not isinstance(contract, list) or not contract:
        raise ValueError("lidar_input_contract requires a nonempty list")
    names = []
    for item in contract:
        if (not isinstance(item, dict) or set(item) != {"name", "shape", "dtype"}
                or not isinstance(item["name"], str) or not item["name"]
                or item["dtype"] not in ("float32", "int32") or not isinstance(item["shape"], list)
                or not item["shape"]):
            raise ValueError("Invalid compatible input contract record")
        names.append(item["name"])
        for axis, dimension in enumerate(item["shape"]):
            if dimension is None and item["name"] == "camera_points" and axis == 0:
                continue
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
                raise ValueError("Only camera_points may have a dynamic point dimension")
    if len(names) != len(set(names)) or names[:4] != ["images", "points", "point_indices", "inverse_counts"]:
        raise ValueError("Compatible input contract requires unique names and the canonical four leading inputs")
    by_name = {item["name"]: item for item in contract}
    for name, expected, dtype in (("points", [profile.point_capacity, 5], "float32"),
                                  ("point_indices", [profile.voxel_capacity, 10], "int32"),
                                  ("inverse_counts", [profile.voxel_capacity, 1], "float32")):
        if by_name[name]["shape"] != expected or by_name[name]["dtype"] != dtype:
            raise ValueError(f"{name} does not match the capacity profile")
    image = by_name["images"]
    if image["dtype"] != "float32" or len(image["shape"]) != 5 or image["shape"][:3] != [1, 6, 3]:
        raise ValueError("Compatible images require normalized FP32 [1,6,3,H,W]")
    task, mode, pool = metadata.get("task"), metadata.get("calibration_mode"), metadata.get("pool_backend")
    if task not in ("det", "seg"):
        raise ValueError("Compatible ONNX requires task='det' or 'seg'")
    fixed = None
    if mode == "fixed":
        if pool not in ("reference", "hat", "compatible") or metadata.get("pooling_precomputed") != "true":
            raise ValueError("Fixed compatible ONNX requires a declared precomputed pooling backend")
        fixed = _json_record(metadata, "fixed_calibration")
        if not isinstance(fixed, dict) or set(fixed) != set(SENSOR_NAMES[2:]):
            raise ValueError("fixed_calibration must contain exactly the three camera matrices")
        try:
            _fixed_calibration(tuple(torch.as_tensor(fixed[name], dtype=torch.float32) for name in SENSOR_NAMES[2:]), "cpu")
        except (TypeError, RuntimeError) as exc:
            raise ValueError("Invalid fixed calibration") from exc
    elif mode != "dynamic" or pool != "reference" or metadata.get("pooling_precomputed") != "false":
        raise ValueError("Dynamic camera geometry requires reference pooling without precomputation")
    expected_tail = (["camera_points"] if task == "det" else []) + (list(SENSOR_NAMES[2:]) if mode == "dynamic" else [])
    tail_start = names.index("dense_lookup") + 1 if "dense_lookup" in names else 0
    if tail_start < 5 or names[tail_start:] != expected_tail:
        raise ValueError("Compatible graph requires dense_lookup followed by its task/calibration inputs")
    sparse_names = names[4:tail_start - 1]
    if not sparse_names or len(sparse_names) % 2 or sparse_names != [n for i in range(len(sparse_names)//2) for n in (f"neighbors_{i}", f"mask_{i}")]:
        raise ValueError("Sparse plan inputs must be consecutive neighbor/mask pairs")
    for index in range(len(sparse_names)//2):
        neighbor, mask = by_name[f"neighbors_{index}"], by_name[f"mask_{index}"]
        if (neighbor["dtype"] != "int32" or len(neighbor["shape"]) != 2
                or neighbor["shape"][1] != profile.sparse_capacity or mask["dtype"] != "float32"
                or mask["shape"] != [profile.sparse_capacity, 1]):
            raise ValueError("Sparse plan tensor types/capacities differ from the profile")
    if by_name["dense_lookup"]["dtype"] != "int32" or len(by_name["dense_lookup"]["shape"]) != 1:
        raise ValueError("dense_lookup requires an INT32 vector")
    if task == "det" and (by_name["camera_points"]["shape"] != [None, 5] or by_name["camera_points"]["dtype"] != "float32"):
        raise ValueError("camera_points must contain dynamic FP32 raw [P,5] points")
    if mode == "dynamic":
        for name, dimension in zip(SENSOR_NAMES[2:], (3, 4, 4)):
            if by_name[name]["dtype"] != "float32" or by_name[name]["shape"] != [1, 6, dimension, dimension]:
                raise ValueError("Runtime camera matrices have invalid shapes or types")
    return {"profile": profile, "geometry": geometry, "inputs": contract, "calibration_mode": mode,
            "fixed_calibration": fixed, "task": task, "pool_backend": pool}


def prepare_compatible_feed(metadata, sensor_feed, *, backbone=None):
    """Build current-frame runtime inputs without loading checkpoint weights.

    ``sensor_feed`` uses the ordinary five normalized FP32 sensor inputs. Fixed
    camera geometry is checked before dropping matrices from the returned feed.
    Capacity overflow raises; neither the cloud nor its sparse sites are cropped.
    """
    if metadata.get("lidar_backend") != "compatible":
        raise ValueError("Compatible planning requires lidar_backend='compatible'")
    if metadata.get("cpu_planning_required") != "true" or metadata.get("plan_tensors") != "runtime_inputs":
        raise ValueError("Compatible ONNX must declare per-frame CPU planning and runtime plan inputs")
    validated = validate_compatible_metadata(metadata)
    profile, geometry, contract = (validated[name] for name in ("profile", "geometry", "inputs"))
    if not isinstance(geometry, dict) or not isinstance(contract, list) or not contract:
        raise ValueError("Compatible ONNX geometry/input contract must be explicit mappings/lists")
    values = _sensor_tensors(sensor_feed, contract[0].get("shape"))
    images, points, *matrices = values
    mode, pool = metadata.get("calibration_mode"), metadata.get("pool_backend")
    if mode == "fixed":
        if pool not in ("reference", "hat", "compatible") or metadata.get("pooling_precomputed") != "true":
            raise ValueError("Compatible fixed ONNX must declare a precomputed pooling backend")
        bound = _json_record(metadata, "fixed_calibration")
        if not isinstance(bound, dict) or set(bound) != set(SENSOR_NAMES[2:]):
            raise ValueError("fixed_calibration must contain exactly the three camera matrices")
        expected = _fixed_calibration(tuple(torch.as_tensor(bound[name], dtype=torch.float32)
                                            for name in SENSOR_NAMES[2:]), "cpu")
        for name, actual, wanted in zip(SENSOR_NAMES[2:], matrices, expected):
            if not torch.allclose(actual, wanted, atol=1e-6, rtol=1e-6):
                raise ValueError(f"Runtime {name} differs from fixed calibration; re-export for changed calibration")
    elif mode != "dynamic" or pool != "reference":
        raise ValueError("Dynamic compatible ONNX camera geometry supports reference pooling only")
    try:
        shape = tuple(int(size) for size in geometry["sparse_shape"])
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError("sparse_shape must have three positive dimensions")
        backbone = _planning_backbone(shape) if backbone is None else backbone
        if tuple(backbone.sparse_shape) != shape or _layout(backbone) != geometry["convolutions"]:
            raise ValueError("Sparse encoder architecture differs from the deployment geometry")
        voxel = build_voxel_plan(points, voxel_size=geometry["voxel_size"],
            point_cloud_range=geometry["point_cloud_range"], max_voxels=geometry["max_voxels"],
            max_points=10, point_capacity=profile.point_capacity, voxel_capacity=profile.voxel_capacity)
        topology = _topology(backbone, voxel, profile)
    except KeyError as exc:
        raise ValueError(f"Missing compatible LiDAR geometry field: {exc}") from exc
    tensors = OrderedDict(images=images)
    tensors.update(zip(("points", "point_indices", "inverse_counts"), voxel.tensor_feed("cpu")))
    tensors.update(_plan_tensors(backbone, topology))
    if metadata.get("task") == "det":
        tensors["camera_points"] = points
    elif metadata.get("task") != "seg":
        raise ValueError("Compatible ONNX requires task='det' or 'seg'")
    if mode == "dynamic":
        tensors.update(zip(SENSOR_NAMES[2:], matrices))
    result = OrderedDict((name, np.ascontiguousarray(value.numpy())) for name, value in tensors.items())
    _validate_contract(result, contract)
    return result


class CompatibleOnnxBEVFusion(nn.Module):
    """Complete ordinary ONNX with compatible LiDAR and source task outputs."""
    def __init__(self, model, values, *, profile: DeploymentProfile, pool_backend="reference",
                 pool_num_points=10, score_threshold=None, fixed_calibration=False):
        super().__init__()
        if model.training:
            raise ValueError("Compatible ONNX export requires model.eval()")
        if not isinstance(profile, DeploymentProfile):
            raise ValueError("profile must be DeploymentProfile")
        if pool_backend not in ("reference", "hat", "compatible"):
            raise ValueError("Compatible ONNX pooling must be reference, hat or compatible")
        if pool_backend != "reference" and not fixed_calibration:
            raise ValueError(f"{pool_backend} pooling requires explicitly fixed calibration")
        if len(values) != 5:
            raise ValueError("values must contain the five normalized sensor tensors")
        self.task = getattr(model, "task", None)
        if self.task not in ("det", "seg"):
            raise ValueError("model.task must be det or seg")
        self.profile, self.pool_backend = profile, pool_backend
        self.fixed_calibration = bool(fixed_calibration)
        original = model.encoders["lidar"]["backbone"]
        self.geometry = {"voxel_size": list(model.voxel_size),
                         "point_cloud_range": list(model.point_cloud_range),
                         "max_voxels": int(model.max_voxels),
                         "sparse_shape": list(original.sparse_shape), "convolutions": _layout(original)}
        points = values[1].detach().cpu().float()
        voxel = build_voxel_plan(points, voxel_size=model.voxel_size, point_cloud_range=model.point_cloud_range,
            max_voxels=model.max_voxels, max_points=10, point_capacity=profile.point_capacity,
            voxel_capacity=profile.voxel_capacity)
        topology = _topology(original, voxel, profile)
        self.voxel = CompatibleVoxelMean(profile.point_capacity, profile.voxel_capacity, 10)
        self.lidar = CompatibleSparseEncoder(original, topology, page_size=profile.page_size)
        self.camera = OnnxCamera(model.encoders["camera"], model.point_cloud_range,
            pool_backend="reference" if pool_backend == "compatible" else pool_backend,
            pool_num_points=pool_num_points, fixed_calibration=tuple(values[2:]) if fixed_calibration else None)
        if pool_backend == "compatible":
            from bevfusion.operators.pooling.compatible import CompatibleBEVPool
            vt = self.camera.vtransform
            self.camera.pool = CompatibleBEVPool.from_transform(vt, self.camera.fixed_camera2lidar,
                self.camera.fixed_camera_intrinsics, self.camera.fixed_img_aug_matrix,
                torch.eye(4, device=vt.frustum.device)[None])
            self.camera.pool_backend = "compatible"
        self.fuser, self.decoder = model.fuser, model.decoder
        self.quantize_decoder = bool(getattr(model, "quantize_decoder", False))
        if self.task == "det":
            if score_threshold is None:
                score_threshold = model.config["model"]["heads"]["object"]["bbox_coder"]["score_threshold"]
            self.head = OnnxTransFusionHead(model.heads["object"], score_threshold)
        else:
            self.head = model.heads["map"]
        self.input_specs = OrderedDict()
        tensors = OrderedDict(images=values[0], points=voxel.padded_points,
                              point_indices=voxel.point_indices, inverse_counts=voxel.inverse_counts)
        tensors.update(_plan_tensors(original, topology))
        if self.task == "det":
            tensors["camera_points"] = points
        if not fixed_calibration:
            tensors.update(zip(SENSOR_NAMES[2:], values[2:]))
        for name, tensor in tensors.items():
            self.input_specs[name] = {"shape": [None, 5] if name == "camera_points" else list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch.")}
        self.input_names = list(tensors)
        self.output_names = ["probabilities"] if self.task == "seg" else ["boxes_3d", "scores", "labels"]
        self.example_inputs = tuple(tensor.to(device=values[0].device) for tensor in tensors.values())
        self.training = False

    def deployment_metadata(self):
        result = {"task": self.task, "lidar_backend": "compatible",
                  "lidar_profile": json.dumps(self.profile.to_dict()), "lidar_geometry": json.dumps(self.geometry),
                  "lidar_input_contract": json.dumps([{"name": name, **spec} for name, spec in self.input_specs.items()]),
                  "cpu_planning_required": "true", "plan_tensors": "runtime_inputs",
                  "pool_backend": self.pool_backend, "calibration_mode": "fixed" if self.fixed_calibration else "dynamic",
                  "pooling_precomputed": str(self.fixed_calibration).lower()}
        if self.fixed_calibration:
            result["fixed_calibration"] = json.dumps({name: getattr(self.camera, "fixed_" + name).detach().cpu().tolist()
                                                     for name in SENSOR_NAMES[2:]})
        return result

    def pool_statistics(self):
        return ({"backend": self.pool_backend, **self.camera.pool.statistics()}
                if self.camera.pool is not None else None)

    def prepare_feed(self, sensor_feed):
        return prepare_compatible_feed(self.deployment_metadata(), sensor_feed)

    def forward(self, *inputs):
        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()) and len(inputs) != len(self.input_names):
            raise ValueError(f"Expected {len(self.input_names)} compatible ONNX inputs")
        values = dict(zip(self.input_names, inputs))
        features = self.voxel(values["points"], values["point_indices"], values["inverse_counts"])
        lidar_bev = self.lidar(features, *(values[name] for name in self.lidar.input_names[1:]))
        matrices = tuple(getattr(self.camera, "fixed_" + name) if self.fixed_calibration else values[name]
                         for name in SENSOR_NAMES[2:])
        # Projection must use raw points, not zero-padded voxel-mean inputs.
        camera_bev = self.camera(values["images"], *matrices, points=values.get("camera_points"))
        fused = self.fuser(torch.cat((camera_bev, lidar_bev), dim=1))
        decoded = self.decoder["neck"](self.decoder["backbone"](fused))[0]
        if self.quantize_decoder:
            decoded = decoded.half().float()
        return self.head(decoded)


__all__ = ["CompatibleOnnxBEVFusion", "prepare_compatible_feed", "validate_compatible_metadata"]
