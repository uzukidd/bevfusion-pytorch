"""Static BEVFusion compute graph with explicit per-frame deployment inputs.

Grouping, rulebooks, LiDAR depth projection and detection filtering belong to the
application. This module never embeds a frame's sparse topology. Camera geometry
is intentionally fixed to the supplied calibration. Standard ONNX export alone
does not establish that the full model compiles or executes on a target BPU.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F

from bevfusion.operators.pooling.compatible import CompatibleBEVPool
from bevfusion.operators.sparse_conv.compatible import CompatibleSparseEncoder
from bevfusion.operators.voxelization.compatible import CompatibleVoxelMean
from bevfusion.operators.pooling.hat import HATLSSPool
from bevfusion.operators.sparse_conv.reference import ReferenceSparseConv
from tools.onnx_model import OnnxTransFusionHead, _fixed_calibration, atan2


BOX_COLUMNS = ["x", "y", "z_bottom", "size_x", "size_y", "size_z", "yaw", "vx", "vy"]


def _threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score_threshold must be a finite number in [0,1]")
    return float(value)


def _detection_contract(model):
    head = model.heads["object"]
    configured = model.config["model"]["heads"]["object"]["bbox_coder"]["score_threshold"]
    origin = head.decoded_z_origin
    if origin not in ("center", "bottom"):
        raise ValueError("decoded_z_origin must be center or bottom")
    bounds = head.post_center_range
    if bounds is not None:
        bounds = [float(value) for value in bounds]
        if len(bounds) != 6 or not all(math.isfinite(value) for value in bounds) or any(bounds[i] > bounds[i + 3] for i in range(3)):
            raise ValueError("post_center_range must contain six finite ordered bounds")
    return {"box_origin": "bottom", "box_format_version": 2, "box_columns": BOX_COLUMNS.copy(),
            "decoded_z_origin": origin, "checkpoint_decoded_z_origin": origin,
            "post_center_range": bounds, "score_threshold": _threshold(configured),
            "classes": list(model.classes), "num_proposals": int(head.num_proposals),
            "graph_labels_dtype": "int32", "application_labels_dtype": "int64",
            "score_comparison": ">", "range_comparison": "inclusive", "nms": None}


class _StaticDetectionHead(OnnxTransFusionHead):
    """Fixed proposal count, physical bottom-Z boxes, no data-dependent filtering."""
    def decode(self, predictions):
        one_hot = F.one_hot(predictions["query_labels"], self.num_classes).permute(0, 2, 1)
        class_scores = predictions["heatmap"].sigmoid() * predictions["query_heatmap_score"] * one_hot
        scores, labels = class_scores.max(dim=1)
        center = predictions["center"]
        xy = torch.stack((center[:, 0] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0],
                          center[:, 1] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]), dim=1)
        dimensions = predictions["dim"].exp()
        # Preserve the released checkpoint's legacy decode, then repair origin.
        decoded_height = predictions["height"] - dimensions[:, 2:3] * 0.5
        bottom = decoded_height - dimensions[:, 2:3] * 0.5 if self.decoded_z_origin == "center" else decoded_height
        rotation = atan2(predictions["rot"][:, 0:1], predictions["rot"][:, 1:2])
        boxes = torch.cat((xy, bottom, dimensions, rotation, predictions["vel"]), dim=1)[0].transpose(0, 1)
        return (boxes.reshape(self.num_proposals, 9), scores[0].reshape(self.num_proposals),
                labels[0].to(dtype=torch.int32).reshape(self.num_proposals))


class StaticDeploymentModel(nn.Module):
    """One static compute graph, with all sparse indices supplied every frame.

    Input order is public in ``input_names`` and ``input_specs``. ``self.sparse``
    exposes ``plan_inputs(new_encoder_plan, device)`` to pack new CPU rulebooks.
    If spconv runs on CPU, voxel_backend describes application preprocessing only
    and the graph consumes dense lidar_bev. The det graph always consumes an
    application-projected lidar_depth map. No raw point projection is traced.
    """
    def __init__(self, model, calibration, *, pool_backend="hat", pool_num_points=10,
                 voxel_backend="compatible", spconv_backend="compatible",
                 voxel_plan=None, encoder_plan=None, page_size=32768):
        super().__init__()
        if model.training:
            raise ValueError("StaticDeploymentModel requires an eval() source model")
        if pool_backend not in ("hat", "compatible"):
            raise ValueError("pool_backend must be hat or compatible")
        if voxel_backend not in ("compatible", "cpu"):
            raise ValueError("voxel_backend must be compatible or cpu")
        if spconv_backend not in ("compatible", "cpu"):
            raise ValueError("spconv_backend must be compatible or cpu")
        if isinstance(pool_num_points, bool) or not isinstance(pool_num_points, int) or not 1 <= pool_num_points <= 1024:
            raise ValueError("pool_num_points must be an integer in [1,1024]")
        self.task = getattr(model, "task", None)
        if self.task not in ("det", "seg"):
            raise ValueError("model.task must be det or seg")
        self.pool_backend, self.pool_num_points = pool_backend, pool_num_points
        self.voxel_backend, self.spconv_backend = voxel_backend, spconv_backend
        camera = model.encoders["camera"]
        vt = camera["vtransform"]
        device = vt.frustum.device
        intrinsic, extrinsic, augmentation = _fixed_calibration(calibration, device)
        # Persist calibration as a contract, not live geometry-discovery inputs.
        for name, value in zip(("camera_intrinsics", "camera2lidar", "img_aug_matrix"),
                               (intrinsic, extrinsic, augmentation)):
            self.register_buffer(name, value)
        self.image_size = tuple(vt.image_size)
        self.feature_size = tuple(vt.feature_size)
        self.depth_channels, self.camera_channels = int(vt.D), int(vt.C)
        self.backbone, self.neck = camera["backbone"], camera["neck"]
        self.depthnet, self.downsample = vt.depthnet, vt.downsample
        self.depth_conditioned = hasattr(vt, "dtransform")
        if self.depth_conditioned != (self.task == "det"):
            raise ValueError("det requires DepthLSSTransform and seg requires LSSTransform")
        self.dtransform = vt.dtransform if self.depth_conditioned else None
        with torch.no_grad():
            arguments = (vt, extrinsic, intrinsic, augmentation, torch.eye(4, device=device)[None])
            self.pool = (HATLSSPool.from_transform(*arguments, num_points=pool_num_points) if pool_backend == "hat"
                         else CompatibleBEVPool.from_transform(*arguments))
        self.voxel, self.sparse = None, None
        self.input_names = ["images"]
        self.input_specs = {"images": {"shape": [1, 6, 3, *self.image_size], "dtype": "float32"}}
        original = model.encoders["lidar"]["backbone"]
        if spconv_backend == "compatible":
            if voxel_plan is None or encoder_plan is None:
                raise ValueError("Compatible spconv requires voxel_plan and encoder_plan capacity profiles")
            if int(encoder_plan.input_capacity) != int(voxel_plan.voxel_capacity):
                raise ValueError("Voxel and encoder input capacities must match")
            self.sparse = CompatibleSparseEncoder(original, encoder_plan, page_size=page_size)
            if voxel_backend == "compatible":
                self.voxel = CompatibleVoxelMean(voxel_plan.point_capacity, voxel_plan.voxel_capacity,
                                                  voxel_plan.max_points)
                self._input("points", (voxel_plan.point_capacity, 5), "float32")
                self._input("point_indices", (voxel_plan.voxel_capacity, voxel_plan.max_points), "int32")
                self._input("inverse_counts", (voxel_plan.voxel_capacity, 1), "float32")
            else:
                self._input("voxel_features", (voxel_plan.voxel_capacity, 5), "float32")
            # Only inspect tensor dimensions/dtypes; never register sample indices.
            for name, value in zip(self.sparse.input_names[1:], self.sparse.plan_inputs(encoder_plan)):
                self._input(name, tuple(value.shape), "int32" if value.dtype == torch.int32 else "float32")
            self.lidar_bev_shape = (1, int(original.conv_out[0].weight.shape[-1]) * encoder_plan.final_shape[2],
                                    *tuple(encoder_plan.final_shape[:2]))
        else:
            shape = tuple(int(value) for value in original.sparse_shape)
            channels = int(original.conv_out[0].weight.shape[-1])
            final = shape
            for module in original.modules():
                # Topology is unavailable here; real source profiles have fixed
                # stride reductions. Derive dense output from encoder convolution metadata.
                if isinstance(module, ReferenceSparseConv) and not module.subm:
                    final = tuple((size + 2 * pad - kernel) // stride + 1 for size, pad, kernel, stride in
                                  zip(final, module.padding, module.kernel, module.stride))
            self.lidar_bev_shape = (1, channels * final[2], final[0], final[1])
            self._input("lidar_bev", self.lidar_bev_shape, "float32")
        if self.depth_conditioned:
            self._input("lidar_depth", (1, 6, 1, *self.image_size), "float32")
        self.fuser, self.decoder = model.fuser, model.decoder
        self.quantize_decoder = bool(getattr(model, "quantize_decoder", False))
        self.detection_contract = _detection_contract(model) if self.task == "det" else None
        self.head = _StaticDetectionHead(model.heads["object"], score_threshold=0.0) if self.task == "det" else model.heads["map"]
        self.output_names = ["boxes_3d", "scores", "labels"] if self.task == "det" else ["probabilities"]
        # Source modules are required to be in eval mode, so no recursive .eval()
        # is needed (and the constructor never changes source training states).
        self.training = False

    def _input(self, name, shape, dtype):
        self.input_names.append(name)
        self.input_specs[name] = {"shape": [int(value) for value in shape], "dtype": dtype}

    def _camera(self, images, lidar_depth):
        h, w = self.feature_size
        feature = self.neck(self.backbone(images.reshape(6, 3, *self.image_size)))[0].float()
        if self.depth_conditioned:
            feature = torch.cat((self.dtransform(lidar_depth.reshape(6, 1, *self.image_size)), feature), dim=1)
        prediction = self.depthnet(feature).reshape(1, 6, self.depth_channels + self.camera_channels, h, w)
        depth = prediction[:, :, :self.depth_channels].softmax(dim=2)
        channels = prediction[:, :, self.depth_channels:]
        return self.downsample(self.pool(depth, channels).contiguous())

    def forward(self, *inputs):
        exporting = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
        if not exporting:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("StaticDeploymentModel is inference-only; use eval() and inference_mode()")
            if len(inputs) != len(self.input_names):
                raise ValueError(f"Expected {len(self.input_names)} inputs in order {self.input_names}")
            device = self.camera_intrinsics.device
            for name, value in zip(self.input_names, inputs):
                spec = self.input_specs[name]
                dtype = torch.int32 if spec["dtype"] == "int32" else torch.float32
                if not isinstance(value, torch.Tensor) or list(value.shape) != spec["shape"] or value.dtype != dtype or value.device != device:
                    raise ValueError(f"{name} requires {spec['dtype']} {spec['shape']} on {device}")
        values = dict(zip(self.input_names, inputs))
        if self.spconv_backend == "cpu":
            lidar_bev = values["lidar_bev"]
        else:
            features = (self.voxel(values["points"], values["point_indices"], values["inverse_counts"])
                        if self.voxel_backend == "compatible" else values["voxel_features"])
            lidar_bev = self.sparse(features, *(values[name] for name in self.sparse.input_names[1:]))
        camera_bev = self._camera(values["images"], values.get("lidar_depth"))
        fused = self.fuser(torch.cat((camera_bev, lidar_bev), dim=1))
        decoded = self.decoder["neck"](self.decoder["backbone"](fused))[0]
        if self.quantize_decoder:
            decoded = decoded.half().float()
        return self.head(decoded)


def postprocess_detection(outputs, contract_or_model, score_threshold=None):
    """Restore source filtering on CPU; graph outputs all fixed proposals.

    Returns boxes_3d FP32, scores FP32, labels INT64. Range filtering uses the
    legacy pre-bottom-repair Z (inclusive bounds), scores use a strict > test
    when the configured threshold is nonzero. No NMS is introduced.
    """
    if isinstance(contract_or_model, Mapping):
        contract = contract_or_model
    elif isinstance(contract_or_model, StaticDeploymentModel):
        contract = contract_or_model.detection_contract
    else:
        contract = _detection_contract(contract_or_model)
    if not isinstance(contract, Mapping) or contract.get("box_origin") != "bottom" or contract.get("box_format_version") != 2:
        raise ValueError("Detection postprocess requires a physical bottom-Z v2 contract")
    if isinstance(outputs, Mapping):
        outputs = tuple(outputs[name] for name in ("boxes_3d", "scores", "labels"))
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise ValueError("Detection outputs must be boxes_3d, scores, labels")
    boxes, scores, labels = (torch.as_tensor(value).detach().cpu() for value in outputs)
    if (boxes.dtype != torch.float32 or scores.dtype != torch.float32 or labels.dtype not in (torch.int32, torch.int64)
            or boxes.ndim != 2 or boxes.shape[1] != 9 or scores.shape != boxes.shape[:1] or labels.shape != scores.shape):
        raise ValueError("Detection outputs require FP32 [N,9], FP32 [N], integer [N]")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("Detection outputs must be finite")
    threshold = _threshold(contract.get("score_threshold", 0.0) if score_threshold is None else score_threshold)
    keep = torch.ones_like(scores, dtype=torch.bool)
    bounds = contract.get("post_center_range")
    if bounds is not None:
        bounds = torch.as_tensor(bounds, dtype=boxes.dtype)
        if bounds.shape != (6,) or not torch.isfinite(bounds).all() or (bounds[:3] > bounds[3:]).any():
            raise ValueError("post_center_range must contain six finite ordered bounds")
        origin = contract.get("decoded_z_origin", contract.get("checkpoint_decoded_z_origin"))
        if origin not in ("center", "bottom"):
            raise ValueError("decoded_z_origin must be center or bottom")
        z = boxes[:, 2] + boxes[:, 5] * 0.5 if origin == "center" else boxes[:, 2]
        position = torch.stack((boxes[:, 0], boxes[:, 1], z), dim=1)
        keep &= (position >= bounds[:3]).all(1) & (position <= bounds[3:]).all(1)
    if threshold:
        keep &= scores > threshold
    return {"boxes_3d": boxes[keep], "scores": scores[keep], "labels": labels[keep].long()}


__all__ = ["StaticDeploymentModel", "postprocess_detection"]
