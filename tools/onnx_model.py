"""Standard-operator ONNX camera geometry and BEVFusion inference adapters.

Inputs are normalized/cropped RGB, raw points and per-frame calibration.
中文：以标准 ONNX 算子实现动态点云、深度投影及检测/分割推理。
"""
import torch
from torch import nn
from torch.nn import functional as F

from .onnx_lidar import OnnxLidarEncoder, voxelize


def inverse3(matrix):
    """Standard-op 3x3 inverse with runtime camera-intrinsic specialization."""
    a, b, c = matrix.unbind(-2)

    def cross(x, y):
        return torch.stack((x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1],
                            x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2],
                            x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]), -1)

    first, second, third = cross(b, c), cross(c, a), cross(a, b)
    determinant = (a * first).sum(-1)
    general_inverse = torch.stack((first, second, third), -1) / determinant[..., None, None]
    # For K=[[fx,0,cx],[0,fy,cy],[0,0,1]], match the source LU inverse's
    # reciprocal-then-multiply rounding. Direct -cx/fx differs by one FP32
    # ULP for common calibrations (500,352), enough to cross BEV voxel edges.
    # Select per matrix at runtime; skewed/general matrices keep the adjugate.
    canonical = ((a[..., 1] == 0) & (b[..., 0] == 0) & (c[..., 0] == 0)
                 & (c[..., 1] == 0) & (c[..., 2] == 1))
    one, zero = torch.ones_like(a[..., 0]), torch.zeros_like(a[..., 0])
    # Unused specialized denominators must also be safe for invertible general
    # matrices with zero diagonal entries.
    inverse_fx = torch.where(canonical, a[..., 0], one).reciprocal()
    inverse_fy = torch.where(canonical, b[..., 1], one).reciprocal()
    canonical_inverse = torch.stack((
        inverse_fx, zero, -a[..., 2] * inverse_fx,
        zero, inverse_fy, -b[..., 2] * inverse_fy,
        zero, zero, one), -1).reshape_as(matrix)
    return torch.where(canonical[..., None, None], canonical_inverse, general_inverse)


def project_lidar_depth(points, camera2lidar, camera_intrinsics, img_aug_matrix,
                        image_size=(256, 704), point_cloud_range=None):
    """ONNX-safe scalar depth; the last input point at each pixel wins.

    LiDAR augmentation is identity, as in the public inference adapter. Keep
    the source projector's clamped camera Z, including points behind a camera.
    A dummy point and a sentinel pixel keep every Gather valid at P=0.
    """
    height, width = image_size
    points = points.float()
    if point_cloud_range is not None:
        lower = points.new_tensor(point_cloud_range[:3])
        upper = points.new_tensor(point_cloud_range[3:])
        points = points[((points[:, :3] >= lower) & (points[:, :3] < upper)).all(1)]
    xyz = torch.cat((points[:, :3], points.new_zeros((1, 3))), dim=0).T
    camera = camera2lidar[0].float()
    rotation = inverse3(camera[:, :3, :3])
    translation = -(rotation @ camera[:, :3, 3, None])
    intrinsic = camera_intrinsics[0, :, :3, :3].float()
    lidar2image_rotation = intrinsic @ rotation
    lidar2image_translation = intrinsic @ translation
    projected = lidar2image_rotation @ xyz + lidar2image_translation
    distance = projected[:, 2].clamp(1e-5, 1e5)
    normalized = torch.cat((projected[:, :2] / distance[:, None], distance[:, None]), dim=1)
    augmentation = img_aug_matrix[0].float()
    projected = augmentation[:, :3, :3] @ normalized + augmentation[:, :3, 3, None]
    u, v = projected[:, 0], projected[:, 1]
    real_point = torch.cat((torch.ones_like(points[:, 0], dtype=torch.bool),
                            torch.zeros(1, device=points.device, dtype=torch.bool)))
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height) & real_point[None]
    num_cameras = camera.shape[0]
    offsets = torch.arange(num_cameras, device=points.device)[:, None] * (height * width)
    pixels = v.long() * width + u.long() + offsets
    pixels = torch.where(valid, pixels, num_cameras * height * width)
    indices = torch.arange(distance.numel(), device=points.device, dtype=torch.int64) + 1
    indices = torch.where(valid.reshape(-1), indices, torch.zeros_like(indices))
    last = torch.zeros(num_cameras * height * width + 1, device=points.device, dtype=torch.int64)
    last = last.scatter_reduce(0, pixels.reshape(-1), indices, reduce="amax", include_self=True)
    values = torch.cat((distance.new_zeros(1), distance.reshape(-1)))
    return values[last[:-1]].reshape(1, num_cameras, 1, height, width)


def atan2(y, x):
    """Standard Atan/Where decomposition, including zero and signed-zero axes."""
    negative_y = (y < 0) | ((y == 0) & (y.reciprocal() < 0))
    negative_x = (x < 0) | ((x == 0) & (x.reciprocal() < 0))
    signed_pi = torch.where(negative_y, -torch.ones_like(y) * torch.pi,
                            torch.ones_like(y) * torch.pi)
    angle = torch.atan(y / torch.where(x == 0, torch.ones_like(x), x))
    angle = torch.where(negative_x, angle + signed_pi, angle)
    angle = torch.where((x == 0) & (y != 0), signed_pi * 0.5, angle)
    # ORT's vectorized Where can canonicalize -0 to +0. Restore the sign
    # with multiplication after all selections, including atan2(-0, +0).
    return angle.abs() * torch.where(negative_y, -torch.ones_like(y), torch.ones_like(y))


def _fixed_calibration(calibration, device):
    """Clone the explicit six-camera contract; never synthesize calibration."""
    if not isinstance(calibration, tuple) or len(calibration) != 3:
        raise ValueError("ONNX precomputed pooling requires fixed_calibration=(K, camera2lidar, img_aug_matrix)")
    result = []
    for value, name, size in zip(calibration, ("camera_intrinsics", "camera2lidar", "img_aug_matrix"), (3, 4, 4)):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 6, size, size):
            raise ValueError(f"fixed {name} must have shape [1,6,{size},{size}]")
        value = value.detach().to(device=device, dtype=torch.float32).clone()
        if not torch.isfinite(value).all() or (torch.linalg.det(value[..., :3, :3]).abs() < 1e-8).any():
            raise ValueError(f"fixed {name} must be finite and invertible")
        if size == 4 and not torch.allclose(value[..., 3, :], value.new_tensor([0., 0., 0., 1.]).expand_as(value[..., 3, :]), atol=1e-6, rtol=0):
            raise ValueError(f"fixed {name} must be homogeneous with last row [0,0,0,1]")
        result.append(value)
    return tuple(result)


class _OnnxReferencePool(nn.Module):
    """All-contribution reference pooling with geometry-only constant indices.

    Keep each source camera/depth chunk and its original valid-point ordering.
    New depth probabilities and image features are gathered on every invocation;
    only geometry-derived indices are retained. Scatter-add preserves duplicate
    cell contributions without the ONNX duplicate-index caveat of index_add_.
    """
    def __init__(self, transform, calibration):
        super().__init__()
        from bevfusion.operators.pooling.reference import build_pool_plan

        intrinsic, extrinsic, augmentation = calibration
        with torch.no_grad():
            geometry = transform.get_geometry(extrinsic, intrinsic, augmentation,
                                              torch.eye(4, device=intrinsic.device)[None])
            coordinates = transform._voxel_indices(geometry)
            plan = build_pool_plan(coordinates, tuple(int(value) for value in transform.nx),
                                   transform.pool_depth_chunk)
        self.shape, self.batch_size, self.pixels = plan.shape, plan.batch_size, plan.pixels
        self.channels = int(transform.C)
        self.chunks = tuple((chunk.batch, chunk.camera, chunk.start, chunk.end)
                            for chunk in plan.chunks)
        self.valid_points = sum(chunk.indices.numel() for chunk in plan.chunks)
        for index, chunk in enumerate(plan.chunks):
            self.register_buffer(f"indices_{index}", chunk.indices.detach().clone())
            self.register_buffer(f"ranks_{index}", chunk.ranks.detach().clone())

    def statistics(self):
        return {"backend": "reference", "precomputed": True, "approximate": False,
                "operator": "Gather/Mul/ScatterElements(add)", "groups": len(self.chunks),
                "chunks": len(self.chunks), "valid_points": self.valid_points,
                "selected_points": self.valid_points, "dropped_points": 0,
                "grid_shape": list(self.shape), "normalization": "none",
                "index_bytes": sum(value.numel() * value.element_size() for value in self.buffers())}

    def forward(self, depth, features):
        nx, ny, nz = self.shape
        volume = nz * nx * ny
        output = features.new_zeros((self.batch_size * volume, self.channels), dtype=torch.float32)
        image_features = features.flatten(3).permute(0, 1, 3, 2)
        for index, (batch, camera, start, end) in enumerate(self.chunks):
            selected = getattr(self, f"indices_{index}")
            ranks = getattr(self, f"ranks_{index}") + batch * volume
            probability = depth[batch, camera, start:end].reshape(-1)[selected]
            values = probability[:, None] * image_features[batch, camera][selected % self.pixels]
            output = output.scatter_add(0, ranks[:, None].expand(-1, self.channels), values.float())
        return output.reshape(self.batch_size, nz, nx, ny, self.channels).permute(0, 1, 4, 2, 3).reshape(
            self.batch_size, nz * self.channels, nx, ny)


class OnnxCamera(nn.Module):
    def __init__(self, camera, point_cloud_range=None, *, pool_backend="reference",
                 pool_num_points=10, fixed_calibration=None):
        super().__init__()
        if pool_backend not in ("reference", "hat"):
            raise ValueError("ONNX camera pool_backend must be reference or hat")
        self.backbone = camera["backbone"]
        self.neck = camera["neck"]
        self.vtransform = camera["vtransform"]
        self.point_cloud_range = point_cloud_range
        # Architecture metadata is fixed; calibration and points remain inputs.
        self.bev_shape = tuple(int(value) for value in self.vtransform.nx)
        self.depth_conditioned = hasattr(self.vtransform, "dtransform")
        self.pool_backend = pool_backend
        self.fixed_calibration = pool_backend == "hat" or fixed_calibration is not None
        self.pool = None
        if self.fixed_calibration:
            intrinsic, extrinsic, augmentation = _fixed_calibration(
                fixed_calibration, self.vtransform.frustum.device)
            for name, value in zip(("fixed_camera_intrinsics", "fixed_camera2lidar", "fixed_img_aug_matrix"),
                                   (intrinsic, extrinsic, augmentation)):
                self.register_buffer(name, value)
            with torch.no_grad():
                if pool_backend == "hat":
                    from bevfusion.operators.pooling.hat import HATLSSPool
                    self.pool = HATLSSPool.from_transform(
                        self.vtransform, extrinsic, intrinsic, augmentation,
                        torch.eye(4, device=intrinsic.device)[None], num_points=pool_num_points)
                else:
                    self.pool = _OnnxReferencePool(self.vtransform, (intrinsic, extrinsic, augmentation))

    def forward(self, images, camera_intrinsics, camera2lidar, img_aug_matrix, points=None):
        vt = self.vtransform
        num_cameras = images.shape[1]
        feature = self.neck(self.backbone(images.reshape(-1, 3, *vt.image_size)))[0]
        _, channels, height, width = feature.shape
        feature = feature.reshape(1, num_cameras, channels, height, width)
        if self.fixed_calibration:
            if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
                for value, bound in zip((camera_intrinsics, camera2lidar, img_aug_matrix),
                                         (self.fixed_camera_intrinsics, self.fixed_camera2lidar,
                                          self.fixed_img_aug_matrix)):
                    if tuple(value.shape) != tuple(bound.shape) or not torch.equal(value, bound):
                        raise ValueError(f"{self.pool_backend} camera calibration differs from its fixed sampling geometry; rebuild the wrapper")
        else:
            frustum = vt.frustum.float()[None, None]
            post_rot, post_trans = img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3]
            geometry = frustum - post_trans.reshape(1, num_cameras, 1, 1, 1, 3)
            geometry = inverse3(post_rot).reshape(1, num_cameras, 1, 1, 1, 3, 3) @ geometry.unsqueeze(-1)
            geometry = torch.cat((geometry[..., :2, :] * geometry[..., 2:3, :], geometry[..., 2:3, :]), -2)
            combine = camera2lidar[..., :3, :3] @ inverse3(camera_intrinsics[..., :3, :3])
            geometry = (combine.reshape(1, num_cameras, 1, 1, 1, 3, 3) @ geometry).squeeze(-1)
            geometry = geometry + camera2lidar[..., :3, 3].reshape(1, num_cameras, 1, 1, 1, 3)
            coords = ((geometry - (vt.bx - vt.dx / 2)) / vt.dx).long()
        flat_feature = feature.reshape(num_cameras, channels, height, width).float()
        if self.depth_conditioned:
            if points is None:
                raise ValueError("Detection camera depth requires raw LiDAR points")
            scalar_depth = project_lidar_depth(
                points, camera2lidar, camera_intrinsics, img_aug_matrix,
                vt.image_size, self.point_cloud_range)
            depth_feature = vt.dtransform(scalar_depth.reshape(num_cameras, 1, *vt.image_size))
            flat_feature = torch.cat((depth_feature, flat_feature), dim=1)
        prediction = vt.depthnet(flat_feature).reshape(1, num_cameras, vt.D + vt.C, height, width)
        depth, channels = prediction[:, :, :vt.D].softmax(2), prediction[:, :, vt.D:]
        if self.fixed_calibration:
            return vt.downsample(self.pool(depth, channels).contiguous())
        nx, ny, nz = self.bev_shape
        out = feature.new_zeros((nz * nx * ny, vt.C), dtype=torch.float32)
        bounds = torch.tensor(self.bev_shape, device=images.device)
        for camera_index in range(num_cameras):
            for start in range(0, vt.D, vt.pool_depth_chunk):
                end = min(vt.D, start + vt.pool_depth_chunk)
                lifted = (depth[0, camera_index, start:end, :, :, None]
                          * channels[0, camera_index].permute(1, 2, 0)[None])
                co = coords[0, camera_index, start:end].reshape(-1, 3)
                valid = ((co >= 0) & (co < bounds)).all(-1)
                co = co[valid]
                ranks = (co[:, 2] * nx + co[:, 0]) * ny + co[:, 1]
                values = lifted.reshape(-1, vt.C)[valid]
                out = out.scatter_add(0, ranks[:, None].expand(-1, vt.C), values)
        out = out.reshape(1, nz, nx, ny, vt.C).permute(0, 1, 4, 2, 3)
        return vt.downsample(out.reshape(1, nz * vt.C, nx, ny).contiguous())


class _OnnxAttention(nn.Module):
    """Reuse packed projections with a fixed architectural self/cross grouping."""
    def __init__(self, original, self_attention):
        super().__init__()
        self.in_proj_weight = original.in_proj_weight
        self.in_proj_bias = original.in_proj_bias
        self.out_proj = original.out_proj
        self.num_heads = original.num_heads
        self.dropout = original.dropout
        self.self_attention = self_attention
        self.training = original.training

    def forward(self, query, key, value):
        target_length, batch_size, channels = query.shape
        head_dim = channels // self.num_heads
        if self.self_attention:
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, -1)
        else:
            q = F.linear(query, self.in_proj_weight[:channels], self.in_proj_bias[:channels])
            k, v = F.linear(key, self.in_proj_weight[channels:], self.in_proj_bias[channels:]).chunk(2, -1)
        q = q * (float(head_dim) ** -0.5)
        q = q.contiguous().view(target_length, batch_size * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * self.num_heads, head_dim).transpose(0, 1)
        weights = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.bmm(weights, v)
        output = output.transpose(0, 1).contiguous().view(target_length, batch_size, channels)
        return F.linear(output, self.out_proj.weight, self.out_proj.bias)


class _OnnxDecoderLayer(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.self_attn = _OnnxAttention(original.self_attn, True)
        self.multihead_attn = _OnnxAttention(original.multihead_attn, False)
        for name in ("linear1", "linear2", "dropout", "norm1", "norm2", "norm3",
                     "dropout1", "dropout2", "dropout3", "self_posembed", "cross_posembed"):
            setattr(self, name, getattr(original, name))
        self.training = original.training

    def forward(self, query, key, query_pos, key_pos):
        query_embedding = self.self_posembed(query_pos).permute(2, 0, 1)
        key_embedding = self.cross_posembed(key_pos).permute(2, 0, 1)
        query, key = query.permute(2, 0, 1), key.permute(2, 0, 1)
        positioned_query = query + query_embedding
        query = self.norm1(query + self.dropout1(
            self.self_attn(positioned_query, positioned_query, positioned_query)))
        positioned_key = key + key_embedding
        query = self.norm2(query + self.dropout2(
            self.multihead_attn(query + query_embedding, positioned_key, positioned_key)))
        query = self.norm3(query + self.dropout3(
            self.linear2(self.dropout(F.relu(self.linear1(query))))))
        return query.permute(1, 2, 0)


class OnnxTransFusionHead(nn.Module):
    """Fixed proposal TopK, dynamic final detections, and shared head weights.

    Positive, non-tied proposal scores retain reference order. Zero/tied
    candidates follow TopK's tie order; no box NMS is introduced.
    """
    def __init__(self, original, score_threshold=0.0):
        super().__init__()
        if not isinstance(score_threshold, (int, float)) or not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0,1]")
        self.score_threshold = score_threshold
        for name in ("num_classes", "num_proposals", "voxel_size", "pc_range",
                     "out_size_factor", "post_center_range", "decoded_z_origin"):
            setattr(self, name, getattr(original, name))
        self.shared_conv = original.shared_conv
        self.heatmap_head = original.heatmap_head
        self.class_encoding = original.class_encoding
        self.decoder = _OnnxDecoderLayer(original.decoder[0])
        self.prediction_head = original.prediction_heads[0]
        self.register_buffer("bev_pos", original.bev_pos, persistent=False)
        self.training = original.training

    def forward_raw(self, features):
        lidar_feat = self.shared_conv(features)
        lidar_flat = lidar_feat.flatten(2)
        bev_pos = self.bev_pos.to(dtype=lidar_feat.dtype).expand(features.shape[0], -1, -1)
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        local_max = F.pad(F.max_pool2d(heatmap, 3, stride=1), (1, 1, 1, 1))
        # Fixed class metadata, never values of this frame's heatmap.
        unsuppressed = torch.arange(self.num_classes, device=features.device) >= 8
        unsuppressed = unsuppressed & (torch.arange(self.num_classes, device=features.device) <= 9)
        local_max = torch.where(unsuppressed[None, :, None, None], heatmap, local_max)
        heatmap = (heatmap * (heatmap == local_max)).flatten(2)
        top_proposals = heatmap.flatten(1).topk(self.num_proposals, dim=-1, sorted=True).indices
        query_labels = top_proposals // heatmap.shape[-1]
        query_indices = top_proposals % heatmap.shape[-1]
        query_feat = lidar_flat.gather(2, query_indices[:, None].expand(-1, lidar_flat.shape[1], -1))
        one_hot = F.one_hot(query_labels, self.num_classes).permute(0, 2, 1)
        query_feat = query_feat + self.class_encoding(one_hot.to(dtype=lidar_feat.dtype))
        query_pos = bev_pos.gather(1, query_indices[:, :, None].expand(-1, -1, 2))
        query_feat = self.decoder(query_feat, lidar_flat, query_pos, bev_pos)
        predictions = self.prediction_head(query_feat)
        predictions["center"] = predictions["center"] + query_pos.transpose(1, 2)
        predictions["query_labels"] = query_labels
        predictions["query_heatmap_score"] = heatmap.gather(
            2, query_indices[:, None].expand(-1, self.num_classes, -1))
        predictions["dense_heatmap"] = dense_heatmap
        return predictions

    def decode(self, predictions):
        one_hot = F.one_hot(predictions["query_labels"], self.num_classes).permute(0, 2, 1)
        class_scores = predictions["heatmap"].sigmoid() * predictions["query_heatmap_score"] * one_hot
        scores, labels = class_scores.max(dim=1)
        center = predictions["center"]
        xy = torch.stack((
            center[:, 0] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0],
            center[:, 1] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]), dim=1)
        dimensions = predictions["dim"].exp()
        decoded_height = predictions["height"] - dimensions[:, 2:3] * 0.5
        bottom_height = (decoded_height - dimensions[:, 2:3] * 0.5
                         if self.decoded_z_origin == "center" else decoded_height)
        rotation = atan2(predictions["rot"][:, 0:1], predictions["rot"][:, 1:2])
        boxes = torch.cat((xy, bottom_height, dimensions, rotation, predictions["vel"]), dim=1)[0].T
        scores, labels = scores[0], labels[0]
        keep = torch.ones_like(scores, dtype=torch.bool)
        if self.post_center_range is not None:
            bounds = boxes.new_tensor(self.post_center_range)
            # Match the legacy reference point before physical bottom-Z repair.
            positions = torch.cat((xy, decoded_height), dim=1)[0].T
            keep = (positions >= bounds[:3]).all(-1) & (positions <= bounds[3:]).all(-1)
        if self.score_threshold:
            keep = keep & (scores > self.score_threshold)
        return boxes[keep], scores[keep], labels[keep]

    def forward(self, features):
        return self.decode(self.forward_raw(features))


class OnnxBEVFusion(nn.Module):
    """Shared weights, standard ONNX ops, and task-specific inference outputs.

    Segmentation keeps its single probability tensor. Detection returns
    (boxes_3d[N,9], scores[N], labels[N]), with a dynamic final box count.
    Geometry is dynamic unless a fixed-calibration pooling plan is supplied.
    Raw-point voxelization, sparse searches and detection depth stay dynamic.
    """
    def __init__(self, model, score_threshold=None, *, pool_backend="reference",
                 pool_num_points=10, fixed_calibration=None):
        super().__init__()
        self.task = getattr(model, "task", "seg")
        if self.task not in ("det", "seg"):
            raise ValueError(f"Unsupported BEVFusion task: {self.task!r}")
        self.voxel_size = model.voxel_size
        self.point_cloud_range = model.point_cloud_range
        self.max_voxels = model.max_voxels
        self.camera = OnnxCamera(model.encoders["camera"], self.point_cloud_range,
                                 pool_backend=pool_backend, pool_num_points=pool_num_points,
                                 fixed_calibration=fixed_calibration)
        self.pool_backend = pool_backend
        self.lidar = OnnxLidarEncoder(model.encoders["lidar"]["backbone"])
        self.fuser = model.fuser
        self.decoder = model.decoder
        if self.task == "det":
            if score_threshold is None:
                score_threshold = model.config["model"]["heads"]["object"]["bbox_coder"]["score_threshold"]
            self.head = OnnxTransFusionHead(model.heads["object"], score_threshold)
        else:
            self.head = model.heads["map"]
        self.quantize_decoder = getattr(model, "quantize_decoder", False)

    def forward(self, images, points, camera_intrinsics, camera2lidar, img_aug_matrix):
        features, coords = voxelize(points, max_voxels=self.max_voxels,
                                    voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
        lidar_bev = self.lidar(features, coords)
        camera_bev = self.camera(images, camera_intrinsics, camera2lidar, img_aug_matrix, points)
        fused = self.fuser(torch.cat((camera_bev, lidar_bev), dim=1))
        decoded = self.decoder["neck"](self.decoder["backbone"](fused))[0]
        if self.quantize_decoder:
            decoded = decoded.half().float()
        return self.head(decoded)


class FixedCalibrationOnnxBEVFusion(OnnxBEVFusion):
    """Two runtime inputs (images, points), with explicitly bound pool geometry.

    The fixed original-image intrinsics, camera-to-LiDAR and image augmentation
    matrices are cloned FP32 buffers. Rebuild/export after calibration changes.
    LiDAR still uses the dynamic standard ONNX adapter (Unique and Loop); this
    wrapper does not establish whole-model BPU compatibility.
    """
    def __init__(self, model, calibration, score_threshold=None, pool_num_points=10, *, pool_backend="hat"):
        super().__init__(model, score_threshold, pool_backend=pool_backend,
                         pool_num_points=pool_num_points, fixed_calibration=calibration)
        for name in ("camera_intrinsics", "camera2lidar", "img_aug_matrix"):
            self.register_buffer(name, getattr(self.camera, "fixed_" + name).detach().clone())

    def forward(self, images, points):
        return super().forward(images, points, self.camera_intrinsics,
                               self.camera2lidar, self.img_aug_matrix)
