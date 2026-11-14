"""Inference-only, checkpoint-compatible PyTorch components for BEVFusion seg.

Runtime dependency: torch. Inputs are already normalized/augmented; this module
performs no image preparation, point voxelization, or sparse LiDAR convolution.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


MAP_CLASSES = (
    "drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "divider"
)


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(96)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 4, 0, (-h) % 4))
        x = self.projection(x)
        hw = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), hw


class PatchMerging(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # MMDetection uses nn.Unfold: channel-major [c00,c01,c10,c11].
        # Concatenating four spatial slices would silently change checkpoint semantics.
        self.sampler = nn.Unfold(kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(4 * channels)
        self.reduction = nn.Linear(4 * channels, 2 * channels, bias=False)

    def forward(self, x, hw):
        b, _, c = x.shape
        h, w = hw
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = F.pad(x, (0, w % 2, 0, h % 2))
        x = self.sampler(x).transpose(1, 2)
        return self.reduction(self.norm(x)), ((h + 1) // 2, (w + 1) // 2)


class WindowMSA(nn.Module):
    def __init__(self, channels, num_heads, window_size=7):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (channels // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        )).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous() + window_size - 1
        index = relative[..., 0] * (2 * window_size - 1) + relative[..., 1]
        self.register_buffer("relative_position_index", index)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, mask=None):
        b, length, c = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(length, length, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.reshape(b // nw, nw, self.num_heads, length, length)
            attn = attn + mask[None, :, None]
            attn = attn.reshape(b, self.num_heads, length, length)
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b, length, c)
        return self.proj(x)


class ShiftWindowMSA(nn.Module):
    def __init__(self, channels, num_heads, shift_size=0):
        super().__init__()
        self.window_size = 7
        self.shift_size = shift_size
        self.w_msa = WindowMSA(channels, num_heads)

    def _partition(self, x):
        b, h, w, c = x.shape
        s = self.window_size
        return x.reshape(b, h // s, s, w // s, s, c).permute(
            0, 1, 3, 2, 4, 5
        ).reshape(-1, s * s, c)

    def forward(self, x, hw):
        b, _, c = x.shape
        h, w = hw
        s, shift = self.window_size, self.shift_size
        x = F.pad(x.reshape(b, h, w, c), (0, 0, 0, (-w) % s, 0, (-h) % s))
        hp, wp = x.shape[1:3]
        mask = None
        if shift:
            x = torch.roll(x, (-shift, -shift), (1, 2))
            labels = x.new_zeros((1, hp, wp, 1))
            slices = (slice(0, -s), slice(-s, -shift), slice(-shift, None))
            for i, hs in enumerate(slices):
                for j, ws in enumerate(slices):
                    labels[:, hs, ws, :] = 3 * i + j
            windows = self._partition(labels).squeeze(-1)
            delta = windows.unsqueeze(1) - windows.unsqueeze(2)
            mask = delta.masked_fill(delta != 0, -100.0).masked_fill(delta == 0, 0.0)
        x = self.w_msa(self._partition(x), mask)
        x = x.reshape(b, hp // s, wp // s, s, s, c).permute(
            0, 1, 3, 2, 4, 5
        ).reshape(b, hp, wp, c)
        if shift:
            x = torch.roll(x, (shift, shift), (1, 2))
        return x[:, :h, :w, :].reshape(b, h * w, c)


class FFN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(), nn.Dropout(0)),
            nn.Linear(4 * channels, channels), nn.Dropout(0),
        )

    def forward(self, x):
        return self.layers(x)


class SwinBlock(nn.Module):
    def __init__(self, channels, heads, shift):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = ShiftWindowMSA(channels, heads, 3 if shift else 0)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = FFN(channels)

    def forward(self, x, hw):
        x = x + self.attn(self.norm1(x), hw)
        return x + self.ffn(self.norm2(x))


class SwinStage(nn.Module):
    def __init__(self, channels, heads, depth, downsample):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(channels, heads, i % 2) for i in range(depth)
        ])
        self.downsample = PatchMerging(channels) if downsample else None

    def forward(self, x, hw):
        for block in self.blocks:
            x = block(x, hw)
        out, out_hw = x, hw
        if self.downsample is not None:
            x, hw = self.downsample(x, hw)
        return x, hw, out, out_hw


class SwinTransformer(nn.Module):
    """Swin-T in evaluation mode, with MMDetection state_dict names and ordering."""
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.stages = nn.ModuleList([
            SwinStage(96 * 2**i, heads, depth, i < 3)
            for i, (heads, depth) in enumerate(zip((3, 6, 12, 24), (2, 2, 6, 2)))
        ])
        for i in (1, 2, 3):
            self.add_module(f"norm{i}", nn.LayerNorm(96 * 2**i))

    def forward(self, x):
        x, hw = self.patch_embed(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw, out, (h, w) = stage(x, hw)
            if i > 0:
                out = getattr(self, f"norm{i}")(out)
                outs.append(out.reshape(out.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous())
        return outs


class ConvModule(nn.Module):
    def __init__(self, cin, cout, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.activate = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class GeneralizedLSSFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvModule(448, 256, 1), ConvModule(1152, 256, 1)])
        self.fpn_convs = nn.ModuleList([ConvModule(256, 256, 3), ConvModule(256, 256, 3)])

    def forward(self, inputs):
        laterals = list(inputs)
        for i in (1, 0):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:],
                               mode="bilinear", align_corners=False)
            laterals[i] = self.fpn_convs[i](self.lateral_convs[i](torch.cat([laterals[i], up], 1)))
        return tuple(laterals[:2])


class LSSTransform(nn.Module):
    def __init__(self, image_size=(256, 704), pool_depth_chunk=16, pool_backend="reference",
                 precompute=True, pool_num_points=10):
        super().__init__()
        if pool_backend not in ("reference", "cuda", "auto", "compatible", "horizon", "hat", "hat_plugin"):
            raise ValueError("pool_backend must be reference, cuda, auto, compatible, horizon, hat or hat_plugin")
        from .configuration import validate_pool_num_points
        self.pool_num_points = validate_pool_num_points(pool_num_points)
        self.pool_backend = pool_backend
        self.precompute_enabled = bool(precompute)
        self._bev_cache = None
        self._bev_cache_hits = 0
        self._bev_cache_misses = 0
        self.image_size = tuple(image_size)
        self.feature_size = tuple(s // 8 for s in image_size)
        self.C, self.D = 80, 118
        self.pool_depth_chunk = pool_depth_chunk
        self.dx = nn.Parameter(torch.tensor([0.4, 0.4, 20.0]), requires_grad=False)
        self.bx = nn.Parameter(torch.tensor([-51.0, -51.0, 0.0]), requires_grad=False)
        self.nx = nn.Parameter(torch.tensor([256, 256, 1]), requires_grad=False)
        fh, fw = self.feature_size
        d = torch.arange(1.0, 60.0, 0.5).reshape(-1, 1, 1).expand(-1, fh, fw)
        x = torch.linspace(0, image_size[1] - 1, fw).reshape(1, 1, fw).expand(self.D, fh, fw)
        y = torch.linspace(0, image_size[0] - 1, fh).reshape(1, fh, 1).expand(self.D, fh, fw)
        self.frustum = nn.Parameter(torch.stack([x, y, d], -1), requires_grad=False)
        self.depthnet = nn.Conv2d(256, self.D + self.C, 1)
        self.downsample = nn.Sequential(
            nn.Conv2d(80, 80, 3, padding=1, bias=False), nn.BatchNorm2d(80), nn.ReLU(True),
            nn.Conv2d(80, 80, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(80), nn.ReLU(True),
            nn.Conv2d(80, 80, 3, padding=1, bias=False), nn.BatchNorm2d(80), nn.ReLU(True),
        )

    def clear_bev_cache(self):
        """Release the single cached calibration plan; leave diagnostic counters."""
        self._bev_cache = None

    def bev_cache_info(self):
        """Return cache counters and resident index bytes, without exposing tensors."""
        cache = self._bev_cache
        plan = cache[3] if cache is not None else None
        if plan is not None and hasattr(plan, "statistics"):
            statistics = plan.statistics()
            chunks, index_bytes = statistics["groups"], statistics["index_bytes"]
        else:
            chunks = len(plan.chunks) if plan is not None else 0
            index_bytes = plan.nbytes if plan is not None else 0
        return {"enabled": self.precompute_enabled, "entries": int(cache is not None),
                "hits": self._bev_cache_hits, "misses": self._bev_cache_misses,
                "chunks": chunks,
                "bytes": (index_bytes + cache[1].numel() * cache[1].element_size()
                          + cache[2].numel() * cache[2].element_size()) if cache is not None else 0}

    def _apply(self, fn, recurse=True):
        # Plans are ordinary runtime attributes, never checkpoint buffers.
        self.clear_bev_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        # Parent model.load_state_dict recursively invokes this method.
        self.clear_bev_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def train(self, mode=True):
        if mode:
            self.clear_bev_cache()
        return super().train(mode)

    def _get_bev_plan(self, camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix):
        from .operators.pooling.reference import build_pool_plan

        matrices = (camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix)
        geometry_values = (*matrices, self.dx, self.bx, self.nx)
        metadata = (tuple((tuple(t.shape), t.device, t.dtype) for t in geometry_values),
                    tuple(self.frustum.shape), self.frustum.device, self.frustum.dtype,
                    self.image_size, self.feature_size, self.D, self.C, self.pool_depth_chunk,
                    self.pool_backend, self.pool_num_points)
        # Small matrices are compared by value because the public input validator
        # clones them each frame. Exact comparisons also catch inplace edits to
        # inference tensors, which do not have a usable tensor version counter.
        values = torch.cat([t.detach().reshape(-1).double() for t in geometry_values])
        cache = self._bev_cache
        retain_plan = self.precompute_enabled or self.pool_backend not in ("compatible", "horizon", "hat", "hat_plugin")
        if retain_plan and cache is not None and cache[0] == metadata:
            same = torch.stack(((values == cache[1]).all(),
                                (self.frustum.detach() == cache[2]).all())).all().item()
            if same:
                self._bev_cache_hits += 1
                return cache[3]
        # Retain at most one calibration: moving sensors cannot grow an LRU
        # indefinitely, and a new frame must never use a stale association.
        self.clear_bev_cache()
        geometry = self.get_geometry(*matrices)
        coordinates = self._voxel_indices(geometry)
        shape = tuple(int(value) for value in self.nx)
        if self.pool_backend == "compatible":
            from .operators.pooling.compatible import CompatibleBEVPool
            # Keep runtime indices inside the ordinary cache tuple, outside the
            # model module/state_dict tree, exactly like the native pool plan.
            plan = CompatibleBEVPool(coordinates, shape, self.C).to(device=self.frustum.device)
        elif self.pool_backend == "hat":
            from .operators.pooling.hat import HATLSSPool
            plan = HATLSSPool(coordinates, shape, self.C, num_points=self.pool_num_points).to(
                device=self.frustum.device)
        elif self.pool_backend == "hat_plugin":
            from .operators.pooling.hat_plugin import HATPluginLSSPool
            plan = HATPluginLSSPool(coordinates, shape, self.C, num_points=self.pool_num_points).to(
                device=self.frustum.device)
        elif self.pool_backend == "horizon":
            from .operators.pooling.horizon import HorizonBEVPool
            plan = HorizonBEVPool(coordinates, shape, self.C).to(device=self.frustum.device)
        else:
            plan = build_pool_plan(coordinates, shape, self.pool_depth_chunk)
        if retain_plan:
            self._bev_cache = (metadata, values.detach().clone(), self.frustum.detach().clone(), plan)
        self._bev_cache_misses += 1
        return plan

    def _pool_predictions(self, depth, features, camera2lidar, camera_intrinsics,
                          img_aug_matrix, lidar_aug_matrix):
        matrices = (camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix)
        if self.pool_backend in ("compatible", "horizon", "hat", "hat_plugin"):
            if self.training or torch.is_grad_enabled():
                raise RuntimeError(f"{self.pool_backend} pooling is inference-only; use eval() and inference_mode()")
            if self.pool_backend == "hat_plugin" and (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
                raise RuntimeError("Export a prebuilt HATPluginLSSPool with the Horizon HBIR exporter; standard ONNX export is unsupported")
            if self.pool_backend == "compatible" and (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
                raise RuntimeError("Export compatible pooling explicitly with tools/export_bev_pool.py")
            if self.pool_backend == "hat" and (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
                raise RuntimeError("Export HAT pooling explicitly with tools/export_bev_pool.py --pool-backend hat")
            if self.pool_backend == "horizon" and (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
                raise RuntimeError("Export a prebuilt HorizonBEVPool with the Horizon HBIR exporter")
            return self._get_bev_plan(*matrices)(depth, features)
        # Bare inference CUDA kernels have no autograd registration; eval alone
        # is insufficient. Bypass for gradients, training, tracing and ONNX export.
        if (self.precompute_enabled and not self.training and not torch.is_grad_enabled()
                and not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export()):
            from .operators.pooling.reference import pool_with_plan
            return pool_with_plan(self._get_bev_plan(*matrices), depth, features, self.pool_backend)
        geometry = self.get_geometry(*matrices)
        coords = self._voxel_indices(geometry)
        b, n, _, _, _ = depth.shape
        nx, ny, nz = (int(v) for v in self.nx)
        out = features.new_zeros((b * nz * nx * ny, self.C), dtype=torch.float32)
        for bi in range(b):
            for ni in range(n):
                for start in range(0, self.D, self.pool_depth_chunk):
                    end = min(self.D, start + self.pool_depth_chunk)
                    lifted = depth[bi, ni, start:end, :, :, None] * features[bi, ni].permute(1, 2, 0)[None]
                    self._accumulate(out, coords[bi, ni, start:end], lifted, bi)
        return out.reshape(b, nz, nx, ny, self.C).permute(0, 1, 4, 2, 3).reshape(b, nz * self.C, nx, ny)

    def get_geometry(self, camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix):
        b, n = camera2lidar.shape[:2]
        camera2lidar = camera2lidar.float()
        intrins = camera_intrinsics[..., :3, :3].float()
        post_rot = img_aug_matrix[..., :3, :3].float()
        post_trans = img_aug_matrix[..., :3, 3].float()
        points = self.frustum.float()[None, None] - post_trans.reshape(b, n, 1, 1, 1, 3)
        points = torch.linalg.inv(post_rot).reshape(b, n, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)
        points = torch.cat([points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]], -2)
        combine = camera2lidar[..., :3, :3] @ torch.linalg.inv(intrins)
        points = (combine.reshape(b, n, 1, 1, 1, 3, 3) @ points).squeeze(-1)
        points = points + camera2lidar[..., :3, 3].reshape(b, n, 1, 1, 1, 3)
        extra = lidar_aug_matrix.float()
        points = (extra[:, None, None, None, None, :3, :3] @ points.unsqueeze(-1)).squeeze(-1)
        return points + extra[:, None, None, None, None, :3, 3]

    def _voxel_indices(self, geometry):
        # Match original .long() truncation (including a <1-voxel lower-bound fringe).
        return ((geometry - (self.bx - self.dx / 2)) / self.dx).long()

    def _accumulate(self, out, coords, values, batch_index):
        nx, ny, nz = (int(v) for v in self.nx)
        coords = coords.reshape(-1, 3)
        valid = ((coords >= 0) & (coords < self.nx)).all(-1)
        coords = coords[valid]
        backend = getattr(self, "pool_backend", "reference")
        if backend == "cuda" and not values.is_cuda:
            raise ValueError("The cuda pooling backend requires CUDA tensors")
        if values.is_cuda and backend in ("auto", "cuda"):
            if not len(coords):
                return
            try:
                from ops.bev_pool import bev_pool
            except (ImportError, OSError) as exc:
                raise RuntimeError("Install bevfusion-bev-pool-ops with scripts/install_ops.bat") from exc
            # Each chunk is one camera of one batch: use a local batch index
            # and accumulate the CUDA B,C,Z,X,Y result into the shared layout.
            coordinates = F.pad(coords, (0, 1)).int().contiguous()
            pooled = bev_pool(values.reshape(-1, self.C)[valid].float().contiguous(),
                              coordinates, 1, nz, nx, ny)
            start = batch_index * nz * nx * ny
            out[start:start + nz * nx * ny] += pooled.permute(0, 2, 3, 4, 1).reshape(-1, self.C)
            return
        # Original CUDA output is B,C,Z,X,Y: X is the first spatial axis.
        ranks = ((batch_index * nz + coords[:, 2]) * nx + coords[:, 0]) * ny + coords[:, 1]
        out.index_add_(0, ranks, values.reshape(-1, self.C)[valid].float())

    def bev_pool(self, geometry, lifted):
        """Reference full-lift API: geometry B,N,D,H,W,3; lifted B,N,D,H,W,C."""
        b = lifted.shape[0]
        nx, ny, nz = (int(v) for v in self.nx)
        out = lifted.new_zeros((b * nz * nx * ny, self.C), dtype=torch.float32)
        coords = self._voxel_indices(geometry)
        for bi in range(b):
            self._accumulate(out, coords[bi], lifted[bi], bi)
        return out.reshape(b, nz, nx, ny, self.C).permute(0, 1, 4, 2, 3).reshape(b, nz * self.C, nx, ny)

    def get_cam_feats(self, x):
        b, n, c, h, w = x.shape
        prediction = self.depthnet(x.reshape(b * n, c, h, w).float())
        depth = prediction[:, :self.D].softmax(1)
        lifted = prediction[:, self.D:, None] * depth[:, None]
        return lifted.reshape(b, n, self.C, self.D, h, w).permute(0, 1, 3, 4, 5, 2)

    def forward(self, x, camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix):
        b, n, c, h, w = x.shape
        if (h, w) != self.feature_size:
            raise ValueError(f"Expected LSS feature size {self.feature_size}, got {(h, w)}")
        prediction = self.depthnet(x.reshape(b * n, c, h, w).float()).reshape(b, n, self.D + self.C, h, w)
        depth = prediction[:, :, :self.D].softmax(2)
        features = prediction[:, :, self.D:]
        out = self._pool_predictions(depth, features, camera2lidar, camera_intrinsics,
                                     img_aug_matrix, lidar_aug_matrix)
        return self.downsample(out.contiguous())


class NativeCameraEncoder(nn.ModuleDict):
    def __init__(self, camera_batch_size=1, pool_depth_chunk=16, *, vtransform=None,
                 pool_backend="reference", precompute=True, pool_num_points=10):
        from .configuration import validate_pool_num_points
        pool_num_points = validate_pool_num_points(pool_num_points)
        super().__init__({"backbone": SwinTransformer(), "neck": GeneralizedLSSFPN(),
                          "vtransform": vtransform if vtransform is not None else LSSTransform(
                              pool_depth_chunk=pool_depth_chunk, pool_backend=pool_backend,
                              precompute=precompute, pool_num_points=pool_num_points)})
        self.camera_batch_size = camera_batch_size

    def encode_images(self, images):
        """Normalized RGB B,N,3,256,704 -> camera features B,N,256,32,88."""
        b, n, c, h, w = images.shape
        if (c, h, w) != (3, 256, 704):
            raise ValueError(f"Expected B,N,3,256,704 images, got {tuple(images.shape)}")
        flat = images.reshape(b * n, c, h, w)
        step = self.camera_batch_size or b * n
        features = [self["neck"](self["backbone"](part))[0] for part in flat.split(step)]
        features = torch.cat(features)
        return features.reshape(b, n, *features.shape[1:])

    def forward(self, images, camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix):
        """Matrices: camera tensors B,N,4,4 (intrinsics may be 3,3); LiDAR aug B,4,4."""
        return self["vtransform"](self.encode_images(images), camera2lidar,
                                  camera_intrinsics, img_aug_matrix, lidar_aug_matrix)


class SECOND(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for cin, cout, stride in ((256, 128, 1), (128, 256, 2)):
            layers = []
            for j in range(6):
                layers.extend([
                    nn.Conv2d(cin if j == 0 else cout, cout, 3,
                              stride=stride if j == 0 else 1, padding=1, bias=False),
                    nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01), nn.ReLU(True),
                ])
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return tuple(outs)


class SECONDFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128, 256, 1, bias=False),
                          nn.BatchNorm2d(256, eps=1e-3, momentum=0.01), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
                          nn.BatchNorm2d(256, eps=1e-3, momentum=0.01), nn.ReLU(True)),
        ])

    def forward(self, x):
        return [torch.cat([block(feature) for block, feature in zip(self.deblocks, x)], 1)]


class BEVGridTransform(nn.Module):
    def forward(self, x):
        # Preserve original grid_sample axis order: row=X, column=Y.
        coords = torch.arange(-49.75, 50.0, 0.5, device=x.device, dtype=torch.float32)
        coords = (coords + 51.2) / 102.4 * 2 - 1
        u, v = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack([v, u], -1)[None].expand(x.shape[0], -1, -1, -1)
        return F.grid_sample(x, grid, mode="bilinear", align_corners=False)


class BEVSegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes = MAP_CLASSES
        self.transform = BEVGridTransform()
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, len(self.classes), 1),
        )

    def forward(self, x, return_logits=False):
        if isinstance(x, (list, tuple)):
            x = x[0]
        logits = self.classifier(self.transform(x))
        return logits if return_logits else logits.sigmoid()


def build_camera_encoder(camera_batch_size=1, pool_depth_chunk=16):
    return NativeCameraEncoder(camera_batch_size, pool_depth_chunk).eval()


def build_decoder():
    return nn.ModuleDict({"backbone": SECOND(), "neck": SECONDFPN()}).eval()


def build_map_head():
    return BEVSegmentationHead().eval()


def build_fuser():
    """Call on torch.cat([camera_bev, lidar_bev], dim=1), in this order."""
    return nn.Sequential(nn.Conv2d(336, 256, 3, padding=1, bias=False),
                         nn.BatchNorm2d(256), nn.ReLU(True)).eval()


def load_checkpoint_subset(module, checkpoint, prefix=""):
    """Strictly load all keys under prefix; raise on missing, extra, or wrong shapes.

    checkpoint may be a path, a raw state_dict, or {'state_dict': state_dict}.
    Prefix examples: 'encoders.camera.', 'decoder.', 'heads.map.', 'fuser.'.
    Returns the number of loaded state tensors. No weight conversion is performed.
    """
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    subset = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    module.load_state_dict(subset, strict=True)
    return len(subset)


class NativeBEVFusionSeg(nn.Module):
    """Camera and dense fusion components; caller supplies the sparse LiDAR BEV."""
    def __init__(self, camera_batch_size=1, pool_depth_chunk=16):
        super().__init__()
        self.encoders = nn.ModuleDict({"camera": build_camera_encoder(camera_batch_size, pool_depth_chunk)})
        self.fuser = build_fuser()
        self.decoder = build_decoder()
        self.heads = nn.ModuleDict({"map": build_map_head()})
        self.eval()

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        # Reject unknown dense keys; only the explicitly external LiDAR branch is excluded.
        selected = {k: v for k, v in state.items() if not k.startswith("encoders.lidar.")}
        self.load_state_dict(selected, strict=True)
        return {"loaded_tensors": len(selected),
                "external_lidar_tensors": sum(k.startswith("encoders.lidar.") for k in state)}

    def extract_camera_features(self, images, camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix):
        return self.encoders["camera"](images, camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix)

    def decode(self, fused_bev, return_logits=False):
        features = self.decoder["neck"](self.decoder["backbone"](fused_bev))
        return self.heads["map"](features, return_logits=return_logits)

    def fuse_and_decode(self, camera_bev, lidar_bev, return_logits=False):
        if camera_bev.shape[1:] != (80, 128, 128) or lidar_bev.shape[1:] != (256, 128, 128):
            raise ValueError(f"Expected camera B,80,128,128 and LiDAR B,256,128,128; "
                             f"got {tuple(camera_bev.shape)}, {tuple(lidar_bev.shape)}")
        fused = self.fuser(torch.cat([camera_bev, lidar_bev], 1))
        return self.decode(fused, return_logits=return_logits)

    @torch.inference_mode()
    def forward(self, images, lidar_bev, camera2lidar, camera_intrinsics, img_aug_matrix, lidar_aug_matrix,
                return_logits=False):
        camera_bev = self.extract_camera_features(images, camera2lidar, camera_intrinsics,
                                                  img_aug_matrix, lidar_aug_matrix)
        return self.fuse_and_decode(camera_bev, lidar_bev, return_logits=return_logits)
