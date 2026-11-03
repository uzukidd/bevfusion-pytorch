# Copyright 2018-2019 Open-MMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from BEVFusion's mmdet3d/models/heads/bbox/transfusion.py,
# models/utils/transformer.py and core/bbox/coders/transfusion_bbox_coder.py.
# Upstream: https://github.com/mit-han-lab/bevfusion
# License: ../references/bevfusion/LICENSE (Apache License, Version 2.0).
# Modified: standalone torch modules, explicit query metadata, functional decode.
"""Checkpoint-compatible, torch-only TransFusion inference head.

BEV tensors use [batch, channels, X, Y], with X preceding Y when flattened.
``grid_size`` is the BEV feature size, already divided by ``out_size_factor``.
Forward returns raw predictions; callers decide whether/when to call decode.
"""

import torch
from torch import nn
from torch.nn import functional as F


OBJECT_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)

__all__ = ["TransFusionHead", "OBJECT_CLASSES"]


class _ConvModule(nn.Module):
    """Conv/BN/ReLU with MMCV's checkpoint names and bias='auto' semantics."""

    def __init__(self, in_channels, out_channels, kernel_size, *, dim, padding=0):
        super().__init__()
        conv, bn = (nn.Conv1d, nn.BatchNorm1d) if dim == 1 else (nn.Conv2d, nn.BatchNorm2d)
        self.conv = conv(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = bn(out_channels, eps=1e-5, momentum=0.1)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class _PositionEmbeddingLearned(nn.Module):
    def __init__(self, hidden_channel):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(2, hidden_channel, 1),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channel, hidden_channel, 1),
        )

    def forward(self, positions):
        return self.position_embedding_head(positions.transpose(1, 2).contiguous())


class _MultiheadAttention(nn.Module):
    """BEVFusion's packed-projection attention, using only torch operations.

    The upstream transformer implements the original PyTorch attention formula.
    Keep its projection grouping and explicit softmax/bmm (including dropout),
    avoiding backend-dependent fused attention paths during reference checks.
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("hidden_channel must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, key, value):
        target_length, batch_size, channels = query.shape
        head_dim = channels // self.num_heads
        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, -1)
        elif torch.equal(key, value):
            q = F.linear(query, self.in_proj_weight[:channels], self.in_proj_bias[:channels])
            k, v = F.linear(
                key, self.in_proj_weight[channels:], self.in_proj_bias[channels:]
            ).chunk(2, -1)
        else:
            weights = self.in_proj_weight.chunk(3, 0)
            biases = self.in_proj_bias.chunk(3, 0)
            q, k, v = [F.linear(x, w, b) for x, w, b in zip(
                (query, key, value), weights, biases
            )]
        q = q * (float(head_dim) ** -0.5)
        q = q.contiguous().view(target_length, batch_size * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * self.num_heads, head_dim).transpose(0, 1)
        weights = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.bmm(weights, v)
        output = output.transpose(0, 1).contiguous().view(target_length, batch_size, channels)
        return F.linear(output, self.out_proj.weight, self.out_proj.bias)


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_channel):
        super().__init__()
        self.self_attn = _MultiheadAttention(hidden_channel)
        self.multihead_attn = _MultiheadAttention(hidden_channel)
        self.linear1 = nn.Linear(hidden_channel, 256)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(256, hidden_channel)
        self.norm1 = nn.LayerNorm(hidden_channel)
        self.norm2 = nn.LayerNorm(hidden_channel)
        self.norm3 = nn.LayerNorm(hidden_channel)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.self_posembed = _PositionEmbeddingLearned(hidden_channel)
        self.cross_posembed = _PositionEmbeddingLearned(hidden_channel)

    def forward(self, query, key, query_pos, key_pos):
        query_embedding = self.self_posembed(query_pos).permute(2, 0, 1)
        key_embedding = self.cross_posembed(key_pos).permute(2, 0, 1)
        query, key = query.permute(2, 0, 1), key.permute(2, 0, 1)
        # Upstream includes position in V as well as Q/K in both attentions.
        positioned_query = query + query_embedding
        query = self.norm1(query + self.dropout1(
            self.self_attn(positioned_query, positioned_query, positioned_query)
        ))
        positioned_key = key + key_embedding
        query = self.norm2(query + self.dropout2(
            self.multihead_attn(query + query_embedding, positioned_key, positioned_key)
        ))
        query = self.norm3(query + self.dropout3(
            self.linear2(self.dropout(F.relu(self.linear1(query))))
        ))
        return query.permute(1, 2, 0)


class _PredictionHead(nn.Module):
    def __init__(self, hidden_channel, num_classes):
        super().__init__()
        self.heads = dict(center=2, height=1, dim=3, rot=2, vel=2, heatmap=num_classes)
        for name, channels in self.heads.items():
            self.add_module(name, nn.Sequential(
                _ConvModule(hidden_channel, 64, 1, dim=1),
                nn.Conv1d(64, channels, 1),
            ))

    def forward(self, x):
        return {name: getattr(self, name)(x) for name in self.heads}


class TransFusionHead(nn.Module):
    """The single-decoder nuScenes detection head used by bevfusion-det.pth.

    ``forward(features)`` returns a dict of [B, channels, proposals] tensors,
    plus ``dense_heatmap`` [B, classes, X, Y], ``query_labels`` [B, proposals]
    and ``query_heatmap_score`` [B, classes, proposals]. No query state is kept
    on the module. ``decode`` returns bottom-centred LiDAR boxes ordered as
    [x, y, z, dx, dy, dz, yaw, vx, vy], without box NMS, for batch size one.
    """

    def __init__(
        self,
        in_channels=512,
        hidden_channel=128,
        num_classes=10,
        num_proposals=200,
        grid_size=(180, 180),
        voxel_size=(0.075, 0.075),
        pc_range=(-54.0, -54.0),
        out_size_factor=8,
        post_center_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
        decoded_z_origin="bottom",
    ):
        super().__init__()
        if len(grid_size) != 2 or any(int(n) != n or n < 3 for n in grid_size):
            raise ValueError("grid_size must contain two integer BEV sizes >= 3")
        if num_classes < 1 or num_proposals < 1:
            raise ValueError("num_classes and num_proposals must be positive")
        if num_proposals > num_classes * grid_size[0] * grid_size[1]:
            raise ValueError("num_proposals exceeds the number of class/grid candidates")
        if len(voxel_size) != 2 or len(pc_range) != 2:
            raise ValueError("voxel_size and pc_range must each contain X and Y")
        if post_center_range is not None and len(post_center_range) != 6:
            raise ValueError("post_center_range must be None or contain six bounds")
        if decoded_z_origin not in ("bottom", "center"):
            raise ValueError("decoded_z_origin must be center or bottom")
        self.decoded_z_origin = decoded_z_origin
        self.in_channels = in_channels
        self.hidden_channel = hidden_channel
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.grid_size = tuple(int(n) for n in grid_size)
        self.voxel_size = tuple(voxel_size)
        self.pc_range = tuple(pc_range)
        self.out_size_factor = out_size_factor
        self.post_center_range = None if post_center_range is None else tuple(post_center_range)
        self.nms_kernel_size = 3
        self.shared_conv = nn.Conv2d(in_channels, hidden_channel, 3, padding=1)
        self.heatmap_head = nn.Sequential(
            _ConvModule(hidden_channel, hidden_channel, 3, dim=2, padding=1),
            nn.Conv2d(hidden_channel, num_classes, 3, padding=1),
        )
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        self.decoder = nn.ModuleList([_TransformerDecoderLayer(hidden_channel)])
        self.prediction_heads = nn.ModuleList([_PredictionHead(hidden_channel, num_classes)])
        # Match the reference initialization; BN retains eps=1e-5, momentum=.1.
        for parameter in self.decoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        x, y = torch.meshgrid(
            torch.arange(self.grid_size[0], dtype=torch.float32) + 0.5,
            torch.arange(self.grid_size[1], dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        # Derived coordinates must not add a checkpoint state_dict entry.
        self.register_buffer("bev_pos", torch.stack((x, y), -1).reshape(1, -1, 2), persistent=False)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError("features must have shape [B, in_channels, X, Y]")
        if tuple(features.shape[-2:]) != self.grid_size:
            raise ValueError(f"expected BEV grid {self.grid_size}, got {tuple(features.shape[-2:])}")
        batch_size = features.shape[0]
        lidar_feat = self.shared_conv(features)
        lidar_flat = lidar_feat.flatten(2)
        bev_pos = self.bev_pos.to(dtype=lidar_feat.dtype).expand(batch_size, -1, -1)
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        local_max = torch.zeros_like(heatmap)
        local_max[:, :, 1:-1, 1:-1] = F.max_pool2d(heatmap, 3, stride=1)
        # nuScenes pedestrian and traffic_cone bypass heatmap suppression.
        for label in (8, 9):
            if label < self.num_classes:
                local_max[:, label] = heatmap[:, label]
        heatmap = (heatmap * (heatmap == local_max)).flatten(2)
        # argsort also preserves the reference treatment of tied/zero scores.
        top_proposals = heatmap.flatten(1).argsort(dim=-1, descending=True)[:, :self.num_proposals]
        query_labels = top_proposals // heatmap.shape[-1]
        query_indices = top_proposals % heatmap.shape[-1]
        query_feat = lidar_flat.gather(2, query_indices[:, None].expand(-1, lidar_flat.shape[1], -1))
        one_hot = F.one_hot(query_labels, self.num_classes).permute(0, 2, 1)
        query_feat = query_feat + self.class_encoding(one_hot.to(dtype=lidar_feat.dtype))
        query_pos = bev_pos.gather(1, query_indices[:, :, None].expand(-1, -1, 2))
        query_feat = self.decoder[0](query_feat, lidar_flat, query_pos, bev_pos)
        predictions = self.prediction_heads[0](query_feat)
        predictions["center"] = predictions["center"] + query_pos.transpose(1, 2)
        predictions["query_labels"] = query_labels
        predictions["query_heatmap_score"] = heatmap.gather(
            2, query_indices[:, None].expand(-1, self.num_classes, -1)
        )
        predictions["dense_heatmap"] = dense_heatmap
        return predictions

    def decode(self, predictions, score_threshold=0.0):
        """Functionally decode a raw forward dict; do not alter its tensors.

        Bounds are inclusive and apply to the legacy decoded reference point.
        With decoded_z_origin='center', its Z is normalized to the physical box
        bottom before returning; range filtering keeps the checkpoint semantics.
        Threshold zero disables score filtering, matching the original coder.
        """
        if predictions["heatmap"].shape[0] != 1:
            raise ValueError("decode supports batch size one")
        one_hot = F.one_hot(predictions["query_labels"], self.num_classes).permute(0, 2, 1)
        class_scores = predictions["heatmap"].sigmoid() * predictions["query_heatmap_score"] * one_hot
        scores, labels = class_scores.max(dim=1)
        center = predictions["center"]
        xy = torch.stack((
            center[:, 0] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0],
            center[:, 1] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1],
        ), dim=1)
        dimensions = predictions["dim"].exp()
        decoded_height = predictions["height"] - dimensions[:, 2:3] * 0.5
        # Keep the default coder algebra for correctly bottom-origin training
        # boxes. The released BEVFusion detection profile instead retained
        # annotation centre Z before encode: its decoded Z is also a centre.
        bottom_height = (decoded_height - dimensions[:, 2:3] * 0.5
                         if self.decoded_z_origin == "center" else decoded_height)
        rotation = torch.atan2(predictions["rot"][:, 0:1], predictions["rot"][:, 1:2])
        boxes = torch.cat((xy, bottom_height, dimensions, rotation, predictions["vel"]), dim=1)[0].T
        scores, labels = scores[0], labels[0]
        keep = torch.ones_like(scores, dtype=torch.bool)
        if self.post_center_range is not None:
            bounds = boxes.new_tensor(self.post_center_range)
            # Preserve checkpoint range filtering in the legacy decoded frame;
            # normalizing the reported Z must not silently alter query selection.
            range_positions = torch.cat((xy, decoded_height), dim=1)[0].T
            keep = (range_positions >= bounds[:3]).all(-1) & (range_positions <= bounds[3:]).all(-1)
        if score_threshold:
            keep = keep & (scores > score_threshold)
        return {"boxes_3d": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
