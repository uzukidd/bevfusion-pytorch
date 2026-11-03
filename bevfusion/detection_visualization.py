# Copyright 2018-2019 Open-MMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Corner convention adapted from references/bevfusion/mmdet3d/core/bbox/
# structures/lidar_box3d.py and utils.py; see references/bevfusion/LICENSE.
# Modified: standalone torch geometry and optional Matplotlib rendering.
"""Six-camera box projections and LiDAR XY detection views.

Matplotlib is imported only by plot_detection. Boxes use bottom-centred LiDAR
[x, y, z, dx, dy, dz, yaw, vx, vy], following this fork's clockwise yaw convention.
"""

from pathlib import Path

import torch

from .detection_head import OBJECT_CLASSES


BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def boxes_to_corners(boxes):
    """Return [N,8,3], matching LiDARInstance3DBoxes.corners exactly.

    Positive yaw rotates row vectors with [[cos,-sin],[sin,cos]]. The corner
    order is 000,001,011,010,100,101,111,110, and origin is (.5,.5,0).
    """
    if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] not in (7, 9):
        raise ValueError("boxes must have shape [N,7] or [N,9]")
    if not boxes.is_floating_point() or not torch.isfinite(boxes).all():
        raise ValueError("boxes must contain finite floats")
    if (boxes[:, 3:6] <= 0).any():
        raise ValueError("box dimensions must be positive")
    offsets = boxes.new_tensor([
        [-.5, -.5, 0], [-.5, -.5, 1], [-.5, .5, 1], [-.5, .5, 0],
        [.5, -.5, 0], [.5, -.5, 1], [.5, .5, 1], [.5, .5, 0],
    ])
    local = boxes[:, None, 3:6] * offsets
    cos, sin = boxes[:, 6].cos()[:, None], boxes[:, 6].sin()[:, None]
    corners = torch.stack((local[..., 0] * cos + local[..., 1] * sin,
                           -local[..., 0] * sin + local[..., 1] * cos,
                           local[..., 2]), -1)
    return corners + boxes[:, None, :3]


def _clip_image_segment(segment, width, height):
    """Liang-Barsky clipping; reject lines that do not intersect the image."""
    start, finish = segment
    dx, dy = (finish - start).tolist()
    x, y = start.tolist()
    lower, upper = 0., 1.
    for p, q in ((-dx, x), (dx, width - 1 - x), (-dy, y), (dy, height - 1 - y)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        elif p < 0:
            lower = max(lower, q / p)
        else:
            upper = min(upper, q / p)
        if lower > upper:
            return None
    direction = finish - start
    return torch.stack((start + lower * direction, start + upper * direction))


def project_box_edges(corners, camera_intrinsic, camera2lidar, image_size, *,
                      img_aug_matrix=None, near_clip=.1):
    """Project visible box edges, clipping in 3D before perspective division.

    Returns one [visible_edges,2,2] tensor per box. image_size is (height,width).
    K is always the original-image intrinsic. Supply img_aug_matrix only when
    rendering processed images; it transforms projected original pixels.
    """
    corners = torch.as_tensor(corners).detach().float().cpu()
    intrinsic = torch.as_tensor(camera_intrinsic).detach().float().cpu()
    transform = torch.as_tensor(camera2lidar).detach().float().cpu()
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError("corners must have shape [N,8,3]")
    if intrinsic.shape not in ((3, 3), (4, 4)) or transform.shape != (4, 4):
        raise ValueError("expected intrinsic [3,3]/[4,4] and camera2lidar [4,4]")
    if near_clip <= 0 or len(image_size) != 2 or min(image_size) < 1:
        raise ValueError("near_clip and image dimensions must be positive")
    augmentation = None if img_aug_matrix is None else torch.as_tensor(img_aug_matrix).detach().float().cpu()
    if augmentation is not None and augmentation.shape != (4, 4):
        raise ValueError("img_aug_matrix must have shape [4,4]")
    tensors = (corners, intrinsic, transform) + (() if augmentation is None else (augmentation,))
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("projection inputs must be finite")
    lidar2camera = torch.linalg.inv(transform)
    homogeneous = torch.cat((corners, torch.ones_like(corners[..., :1])), -1)
    camera_corners = (homogeneous @ lidar2camera.T)[..., :3]
    result = []
    for box in camera_corners:
        visible = []
        for first, second in BOX_EDGES:
            start, finish = box[first].clone(), box[second].clone()
            if start[2] < near_clip and finish[2] < near_clip:
                continue
            if start[2] < near_clip:
                start = start + (finish - start) * ((near_clip - start[2]) / (finish[2] - start[2]))
            elif finish[2] < near_clip:
                finish = finish + (start - finish) * ((near_clip - finish[2]) / (start[2] - finish[2]))
            projected = torch.stack((start, finish)) @ intrinsic[:3, :3].T
            pixels = projected[:, :2] / projected[:, 2:3]
            if augmentation is not None:
                # Image augmentation acts on [u,v,depth] after perspective division.
                uv_depth = torch.cat((pixels, torch.stack((start[2], finish[2]))[:, None]), -1)
                pixels = (uv_depth @ augmentation[:3, :3].T + augmentation[:3, 3])[:, :2]
            clipped = _clip_image_segment(pixels, image_size[1], image_size[0])
            if clipped is not None and torch.isfinite(clipped).all():
                visible.append(clipped)
        result.append(torch.stack(visible) if visible else corners.new_empty((0, 2, 2)))
    return result


def _camera_matrices(value, name, sizes):
    if value is None:
        raise ValueError(f"{name} is required for camera box projection")
    value = torch.as_tensor(value).detach().float().cpu()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 6 or tuple(value.shape[-2:]) not in sizes:
        raise ValueError(f"{name} must contain six camera matrices")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def plot_detection(images, points, predictions, classes=OBJECT_CLASSES, *,
                   camera_intrinsics, camera2lidar, img_aug_matrix=None,
                   camera_names=None, threshold=.3, save_path=None, show=True,
                   title=None, max_plot_points=100000, bev_range=(-54., -54., 54., 54.)):
    """Draw six camera projections and BEV points/rotated boxes with scores.

    uint8 images use the original K directly. ImageNet-normalized floating
    images require their img_aug_matrix. A leading batch dimension of one is
    accepted. Display threshold never changes the supplied prediction dict.
    PNG outputs use exclusive creation and cannot overwrite an existing file.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch

    if not 0 <= threshold <= 1 or max_plot_points < 1:
        raise ValueError("threshold must be in [0,1] and max_plot_points must be positive")
    if not isinstance(images, torch.Tensor):
        raise ValueError("images must be a Tensor")
    images = images.detach().cpu()
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4 or tuple(images.shape[:2]) != (6, 3):
        raise ValueError("expected six RGB images [6,3,H,W] or [1,6,3,H,W]")
    raw_images = images.dtype == torch.uint8
    if raw_images:
        rgb = images.permute(0, 2, 3, 1).numpy()
    else:
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("processed images must be finite ImageNet-normalized floats")
        if img_aug_matrix is None:
            raise ValueError("img_aug_matrix is required when projecting onto processed images")
        mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
        std = images.new_tensor([.229, .224, .225])[None, :, None, None]
        rgb = (images.float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    intrinsics = _camera_matrices(camera_intrinsics, "camera_intrinsics", ((3, 3), (4, 4)))
    extrinsics = _camera_matrices(camera2lidar, "camera2lidar", ((4, 4),))
    augmentation = None if raw_images else _camera_matrices(img_aug_matrix, "img_aug_matrix", ((4, 4),))
    names = list(camera_names) if camera_names is not None else [f"Camera {i + 1}" for i in range(6)]
    if len(names) != 6 or any(not isinstance(name, str) for name in names):
        raise ValueError("camera_names must contain six names")
    classes = tuple(classes)
    boxes = predictions["boxes_3d"].detach().float().cpu()
    scores = predictions["scores"].detach().float().cpu()
    labels = predictions["labels"].detach().cpu()
    if boxes.ndim != 2 or boxes.shape[1] != 9 or scores.shape != (len(boxes),) or labels.shape != scores.shape:
        raise ValueError("expected boxes_3d [N,9], scores [N], labels [N]")
    if not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities")
    if labels.dtype not in (torch.int32, torch.int64) or ((labels < 0) | (labels >= len(classes))).any():
        raise ValueError("labels must index classes")
    keep = scores >= threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    corners = boxes_to_corners(boxes)
    if not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] not in (4, 5):
        raise ValueError("points must have shape [P,4/5]")
    if len(bev_range) != 4 or bev_range[0] >= bev_range[2] or bev_range[1] >= bev_range[3]:
        raise ValueError("bev_range must contain increasing XY bounds")
    xyz = points.detach().float().cpu()[:, :3]
    valid = torch.isfinite(xyz).all(-1)
    valid &= (xyz[:, 0] >= bev_range[0]) & (xyz[:, 0] <= bev_range[2])
    valid &= (xyz[:, 1] >= bev_range[1]) & (xyz[:, 1] <= bev_range[3])
    xyz = xyz[valid]
    point_count = len(xyz)
    if point_count > max_plot_points:
        xyz = xyz[::(point_count + max_plot_points - 1) // max_plot_points]
    palette = [f"C{i % 10}" for i in range(len(classes))]
    style = {
        "figure.facecolor": "#0e1621", "axes.facecolor": "#111d29",
        "savefig.facecolor": "#0e1621", "text.color": "#e4edf7",
        "axes.labelcolor": "#c5d2e1", "axes.edgecolor": "#526073",
        "xtick.color": "#aab9ca", "ytick.color": "#aab9ca", "font.size": 10,
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(18, 14), layout="constrained")
        grid = figure.add_gridspec(3, 3, height_ratios=(1., 1., 1.75))
        for index in range(6):
            ax = figure.add_subplot(grid[index // 3, index % 3])
            ax.imshow(rgb[index])
            projections = project_box_edges(
                corners, intrinsics[index], extrinsics[index], images.shape[-2:],
                img_aug_matrix=None if augmentation is None else augmentation[index],
            )
            visible_count = 0
            for box_index, segments in enumerate(projections):
                if not len(segments):
                    continue
                visible_count += 1
                color = palette[labels[box_index].item()]
                ax.add_collection(LineCollection(segments.numpy(), colors=color, linewidths=1.2))
                anchor = segments.reshape(-1, 2).mean(0)
                ax.text(float(anchor[0]), float(anchor[1]),
                        f"{classes[labels[box_index].item()]} {scores[box_index]:.2f}",
                        color=color, fontsize=7, bbox=dict(facecolor="black", alpha=.5, edgecolor="none"))
            ax.set_title(f"{index + 1}. {names[index]} | {visible_count} boxes")
            ax.set_xlim(-.5, images.shape[-1] - .5)
            ax.set_ylim(images.shape[-2] - .5, -.5)
            ax.set_axis_off()
        bev = figure.add_subplot(grid[2, :])
        bev.scatter(xyz[:, 0].numpy(), xyz[:, 1].numpy(), c=xyz[:, 2].numpy(),
                    cmap="viridis", s=.6, alpha=.65, edgecolors="none", rasterized=True)
        for index, box in enumerate(boxes):
            color = palette[labels[index].item()]
            polygon = corners[index, [0, 3, 7, 4, 0], :2]
            bev.plot(polygon[:, 0].numpy(), polygon[:, 1].numpy(), color=color, linewidth=1.4)
            front = corners[index, [0, 4], :2].mean(0)
            bev.plot([float(box[0]), float(front[0])], [float(box[1]), float(front[1])], color=color, linewidth=1.4)
            bev.text(float(box[0]), float(box[1]), f"{classes[labels[index].item()]} {scores[index]:.2f}",
                     color=color, fontsize=7)
        bev.plot(0, 0, "+", color="white", markersize=8)
        bev.set(xlim=(bev_range[0], bev_range[2]), ylim=(bev_range[1], bev_range[3]),
                aspect="equal", xlabel="LiDAR X (m)", ylabel="LiDAR Y (m)",
                title=f"LiDAR XY + detections | {len(boxes)} boxes at score >= {threshold:.2f} | {point_count:,} points")
        bev.grid(alpha=.2)
        displayed_labels = sorted(set(labels.tolist()))
        if displayed_labels:
            bev.legend(handles=[Patch(color=palette[i], label=classes[i]) for i in displayed_labels],
                       loc="upper left", ncol=min(5, len(displayed_labels)), fontsize=8)
        figure.suptitle(title or "BEVFusion detection | six cameras + LiDAR", fontsize=18)
        figure.supxlabel("LiDAR frame: +X right, +Y up | box Z is the bottom centre | scores shown per box")
        if save_path is not None:
            path = Path(save_path)
            if path.suffix.lower() != ".png":
                plt.close(figure)
                raise ValueError("save_path must end with .png")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as stream:
                    figure.savefig(stream, format="png", dpi=150, bbox_inches="tight")
            except BaseException:
                plt.close(figure)
                raise
        if show:
            plt.show()
    return figure
