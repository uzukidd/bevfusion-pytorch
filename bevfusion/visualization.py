"""Optional Matplotlib display for tensor-only BEVFusion inference.

English: camera order is preserved; all BEV panels use +X right, +Y up.
中文：保留输入相机顺序；点云与分割均显示为 X 向右、Y 向上的 LiDAR 俯视图。
Matplotlib is imported only when plotting, and is not a model dependency.
"""
from pathlib import Path

import torch


COLORS = {
    "sidewalk": "#d755b4", "road": "#41a55f", "vegetation": "#508228",
    "crosswalk": "#f5dc4b", "drivable_area": "#41a55f",
    "ped_crossing": "#f5dc4b", "walkway": "#d755b4",
    "stop_line": "#f56450", "carpark_area": "#5a9bf0", "divider": "#f0f0f0",
}


def plot_prediction(images, points, probabilities, classes, *, camera_names=None,
                    threshold=0.5, save_path=None, show=True, title=None,
                    max_plot_points=100000):
    """Plot six cameras, flattened LiDAR, segmentation and individual masks.

    images: uint8 RGB [6,3,H,W], or ImageNet-normalized float [6,3,256,704].
    A leading batch=1 is accepted. probabilities: [1,C,200,200] or [C,200,200].
    points: [P,4/5], in reference LiDAR coordinates. Height only controls colour;
    point positions are projected onto XY. No teacher labels are read.
    Combined map chooses the largest probability among masks >= threshold;
    cells below all thresholds remain dark. Per-class masks retain overlaps.
    Returns a Matplotlib Figure. PNG saves refuse to overwrite existing files.
    中文：可独立调用；浮点图像先反标准化，点云 Z 仅着色，原生 [X,Y] 转为显示 [Y,X]。
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, to_rgb
    from matplotlib.patches import Patch

    if not 0 <= threshold <= 1 or max_plot_points < 1:
        raise ValueError("threshold must be in [0,1] and max_plot_points must be positive")
    if not isinstance(images, torch.Tensor):
        raise ValueError("images must be a Tensor")
    images = images.detach().cpu()
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4 or images.shape[:2] != (6, 3):
        raise ValueError("Expected six RGB images [6,3,H,W] or [1,6,3,H,W]")
    raw = images.dtype == torch.uint8
    if raw:
        rgb = images.permute(0, 2, 3, 1).numpy()
    else:
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("Processed images must be finite ImageNet-normalized floats")
        mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        rgb = (images.float() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    names = [f"Camera {i + 1}" for i in range(6)] if camera_names is None else list(camera_names)
    if len(names) != 6 or any(not isinstance(name, str) for name in names):
        raise ValueError("camera_names must contain six names in tensor order")
    if not isinstance(probabilities, torch.Tensor):
        raise ValueError("probabilities must be a Tensor")
    probability = probabilities.detach().float().cpu()
    if probability.ndim == 4 and probability.shape[0] == 1:
        probability = probability[0]
    classes = tuple(classes)
    if probability.ndim != 3 or probability.shape != (len(classes), 200, 200) or not classes:
        raise ValueError("Expected probabilities [1,C,200,200] with C matching classes")
    if not torch.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] not in (4, 5):
        raise ValueError("Expected points [P,4/5]")
    xyz = points.detach().float().cpu()[:, :3]
    keep = torch.isfinite(xyz).all(1) & (xyz[:, :2] >= -50).all(1) & (xyz[:, :2] < 50).all(1)
    keep &= (xyz[:, 2] >= -5) & (xyz[:, 2] < 3)
    xyz = xyz[keep]
    visible_count = len(xyz)
    if visible_count > max_plot_points:
        step = (visible_count + max_plot_points - 1) // max_plot_points
        xyz = xyz[::step]
    palette = torch.tensor([to_rgb(COLORS.get(name, f"C{i % 10}")) for i, name in enumerate(classes)])
    best_probability, best_class = probability.max(0)
    active = best_probability >= threshold
    combined = torch.full((200, 200, 3), 0.065)
    combined[active] = palette[best_class[active]]
    # Native tensor is [X,Y]. imshow rows are Y and columns are X.
    combined_display = combined.permute(1, 0, 2).numpy()
    extent = (-50, 50, -50, 50)

    style = {
        "figure.facecolor": "#0e1621", "axes.facecolor": "#111d29",
        "savefig.facecolor": "#0e1621", "text.color": "#e4edf7",
        "axes.labelcolor": "#c5d2e1", "axes.edgecolor": "#526073",
        "xtick.color": "#aab9ca", "ytick.color": "#aab9ca",
        "font.size": 10, "axes.titlesize": 12, "grid.color": "#8193a5",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(18, 15), layout="constrained")
        grid = fig.add_gridspec(4, 6, height_ratios=(1.05, 1.05, 1.75, 0.85))
        for index in range(6):
            row, column = divmod(index, 3)
            ax = fig.add_subplot(grid[row, 2 * column:2 * column + 2])
            ax.imshow(rgb[index])
            kind = "RGB input" if raw else "processed input (denormalized)"
            ax.set_title(f"{index + 1}. {names[index]} | {kind}")
            ax.set_axis_off()

        def bev_axes(ax, heading):
            ax.set(xlim=(-50, 50), ylim=(-50, 50), aspect="equal", title=heading,
                   xlabel="LiDAR X (m)  ->", ylabel="LiDAR Y (m)  ->")
            ax.set_xticks([-50, -25, 0, 25, 50])
            ax.set_yticks([-50, -25, 0, 25, 50])
            ax.grid(alpha=0.2, linewidth=0.6)
            ax.plot(0, 0, "+", color="white", markersize=9, markeredgewidth=1.2)

        ax_cloud = fig.add_subplot(grid[2, 0:2])
        scatter = ax_cloud.scatter(xyz[:, 0].numpy(), xyz[:, 1].numpy(), c=xyz[:, 2].numpy(),
                                   cmap="viridis", norm=Normalize(-5, 3), s=0.7, alpha=0.8,
                                   edgecolors="none", rasterized=True)
        bev_axes(ax_cloud, f"Flattened LiDAR XY | {visible_count:,} points in grid")
        colorbar = fig.colorbar(scatter, ax=ax_cloud, shrink=0.72, pad=0.02)
        colorbar.set_label("LiDAR height Z (m)")

        ax_seg = fig.add_subplot(grid[2, 2:4])
        ax_seg.imshow(combined_display, origin="lower", extent=extent, interpolation="nearest")
        bev_axes(ax_seg, f"Segmentation | highest active class, threshold {threshold:.2f}")

        ax_overlay = fig.add_subplot(grid[2, 4:6])
        ax_overlay.imshow(combined_display, origin="lower", extent=extent, interpolation="nearest")
        ax_overlay.scatter(xyz[:, 0].numpy(), xyz[:, 1].numpy(), c="white", s=0.4,
                           alpha=0.3, edgecolors="none", rasterized=True)
        bev_axes(ax_overlay, "Segmentation + flattened LiDAR")
        handles = [Patch(facecolor=palette[i].numpy(), label=name) for i, name in enumerate(classes)]
        handles.append(Patch(facecolor="#111111", label="No mask above threshold"))
        legend = ax_seg.legend(handles=handles, loc="lower left", fontsize=8,
                               facecolor="#132131", edgecolor="#526073", framealpha=0.9)
        for text in legend.get_texts():
            text.set_color("#e4edf7")

        mask_grid = grid[3, :].subgridspec(1, len(classes))
        for index, name in enumerate(classes):
            ax = fig.add_subplot(mask_grid[0, index])
            mask = torch.full((200, 200, 3), 0.065)
            mask[probability[index] >= threshold] = palette[index]
            ax.imshow(mask.permute(1, 0, 2).numpy(), origin="lower", extent=extent,
                      interpolation="nearest")
            ax.set_title(name, fontsize=10)
            ax.set(xlim=(-50, 50), ylim=(-50, 50), aspect="equal")
            ax.set_xticks([-50, 0, 50])
            ax.set_yticks([-50, 0, 50])
            ax.tick_params(labelsize=8)
        fig.suptitle(title or f"BEVFusion | six cameras + LiDAR + {len(classes)}-class segmentation", fontsize=18)
        fig.supxlabel("Reference LiDAR frame: +X right, +Y up | XY [-50,50) m | 0.5 m/cell | "
                       "bottom row: independent masks (overlaps preserved)", fontsize=11)
        if save_path is not None:
            path = Path(save_path)
            if path.suffix.lower() != ".png":
                plt.close(fig)
                raise ValueError("save_path must end with .png")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as stream:
                    fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
            except BaseException:
                plt.close(fig)
                raise
        if show:
            plt.show()
    return fig
