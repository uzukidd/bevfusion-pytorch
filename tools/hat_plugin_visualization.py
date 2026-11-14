"""Render floating HAT versus Horizon-plugin predictions without loading the plugin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _save(figure, path, plt):
    if path.exists():
        plt.close(figure)
        raise FileExistsError(f"Refusing to overwrite visualization: {path}")
    try:
        figure.savefig(path, dpi=140, facecolor="white")
    finally:
        plt.close(figure)
    return str(path.resolve())


def _inputs(sample, directory, plt):
    path = directory / "inputs.png"
    if path.exists():
        # Both task comparisons share the exact same sensor snapshot.
        return str(path.resolve())
    images = _array(sample["images"])
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4 or images.shape[:2] != (6, 3):
        raise ValueError("Input montage requires images [6,3,H,W] or [1,6,3,H,W]")
    names = sample.get("camera_names") or [f"Camera {index + 1}" for index in range(6)]
    if len(names) != 6:
        raise ValueError("camera_names must have six entries")
    figure, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    for index, axis in enumerate(axes.flat):
        image = images[index].transpose(1, 2, 0)
        if images.dtype == np.uint8:
            displayed = image
        else:
            # Model float inputs use ImageNet normalization.
            displayed = np.clip(image * np.array([.229, .224, .225]) + np.array([.485, .456, .406]), 0, 1)
        axis.imshow(displayed)
        axis.set_title(str(names[index]), fontsize=10)
        axis.axis("off")
    figure.suptitle("Input cameras shared by HAT and hat_plugin", fontsize=14)
    return _save(figure, path, plt)


def _segmentation(expected, actual, classes, directory, plt):
    from matplotlib.colors import ListedColormap

    reference, candidate = _array(expected), _array(actual)
    if reference.ndim != 4 or reference.shape[0] != 1 or reference.shape[-2:] != (200, 200):
        raise ValueError("Segmentation visualization requires [1,C,200,200]")
    if candidate.shape != reference.shape or len(classes) != reference.shape[1]:
        raise ValueError("Segmentation outputs and class names must agree")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Segmentation outputs must be finite")
    reference, candidate = reference[0], candidate[0]
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    scale = max(float(difference.max()), 1e-12)
    count = reference.shape[0]
    figure, axes = plt.subplots(3, count, figsize=(3.1 * count, 9.0), squeeze=False, layout="constrained")
    colors = plt.get_cmap("tab10")
    extent = (-50, 50, -50, 50)
    heat = None
    for index, name in enumerate(classes):
        cmap = ListedColormap(["#f6f7f9", colors(index % 10)])
        disagreements = int(np.count_nonzero((reference[index] >= .5) != (candidate[index] >= .5)))
        for row, values in enumerate((reference[index] >= .5, candidate[index] >= .5, difference[index])):
            axis = axes[row, index]
            # Tensor axes are [X,Y]; transpose for horizontal X / vertical Y.
            heat = axis.imshow(values.T, origin="lower", extent=extent, interpolation="nearest",
                               cmap="magma" if row == 2 else cmap, vmin=0, vmax=scale if row == 2 else 1)
            axis.set_aspect("equal")
            axis.set_xticks([-50, 0, 50])
            axis.set_yticks([-50, 0, 50])
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(str(name).replace("_", " "), fontsize=11)
            if index == 0:
                axis.set_ylabel(("HAT mask (p >= 0.5)", "hat_plugin mask (p >= 0.5)", "Absolute probability error")[row]
                                + "\nY (m)", fontsize=10)
            if row == 2:
                axis.set_xlabel(f"X (m)\nmax={difference[index].max():.2e}; mask changes={disagreements}", fontsize=9)
    colorbar = figure.colorbar(heat, ax=axes[2, :].tolist(), shrink=.8, pad=.015)
    colorbar.set_label("Absolute probability difference (shared scale)", fontsize=9)
    colorbar.formatter.set_powerlimits((-2, 2))
    colorbar.update_ticks()
    total_changes = int(np.count_nonzero((reference >= .5) != (candidate >= .5)))
    figure.suptitle(f"Segmentation: independent class masks; max error={scale:.3e}; total mask changes={total_changes}\n"
                   "Each class is thresholded independently; overlapping classes are preserved.", fontsize=13)
    return _save(figure, directory / "seg_comparison.png", plt)


def _detections(prediction):
    values = {name: _array(prediction[name]) for name in ("boxes_3d", "scores", "labels")}
    boxes, scores, labels = (values[name] for name in ("boxes_3d", "scores", "labels"))
    if boxes.ndim != 2 or boxes.shape[1] != 9 or scores.shape != (len(boxes),) or labels.shape != scores.shape:
        raise ValueError("Detection visualization requires boxes [N,9], scores [N], labels [N]")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Detection boxes and scores must be finite")
    return boxes, scores, labels.astype(np.int64)


def _draw_boxes(axis, boxes, scores, labels, classes, colors, threshold):
    chosen = np.flatnonzero(scores > threshold)
    # A visual top-50 limit avoids illegible text; comparison metrics use all rows.
    chosen = chosen[np.argsort(-scores[chosen], kind="stable")[:50]]
    template = np.array([[-.5, -.5], [-.5, .5], [.5, .5], [.5, -.5], [-.5, -.5]])
    for index in chosen:
        box = boxes[index]
        local = template * box[3:5]
        cosine, sine = np.cos(box[6]), np.sin(box[6])
        # Match this checkpoint's clockwise LiDAR yaw convention.
        corners = np.column_stack((local[:, 0] * cosine + local[:, 1] * sine,
                                   -local[:, 0] * sine + local[:, 1] * cosine)) + box[:2]
        label = int(labels[index])
        color = colors(label % 10)
        axis.plot(corners[:, 0], corners[:, 1], color=color, linewidth=1.1)
        if len(chosen) <= 30:
            name = str(classes[label]) if 0 <= label < len(classes) else str(label)
            axis.text(box[0], box[1], f"{name} {scores[index]:.2f}", color=color, fontsize=5.5,
                      clip_on=True, bbox={"facecolor": "white", "alpha": .6, "edgecolor": "none", "pad": .3})
    return len(chosen), len(np.flatnonzero(scores > threshold))


def _detection(expected, actual, sample, classes, directory, plt):
    reference, candidate = _detections(expected), _detections(actual)
    points = _array(sample["points"])
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("LiDAR points must be [P,>=3]")
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    points = points[::max(1, int(np.ceil(len(points) / 30000)))]
    threshold = .2
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.4), layout="constrained")
    colors = plt.get_cmap("tab10")
    for axis, (boxes, scores, labels), name in zip(axes[:2], (reference, candidate), ("HAT", "hat_plugin")):
        axis.scatter(points[:, 0], points[:, 1], s=.2, c="#9aa3ae", alpha=.5, rasterized=True)
        shown, eligible = _draw_boxes(axis, boxes, scores, labels, classes, colors, threshold)
        axis.scatter([0], [0], s=25, c="black", marker="+")
        axis.set(xlim=(-54, 54), ylim=(-54, 54), xlabel="LiDAR X (m)", ylabel="LiDAR Y (m)",
                 title=f"{name}: score > {threshold:.1f}; {shown}/{eligible} boxes shown")
        axis.set_aspect("equal")
        axis.grid(alpha=.15)
    axis = axes[2]
    rb, rs, rl = reference
    cb, cs, cl = candidate
    if rb.shape == cb.shape and np.array_equal(rl, cl):
        delta = (cb[:, :2].astype(np.float64) - rb[:, :2].astype(np.float64)) * 1000
        score_error = cs.astype(np.float64) - rs.astype(np.float64)
        if len(delta):
            span = max(float(np.abs(delta).max()) * 1.15, .001)
            color_limit = max(float(np.abs(score_error).max()), 1e-12)
            dots = axis.scatter(delta[:, 0], delta[:, 1], c=score_error, cmap="coolwarm", s=14, alpha=.8,
                                vmin=-color_limit, vmax=color_limit)
            bar = figure.colorbar(dots, ax=axis, shrink=.7)
            bar.set_label("Score difference (hat_plugin - HAT)")
            axis.set_xlim(-span, span)
            axis.set_ylim(-span, span)
            max_box = float(np.abs(cb.astype(np.float64) - rb.astype(np.float64)).max())
            axis.set_title(f"All {len(delta)} aligned output rows\nmax box-component error={max_box:.2e}")
        else:
            axis.text(.5, .5, "Both predictions are empty", transform=axis.transAxes, ha="center")
        axis.set_xlabel("Center X difference (mm)")
        axis.set_ylabel("Center Y difference (mm)")
        axis.axhline(0, color="gray", linewidth=.6)
        axis.axvline(0, color="gray", linewidth=.6)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.2)
    else:
        axis.text(.5, .5, "Output row counts or labels differ.\nNo row-wise coordinate comparison performed.",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
    figure.suptitle("Detection comparison: same LiDAR snapshot; score/top-50 filtering is for display only", fontsize=13)
    return _save(figure, directory / "det_comparison.png", plt)


def render_comparison(task, expected, actual, sample, classes, output_dir):
    """Return absolute PNG paths for sensor inputs and backend prediction comparison.

    Predictions are floating, unquantized outputs. Segmentation preserves separate
    class membership; detection row differences are shown only when labels/counts
    agree and presume the verifier has already established row correspondence.
    """
    if task not in ("seg", "det"):
        raise ValueError("task must be seg or det")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{task}_comparison.png"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite visualization: {target}")
    plt = _pyplot()
    comparison = (_segmentation(expected, actual, classes, directory, plt) if task == "seg" else
                  _detection(expected, actual, sample, classes, directory, plt))
    return {f"{task}_comparison": comparison, "inputs": _inputs(sample, directory, plt)}


def main(argv=None):
    """Render saved verification predictions in a native Matplotlib environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=ROOT / "data/samples")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--inputs", type=Path, help="Optional original sensor snapshot saved as a PyTorch dictionary")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        import torch
        from bevfusion.sample_io import load_samples

        source = args.predictions_dir.expanduser().resolve(strict=True)
        directory = (args.output_dir or source / "figures").expanduser().resolve()
        if directory.exists():
            raise ValueError(f"Refusing to overwrite visualization directory: {directory}")
        predictions = {}
        for task in ("seg", "det"):
            path = source / f"{task}.predictions.pt"
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict) or not {"hat", "hat_plugin", "classes"}.issubset(payload):
                    raise ValueError(f"Saved predictions require hat, hat_plugin and classes: {path}")
                predictions[task] = (path, payload)
        if not predictions:
            raise ValueError(f"No seg.predictions.pt or det.predictions.pt snapshots found in {source}")
        sample = (torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else
                  load_samples(args.samples, calibration=args.calibration))
        manifest = {"scope": "floating_hat_plugin_visual_comparison", "quantized_accuracy_verified": False,
                    "predictions_dir": str(source), "sensor_inputs": str(args.inputs.resolve()) if args.inputs else None,
                    "samples": str(args.samples.resolve()) if not args.inputs else None,
                    "calibration": str(args.calibration.resolve()) if args.calibration else None, "tasks": {}}
        for task, (path, payload) in predictions.items():
            paths = render_comparison(task, payload["hat"], payload["hat_plugin"], sample, payload["classes"], directory)
            manifest["tasks"][task] = {"source_predictions": str(path), "images": paths}
        with (directory / "visualizations.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
