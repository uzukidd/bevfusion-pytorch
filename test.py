"""Minimal tensor inference example.

Only the example reads files. The BEVFusion forward method accepts tensors only.
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch

from bevfusion import BEVFusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Original six-class checkpoint or packed complete model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path,
                        help="torch.save dict: images, points, and calibration unless bound in model")
    source.add_argument("--samples", type=Path,
                        help="Directory containing six PNG images and one float32 BIN point cloud")
    parser.add_argument("--calibration", type=Path,
                        help="Original-image camera calibration JSON/PT; defaults to <samples>/calibration.json")
    parser.add_argument("--image-files", nargs=6, metavar="PNG",
                        help="Six image paths relative to --samples, in calibration order")
    parser.add_argument("--lidar-file", type=Path, help="BIN path relative to --samples; otherwise discover one BIN")
    parser.add_argument("--point-dim", type=int, choices=(4, 5), default=5,
                        help="Actual float32 columns per BIN point; default=5 (nuScenes)")
    parser.add_argument("--fifth-column", choices=("ring", "time-lag"), default="ring",
                        help="For five-column BIN: zero ring IDs for current frame, or preserve time lag in seconds")
    parser.add_argument("--save-inputs", type=Path, help="Optional new .pt file to save loaded sensor/calibration tensors")
    parser.add_argument("--head-checkpoint", type=Path,
                        help="Optional trained four-class head paired with the original base checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Optional output .pt containing probabilities and class names")
    parser.add_argument("--save-model", type=Path, help="Optional new file: complete model including bound calibration")
    parser.add_argument("--figure", type=Path, help="Visualization PNG; default: outputs/visualizations/<timestamp>.png")
    parser.add_argument("--threshold", type=float, default=0.5, help="Independent segmentation mask display threshold")
    parser.add_argument("--no-show", action="store_true", help="Save the Matplotlib figure without opening a GUI window")
    parser.add_argument("--no-visualization", action="store_true", help="Run inference without importing Matplotlib")
    args = parser.parse_args()
    if args.inputs and (args.calibration or args.image_files or args.lidar_file):
        parser.error("--calibration, --image-files and --lidar-file are options for --samples")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0,1]")
    if args.figure and args.no_visualization:
        parser.error("--figure cannot be combined with --no-visualization")
    if args.figure and (args.figure.suffix.lower() != ".png" or args.figure.exists()):
        parser.error("--figure must be a new .png file")
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.samples:
        from bevfusion.sample_io import load_samples
        try:
            sample = load_samples(args.samples, calibration=args.calibration,
                                  image_files=args.image_files, lidar_file=args.lidar_file,
                                  point_dim=args.point_dim, fifth_column=args.fifth_column)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        input_name = args.samples.name
        print("camera order:", sample["camera_names"])
    else:
        sample = torch.load(args.inputs, map_location="cpu", weights_only=True)
        input_name = args.inputs.stem
    if not isinstance(sample, dict) or not {"images", "points"}.issubset(sample):
        parser.error("--inputs must contain a dict with images and points tensors")
    print("input images:", tuple(sample["images"].shape), sample["images"].dtype)
    print("input points:", tuple(sample["points"].shape), sample["points"].dtype)
    calibration = {key: sample[key] for key in ("camera_intrinsics", "camera2lidar", "img_aug_matrix")
                   if key in sample}

    # 1. Load all model modules and optionally bind camera calibration once.
    model = BEVFusion.from_pretrained(args.checkpoint, head_checkpoint=args.head_checkpoint,
                                     device=args.device, **calibration)
    if model._camera_intrinsics is None or model._camera2lidar is None:
        parser.error("Camera calibration is required. Put calibration.json in --samples, pass --calibration, "
                     "or load a complete model with matching calibration already bound.")
    # 2. The actual model invocation needs only the two sensor tensors.
    with torch.inference_mode():
        probabilities = model(sample["images"], sample["points"])

    print("classes:", model.classes)
    print("probabilities:", tuple(probabilities.shape), probabilities.dtype, probabilities.device)
    print("range:", float(probabilities.min()), float(probabilities.max()))
    print("grid: [batch,class,increasing_X,increasing_Y], 200x200, 0.5 m/cell, XY in [-50,50)")
    if args.save_inputs:
        args.save_inputs.parent.mkdir(parents=True, exist_ok=True)
        with args.save_inputs.open("xb") as stream:
            torch.save(sample, stream)
        print("saved input tensors:", args.save_inputs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            torch.save({"probabilities": probabilities.cpu(), "classes": model.classes,
                        "axis_order": ["batch", "class", "X", "Y"], "resolution_m": 0.5}, stream)
        print("saved predictions:", args.output)
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_model)
        print("saved complete model:", args.save_model)
    if not args.no_visualization:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        from bevfusion.visualization import plot_prediction
        import matplotlib.pyplot as plt

        figure_path = args.figure or Path("outputs/visualizations") / (
            input_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png")
        figure = plot_prediction(sample["images"], sample["points"], probabilities, model.classes,
                                 camera_names=sample.get("camera_names"), threshold=args.threshold,
                                 save_path=figure_path, show=False)
        print("saved visualization:", figure_path, flush=True)
        if args.no_show:
            plt.close(figure)
        else:
            plt.show()


if __name__ == "__main__":
    main()
