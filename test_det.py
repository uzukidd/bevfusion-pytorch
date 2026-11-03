"""Run BEVFusion 3D detection on six-camera/LiDAR tensors or PNG/BIN samples."""

import argparse
from datetime import datetime
from pathlib import Path
import time

import torch

from bevfusion import BEVFusionDet, OBJECT_CLASSES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path, help="torch.save dict containing images, points and optional calibration")
    source.add_argument("--samples", type=Path, help="Directory containing six PNG images and one float32 BIN point cloud")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/bevfusion-det.pth"))
    parser.add_argument("--config", type=Path, help="Optional detection YAML, e.g. config/convfuser.yaml")
    parser.add_argument("--backend", choices=("auto", "reference", "cuda"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration", type=Path, help="Original-image calibration JSON/PT; defaults to samples/calibration.json")
    parser.add_argument("--image-files", nargs=6, metavar="PNG", help="Six paths relative to --samples, in camera/calibration order")
    parser.add_argument("--lidar-file", type=Path, help="BIN path relative to --samples; otherwise discover one BIN")
    parser.add_argument("--point-dim", type=int, choices=(4, 5), default=5)
    parser.add_argument("--fifth-column", choices=("ring", "time-lag"), default="ring")
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="Filter returned/saved box scores; default: model configuration (original=0)")
    parser.add_argument("--threshold", type=float, default=.3, help="Display-only box score threshold; default=.3")
    parser.add_argument("--output", type=Path, help="Optional new .pt file containing boxes_3d, scores, labels and classes")
    parser.add_argument("--save-inputs", type=Path, help="Optional new .pt file containing the loaded sensor/calibration tensors")
    parser.add_argument("--save-model", type=Path, help="Optional new file containing complete model and bound calibration")
    parser.add_argument("--figure", type=Path, help="New visualization PNG; default: outputs/visualizations/det_<sample>_<timestamp>.png")
    parser.add_argument("--no-show", action="store_true", help="Save a figure without opening a GUI window")
    parser.add_argument("--no-visualization", action="store_true", help="Skip visualization and Matplotlib imports")
    args = parser.parse_args(argv)
    if args.inputs and (args.calibration or args.image_files or args.lidar_file):
        parser.error("--calibration, --image-files and --lidar-file are options for --samples")
    for name in ("score_threshold", "threshold"):
        if getattr(args, name) is not None and not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.figure and args.no_visualization:
        parser.error("--figure cannot be combined with --no-visualization")
    if args.figure and args.figure.suffix.lower() != ".png":
        parser.error("--figure must be a new .png file")
    for name in ("output", "save_inputs", "save_model", "figure"):
        destination = getattr(args, name)
        if destination is not None and destination.exists():
            parser.error(f"--{name.replace('_', '-')} must be a new file: {destination}")
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if args.samples:
            from bevfusion.sample_io import load_samples
            sample = load_samples(args.samples, calibration=args.calibration,
                                  image_files=args.image_files, lidar_file=args.lidar_file,
                                  point_dim=args.point_dim, fifth_column=args.fifth_column)
            input_name = args.samples.name
        else:
            sample = torch.load(args.inputs, map_location="cpu", weights_only=True)
            input_name = args.inputs.stem
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not isinstance(sample, dict) or not {"images", "points"}.issubset(sample):
        parser.error("--inputs must contain a dict with images and points tensors")
    if any(not isinstance(sample[name], torch.Tensor) for name in ("images", "points")):
        parser.error("images and points must be tensors")
    if "camera_names" in sample:
        print("camera order:", sample["camera_names"])
    print("input images:", tuple(sample["images"].shape), sample["images"].dtype)
    print("input points:", tuple(sample["points"].shape), sample["points"].dtype)
    calibration = {name: sample[name] for name in ("camera_intrinsics", "camera2lidar", "img_aug_matrix") if name in sample}
    load_options = dict(device=args.device, backend=args.backend, **calibration)
    if args.config is not None:
        load_options["config"] = args.config
    model = BEVFusionDet.from_pretrained(args.checkpoint, **load_options)
    if getattr(model, "_camera_intrinsics", None) is None or getattr(model, "_camera2lidar", None) is None:
        parser.error("Camera calibration is required. Supply samples/calibration.json, --calibration, "
                     "input tensor calibration, or a complete model with bound calibration.")
    score_threshold = args.score_threshold
    if score_threshold is None:
        score_threshold = model.config["model"]["heads"]["object"]["bbox_coder"]["score_threshold"]
    started = time.perf_counter()
    with torch.inference_mode():
        predictions = model(sample["images"], sample["points"], score_threshold=score_threshold)
    elapsed = time.perf_counter() - started
    classes = tuple(getattr(model, "classes", OBJECT_CLASSES))
    print("classes:", classes)
    print("boxes_3d:", tuple(predictions["boxes_3d"].shape), predictions["boxes_3d"].dtype)
    print("scores:", tuple(predictions["scores"].shape))
    print(f"inference time: {elapsed:.3f} s")
    print("boxes: [x, y, z_bottom, size_x, size_y, size_z, yaw, vx, vy] in reference LiDAR frame")
    if args.save_inputs:
        args.save_inputs.parent.mkdir(parents=True, exist_ok=True)
        with args.save_inputs.open("xb") as stream:
            torch.save(sample, stream)
        print("saved input tensors:", args.save_inputs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: predictions[name].detach().cpu() for name in ("boxes_3d", "scores", "labels")}
        payload.update(classes=classes, box_columns=("x", "y", "z_bottom", "size_x", "size_y", "size_z", "yaw", "vx", "vy"),
                       coordinate_frame="reference_lidar", score_threshold=score_threshold,
                       box_origin="bottom", box_format_version=2,
                       checkpoint_decoded_z_origin=model.config["model"]["heads"]["object"]["bbox_coder"].get(
                           "decoded_z_origin", "bottom"))
        with args.output.open("xb") as stream:
            torch.save(payload, stream)
        print("saved predictions:", args.output)
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_model)
        print("saved complete model:", args.save_model)
    if not args.no_visualization:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from bevfusion.detection_visualization import plot_detection
        figure_path = args.figure or Path("outputs/visualizations") / (
            "det_" + input_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png")
        figure = plot_detection(
            sample["images"], sample["points"], predictions, classes,
            camera_intrinsics=sample.get("camera_intrinsics", model._camera_intrinsics),
            camera2lidar=sample.get("camera2lidar", model._camera2lidar),
            img_aug_matrix=sample.get("img_aug_matrix", getattr(model, "_img_aug_matrix", None)),
            camera_names=sample.get("camera_names"), threshold=args.threshold,
            save_path=figure_path, show=False,
        )
        print("saved visualization:", figure_path, flush=True)
        if args.no_show:
            plt.close(figure)
        else:
            plt.show()
    return predictions


if __name__ == "__main__":
    main()
