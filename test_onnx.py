"""ONNX Runtime version of test.py, with the same input and visualization flow.

中文：读取张量或六张 PNG + BIN，预处理后执行 ONNX，保存预测并用 Matplotlib 展示。
The ONNX graph must use tools/export_onnx.py's five-input segmentation contract.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from bevfusion import MAP_CLASSES, SURFACE_CLASSES, preprocess_images
from bevfusion.model import _camera_matrix
from bevfusion.sample_io import load_calibration, load_samples

INPUT_NAMES = ("images", "points", "camera_intrinsics", "camera2lidar", "img_aug_matrix")


def prepare_feed(sample):
    """Prepare CPU contiguous FP32 ONNX inputs without loading a PyTorch model."""
    if not isinstance(sample, dict) or not {"images", "points"}.issubset(sample):
        raise ValueError("Input must be a dictionary containing images and points tensors")
    images, points = sample["images"], sample["points"]
    if not isinstance(images, torch.Tensor):
        raise ValueError("images must be a Tensor")
    images = images.detach().cpu()
    if images.dtype == torch.uint8:
        images, augmentation = preprocess_images(images)
    else:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("Normalized images must be finite floating point values")
        augmentation = sample.get("img_aug_matrix")
        if augmentation is None:
            raise ValueError("Normalized float images require img_aug_matrix; "
                             "use explicit identity only when K describes processed images")
    if tuple(images.shape) != (1, 6, 3, 256, 704):
        raise ValueError("Expected six images [1,6,3,256,704] after preprocessing")
    if (not isinstance(points, torch.Tensor) or points.ndim != 2
            or points.shape[1] not in (4, 5) or not points.is_floating_point()
            or not torch.isfinite(points).all()):
        raise ValueError("points must be finite floating Tensor[P,4/5]")
    points = points.detach().cpu().float()
    if points.shape[1] == 4:
        points = torch.cat((points, points.new_zeros((len(points), 1))), dim=1)
    intrinsic = _camera_matrix(sample.get("camera_intrinsics"), "camera_intrinsics", ((3, 3), (4, 4)))
    extrinsic = _camera_matrix(sample.get("camera2lidar"), "camera2lidar", ((4, 4),))
    augmentation = _camera_matrix(augmentation, "img_aug_matrix", ((4, 4),))
    if intrinsic is None or extrinsic is None:
        raise ValueError("ONNX needs camera_intrinsics and camera2lidar for each frame; "
                         "provide them in --inputs or --calibration / <samples>/calibration.json")
    tensors = (images, points, intrinsic[..., :3, :3], extrinsic, augmentation)
    return {name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
            for name, value in zip(INPUT_NAMES, tensors)}


def create_session(model_path, *, device="cpu", threads=8):
    if threads < 1:
        raise ValueError("threads must be positive")
    provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}.get(device)
    if provider is None or provider not in ort.get_available_providers():
        raise ValueError(f"Requested provider {provider} is unavailable; installed providers: "
                         f"{ort.get_available_providers()}. Use --device cpu or a compatible GPU runtime.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    providers = [provider]
    if device == "cuda":
        # Each sparse-search Loop owns a subgraph session. Default per-subgraph
        # CUDA streams retain separate cuDNN/cuBLAS handles and can exhaust GPU
        # resources. Reuse the EP stream/handles across all subgraphs instead.
        providers = [(provider, {"use_ep_level_unified_stream": "1", "use_tf32": "0"}),
                     "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise ValueError("CUDA provider could not initialize; check ONNX Runtime CUDA dependencies")
    inputs = session.get_inputs()
    if {value.name for value in inputs} != set(INPUT_NAMES) or any(value.type != "tensor(float)" for value in inputs):
        raise ValueError("Expected the FP32 five-input ONNX exported by tools/export_onnx.py")
    if not any(value.name == "probabilities" for value in session.get_outputs()):
        raise ValueError("ONNX must expose the probabilities segmentation output")
    return session


def output_classes(session, count, override=None):
    classes = override
    if classes is None:
        encoded = session.get_modelmeta().custom_metadata_map.get("classes")
        if encoded is not None:
            try:
                classes = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise ValueError("ONNX class metadata is invalid JSON; pass --classes") from exc
        elif count in (4, 6):
            classes = SURFACE_CLASSES if count == 4 else MAP_CLASSES
        else:
            raise ValueError("ONNX has no class names; provide --classes in output channel order")
    if (not isinstance(classes, (list, tuple)) or len(classes) != count
            or any(not isinstance(name, str) or not name for name in classes)
            or len(set(classes)) != count):
        raise ValueError("Class names must be unique and match the ONNX output channels")
    return tuple(classes)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--checkpoint", dest="model", type=Path,
                        default=Path("outputs/onnx/bevfusion-seg.onnx"), help="Exported .onnx model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path, help="torch.save sensor tensor dictionary with calibration")
    source.add_argument("--samples", type=Path, help="Six PNGs, one BIN and optional calibration.json")
    parser.add_argument("--calibration", type=Path, help="Camera calibration JSON/PT override for either input mode")
    parser.add_argument("--image-files", nargs=6, metavar="PNG", help="PNG paths relative to --samples, in calibration order")
    parser.add_argument("--lidar-file", type=Path, help="BIN path relative to --samples")
    parser.add_argument("--point-dim", type=int, choices=(4, 5), default=5)
    parser.add_argument("--fifth-column", choices=("ring", "time-lag"), default="ring")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="ONNX Runtime provider; default CUDA")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--classes", nargs="+", help="Optional class names in ONNX output channel order")
    parser.add_argument("--save-inputs", type=Path, help="Save the original loaded sensor/calibration dictionary")
    parser.add_argument("--output", type=Path, help="Save probabilities and classes to a new .pt file")
    parser.add_argument("--figure", type=Path, help="Save visualization to a new PNG (default timestamped path)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-show", action="store_true", help="Save the figure using Agg without a GUI window")
    parser.add_argument("--no-visualization", action="store_true", help="Do not import Matplotlib")
    args = parser.parse_args()
    if args.inputs and (args.image_files or args.lidar_file):
        parser.error("--image-files and --lidar-file require --samples")
    if not args.model.is_file() or args.model.suffix.lower() != ".onnx":
        parser.error("--model must point to an existing .onnx file")
    if args.threads < 1 or not 0 <= args.threshold <= 1:
        parser.error("--threads must be positive and --threshold must be in [0,1]")
    if args.no_visualization and args.figure:
        parser.error("--figure cannot be combined with --no-visualization")
    for path in (args.save_inputs, args.output, args.figure):
        if path is not None and path.exists():
            parser.error(f"Refusing to overwrite existing output: {path}")
    if args.figure and args.figure.suffix.lower() != ".png":
        parser.error("--figure must end with .png")
    return parser, args


def main():
    parser, args = parse_args()
    torch.set_num_threads(args.threads)
    try:
        if args.samples:
            sample = load_samples(args.samples, calibration=args.calibration,
                                  image_files=args.image_files, lidar_file=args.lidar_file,
                                  point_dim=args.point_dim, fifth_column=args.fifth_column)
            input_name = args.samples.name
        else:
            sample = torch.load(args.inputs, map_location="cpu", weights_only=True)
            if not isinstance(sample, dict):
                raise ValueError("--inputs must contain a tensor dictionary")
            if args.calibration:
                sample = {**sample, **load_calibration(args.calibration)}
            input_name = args.inputs.stem
        feed = prepare_feed(sample)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if sample.get("camera_names"):
        print("camera order:", sample["camera_names"])
    print("input images:", feed["images"].shape, feed["images"].dtype)
    print("input points:", feed["points"].shape, feed["points"].dtype)
    print("[load] ONNX:", args.model, flush=True)
    try:
        session = create_session(args.model, device=args.device, threads=args.threads)
    except ValueError as exc:
        parser.error(str(exc))
    print("providers:", session.get_providers(), flush=True)
    started = time.perf_counter()
    array = session.run(["probabilities"], feed)[0]
    elapsed = time.perf_counter() - started
    if (array.ndim != 4 or array.shape[0] != 1 or array.shape[2:] != (200, 200)
            or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1):
        raise ValueError("ONNX output must be finite sigmoid probabilities [1,C,200,200]")
    try:
        classes = output_classes(session, array.shape[1], args.classes)
    except ValueError as exc:
        parser.error(str(exc))
    probabilities = torch.from_numpy(array)
    print("classes:", classes)
    print("probabilities:", tuple(probabilities.shape), probabilities.dtype)
    print("range:", float(array.min()), float(array.max()))
    print(f"ONNX inference: {elapsed:.3f} s")
    print("grid: [batch,class,increasing_X,increasing_Y], 200x200, 0.5 m/cell, XY in [-50,50)")
    if args.save_inputs:
        args.save_inputs.parent.mkdir(parents=True, exist_ok=True)
        with args.save_inputs.open("xb") as stream:
            torch.save(sample, stream)
        print("saved input tensors:", args.save_inputs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            torch.save({"probabilities": probabilities, "classes": classes,
                        "axis_order": ["batch", "class", "X", "Y"], "resolution_m": 0.5,
                        "backend": "onnxruntime", "model": str(args.model.resolve()),
                        "providers": session.get_providers(), "inference_seconds": elapsed}, stream)
        print("saved predictions:", args.output)
    if not args.no_visualization:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from bevfusion.visualization import plot_prediction
        figure_path = args.figure or Path("outputs/visualizations") / (
            input_name + "_onnx_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png")
        figure = plot_prediction(sample["images"], sample["points"], probabilities, classes,
                                 camera_names=sample.get("camera_names"), threshold=args.threshold,
                                 title=f"BEVFusion ONNX | six cameras + LiDAR + {len(classes)}-class segmentation",
                                 save_path=figure_path, show=False)
        print("saved visualization:", figure_path, flush=True)
        if args.no_show:
            plt.close(figure)
        else:
            plt.show()


if __name__ == "__main__":
    main()
