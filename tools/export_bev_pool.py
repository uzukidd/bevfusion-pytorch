"""Export fixed-calibration BEV pooling for Nash-P compilation experiments.

Exports pooling before the learned downsample: det 360x360, seg 256x256.
This tool neither invokes a BPU compiler nor establishes BPU support.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bevfusion.base import _camera_matrix
from bevfusion.camera import LSSTransform
from bevfusion.depth import DepthLSSTransform
from bevfusion.operators.pooling.reference import build_pool_plan, pool_with_plan
from bevfusion.preprocessing import preprocess_images
from bevfusion.sample_io import load_calibration

INPUT_NAMES = ["depth", "features"]
OUTPUT_NAMES = ["bev"]
FORBIDDEN_OPS = {"Loop", "If", "Scan", "NonZero", "Unique", "TopK"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("det", "seg"), default="det")
    parser.add_argument("--calibration", type=Path, default=ROOT / "data/samples/calibration.json",
                        help="Original-image intrinsics and camera-to-reference-LiDAR matrices")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"), default=(900, 1600),
                        help="Original image dimensions; must match raw images in --inputs")
    parser.add_argument("--inputs", type=Path,
                        help="Optional calibration-source tensor dictionary; normalized images require img_aug_matrix")
    parser.add_argument("--output", type=Path,
                        help="Default: outputs/onnx_export/bevfusion-{task}-bevpool_<timestamp>.onnx")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32",
                        help="Input depth/features dtype; multiplication, accumulation and output remain FP32")
    parser.add_argument("--pool-backend", choices=("compatible", "hat"), default="compatible",
                        help="Exact compatible pooling or fixed-slot HAT-style GridSample adaptation")
    parser.add_argument("--num-points", type=int, default=10,
                        help="HAT-style contributions per cell, 1..1024; ignored by compatible pooling")
    parser.add_argument("--max-pairs-per-chunk", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--verify", action="store_true",
                        help="Verify CPU ORT semantics; HAT-style full-pooling differences are measured, not asserted equal")
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.image_size):
        parser.error("--image-size H W must contain positive integers")
    if min(int(value * .48) for value in args.image_size) < 1:
        parser.error("--image-size is too small for preprocess_images resize=0.48")
    if args.max_pairs_per_chunk < 1 or args.max_pairs_per_chunk > 65535 or args.threads < 1:
        parser.error("--max-pairs-per-chunk must be in [1,65535] and --threads must be positive")
    if args.pool_backend == "hat":
        if not 1 <= args.num_points <= 1024:
            parser.error("--num-points must be in [1,1024] for --pool-backend hat")
        if args.dtype != "float32":
            parser.error("--pool-backend hat requires --dtype float32; float16 is not silently cast")
    args.image_size = tuple(args.image_size)
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = f"-hat-lss-k{args.num_points}" if args.pool_backend == "hat" else ""
        args.output = ROOT / "outputs/onnx_export" / f"bevfusion-{args.task}-bevpool{suffix}_{timestamp}.onnx"
    args.output = args.output.expanduser()
    args.report_path = args.output.with_suffix(".verification.json")
    if args.output.suffix.lower() != ".onnx":
        parser.error("--output must have .onnx suffix")
    for path in (args.output, args.report_path):
        if path.exists():
            parser.error(f"Refusing to overwrite existing path: {path}")
    return parser, args


def _lidar_matrix(value):
    if value is None:
        return torch.eye(4, dtype=torch.float32)[None]
    matrix = torch.as_tensor(value, dtype=torch.float32).detach().clone()
    if matrix.shape == (4, 4):
        matrix = matrix[None]
    if matrix.shape != (1, 4, 4) or not torch.isfinite(matrix).all():
        raise ValueError("lidar_aug_matrix must be finite [4,4] or [1,4,4]")
    if torch.linalg.det(matrix[:, :3, :3]).abs().min() < 1e-8:
        raise ValueError("lidar_aug_matrix must contain invertible rotation/scale")
    if not torch.allclose(matrix[:, 3], matrix.new_tensor([[0, 0, 0, 1]]), atol=1e-6):
        raise ValueError("lidar_aug_matrix must have homogeneous last row [0,0,0,1]")
    return matrix


def resolve_calibration(args):
    """Use exactly the existing preprocessing implementation for raw-image aug."""
    if args.inputs is None:
        matrices = load_calibration(args.calibration)
        # A single image produces the same augmentation as all six; no formula
        # duplication of preprocessing resize/crop/padding is maintained here.
        raw = torch.zeros((1, 3, *args.image_size), dtype=torch.uint8)
        _, augmentation = preprocess_images(raw)
        augmentation = augmentation.expand(1, 6, 4, 4).clone()
        source = {"kind": "calibration", "path": str(args.calibration.resolve()),
                  "original_image_size": list(args.image_size),
                  "augmentation_source": "preprocess_images"}
        lidar = _lidar_matrix(None)
    else:
        sample = torch.load(args.inputs, map_location="cpu", weights_only=True)
        if not isinstance(sample, dict) or "images" not in sample:
            raise ValueError("--inputs must contain an images tensor and calibration matrices")
        images = sample["images"]
        if not isinstance(images, torch.Tensor):
            raise ValueError("--inputs images must be a Tensor")
        matrices = {key: _camera_matrix(sample.get(key), key, sizes) for key, sizes in (
            ("camera_intrinsics", ((3, 3), (4, 4))), ("camera2lidar", ((4, 4),)))}
        if any(value is None for value in matrices.values()):
            raise ValueError("--inputs requires camera_intrinsics and camera2lidar; no calibration fallback is used")
        if images.dtype == torch.uint8:
            if tuple(images.shape) not in ((6, 3, *args.image_size), (1, 6, 3, *args.image_size)):
                raise ValueError("Raw images must be [6,3,H,W] or [1,6,3,H,W] matching --image-size")
            normalized, augmentation = preprocess_images(images)
            if tuple(normalized.shape) != (1, 6, 3, 256, 704):
                raise ValueError("Raw preprocessing did not produce six [3,256,704] images")
            augmentation_source = "preprocess_images"
        else:
            if (not images.is_floating_point() or not torch.isfinite(images).all()
                    or tuple(images.shape) not in ((6, 3, 256, 704), (1, 6, 3, 256, 704))):
                raise ValueError("Normalized images must be finite floating [1,6,3,256,704] or [6,3,256,704]")
            augmentation = _camera_matrix(sample.get("img_aug_matrix"), "img_aug_matrix", ((4, 4),))
            if augmentation is None:
                raise ValueError("Normalized images require explicit img_aug_matrix")
            augmentation_source = "inputs.img_aug_matrix"
        lidar = _lidar_matrix(sample.get("lidar_aug_matrix"))
        source = {"kind": "inputs", "path": str(args.inputs.resolve()),
                  "original_image_size": list(args.image_size), "augmentation_source": augmentation_source}
    return {"camera_intrinsics": matrices["camera_intrinsics"][..., :3, :3].contiguous(),
            "camera2lidar": matrices["camera2lidar"].contiguous(),
            "img_aug_matrix": augmentation.contiguous(), "lidar_aug_matrix": lidar.contiguous()}, source


def make_transform(task):
    factory = DepthLSSTransform if task == "det" else LSSTransform
    # Only geometry is used. Random convolution weights are never exported or
    # evaluated; no complete camera backbone/checkpoint is constructed.
    return factory(image_size=(256, 704), pool_backend="reference", precompute=False).eval()


def tensor_sha256(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def inspect_graph(model):
    """Check standard ops, INT32 Gather indices and known Nash-P size limits.

    Shape inference may leave internal dimensions unknown. Those checks are
    counted as unverified, rather than claiming a fully static validated graph.
    """
    import onnx
    inferred = onnx.shape_inference.infer_shapes(model)
    counts = {}
    validation = {"gather_count": 0, "gather_source_axes_checked": 0, "gather_source_axes_unverified": 0,
                  "max_gather_source_axis": 0, "grid_sample_count": 0, "grid_sample_shapes_checked": 0,
                  "grid_sample_shapes_unverified": 0, "concat_count": 0, "max_concat_inputs": 0,
                  "reduce_sum_count": 0, "reduce_sum_axes_checked": 0, "reduce_sum_axes_unverified": 0,
                  "max_reduce_sum_axis": 0, "tensor_shapes_checked": 0, "max_tensor_dimension": 0,
                  "known_dimensions_checked": 0, "unknown_dimensions": 0, "unknown_rank_tensors": 0}

    def tensor_shape(value):
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            return None
        return tuple(dim.dim_value if dim.HasField("dim_value") else None for dim in tensor.shape.dim)

    def visit(graph):
        values = {value.name: value for value in (*graph.input, *graph.output, *graph.value_info)
                  if value.type.HasField("tensor_type")}
        types = {name: value.type.tensor_type.elem_type for name, value in values.items()}
        shapes = {name: tensor_shape(value) for name, value in values.items()}
        constants = {value.name: onnx.numpy_helper.to_array(value) for value in graph.initializer}
        for value in graph.initializer:
            types[value.name], shapes[value.name] = value.data_type, tuple(value.dims)
        for node in graph.node:
            if node.domain not in ("", "ai.onnx"):
                raise ValueError(f"Custom domain is forbidden: {node.domain}::{node.op_type}")
            if node.op_type in FORBIDDEN_OPS or node.op_type.startswith("Scatter"):
                raise ValueError(f"Forbidden pooling graph operation: {node.op_type}")
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
            for name in node.output:
                if name:
                    shapes.setdefault(name, None)
            attributes = {attr.name: attr for attr in node.attribute}
            if node.op_type == "Constant":
                attr = attributes.get("value")
                if attr is not None and attr.type == onnx.AttributeProto.TENSOR:
                    name = node.output[0]
                    types[name], shapes[name] = attr.t.data_type, tuple(attr.t.dims)
                    constants[name] = onnx.numpy_helper.to_array(attr.t)
            if node.op_type == "Cast" and "to" in attributes:
                types[node.output[0]] = attributes["to"].i
            if node.op_type in ("Gather", "GatherND"):
                validation["gather_count"] += 1
                dtype = types.get(node.input[1])
                if dtype != onnx.TensorProto.INT32:
                    raise ValueError(f"{node.op_type} index must be INT32: {node.name or node.output[0]} (type={dtype})")
                source = shapes.get(node.input[0])
                axis = attributes["axis"].i if "axis" in attributes else 0
                if node.op_type == "Gather" and source is not None and -len(source) <= axis < len(source):
                    length = source[axis]
                    if length is not None:
                        if length > 65536:
                            raise ValueError(f"Gather source axis exceeds 65536: {node.name or node.output[0]} ({length})")
                        validation["gather_source_axes_checked"] += 1
                        validation["max_gather_source_axis"] = max(validation["max_gather_source_axis"], length)
                    else:
                        validation["gather_source_axes_unverified"] += 1
                else:
                    validation["gather_source_axes_unverified"] += 1
            if node.op_type == "GridSample":
                validation["grid_sample_count"] += 1
                source, grid = shapes.get(node.input[0]), shapes.get(node.input[1])
                if source is None or grid is None or any(value is None for value in (*source, *grid)):
                    validation["grid_sample_shapes_unverified"] += 1
                else:
                    if len(source) != 4 or len(grid) != 4 or grid[-1] != 2 or source[0] != grid[0]:
                        raise ValueError(f"GridSample requires static 4D input and [N,H,W,2] grid: {node.name}")
                    for height, width in ((source[2], source[3]), (grid[1], grid[2])):
                        if not 1 <= height <= 1024 or not 1 <= width <= 1024 or height * width > 720 * 1024:
                            raise ValueError(f"GridSample spatial dimensions exceed candidate limits: {node.name} ({height},{width})")
                    validation["grid_sample_shapes_checked"] += 1
            if node.op_type == "Concat":
                count = len(node.input)
                if count > 1024:
                    raise ValueError(f"Concat has more than 1024 inputs: {node.name or node.output[0]} ({count})")
                validation["concat_count"] += 1
                validation["max_concat_inputs"] = max(validation["max_concat_inputs"], count)
            if node.op_type == "ReduceSum":
                validation["reduce_sum_count"] += 1
                source = shapes.get(node.input[0])
                axes = None
                if len(node.input) > 1 and node.input[1]:
                    if node.input[1] in constants:
                        axes = [int(value) for value in constants[node.input[1]].reshape(-1).tolist()]
                elif "axes" in attributes:
                    axes = list(attributes["axes"].ints)
                elif source is not None:
                    axes = list(range(len(source)))
                if axes is None:
                    validation["reduce_sum_axes_unverified"] += 1
                else:
                    if len(axes) != 1:
                        raise ValueError(f"ReduceSum must reduce exactly one axis: {node.name or node.output[0]} ({axes})")
                    axis = axes[0]
                    if source is None or not -len(source) <= axis < len(source) or source[axis] is None:
                        validation["reduce_sum_axes_unverified"] += 1
                    else:
                        length = source[axis]
                        if length > 65535:
                            raise ValueError(f"ReduceSum axis exceeds 65535: {node.name or node.output[0]} ({length})")
                        validation["reduce_sum_axes_checked"] += 1
                        validation["max_reduce_sum_axis"] = max(validation["max_reduce_sum_axis"], length)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    visit(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for child in attr.graphs:
                        visit(child)
        for name, shape in shapes.items():
            if shape is None:
                validation["unknown_rank_tensors"] += 1
                continue
            validation["tensor_shapes_checked"] += 1
            for length in shape:
                if length is None:
                    validation["unknown_dimensions"] += 1
                else:
                    if length >= 131072:
                        raise ValueError(f"Tensor dimension must be <131072: {name} ({length})")
                    validation["known_dimensions_checked"] += 1
                    validation["max_tensor_dimension"] = max(validation["max_tensor_dimension"], length)

    visit(inferred.graph)
    for value in (*model.graph.input, *model.graph.output):
        shape = tensor_shape(value)
        if shape is None or not shape or any(length is None or length < 1 for length in shape):
            raise ValueError(f"Pooling interface must have fixed positive dimensions: {value.name}")
    validation["all_tensor_dimensions_verified"] = not (validation["unknown_dimensions"] or validation["unknown_rank_tensors"])
    validation["all_gather_source_axes_verified"] = validation["gather_source_axes_unverified"] == 0
    validation["all_reduce_sum_axes_verified"] = validation["reduce_sum_axes_unverified"] == 0
    return {"operator_counts": counts, "gather_count": validation["gather_count"],
            "grid_sample_count": validation["grid_sample_count"],
            "all_gather_indices_int32": True, "standard_onnx_operators_only": True,
            "forbidden_operators_absent": True, "shape_validation": validation}


def random_inputs(module, dtype, seed, *, zero=False):
    generator = torch.Generator().manual_seed(seed)
    depth = torch.randn(module.depth_shape, generator=generator).softmax(dim=2).to(dtype)
    features = (torch.randn(module.feature_shape, generator=generator) * .3).to(dtype)
    if zero:
        depth.zero_()
        features.zero_()
    return depth, features


def selected_pool_numpy(coordinates, grid_shape, depth, features, num_points):
    """Independent first-K oracle from input coordinates, never module grids.

    Walk valid contributions in original B,N,D,H,W order and retain each cell's
    first K entries. NumPy add.at then accumulates depth-weighted features.
    """
    import numpy as np
    xyz = coordinates.detach().cpu().numpy().reshape(-1, 3)
    depth = np.asarray(depth, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    batch, cameras, bins, height, width = depth.shape
    channels = features.shape[2]
    nx, ny, nz = (int(value) for value in grid_shape)
    counts, selected_depth, selected_feature, selected_cells = {}, [], [], []
    per_camera, per_batch = bins * height * width, cameras * bins * height * width
    for rank, (x, y, z) in enumerate(xyz):
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            continue
        bi, within_batch = divmod(rank, per_batch)
        ni, within_camera = divmod(within_batch, per_camera)
        pixel = within_camera % (height * width)
        cell = int(((bi * nz + z) * nx + x) * ny + y)
        count = counts.get(cell, 0)
        counts[cell] = count + 1
        if count < num_points:
            selected_depth.append(rank)
            selected_feature.append((bi * cameras + ni) * height * width + pixel)
            selected_cells.append(cell)
    dense = np.zeros((batch * nz * nx * ny, channels), dtype=np.float32)
    if selected_depth:
        feature_rows = features.transpose(0, 1, 3, 4, 2).reshape(-1, channels)
        values = depth.reshape(-1)[selected_depth, None] * feature_rows[selected_feature]
        np.add.at(dense, selected_cells, values)
    return dense.reshape(batch, nz, nx, ny, channels).transpose(0, 1, 4, 2, 3).reshape(
        batch, nz * channels, nx, ny)


def verify_model(path, module, coordinates, grid_shape, dtype, threads):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    plan = build_pool_plan(coordinates, grid_shape, 16)
    statistics = module.statistics()
    is_hat = statistics.get("backend") == "hat"
    cases = []
    with torch.inference_mode():
        for index in range(3):
            name = f"random_{index + 1}" if index < 2 else "all_zero"
            depth, features = random_inputs(module, dtype, 20261105 + index, zero=index == 2)
            expected = pool_with_plan(plan, depth.float(), features.float(), "reference").numpy()
            actual_module = module(depth, features).numpy()
            output = session.run(OUTPUT_NAMES, dict(zip(INPUT_NAMES, (depth.numpy(), features.numpy()))))[0]
            if output.dtype != np.float32 or tuple(output.shape) != tuple(module.output_shape):
                raise AssertionError("ORT pooling output must be FP32 with the fixed pre-downsample shape")
            result = {"name": name}
            parity = []
            for key, value in (("module_vs_source", actual_module), ("ort_vs_source", output)):
                if not np.isfinite(value).all():
                    raise AssertionError(f"{name} {key} contains non-finite values")
                delta = np.abs(value - expected)
                close = bool(np.allclose(value, expected, atol=3e-4, rtol=3e-4))
                result[key] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                               "passed": close}
                parity.append(close)
                if not is_hat:
                    np.testing.assert_allclose(value, expected, atol=3e-4, rtol=3e-4)
            result["sourceparity_passed"] = all(parity)
            result["ort_vs_module_max_abs"] = float(np.abs(output - actual_module).max())
            np.testing.assert_allclose(output, actual_module, atol=3e-4, rtol=3e-4)
            if is_hat:
                selected = selected_pool_numpy(coordinates, grid_shape, depth.numpy(), features.numpy(),
                                               int(statistics["num_points"]))
                for key, value in (("module_vs_selected_oracle", actual_module),
                                   ("ort_vs_selected_oracle", output)):
                    delta = np.abs(value - selected)
                    result[key] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}
                    np.testing.assert_allclose(value, selected, atol=3e-4, rtol=3e-4)
                result["selected_semantics_passed"] = True
            cases.append(result)
            print("[verify]", json.dumps(result), flush=True)
    return {"runtime_verified": True, "runtime": ort.__version__, "providers": session.get_providers(),
            "runtime_verification_scope": "ORT/module/independent first-K NumPy oracle" if is_hat else "full-source parity",
            "sourceparity_passed": all(case["sourceparity_passed"] for case in cases),
            "selected_semantics_verified": is_hat,
            "tolerances": {"atol": 3e-4, "rtol": 3e-4}, "verification_cases": cases}


def main(argv=None):
    parser, args = parse_args(argv)
    torch.set_num_threads(args.threads)
    try:
        calibration, source = resolve_calibration(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    from bevfusion.operators.pooling.compatible import CompatibleBEVPool
    import onnx

    started = time.perf_counter()
    transform = make_transform(args.task)
    with torch.inference_mode():
        coordinates = transform._voxel_indices(transform.get_geometry(
            calibration["camera2lidar"], calibration["camera_intrinsics"],
            calibration["img_aug_matrix"], calibration["lidar_aug_matrix"]))
        grid_shape = tuple(int(value) for value in transform.nx)
        if args.pool_backend == "hat":
            from bevfusion.operators.pooling.hat import HATLSSPool
            module = HATLSSPool.from_transform(transform, calibration["camera2lidar"],
                calibration["camera_intrinsics"], calibration["img_aug_matrix"],
                calibration["lidar_aug_matrix"], num_points=args.num_points).eval()
        else:
            module = CompatibleBEVPool.from_transform(transform, calibration["camera2lidar"],
                calibration["camera_intrinsics"], calibration["img_aug_matrix"],
                calibration["lidar_aug_matrix"], max_pairs_per_chunk=args.max_pairs_per_chunk).eval()
    dtype = getattr(torch, args.dtype)
    contract = {"depth": {"shape": list(module.depth_shape), "dtype": args.dtype,
                           "meaning": "Per-camera depth probabilities; softmax along axis 2"},
                "features": {"shape": list(module.feature_shape), "dtype": args.dtype,
                             "meaning": "Per-camera predicted lift features, not backbone features"},
                "bev": {"shape": list(module.output_shape), "dtype": "float32",
                        "meaning": "Before learned downsample; axes B,(Z*C),X,Y"}}
    geometry_hashes = {"coordinates": tensor_sha256(coordinates), "frustum": tensor_sha256(transform.frustum),
                       "dx": tensor_sha256(transform.dx), "bx": tensor_sha256(transform.bx)}
    calibration_record = {key: {"matrix": value.tolist(), "sha256": tensor_sha256(value)}
                          for key, value in calibration.items()}
    report = {"status": "prepared", "task": args.task, "target": "nash-p_candidate", "bpu_compiled": False,
              "fixed_calibration": True, "calibration_source": source, "calibration": calibration_record,
              "geometry_sha256": geometry_hashes, "contracts": contract, "input_dtype": args.dtype,
              "accumulation_dtype": "float32", "output_dtype": "float32", "runtime_verified": False,
              "grid_profile": {"shape_xyz": list(grid_shape), "voxel_size_xyz": transform.dx.tolist(),
                               "voxel_center_origin_xyz": transform.bx.tolist(),
                               "processed_image_size": list(transform.image_size),
                               "feature_size": list(transform.feature_size), "channels": transform.C,
                               "depth_bins": transform.D, "stage": "before_downsample"},
              "statistics": module.statistics(), "opset": 18, "torch": torch.__version__, "onnx": onnx.__version__}
    report.update(pool_backend=args.pool_backend, backend=args.pool_backend, sourceparity_passed=None,
                  quantization_verified=False, board_verified=False)
    if args.pool_backend == "hat":
        report["sampling"] = {
            "num_points": args.num_points, "selection_policy": "first_depth_rank",
            "approximate": bool(report["statistics"]["dropped_points"]),
            "official_class": False, "official_exact_sort_key": False,
            "interpolation": report["statistics"].get("interpolation"),
            "align_corners": report["statistics"].get("align_corners"),
            "implementation": "official pattern adaptation; sampling order/interpolation are local implementation choices",
            "normalization": "none", "full_source_equivalence_claimed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[export] {args.task} pooling -> {args.output}", flush=True)
        print("[contract]", json.dumps(contract), flush=True)
        with torch.inference_mode(), args.output.open("xb") as stream:
            torch.onnx.export(module, random_inputs(module, dtype, 42), stream,
                              input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                              opset_version=18, dynamo=False)
        graph = onnx.load(str(args.output))
        onnx.checker.check_model(graph, full_check=True)
        report.update(inspect_graph(graph))
        metadata = {"task": args.task, "target": "nash-p_candidate", "bpu_compiled": "false",
                    "fixed_calibration": "true", "contracts": json.dumps(contract),
                    "calibration": json.dumps(calibration_record), "calibration_source": json.dumps(source),
                    "geometry_sha256": json.dumps(geometry_hashes), "grid_profile": json.dumps(report["grid_profile"]),
                    "statistics": json.dumps(report["statistics"]), "input_dtype": args.dtype,
                    "accumulation_dtype": "float32", "output_dtype": "float32",
                    "pool_backend": args.pool_backend, "backend": args.pool_backend, "quantization_verified": "false",
                    "sampling": json.dumps(report.get("sampling"))}
        onnx.helper.set_model_props(graph, metadata)
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, str(args.output))
        report.update(status="exported", file_bytes=args.output.stat().st_size)
        if args.verify:
            report.update(verify_model(args.output, module, coordinates, grid_shape, dtype, args.threads))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        with args.report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print("[report]", args.report_path, flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
