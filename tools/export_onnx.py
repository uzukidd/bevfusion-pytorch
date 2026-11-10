"""Export BEVFusion det/seg to standard ONNX with dynamic point counts.

Images are normalized FP32 [1,6,3,256,704]; PNG decoding and preprocessing stay
outside the graph. Without --output, writes outputs/onnx_export/<checkpoint-stem>_<timestamp>.onnx.
Detection exports decoded bottom-centred boxes/scores/labels; segmentation exports
sigmoid probabilities. No CUDA custom ops required.
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import time
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bevfusion import BEVFusionDet, BEVFusionSeg, preprocess_images
from tools.onnx_model import OnnxBEVFusion

INPUT_NAMES = ["images", "points", "camera_intrinsics", "camera2lidar", "img_aug_matrix"]
OUTPUT_NAMES = {"seg": ["probabilities"], "det": ["boxes_3d", "scores", "labels"]}
BOX_COLUMNS = ["x", "y", "z_bottom", "size_x", "size_y", "size_z", "yaw", "vx", "vy"]
_LAST_EXPORT_TIME = None


def checkpoint_task(payload):
    """Inspect actual head keys rather than trusting the checkpoint filename."""
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state_dict mapping")
    state = payload.get("state_dict", payload)
    keys = [key.removeprefix("module.") for key in state]
    heads = {task for task, prefix in (("det", "heads.object."), ("seg", "heads.map."))
             if any(key.startswith(prefix) for key in keys)}
    if len(heads) != 1:
        raise ValueError("Expected exactly one supported det or seg head in checkpoint")
    task = heads.pop()
    if payload.get("task", task) != task:
        raise ValueError("Checkpoint task metadata conflicts with its head weights")
    return task


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-seg.pth")
    parser.add_argument("--task", choices=("auto", "seg", "det"), default="auto",
                        help="Auto detects the task from checkpoint weights")
    parser.add_argument("--config", type=Path, help="Detection architecture/operator config or segmentation operator config")
    parser.add_argument("--pool-backend", choices=("reference", "hat"), default=None,
                        help="Standard dynamic reference pooling or fixed-calibration full-model HAT export")
    parser.add_argument("--pool-num-points", type=int, default=None, help="HAT contributions per cell, 1..1024")
    parser.add_argument("--head-checkpoint", type=Path, help="Optional historical four-class segmentation head")
    parser.add_argument("--score-threshold", type=float,
                        help="Detection score filtering embedded in ONNX; default: checkpoint config")
    parser.add_argument("--output", type=Path,
                        help="Default: outputs/onnx_export/<checkpoint-stem>_<YYYYMMDD_HHMMSS_microseconds>.onnx")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--inputs", type=Path, help="Real sensor/calibration .pt sample for numerical verification")
    source.add_argument("--samples", type=Path, help="Six PNG cameras, LiDAR BIN and calibration directory")
    parser.add_argument("--calibration", type=Path, help="Optional calibration JSON for --samples")
    parser.add_argument("--verify-device", choices=("cpu", "cuda"), default="cpu", help="ONNX Runtime verification device")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PyTorch export device")
    parser.add_argument("--verify", action="store_true", help="Compare selected ONNX Runtime provider with PyTorch on dynamic inputs")
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing ONNX without overwriting it")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.pool_num_points is not None and not 1 <= args.pool_num_points <= 1024:
        parser.error("--pool-num-points must be in [1,1024]")
    if args.calibration is not None and args.samples is None:
        parser.error("--calibration requires --samples")
    if args.pool_backend == "hat" and args.inputs is None and args.samples is None:
        parser.error("HAT full-model export requires real --samples or --inputs to bind calibration")
    if args.output is None:
        if args.verify_only:
            parser.error("--verify-only requires --output to select the existing ONNX file")
        global _LAST_EXPORT_TIME
        current = datetime.now()
        if _LAST_EXPORT_TIME is not None and current <= _LAST_EXPORT_TIME:
            current = _LAST_EXPORT_TIME + timedelta(microseconds=1)
        _LAST_EXPORT_TIME = current
        timestamp = current.strftime("%Y%m%d_%H%M%S_%f")
        args.output = ROOT / "outputs" / "onnx_export" / f"{args.checkpoint.stem}_{timestamp}.onnx"
    if args.threads < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("--threads must be positive; tolerances must be nonnegative")
    if args.score_threshold is not None and not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be in [0,1]")
    if args.verify_only:
        args.verify = True
    if args.output.suffix.lower() != ".onnx" or (args.output.exists() and not args.verify_only):
        parser.error("--output must be a new .onnx file (or use --verify-only)")
    if args.verify_only and not args.output.is_file():
        parser.error("--verify-only requires an existing ONNX file")
    report_path = args.output.with_suffix(".verification.json")
    # Re-verifying an export must not require deleting its original report.
    if args.verify_only and report_path.exists():
        report_path = args.output.with_name(args.output.stem + ".recheck_" + str(time.time_ns()) + ".json")
    if report_path.exists():
        parser.error("Choose a fresh output name: verification report already exists")
    args.report_path = report_path
    return parser, args


def example_inputs(model, sample, device):
    if sample is None:
        images = torch.zeros((1, 6, 3, 256, 704), device=device)
        generator = torch.Generator().manual_seed(42)
        points = torch.rand((32, 5), generator=generator).to(device)
        points[:, :2] = points[:, :2] * 30 - 15
        points[:, 2] = points[:, 2] * 4 - 3
        points[:, 4] = 0
        intrinsic = torch.tensor([[500., 0., 352.], [0., 500., 128.], [0., 0., 1.]], device=device).repeat(1, 6, 1, 1)
        extrinsic = torch.eye(4, device=device).repeat(1, 6, 1, 1)
        augmentation = extrinsic.clone()
    else:
        images, points = sample["images"].to(device), sample["points"].float().to(device)
        if points.ndim != 2 or points.shape[1] not in (4, 5) or not torch.isfinite(points).all():
            raise ValueError("points must be finite float [P,4/5]")
        if points.shape[1] == 4:
            points = torch.cat((points, points.new_zeros((len(points), 1))), 1)
        intrinsic = sample.get("camera_intrinsics", model._camera_intrinsics)
        extrinsic = sample.get("camera2lidar", model._camera2lidar)
        augmentation = sample.get("img_aug_matrix", model._img_aug_matrix)
        if images.dtype == torch.uint8:
            images, augmentation = preprocess_images(images)
        else:
            images = images.float()
            if images.ndim == 4:
                images = images[None]
        if intrinsic is None or extrinsic is None or augmentation is None:
            raise ValueError("Verification input needs K/camera2lidar and augmentation (auto for uint8)")
        intrinsic = torch.as_tensor(intrinsic, device=device, dtype=torch.float32)
        extrinsic = torch.as_tensor(extrinsic, device=device, dtype=torch.float32)
        augmentation = torch.as_tensor(augmentation, device=device, dtype=torch.float32)
        if intrinsic.ndim == 3:
            intrinsic = intrinsic[None]
        if extrinsic.ndim == 3:
            extrinsic = extrinsic[None]
        if augmentation.ndim == 3:
            augmentation = augmentation[None]
        intrinsic = intrinsic[..., :3, :3]
    if tuple(images.shape) != (1, 6, 3, 256, 704) or not torch.isfinite(images).all():
        raise ValueError("images must be finite [1,6,3,256,704] normalized RGB")
    return images, points, intrinsic, extrinsic, augmentation


def _numpy_outputs(values, task):
    if task == "seg":
        values = (values,)
    elif isinstance(values, dict):
        values = tuple(values[key] for key in OUTPUT_NAMES[task])
    return [value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value for value in values]


def _validate_detection_arrays(values, name):
    import numpy as np
    boxes, scores, labels = values
    if not all(isinstance(value, np.ndarray) for value in values):
        raise AssertionError(f"{name} detection outputs must be NumPy arrays or tensors")
    if boxes.dtype != np.float32 or scores.dtype != np.float32 or labels.dtype != np.int64:
        raise AssertionError(f"{name} detection dtypes must be float32, float32, int64; got "
                             f"{boxes.dtype}, {scores.dtype}, {labels.dtype}")
    if boxes.ndim != 2 or boxes.shape[1] != 9 or scores.ndim != 1 or labels.ndim != 1:
        raise AssertionError(f"{name} detection shapes must be [N,9], [N], [N]")
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise AssertionError(f"{name} detection output lengths do not match")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise AssertionError(f"{name} detection outputs contain non-finite values")
    if ((scores < 0) | (scores > 1)).any() or (labels < 0).any() or (labels >= 10).any():
        raise AssertionError(f"{name} scores/labels are outside the nuScenes output contract")


def compare_outputs(actual, expected, task, atol, rtol):
    """Compare numeric outputs while matching unordered detection queries.

    Runtimes can permute TopK queries, especially under exact score ties. Match
    each query using its full box, yaw modulo 2*pi, score and class, not only XY.
    """
    import numpy as np
    actual, expected = _numpy_outputs(actual, task), _numpy_outputs(expected, task)
    if task == "seg":
        if len(actual) != 1 or len(expected) != 1:
            raise AssertionError("Segmentation must have exactly one output")
        if actual[0].dtype != np.float32 or expected[0].dtype != np.float32:
            raise AssertionError("Segmentation output must be float32")
        if actual[0].shape != expected[0].shape or actual[0].ndim != 4:
            raise AssertionError("Segmentation outputs must have matching [1,C,200,200] shape")
        if not np.isfinite(actual[0]).all() or not np.isfinite(expected[0]).all():
            raise AssertionError("Segmentation outputs contain non-finite values")
        difference = np.abs(actual[0] - expected[0])
        result = {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
                  "mask_disagreements_at_0.5": int(np.count_nonzero((actual[0] >= .5) != (expected[0] >= .5)))}
        np.testing.assert_allclose(actual[0], expected[0], atol=atol, rtol=rtol)
        return result
    if len(actual) != 3 or len(expected) != 3:
        raise AssertionError("Detection must return exactly boxes_3d, scores, labels")
    _validate_detection_arrays(actual, "ONNX")
    _validate_detection_arrays(expected, "PyTorch")
    result = {"boxes": len(actual[1]), "reference_boxes": len(expected[1]),
              "matching": "class/full-box/yaw/score", "excluded_zero_score": int((actual[1] == 0).sum()),
              "reference_excluded_zero_score": int((expected[1] == 0).sum())}
    a_keep, e_keep = actual[1] > 0, expected[1] > 0
    boxes, scores, labels = (v[a_keep] for v in actual)
    ref_boxes, ref_scores, ref_labels = (v[e_keep] for v in expected)
    if len(scores) != len(ref_scores):
        raise AssertionError(f"Different positive-score detection counts: {len(scores)} / {len(ref_scores)}")
    if not len(scores):
        return {**result, "box_max_abs": 0., "score_max_abs": 0.}
    delta = boxes[:, None, :] - ref_boxes[None, :, :]
    delta[..., 6] = np.arctan2(np.sin(delta[..., 6]), np.cos(delta[..., 6]))
    tolerance = np.broadcast_to(atol + rtol * np.abs(ref_boxes[None]), delta.shape).copy()
    tolerance[..., 6] = atol + rtol * np.abs(delta[..., 6])
    normalized = np.abs(delta) / np.maximum(tolerance, np.finfo(np.float32).tiny)
    score_tolerance = atol + rtol * np.abs(ref_scores[None, :])
    normalized_score = np.abs(scores[:, None] - ref_scores[None, :]) / np.maximum(
        score_tolerance, np.finfo(np.float32).tiny)
    cost = normalized.sum(-1) + normalized_score
    cost[labels[:, None] != ref_labels[None, :]] = np.inf
    pairs = []
    for _ in range(len(scores)):
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        if not np.isfinite(cost[i, j]):
            raise AssertionError("Detection classes/counts differ")
        pairs.append((i, j))
        cost[i, :] = np.inf
        cost[:, j] = np.inf
    ai, ei = np.array(pairs).T
    matched, target = boxes[ai].copy(), ref_boxes[ei]
    matched[:, 6] = target[:, 6] + np.arctan2(np.sin(matched[:, 6]-target[:, 6]),
                                            np.cos(matched[:, 6]-target[:, 6]))
    result.update(box_max_abs=float(np.abs(matched-target).max()),
                  score_max_abs=float(np.abs(scores[ai]-ref_scores[ei]).max()))
    np.testing.assert_allclose(matched, target, atol=atol, rtol=rtol)
    np.testing.assert_allclose(scores[ai], ref_scores[ei], atol=atol, rtol=rtol)
    return result


def assert_standard_operators(graph):
    import onnx
    for node in graph.node:
        if node.domain not in ("", "ai.onnx"):
            raise RuntimeError(f"Custom runtime operator found: {node.domain}::{node.op_type}")
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                assert_standard_operators(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for nested in attr.graphs:
                    assert_standard_operators(nested)


def validate_graph_contract(graph, task, classes):
    """Reject mismatched output types or class order, including verify-only."""
    import onnx
    expected_types = ([onnx.TensorProto.FLOAT] if task == "seg" else
                      [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT, onnx.TensorProto.INT64])
    outputs = graph.graph.output
    if [value.name for value in outputs] != OUTPUT_NAMES[task]:
        raise ValueError("ONNX output names do not match the checkpoint task")
    if [value.type.tensor_type.elem_type for value in outputs] != expected_types:
        raise ValueError("ONNX output element types do not match the det/seg contract")
    metadata = {item.key: item.value for item in graph.metadata_props}
    if metadata.get("task", task) != task:
        raise ValueError("ONNX task metadata does not match checkpoint")
    if json.loads(metadata.get("classes", "null")) != list(classes):
        raise ValueError("ONNX class metadata/order does not match checkpoint")
    if task == "det":
        shapes = [value.type.tensor_type.shape.dim for value in outputs]
        if len(shapes[0]) != 2 or shapes[0][1].dim_value != 9 or any(len(s) != 1 for s in shapes[1:]):
            raise ValueError("Detection ONNX output shapes must be [N,9], [N], [N]")


def resolve_export_pool(args, payload):
    """Resolve export semantics before forcing eager implementations for tracing.

    Native eager auto/cuda selections map to the existing standard reference
    graph. Explicit deployment backends never silently map to another graph.
    """
    import yaml
    packed = payload.get("config", {}) if isinstance(payload, dict) else {}
    packed = packed if isinstance(packed, dict) else {}
    if args.config is not None:
        with args.config.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        if not isinstance(config, dict) or not isinstance(config.get("operators", {}), dict):
            raise ValueError("--config must contain a mapping with optional operators")
        configured = config.get("operators", {})
        backend = configured.get("bev_pool")
        count = configured.get("bev_pool_num_points", 10)
    else:
        config = packed.get("config", {})
        operators = config.get("operators", {}) if isinstance(config, dict) else {}
        backend = packed.get("pool_backend", operators.get("bev_pool"))
        count = packed.get("pool_num_points", operators.get("bev_pool_num_points", 10))
    backend = args.pool_backend if args.pool_backend is not None else backend
    backend = "reference" if backend in (None, "reference", "auto", "cuda") else backend
    if backend not in ("reference", "hat"):
        raise ValueError(f"Full-model ONNX export does not support pooling backend {backend!r}; select reference or hat explicitly")
    count = args.pool_num_points if args.pool_num_points is not None else count
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1024:
        raise ValueError("HAT pool_num_points must be an integer in [1,1024]")
    if backend == "hat" and args.inputs is None and args.samples is None:
        raise ValueError("HAT full-model export requires real --samples or --inputs to bind calibration")
    return backend, count


def load_export_sample(args):
    if args.samples is not None:
        from bevfusion.sample_io import load_samples
        return load_samples(args.samples, calibration=args.calibration)
    return torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else None


def calibration_record(values):
    return {name: value.detach().cpu().tolist() for name, value in zip(INPUT_NAMES[2:], values[2:])}


def validate_export_input_contract(graph, backend, num_points, values, *, verify_only=False):
    """Verify existing graphs before evaluating them against a requested model."""
    import numpy as np
    names = INPUT_NAMES[:2] if backend == "hat" else INPUT_NAMES
    if [value.name for value in graph.graph.input] != names:
        raise ValueError("Existing ONNX input names do not match the requested calibration/pooling mode")
    if not verify_only:
        return
    metadata = {item.key: item.value for item in graph.metadata_props}
    if metadata.get("pool_backend", "reference") != backend:
        raise ValueError("Existing ONNX pool_backend differs from requested options")
    expected_mode = "fixed" if backend == "hat" else "dynamic"
    if metadata.get("calibration_mode", "dynamic") != expected_mode:
        raise ValueError("Existing ONNX calibration_mode differs from requested options")
    if backend == "hat":
        if metadata.get("pool_num_points") != str(num_points):
            raise ValueError("Existing ONNX pool_num_points differs from requested options")
        try:
            existing = json.loads(metadata["fixed_calibration"])
            for name, matrix in calibration_record(values).items():
                actual, expected = np.asarray(existing[name]), np.asarray(matrix)
                if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-6, rtol=0):
                    raise ValueError(f"Existing ONNX fixed calibration differs: {name}")
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Existing ONNX fixed_calibration metadata is missing or invalid") from exc


def operator_counts(graph):
    import onnx
    counts = {}
    for node in graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
        for attr in node.attribute:
            nested = [attr.g] if attr.type == onnx.AttributeProto.GRAPH else attr.graphs if attr.type == onnx.AttributeProto.GRAPHS else ()
            for child in nested:
                for name, count in operator_counts(child).items():
                    counts[name] = counts.get(name, 0) + count
    return counts


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        import onnx
        if args.verify:
            import onnxruntime as ort
    except ImportError as exc:
        parser.error(f"Missing export dependency {exc.name}; install it in the selected Conda environment")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    print("Python:", sys.executable, flush=True)
    print("[load]", args.checkpoint, flush=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        detected = checkpoint_task(payload)
    except ValueError as exc:
        parser.error(str(exc))
    task = detected if args.task == "auto" else args.task
    if task != detected:
        parser.error(f"--task {task} conflicts with the {detected} checkpoint")
    if task == "seg" and args.score_threshold is not None:
        parser.error("--score-threshold applies to detection only")
    if task == "det" and args.head_checkpoint is not None:
        parser.error("--head-checkpoint applies to segmentation only")
    try:
        pool_backend, num_points = resolve_export_pool(args, payload)
        sample = load_export_sample(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    options = {"backend": "reference", "pool_backend": pool_backend,
               "lidar_backend": "reference", "pool_num_points": num_points,
               "precompute": pool_backend == "hat", "camera_batch_size": 0}
    if args.config is not None:
        options["config"] = args.config
    if task == "seg":
        model = BEVFusionSeg.from_pretrained(args.checkpoint, head_checkpoint=args.head_checkpoint,
                                             device=args.device, **options)
    else:
        model = BEVFusionDet._load_pretrained(args.checkpoint, "bevfusion-det-v1", args.device,
                                              options, payload=payload)
    del payload
    threshold = args.score_threshold
    if task == "det" and threshold is None:
        threshold = model.config["model"]["heads"]["object"]["bbox_coder"]["score_threshold"]
    values = example_inputs(model, sample, args.device)
    if pool_backend == "hat":
        if not isinstance(sample, dict) or sample.get("camera_intrinsics") is None or sample.get("camera2lidar") is None:
            parser.error("HAT --inputs/--samples must explicitly provide six-camera intrinsics and extrinsics")
        if sample["images"].dtype != torch.uint8 and sample.get("img_aug_matrix") is None:
            parser.error("Normalized HAT input images require explicit img_aug_matrix")
        if any(tuple(value.shape[:2]) != (1, 6) or not torch.isfinite(value).all() for value in values[2:]):
            parser.error("HAT calibration must contain finite [1,6,...] camera matrices")
        from tools.onnx_model import FixedCalibrationOnnxBEVFusion
        wrapper = FixedCalibrationOnnxBEVFusion(model, calibration=values[2:],
                    score_threshold=threshold, pool_num_points=num_points).eval()
    else:
        wrapper = OnnxBEVFusion(model, score_threshold=threshold).eval()
    input_names = INPUT_NAMES[:2] if pool_backend == "hat" else INPUT_NAMES
    wrapper_values = values[:2] if pool_backend == "hat" else values
    dynamic_axes = {"points": {0: "num_points"}}
    if task == "det":
        dynamic_axes.update({name: {0: "num_boxes"} for name in OUTPUT_NAMES[task]})
    report = {"status": "exporting", "task": task, "checkpoint": str(args.checkpoint.resolve()),
              "python": sys.executable, "torch": str(torch.__version__), "onnx": onnx.__version__,
              "opset": 18, "classes": list(model.classes), "loaded": model.load_info,
              "inputs": dict(zip(input_names, [list(value.shape) for value in wrapper_values])),
              "dynamic_axes": dynamic_axes, "outputs": OUTPUT_NAMES[task],
              "input_contract": "normalized FP32 RGB; raw [P,5] XYZ/intensity/time_lag_seconds; per-frame K/extrinsics/augmentation"}

    report.update(pool_backend=pool_backend, pool_num_points=num_points,
                  calibration_mode="fixed" if pool_backend == "hat" else "dynamic",
                  scope="entire_model", lidar_backend="standard_dynamic", bpu_compatible=False)
    if pool_backend == "hat":
        report.update(fixed_calibration=calibration_record(values), sampling=wrapper.camera.pool.statistics(),
                      input_contract="normalized FP32 RGB; raw [P,5] XYZ/intensity/time_lag_seconds; fixed sensor calibration embedded in graph")
    if args.verify_only:
        existing_graph = onnx.load(str(args.output))
        validate_export_input_contract(existing_graph, pool_backend, num_points, values, verify_only=True)
        del existing_graph

    def reference(inputs):
        if task == "det":
            return model(*inputs, score_threshold=threshold)
        return model(*inputs)

    with torch.no_grad():
        print("[check]", task, "adapter vs original PyTorch model", flush=True)
        actual, expected = wrapper(*wrapper_values), reference(values)
        report["adapter_comparison"] = compare_outputs(actual, expected, task, 3e-4, 3e-4)
        if not args.verify_only:
            trace = example_inputs(model, None, args.device)
            trace = trace[:2] if pool_backend == "hat" else trace
            args.output.parent.mkdir(parents=True, exist_ok=True)
            print("[export]", task, "->", args.output, flush=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                torch.onnx.export(wrapper, tuple(trace), str(args.output), dynamo=False,
                                  opset_version=18, input_names=input_names, output_names=OUTPUT_NAMES[task],
                                  dynamic_axes=dynamic_axes, do_constant_folding=True,
                                  export_params=True, training=torch.onnx.TrainingMode.EVAL)
    print("[check] ONNX structure", flush=True)
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph, full_check=True)
    assert_standard_operators(graph.graph)
    metadata = {"task": task, "classes": json.dumps(list(model.classes)),
                "images": "normalized RGB float32 [1,6,3,256,704]",
                "points": "dynamic float32 [P,5], XYZ/intensity/time_lag_seconds",
                "calibration": "original-image K, camera2lidar, image augmentation"}
    metadata.update(pool_backend=pool_backend, pool_num_points=str(num_points),
                    calibration_mode=report["calibration_mode"], scope="entire_model",
                    lidar_backend="standard_dynamic", bpu_compatible="false")
    if pool_backend == "hat":
        metadata.update(fixed_calibration=json.dumps(report["fixed_calibration"]),
                        sampling=json.dumps(report["sampling"]),
                        calibration="fixed original-image K, camera2lidar, image augmentation",
                        input_contract=report["input_contract"])
    if task == "det":
        metadata.update(box_origin="bottom", box_format_version="2", box_columns=json.dumps(BOX_COLUMNS),
                        coordinate_frame="reference_lidar", score_threshold=str(threshold),
                        checkpoint_decoded_z_origin=model.heads["object"].decoded_z_origin)
    else:
        metadata["grid"] = "[batch,class,X,Y], 200x200, XY[-50,50), 0.5m"
    if not args.verify_only:
        onnx.helper.set_model_props(graph, metadata)
        onnx.save(graph, str(args.output))
    else:
        existing = {item.key: item.value for item in graph.metadata_props}
        actual_names = [value.name for value in graph.graph.output]
        if actual_names != OUTPUT_NAMES[task] or existing.get("task", task) != task:
            raise ValueError("Existing ONNX outputs/task do not match the checkpoint")
        if task == "det" and any(existing.get(key) != metadata[key] for key in (
                "box_origin", "box_format_version", "checkpoint_decoded_z_origin", "score_threshold")):
            raise ValueError("Existing ONNX box convention/threshold differs from the checkpoint options")
    validate_graph_contract(graph, task, model.classes)
    validate_export_input_contract(graph, pool_backend, num_points, values, verify_only=args.verify_only)
    report["operator_counts"] = operator_counts(graph.graph)
    report.update(status="exported", file_bytes=args.output.stat().st_size,
                  node_count=len(graph.graph.node), initializer_count=len(graph.graph.initializer),
                  standard_onnx_operators_only=True, runtime_verified=False)
    del graph
    if args.verify:
        print("[verify] loading ONNX Runtime", args.verify_device, "session", flush=True)
        from test_onnx import create_session, select_session_feed
        session = create_session(args.output, device=args.verify_device, threads=args.threads)
        cases = []
        report["tolerances"] = {"atol": args.atol, "rtol": args.rtol}
        inputs_cases = [("provided" if sample else "synthetic", values)]
        if sample is not None and pool_backend != "hat":
            inputs_cases.append(("synthetic_calibration", example_inputs(model, None, args.device)))
        if pool_backend == "hat":
            inputs_cases.append(("changed_images", (values[0] * .9 + .03, *values[1:])))
        inputs_cases.extend([
            ("empty_points", (values[0], values[1][:0], *values[2:])),
            ("changed_points", (values[0], values[1][:min(7, len(values[1]))] + values[1].new_tensor([.23,-.17,0,0,0]), *values[2:])),
        ])
        for label, inputs in inputs_cases:
            print("[verify]", label, "points=", inputs[1].shape[0], flush=True)
            try:
                with torch.no_grad():
                    expected = reference(inputs)
                feed = {name: value.detach().cpu().numpy() for name, value in zip(INPUT_NAMES, inputs)}
                output = session.run(OUTPUT_NAMES[task], select_session_feed(session, feed))
                case = {"name": label, "points": len(inputs[1]),
                        **compare_outputs(output[0] if task == "seg" else output, expected, task, args.atol, args.rtol)}
            except Exception as exc:
                case = {"name": label, "points": len(inputs[1]), "error": str(exc)}
                cases.append(case)
                report.update(status="verification_failed", runtime_verified=False,
                              verification_cases=cases, elapsed_seconds=time.perf_counter()-started)
                with args.report_path.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, indent=2)
                raise
            cases.append(case)
            print("[verify] result:", json.dumps(case), flush=True)
        report.update(runtime_verified=True, runtime=ort.__version__, providers=session.get_providers(), verification_cases=cases)
    report["elapsed_seconds"] = time.perf_counter() - started
    with args.report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
