"""Export CPU-planned voxel mean and sparse encoder subgraphs for Nash-P checks.

Plans are fresh runtime inputs, never frozen point-cloud topology. This tool
exports standard ONNX and verifies host execution; it does not compile HBM.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bevfusion import BEVFusionDet, BEVFusionSeg
from bevfusion.operators.sparse_conv.compatible import CompatibleSparseEncoder
from bevfusion.operators.voxelization.compatible import CompatibleVoxelMean, build_voxel_plan
from bevfusion.operators.sparse_conv.reference import ReferenceSparseConv
from bevfusion.preprocessing import prepare_points, voxelize
from bevfusion.sample_io import read_lidar_bin
from bevfusion.operators.sparse_conv.plan import plan_sparse_encoder
from tools.export_bev_pool import inspect_graph

VOXEL_INPUT_NAMES = ["points", "point_indices", "inverse_counts"]
ATOL = RTOL = 3e-4


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("det", "seg"), default="det")
    parser.add_argument("--checkpoint", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--samples", type=Path, help="Directory with exactly one LiDAR BIN; no images are loaded")
    source.add_argument("--inputs", type=Path, help="Tensor dictionary containing raw points[P,4/5]")
    source.add_argument("--point-file", type=Path, help="Raw little-endian float32 LiDAR BIN")
    parser.add_argument("--point-dim", type=int, choices=(4, 5), default=5)
    parser.add_argument("--fifth-column", choices=("ring", "time-lag"), default="ring")
    parser.add_argument("--point-capacity", type=int, default=65535)
    parser.add_argument("--voxel-capacity", type=int, default=32768)
    parser.add_argument("--capacity", type=int, default=32768, help="Fixed output capacity for every sparse convolution")
    parser.add_argument("--page-size", type=int, default=32768)
    parser.add_argument("--output-dir", type=Path, help="New directory only; default outputs/onnx_export/lidar_<task>_<timestamp>")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify", action="store_true", help="CPU ORT plus source encoder comparison on two freshly planned frames")
    args = parser.parse_args(argv)
    for name in ("point_capacity", "voxel_capacity", "capacity"):
        if not 1 <= getattr(args, name) <= 131071:
            parser.error(f"--{name.replace('_', '-')} must be in [1,131071]")
    if not 1 <= args.page_size <= 65535 or args.threads < 1:
        parser.error("--page-size must be in [1,65535] and --threads must be positive")
    args.checkpoint = (args.checkpoint or ROOT / f"data/checkpoints/bevfusion-{args.task}.pth").expanduser()
    if args.samples is None and args.inputs is None and args.point_file is None:
        args.samples = ROOT / "data/samples"
    args.output_dir = (args.output_dir or ROOT / "outputs/onnx_export" /
                       f"lidar_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}").expanduser()
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite existing output directory: {args.output_dir}")
    return parser, args


def load_points(args):
    if args.inputs is not None:
        payload = torch.load(args.inputs, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "points" not in payload:
            raise ValueError("--inputs must contain a raw points tensor")
        points = prepare_points(payload["points"]).detach().cpu().contiguous()
        source = {"kind": "inputs", "path": str(args.inputs.resolve())}
    else:
        path = args.point_file
        if path is None:
            if not args.samples.is_dir():
                raise ValueError(f"Sample directory does not exist: {args.samples}")
            candidates = sorted(p for p in args.samples.iterdir() if p.is_file() and p.suffix.lower() == ".bin")
            if len(candidates) != 1:
                raise ValueError(f"Expected one BIN in {args.samples}, found {len(candidates)}; use --point-file")
            path = candidates[0]
        points = read_lidar_bin(path, point_dim=args.point_dim, fifth_column=args.fifth_column)
        source = {"kind": "point_file", "path": str(path.resolve()),
                  "point_dim": args.point_dim, "fifth_column": args.fifth_column}
    return points, source


def load_backbone(args):
    factory = BEVFusionDet if args.task == "det" else BEVFusionSeg
    model = factory.from_pretrained(args.checkpoint, device="cpu", backend="reference",
                                    lidar_backend="reference", pool_backend="reference", precompute=False)
    # Drop camera, fuser and heads immediately; only learned LiDAR weights export.
    return (model.encoders["lidar"]["backbone"].eval(), tuple(model.voxel_size),
            tuple(model.point_cloud_range), int(model.max_voxels))


def make_frame(points, backbone, args, voxel_size, point_cloud_range, max_voxels, voxel_module, sparse_module=None):
    started = time.perf_counter()
    voxel_plan = build_voxel_plan(points, voxel_size=voxel_size, point_cloud_range=point_cloud_range,
                                  max_points=10, max_voxels=max_voxels,
                                  point_capacity=args.point_capacity, voxel_capacity=args.voxel_capacity)
    capacities = {name: args.capacity for name, module in backbone.named_modules()
                  if isinstance(module, ReferenceSparseConv)}
    encoder_plan = plan_sparse_encoder(backbone, voxel_plan.coordinates,
                                       capacities=capacities, input_capacity=args.voxel_capacity)
    planning_seconds = time.perf_counter() - started
    if sparse_module is None:
        sparse_module = CompatibleSparseEncoder(backbone, encoder_plan, page_size=args.page_size).eval()
    with torch.inference_mode():
        voxel_inputs = voxel_plan.tensor_feed("cpu")
        padded_features = voxel_module(*voxel_inputs)
        reference_features, reference_coords = voxelize(points, voxel_size=voxel_size,
            point_cloud_range=point_cloud_range, max_points=10, max_voxels=max_voxels, backend="reference")
        if len(reference_features) != voxel_plan.voxel_count:
            raise AssertionError("Voxel count differs from source reference")
        torch.testing.assert_close(voxel_plan.coordinates.long(), reference_coords, atol=0, rtol=0)
        torch.testing.assert_close(padded_features[:voxel_plan.voxel_count], reference_features,
                                   atol=ATOL, rtol=RTOL)
        if torch.count_nonzero(padded_features[voxel_plan.voxel_count:]):
            raise AssertionError("Padded voxel features must be zero")
    feed = dict(zip(VOXEL_INPUT_NAMES, voxel_inputs))
    feed.update(zip(sparse_module.input_names, (padded_features, *sparse_module.plan_inputs(encoder_plan, "cpu"))))
    stats = {"cpu_planning_seconds": planning_seconds, "voxel": voxel_plan.statistics(),
             "sparse_convolutions": {name: {"input_count": plan.input_count, "output_count": plan.output_count,
                  "input_capacity": plan.input_capacity, "output_capacity": plan.output_capacity}
                  for name, plan in encoder_plan.plans.items()},
             "final_shape_xyz": list(encoder_plan.final_shape), "final_count": encoder_plan.output_count,
             "dense_volume": int(encoder_plan.dense_lookup.numel()), "voxel_module_vs_source_verified": True}
    return sparse_module, voxel_plan, encoder_plan, feed, stats


def _contract(names, tensors):
    return [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
            for name, tensor in zip(names, tensors)]


def export_graph(module, tensors, names, output_name, path):
    import onnx
    with torch.inference_mode(), path.open("xb") as stream:
        torch.onnx.export(module, tuple(tensors), stream, input_names=list(names), output_names=[output_name],
                          opset_version=18, dynamo=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph, full_check=True)
    # CPU planning produces per-frame INT32 inputs. Reuse the pooling audit for
    # standard domains, forbidden ops and bounded Gather/ReduceSum dimensions.
    inspected = inspect_graph(graph)
    actual = [value.name for value in graph.graph.input]
    if actual != list(names):
        raise AssertionError(f"Runtime plan inputs were removed or renamed: {actual} / {names}")
    expected_dtypes = {name: tensor.dtype for name, tensor in zip(names, tensors)}
    for value in graph.graph.input:
        expected = onnx.TensorProto.INT32 if expected_dtypes[value.name] == torch.int32 else onnx.TensorProto.FLOAT
        if value.type.tensor_type.elem_type != expected:
            raise AssertionError(f"Wrong ONNX input dtype: {value.name}")
    onnx.helper.set_model_props(graph, {"target": "nash-p_candidate", "bpu_compiled": "false",
        "cpu_planning_required": "true", "capacity_overflow": "raise", "plan_tensors": "runtime_inputs",
        "stage": path.stem, "input_contract": json.dumps(_contract(names, tensors))})
    onnx.save(graph, str(path))
    onnx.checker.check_model(graph, full_check=True)
    output_contract = [{"name": value.name,
                        "shape": [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim],
                        "dtype": "float32" if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT else str(value.type.tensor_type.elem_type)}
                       for value in graph.graph.output]
    return {"path": path.name, "file_bytes": path.stat().st_size,
            "inputs": _contract(names, tensors), "outputs": output_contract,
            "input_names": list(names), "output_name": output_name, **inspected}


def _compare(actual, expected):
    expected = expected.detach().cpu().numpy() if isinstance(expected, torch.Tensor) else expected
    if actual.shape != expected.shape or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise AssertionError("Output must be finite FP32 with reference shape")
    delta = np.abs(actual - expected)
    metrics = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
    return metrics


def verify_frames(args, points, backbone, voxel_module, sparse_module, profile, initial):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = args.threads, 1
    sessions = {name: ort.InferenceSession(str(args.output_dir / f"{name}.onnx"), sess_options=options,
                   providers=["CPUExecutionProvider"]) for name in ("voxel_mean", "sparse_encoder")}
    cases, previous_plan = [], None
    for index in range(2):
        if index == 0:
            voxel_plan, encoder_plan, feed, statistics = initial
        else:
            changed = points.clone()
            changed[:, :3] += changed.new_tensor([.137, -.137, .137])
            _, voxel_plan, encoder_plan, feed, statistics = make_frame(
                changed, backbone, args, *profile, voxel_module, sparse_module)
            if not voxel_plan.voxel_count or not encoder_plan.output_count:
                raise AssertionError("Changed-point verification frame must contain nonempty voxel and sparse outputs")
            np.savez_compressed(args.output_dir / "sample_inputs_changed.npz",
                                **{key: value.detach().cpu().numpy() for key, value in feed.items()})
            if torch.equal(previous_plan.coordinates, voxel_plan.coordinates):
                raise AssertionError("Translation did not change voxel coordinates; runtime-plan freshness is unverified")
        previous_plan = voxel_plan
        voxel_feed = {name: feed[name].numpy() for name in VOXEL_INPUT_NAMES}
        voxel_actual = sessions["voxel_mean"].run(None, voxel_feed)[0]
        sparse_feed = {name: feed[name].numpy() for name in sparse_module.input_names}
        with torch.inference_mode():
            expected_voxels = voxel_module(*(feed[name] for name in VOXEL_INPUT_NAMES))
            expected_sparse = sparse_module(*(feed[name] for name in sparse_module.input_names))
            source_sparse = backbone(expected_voxels[:voxel_plan.voxel_count], voxel_plan.coordinates, verbose=False)
        sparse_actual = sessions["sparse_encoder"].run(None, sparse_feed)[0]
        chained_feed = {**sparse_feed, "features": voxel_actual}
        chained = sessions["sparse_encoder"].run(None, chained_feed)[0]
        case = {"name": "provided" if index == 0 else "translated_points", "statistics": statistics,
                "voxel_ort_vs_module": _compare(voxel_actual, expected_voxels),
                "sparse_module_vs_source": _compare(expected_sparse.numpy(), source_sparse),
                "sparse_ort_vs_module": _compare(sparse_actual, expected_sparse),
                "sparse_ort_vs_source": _compare(sparse_actual, source_sparse),
                "chained_ort_vs_source": _compare(chained, source_sparse)}
        cases.append(case)
        print("[verify]", json.dumps(case), flush=True)
    return {"runtime_verified": True, "runtime": ort.__version__, "verification_cases": cases,
            "source_sparse_reference_verified": True, "tolerances": {"atol": ATOL, "rtol": RTOL}}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser, args = parse_args(argv)
    torch.set_num_threads(args.threads)
    try:
        points, source = load_points(args)
        backbone, voxel_size, point_cloud_range, max_voxels = load_backbone(args)
        voxel_module = CompatibleVoxelMean(args.point_capacity, args.voxel_capacity, 10).eval()
        sparse_module, voxel_plan, encoder_plan, feed, statistics = make_frame(
            points, backbone, args, voxel_size, point_cloud_range, max_voxels, voxel_module)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "prepared", "task": args.task, "target": "nash-p_candidate", "bpu_compiled": False,
              "cpu_planning_required": True, "capacity_overflow": "raise", "runtime_verified": False,
              "source_sparse_reference_verified": False, "source": source,
              "source_checkpoint": str(args.checkpoint.resolve()),
              "source_checkpoint_sha256": sha256_file(args.checkpoint),
              "profile": {"voxel_size": list(voxel_size), "point_cloud_range": list(point_cloud_range),
                          "max_points": 10, "max_voxels": max_voxels, "page_size": args.page_size},
              "sentinels": {"voxel_mean": "point_capacity: appended zero point",
                            "sparse_encoder": "0: missing neighbor; valid input row i encoded i+1"},
              "runtime_plan_contract": "Rebuild voxel grouping and sparse topology on CPU whenever point XYZ/order changes; capacities stay fixed",
              "statistics": statistics, "opset": 18, "torch": torch.__version__}
    try:
        np.savez_compressed(args.output_dir / "sample_inputs.npz",
                            **{key: value.detach().cpu().numpy() for key, value in feed.items()})
        report["sample_inputs"] = "sample_inputs.npz (select each graph's input_names)"
        report["graphs"] = {
            "voxel_mean": export_graph(voxel_module, [feed[name] for name in VOXEL_INPUT_NAMES],
                                        VOXEL_INPUT_NAMES, "voxel_features", args.output_dir / "voxel_mean.onnx"),
            "sparse_encoder": export_graph(sparse_module, [feed[name] for name in sparse_module.input_names],
                                            sparse_module.input_names, "lidar_bev", args.output_dir / "sparse_encoder.onnx")}
        if args.verify:
            report.update(verify_frames(args, points, backbone, voxel_module, sparse_module,
                (voxel_size, point_cloud_range, max_voxels), (voxel_plan, encoder_plan, feed, statistics)))
        report["status"] = "exported"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print("[manifest]", args.output_dir / "manifest.json", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
