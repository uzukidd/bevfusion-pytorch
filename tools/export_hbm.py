"""Export a checkpoint, static deployment inputs and an S600 HBM bundle.

Topology discovery and optional CPU operators remain per-frame application work.
Only a successful OE compile with an existing HBM is reported as HBM success.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bevfusion import BEVFusionDet, BEVFusionSeg
from bevfusion.deployment_inputs import DeploymentProfile, build_deployment_frame
from bevfusion.sample_io import load_samples
from tools.deployment_model import StaticDeploymentModel, postprocess_detection
from tools.export_onnx import checkpoint_task, compare_outputs, operator_counts, assert_standard_operators
from tools.oe_compile import compile_hbm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-seg.pth")
    parser.add_argument("--task", choices=("auto", "det", "seg"), default="auto")
    parser.add_argument("--config", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--samples", type=Path)
    source.add_argument("--inputs", type=Path)
    parser.add_argument("--calibration", type=Path, help="Original-image calibration for --samples")
    parser.add_argument("--calibration-inputs", type=Path, action="append", default=[], help="Additional real PT frames; repeat for calibration samples")
    parser.add_argument("--pool-backend", choices=("hat", "compatible"), default=None)
    parser.add_argument("--pool-num-points", type=int, default=None)
    parser.add_argument("--voxel-backend", choices=("cpu", "reference", "compatible"), default="compatible")
    parser.add_argument("--spconv-backend", choices=("cpu", "reference", "compatible"), default="compatible")
    parser.add_argument("--point-capacity", type=int, default=65535)
    parser.add_argument("--voxel-capacity", type=int, default=32768)
    parser.add_argument("--sparse-capacity", type=int, default=32768)
    parser.add_argument("--page-size", type=int, default=32768)
    parser.add_argument("--score-threshold", type=float, help="Detection filtering runs outside HBM; default checkpoint threshold")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PyTorch graph export device")
    parser.add_argument("--verify-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-verify", action="store_true", help="Skip ORT checks; source-adapter parity still required")
    parser.add_argument("--export-only", action="store_true", help="Generate and verify ONNX, inputs and OE config without invoking the Conda compiler")
    parser.add_argument("--dry-run", action="store_true", help="Alias of --export-only; compiler invocation is recorded only")
    parser.add_argument("--precision", choices=("float16", "int16", "int8"), default="int16")
    parser.add_argument("--optimize-level", choices=("O0", "O1", "O2"), default="O0")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--compile-timeout", type=int, default=600, help="Compiler seconds; 0 disables its time limit")
    parser.add_argument("--oe-image", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04", help="WSL distribution used by Windows HBM compilation")
    parser.add_argument("--hbm-python", help="Absolute Linux Conda Python for HBM compilation; default bevfusion-pytorch")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, help="New bundle directory; default outputs/hbm_export/<checkpoint>_<timestamp>")
    args = parser.parse_args(argv)
    if args.oe_image is not None:
        parser.error("--oe-image is no longer supported; use the bevfusion-pytorch Conda environment (--hbm-python)")
    if args.calibration is not None and args.inputs is not None:
        parser.error("--calibration applies to --samples; PT inputs carry their own calibration")
    if args.samples is None and args.inputs is None:
        args.samples = ROOT / "data/samples"
    if args.threads < 1 or args.jobs < 1 or args.compile_timeout < 0:
        parser.error("threads/jobs must be positive and compile-timeout nonnegative")
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("atol/rtol must be finite and nonnegative")
    if args.pool_num_points is not None and not 1 <= args.pool_num_points <= 1024:
        parser.error("--pool-num-points must be in [1,1024]")
    if args.score_threshold is not None and not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be in [0,1]")
    try:
        args.profile = DeploymentProfile(args.point_capacity, args.voxel_capacity, args.sparse_capacity, args.page_size)
    except ValueError as exc:
        parser.error(str(exc))
    args.voxel_backend = "cpu" if args.voxel_backend == "reference" else args.voxel_backend
    args.spconv_backend = "cpu" if args.spconv_backend == "reference" else args.spconv_backend
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = (args.output_dir or ROOT / "outputs/hbm_export" /
        f"{args.checkpoint.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}").expanduser().resolve()
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite existing bundle: {args.output_dir}")
    try:
        args.output_dir.relative_to(ROOT)
    except ValueError:
        parser.error("--output-dir must be inside the project so the WSL compiler can access it")
    return parser, args


def _json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(args):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    task = checkpoint_task(payload)
    if args.task != "auto" and task != args.task:
        raise ValueError(f"Requested task {args.task} conflicts with checkpoint {task}")
    if task == "seg" and args.score_threshold is not None:
        raise ValueError("--score-threshold applies to detection only")
    factory = BEVFusionDet if task == "det" else BEVFusionSeg
    options = dict(backend="reference", lidar_backend="reference", pool_backend="reference",
                   precompute=True, camera_batch_size=0)
    if args.config is not None:
        options["config"] = args.config
    model = factory.from_pretrained(args.checkpoint, device="cpu", **options).eval()
    packed = payload.get("config", {}) if payload.get("format") else {}
    packed = packed if isinstance(packed, dict) else {}
    operators = getattr(model, "config", {}).get("operators", {})
    configured_pool = operators.get("bev_pool") if args.config is not None else packed.get("pool_backend", operators.get("bev_pool"))
    configured_count = operators.get("bev_pool_num_points", 10) if args.config is not None else packed.get("pool_num_points", operators.get("bev_pool_num_points", 10))
    pool = args.pool_backend or configured_pool or "hat"
    if pool not in ("hat", "compatible"):
        raise ValueError(f"S600 deployment pooling must be hat or compatible, got {pool!r}; select --pool-backend explicitly")
    count = args.pool_num_points if args.pool_num_points is not None else configured_count
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1024:
        raise ValueError("HAT sample count must be an integer in [1,1024]")
    if task == "det" and int(model.heads["object"].num_proposals) > 1024:
        raise ValueError("Nash-P TopK supports at most 1024 proposals; the model is not truncated automatically")
    vt = model.encoders["camera"]["vtransform"]
    vt.pool_backend, vt.pool_num_points = pool, count
    vt.clear_bev_cache()
    threshold = args.score_threshold
    if task == "det" and threshold is None:
        threshold = model.config["model"]["heads"]["object"]["bbox_coder"]["score_threshold"]
    return model, task, pool, count, threshold


def load_frames(args):
    first = (torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else
             load_samples(args.samples, calibration=args.calibration))
    return [first] + [torch.load(path, map_location="cpu", weights_only=True) for path in args.calibration_inputs]


def _source_prediction(model, sample, frame, threshold):
    points = sample["points"].detach().cpu()
    images = frame.feed["images"].cpu()
    values = (images, points, *frame.calibration)
    with torch.inference_mode():
        return model(*values, score_threshold=threshold) if model.task == "det" else model(*values)


def _compare_outputs(raw, expected, task, contract, threshold, atol, rtol):
    actual = postprocess_detection(raw, contract, score_threshold=threshold) if task == "det" else raw[0]
    return compare_outputs(actual, expected, task, atol, rtol)


def _session(path, device, threads):
    import onnxruntime as ort
    provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[device]
    if provider not in ort.get_available_providers():
        raise ValueError(f"Requested ORT provider unavailable: {provider}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.log_severity_level = 3
    providers = [provider]
    if device == "cuda":
        providers = [(provider, {"use_ep_level_unified_stream": "1", "use_tf32": "0",
            "cudnn_conv_use_max_workspace": "0", "cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), opts, providers=providers)
    if provider not in session.get_providers():
        raise RuntimeError(f"Requested ORT provider failed to initialize: {provider}")
    return session


def audit_graph(path, names):
    import onnx
    from tools.static_onnx import specialize_static_shapes
    graph, specialization = specialize_static_shapes(onnx.load(str(path)))
    onnx.checker.check_model(graph, full_check=True)
    assert_standard_operators(graph.graph)
    counts = operator_counts(graph.graph)
    forbidden = {name: counts[name] for name in ("Loop", "Unique", "NonZero", "Scan", "If") if counts.get(name)}
    if forbidden:
        raise ValueError(f"Data-dependent operators remain in the deployment graph: {forbidden}")
    if [value.name for value in graph.graph.input] != list(names):
        raise ValueError("Export changed or removed runtime preprocessing inputs")
    contracts = {}
    for category, values in (("inputs", graph.graph.input), ("outputs", graph.graph.output)):
        items = []
        for value in values:
            shape = [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]
            if not shape or any(dimension < 1 for dimension in shape):
                raise ValueError(f"Deployment {category} must have fixed positive shapes: {value.name}")
            dtype = onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type).lower()
            items.append({"name": value.name, "shape": shape, "dtype": {"float": "float32"}.get(dtype, dtype)})
        contracts[category] = items
    contracts["static_specialization"] = specialization
    return graph, counts, contracts


def run(args):
    import onnx
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    report = {"status": "preparing", "stage": "load", "hbm_compiled": False,
              "board_verified": False, "quantized_accuracy_verified": False}
    # Validate checkpoint/source/plans before creating a bundle directory.
    model, task, pool, count, threshold = load_source(args)
    samples = load_frames(args)
    frames = []
    calibration = None
    for sample in samples:
        frame = build_deployment_frame(model, sample, profile=args.profile, voxel_backend=args.voxel_backend,
            spconv_backend=args.spconv_backend, calibration=calibration)
        calibration = frame.calibration
        frames.append(frame)
    first = frames[0]
    wrapper = StaticDeploymentModel(model, first.calibration, pool_backend=pool, pool_num_points=count,
        voxel_backend=args.voxel_backend, spconv_backend=args.spconv_backend,
        voxel_plan=first.voxel_plan, encoder_plan=first.encoder_plan, page_size=args.page_size).eval()
    if task == "det":
        wrapper.detection_contract["score_threshold"] = threshold
    if wrapper.input_names != list(first.feed):
        raise ValueError("Preprocessor and graph input order differ")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "report.json"
    manifest = None
    try:
        print(f"[backends] pool={pool} voxel={args.voxel_backend} spconv={args.spconv_backend}", flush=True)
        print(f"[bundle] {args.output_dir}", flush=True)
        if args.spconv_backend == "cpu":
            print("[placement] Entire LiDAR feature extraction runs on CPU before the HBM; image/fusion/task computation remains in the graph.", flush=True)
        report.update(stage="source_parity", task=task, backends={"pool": pool, "voxel": args.voxel_backend,
            "spconv": args.spconv_backend}, profile=args.profile.to_dict(), pool_num_points=count,
            calibration_samples=len(frames))
        # Build additional dynamic-cloud cases before moving shared parameters.
        check_samples = [(f"calibration_{index}", sample, frame) for index, (sample, frame) in enumerate(zip(samples, frames))]
        if not args.no_verify:
            base = samples[0]
            for label, points in (("empty_points", base["points"][:0]),
                ("changed_points", base["points"][:min(7, len(base["points"]))].clone())):
                changed = dict(base, points=points)
                if label == "changed_points" and len(points):
                    points[:, :3] += points.new_tensor([.23, -.17, .11])
                frame = build_deployment_frame(model, changed, profile=args.profile,
                    voxel_backend=args.voxel_backend, spconv_backend=args.spconv_backend, calibration=first.calibration)
                check_samples.append((label, changed, frame))
        golden = []
        adapter_metrics = []
        with torch.inference_mode():
            for label, sample, frame in check_samples:
                expected = _source_prediction(model, sample, frame, threshold)
                raw = wrapper(*frame.tensor_feed())
                raw = (raw,) if isinstance(raw, torch.Tensor) else tuple(raw)
                metrics = _compare_outputs(raw, expected, task, wrapper.detection_contract, threshold, 3e-4, 3e-4)
                adapter_metrics.append({"case": label, **metrics})
                golden.append((label, frame, expected))
        report["source_adapter_comparison"] = adapter_metrics
        report["stage"] = "onnx_export"
        onnx_path = args.output_dir / "model.onnx"
        wrapper.to(args.device)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            torch.onnx.export(wrapper, first.tensor_feed(args.device), str(onnx_path), dynamo=False,
                opset_version=18, input_names=wrapper.input_names, output_names=wrapper.output_names,
                do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL)
        graph, counts, contracts = audit_graph(onnx_path, wrapper.input_names)
        report["static_specialization"] = contracts["static_specialization"]
        fixed = {name: tensor.tolist() for name, tensor in zip(
            ("camera_intrinsics", "camera2lidar", "img_aug_matrix"), first.calibration)}
        manifest = {"schema": "bevfusion-hbm-bundle-v1", "task": task,
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha256(args.checkpoint),
            "model_args": {"config": model.config} if hasattr(model, "config") else {},
            "backends": report["backends"], "pool_num_points": count,
            "profile": args.profile.to_dict(), "fixed_calibration": fixed,
            "inputs": contracts["inputs"], "outputs": contracts["outputs"],
            "detection_contract": wrapper.detection_contract,
            "score_threshold": threshold, "onnx": "model.onnx", "target": "nash-p",
            "cpu_preprocessing_required": True, "capacity_overflow": "raise",
            "input_generation": "tools/prepare_hbm_inputs.py --manifest <bundle>/manifest.json --samples <frame> --output-dir <new-dir>",
            "placement": {"grouping": "cpu", "topology": "cpu" if args.spconv_backend == "compatible" else "not_required",
                "voxel_mean": "graph" if args.voxel_backend == args.spconv_backend == "compatible" else "cpu",
                "spconv": "graph" if args.spconv_backend == "compatible" else "cpu",
                "detection_depth_projection": "cpu" if task == "det" else "not_required"}}
        onnx.helper.set_model_props(graph, {"task": task, "deployment_schema": manifest["schema"],
            "backends": json.dumps(report["backends"]), "profile": json.dumps(manifest["profile"]),
            "fixed_calibration": json.dumps(fixed), "cpu_preprocessing_required": "true",
            "detection_contract": json.dumps(wrapper.detection_contract), "hbm_compiled": "false"})
        onnx.save(graph, str(onnx_path))
        _json(args.output_dir / "manifest.json", manifest)
        _json(args.output_dir / "preprocessing.json", {"fixed_calibration": fixed,
            "profile": args.profile.to_dict(), "backends": report["backends"],
            "calibration_frames": [frame.statistics for frame in frames]})
        np.savez_compressed(args.output_dir / "sample_inputs.npz", **first.feed_np())
        calibration_dirs = []
        for name in wrapper.input_names:
            directory = args.output_dir / "calibration" / name
            directory.mkdir(parents=True)
            calibration_dirs.append(directory)
            for index, frame in enumerate(frames):
                np.save(directory / f"{index:06d}.npy", frame.feed_np()[name], allow_pickle=False)
        report.update(operator_counts=counts, input_count=len(wrapper.input_names),
            onnx_file_bytes=onnx_path.stat().st_size, sampling=wrapper.pool.statistics(),
            runtime_verified=False, stage="ort_verification")
        if not args.no_verify:
            session = _session(onnx_path, args.verify_device, args.threads)
            if [value.name for value in session.get_inputs()] != wrapper.input_names:
                raise ValueError("ORT input contract differs from the export")
            cases = []
            for label, frame, expected in golden:
                raw = session.run(wrapper.output_names, frame.feed_np())
                metrics = _compare_outputs(raw, expected, task, wrapper.detection_contract, threshold,
                                           args.atol, args.rtol)
                cases.append({"case": label, **metrics})
                print("[verify]", label, json.dumps(metrics), flush=True)
            report.update(runtime_verified=True, verification_cases=cases, providers=session.get_providers())
            del session
        report["stage"] = "oe_compile"
        compiled = compile_hbm(onnx_path, args.output_dir, contracts["inputs"], calibration_dirs,
            march="nash-p", image=args.oe_image, wsl_distro=args.wsl_distro, python_path=getattr(args, "hbm_python", None),
            precision=args.precision, optimize_level=args.optimize_level, jobs=args.jobs,
            timeout_seconds=args.compile_timeout, dry_run=args.export_only or args.dry_run)
        report["compile"] = compiled
        report["hbm_compiled"] = bool(compiled.get("hbm_compiled", False))
        report["status"] = "compiled" if report["hbm_compiled"] else (
            "exported" if args.export_only or args.dry_run else "compile_failed")
        report["stage"] = "completed" if report["status"] in ("compiled", "exported") else "oe_compile"
        report.update(compiler_status=compiled.get("status"),
                      compiler_returncode=compiled.get("returncode"),
                      compiler_log=compiled.get("log_path"))
        if report["status"] == "compile_failed":
            reason = compiled.get("reason") or compiled.get("error")
            if not reason:
                reason = f"OE compilation failed (status={compiled.get('status', 'unknown')}, exit code={compiled.get('returncode', 'unknown')}); no successful HBM was produced"
            report["error"] = reason
            print(f"[compile failed] {reason}", file=sys.stderr, flush=True)
            if compiled.get("log_path"):
                print(f"[compiler log] {compiled['log_path']}", file=sys.stderr, flush=True)
        return 0 if report["status"] in ("compiled", "exported") else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", hbm_compiled=False)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _json(report_path, report)
        print("[report]", report_path, flush=True)
        print(json.dumps({key: report.get(key) for key in
            ("status", "stage", "hbm_compiled", "runtime_verified", "elapsed_seconds", "error",
             "compiler_status", "compiler_returncode", "compiler_log")}, ensure_ascii=False), flush=True)


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, AssertionError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
