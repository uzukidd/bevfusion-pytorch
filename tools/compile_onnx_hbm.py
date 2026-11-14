"""Compile an existing fixed-shape ONNX and matching NPZ calibration to S600 HBM."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.oe_compile import compile_hbm
from tools.static_onnx import specialize_static_shapes


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_ort(path, feed, device, threads):
    # Import PyTorch first so its installed CUDA/cuDNN DLLs are loaded on Windows,
    # as in the existing export and inference entries. No environment changes.
    if device == "cuda":
        import torch
    import onnxruntime as ort
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        raise ValueError(f"Unavailable verification provider: {provider}")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.log_severity_level = 3
    providers = [provider]
    if device == "cuda":
        providers = [(provider, {"use_ep_level_unified_stream": "1", "use_tf32": "0",
            "cudnn_conv_use_max_workspace": "0", "cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), options, providers=providers)
    if provider not in session.get_providers():
        raise RuntimeError(f"Verification provider failed to initialize: {provider}")
    return session.run(None, feed)


def run(args):
    import onnx
    source = args.onnx.resolve(strict=True)
    inputs = (args.inputs or source.with_suffix(".sample_inputs.npz")).resolve(strict=True)
    if source.suffix.lower() != ".onnx" or inputs.suffix.lower() != ".npz":
        raise ValueError("--onnx requires an ONNX model and --inputs a matching NPZ archive")
    output = (args.output_dir or ROOT / "outputs/onnx_to_hbm" /
        (source.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output directory: {output}")
    graph = onnx.load(str(source))
    onnx.checker.check_model(graph, full_check=True)
    contract = []
    for value in graph.graph.input:
        tensor = value.type.tensor_type
        shape = [dim.dim_value if dim.HasField("dim_value") else None for dim in tensor.shape.dim]
        if not shape or any(size is None or size < 1 for size in shape):
            raise ValueError(f"Input {value.name} has dynamic dimensions; export a fixed deployment graph before compilation")
        dtype = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.INT32: "int32",
                 onnx.TensorProto.INT64: "int64"}.get(tensor.elem_type)
        if dtype is None:
            raise ValueError(f"Unsupported calibration input dtype for {value.name}: {tensor.elem_type}")
        if Path(value.name).name != value.name or any(char in value.name for char in "/\\;:\n\r"):
            raise ValueError(f"Input name is not safe as a calibration directory: {value.name!r}")
        contract.append({"name": value.name, "shape": shape, "dtype": dtype})
    feed = {}
    with np.load(inputs, allow_pickle=False) as archive:
        if set(archive.files) != {item["name"] for item in contract}:
            raise ValueError("Calibration NPZ names must match every actual ONNX input exactly")
        for item in contract:
            value = archive[item["name"]]
            if list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"] or not np.isfinite(value).all():
                raise ValueError(f"Calibration shape/dtype/finite check failed: {item['name']}")
            feed[item["name"]] = np.ascontiguousarray(value)
    fingerprint = _hash(source)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "preparing", "stage": "static_specialization", "source_onnx": str(source),
        "source_sha256": fingerprint, "calibration_npz": str(inputs), "calibration_samples": 1,
        "calibration_accuracy_verified": False, "board_verified": False, "hbm_compiled": False,
        "inputs": contract, "output_dir": str(output)}
    try:
        prepared, specialization = specialize_static_shapes(graph)
        report["static_specialization"] = specialization
        prepared_path = output / "model.onnx"
        onnx.save(prepared, str(prepared_path))
        del prepared, graph
        gc.collect()
        report["stage"] = "specialization_verification"
        print("[verify] Original ONNX against compilation copy", flush=True)
        before = _run_ort(source, feed, args.verify_device, args.threads)
        gc.collect()
        after = _run_ort(prepared_path, feed, args.verify_device, args.threads)
        if len(before) != len(after):
            raise AssertionError("Static specialization changed the output count")
        metrics = []
        for index, (expected, actual) in enumerate(zip(before, after)):
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise AssertionError("Static specialization changed output shape/dtype")
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise AssertionError("Original or specialized ONNX outputs contain nonfinite values")
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            metrics.append({"output": index, "max_abs": float(np.abs(actual.astype(np.float64) - expected).max(initial=0))})
        report["specialization_comparison"] = metrics
        print("[verify]", json.dumps(metrics), flush=True)
        del before, after
        gc.collect()
        directories = []
        for item in contract:
            directory = output / "calibration" / item["name"]
            directory.mkdir(parents=True)
            np.save(directory / "000000.npy", feed[item["name"]], allow_pickle=False)
            directories.append(directory)
        report["stage"] = "oe_compile"
        print("[compile]", args.precision, args.optimize_level, "jobs=", args.jobs, "->", output, flush=True)
        compiled = compile_hbm(prepared_path, output, contract, directories, precision=args.precision,
            optimize_level=args.optimize_level, jobs=args.jobs, timeout_seconds=args.compile_timeout,
            image=args.oe_image, wsl_distro=args.wsl_distro, python_path=getattr(args, "hbm_python", None), dry_run=args.dry_run)
        report["compile"] = compiled
        report["hbm_compiled"] = bool(compiled["hbm_compiled"])
        report["status"] = "compiled" if report["hbm_compiled"] else "prepared" if args.dry_run else "compile_failed"
        report["error"] = compiled.get("reason") or compiled.get("error")
        report["compiler_returncode"] = compiled.get("returncode")
        report["compiler_log"] = compiled.get("log_path")
        report["stage"] = "completed" if report["status"] in ("compiled", "prepared") else "oe_compile"
        return 0 if report["status"] in ("compiled", "prepared") else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", hbm_compiled=False)
        raise
    finally:
        report["source_unchanged"] = _hash(source) == fingerprint
        report["elapsed_seconds"] = time.perf_counter() - started
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps({key: report.get(key) for key in ("status", "stage", "hbm_compiled", "error",
            "compiler_returncode", "compiler_log", "source_unchanged", "elapsed_seconds")}, ensure_ascii=False), flush=True)
        print("[report]", output / "report.json", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Default: adjacent <model>.sample_inputs.npz")
    parser.add_argument("--output-dir", type=Path, help="New directory under this project")
    parser.add_argument("--precision", choices=("int16", "int8", "float16"), default="int16")
    parser.add_argument("--optimize-level", choices=("O0", "O1", "O2"), default="O0")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--oe-image", help=argparse.SUPPRESS)
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04", help="WSL distribution used by Windows HBM compilation")
    parser.add_argument("--hbm-python", help="Absolute Linux Conda Python for HBM compilation; default bevfusion-pytorch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.oe_image is not None:
        parser.error("--oe-image is no longer supported; use the bevfusion-pytorch Conda environment (--hbm-python)")
    if args.jobs < 1 or args.threads < 1 or args.compile_timeout < 0:
        parser.error("jobs/threads must be positive; compile-timeout must be nonnegative")
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, AssertionError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
