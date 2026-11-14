"""Split an existing BEVFusion ONNX into camera/LiDAR/fusion graphs and compile independently."""
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

from tools.compile_onnx_hbm import _run_ort
from tools.oe_compile import compile_hbm


BRANCHES = ("fusion", "camera", "lidar")


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_feed(path, contract):
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {item["name"] for item in contract}:
            raise ValueError(f"NPZ must contain exactly the graph inputs: {path}")
        for item in contract:
            value = archive[item["name"]]
            if list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"]:
                raise ValueError(f"Input shape/dtype mismatch for {item['name']}: {path}")
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite input {item['name']}: {path}")
            result[item["name"]] = np.ascontiguousarray(value)
    return result


def _graph_inputs(path):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    result = []
    for value in model.graph.input:
        tensor = value.type.tensor_type
        shape = [dim.dim_value if dim.HasField("dim_value") else None for dim in tensor.shape.dim]
        if not shape or any(size is None or size < 1 for size in shape):
            raise ValueError(f"Splitting for HBM requires fixed input dimensions: {value.name}")
        dtype = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.INT32: "int32",
                 onnx.TensorProto.INT64: "int64", onnx.TensorProto.FLOAT16: "float16"}.get(tensor.elem_type)
        if dtype is None:
            raise ValueError(f"Unsupported source input dtype: {value.name}")
        result.append({"name": value.name, "shape": shape, "dtype": dtype})
    return result


def _validate_outputs(outputs, contract):
    if len(outputs) != len(contract):
        raise ValueError("Branch output count differs from its contract")
    for value, item in zip(outputs, contract):
        if (list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"]
                or not np.isfinite(value).all()):
            raise ValueError(f"Branch output violates its shape/dtype contract: {item['name']}")


def _compare(actual, expected, atol, rtol):
    if len(actual) != len(expected):
        raise AssertionError("Split pipeline changed output count")
    metrics = []
    for index, (value, reference) in enumerate(zip(actual, expected)):
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise AssertionError("Split pipeline changed output shape or dtype")
        if not np.isfinite(value).all() or not np.isfinite(reference).all():
            raise AssertionError("Split or whole-model output contains nonfinite values")
        if value.dtype.kind in "iu":
            np.testing.assert_array_equal(value, reference)
        else:
            np.testing.assert_allclose(value, reference, atol=atol, rtol=rtol)
        difference = np.abs(value.astype(np.float64) - reference.astype(np.float64))
        record = {"output": index, "max_abs": float(difference.max(initial=0)),
                  "mean_abs": float(difference.mean()) if difference.size else 0.0}
        if value.dtype.kind == "f" and value.ndim == 4 and value.shape[-2:] == (200, 200):
            record["mask_disagreements_at_0.5"] = int(np.count_nonzero((value >= .5) != (reference >= .5)))
        metrics.append(record)
    return metrics


def run(args):
    from tools.split_onnx_branches import split_model

    source = args.onnx.expanduser().resolve(strict=True)
    if source.suffix.lower() != ".onnx":
        raise ValueError("--onnx must select an existing ONNX model")
    source_inputs = _graph_inputs(source)
    samples = [(args.inputs or source.with_suffix(".sample_inputs.npz")).expanduser().resolve(strict=True)]
    samples += [path.expanduser().resolve(strict=True) for path in args.calibration_inputs]
    # Validate all snapshots before writing the output directory.
    for path in samples:
        _load_feed(path, source_inputs)
    output = (args.output_dir or ROOT / "outputs/split_hbm" /
              (source.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).expanduser().resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    fingerprint = _hash(source)
    report = {"status": "preparing", "stage": "split", "source_onnx": str(source),
              "source_sha256": fingerprint, "calibration_npz": [str(path) for path in samples],
              "output_dir": str(output), "selected_branches": list(args.branches),
              "hbm_compiled": False, "board_verified": False, "quantized_accuracy_verified": False,
              "float_pipeline_verified": False, "compile": {}, "verification_cases": []}
    try:
        manifest = split_model(source, output / "graphs", camera_output=args.camera_output, lidar_output=args.lidar_output)
        report["manifest"] = manifest
        report["stage"] = "float_pipeline_verification"
        directories = {}
        for name, branch in manifest["branches"].items():
            directories[name] = []
            for item in branch["inputs"]:
                directory = output / "calibration" / name / item["name"]
                directory.mkdir(parents=True)
                directories[name].append(directory)
        for index, path in enumerate(samples):
            print(f"[verify] frame {index}: {path}", flush=True)
            sensor = _load_feed(path, source_inputs)
            expected = _run_ort(source, sensor, args.verify_device, args.threads)
            intermediate = {}
            for name in ("camera", "lidar", "fusion"):
                branch = manifest["branches"][name]
                available = intermediate if name == "fusion" else sensor
                feed = {item["name"]: available[item["name"]] for item in branch["inputs"]}
                values = _run_ort(Path(branch["onnx"]), feed, args.verify_device, args.threads)
                _validate_outputs(values, branch["outputs"])
                for item, directory in zip(branch["inputs"], directories[name]):
                    np.save(directory / f"{index:06d}.npy", feed[item["name"]], allow_pickle=False)
                if index == 0:
                    np.savez_compressed(output / f"{name}.sample_inputs.npz", **feed)
                intermediate.update(zip((item["name"] for item in branch["outputs"]), values))
                if name == "fusion":
                    actual = values
            metrics = _compare(actual, expected, args.atol, args.rtol)
            report["verification_cases"].append({"frame": index, "source": str(path), "outputs": metrics})
            print("[verify] chained branches vs whole ONNX:", json.dumps(metrics), flush=True)
            del sensor, expected, intermediate, actual, values, feed
            gc.collect()
        report["float_pipeline_verified"] = True
        report["stage"] = "oe_compile"
        # Run sequentially so compiler memory cannot accumulate across branches.
        for name in args.branches:
            branch = manifest["branches"][name]
            print(f"[compile branch] {name}; precision={args.precision}; jobs={args.jobs}", flush=True)
            result = compile_hbm(Path(branch["onnx"]), output / "compile" / name, branch["inputs"], directories[name],
                precision=args.precision, optimize_level=args.optimize_level, jobs=args.jobs,
                timeout_seconds=args.compile_timeout, image=args.oe_image, wsl_distro=args.wsl_distro, python_path=getattr(args, "hbm_python", None),
                dry_run=args.export_only)
            report["compile"][name] = result
            print("[branch result]", json.dumps({"branch": name, **{key: result.get(key) for key in
                ("status", "hbm_compiled", "hbm_path", "reason", "returncode", "log_path", "elapsed_seconds")}},
                ensure_ascii=False), flush=True)
            # A previous branch failure must not prevent the independent checks.
            _json(output / f"{name}.result.json", result)
        complete = all(result.get("hbm_compiled", False) for result in report["compile"].values())
        report["selected_branches_compiled"] = complete
        report["hbm_compiled"] = complete and set(args.branches) == set(BRANCHES)
        report["status"] = "exported" if args.export_only else "compiled" if complete else "compile_failed"
        report["stage"] = "completed"
        if report["status"] == "compile_failed":
            report["error"] = "; ".join(f"{name}: {result.get('reason') or result.get('error') or result.get('status')}"
                for name, result in report["compile"].items() if not result.get("hbm_compiled"))
        return 0 if report["status"] in ("exported", "compiled") else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", hbm_compiled=False)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["source_unchanged"] = _hash(source) == fingerprint
        _json(output / "report.json", report)
        print("[report]", output / "report.json", flush=True)
        print(json.dumps({key: report.get(key) for key in ("status", "stage", "float_pipeline_verified",
            "hbm_compiled", "error", "elapsed_seconds", "source_unchanged")}, ensure_ascii=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Default: adjacent source .sample_inputs.npz")
    parser.add_argument("--calibration-inputs", type=Path, action="append", default=[], help="Additional matching NPZ snapshots")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--branches", nargs="+", choices=BRANCHES, default=list(BRANCHES),
                        help="Compile selected branches sequentially; default fusion camera lidar")
    parser.add_argument("--camera-output", help="Optional exact original Camera BEV tensor name")
    parser.add_argument("--lidar-output", help="Optional exact original LiDAR BEV tensor name")
    parser.add_argument("--precision", choices=("int16", "int8", "float16"), default="int16")
    parser.add_argument("--optimize-level", choices=("O0", "O1", "O2"), default="O0")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--compile-timeout", type=int, default=600, help="Seconds per branch; 0 disables timeout")
    parser.add_argument("--verify-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--oe-image", help=argparse.SUPPRESS)
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04", help="WSL distribution used by Windows HBM compilation")
    parser.add_argument("--hbm-python", help="Absolute Linux Conda Python for HBM compilation; default bevfusion-pytorch")
    parser.add_argument("--export-only", action="store_true", help="Verify all ONNX branches, generate OE configs without compiling")
    args = parser.parse_args(argv)
    if args.oe_image is not None:
        parser.error("--oe-image is no longer supported; use the bevfusion-pytorch Conda environment (--hbm-python)")
    if args.jobs < 1 or args.threads < 1 or args.compile_timeout < 0:
        parser.error("jobs/threads must be positive; compile-timeout must be nonnegative")
    if len(args.branches) != len(set(args.branches)):
        parser.error("--branches must not contain duplicates")
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("atol/rtol must be finite and nonnegative")
    if bool(args.camera_output) != bool(args.lidar_output):
        parser.error("--camera-output and --lidar-output must be supplied together")
    return parser, args


def main(argv=None):
    _, args = parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, AssertionError, KeyError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
