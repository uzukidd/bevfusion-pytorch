"""Convert and compile an existing HBIR using the active WSL bevfusion-pytorch environment."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _node_record(node):
    result = {"name": node.name, "device": str(node.device)}
    # HBRT node names can inherit a preceding op name. HBDK 4.7.5 exposes
    # the real CPU operation signature only through its native Node handle.
    handle = getattr(node, "_Node__handle", None)
    operation_name = getattr(handle, "operation_name", None)
    if callable(operation_name):
        result["operation_name"] = operation_name() or None
    return result


def run(args):
    source = args.hbir.resolve(strict=True)
    output = (args.output_dir or ROOT / "outputs/hbir_to_hbm" /
              (source.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    fingerprint = _hash(source)
    report = {"status": "running", "stage": "environment", "runtime_python": sys.executable,
              "source_hbir": str(source), "source_sha256": fingerprint,
              "march": args.march, "opt": args.opt, "jobs": args.jobs,
              "hbm_compiled": False, "board_verified": False, "quantized_accuracy_verified": False,
              "automatic_quantization": False, "note": "Conversion preserves the input HBIR precision; it does not calibrate a float graph."}
    started = time.perf_counter()
    try:
        import hbdk4.compiler as hbc
        report["stage"] = "load_hbir"
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[load]", source, flush=True)
        module = hbc.load(str(source))
        if args.resume_hbo:
            converted_path = args.resume_hbo.resolve(strict=True).with_name("camera.converted.bc")
            converted = hbc.load(str(converted_path))
            report["resume_hbo"] = str(args.resume_hbo.resolve())
        else:
            report["stage"] = "convert"
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[convert] march={args.march}; advice={output}", flush=True)
            converted = hbc.convert(module, march=args.march, advice=True, advice_path=str(output))
            converted_path = output / "camera.converted.bc"
            hbc.save(converted, str(converted_path))
        report["converted_hbir"] = str(converted_path)
        report["operation_statistics"] = hbc.statistics(converted)
        report["stage"] = "link" if args.resume_hbo else "compile"
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        hbm_path = output / "camera.hbm"
        print(f"[compile] march={args.march}; opt={args.opt}; jobs={args.jobs}; output={hbm_path}", flush=True)
        if args.resume_hbo:
            from hbdk4.compiler.hbm import Hbo
            hbc.link([Hbo(str(args.resume_hbo.resolve(strict=True)))], str(hbm_path))
        else:
            hbc.compile(converted, str(hbm_path), march=args.march, opt=args.opt, jobs=args.jobs)
        if not hbm_path.is_file() or not hbm_path.stat().st_size:
            raise RuntimeError("Compiler returned without producing a nonempty HBM")
        report.update(status="compiled", stage="completed", hbm_compiled=True,
                      hbm_path=str(hbm_path), hbm_bytes=hbm_path.stat().st_size)
        if hasattr(hbc, "Hbm"):
            try:
                artifact = hbc.Hbm(str(hbm_path))
                graphs = []
                for graph in artifact.graphs:
                    graph_record = {"name": graph.name,
                        "device_counts": dict(Counter(str(node.device) for node in graph.nodes)),
                        "nodes": [_node_record(node) for node in graph.nodes]}
                    for field, target in (("flatten_inputs", "inputs"), ("flatten_outputs", "outputs")):
                        graph_record[target] = [{"name": value.name,
                            "shape": [int(d) if d is not None else None for d in value.type.shape],
                            "dtype": str(value.type.np_dtype)} for value in getattr(graph, field)]
                    graphs.append(graph_record)
                report["hbm_metadata"] = {"march": str(artifact.march_name),
                    "toolkit_version": str(artifact.toolkit_version), "loaded": True, "graphs": graphs}
            except Exception as exc:
                report["hbm_metadata_error"] = f"{type(exc).__name__}: {exc}"
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("[failed]", report["error"], flush=True)
        return 1
    finally:
        records = []
        advice_paths = list(output.glob("*advice*.json")) + list(output.parent.glob(output.name + "*advice*.json"))
        report["advice_paths"] = [str(path) for path in advice_paths]
        for path in advice_paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    records.extend(item for item in data if isinstance(item, dict))
            except (ValueError, OSError):
                pass
        if records:
            report["placement_records"] = dict(Counter(item.get("backend", "unknown") for item in records))
            failures = [item for item in records if item.get("check_info")]
            report["constraint_failures"] = len(failures)
            report["constraint_failure_details"] = failures[:20]
        report["elapsed_seconds"] = time.perf_counter() - started
        report["source_unchanged"] = _hash(source) == fingerprint
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[report]", output / "report.json", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hbir", type=Path, required=True)
    parser.add_argument("--resume-hbo", type=Path, help="Link an existing compilation object without recompiling")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--march", choices=("nash-p",), default="nash-p")
    parser.add_argument("--opt", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("jobs must be positive")
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
