"""Compile separate float HBIR branches to HBO, then link a scheduled multi-graph HBM."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.compile_hbir_hbm import _node_record

BRANCHES = ("camera", "lidar", "fusion")
LIDAR_STAGES = tuple(f"lidar_stage{i}" for i in range(4))
STAGED_GRAPHS = ("camera", *LIDAR_STAGES, "fusion")
GRAPH_NAMES = (*BRANCHES, *LIDAR_STAGES)
SCRIPT = Path(__file__).resolve()
PRECISION_NOTE = ("Float source precision is preserved; conversion does not calibrate or quantize. "
                  "CPU fallback operations may remain. Quantized accuracy and board execution are unverified.")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def nonempty(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Expected nonempty artifact: {path}")
    return path


def dtype_name(dtype):
    text = str(dtype)
    if text.startswith("<class '") and text.endswith("'>"):
        return text[8:-2].rsplit(".", 1)[-1]
    return text


def normalized_graph(record):
    return {**record, **{direction: [{**item, "dtype": dtype_name(item["dtype"])} for item in record[direction]]
                        for direction in ("inputs", "outputs")}}


def specs(values):
    return [{"name": value.name, "shape": [int(d) if d is not None else None for d in value.type.shape],
             "dtype": dtype_name(value.type.np_dtype)} for value in values]


def graph_record(graph, nodes=False):
    record = {"name": graph.name, "inputs": specs(graph.flatten_inputs), "outputs": specs(graph.flatten_outputs)}
    if nodes:
        record["nodes"] = [_node_record(node) for node in graph.nodes]
        record["node_count"] = len(record["nodes"])
        record["device_counts"] = dict(Counter(node["device"] for node in record["nodes"]))
        record["actual_cpu_operations"] = [node for node in record["nodes"] if "cpu" in node["device"].lower()]
    return record


def validate_branch(record, name):
    expected = {"camera": ({"images"}, {"camera_bev"}),
                "lidar": ({"voxel_features"}, {"lidar_bev"}),
                "fusion": ({"camera_bev", "lidar_bev"}, {"probabilities"})}
    expected.update({name: ({"voxel_features" if index == 0 else f"lidar_stage{index - 1}_features"},
                            {"lidar_bev" if index == 3 else name + "_features"})
                     for index, name in enumerate(LIDAR_STAGES)})
    if name not in expected:
        raise ValueError(f"Unsupported graph name: {name}")
    if record["name"] != name:
        raise ValueError(f"Expected graph {name}, got {record['name']}")
    for direction, names in zip(("inputs", "outputs"), expected[name]):
        actual = [item["name"] for item in record[direction]]
        if len(actual) != len(names) or set(actual) != names:
            raise ValueError(f"{name} {direction} contract names: expected {sorted(names)}, got {actual}")
    if name == "fusion" and record["outputs"][0]["shape"] != [1, 6, 200, 200]:
        raise ValueError("Fusion probabilities must have shape [1, 6, 200, 200]")


def validate_contracts(records, expected_names=None):
    records = [normalized_graph(record) for record in records]
    names = [record["name"] for record in records]
    if expected_names is None:
        expected_names = STAGED_GRAPHS if any(name in LIDAR_STAGES for name in names) else BRANCHES
    if set(expected_names) not in (set(BRANCHES), set(STAGED_GRAPHS)) or len(expected_names) != len(set(expected_names)):
        raise ValueError(f"Unsupported or duplicate graph manifest: {expected_names}")
    if len(names) != len(expected_names) or set(names) != set(expected_names):
        raise ValueError(f"HBM must contain exactly unique graphs {list(expected_names)}; got {names}")
    graphs = {record["name"]: record for record in records}
    for name, record in graphs.items():
        validate_branch(record, name)
    staged = set(expected_names) == set(STAGED_GRAPHS)
    edges = [("camera", "fusion", "camera_bev"),
             ("lidar_stage3" if staged else "lidar", "fusion", "lidar_bev")]
    if staged:
        edges = [(LIDAR_STAGES[index], LIDAR_STAGES[index + 1], f"lidar_stage{index}_features")
                 for index in range(3)] + edges
    contracts = []
    for producer, consumer, tensor in edges:
        source = next(item for item in graphs[producer]["outputs"] if item["name"] == tensor)
        target = next(item for item in graphs[consumer]["inputs"] if item["name"] == tensor)
        if source != target:
            raise ValueError(f"{tensor} shape/dtype/name differs between {producer} output and {consumer} input")
        contracts.append({"producer": producer, "consumer": consumer, **source, "matched": True})
    return contracts


def stage_pipeline():
    return {"graphs": list(LIDAR_STAGES), "external_input": "voxel_features", "output": "lidar_bev",
            "intermediate_tensors": [f"lidar_stage{index}_features" for index in range(3)],
            "execution": "Application invokes stages in order, then fusion after camera and lidar_bev are ready",
            "application_scheduling_required": True, "single_end_to_end_graph": False}


def source_provenance(report_paths, expected_topology=None):
    records = []
    for path in sorted({Path(path).resolve() for path in report_paths}, key=str):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append({"report": str(path), "report_sha256": digest(path),
                        "checkpoint_sha256": data.get("checkpoint_sha256"),
                        "coordinates_sha256": data.get("topology", {}).get("coordinates_sha256"),
                        "fixed_lidar_topology": data.get("fixed_lidar_topology")})
    result = {"reports": records}
    for field in ("checkpoint_sha256", "coordinates_sha256", "fixed_lidar_topology"):
        values = {item[field] for item in records if item[field] is not None}
        if len(values) > 1:
            raise ValueError(f"Export provenance mismatch: {field} differs between source reports")
        result[field] = next(iter(values), None)
    actual_topology = result["fixed_lidar_topology"]
    if expected_topology is not None and actual_topology is not None and actual_topology != expected_topology:
        raise ValueError("Export provenance mismatch: fixed_lidar_topology differs from link manifest")
    return result


def provenance_fields(provenance):
    return {"source_provenance": provenance, "checkpoint_sha256": provenance["checkpoint_sha256"],
            "topology_coordinates_sha256": provenance["coordinates_sha256"]}


def rename_graph(graph, name):
    # HBDK 4.7.5 Function.name has no setter. FuncOp.sym_name accepts StringAttr.
    if graph.name != name:
        from hbdk4.compiler.ir import StringAttr
        graph.opview.sym_name = StringAttr.get(name, context=graph.opview.context)
    if graph.name != name:
        raise ValueError(f"Failed to rename sole HBIR function to {name}")


def only_graph(module):
    functions = list(module.functions)
    if len(functions) != 1:
        raise ValueError(f"Branch HBIR must contain exactly one function; got {len(functions)}")
    return functions[0]


def base_report():
    return {"status": "running", "runtime_python": sys.executable, "march": "nash-p", "opt": 0, "jobs": 1,
            "precision": "float_source", "automatic_quantization": False, "precision_limitations": PRECISION_NOTE,
            "board_verified": False, "quantized_accuracy_verified": False, "hbm_end_to_end_verified": False,
            "application_scheduling_required": True, "hbm_compiled": False, "hbo_compiled": False}


def advice_metadata(output, report):
    paths = sorted(output.glob("*advice*.json"))
    report["advice_paths"] = [str(path) for path in paths]
    records = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(item for item in data if isinstance(item, dict))
        except (OSError, ValueError):
            pass
    if records:
        report["placement_records"] = dict(Counter(item.get("backend", "unknown") for item in records))
        report["constraint_failure_details"] = [item for item in records if item.get("check_info")]


def worker(args):
    output = args.worker_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = base_report()
    report["stage"] = "environment"
    summary = output / "report.json"
    started = time.monotonic()
    source = None

    def stage(name):
        report["stage"] = name
        save(summary, report)
        print("[stage]", name, flush=True)

    try:
        stage("environment")
        import hbdk4.compiler as hbc
        from hbdk4.compiler.hbm import Hbo
        if args.link_manifest:
            manifest = json.loads(args.link_manifest.read_text(encoding="utf-8"))
            report["fixed_lidar_topology"] = manifest["fixed_lidar_topology"]
            graph_order = manifest.get("graph_order", list(BRANCHES))
            if "staged_lidar_pipeline" in manifest:
                report["staged_lidar_pipeline"] = manifest["staged_lidar_pipeline"]
            report["expected_graph_names"] = graph_order
            stage("validate_source_provenance")
            # Derive source export reports even for manifests made by an already-running older parent.
            source_reports = [Path(item["source_hbir"]).parent / "report.json"
                              for item in manifest["branches"].values() if item.get("source_hbir")]
            provenance = source_provenance(source_reports, report["fixed_lidar_topology"])
            for key, field in (("checkpoint_sha256", "checkpoint_sha256"),
                               ("topology_coordinates_sha256", "coordinates_sha256")):
                if manifest.get(key) is not None and provenance[field] is not None and manifest[key] != provenance[field]:
                    raise ValueError(f"Export provenance mismatch: {key} differs from link manifest")
            report.update(provenance_fields(provenance))
            manifest.update(provenance_fields(provenance))
            save(args.link_manifest, manifest)
            stage("validate_link_inputs")
            if len(graph_order) != len(set(graph_order)) or set(graph_order) != set(manifest["branches"]):
                raise ValueError("Link graph order must identify every manifest branch exactly once")
            records, objects = [], []
            for name in graph_order:
                item = manifest["branches"][name]
                # Reports from workers started before normalization may contain NumPy class reprs.
                item["graph"] = normalized_graph(item["graph"])
                path = nonempty(item["hbo_path"])
                if digest(path) != item["hbo_sha256"]:
                    raise ValueError(f"{name} HBO changed before linking")
                records.append(item["graph"])
                objects.append(Hbo(str(path)))
            report["contracts"] = validate_contracts(records, graph_order)
            hbm_path = Path(manifest["hbm_path"])
            if hbm_path.exists():
                raise ValueError(f"Refusing to overwrite HBM: {hbm_path}")
            stage("link")
            linked = hbc.link(objects, str(hbm_path))
            nonempty(hbm_path)
            stage("inspect_hbm")
            # Reopen the persisted HBM rather than trusting linker return metadata.
            artifact = hbc.Hbm(str(hbm_path))
            from hbdk4.compiler.hbm import Hbm
            if not isinstance(linked, Hbm) or not isinstance(artifact, Hbm):
                raise TypeError("Link must return and persist an Hbm")
            records = [graph_record(graph, nodes=True) for graph in artifact.graphs]
            report["hbm_metadata"] = {"loaded": True, "march": str(artifact.march_name),
                "toolkit_version": str(artifact.toolkit_version), "graphs": records}
            report["contracts"] = validate_contracts(records, graph_order)
            source_graphs = {item["name"]: item for item in records}
            for name in graph_order:
                for direction in ("inputs", "outputs"):
                    if source_graphs[name][direction] != manifest["branches"][name]["graph"][direction]:
                        raise ValueError(f"Linked {name} {direction} differ from verified HBO source")
            report.update(hbm_compiled=True, complete_multi_graph_hbm=True, hbm_path=str(hbm_path),
                hbm_bytes=hbm_path.stat().st_size, hbm_sha256=digest(hbm_path),
                graph_names=[item["name"] for item in records],
                node_count=sum(item["node_count"] for item in records),
                actual_cpu_operations=[{"graph": item["name"], **node} for item in records
                                       for node in item["actual_cpu_operations"]])
        else:
            source = nonempty(args.hbir.resolve(strict=True))
            name = args.graph_name
            report.update(graph_name=name, source_hbir=str(source), source_sha256=digest(source))
            stage("load_hbir")
            module = hbc.load(str(source))
            graph = only_graph(module)
            report["original_graph_name"] = graph.name
            stage("rename_graph")
            rename_graph(graph, name)
            report["graph"] = graph_record(graph)
            validate_branch(report["graph"], name)
            if args.resume_object:
                stage("validate_resume")
                hbo_path = nonempty(args.resume_object.resolve(strict=True))
                if hbo_path.suffix != ".hbo":
                    raise ValueError("Resume object must have .hbo suffix")
                previous = json.loads((hbo_path.parent / "report.json").read_text(encoding="utf-8"))
                if previous.get("source_sha256") != report["source_sha256"]:
                    raise ValueError("Resume HBO source hash does not match current HBIR")
                if previous.get("graph_name") != name or normalized_graph(previous["graph"]) != report["graph"]:
                    raise ValueError("Resume HBO graph identity/contracts do not match current branch")
                if previous.get("hbo_sha256") != digest(hbo_path) or not previous.get("hbo_compiled"):
                    raise ValueError("Resume HBO requires a verified matching report and artifact hash")
                converted_path = nonempty(hbo_path.parent / (name + ".converted.bc"))
                if previous.get("converted_sha256") != digest(converted_path):
                    raise ValueError("Resume converted HBIR hash mismatch")
                converted = hbc.load(str(converted_path))
                if graph_record(only_graph(converted)) != report["graph"]:
                    raise ValueError("Resume converted HBIR graph does not match current branch")
                hbo = Hbo(str(hbo_path))
                report["resume_hbo"] = str(hbo_path)
            else:
                stage("convert")
                converted = hbc.convert(module, march="nash-p", advice=True, advice_path=str(output) + os.sep)
                converted_path = output / (name + ".converted.bc")
                stage("save_converted")
                hbc.save(converted, str(converted_path))
                nonempty(converted_path)
                if graph_record(only_graph(converted)) != report["graph"]:
                    raise ValueError("Converted HBIR changed graph identity or contracts")
                hbo_path = output / (name + ".hbo")
                stage("compile_hbo")
                hbo = hbc.compile(converted, str(hbo_path), march="nash-p", opt=0, jobs=1)
            nonempty(hbo_path)
            if not isinstance(hbo, Hbo):
                raise TypeError("Branch compiler must return an Hbo object")
            stage("verify_hbo")
            verified = Hbo(str(hbo_path))
            if not isinstance(verified, Hbo):
                raise TypeError("Persisted compilation object must be Hbo")
            report.update(hbo_compiled=True, hbo_path=str(hbo_path), hbo_bytes=hbo_path.stat().st_size,
                          hbo_sha256=digest(hbo_path), converted_hbir=str(converted_path),
                          converted_sha256=digest(converted_path))
        if source and digest(source) != report["source_sha256"]:
            report["hbo_compiled"] = False
            raise ValueError("Source HBIR changed during compilation")
        report.update(status="compiled", stage="completed")
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("[failed]", report["error"], flush=True)
        return 1
    finally:
        if source:
            report["source_unchanged"] = source.is_file() and digest(source) == report.get("source_sha256")
            if not report["source_unchanged"]:
                report.update(status="failed", hbo_compiled=False, error="Source HBIR changed during compilation")
        advice_metadata(output, report)
        report["elapsed_seconds"] = time.monotonic() - started
        save(summary, report)


def process_group_rss(pgid):
    total = 0
    if sys.platform == "linux":
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid:
                    lines = (entry / "status").read_text().splitlines()
                    total += next((int(line.split()[1]) for line in lines if line.startswith("VmRSS:")), 0)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    return total


def terminate_group(child, grace=1):
    if sys.platform != "linux":
        if child.poll() is None:
            child.kill()
        child.wait()
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            child.wait()
            return
        time.sleep(.05)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run_worker(command, output, timeout, rss_limit_mib, artifact_flag, artifact_path=None):
    """Monitor the whole dedicated Linux process group, including orphan descendants."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite worker output: {output}")
    if rss_limit_mib and sys.platform != "linux":
        raise ValueError("RSS budgets require Linux /proc; use scripts/compile_hybrid_branches.bat under WSL")
    log = output.parent / (output.name + ".log")
    started = time.monotonic()
    peak, code, status = 0, None, "failed"
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=sys.platform == "linux")
        try:
            while True:
                peak = max(peak, process_group_rss(child.pid))
                if rss_limit_mib and peak > rss_limit_mib * 1024:
                    status, code = "memory_limit", 137
                    break
                if timeout and time.monotonic() - started >= timeout:
                    status, code = "timeout", 124
                    break
                try:
                    code = child.wait(timeout=.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Even a successful leader can leave TERM-ignoring compiler descendants.
            terminate_group(child)
    report_path = output / "report.json"
    try:
        result = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result = {"error": f"No valid worker report: {exc}"}
    if isinstance(result.get("graph"), dict):
        result["graph"] = normalized_graph(result["graph"])
    path = artifact_path or result.get("hbo_path" if artifact_flag == "hbo_compiled" else "hbm_path")
    confirmed = bool(code == 0 and result.get(artifact_flag) and path and Path(path).is_file() and Path(path).stat().st_size)
    if confirmed:
        expected_hash = result.get("hbo_sha256" if artifact_flag == "hbo_compiled" else "hbm_sha256")
        confirmed = bool(expected_hash and expected_hash == digest(path))
    result.update(returncode=code, status="compiled" if confirmed else status, log=str(log),
                  process_group_peak_rss_kib=peak, rss_limit_mib=rss_limit_mib,
                  rss_monitor_available=sys.platform == "linux", elapsed_seconds=time.monotonic() - started)
    result[artifact_flag] = confirmed
    if not confirmed:
        result["reason"] = (f"Compiler process group exceeded {rss_limit_mib} MiB RSS budget" if status == "memory_limit"
                            else f"Compiler process group exceeded {timeout}s timeout" if status == "timeout"
                            else result.get("error") or f"Worker exited {code}; no verified nonempty artifact")
    save(output.parent / (output.name + ".process.json"), result)
    return result


def parse_resume(items):
    result = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or name not in GRAPH_NAMES or not value or name in result:
            raise ValueError("--resume-hbo must be unique NAME=path entries for supported branch/stage names")
        result[name] = nonempty(Path(value).resolve(strict=True))
    return result


def run(args):
    source = args.branch_dir.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("--branch-dir must be an existing directory")
    staged_dir = getattr(args, "staged_lidar_dir", None)
    if staged_dir:
        staged_dir = staged_dir.resolve(strict=True)
        if not staged_dir.is_dir():
            raise ValueError("--staged-lidar-dir must be an existing directory")
    expected_graphs = STAGED_GRAPHS if staged_dir else BRANCHES
    selected = [stage for name in args.branches for stage in (LIDAR_STAGES if staged_dir and name == "lidar" else (name,))]
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate branch names are not allowed")
    resume = parse_resume(args.resume_hbo)
    if not set(resume).issubset(selected):
        raise ValueError("Resume names must be included in the selected branches/stages")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = base_report()
    report.update(stage="compile_branches", branch_dir=str(source), requested_branches=args.branches,
                  branches={}, complete_multi_graph_hbm=False, fixed_lidar_topology=None,
                  selected_graphs=selected, expected_graph_names=list(expected_graphs),
                  dynamic_pointcloud_supported=False)
    if staged_dir:
        report.update(staged_lidar_dir=str(staged_dir), staged_lidar_pipeline=stage_pipeline())
    summary = output / "report.json"
    export_report = source / "report.json"
    if export_report.is_file():
        try:
            exported = json.loads(export_report.read_text(encoding="utf-8"))
            report.update(export_report=str(export_report), export_report_sha256=digest(export_report),
                          fixed_lidar_topology=exported.get("fixed_lidar_topology"))
        except (OSError, ValueError) as exc:
            report["export_report_error"] = str(exc)
    if staged_dir:
        staged_report_path = staged_dir / "report.json"
        if staged_report_path.is_file():
            try:
                staged_report = json.loads(staged_report_path.read_text(encoding="utf-8"))
                report.update(staged_export_report=str(staged_report_path),
                              staged_export_report_sha256=digest(staged_report_path))
                if report["fixed_lidar_topology"] is None:
                    report["fixed_lidar_topology"] = staged_report.get("fixed_lidar_topology")
            except (OSError, ValueError) as exc:
                report["staged_export_report_error"] = str(exc)
    report["stage"] = "validate_source_provenance"
    save(summary, report)
    try:
        reports = [export_report] + ([staged_dir / "report.json"] if staged_dir else [])
        report.update(provenance_fields(source_provenance(reports, report["fixed_lidar_topology"])))
    except (OSError, ValueError) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(summary, report)
        return 1
    save(summary, report)
    for name in selected:
        report["stage"] = "compile_" + name
        save(summary, report)
        try:
            branch_source = staged_dir if name in LIDAR_STAGES else source
            hbir = nonempty(branch_source / (name + ".bc"))
            command = [sys.executable, str(SCRIPT), "--worker", "--hbir", str(hbir),
                       "--graph-name", name, "--worker-output", str(output / name)]
            if name in resume:
                command += ["--resume-object", str(resume[name])]
            report["branches"][name] = run_worker(command, output / name, args.timeout, args.rss_limit_mib, "hbo_compiled")
        except Exception as exc:
            report["branches"][name] = {"status": "failed", "hbo_compiled": False,
                                        "error": f"{type(exc).__name__}: {exc}"}
        save(summary, report)
    complete = set(report["branches"]) == set(expected_graphs) and all(
        report["branches"][name].get("hbo_compiled") for name in expected_graphs)
    report["all_required_hbo_compiled"] = complete
    report["all_six_hbo_compiled" if staged_dir else "all_three_hbo_compiled"] = complete
    if complete:
        report["stage"] = "link"
        manifest = {"fixed_lidar_topology": report["fixed_lidar_topology"], "branches": report["branches"],
                    "graph_order": selected, "hbm_path": str(output / "bevfusion.hbm"),
                    **provenance_fields(report["source_provenance"])}
        if staged_dir:
            manifest["staged_lidar_pipeline"] = report["staged_lidar_pipeline"]
        manifest_path = output / "link_manifest.json"
        save(manifest_path, manifest)
        save(summary, report)
        command = [sys.executable, str(SCRIPT), "--worker", "--link-manifest", str(manifest_path),
                   "--worker-output", str(output / "link")]
        try:
            report["link"] = run_worker(command, output / "link", args.timeout, args.rss_limit_mib,
                                         "hbm_compiled", output / "bevfusion.hbm")
        except Exception as exc:
            report["link"] = {"status": "failed", "hbm_compiled": False, "error": f"{type(exc).__name__}: {exc}"}
        report["hbm_compiled"] = bool(report["link"].get("hbm_compiled"))
        report["complete_multi_graph_hbm"] = report["hbm_compiled"]
        for key in ("hbm_path", "hbm_bytes", "hbm_sha256", "graph_names", "node_count", "hbm_metadata",
                    "actual_cpu_operations", "contracts", "source_provenance", "checkpoint_sha256", "topology_coordinates_sha256"):
            if key in report["link"]:
                report[key] = report["link"][key]
    else:
        report["link_skipped_reason"] = f"All {len(expected_graphs)} independently verified nonempty HBO graphs are required"
    report.update(status="compiled" if report["hbm_compiled"] else "partial", stage="completed")
    save(summary, report)
    print("[report]", summary, flush=True)
    requested_ok = all(item.get("hbo_compiled") for item in report["branches"].values())
    return 0 if (report["hbm_compiled"] if set(selected) == set(expected_graphs) else requested_ok) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path)
    parser.add_argument("--staged-lidar-dir", type=Path,
                        help="Replace the lidar branch with lidar_stage0.bc through lidar_stage3.bc from this directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--branches", nargs="+", choices=BRANCHES, default=["lidar", "fusion", "camera"])
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--rss-limit-mib", type=int, default=18432)
    parser.add_argument("--resume-hbo", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hbir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--graph-name", choices=GRAPH_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-object", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--link-manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.timeout < 0 or args.rss_limit_mib < 0:
        parser.error("Timeout and RSS budget must be nonnegative; zero disables the limit")
    if args.worker:
        if not args.worker_output or (bool(args.link_manifest) == bool(args.hbir)) or (args.hbir and not args.graph_name):
            parser.error("Worker requires fresh --worker-output and either --hbir/--graph-name or --link-manifest")
    elif not args.branch_dir or not args.output_dir:
        parser.error("--branch-dir and fresh --output-dir are required")
    try:
        return worker(args) if args.worker else run(args)
    except (ValueError, OSError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
