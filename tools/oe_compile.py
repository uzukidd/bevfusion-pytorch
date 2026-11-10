"""Bounded OE Docker compilation for project-owned ONNX and calibration files.

Windows invokes the existing WSL Docker daemon; Linux invokes Docker directly.
No image is pulled and no shell interprets command arguments. Dry runs write a
reviewable YAML/manifest without starting WSL, Docker, or a compiler.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0"
DEFAULT_WSL_DISTRO = "Ubuntu-26.04"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _runtime(image, wsl_distro):
    image = image or os.environ.get("OE_S600_IMAGE") or DEFAULT_IMAGE
    if not isinstance(image, str) or not image.strip() or image.startswith("-"):
        raise ValueError("image must name an existing Docker image")
    windows = platform.system() == "Windows"
    if windows and (not isinstance(wsl_distro, str) or not wsl_distro.strip()):
        raise ValueError("wsl_distro must name an existing WSL distribution")
    prefix = (["wsl.exe", "--distribution", wsl_distro, "--user", "root", "--exec"]
              if windows else [])
    return image, windows, prefix


def _probe(command, timeout_seconds):
    try:
        process = subprocess.run(command, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=timeout_seconds,
                                 check=False)
        return {"command": command, "returncode": process.returncode,
                "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None,
                "error": f"{type(exc).__name__}: {exc}"}


def detect(*, image=None, wsl_distro=DEFAULT_WSL_DISTRO, timeout_seconds=30):
    """Check the selected daemon and already-installed image without pulling it."""
    _positive_integer(timeout_seconds, "timeout_seconds")
    image, windows, prefix = _runtime(image, wsl_distro)
    info = _probe(prefix + ["docker", "info", "--format", "{{.ServerVersion}}"], timeout_seconds)
    result = {"available": False, "image": image, "platform": "windows-wsl" if windows else "linux",
              "wsl_distro": wsl_distro if windows else None, "checks": [info]}
    if info["returncode"] != 0:
        result["reason"] = "Docker is unavailable or its daemon is inaccessible"
        return result
    installed = _probe(prefix + ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                       timeout_seconds)
    result["checks"].append(installed)
    if installed["returncode"] != 0:
        result["reason"] = "The selected OE image is not installed; no implicit pull was attempted"
        return result
    result.update(available=True, docker_version=info["stdout"], image_id=installed["stdout"])
    return result


def _owned_path(value, name, project_root=None):
    root = Path(project_root).resolve() if project_root is not None else ROOT.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must resolve inside the project root: {root}") from exc
    return path, PurePosixPath("/workspace", *relative.parts).as_posix()


def _input_contract(inputs):
    if not isinstance(inputs, (list, tuple)) or not inputs:
        raise ValueError("inputs must be a nonempty list of name/shape/dtype mappings")
    result, names = [], set()
    supported = {"float32", "float16", "int8", "uint8", "int16", "int32", "int64", "bool"}
    for item in inputs:
        if not isinstance(item, dict) or not {"name", "shape", "dtype"}.issubset(item):
            raise ValueError("Each input must declare name, shape and dtype")
        name, shape, dtype = item["name"], item["shape"], item["dtype"]
        if (not isinstance(name, str) or not name.strip() or name in names
                or ";" in name or "\n" in name or "\r" in name):
            raise ValueError("Input names must be distinct, nonempty and contain no semicolons/newlines")
        if not isinstance(shape, (list, tuple)) or not shape:
            raise ValueError(f"Input {name} requires a nonempty static shape")
        shape = [_positive_integer(dimension, f"{name} shape dimension") for dimension in shape]
        if not isinstance(dtype, str) or dtype not in supported:
            raise ValueError(f"Unsupported input dtype for {name}: {dtype!r}")
        names.add(name)
        result.append({"name": name, "shape": shape, "dtype": dtype})
    return result


def _mount_source(root, windows, prefix, dry_run):
    if not windows:
        return root.as_posix(), False
    if dry_run:
        value = PureWindowsPath(str(root))
        if len(value.drive) == 2 and value.drive[1] == ":":
            return PurePosixPath("/mnt", value.drive[0].lower(), *value.parts[1:]).as_posix(), True
        return root.as_posix(), True
    converted = _probe(prefix + ["wslpath", "-a", "-u", str(root)], 30)
    if converted["returncode"] != 0 or not converted.get("stdout", "").startswith("/"):
        raise RuntimeError("Cannot translate the project root into WSL: " +
                           converted.get("stderr", converted.get("error", "invalid wslpath output")))
    return converted["stdout"], False


def _artifacts(build_dir):
    files = sorted(path for path in build_dir.rglob("*") if path.is_file()) if build_dir.exists() else []
    result = {"artifacts": [str(path) for path in files], "hbm_files": [str(path) for path in files if path.suffix == ".hbm"]}
    advice = []
    for path in files:
        if path.name.endswith("_advice.json"):
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(values, list):
                    advice.extend(item for item in values if isinstance(item, dict))
            except (OSError, ValueError):
                result.setdefault("unreadable_advice_files", []).append(str(path))
    if advice:
        result["placement_records"] = dict(Counter(item.get("backend", "unknown") for item in advice))
        result["constraint_failures"] = sum(bool(item.get("check_info")) for item in advice)
    fallback_reasons = []
    for path in files:
        if path.name.endswith("_advice.csv"):
            try:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        reason = row.get("FallbackReason", "").strip()
                        if row.get("Device", "").lower() != "bpu" or reason not in ("", "N/A"):
                            fallback_reasons.append({"node": row.get("Node"), "device": row.get("Device"),
                                                     "reason": reason})
            except (OSError, ValueError, csv.Error):
                result.setdefault("unreadable_advice_files", []).append(str(path))
    if fallback_reasons:
        result["fallback_reasons"] = fallback_reasons
    return result


def compile_hbm(onnx_path, output_dir, inputs, calibration_dirs, *, march="nash-p", image=None,
                wsl_distro=DEFAULT_WSL_DISTRO, precision="float16", optimize_level="O0", jobs=4,
                timeout_seconds=600, dry_run=False, project_root=None):
    """Write an OE PTQ configuration and compile with a bounded Docker lifetime.

    Paths must resolve inside ROOT (including symlink targets). Inputs describe
    fixed logical ONNX tensors; calibration_dirs contains one directory of NPY
    samples per input, with equal sample counts. Integer index tensors remain
    integer inputs; ``precision`` chooses node computation, not physical HBM I/O.
    Existing helper outputs/build artifacts are never overwritten. A successful
    return requires both compiler exit zero and the expected nonempty HBM file.
    Runtime errors are returned and persisted; invalid contracts raise ValueError.
    ``timeout_seconds=0`` disables both the container and host wait budgets.
    """
    import yaml

    if march != "nash-p":
        raise ValueError("This S600 helper supports march='nash-p' only")
    if precision not in ("float16", "int16", "int8"):
        raise ValueError("precision must be float16, int16 or int8")
    if optimize_level not in ("O0", "O1", "O2"):
        raise ValueError("optimize_level must be O0, O1 or O2")
    _positive_integer(jobs, "jobs")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds < 0:
        raise ValueError("timeout_seconds must be a nonnegative integer; 0 disables the time budget")
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be bool")
    image, windows, prefix = _runtime(image, wsl_distro)
    root = Path(project_root).expanduser().resolve() if project_root is not None else ROOT.resolve()
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory")
    contract = _input_contract(inputs)
    onnx_path, container_onnx = _owned_path(onnx_path, "onnx_path", root)
    output_dir, container_output = _owned_path(output_dir, "output_dir", root)
    if not onnx_path.is_file() or onnx_path.suffix.lower() != ".onnx":
        raise ValueError("onnx_path must point to an existing .onnx file")
    if not isinstance(calibration_dirs, (list, tuple)) or len(calibration_dirs) != len(contract):
        raise ValueError("calibration_dirs must provide exactly one directory per input")
    calibration_paths, container_calibration, sample_counts = [], [], []
    for index, value in enumerate(calibration_dirs):
        path, container_path = _owned_path(value, f"calibration_dirs[{index}]", root)
        if not path.is_dir():
            raise ValueError(f"Calibration directory does not exist: {path}")
        samples = sorted(path.glob("*.npy"))
        if not samples or any(not sample.is_file() or sample.stat().st_size == 0 for sample in samples):
            raise ValueError(f"Calibration directory must contain nonempty NPY samples: {path}")
        for sample in samples:
            _owned_path(sample, "calibration sample", root)
        if ";" in container_path or "\n" in container_path or "\r" in container_path:
            raise ValueError("Calibration paths cannot contain semicolons/newlines")
        calibration_paths.append(path)
        container_calibration.append(container_path)
        sample_counts.append(len(samples))
    if len(set(sample_counts)) != 1:
        raise ValueError("Calibration input directories must contain equal NPY sample counts")
    config_path, log_path = output_dir / "oe_compile.yaml", output_dir / "oe_compile.log"
    result_path, build_dir = output_dir / "oe_compile.result.json", output_dir / "oe_build"
    if any(path.exists() for path in (config_path, log_path, result_path, build_dir)):
        raise ValueError("Refusing to overwrite existing OE configuration, log, result or build directory")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output_dir must be a directory")
    container_build = PurePosixPath(container_output, "oe_build").as_posix()
    container_config = PurePosixPath(container_output, config_path.name).as_posix()
    model_prefix = onnx_path.stem
    config = {
        "model_parameters": {"onnx_model": container_onnx, "march": march,
                             "working_dir": container_build, "output_model_file_prefix": model_prefix},
        "input_parameters": {"input_name": ";".join(item["name"] for item in contract),
                             "input_shape": ";".join("x".join(map(str, item["shape"])) for item in contract),
                             "input_type_rt": ";".join("featuremap" for _ in contract),
                             "input_type_train": ";".join("featuremap" for _ in contract)},
        "calibration_parameters": {"cal_data_dir": ";".join(container_calibration),
                                   "calibration_type": "max",
                                   "quant_config": {"model_config": {"all_node_type": precision}}},
        "compiler_parameters": {"compile_mode": "latency", "core_num": 1,
                                "optimize_level": optimize_level, "jobs": jobs},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    started = time.perf_counter()
    result = {"status": "preparing", "image": image, "march": march, "precision": precision,
              "inputs": contract, "calibration_dirs": [str(path) for path in calibration_paths],
              "calibration_samples": sample_counts[0], "config_path": str(config_path),
              "log_path": str(log_path), "result_path": str(result_path), "build_dir": str(build_dir),
              "hbm_path": None, "hbm_generated": False, "hbm_compiled": False,
              "stage": "preparing", "board_verified": False,
              "quantized_accuracy_verified": False, "dry_run": dry_run}
    container_name = "bevfusion-oe-" + uuid.uuid4().hex[:12]
    try:
        if not dry_run:
            environment = detect(image=image, wsl_distro=wsl_distro)
            result["environment"] = environment
            if not environment["available"]:
                result.update(status="unavailable", error=environment["reason"])
                return result
        mount_source, estimated = _mount_source(root, windows, prefix, dry_run)
        if "," in mount_source:
            raise ValueError("Docker bind source cannot contain commas with this --mount interface")
        command = prefix + ["docker", "run", "--rm", "--pull", "never", "--name", container_name, "--network", "none",
                            "--mount", f"type=bind,source={mount_source},target=/workspace",
                            "--workdir", "/workspace", image,
                            "timeout", "--signal=TERM", "--kill-after=10s", f"{timeout_seconds}s",
                            "hb_compile", "-c", container_config]
        result.update(command=command, mount_source=mount_source, mount_source_estimated=estimated)
        if dry_run:
            result["status"] = "dry_run"
            return result
        with log_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"command": command}, ensure_ascii=False) + "\n")
            stream.flush()
            try:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=timeout_seconds + 45 if timeout_seconds else None, check=False)
                result["returncode"] = process.returncode
                result["status"] = "timeout" if process.returncode == 124 else "failed"
            except subprocess.TimeoutExpired:
                result.update(status="timeout", returncode=124, error="Host compilation wait budget expired")
                _stop_container(prefix, container_name, stream)
            except KeyboardInterrupt:
                result.update(status="interrupted", error="Compilation interrupted")
                _stop_container(prefix, container_name, stream)
                raise
        result.update(_artifacts(build_dir))
        expected_hbm = build_dir / (model_prefix + ".hbm")
        if expected_hbm.is_file() and expected_hbm.stat().st_size > 0:
            result.update(hbm_path=str(expected_hbm), hbm_generated=True)
        if result.get("returncode") == 0:
            result["status"] = "compiled" if result["hbm_generated"] else "failed"
            if not result["hbm_generated"]:
                result["error"] = "Compiler exited zero without producing the expected nonempty HBM"
        elif result["status"] == "failed":
            result["error"] = ("Compiler/container was killed (137); memory exhaustion or external termination "
                               "is possible, but timeout is not established"
                               if result.get("returncode") == 137 else
                               f"OE compiler exited with code {result.get('returncode')}")
        elif result["status"] == "timeout":
            result.setdefault("error", f"OE compilation exceeded its {timeout_seconds}-second budget")
        if log_path.is_file():
            warnings = []
            with log_path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if "Failed to generate node info" in line:
                        warnings.append(line.strip())
            if warnings:
                result["compiler_report_warnings"] = warnings
        return result
    except (OSError, RuntimeError, ValueError) as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return result
    finally:
        result["hbm_compiled"] = result["status"] == "compiled" and result["hbm_generated"]
        result["stage"] = ("complete" if result["hbm_compiled"] else
                           "dry_run" if dry_run else "environment" if result["status"] == "unavailable" else "compile")
        if result.get("error"):
            result["reason"] = result["error"]
        result["elapsed_seconds"] = time.perf_counter() - started
        if not dry_run and not log_path.exists():
            with log_path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump({"status": result["status"], "error": result.get("error"),
                           "environment": result.get("environment")}, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        with result_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def _stop_container(prefix, name, log_stream):
    """Clean up only the container created for this invocation."""
    try:
        subprocess.run(prefix + ["docker", "stop", "--timeout", "5", name], stdout=log_stream,
                       stderr=subprocess.STDOUT, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_stream.write(f"Container cleanup failed: {exc}\n")
