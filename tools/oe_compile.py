"""Bounded ONNX compilation in the selected bevfusion-pytorch Conda environment.

Windows invokes WSL; Linux invokes the same compiler environment directly.
No Docker, root account, shell command interpolation, or environment activation is
used. Dry runs write reviewable configuration without launching a compiler.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shlex
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
# Kept only for import compatibility; images are no longer accepted by this API.
DEFAULT_IMAGE = "registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0"
DEFAULT_WSL_DISTRO = "Ubuntu-26.04"
DEFAULT_WSL_PYTHON = "/home/administrator/miniconda3/envs/bevfusion-pytorch/bin/python"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _runtime(image, wsl_distro, python_path=None):
    if image is not None:
        raise ValueError("--oe-image is no longer supported; HBM compilation uses the bevfusion-pytorch Conda environment")
    system = platform.system()
    if system not in ("Windows", "Linux"):
        raise ValueError("HBM compilation requires WSL on Windows or native Linux")
    windows = system == "Windows"
    if windows and (not isinstance(wsl_distro, str) or not wsl_distro.strip()):
        raise ValueError("wsl_distro must name an existing WSL distribution")
    python_path = (python_path or os.environ.get("BEVFUSION_HBM_PYTHON")
                   or os.environ.get("BEVFUSION_WSL_PYTHON") or DEFAULT_WSL_PYTHON)
    if (not isinstance(python_path, str) or not python_path.startswith("/")
            or any(char in python_path for char in ("\n", "\r", "\0"))):
        raise ValueError("python_path must be an absolute Linux Conda Python path")
    prefix = ["wsl.exe", "--distribution", wsl_distro, "--exec"] if windows else []
    binary = PurePosixPath(python_path).parent
    env = ["env", f"PATH={binary}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
           "GLIBC_TUNABLES=glibc.rtld.execstack=2"]
    return python_path, windows, prefix, env


def _format_command(command):
    """Render argv for the host shell; execution still uses argv, never a shell."""
    if platform.system() == "Windows":
        # PowerShell single-quoted arguments preserve spaces, $, &, and quotes.
        return "& " + " ".join("'" + str(part).replace("'", "''") + "'" for part in command)
    return shlex.join(str(part) for part in command)


def _print_command(command, *, dry_run=False):
    rendered = _format_command(command)
    label = "[dry-run command; not executed]" if dry_run else "[command]"
    print(f"{label} {rendered}", flush=True)
    return rendered


def _probe(command, timeout_seconds):
    _print_command(command)
    try:
        process = subprocess.run(command, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=timeout_seconds,
                                 check=False)
        return {"command": command, "returncode": process.returncode,
                "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None,
                "error": f"{type(exc).__name__}: {exc}"}


_DETECT_CODE = """import importlib.metadata as m, json, pathlib, shutil, sys
import horizon_tc_ui, hmct, hbdk4.compiler
command = shutil.which('hb_compile')
expected = pathlib.Path(sys.executable).parent / 'hb_compile'
if command is None or pathlib.Path(command).resolve() != expected.resolve():
    raise RuntimeError('hb_compile is missing from the selected Conda bin directory')
print(json.dumps({'python':sys.executable, 'prefix':sys.prefix, 'hb_compile':command,
    'versions':{name:m.version(name) for name in ('horizon-tc-ui','hbdk4-compiler')}}))
"""


def detect(*, image=None, wsl_distro=DEFAULT_WSL_DISTRO, timeout_seconds=30, python_path=None):
    """Import the compiler stack and verify its entry point in the selected Conda bin."""
    _positive_integer(timeout_seconds, "timeout_seconds")
    python_path, windows, prefix, env = _runtime(image, wsl_distro, python_path)
    check = _probe(prefix + env + [python_path, "-c", _DETECT_CODE], timeout_seconds)
    result = {"available": False, "backend": "wsl-conda" if windows else "linux-conda",
              "runtime_python": python_path, "wsl_distro": wsl_distro if windows else None, "checks": [check]}
    if check["returncode"] != 0:
        result["reason"] = "Selected Conda environment cannot load horizon-tc-ui, HMCT, HBDK or its hb_compile entry point"
        return result
    try:
        details = json.loads(check["stdout"].splitlines()[-1])
        if not isinstance(details, dict) or not details.get("hb_compile"):
            raise ValueError("Missing hb_compile path")
    except (ValueError, IndexError) as exc:
        result["reason"] = f"Invalid compiler environment probe response: {exc}"
        return result
    result.update(available=True, **details)
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
    return path, PurePosixPath(*relative.parts).as_posix()


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


def _runtime_root(root, windows, prefix, dry_run):
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


# A small Linux supervisor owns a single process group. The token lives in the
# child's environment and a private pid record; cancellation never searches by
# compiler name and cannot terminate an unrelated user's compiler invocation.
_RUNNER_CODE = """import json, os, pathlib, signal, subprocess, sys
pid_path, token, *command = sys.argv[1:]
env = dict(os.environ, BEVFUSION_COMPILE_ID=token)
process = subprocess.Popen(command, env=env, start_new_session=True)
record = pathlib.Path(pid_path)
try:
    with record.open('x', encoding='utf-8') as stream:
        json.dump({'token':token, 'pid':process.pid}, stream)
    code = process.wait()
except BaseException:
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait()
    raise
finally:
    if record.exists():
        data = json.loads(record.read_text(encoding='utf-8'))
        if data.get('token') == token: record.unlink()
sys.exit(code if code >= 0 else 128 - code)
"""
_STOP_CODE = """import json, os, pathlib, signal, sys, time
record, token = pathlib.Path(sys.argv[1]), sys.argv[2]
if not record.exists(): sys.exit(0)
data = json.loads(record.read_text(encoding='utf-8'))
if data.get('token') != token: raise RuntimeError('Compile owner token mismatch')
pid = int(data['pid'])
if pid <= 1: raise RuntimeError('Invalid compile process group')
proc = pathlib.Path('/proc') / str(pid)
try: environment = (proc / 'environ').read_bytes().split(b'\\0')
except FileNotFoundError: sys.exit(0)
if ('BEVFUSION_COMPILE_ID=' + token).encode() not in environment:
    raise RuntimeError('Compile process identity changed; refusing cleanup')
if os.getpgid(pid) != pid: raise RuntimeError('Compiler is not its own process-group leader')
os.killpg(pid, signal.SIGTERM)
for _ in range(50):
    if not proc.exists(): sys.exit(0)
    time.sleep(.1)
try:
    environment = (proc / 'environ').read_bytes().split(b'\\0')
    if ('BEVFUSION_COMPILE_ID=' + token).encode() not in environment or os.getpgid(pid) != pid:
        raise RuntimeError('Compile identity changed while waiting; refusing forced cleanup')
    os.killpg(pid, signal.SIGKILL)
except (FileNotFoundError, ProcessLookupError): pass
"""


def _stop_compiler(prefix, env, python_path, pid_path, token, log_stream):
    command = prefix + env + [python_path, "-c", _STOP_CODE, pid_path, token]
    _print_command(command)
    try:
        subprocess.run(command, stdout=log_stream, stderr=subprocess.STDOUT, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_stream.write(f"Compiler process-group cleanup failed: {exc}\n")


def compile_hbm(onnx_path, output_dir, inputs, calibration_dirs, *, march="nash-p", image=None,
                wsl_distro=DEFAULT_WSL_DISTRO, precision="float16", optimize_level="O0", jobs=4,
                timeout_seconds=600, dry_run=False, project_root=None, python_path=None):
    """Compile with hb_compile from the selected Linux Conda environment.

    Source/calibration/output paths stay project-owned and unchanged. Logical
    integer inputs retain their dtype. A successful result requires exit zero
    and the expected nonempty HBM file. Zero timeout disables both budgets.
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
    python_path, windows, prefix, env = _runtime(image, wsl_distro, python_path)
    root = Path(project_root).expanduser().resolve() if project_root is not None else ROOT.resolve()
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory")
    contract = _input_contract(inputs)
    onnx_path, relative_onnx = _owned_path(onnx_path, "onnx_path", root)
    output_dir, relative_output = _owned_path(output_dir, "output_dir", root)
    if not onnx_path.is_file() or onnx_path.suffix.lower() != ".onnx":
        raise ValueError("onnx_path must point to an existing .onnx file")
    if not isinstance(calibration_dirs, (list, tuple)) or len(calibration_dirs) != len(contract):
        raise ValueError("calibration_dirs must provide exactly one directory per input")
    calibration_paths, relative_calibration, sample_counts = [], [], []
    for index, value in enumerate(calibration_dirs):
        path, relative = _owned_path(value, f"calibration_dirs[{index}]", root)
        if not path.is_dir():
            raise ValueError(f"Calibration directory does not exist: {path}")
        samples = sorted(path.glob("*.npy"))
        if not samples or any(not sample.is_file() or sample.stat().st_size == 0 for sample in samples):
            raise ValueError(f"Calibration directory must contain nonempty NPY samples: {path}")
        for sample in samples:
            _owned_path(sample, "calibration sample", root)
        if any(char in relative for char in (";", "\n", "\r")):
            raise ValueError("Calibration paths cannot contain semicolons/newlines")
        calibration_paths.append(path)
        relative_calibration.append(relative)
        sample_counts.append(len(samples))
    if len(set(sample_counts)) != 1:
        raise ValueError("Calibration input directories must contain equal NPY sample counts")
    config_path, log_path = output_dir / "oe_compile.yaml", output_dir / "oe_compile.log"
    result_path, build_dir = output_dir / "oe_compile.result.json", output_dir / "oe_build"
    if any(path.exists() for path in (config_path, log_path, result_path, build_dir)):
        raise ValueError("Refusing to overwrite existing OE configuration, log, result or build directory")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output_dir must be a directory")
    runtime_root, estimated = _runtime_root(root, windows, prefix, dry_run)
    runtime_onnx = PurePosixPath(runtime_root, relative_onnx).as_posix()
    runtime_output = PurePosixPath(runtime_root, relative_output)
    runtime_calibration = [PurePosixPath(runtime_root, value).as_posix() for value in relative_calibration]
    config = {
        "model_parameters": {"onnx_model": runtime_onnx, "march": march,
            "working_dir": (runtime_output / "oe_build").as_posix(), "output_model_file_prefix": onnx_path.stem},
        "input_parameters": {"input_name": ";".join(item["name"] for item in contract),
            "input_shape": ";".join("x".join(map(str, item["shape"])) for item in contract),
            "input_type_rt": ";".join("featuremap" for _ in contract),
            "input_type_train": ";".join("featuremap" for _ in contract)},
        "calibration_parameters": {"cal_data_dir": ";".join(runtime_calibration), "calibration_type": "max",
            "quant_config": {"model_config": {"all_node_type": precision}}},
        "compiler_parameters": {"compile_mode": "latency", "core_num": 1,
            "optimize_level": optimize_level, "jobs": jobs},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    token = uuid.uuid4().hex
    pid_path = (runtime_output / (".compile-" + token + ".pid.json")).as_posix()
    compiler_command = ["timeout", "--signal=TERM", "--kill-after=10s", f"{timeout_seconds}s",
                        "hb_compile", "-c", (runtime_output / config_path.name).as_posix()]
    command = prefix + env + [python_path, "-c", _RUNNER_CODE, pid_path, token] + compiler_command
    started = time.perf_counter()
    result = {"status": "preparing", "backend": "wsl-conda" if windows else "linux-conda",
              "runtime_python": python_path, "runtime_root": runtime_root, "runtime_root_estimated": estimated,
              "march": march, "precision": precision, "inputs": contract,
              "calibration_dirs": [str(path) for path in calibration_paths], "calibration_samples": sample_counts[0],
              "config_path": str(config_path), "log_path": str(log_path), "result_path": str(result_path),
              "build_dir": str(build_dir), "hbm_path": None, "hbm_generated": False, "hbm_compiled": False,
              "stage": "preparing", "board_verified": False, "quantized_accuracy_verified": False,
              "dry_run": dry_run, "command": command, "compiler_command": compiler_command}
    try:
        if not dry_run:
            environment = detect(wsl_distro=wsl_distro, python_path=python_path)
            result["environment"] = environment
            if not environment["available"]:
                result.update(status="unavailable", error=environment["reason"])
                return result
        invocation_label = "[planned compiler invocation; not executed]" if dry_run else "[compiler invocation]"
        print(invocation_label + " " + _format_command(prefix + env + compiler_command), flush=True)
        result["command_display"] = _print_command(command, dry_run=dry_run)
        print(f"[compiler config] {config_path}", flush=True)
        print(f"[compiler log] {log_path}", flush=True)
        if dry_run:
            result["status"] = "dry_run"
            return result
        with log_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"command": command, "compiler_command": compiler_command}, ensure_ascii=False) + "\n")
            stream.flush()
            try:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=timeout_seconds + 45 if timeout_seconds else None, check=False)
                result["returncode"] = process.returncode
                result["status"] = "timeout" if process.returncode == 124 else "failed"
            except subprocess.TimeoutExpired:
                result.update(status="timeout", returncode=124, error="Host compilation wait budget expired")
                _stop_compiler(prefix, env, python_path, pid_path, token, stream)
            except KeyboardInterrupt:
                result.update(status="interrupted", error="Compilation interrupted")
                _stop_compiler(prefix, env, python_path, pid_path, token, stream)
                raise
        result.update(_artifacts(build_dir))
        expected = build_dir / (onnx_path.stem + ".hbm")
        if expected.is_file() and expected.stat().st_size > 0:
            result.update(hbm_path=str(expected), hbm_generated=True)
        if result.get("returncode") == 0:
            result["status"] = "compiled" if result["hbm_generated"] else "failed"
            if not result["hbm_generated"]:
                result["error"] = "Compiler exited zero without producing the expected nonempty HBM"
        elif result["status"] == "failed":
            result["error"] = ("Compiler was killed (137); memory exhaustion or external termination is possible, "
                "but timeout is not established" if result.get("returncode") == 137 else
                f"OE compiler exited with code {result.get('returncode')}")
        elif result["status"] == "timeout":
            result.setdefault("error", f"OE compilation exceeded its {timeout_seconds}-second budget")
        if log_path.is_file():
            warnings = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if "Failed to generate node info" in line]
            if warnings:
                result["compiler_report_warnings"] = warnings
        return result
    except (OSError, RuntimeError, ValueError) as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return result
    finally:
        result["hbm_compiled"] = result["status"] == "compiled" and result["hbm_generated"]
        result["stage"] = ("complete" if result["hbm_compiled"] else "dry_run" if dry_run else
                           "environment" if result["status"] == "unavailable" else "compile")
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
