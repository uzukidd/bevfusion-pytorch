"""Compile the current seg LiDAR/fusion ONNX and plugin-camera HBIR sequentially."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import oe_compile as oe


def _save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args):
    split = args.split_dir.resolve(strict=True)
    hbir = args.camera_hbir.resolve(strict=True)
    manifest = json.loads((split / "graphs/manifest.json").read_text(encoding="utf-8"))
    output = (args.output_dir or ROOT / "outputs/hbm_branch_validation" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise ValueError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {"status": "running", "scope": "seg_camera_lidar_fusion", "runtime": "bevfusion-pytorch",
              "branch_order": ["lidar", "fusion", "camera"], "branches": {},
              "end_to_end_hbm_verified": False, "board_verified": False,
              "quantized_accuracy_verified": False, "output_dir": str(output)}
    summary = output / "report.json"
    _save(summary, report)
    for name in ("lidar", "fusion"):
        branch = manifest["branches"][name]
        print(f"[branch] {name}: INT16 O0 jobs=1", flush=True)
        source = Path(branch["onnx"])
        if not source.is_file():
            source = split / "graphs" / name / "model.onnx"
        calibration = [split / "calibration" / name / item["name"] for item in branch["inputs"]]
        result = oe.compile_hbm(source, output / name, branch["inputs"], calibration,
            precision="int16", optimize_level="O0", jobs=1, timeout_seconds=args.onnx_timeout,
            python_path=args.hbm_python, wsl_distro=args.wsl_distro)
        report["branches"][name] = result
        _save(summary, report)
        print("[branch result]", name, result["status"], result.get("reason"), flush=True)
    python, windows, prefix, env = oe._runtime(None, args.wsl_distro, args.hbm_python)
    runtime_root, _ = oe._runtime_root(ROOT.resolve(), windows, prefix, False)
    translate = lambda path: str(PurePosixPath(runtime_root, path.resolve().relative_to(ROOT).as_posix()))
    camera_output = output / "camera"
    pid_path = translate(output / ".camera.pid.json")
    token = uuid.uuid4().hex
    child = [python, translate(ROOT / "tools/compile_hbir_hbm.py"), "--hbir", translate(hbir),
             "--output-dir", translate(camera_output), "--opt", "0", "--jobs", "1"]
    command = prefix + env + [python, "-c", oe._RUNNER_CODE, pid_path, token,
        "timeout", "--signal=TERM", "--kill-after=10s", f"{args.camera_timeout}s"] + child
    log_path = output / "camera.log"
    camera_started = time.perf_counter()
    print("[branch] camera: unchanged floating HBIR, O0 jobs=1", flush=True)
    oe._print_command(command)
    with log_path.open("x", encoding="utf-8") as stream:
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                timeout=args.camera_timeout + 45 if args.camera_timeout else None, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            oe._stop_compiler(prefix, env, python, pid_path, token, stream)
            code = 124
        except KeyboardInterrupt:
            oe._stop_compiler(prefix, env, python, pid_path, token, stream)
            raise
    camera_report = camera_output / "report.json"
    result = json.loads(camera_report.read_text(encoding="utf-8")) if camera_report.exists() else {}
    expected = camera_output / "camera.hbm"
    success = code == 0 and result.get("hbm_compiled") and expected.is_file() and expected.stat().st_size > 0
    result.update(status="compiled" if success else "timeout" if code == 124 else "failed",
                  returncode=code, hbm_compiled=bool(success), log_path=str(log_path),
                  elapsed_seconds=time.perf_counter() - camera_started)
    if not success:
        result["reason"] = result.get("error") or ("Camera compilation time budget exceeded" if code == 124 else
            "Camera compiler returned zero without a confirmed nonempty HBM" if code == 0 else
            f"Camera compiler exited with code {code}")
    if success:
        result["hbm_path"] = str(expected)
    report["branches"]["camera"] = result
    report["all_branches_compiled"] = all(value.get("hbm_compiled") for value in report["branches"].values())
    report["status"] = "compiled" if report["all_branches_compiled"] else "partial"
    report["elapsed_seconds"] = time.perf_counter() - started
    _save(summary, report)
    print("[report]", summary, flush=True)
    print(json.dumps({name: {key: value.get(key) for key in
        ("status", "hbm_compiled", "hbm_path", "elapsed_seconds", "reason")}
        for name, value in report["branches"].items()}, ensure_ascii=False), flush=True)
    return 0 if report["all_branches_compiled"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--camera-hbir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hbm-python")
    parser.add_argument("--wsl-distro", default="Ubuntu-26.04")
    parser.add_argument("--onnx-timeout", type=int, default=900)
    parser.add_argument("--camera-timeout", type=int, default=1200)
    args = parser.parse_args(argv)
    if min(args.onnx_timeout, args.camera_timeout) < 0:
        parser.error("Timeouts must be nonnegative; zero disables the time budget")
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
