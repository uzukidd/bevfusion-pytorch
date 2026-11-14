"""Export and compile a fixed-topology seg Hybrid + hat_plugin S600 test graph."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    from tools.horizon_runtime import ensure_plugin_loader
    ensure_plugin_loader()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def compare(actual, expected, atol, rtol):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    equal_shape = actual.shape == expected.shape
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64)) if equal_shape else np.array([float("inf")])
    return {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
        "max_abs": float(difference.max()) if difference.size else 0.,
        "mean_abs": float(difference.mean()) if difference.size else 0.,
        "passed": bool(equal_shape and np.isfinite(actual).all() and np.allclose(actual, expected, atol=atol, rtol=rtol))}


def compile_artifact(source, output, timeout_seconds, rss_limit_mib=18432):
    command = [sys.executable, str(ROOT / "tools/compile_hbir_hbm.py"), "--hbir", str(source),
               "--output-dir", str(output), "--opt", "0", "--jobs", "1"]
    if sys.platform == "linux" and timeout_seconds:
        command = ["timeout", "--signal=TERM", "--kill-after=10s", str(timeout_seconds)] + command
    log = output.parent / (output.name + ".compile.log")
    started = time.perf_counter()
    print("[compile]", source, "->", output, flush=True)
    peak_rss_kib, memory_limited = 0, False

    def process_group_rss(pgid):
        total = 0
        if sys.platform == "linux":
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    if int(fields[2]) == pgid:
                        status = (entry / "status").read_text()
                        total += next((int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:")), 0)
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
        return total

    def terminate_group(child, grace=5):
        import signal
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
            time.sleep(.1)
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()

    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                rss = process_group_rss(child.pid)
                peak_rss_kib = max(peak_rss_kib, rss)
                if rss_limit_mib and rss > rss_limit_mib * 1024:
                    memory_limited = True
                    terminate_group(child)
                    code = 137
                    break
                try:
                    code = child.wait(timeout=.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            terminate_group(child)
            raise
    report_path = output / "report.json"
    result = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    expected = output / "camera.hbm"
    passed = bool(code == 0 and result.get("hbm_compiled") and expected.is_file() and expected.stat().st_size)
    result.update(returncode=code, log=str(log), elapsed_seconds=time.perf_counter() - started,
        status="compiled" if passed else "memory_limit" if memory_limited else "timeout" if code == 124 else "failed",
        hbm_compiled=passed, process_group_peak_rss_kib=peak_rss_kib, rss_limit_mib=rss_limit_mib)
    if not passed:
        result["reason"] = (f"Compiler process group exceeded {rss_limit_mib} MiB RSS budget" if memory_limited else
                            result.get("error") or f"Compiler exited {code}; no confirmed HBM")
    save(output.parent / (output.name + ".compile.json"), result)
    return result


def run(args):
    import numpy as np
    import torch
    from torch import nn
    from bevfusion import BEVFusionSeg
    from bevfusion.deployment_inputs import _prepare_sample
    from bevfusion.sample_io import load_samples
    from bevfusion.preprocessing import voxelize
    from bevfusion.operators.sparse_conv.plan import plan_sparse_encoder
    from bevfusion.operators.sparse_conv.hybrid_export import HybridExportEncoder
    from tools.export_hat_plugin_camera import build_camera, _flat_arrays
    from horizon_plugin_pytorch.quantization.hbdk4 import export
    import hbdk4.compiler as hbc

    class Fusion(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.fuser, self.decoder, self.head = model.fuser, model.decoder, model.heads["map"]

        def forward(self, camera_bev, lidar_bev):
            fused = self.fuser(torch.cat((camera_bev, lidar_bev), dim=1))
            return self.head(self.decoder["neck"](self.decoder["backbone"](fused))[0])

    class Combined(nn.Module):
        def __init__(self, camera, lidar, fusion):
            super().__init__()
            self.camera, self.lidar, self.fusion = camera, lidar, fusion

        def forward(self, images, voxel_features):
            return self.fusion(self.camera(images), self.lidar(voxel_features))

    torch.set_num_threads(args.threads)
    torch.manual_seed(831)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = (args.output_dir or ROOT / "outputs/hybrid_hat_hbm" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "stage": "load", "task": "seg", "runtime_python": sys.executable,
        "lidar_backend": "hybrid", "pool_backend": "hat_plugin", "pool_num_points": args.pool_num_points,
        "fixed_lidar_topology": True, "dynamic_pointcloud_supported": False,
        "voxelization": "cpu_outside_graph", "precision": "float32_source",
        "qat_prepared": False, "quantized_accuracy_verified": False, "board_verified": False,
        "hbm_end_to_end_verified": False, "exports": {}, "compilation": {}, "atol": args.atol, "rtol": args.rtol}
    summary = output / "report.json"
    started = time.perf_counter()
    checkpoint_hash = digest(args.checkpoint)
    try:
        model = BEVFusionSeg.from_pretrained(args.checkpoint, backend="reference", lidar_backend="hybrid",
            pool_backend="hat_plugin", pool_num_points=args.pool_num_points, device="cpu", camera_batch_size=0)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        report.update(checkpoint_sha256=checkpoint_hash, strict_load=model.load_info)
        sample = (torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else
                  load_samples(args.samples, calibration=args.calibration))
        images, points, calibration = _prepare_sample(model, sample, None)
        with torch.inference_mode():
            features, coordinates = voxelize(points, max_voxels=model.max_voxels, voxel_size=model.voxel_size,
                point_cloud_range=model.point_cloud_range, backend="reference")
            plan = plan_sparse_encoder(model.encoders["lidar"]["backbone"], coordinates)
            lidar = HybridExportEncoder(model.encoders["lidar"]["backbone"], plan, coordinates, tile_size=args.tile_size)
            lidar.validate_topology(coordinates)
            camera = build_camera(model, calibration, num_points=args.pool_num_points)
            fusion = Fusion(model).eval()
            combined = Combined(camera, lidar, fusion).eval()
            torch.save({"coords": coordinates, "voxel_features": features, "images": images}, output / "fixed_inputs.pt")
            report["topology"] = {**lidar.stats, "coordinates_sha256": hashlib.sha256(coordinates.numpy().tobytes()).hexdigest(),
                "sparse_shape": list(lidar.sparse_shape), "final_shape": list(lidar.final_shape),
                "active_counts": {name: item.output_count for name, item in plan.plans.items()}}
            save(summary, report)
            print("[topology]", json.dumps(report["topology"]), flush=True)
            # CUDA only for eager equivalence: topology is still computed by the host.
            if args.verify_device == "cuda":
                combined.cuda()
                model.cuda()
            device = args.verify_device
            current_images, current_features = images.to(device), features.to(device)
            report["stage"] = "adapter_verify"
            save(summary, report)
            expected_lidar = model.encoders["lidar"]["backbone"](current_features, coordinates.to(device), verbose=False)
            actual_lidar = lidar(current_features)
            expected_camera = camera(current_images)
            expected = fusion(expected_camera, expected_lidar)
            actual = fusion(expected_camera, actual_lidar)
            report["adapter_lidar"] = compare(actual_lidar.cpu().numpy(), expected_lidar.cpu().numpy(), args.atol, args.rtol)
            report["adapter_combination"] = compare(actual.cpu().numpy(), expected.cpu().numpy(), args.atol, args.rtol)
            report["adapter_combination"]["mask_disagreements_at_0.5"] = int(((actual >= .5) != (expected >= .5)).sum())
            if not report["adapter_lidar"]["passed"] or not report["adapter_combination"]["passed"]:
                raise AssertionError("Fixed-topology hybrid adapter differs from eager hybrid")
            expected_lidar, expected_camera, expected = (value.detach().cpu() for value in (actual_lidar, expected_camera, actual))
            del actual_lidar, actual, current_images, current_features
            combined.cpu(); model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            camera.pool.set_qconfig()
            scopes = args.scopes
            examples = {"lidar": (lidar, (features,), ["voxel_features"], ["lidar_bev"], expected_lidar),
                        "fusion": (fusion, (expected_camera, expected_lidar), ["camera_bev", "lidar_bev"], ["probabilities"], expected),
                        "camera": (camera, (images,), ["images"], ["camera_bev"], expected_camera),
                        "combined": (combined, (images, features), ["images", "voxel_features"], ["probabilities"], expected)}
            for scope in scopes:
                module, example, input_names, output_names, reference = examples[scope]
                report["stage"] = "export_" + scope
                save(summary, report)
                path = output / (scope + ".bc")
                print("[export]", scope, flush=True)
                t0 = time.perf_counter()
                exported = export(module, example, input_names=input_names, output_names=output_names)
                hbc.save(exported, str(path))
                del exported
                gc.collect()
                record = {"hbir": str(path), "bytes": path.stat().st_size,
                    "export_seconds": time.perf_counter() - t0,
                    "inputs": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                               for name, value in zip(input_names, example)], "hbir_replayed": False}
                report["exports"][scope] = record
                save(summary, report)
                if args.hbir_replay:
                    report["stage"] = "hbir_replay_" + scope
                    save(summary, report)
                    replay_module = hbc.load(str(path))
                    replay = _flat_arrays(replay_module.functions[0](*(value.numpy() for value in example)))
                    record.update(hbir_replayed=True, comparison=compare(replay[0], reference.numpy(), args.atol, args.rtol))
                    del replay_module
                    if not record["comparison"]["passed"]:
                        raise AssertionError(f"{scope} HBIR replay differs from eager output")
                gc.collect()
                save(summary, report)
            current = model.state_dict()
            report["source_state_unchanged"] = set(before) == set(current) and all(
                torch.equal(before[name], value.cpu()) for name, value in current.items())
            if not report["source_state_unchanged"]:
                raise AssertionError("Export modified source model state")
        del combined, model, lidar, camera, fusion, before, current, examples, scopes, module
        gc.collect()
        if args.compile:
            for scope, item in report["exports"].items():
                report["stage"] = "compile_" + scope
                save(summary, report)
                report["compilation"][scope] = compile_artifact(Path(item["hbir"]), output / (scope + "_hbm"), args.timeout, args.rss_limit_mib)
                save(summary, report)
            report["all_requested_hbm_compiled"] = all(item["hbm_compiled"] for item in report["compilation"].values())
        report["status"] = ("compiled" if report.get("all_requested_hbm_compiled") else
                            "partial" if args.compile else "exported")
        report["stage"] = "completed"
        return 0 if not args.compile or report["all_requested_hbm_compiled"] else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["checkpoint_unchanged"] = digest(args.checkpoint) == checkpoint_hash
        report["elapsed_seconds"] = time.perf_counter() - started
        save(summary, report)
        print("[report]", summary, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-seg.pth")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/samples")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scopes", nargs="+", choices=("lidar", "fusion", "camera", "combined"), default=["combined"])
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--pool-num-points", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--hbir-replay", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--rss-limit-mib", type=int, default=18432,
                        help="Stop only this compiler process group above this RSS budget; 0 disables")
    parser.add_argument("--hbir", type=Path, help="Compile saved HBIR directly without loading the PyTorch model")
    parser.add_argument("--tile-hbir-dir", type=Path, help="Compile the four separately exported Conv3D runtime tiles")
    args = parser.parse_args(argv)
    if min(args.tile_size, args.threads) < 1 or not 1 <= args.pool_num_points <= 1024 or min(args.timeout, args.rss_limit_mib) < 0:
        parser.error("Invalid tile/thread count, pooling K, memory budget or timeout")
    if args.tile_hbir_dir:
        if args.hbir or args.output_dir is None:
            parser.error("--tile-hbir-dir requires --output-dir and cannot combine with --hbir")
        source = args.tile_hbir_dir.resolve(strict=True)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        result = {"scope": "four_hybrid_conv3d_tiles", "whole_combination_compiled": False,
                  "board_verified": False, "operators": {}}
        for name in ("encoder_layers_encoder_layer1_2_0", "encoder_layers_encoder_layer2_2_0",
                     "encoder_layers_encoder_layer3_2_0", "conv_out_0"):
            result["operators"][name] = compile_artifact((source / (name + ".bc")).resolve(strict=True),
                output / name, args.timeout, args.rss_limit_mib)
            save(output / "report.json", result)
        result["all_operators_compiled"] = all(value["hbm_compiled"] for value in result["operators"].values())
        result["status"] = "compiled" if result["all_operators_compiled"] else "partial"
        save(output / "report.json", result)
        print("[report]", output / "report.json", flush=True)
        return 0 if result["all_operators_compiled"] else 1
    if args.hbir:
        if args.output_dir is None:
            parser.error("--hbir requires --output-dir")
        source = args.hbir.resolve(strict=True)
        output = args.output_dir.resolve()
        if output.exists():
            parser.error("Output directory already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        result = compile_artifact(source, output, args.timeout, args.rss_limit_mib)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["hbm_compiled"] else 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
