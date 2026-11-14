"""Compare mixed SubMConv/Conv3D inference and export the four dense operators."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    from tools.horizon_runtime import ensure_plugin_loader
    ensure_plugin_loader()

import torch
from torch import nn
from bevfusion import BEVFusionSeg, BEVFusionDet
from bevfusion.operators.sparse_conv.reference import ReferenceSparseConv, keys
from bevfusion.sample_io import load_samples


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metrics(actual, expected, atol, rtol):
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    difference = (actual.double() - expected.double()).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {"shape": list(actual.shape), "finite": finite,
            "max_abs": float(difference.max()) if difference.numel() else 0.,
            "mean_abs": float(difference.mean()) if difference.numel() else 0.,
            "passed": finite and bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))}


def synchronized(device):
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def export_operators(backbone, directory, *, full_tiles=False, tile_size=16):
    """Probe float Conv3D export with original weights, not whole-backbone placement."""
    from horizon_plugin_pytorch.quantization.hbdk4 import export
    from hbdk4 import compiler as hbc
    from bevfusion.operators.sparse_conv.conv3d import DenseConv3dPatch

    class BoundPatch(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.weight = nn.Parameter(source.weight.detach().cpu().clone(), requires_grad=False)
            self.operation = DenseConv3dPatch(source.stride, (0, source.padding[1], source.padding[2]))

        def forward(self, patch):
            return self.operation(patch, self.weight)

    directory.mkdir(parents=True)
    records = []
    spatial_shape = tuple(backbone.sparse_shape)
    for name, source in backbone.named_modules():
        if not isinstance(source, ReferenceSparseConv) or source.subm:
            continue
        # Full mode uses the real runtime patch, including explicit X halos.
        module = BoundPatch(source).eval()
        patch_shape = (((tile_size - 1) * source.stride[0] + source.kernel[0],
                        spatial_shape[1], spatial_shape[2]) if full_tiles else (7, 9, 7))
        value = torch.randn(1, source.weight.shape[-2], *patch_shape)
        spatial_shape = tuple((s + 2 * p - k) // st + 1 for s, p, k, st in
                              zip(spatial_shape, source.padding, source.kernel, source.stride))
        path = directory / (name.replace(".", "_") + ".bc")
        with torch.no_grad():
            exported = export(module, (value,), input_names=["patch"], output_names=["output"])
            hbc.save(exported, str(path))
            loaded = hbc.load(str(path))
            comparisons = []
            for current in (value, value * .7):
                result = loaded.functions[0](current.numpy())
                result = result[0] if isinstance(result, (tuple, list)) else result
                comparisons.append(metrics(torch.as_tensor(result), module(current), 1e-4, 1e-4))
        records.append({"layer": name, "weight_shape": list(source.weight.shape),
            "input_shape": list(value.shape), "hbir": str(path), "comparisons": comparisons,
            "passed": all(item["passed"] for item in comparisons)})
        print("[hbir]", json.dumps(records[-1]), flush=True)
    return {"float_operator_probes": records, "qat_prepared": False, "hbm_compiled": False,
            "whole_lidar_exported": False, "passed": len(records) == 4 and all(r["passed"] for r in records)}


def run(args):
    torch.manual_seed(831)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = (args.output_dir or ROOT / "outputs/hybrid_lidar" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "device": args.device, "torch": torch.__version__,
        "baseline": "auto", "candidate": "hybrid", "pool_backend": args.pool_backend,
        "atol": args.atol, "rtol": args.rtol, "tasks": [], "hbir": []}
    started = time.perf_counter()
    try:
        sample = (torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs
                  else load_samples(args.samples, calibration=args.calibration))
        for task in args.tasks:
            factory = BEVFusionSeg if task == "seg" else BEVFusionDet
            checkpoint = args.seg_checkpoint if task == "seg" else args.det_checkpoint
            model = factory.from_pretrained(checkpoint, device=args.device, backend="auto",
                lidar_backend="auto", pool_backend=args.pool_backend, camera_batch_size=1)
            before = fingerprint(model)
            backbone = model.encoders["lidar"]["backbone"]
            convolutions = {name: layer for name, layer in backbone.named_modules()
                            if isinstance(layer, ReferenceSparseConv)}
            ordinary = {name: layer for name, layer in convolutions.items() if not layer.subm}
            task_record = {"task": task, "strict_load": model.load_info,
                "submanifold_layers": sum(layer.subm for layer in convolutions.values()),
                "conv3d_layers": list(ordinary), "state_before": before, "cases": []}
            report["tasks"].append(task_record)
            for case in args.cases:
                arguments = [sample["images"], sample["points"], sample["camera_intrinsics"],
                             sample["camera2lidar"], sample.get("img_aug_matrix")]
                if case == "half_points":
                    arguments[1] = arguments[1][::2].contiguous()
                baseline_layers = {}
                def capture(name):
                    def hook(module, inputs, result):
                        order = keys(result.coords, result.shape).argsort()
                        baseline_layers[name] = (result.features[order].detach().cpu(),
                                                 result.coords[order].detach().cpu(), result.shape)
                    return hook
                handles = [layer.register_forward_hook(capture(name)) for name, layer in ordinary.items()]
                baseline_bev = []
                handles.append(backbone.register_forward_hook(lambda module, inputs, result:
                    baseline_bev.append(result.detach().cpu())))
                try:
                    model.lidar_backend = "auto"
                    for layer in convolutions.values():
                        layer.backend = "auto"
                    with torch.inference_mode():
                        start = synchronized(args.device)
                        expected = model(*arguments)
                        baseline_seconds = synchronized(args.device) - start
                finally:
                    for handle in handles:
                        handle.remove()
                layer_records = []
                def compare_layer(name):
                    def hook(module, inputs, result):
                        order = keys(result.coords, result.shape).argsort()
                        expected_features, expected_coords, expected_shape = baseline_layers[name]
                        actual_features = result.features[order].detach().cpu()
                        record = {"layer": name, "coordinates_equal": result.shape == expected_shape and
                            torch.equal(result.coords[order].cpu(), expected_coords),
                            "features": metrics(actual_features, expected_features, args.atol, args.rtol)}
                        record["passed"] = record["coordinates_equal"] and record["features"]["passed"]
                        layer_records.append(record)
                    return hook
                actual_bev = []
                handles = [layer.register_forward_hook(compare_layer(name)) for name, layer in ordinary.items()]
                handles.append(backbone.register_forward_hook(lambda module, inputs, result:
                    actual_bev.append(result.detach().cpu())))
                try:
                    model.lidar_backend = "hybrid"
                    for layer in convolutions.values():
                        layer.backend = "hybrid"
                    if args.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():
                        start = synchronized(args.device)
                        actual = model(*arguments)
                        hybrid_seconds = synchronized(args.device) - start
                    peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
                finally:
                    for handle in handles:
                        handle.remove()
                outputs = ({"probabilities": metrics(actual, expected, args.atol, args.rtol)} if task == "seg"
                    else {name: metrics(actual[name], expected[name], args.atol, args.rtol)
                          for name in ("boxes_3d", "scores", "labels")})
                if task == "seg":
                    outputs["probabilities"]["mask_disagreements_at_0.5"] = int(((actual >= .5) != (expected >= .5)).sum())
                else:
                    outputs["labels"]["equal"] = torch.equal(actual["labels"], expected["labels"])
                bev_record = metrics(actual_bev[0], baseline_bev[0], args.atol, args.rtol)
                record = {"case": case, "baseline_seconds": baseline_seconds, "hybrid_seconds": hybrid_seconds,
                    "hybrid_peak_allocated_bytes": peak, "ordinary_layers": layer_records,
                    "lidar_bev": bev_record, "outputs": outputs}
                record["passed"] = all(r["passed"] for r in layer_records) and bev_record["passed"] and all(
                    r["passed"] for r in outputs.values())
                task_record["cases"].append(record)
                print("[case]", json.dumps({"task": task, **record}), flush=True)
                del actual, expected, baseline_layers, baseline_bev, actual_bev
            after = fingerprint(model)
            task_record.update(state_after=after, state_unchanged=before == after)
            if args.export_hbir:
                report["hbir"].append({"task": task, **export_operators(
                    backbone, output / task / "hbir", full_tiles=args.full_tiles)})
            del model, backbone, convolutions, ordinary
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        passed = all(task["state_unchanged"] and task["submanifold_layers"] == 17 and len(task["conv3d_layers"]) == 4
                     and all(case["passed"] for case in task["cases"]) for task in report["tasks"])
        passed = passed and all(item["passed"] for item in report["hbir"])
        report["status"] = "verified" if passed else "comparison_failed"
        return 0 if passed else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[report]", output / "report.json", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("seg", "det"), default=["seg", "det"])
    parser.add_argument("--cases", nargs="+", choices=("provided", "half_points"), default=["provided", "half_points"])
    parser.add_argument("--seg-checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-seg.pth")
    parser.add_argument("--det-checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-det.pth")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--samples", type=Path, default=ROOT / "data/samples")
    source.add_argument("--inputs", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pool-backend", choices=("reference", "cuda", "hat", "hat_plugin"), default="reference")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--export-hbir", action="store_true")
    parser.add_argument("--full-tiles", action="store_true",
                        help="Export operator probes at the actual runtime tile dimensions")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    import math
    if args.threads < 1 or not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error("threads must be positive; tolerances must be finite and nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
