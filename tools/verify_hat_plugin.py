"""Verify original det/seg checkpoints and real HAT-plugin floating inference."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bevfusion import BEVFusionDet, BEVFusionSeg
from bevfusion.sample_io import load_samples


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _compare(actual, expected, task, atol, rtol):
    names = ("probabilities",) if task == "seg" else ("boxes_3d", "scores", "labels")
    actual = (actual,) if task == "seg" else tuple(actual[name] for name in names)
    expected = (expected,) if task == "seg" else tuple(expected[name] for name in names)
    metrics = []
    for name, value, reference in zip(names, actual, expected):
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise AssertionError(f"Output shape/dtype changed: {name}")
        if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
            raise AssertionError(f"Nonfinite output: {name}")
        torch.testing.assert_close(value, reference, atol=atol, rtol=rtol)
        difference = (value.double() - reference.double()).abs()
        record = {"output": name, "shape": list(value.shape),
                  "max_abs": float(difference.max()) if difference.numel() else 0.0,
                  "mean_abs": float(difference.mean()) if difference.numel() else 0.0}
        if task == "seg":
            record["mask_disagreements_at_0.5"] = int(((value >= .5) != (reference >= .5)).sum())
        metrics.append(record)
    return metrics


def run(args):
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = (args.output_dir or ROOT / "outputs/hat_plugin_verification" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite output directory: {output}")
    sample = None if args.checkpoint_only else (
        torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else
        load_samples(args.samples, calibration=args.calibration))
    output.mkdir(parents=True)
    report = {"status": "running", "torch": torch.__version__, "device": args.device,
              "comparison_backend": "hat", "new_backend": "hat_plugin", "cases": [],
              "comparison_tolerance": {"atol": args.atol, "rtol": args.rtol},
              "checkpoint_only": args.checkpoint_only, "floating_inference_verified": False,
              "qat_prepared": False, "hbir_exported": False, "hbm_compiled": False,
              "quantized_accuracy_verified": False}
    started = time.perf_counter()
    try:
        report["plugin"] = (None if args.checkpoint_only else
                            importlib.metadata.version("horizon-plugin-pytorch"))
        for task in args.tasks:
            path = args.det_checkpoint if task == "det" else args.seg_checkpoint
            fingerprint = _hash(path)
            factory = BEVFusionDet if task == "det" else BEVFusionSeg
            model = factory.from_pretrained(path, device=args.device, backend="reference",
                lidar_backend="reference", pool_backend="hat_plugin", pool_num_points=args.pool_num_points,
                precompute=True, camera_batch_size=1)
            before = _state_hash(model)
            if args.checkpoint_only:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                state = payload.get("state_dict", payload)
                state = {name.removeprefix("module."): value for name, value in state.items()}
                if set(state) != set(model.state_dict()) or not all(
                        torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items()):
                    raise AssertionError("Original checkpoint tensors differ after strict loading")
                unchanged = _hash(path) == fingerprint
                if not unchanged:
                    raise AssertionError("Original checkpoint file changed")
                record = {"task": task, "case": "strict_checkpoint_load", "strict_load": model.load_info,
                          "state_dict_tensors": len(state), "all_weights_equal": True,
                          "plugin_required": False, "checkpoint_unchanged": unchanged}
                report["cases"].append(record)
                report.setdefault("checkpoints", {})[task] = {"sha256": fingerprint, "unchanged": unchanged}
                print(json.dumps(record, ensure_ascii=False), flush=True)
                del model, payload, state
                gc.collect()
                continue
            transform = model.encoders["camera"]["vtransform"]
            values = (sample["images"], sample["points"], sample["camera_intrinsics"],
                      sample["camera2lidar"], sample.get("img_aug_matrix"))
            for case in ("provided", "changed_images"):
                current = list(values)
                if case == "changed_images":
                    images = values[0]
                    current[0] = ((images.float() * .7).round().to(torch.uint8) if images.dtype == torch.uint8
                                  else images * .7)
                with torch.inference_mode():
                    transform.pool_backend = "hat"
                    model.clear_bev_cache()
                    expected = model(*current)
                    transform.pool_backend = "hat_plugin"
                    model.clear_bev_cache()
                    actual = model(*current)
                    metrics = _compare(actual, expected, task, args.atol, args.rtol)
                statistics = model.bev_cache_info()
                plan = transform._bev_cache[3]
                plan.set_qconfig()
                coordinate_dtype = str(plan._runtime.feature_quant.qconfig.activation().dtype)
                if coordinate_dtype != "qint16":
                    raise AssertionError(f"Coordinate quantization must be qint16, got {coordinate_dtype}")
                state_unchanged = _state_hash(model) == before
                if not state_unchanged:
                    raise AssertionError("Inference/geometry/qconfig changed parent checkpoint state")
                record = {"task": task, "case": case, "checkpoint": str(path.resolve()),
                          "strict_load": model.load_info, "state_dict_tensors": len(model.state_dict()),
                          "model_state_unchanged": state_unchanged, "outputs": metrics,
                          "pool": statistics, "coordinate_qconfig_dtype": coordinate_dtype}
                if (args.visualize or args.save_predictions) and case == "provided":
                    if args.visualize:
                        from tools.hat_plugin_visualization import render_comparison
                        record["visualizations"] = render_comparison(
                            task, expected, actual, sample, model.classes, output)
                    snapshot = output / f"{task}.predictions.pt"
                    to_cpu = lambda value: ({name: tensor.detach().cpu() for name, tensor in value.items()}
                                            if isinstance(value, dict) else value.detach().cpu())
                    torch.save({"hat": to_cpu(expected), "hat_plugin": to_cpu(actual),
                                "classes": list(model.classes)}, snapshot)
                    record["prediction_snapshot"] = str(snapshot)
                report["cases"].append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                del expected, actual, plan
            if _hash(path) != fingerprint:
                raise AssertionError("Original checkpoint file changed")
            report.setdefault("checkpoints", {})[task] = {"sha256": fingerprint, "unchanged": True}
            del model, transform
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["floating_inference_verified"] = not args.checkpoint_only
        report["status"] = "verified"
        return 0
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print("[report]", output / "report.json", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save same-frame CPU prediction tensors for plotting in a separate environment")
    parser.add_argument("--visualize", action="store_true",
                        help="Save same-frame segmentation/detection comparison figures and prediction tensors")
    parser.add_argument("--checkpoint-only", action="store_true",
                        help="Verify original weights and strict loading without importing the optional plugin")
    parser.add_argument("--tasks", nargs="+", choices=("seg", "det"), default=["seg", "det"])
    parser.add_argument("--seg-checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-seg.pth")
    parser.add_argument("--det-checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-det.pth")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--samples", type=Path, default=ROOT / "data/samples")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pool-num-points", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.threads < 1 or not 1 <= args.pool_num_points <= 1024:
        parser.error("threads must be positive and pool-num-points must be in [1,1024]")
    if not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        parser.error("atol and rtol must be finite and nonnegative")
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, AssertionError, ImportError, KeyError) as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
