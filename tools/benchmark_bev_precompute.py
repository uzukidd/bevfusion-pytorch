"""Benchmark BEV precomputation with prepared, device-resident sensor inputs."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from bevfusion import BEVFusionDet, BEVFusionSeg
from bevfusion.preprocessing import preprocess_images
from bevfusion.sample_io import load_samples


ATOL = RTOL = 2e-4


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_forward(forward, device):
    _sync(device)
    started = time.perf_counter()
    output = forward()
    _sync(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return output, elapsed_ms


def _cpu_output(output):
    if isinstance(output, torch.Tensor):
        return output.detach().cpu()
    if isinstance(output, dict):
        return {key: _cpu_output(value) for key, value in output.items()}
    raise ValueError(f"Unsupported model output type: {type(output).__name__}")


def _tensor_summary(tensor):
    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel()}
    if tensor.numel():
        values = tensor.double()
        if not torch.isfinite(values).all():
            raise ValueError("Model output contains NaN or infinity")
        result.update(min=values.min().item(), max=values.max().item(), mean=values.mean().item())
    return result


def _output_summary(output, kind):
    if isinstance(output, torch.Tensor):
        result = _tensor_summary(output)
        if kind == "seg":
            result["mask_threshold"] = 0.5
            result["mask_positive_count"] = int((output >= 0.5).sum())
        return result
    result = {key: _tensor_summary(value) for key, value in output.items()}
    labels, counts = output["labels"].unique(return_counts=True)
    result["box_count"] = len(output["labels"])
    result["label_counts"] = {str(int(label)): int(count) for label, count in zip(labels, counts)}
    return result


def _numeric_difference(reference, candidate, name):
    if reference.shape != candidate.shape:
        raise ValueError(f"{name}: precompute changed output shape: "
                         f"{tuple(reference.shape)} -> {tuple(candidate.shape)}")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError(f"{name}: output contains NaN or infinity")
    difference = (candidate.double() - reference.double()).abs()
    close = torch.allclose(candidate, reference, atol=ATOL, rtol=RTOL)
    result = {"maxabs": difference.max().item() if difference.numel() else 0.0,
              "meanabs": difference.mean().item() if difference.numel() else 0.0,
              "allclose": close}
    if not close:
        raise ValueError(f"{name}: precompute numerical validation failed with "
                         f"atol={ATOL}, rtol={RTOL}, maxabs={result['maxabs']:.9g}, "
                         f"meanabs={result['meanabs']:.9g}")
    return result


def _compare(reference, candidate, kind, name):
    result = {"atol": ATOL, "rtol": RTOL}
    if kind == "det":
        labels_equal = torch.equal(reference["labels"], candidate["labels"])
        if not labels_equal:
            raise ValueError(f"{name}: precompute changed detection labels or their order "
                             f"(counts {len(reference['labels'])} -> {len(candidate['labels'])})")
        result.update(labels_equal=True, box_count=len(reference["labels"]))
        for key in ("boxes_3d", "scores"):
            result[key] = _numeric_difference(reference[key], candidate[key], f"{name}.{key}")
    else:
        result.update(_numeric_difference(reference, candidate, name))
        if kind == "seg":
            result["mask_threshold"] = 0.5
            result["mask_changes"] = int(((reference >= 0.5) != (candidate >= 0.5)).sum())
    return result


def _benchmark_mode(model, forward, enabled, kind, device, warmup, repeat):
    model.encoders["camera"]["vtransform"].precompute_enabled = enabled
    model.clear_bev_cache()
    warmup_summaries = []
    for _ in range(warmup):
        output = forward()
        cpu = _cpu_output(output)
        del output
        warmup_summaries.append(_output_summary(cpu, kind))
        del cpu
    _sync(device)
    model.clear_bev_cache()
    before = dict(model.bev_cache_info())
    output, cold_ms = _timed_forward(forward, device)
    cpu = _cpu_output(output)
    del output
    cold_summary = _output_summary(cpu, kind)
    del cpu
    after_cold = dict(model.bev_cache_info())
    elapsed = []
    summaries = []
    for _ in range(repeat):
        output, elapsed_ms = _timed_forward(forward, device)
        cpu = _cpu_output(output)
        del output
        elapsed.append(elapsed_ms)
        summaries.append(_output_summary(cpu, kind))
        # Only the last CPU output survives for comparison; JSON stores summaries.
        last_output = cpu
        del cpu
    mean_ms = statistics.mean(elapsed)
    report = {
        "precompute": enabled,
        "mean_ms": mean_ms, "median_ms": statistics.median(elapsed),
        "min_ms": min(elapsed), "max_ms": max(elapsed), "fps": 1000.0 / mean_ms,
        "repeat_ms": elapsed, "cache_cold_ms": cold_ms,
        "cache_info_before_cold": before, "cache_info_after_cold": after_cold,
        "cache_info": dict(model.bev_cache_info()),
        "output_summaries": {"warmup": warmup_summaries, "cache_cold": cold_summary,
                             "repeat": summaries},
    }
    return report, last_output


def _benchmark_pair(model, forward, kind, name, args, device):
    off_report, reference = _benchmark_mode(
        model, forward, False, kind, device, args.warmup, args.repeat)
    on_report, candidate = _benchmark_mode(
        model, forward, True, kind, device, args.warmup, args.repeat)
    comparison = _compare(reference, candidate, kind, name)
    return {"precompute_off": off_report, "precompute_on": on_report,
            "mean_speedup": off_report["mean_ms"] / on_report["mean_ms"],
            "comparison": comparison}


@torch.inference_mode()
def _benchmark_task(task, args, device, sample, images, points, augmentation):
    checkpoint = args.checkpoint_dir / f"bevfusion-{task}.pth"
    model_class = BEVFusionSeg if task == "seg" else BEVFusionDet
    started = time.perf_counter()
    model = model_class.from_pretrained(
        checkpoint, device=device, backend=args.backend,
        camera_batch_size=args.camera_batch_size, precompute=True)
    model.eval()
    intrinsic = sample.get("camera_intrinsics", model._camera_intrinsics)
    extrinsic = sample.get("camera2lidar", model._camera2lidar)
    if intrinsic is None or extrinsic is None:
        raise ValueError("Camera calibration is required in samples/calibration.json "
                         "or bound in the pretrained checkpoint")
    model.set_calibration(intrinsic, extrinsic, augmentation)
    _sync(device)
    model_setup_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "checkpoint": str(checkpoint.resolve()), "classes": list(model.classes),
        "excluded_model_load_and_calibration_ms": model_setup_ms,
        "full_forward": _benchmark_pair(
            model, lambda: model(images, points), task, f"{task}.full_forward", args, device),
    }
    camera = model.encoders["camera"]
    features = camera.encode_images(images)
    intrinsic, extrinsic, augmentation = (
        model._camera_intrinsics, model._camera2lidar, model._img_aug_matrix)
    identity = torch.eye(4, dtype=torch.float32, device=device)[None]
    vt = camera["vtransform"]
    if task == "det":
        lower = points.new_tensor(model.point_cloud_range[:3])
        upper = points.new_tensor(model.point_cloud_range[3:])
        selected = points[((points[:, :3] >= lower) & (points[:, :3] < upper)).all(1)]
        def transform():
            return vt(features, selected, extrinsic, intrinsic, augmentation, identity)
        result["vtransform_point_count"] = len(selected)
    else:
        def transform():
            return vt(features, extrinsic, intrinsic, augmentation, identity)
    _sync(device)
    result["vtransform"] = _benchmark_pair(
        model, transform, "vtransform", f"{task}.vtransform", args, device)
    model.clear_bev_cache()
    return result


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("seg", "det", "both"), default="both")
    parser.add_argument("--samples", type=Path, default=Path("data/samples"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("auto", "reference", "cuda"), default="auto")
    parser.add_argument("--camera-batch-size", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.camera_batch_size < 0:
        parser.error("--camera-batch-size must be >= 0")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1")
    if args.output is not None:
        if args.output.suffix.lower() != ".json":
            parser.error("--output must be a new .json file")
        if args.output.exists():
            parser.error(f"--output must be a new file: {args.output}")
    return args


def main(argv=None):
    args = _parse_args(argv)
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device cpu for a CPU benchmark")
    started = time.perf_counter()
    sample = load_samples(args.samples)
    io_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    images, augmentation = preprocess_images(sample["images"])
    preprocessing_ms = (time.perf_counter() - started) * 1000.0
    _sync(device)
    started = time.perf_counter()
    images = images.to(device)
    points = sample["points"].to(device)
    augmentation = augmentation.to(device)
    _sync(device)
    transfer_ms = (time.perf_counter() - started) * 1000.0
    report = {
        "settings": {"task": args.task, "samples": str(args.samples.resolve()),
                     "device": str(device), "backend": args.backend,
                     "camera_batch_size": args.camera_batch_size,
                     "warmup": args.warmup, "repeat": args.repeat,
                     "threads": 8, "tf32": False, "eval": True, "inference_mode": True,
                     "atol": ATOL, "rtol": RTOL},
        "timing": {
            "method": "CUDA synchronize (when CUDA), perf_counter, forward, synchronize",
            "units": "ms", "fps": "1000 / mean_ms; one six-camera/LiDAR frame per forward",
            "excluded": ["sample and checkpoint I/O", "image resize/crop/normalization",
                         "host-to-device input/calibration transfer", "model loading and calibration binding",
                         "output-to-CPU transfer, summaries, numerical validation",
                         "camera.encode_images and detection point-range filtering for standalone vtransform"],
            "included": "Full forward includes normal model input validation, point voxelization, "
                        "camera/LiDAR encoders, fusion and task head; standalone vtransform includes "
                        "depth prediction/projection, pooling and downsample.",
            "cache_cold_ms": "One forward after warmup and clear_bev_cache; repeated forwards follow "
                             "this call with the cache retained. Off mode has precompute disabled.",
        },
        "excluded_setup_ms": {"sample_io": io_ms, "preprocess_images": preprocessing_ms,
                              "input_device_transfer": transfer_ms},
        "input": {"images_shape": list(images.shape), "points_shape": list(points.shape),
                  "camera_names": sample["camera_names"], "image_paths": sample["image_paths"],
                  "lidar_path": sample["lidar_path"]},
        "tasks": {},
    }
    tasks = ("seg", "det") if args.task == "both" else (args.task,)
    for task in tasks:
        try:
            report["tasks"][task] = _benchmark_task(
                task, args, device, sample, images, points, augmentation)
        finally:
            # Function-local model/features are released before the next task loads.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text, flush=True)
    return report


if __name__ == "__main__":
    main()
