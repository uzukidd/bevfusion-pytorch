"""Rebuild one frame's deployment inputs from a bundle manifest and raw sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bevfusion import BEVFusionDet, BEVFusionSeg
from bevfusion.deployment_inputs import DeploymentProfile, build_deployment_frame
from bevfusion.sample_io import load_samples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path, help="Raw/normalized sensor tensor dictionary")
    source.add_argument("--samples", type=Path, help="Six ordered PNGs, LiDAR BIN and calibration")
    parser.add_argument("--calibration", type=Path, help="Override calibration JSON for --samples")
    parser.add_argument("--checkpoint", type=Path, help="Override the manifest source checkpoint location")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.calibration is not None and args.samples is None:
        parser.error("--calibration requires --samples")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir must be new; existing files are never overwritten")
    return parser, args


def _checkpoint(manifest, manifest_path, override):
    if override is not None:
        return override.expanduser().resolve(strict=True)
    value = manifest.get("checkpoint")
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest requires source checkpoint or --checkpoint override")
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [manifest_path.parent / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("Manifest checkpoint cannot be found; supply --checkpoint")


def verify_checkpoint(path, expected):
    """Allow relocation, never replacement, of a bundle's source weights."""
    if expected is None:
        return {"verified": False, "reason": "manifest_missing_checkpoint_sha256"}
    if (not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected)):
        raise ValueError("Manifest checkpoint_sha256 must be a 64-character hexadecimal digest")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise ValueError("Source checkpoint SHA256 differs from the bundle; --checkpoint may change its path, not its weights")
    return {"verified": True, "sha256": actual}


def load_model(manifest, manifest_path, override=None):
    task = manifest.get("task")
    if task not in ("det", "seg"):
        raise ValueError("Manifest task must be det or seg")
    options = manifest.get("model_options", manifest.get("model_args", {}))
    if not isinstance(options, dict):
        raise ValueError("Manifest model_args must be a mapping")
    options = dict(options)
    options.update(backend="reference", lidar_backend="reference", pool_backend="reference", precompute=False)
    factory = BEVFusionDet if task == "det" else BEVFusionSeg
    path = _checkpoint(manifest, manifest_path, override)
    verification = verify_checkpoint(path, manifest.get("checkpoint_sha256"))
    model = factory.from_pretrained(path, device="cpu", **options)
    model._deployment_checkpoint_verification = verification
    return model


def validate_feed(feed, contract):
    if not isinstance(contract, list) or not contract:
        raise ValueError("Manifest inputs must be a nonempty list of name/shape/dtype records")
    names = [item.get("name") if isinstance(item, dict) else None for item in contract]
    if any(not isinstance(name, str) or not name or "/" in name or "\\" in name or name in (".", "..")
           for name in names) or len(names) != len(set(names)):
        raise ValueError("Manifest input names must be unique safe filenames")
    if list(feed) != names:
        raise ValueError(f"Prepared input order/names differ from manifest: {list(feed)} != {names}")
    for item in contract:
        value = feed[item["name"]]
        dtype = str(item.get("dtype", "")).removeprefix("torch.").lower()
        dtype = {"float": "float32", "tensor(float)": "float32", "tensor(int32)": "int32"}.get(dtype, dtype)
        if list(value.shape) != item.get("shape") or value.dtype != np.dtype(dtype):
            raise ValueError(f"Prepared {item['name']} shape/dtype differs from manifest")
        if not np.isfinite(value).all():
            raise ValueError(f"Prepared {item['name']} contains nonfinite values")


def main(argv=None):
    parser, args = parse_args(argv)
    torch.set_num_threads(args.threads)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != "bevfusion-hbm-bundle-v1":
            raise ValueError("Manifest schema must be bevfusion-hbm-bundle-v1")
        fixed = manifest.get("fixed_calibration")
        if not isinstance(fixed, dict) or set(fixed) != {"camera_intrinsics", "camera2lidar", "img_aug_matrix"}:
            raise ValueError("Manifest fixed_calibration must contain exactly camera_intrinsics, camera2lidar, img_aug_matrix")
        if any(value is None for value in fixed.values()):
            raise ValueError("Manifest fixed_calibration matrices cannot be null")
        profile = DeploymentProfile.from_dict(manifest["profile"])
        model = load_model(manifest, args.manifest, args.checkpoint)
        sample = (load_samples(args.samples, calibration=args.calibration) if args.samples is not None else
                  torch.load(args.inputs, map_location="cpu", weights_only=True))
        backends = manifest["backends"]
        if not isinstance(backends, dict):
            raise ValueError("Manifest backends must be a mapping")
        frame = build_deployment_frame(model, sample, profile=profile,
            voxel_backend=backends["voxel"], spconv_backend=backends["spconv"],
            calibration=fixed)
        feed = frame.feed_np()
        validate_feed(feed, manifest["inputs"])
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(args.output_dir / "sample_inputs.npz", **feed)
    for name, value in feed.items():
        np.save(args.output_dir / f"{name}.npy", value, allow_pickle=False)
    result = {"source_manifest": str(args.manifest.resolve()), "task": manifest["task"],
              "checkpoint_verification": getattr(model, "_deployment_checkpoint_verification",
                                                   {"verified": False, "reason": "checkpoint_verification_unavailable"}),
              "inputs": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                          "file": f"{name}.npy"} for name, value in feed.items()],
              "statistics": frame.statistics}
    with (args.output_dir / "input_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
