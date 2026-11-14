"""Export a checkpoint-preserving fixed-calibration camera branch using Horizon HBIR."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compiler_overlay(directory, wheels):
    """Expose local vendor wheels in this run only; never install into Conda."""
    if wheels is None:
        return None
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    compiler = list(wheels.glob(f"hbdk4_compiler-*-{tag}-{tag}-manylinux*.whl"))
    march = list(wheels.glob("hbdk4_march-*-abi3-manylinux*.whl"))
    if len(compiler) != 1 or len(march) != 1:
        raise ValueError(f"Select a directory containing exactly one {tag} HBDK compiler wheel and one march wheel")
    directory.mkdir()
    for wheel in compiler + march:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                target = (directory / name).resolve()
                target.relative_to(directory.resolve())
            archive.extractall(directory)
    paths = [directory] + sorted(directory.glob("*.data/purelib"))
    for path in paths:
        sys.path.insert(0, str(path))
    return {"path": str(directory), "python_paths": [str(path) for path in paths],
            "wheels": [{"path": str(path), "sha256": _hash(path)} for path in compiler + march]}


def build_camera(model, calibration, *, num_points=10, camera_batch_size=0):
    import torch
    from torch import nn
    from bevfusion.operators.pooling.hat_plugin import HATPluginLSSPool

    import copy

    class Unfold2x2(nn.Module):
        def forward(self, value):
            batch, channels, height, width = value.shape
            # nn.Unfold orders channels before the four kernel positions.
            # Stack inside each channel; concatenating spatial slices is wrong.
            patches = torch.stack((value[:, :, 0::2, 0::2], value[:, :, 0::2, 1::2],
                                   value[:, :, 1::2, 0::2], value[:, :, 1::2, 1::2]), dim=2)
            return patches.reshape(batch, channels * 4, (height // 2) * (width // 2))

    class FixedCamera(nn.Module):
        def __init__(self):
            super().__init__()
            camera = model.encoders["camera"]
            transform = camera["vtransform"]
            self.backbone, self.neck = copy.deepcopy(camera["backbone"]), camera["neck"]
            from bevfusion.camera import PatchMerging
            self.unfold_rewrites = 0
            for module in self.backbone.modules():
                if isinstance(module, PatchMerging):
                    module.sampler = Unfold2x2()
                    self.unfold_rewrites += 1
            self.depthnet, self.downsample = transform.depthnet, transform.downsample
            self.depth_conditioned = hasattr(transform, "dtransform")
            if self.depth_conditioned:
                self.dtransform = transform.dtransform
            self.pool = HATPluginLSSPool.from_transform(transform, calibration[1], calibration[0], calibration[2],
                torch.eye(4)[None], num_points=num_points)
            self.image_size, self.D, self.C = tuple(transform.image_size), transform.D, transform.C
            self.camera_batch_size = camera_batch_size
            self.eval()

        def forward(self, images, lidar_depth=None):
            batch, cameras = images.shape[:2]
            flat = images.reshape(batch * cameras, 3, *self.image_size)
            if self.camera_batch_size:
                encoded = [self.neck(self.backbone(part))[0] for part in flat.split(self.camera_batch_size)]
                feature = torch.cat(encoded, dim=0)
            else:
                feature = self.neck(self.backbone(flat))[0]
            if self.depth_conditioned:
                depth_features = self.dtransform(lidar_depth.reshape(batch * cameras, 1, *self.image_size))
                prediction = self.depthnet(torch.cat((depth_features, feature), dim=1))
            else:
                prediction = self.depthnet(feature)
            height, width = prediction.shape[-2:]
            prediction = prediction.reshape(batch, cameras, self.D + self.C, height, width)
            return self.downsample(self.pool(prediction[:, :, :self.D].softmax(2), prediction[:, :, self.D:]))

    return FixedCamera()


def _flat_arrays(value):
    if isinstance(value, dict):
        return [array for item in value.values() for array in _flat_arrays(item)]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in _flat_arrays(item)]
    import numpy as np
    return [np.asarray(value)]


def run(args):
    output = (args.output_dir or ROOT / "outputs/hat_plugin_camera_export" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    report = {"status": "running", "stage": "environment", "hbir_exported": False,
              "hbir_verified": False, "hbm_compiled": False, "quantized_accuracy_verified": False,
              "output_dir": str(output), "pool_backend": "hat_plugin", "pool_num_points": args.pool_num_points}
    started = time.perf_counter()
    checkpoint_hash = None
    try:
        if args.compiler_python_path:
            directory = args.compiler_python_path.resolve(strict=True)
            paths = [directory] + sorted(directory.glob("*.data/purelib"))
            for path in paths:
                sys.path.insert(0, str(path))
            report["compiler_overlay"] = {"path": str(directory), "python_paths": [str(path) for path in paths], "existing": True}
        else:
            report["compiler_overlay"] = _compiler_overlay(output / "compiler_python", args.compiler_wheels)
        import numpy as np
        import torch
        import hbdk4.compiler as hbc
        import horizon_plugin_pytorch as hp
        from horizon_plugin_pytorch.quantization.hbdk4 import export, check_hbdk4
        from bevfusion import BEVFusionSeg, BEVFusionDet
        from bevfusion.sample_io import load_samples
        from bevfusion.preprocessing import preprocess_images
        from bevfusion.depth import project_lidar_depth
        check_hbdk4()
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report.update(torch=torch.__version__, plugin=hp.__version__)
        report["stage"] = "load_checkpoint"
        checkpoint_hash = _hash(args.checkpoint)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload)
        task = payload.get("task") or ("det" if any("heads.object." in key for key in state) else "seg")
        factory = BEVFusionDet if task == "det" else BEVFusionSeg
        model = factory.from_pretrained(args.checkpoint, device="cpu", backend="reference", pool_backend="hat_plugin",
                                       pool_num_points=args.pool_num_points, precompute=True)
        report.update(task=task, strict_load=model.load_info)
        del payload, state
        sample = (torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else
                  load_samples(args.samples, calibration=args.calibration))
        images = sample["images"]
        if images.dtype == torch.uint8:
            images, augmentation = preprocess_images(images)
        else:
            if images.ndim == 4:
                images = images.unsqueeze(0)
            augmentation = sample.get("img_aug_matrix")
            if augmentation is None:
                raise ValueError("Normalized images require explicit img_aug_matrix")
        from bevfusion.base import _camera_matrix
        calibration = (_camera_matrix(sample["camera_intrinsics"], "camera_intrinsics", ((3, 3), (4, 4))),
                       _camera_matrix(sample["camera2lidar"], "camera2lidar", ((4, 4),)),
                       _camera_matrix(augmentation, "img_aug_matrix", ((4, 4),)))
        report["fixed_calibration"] = {name: value.tolist() for name, value in zip(
            ("camera_intrinsics", "camera2lidar", "img_aug_matrix"), calibration)}
        report["stage"] = "prepare_camera"
        before_keys = set(model.state_dict())
        with torch.inference_mode():
            branch = build_camera(model, calibration, num_points=args.pool_num_points, camera_batch_size=args.camera_batch_size)
            example = (images.float(),)
            if task == "det":
                points = sample["points"]
                lower, upper = points.new_tensor(model.point_cloud_range[:3]), points.new_tensor(model.point_cloud_range[3:])
                points = points[((points[:, :3] >= lower) & (points[:, :3] < upper)).all(1)]
                depth = project_lidar_depth(points, calibration[1], calibration[0], calibration[2], torch.eye(4)[None], branch.image_size)
                example += (depth,)
            expected = branch(*example).detach().numpy()
            source_camera = model._camera_features(images.float(), sample["points"], *calibration)
            np.testing.assert_allclose(expected, source_camera.numpy(), atol=args.atol, rtol=args.rtol)
            report["adapter_vs_source_max_abs"] = float(np.abs(expected - source_camera.numpy()).max())
            report["unfold_rewrites"] = branch.unfold_rewrites
            branch.pool.set_qconfig()
            if set(model.state_dict()) != before_keys:
                raise AssertionError("Building export adapter changed parent checkpoint keys")
            np.save(output / "camera_bev.pytorch.npy", expected, allow_pickle=False)
            np.savez_compressed(output / "sample_inputs.npz", **dict(zip(("images", "lidar_depth"), (v.numpy() for v in example))))
            report["inputs"] = [{"name": name, "shape": list(v.shape), "dtype": str(v.dtype)}
                                for name, v in zip(("images", "lidar_depth"), example)]
            report["outputs"] = [{"name": "camera_bev", "shape": list(expected.shape), "dtype": str(expected.dtype)}]
            report["pool"] = branch.pool.statistics()
            report["stage"] = "hbir_export"
            print("[export] full camera: backbone + neck + depthnet + hat_plugin pooling + downsample", flush=True)
            module = export(branch, example, input_names=[item["name"] for item in report["inputs"]],
                            output_names=["camera_bev"])
            path = output / "camera.bc"
            hbc.save(module, str(path))
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError("HBIR exporter did not create a nonempty camera.bc")
            report.update(hbir_exported=True, hbir_path=str(path), hbir_bytes=path.stat().st_size)
            report["stage"] = "hbir_verify"
            module = hbc.load(str(path))
            function = module.functions[0]
            cases = []
            for case in ("provided", "changed_images"):
                current = example if case == "provided" else (example[0] * .7,) + example[1:]
                reference = expected if case == "provided" else branch(*current).detach().numpy()
                actual = _flat_arrays(function(*(v.numpy() for v in current)))
                if len(actual) != 1:
                    raise AssertionError("HBIR camera output count changed")
                if actual[0].shape != reference.shape or not np.isfinite(actual[0]).all():
                    raise AssertionError("HBIR camera output shape or finiteness changed")
                np.testing.assert_allclose(actual[0], reference, atol=args.atol, rtol=args.rtol)
                difference = np.abs(actual[0].astype(np.float64) - reference.astype(np.float64))
                cases.append({"case": case, "max_abs": float(difference.max()), "mean_abs": float(difference.mean())})
                np.save(output / f"camera_bev.{case}.hbir.npy", actual[0], allow_pickle=False)
                print("[verify]", json.dumps(cases[-1]), flush=True)
            report.update(status="verified", stage="completed", hbir_verified=True,
                          comparison=cases[0], verification_cases=cases)
            print(json.dumps({key: report.get(key) for key in ("status", "hbir_path", "hbir_verified", "comparison")}), flush=True)
        return 0
    except Exception as exc:
        import traceback
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        if checkpoint_hash:
            report["checkpoint_sha256"] = checkpoint_hash
            report["checkpoint_unchanged"] = _hash(args.checkpoint) == checkpoint_hash
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[report]", output / "report.json", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-seg.pth")
    parser.add_argument("--compiler-python-path", type=Path, help="Reuse a previously staged HBDK Python directory")
    parser.add_argument("--compiler-wheels", type=Path, help="Expose local HBDK cp311/march wheels in a run-local overlay")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/samples")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--camera-batch-size", type=int, default=0, help="0 encodes all cameras together")
    parser.add_argument("--pool-num-points", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.camera_batch_size < 0 or not 1 <= args.pool_num_points <= 1024:
        parser.error("Invalid thread count, camera batch size, or pooling sample count")
    try:
        return run(args)
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
