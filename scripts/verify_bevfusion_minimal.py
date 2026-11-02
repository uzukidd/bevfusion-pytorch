"""Compare the standalone wrapper with the legacy reference assembly.

中文：用真实张量比较新接口与旧计算流程，可选完整模型打包重载验证。
This checks this project's reference backend, not the upstream compiled backend.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bevfusion import BEVFusion, preprocess_images, voxelize
from bevfusion_native_lidar import hard_voxel_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roundtrip", action="store_true", help="Also save/reload the complete model and compare outputs")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sample = torch.load(args.inputs, map_location="cpu", weights_only=True)
    calibration = {key: sample[key] for key in ("camera_intrinsics", "camera2lidar", "img_aug_matrix") if key in sample}
    model = BEVFusion.from_pretrained(args.checkpoint, head_checkpoint=args.head_checkpoint,
                                     device=args.device, **calibration)
    start = time.perf_counter()
    with torch.inference_mode():
        print("[check] standalone tensor forward", flush=True)
        actual = model(sample["images"], sample["points"])
        images = sample["images"].to(args.device)
        if images.dtype == torch.uint8:
            images, augmentation = preprocess_images(images)
        else:
            images = images.float()
            if images.ndim == 4:
                images = images[None]
            augmentation = model._img_aug_matrix
            if augmentation is None:
                augmentation = torch.eye(4, device=args.device).repeat(1, 6, 1, 1)
        points = sample["points"].float()
        if points.shape[1] == 4:
            points = torch.cat((points, points.new_zeros((len(points), 1))), dim=1)
        legacy_features, legacy_coords = hard_voxel_mean(points.numpy())
        features, coords = voxelize(points.to(args.device))
        expected_features = torch.as_tensor(legacy_features, device=args.device, dtype=torch.float32)
        expected_coords = torch.as_tensor(legacy_coords, device=args.device, dtype=torch.long)
        torch.testing.assert_close(coords, expected_coords, rtol=0, atol=0)
        torch.testing.assert_close(features, expected_features, rtol=2e-6, atol=2e-6)
        print("[check] legacy assembly with NumPy voxelization", flush=True)
        lidar_bev = model.encoders["lidar"]["backbone"](expected_features, expected_coords, verbose=False)
        camera_bev = model.encoders["camera"](images, model._camera2lidar,
                                               model._camera_intrinsics, augmentation,
                                               torch.eye(4, device=args.device)[None])
        fused = model.fuser(torch.cat((camera_bev, lidar_bev), dim=1))
        decoder = model.decoder["neck"](model.decoder["backbone"](fused))[0]
        if model.quantize_decoder:
            decoder = decoder.half().float()
        expected = model.heads["map"](decoder)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        if not torch.isfinite(actual).all() or actual.min() < 0 or actual.max() > 1:
            raise AssertionError("Probabilities must be finite and in [0,1]")
        report = {"status": "passed", "device": args.device, "torch": str(torch.__version__),
                  "classes": model.classes, "shape": list(actual.shape), "loaded": model.load_info,
                  "point_count": len(points), "voxel_count": len(coords),
                  "voxel_coordinate_equal": True,
                  "voxel_feature_max_abs": float((features - expected_features).abs().max()) if len(features) else 0.0,
                  "prediction_max_abs": float((actual - expected).abs().max()),
                  "prediction_mean_abs": float((actual - expected).abs().mean()),
                  "roundtrip": False,
                  "scope": "legacy assembly parity within the same PyTorch reference backend"}
        torch.save({"probabilities": actual.cpu(), "classes": model.classes}, out / "prediction.pt")
        if args.roundtrip:
            print("[check] save complete model and reload", flush=True)
            packed_path = out / "model.pth"
            model.save_pretrained(packed_path)
            baseline = actual.cpu()
            del model, actual, expected, images, features, coords, lidar_bev, camera_bev, fused, decoder
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            restored = BEVFusion.from_pretrained(packed_path, device=args.device)
            repeated = restored(sample["images"], sample["points"]).cpu()
            torch.testing.assert_close(baseline, repeated, rtol=1e-4, atol=1e-4)
            report.update(roundtrip=True, roundtrip_max_abs=float((baseline - repeated).abs().max()))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            report["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    report["elapsed_seconds"] = time.perf_counter() - start
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
