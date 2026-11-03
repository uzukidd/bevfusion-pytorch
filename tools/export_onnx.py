"""Export complete BEVFusion weights + model to standard ONNX, on Windows/Linux.

中文：完整模型导出工具，不固定示例点云邻接关系；点数动态，六相机尺寸固定。
Example: python tools/export_onnx.py --checkpoint data/checkpoints/bevfusion-seg.pth
Normalized images [1,6,3,256,704], points [P,5], K [1,6,3,3],
camera2lidar and img_aug_matrix [1,6,4,4] -> sigmoid [1,C,200,200].
RGB PNG decoding/resize/normalization and 4->5-column conversion stay outside ONNX.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bevfusion import BEVFusion, preprocess_images
from tools.onnx_model import OnnxBEVFusion

INPUT_NAMES = ["images", "points", "camera_intrinsics", "camera2lidar", "img_aug_matrix"]


def example_inputs(model, sample, device):
    if sample is None:
        images = torch.zeros((1, 6, 3, 256, 704), device=device)
        generator = torch.Generator().manual_seed(42)
        points = torch.rand((32, 5), generator=generator).to(device)
        points[:, :2] = points[:, :2] * 30 - 15
        points[:, 2] = points[:, 2] * 4 - 3
        points[:, 4] = 0
        intrinsic = torch.tensor([[500., 0., 352.], [0., 500., 128.], [0., 0., 1.]], device=device).repeat(1, 6, 1, 1)
        extrinsic = torch.eye(4, device=device).repeat(1, 6, 1, 1)
        augmentation = extrinsic.clone()
    else:
        images, points = sample["images"].to(device), sample["points"].float().to(device)
        if points.shape[1] == 4:
            points = torch.cat((points, points.new_zeros((len(points), 1))), 1)
        intrinsic = sample.get("camera_intrinsics", model._camera_intrinsics)
        extrinsic = sample.get("camera2lidar", model._camera2lidar)
        augmentation = sample.get("img_aug_matrix", model._img_aug_matrix)
        if images.dtype == torch.uint8:
            images, augmentation = preprocess_images(images)
        else:
            images = images.float()
            if images.ndim == 4:
                images = images[None]
        if intrinsic is None or extrinsic is None or augmentation is None:
            raise ValueError("Verification input needs real K/camera2lidar and augmentation (auto for uint8)")
        intrinsic = torch.as_tensor(intrinsic, device=device, dtype=torch.float32)
        extrinsic = torch.as_tensor(extrinsic, device=device, dtype=torch.float32)
        augmentation = torch.as_tensor(augmentation, device=device, dtype=torch.float32)
        if intrinsic.ndim == 3:
            intrinsic = intrinsic[None]
        if extrinsic.ndim == 3:
            extrinsic = extrinsic[None]
        if augmentation.ndim == 3:
            augmentation = augmentation[None]
        intrinsic = intrinsic[..., :3, :3]
    return images, points, intrinsic, extrinsic, augmentation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-seg.pth")
    parser.add_argument("--head-checkpoint", type=Path, help="Optional four-class head matching the original base weight")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/onnx/bevfusion-seg.onnx")
    parser.add_argument("--inputs", type=Path, help="Optional real tensor sample .pt for eager/ONNX numerical verification")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="PyTorch export device; ONNX Runtime verification uses CPU")
    parser.add_argument("--verify", action="store_true", help="Run ONNX Runtime comparison on real/synthetic input plus another point count")
    parser.add_argument("--verify-only", action="store_true", help="Check an existing ONNX file without exporting or overwriting it")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute probability tolerance for ONNX Runtime comparison")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative probability tolerance for ONNX Runtime comparison")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("--threads must be positive; tolerances must be nonnegative")
    if args.verify_only:
        args.verify = True
    if args.output.suffix.lower() != ".onnx" or (args.output.exists() and not args.verify_only):
        parser.error("--output must be a new .onnx file (or use --verify-only)")
    if args.verify_only and not args.output.is_file():
        parser.error("--verify-only requires an existing --output ONNX file")
    report_path = args.output.with_suffix(".verification.json")
    if report_path.exists():
        parser.error("Choose a fresh output name: verification report already exists")
    try:
        import onnx
        if args.verify:
            import onnxruntime as ort
    except ImportError as exc:
        parser.error(f"Missing export dependency {exc.name}; install onnx and onnxruntime in the selected Conda environment")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    print("Python:", sys.executable, flush=True)
    print("[load]", args.checkpoint, flush=True)
    model = BEVFusion.from_pretrained(args.checkpoint, head_checkpoint=args.head_checkpoint, device=args.device)
    wrapper = OnnxBEVFusion(model).eval()
    sample = torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else None
    values = example_inputs(model, sample, args.device)
    report = {"status": "exporting", "checkpoint": str(args.checkpoint.resolve()),
              "python": sys.executable, "torch": str(torch.__version__), "onnx": onnx.__version__,
              "opset": 18, "classes": list(model.classes), "loaded": model.load_info,
              "inputs": dict(zip(INPUT_NAMES, [list(value.shape) for value in values])),
              "dynamic_axes": {"points": {"0": "num_points"}},
              "input_contract": "images=normalized FP32; points=XYZ,intensity,time_lag_seconds in reference LiDAR; K original image plus img_aug_matrix",
              "output": "FP32 sigmoid [1,C,200,200], spatial axes X,Y, range +/-50 m at 0.5m"}
    with torch.no_grad():
        print("[check] export adapter vs original PyTorch model", flush=True)
        expected = model(values[0], values[1], values[2], values[3], values[4])
        actual = wrapper(*values)
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
        report["adapter_max_abs"] = float((actual - expected).abs().max())
        if not args.verify_only:
            # Trace few points; all point-dependent dimensions stay dynamic.
            trace = list(example_inputs(model, None, args.device))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            print("[export] complete dynamic point graph ->", args.output, flush=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                torch.onnx.export(wrapper, tuple(trace), str(args.output), dynamo=False,
                                  opset_version=18, input_names=INPUT_NAMES, output_names=["probabilities"],
                                  dynamic_axes={"points": {0: "num_points"}},
                                  do_constant_folding=True, export_params=True,
                                  training=torch.onnx.TrainingMode.EVAL)
    print("[check] ONNX structure", flush=True)
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph, full_check=True)
    def check_domains(nodes):
        for node in nodes:
            if node.domain not in ("", "ai.onnx"):
                raise RuntimeError(f"Custom runtime operator found: {node.domain}::{node.op_type}")
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    check_domains(attr.g.node)
    check_domains(graph.graph.node)
    onnx.helper.set_model_props(graph, {
        "classes": json.dumps(list(model.classes)),
        "grid": "[batch,class,X,Y], 200x200, XY[-50,50), 0.5m",
        "images": "normalized RGB float32 [1,6,3,256,704]",
        "points": "dynamic float32 [P,5], XYZ/intensity/time_lag_seconds",
        "calibration": "dynamic values: original-image K, camera2lidar, image augmentation",
    })
    if not args.verify_only:
        onnx.save(graph, str(args.output))
    report.update(status="exported", file_bytes=args.output.stat().st_size,
                  node_count=len(graph.graph.node), initializer_count=len(graph.graph.initializer),
                  standard_onnx_operators_only=True, runtime_verified=False)
    if args.verify:
        print("[verify] loading ONNX Runtime CPU session", flush=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(str(args.output), sess_options=options, providers=["CPUExecutionProvider"])
        cases = []
        # A second scene-size input would be expensive. An empty point cloud and
        # a small changed cloud exercise dynamic sparse shape/empty branches.
        report.update(tolerances={"atol": args.atol, "rtol": args.rtol})
        for label, inputs in (("provided" if sample else "synthetic", values),
                              ("empty_points", (values[0], values[1][:0], *values[2:])),
                              ("changed_points", (values[0], values[1][:min(7, len(values[1]))] + values[1].new_tensor([0.23, -0.17, 0, 0, 0]), *values[2:]))):
            print("[verify]", label, "points=", inputs[1].shape[0], flush=True)
            with torch.no_grad():
                reference = model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]).cpu().numpy()
            feed = {name: value.detach().cpu().numpy() for name, value in zip(INPUT_NAMES, inputs)}
            output = session.run(None, feed)[0]
            import numpy as np
            case = {"name": label, "points": len(inputs[1]),
                    "max_abs": float(np.abs(output - reference).max()),
                    "mean_abs": float(np.abs(output - reference).mean()),
                    "mask_disagreements_at_0.5": int(np.count_nonzero((output >= 0.5) != (reference >= 0.5)))}
            cases.append(case)
            print("[verify] result:", json.dumps(case), flush=True)
            try:
                np.testing.assert_allclose(output, reference, atol=args.atol, rtol=args.rtol)
            except AssertionError:
                report.update(status="verification_failed", runtime_verified=False,
                              verification_cases=cases, elapsed_seconds=time.perf_counter() - started)
                with report_path.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2)
                raise
        report.update(runtime_verified=True, runtime=ort.__version__, verification_cases=cases)
    report["elapsed_seconds"] = time.perf_counter() - started
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
