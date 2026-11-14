"""Export four fixed-topology LiDAR stages; compilation is a separate operation."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    # Must precede torch: the plugin loader reads this setting at process start.
    from tools.horizon_runtime import ensure_plugin_loader
    ensure_plugin_loader()

import torch
from torch import nn

from bevfusion.operators.sparse_conv.compatible import _exporting, _validate_tensor
from bevfusion.operators.sparse_conv.hybrid_export import (
    HybridExportEncoder, _FixedConv3d, _FixedSubmanifold, _gather_rows,
)
from bevfusion.operators.sparse_conv.reference import SparseSequence


class HybridExportStage(nn.Module):
    """Share only one adapter subtree, with no reference to the entire adapter.

    Stages 0..3 respectively execute input+layer1, layer2, layer3, and
    layer4+output+dense. Topology remains the adapter's CPU-planned topology;
    inputs and intermediate outputs contain padded FP32 feature rows only.
    The final output uses the adapter's [1, C*Z, X, Y] BEV layout.
    """

    fixed_topology = True
    _execute = HybridExportEncoder._execute

    def __init__(self, adapter: HybridExportEncoder, stage_index: int):
        super().__init__()
        if not isinstance(adapter, HybridExportEncoder):
            raise TypeError("adapter must be HybridExportEncoder")
        if isinstance(stage_index, bool) or not isinstance(stage_index, int) or not 0 <= stage_index < 4:
            raise ValueError("stage_index must be an integer in [0,3]")
        self.stage_index = stage_index
        self.graph_name = f"lidar_stage{stage_index}"
        self.input_names = ["voxel_features" if stage_index == 0 else f"lidar_stage{stage_index - 1}_features"]
        self.output_names = ["lidar_bev" if stage_index == 3 else f"lidar_stage{stage_index}_features"]
        layer_name = f"encoder_layer{stage_index + 1}"
        if stage_index == 0:
            self.conv_input = adapter.conv_input
        # Keep source key paths, but register only this stage's layer.
        self.encoder_layers = SparseSequence(OrderedDict([(layer_name, getattr(adapter.encoder_layers, layer_name))]))
        if stage_index == 3:
            self.conv_out = adapter.conv_out
            self.final_shape = adapter.final_shape
            self.final_capacity = adapter.final_capacity
            self.out_channels = adapter.out_channels
            self.page_size = adapter.page_size
            self.register_buffer("dense_lookup", adapter.dense_lookup, persistent=False)
        convolutions = [module for module in self.modules() if isinstance(module, (_FixedSubmanifold, _FixedConv3d))]
        first, last = convolutions[0], convolutions[-1]
        self.input_capacity = first.input_capacity
        self.in_channels = int(first.weight.shape[-2])
        self.input_shape = (self.input_capacity, self.in_channels)
        self.output_shape = ((1, self.out_channels * self.final_shape[2], *self.final_shape[:2])
                             if stage_index == 3 else (last.output_capacity, int(last.weight.shape[-1])))
        self.eval()

    def final_dense(self, features):
        """Use the original adapter's sentinel Gather and XYZ-to-BEV layout."""
        dense = _gather_rows(features, self.dense_lookup, self.final_capacity, self.page_size)
        sx, sy, sz = self.final_shape
        return dense.reshape(sx, sy, sz, self.out_channels).permute(3, 2, 0, 1).reshape(
            1, self.out_channels * sz, sx, sy)

    def forward(self, features):
        if not _exporting():
            if self.training:
                raise RuntimeError("HybridExportStage is inference-only; call eval()")
            _validate_tensor(features, self.input_shape, torch.float32, self.input_names[0])
            weight = next(self.parameters())
            if features.device != weight.device or weight.dtype != torch.float32:
                raise ValueError("Features, mapping buffers and FP32 weights must share device")
        x, mask = features, None
        if self.stage_index == 0:
            x, mask = self._execute(self.conv_input, x, mask)
        x, mask = self._execute(self.encoder_layers, x, mask)
        if self.stage_index == 3:
            x, mask = self._execute(self.conv_out, x, mask)
            return self.final_dense(x)
        return x


def build_stages(adapter):
    """Return independent module trees that alias disjoint adapter parameters."""
    return tuple(HybridExportStage(adapter, index) for index in range(4))


def _adapter_outputs(adapter, features):
    """Capture boundaries during one unmodified adapter forward, for eager checks."""
    expected, handles = {}, []
    try:
        for index in range(3):
            layer = getattr(adapter.encoder_layers, f"encoder_layer{index + 1}")
            downsample = layer[-1]
            mask = downsample[0].output_mask

            def capture(module, inputs, output, index=index, mask=mask):
                # _execute applies the sparse mask after this ReLU returns.
                expected[index] = (output * mask).detach().cpu()

            handles.append(downsample[-1].register_forward_hook(capture))
        expected[3] = adapter(features).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return tuple(expected[index] for index in range(4))


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _state_digest(module):
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def _comparison(actual, expected, atol, rtol):
    equal_shape = actual.shape == expected.shape
    difference = (actual.double() - expected.double()).abs() if equal_shape else None
    return {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
            "max_abs": float(difference.max()) if equal_shape and difference.numel() else 0. if equal_shape else None,
            "mean_abs": float(difference.mean()) if equal_shape and difference.numel() else 0. if equal_shape else None,
            "passed": bool(equal_shape and torch.isfinite(actual).all() and torch.isfinite(expected).all()
                           and torch.allclose(actual, expected, atol=atol, rtol=rtol))}


def _spec(name, value):
    return {"name": name, "shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}


def _save(path, value):
    text = json.dumps(value, indent=2, default=str)
    path.write_text(text, encoding="utf-8")
    # The branch orchestration also consumes the conventional report filename.
    if path.name == "manifest.json":
        path.with_name("report.json").write_text(text, encoding="utf-8")


def _name_graph(module, name):
    functions = list(module.functions)
    if len(functions) != 1:
        raise ValueError(f"Expected exactly one exported function, got {len(functions)}")
    graph = functions[0]
    if graph.name != name:
        from hbdk4.compiler.ir import StringAttr
        graph.opview.sym_name = StringAttr.get(name, context=graph.opview.context)
    if graph.name != name:
        raise ValueError(f"Failed to name exported graph {name}")
    return graph


def run(args):
    from bevfusion import BEVFusionSeg
    from bevfusion.deployment_inputs import _prepare_sample
    from bevfusion.operators.sparse_conv.plan import plan_sparse_encoder
    from bevfusion.preprocessing import voxelize
    from bevfusion.sample_io import load_samples
    from horizon_plugin_pytorch.quantization.hbdk4 import export
    import hbdk4.compiler as hbc

    torch.set_num_threads(args.threads)
    torch.manual_seed(831)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.verify_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested eager verification device")
    output = (args.output_dir or ROOT / "outputs/hybrid_stages" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest = {"status": "running", "stage": "load", "task": "seg", "runtime_python": sys.executable,
                "checkpoint": str(args.checkpoint.resolve()), "fixed_lidar_topology": True,
                "dynamic_pointcloud_supported": False, "topology_device": "cpu",
                "voxelization": "cpu_outside_graph", "eager_verify_device": args.verify_device,
                "export_device": "cpu", "precision": "float32_source", "hbir_replayed": False,
                "hbo_compiled": False, "hbm_compiled": False, "board_verified": False,
                "atol": args.atol, "rtol": args.rtol, "exports": {}, "stage_pytorch_comparisons": {}}
    started = time.perf_counter()
    checkpoint_hash = None
    backbone = None
    before = None
    try:
        checkpoint_hash = _digest(args.checkpoint)
        manifest["checkpoint_sha256"] = checkpoint_hash
        model = BEVFusionSeg.from_pretrained(args.checkpoint, backend="reference", lidar_backend="hybrid",
                                             pool_backend="reference", device="cpu", camera_batch_size=0)
        manifest["strict_load"] = model.load_info
        backbone = model.encoders["lidar"]["backbone"]
        before = _state_digest(backbone)
        manifest["source_state_before"] = before
        sample = (torch.load(args.inputs, map_location="cpu", weights_only=True) if args.inputs else
                  load_samples(args.samples, calibration=args.calibration))
        _, points, _ = _prepare_sample(model, sample, None)
        # Camera/fusion weights are neither moved to CUDA nor exported.
        with torch.inference_mode():
            features, coordinates = voxelize(points, max_voxels=model.max_voxels, voxel_size=model.voxel_size,
                                             point_cloud_range=model.point_cloud_range, backend="reference")
        del model, sample, points
        gc.collect()
        manifest["stage"] = "cpu_topology"
        _save(manifest_path, manifest)
        with torch.inference_mode():
            plan = plan_sparse_encoder(backbone, coordinates, input_capacity=args.input_capacity)
            adapter = HybridExportEncoder(backbone, plan, coordinates, tile_size=args.tile_size,
                                          page_size=args.page_size)
            adapter.validate_topology(coordinates)
            stages = build_stages(adapter)
            padded = features.new_zeros((plan.input_capacity, features.shape[1]))
            padded[:len(features)] = features
            torch.save({"coords": coordinates, "voxel_features": padded}, output / "fixed_inputs.pt")
            manifest["topology"] = {**adapter.stats,
                "coordinates_sha256": hashlib.sha256(coordinates.numpy().tobytes()).hexdigest(),
                "sparse_shape": list(adapter.sparse_shape), "final_shape": list(adapter.final_shape),
                "active_counts": {name: item.output_count for name, item in plan.plans.items()}}
            manifest["stage"] = "eager_verify"
            _save(manifest_path, manifest)
            print("[eager] adapter and four stages on", args.verify_device, flush=True)
            adapter.to(args.verify_device)
            # Wrappers share adapter children, but their final buffer needs moving too.
            for stage in stages:
                stage.to(args.verify_device)
            current = padded.to(args.verify_device)
            expected = _adapter_outputs(adapter, current)
            examples, outputs = [], []
            for index, stage in enumerate(stages):
                examples.append(current.detach().cpu())
                current = stage(current)
                actual = current.detach().cpu()
                outputs.append(actual)
                record = _comparison(actual, expected[index], args.atol, args.rtol)
                manifest["stage_pytorch_comparisons"][stage.graph_name] = record
                if not record["passed"]:
                    raise AssertionError(f"{stage.graph_name} differs from the unsplit adapter")
            manifest["adapter_final_comparison"] = manifest["stage_pytorch_comparisons"]["lidar_stage3"]
            del current, expected
            adapter.cpu()
            for stage in stages:
                stage.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for index, stage in enumerate(stages):
                name = stage.graph_name
                manifest["stage"] = "export_" + name
                _save(manifest_path, manifest)
                print("[export]", name, flush=True)
                t0 = time.perf_counter()
                exported = export(stage, (examples[index],), input_names=stage.input_names,
                                  output_names=stage.output_names)
                _name_graph(exported, name)
                path = output / (name + ".bc")
                hbc.save(exported, str(path))
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"Missing exported HBIR: {path}")
                manifest["exports"][name] = {"graph_name": name, "source_hbir": str(path), "hbir": str(path),
                    "sha256": _digest(path), "bytes": path.stat().st_size, "export_seconds": time.perf_counter() - t0,
                    "inputs": [_spec(stage.input_names[0], examples[index])],
                    "outputs": [_spec(stage.output_names[0], outputs[index])], "hbir_replayed": False,
                    "source_parameter_names": list(dict(stage.named_parameters()))}
                del exported
                gc.collect()
                _save(manifest_path, manifest)
        manifest["stage_order"] = [stage.graph_name for stage in stages]
        manifest["contracts"] = [{"producer": stages[index].graph_name, "consumer": stages[index + 1].graph_name,
                                   **_spec(stages[index].output_names[0], outputs[index])}
                                  for index in range(3)]
        manifest.update(status="exported", stage="completed")
        return 0
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if backbone is not None and before is not None:
            manifest["source_state_after"] = _state_digest(backbone)
            manifest["source_state_unchanged"] = manifest["source_state_after"] == before
            if not manifest["source_state_unchanged"]:
                manifest.update(status="failed", error="Export modified source LiDAR state")
        if checkpoint_hash is not None:
            manifest["checkpoint_unchanged"] = _digest(args.checkpoint) == checkpoint_hash
            if not manifest["checkpoint_unchanged"]:
                manifest.update(status="failed", error="Checkpoint changed during export")
        manifest["elapsed_seconds"] = time.perf_counter() - started
        _save(manifest_path, manifest)
        print("[manifest]", manifest_path, flush=True)
        if manifest.get("source_state_unchanged") is False or manifest.get("checkpoint_unchanged") is False:
            raise AssertionError(manifest["error"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/checkpoints/bevfusion-seg.pth")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--samples", type=Path, default=ROOT / "data/samples")
    source.add_argument("--inputs", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output-dir", type=Path, help="New directory; existing directories are refused")
    parser.add_argument("--verify-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=32768)
    parser.add_argument("--input-capacity", type=int, help="Optional padded input row capacity")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if (args.threads < 1 or args.tile_size < 1 or not 1 <= args.page_size <= 65535 or
            (args.input_capacity is not None and not 1 <= args.input_capacity < 131072) or
            not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol))):
        parser.error("Invalid threads, tile size, page size, input capacity, or tolerances")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
