"""Ordinary ONNX export using per-frame CPU plans and compatible LiDAR tensors."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

from bevfusion.deployment_inputs import DeploymentProfile


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _sensor_feed(values):
    from tools.export_onnx import INPUT_NAMES
    return {name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
            for name, value in zip(INPUT_NAMES, values)}


def _contracts(graph):
    import onnx
    result = []
    for value in graph.graph.input:
        tensor = value.type.tensor_type
        dtype = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.INT32: "int32"}.get(tensor.elem_type)
        if dtype is None:
            raise ValueError(f"Unexpected compatible ONNX input dtype: {value.name}")
        result.append({"name": value.name, "shape": [dim.dim_value if dim.HasField("dim_value") else None
            for dim in tensor.shape.dim], "dtype": dtype})
    return result


def _validate_graph(graph, wrapper, metadata, *, verify_only):
    from tools.export_onnx import assert_standard_operators
    import onnx
    onnx.checker.check_model(graph, full_check=True)
    assert_standard_operators(graph.graph)
    names = [value.name for value in graph.graph.input]
    if names != list(wrapper.input_names):
        raise ValueError("Compatible ONNX input names/order differ from requested capacity/pooling contract")
    for item in _contracts(graph):
        spec = wrapper.input_specs[item["name"]]
        if item["dtype"] != spec["dtype"] or len(item["shape"]) != len(spec["shape"]):
            raise ValueError(f"Compatible ONNX input contract differs: {item['name']}")
        for actual, expected in zip(item["shape"], spec["shape"]):
            if expected is not None and actual != expected:
                raise ValueError(f"Compatible ONNX capacity/shape differs: {item['name']}")
    if not verify_only:
        return
    existing = {item.key: item.value for item in graph.metadata_props}
    # All preprocessing-related metadata must agree before evaluating an existing
    # graph. A new sample's coordinate plans must never replace its geometry spec.
    for key, expected in metadata.items():
        if key in ("images", "points", "calibration", "python", "torch"):
            continue
        actual = existing.get(key)
        if key == "fixed_calibration":
            try:
                bound, requested = json.loads(actual), json.loads(expected)
                if set(bound) != set(requested):
                    raise ValueError("different calibration keys")
                for name in requested:
                    left, right = np.asarray(bound[name]), np.asarray(requested[name])
                    if left.shape != right.shape or not np.isfinite(left).all() or not np.allclose(left, right, atol=1e-6, rtol=0):
                        raise ValueError(f"different calibration {name}")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError("Existing compatible ONNX fixed calibration differs or is invalid") from exc
        elif actual != expected:
            raise ValueError(f"Existing compatible ONNX metadata differs: {key}")


def export_compatible_onnx(args, model, sample, values, task, pool_backend, num_points,
                           threshold, fixed_calibration, parser=None, started=None):
    """Export and optionally verify one ordinary compatible det/seg ONNX model.

    Unlike the HBM wrapper, detection retains dynamic filtered INT64-label
    outputs and camera depth projection stays in the graph using raw points.
    Only coordinate grouping and sparse topology discovery move to the CPU.
    """
    import onnx
    from tools.export_onnx import (INPUT_NAMES, OUTPUT_NAMES, BOX_COLUMNS, calibration_record,
                                   compare_outputs, operator_counts, validate_graph_contract)
    from tools.onnx_compatible import CompatibleOnnxBEVFusion

    started = time.perf_counter() if started is None else started
    profile = DeploymentProfile(args.point_capacity, args.voxel_capacity, args.sparse_capacity, args.page_size)
    report_path = args.report_path
    report = {"status": "preparing", "stage": "compatible_preprocessing", "task": task,
              "checkpoint": str(args.checkpoint.resolve()), "python": sys.executable,
              "torch": str(torch.__version__), "onnx": onnx.__version__, "opset": 18,
              "classes": list(model.classes), "loaded": getattr(model, "load_info", {}),
              "lidar_backend": "compatible", "pool_backend": pool_backend, "pool_num_points": num_points,
              "profile": profile.to_dict(), "calibration_mode": "fixed" if fixed_calibration else "dynamic",
              "pooling_precomputed": bool(fixed_calibration), "scope": "entire_model",
              "cpu_preprocessing_required": True, "runtime_verified": False, "bpu_compatible": False,
              "capacity_overflow": "raise", "outputs": OUTPUT_NAMES[task]}
    sample_path = args.output.with_suffix(".sample_inputs.npz")
    preprocessing_path = args.output.with_suffix(".preprocessing.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if Path(report_path).exists():
        raise ValueError("Refusing to overwrite an existing verification report")
    if not args.verify_only:
        for path in (args.output, sample_path, preprocessing_path):
            if path.exists():
                raise ValueError(f"Refusing to overwrite existing compatible export artifact: {path}")
    try:
        wrapper = CompatibleOnnxBEVFusion(model, values, profile=profile, pool_backend=pool_backend,
            pool_num_points=num_points, score_threshold=threshold, fixed_calibration=fixed_calibration).eval()
        wrapper_values = tuple(wrapper.example_inputs)
        metadata = {"task": task, "classes": json.dumps(list(model.classes)),
                    "images": "normalized RGB float32 [1,6,3,256,704]",
                    "points": "fixed-capacity float32 [Pcap,5]; grouping and topology are per-frame CPU inputs",
                    "pool_backend": pool_backend, "pool_num_points": str(num_points),
                    "calibration_mode": report["calibration_mode"], "pooling_precomputed": str(bool(fixed_calibration)).lower(),
                    "scope": "entire_model", "lidar_backend": "compatible", "bpu_compatible": "false",
                    "cpu_preprocessing_required": "true", **wrapper.deployment_metadata()}
        if fixed_calibration:
            record = calibration_record(values)
            metadata["fixed_calibration"] = json.dumps(record)
            report["fixed_calibration"] = record
        if task == "det":
            metadata.update(box_origin="bottom", box_format_version="2", box_columns=json.dumps(BOX_COLUMNS),
                coordinate_frame="reference_lidar", score_threshold=str(threshold),
                checkpoint_decoded_z_origin=model.heads["object"].decoded_z_origin)
        else:
            metadata["grid"] = "[batch,class,X,Y], 200x200, XY[-50,50), 0.5m"
        dynamic_axes = {"camera_points": {0: "num_camera_points"}} if "camera_points" in wrapper.input_names else {}
        if task == "det":
            dynamic_axes.update({name: {0: "num_boxes"} for name in OUTPUT_NAMES[task]})
        report.update(inputs={name: list(value.shape) for name, value in zip(wrapper.input_names, wrapper_values)},
                      dynamic_axes=dynamic_axes, sampling=wrapper.pool_statistics())
        report["stage"] = "adapter_comparison"

        def reference(sensor_values):
            return model(*sensor_values, score_threshold=threshold) if task == "det" else model(*sensor_values)

        with torch.no_grad():
            print("[check]", task, "compatible adapter vs original PyTorch model", flush=True)
            actual, expected = wrapper(*wrapper_values), reference(values)
            report["adapter_comparison"] = compare_outputs(actual, expected, task, 3e-4, 3e-4)
        report["stage"] = "onnx_export"
        if args.verify_only:
            graph = onnx.load(str(args.output))
        else:
            print("[export]", task, "compatible ->", args.output, flush=True)
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                torch.onnx.export(wrapper, wrapper_values, str(args.output), dynamo=False,
                    opset_version=18, input_names=list(wrapper.input_names), output_names=OUTPUT_NAMES[task],
                    dynamic_axes=dynamic_axes or None, do_constant_folding=True,
                    export_params=True, training=torch.onnx.TrainingMode.EVAL)
            graph = onnx.load(str(args.output))
        report["stage"] = "onnx_structure"
        _validate_graph(graph, wrapper, metadata, verify_only=args.verify_only)
        if not args.verify_only:
            onnx.helper.set_model_props(graph, metadata)
            onnx.save(graph, str(args.output))
        validate_graph_contract(graph, task, model.classes)
        report.update(input_contract=_contracts(graph), operator_counts=operator_counts(graph.graph),
            file_bytes=args.output.stat().st_size, node_count=len(graph.graph.node),
            initializer_count=len(graph.graph.initializer), standard_onnx_operators_only=True)
        if not args.verify_only:
            np.savez_compressed(sample_path, **{name: value.detach().cpu().numpy()
                for name, value in zip(wrapper.input_names, wrapper_values)})
            _write_json(preprocessing_path, {"schema": "compatible-onnx-preprocessing-v1",
                "metadata": metadata, "inputs": report["input_contract"], "profile": profile.to_dict(),
                "fixed_calibration": report.get("fixed_calibration"),
                "cpu_preprocessing_required": True, "sample_inputs": sample_path.name})
        report["stage"] = "ort_verification"
        if args.verify:
            from test_onnx import create_session, select_session_feed, cuda_session_kwargs
            import onnxruntime as ort
            session = create_session(args.output, device=args.verify_device, threads=args.threads, **cuda_session_kwargs(args))
            cases = [("provided" if sample is not None else "synthetic", values),
                     ("changed_images", (values[0] * .9 + .03, *values[1:])),
                     ("empty_points", (values[0], values[1][:0], *values[2:])),
                     ("changed_points", (values[0], values[1][:min(7, len(values[1]))] +
                        values[1].new_tensor([.23, -.17, 0., 0., 0.]), *values[2:]))]
            if not fixed_calibration:
                intrinsic = values[2].clone()
                intrinsic[..., 0, 0] *= 1.01
                cases.append(("changed_calibration", (*values[:2], intrinsic, *values[3:])))
            checked = []
            report["tolerances"] = {"atol": args.atol, "rtol": args.rtol}
            for label, sensor_values in cases:
                print("[verify]", label, "points=", len(sensor_values[1]), flush=True)
                try:
                    with torch.no_grad():
                        expected = reference(sensor_values)
                    # The runtime selector validates metadata and rebuilds every
                    # frame's grouping and topology; no source checkpoint needed.
                    feed = select_session_feed(session, _sensor_feed(sensor_values))
                    output = session.run(OUTPUT_NAMES[task], feed)
                    metrics = compare_outputs(output[0] if task == "seg" else output,
                                              expected, task, args.atol, args.rtol)
                    checked.append({"name": label, "points": len(sensor_values[1]), **metrics})
                    print("[verify] result:", json.dumps(checked[-1]), flush=True)
                except Exception as exc:
                    checked.append({"name": label, "points": len(sensor_values[1]), "error": str(exc)})
                    report["verification_cases"] = checked
                    raise
            report.update(runtime_verified=True, runtime=ort.__version__, providers=session.get_providers(),
                          verification_cases=checked)
        report.update(status="exported", stage="completed")
        return report
    except Exception as exc:
        report.update(status="verification_failed" if report["stage"] == "ort_verification" else "failed",
                      error=f"{type(exc).__name__}: {exc}", runtime_verified=False)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_json(report_path, report)
        print("[report]", report_path, flush=True)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
