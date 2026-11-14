"""Split an existing BEVFusion ONNX at its camera/LiDAR BEV fusion boundary.

Weights and tensor operations are extracted from the source graph. Geometry-only
conditional Squeeze guards may be specialized; no sample values are frozen.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import onnx
from onnx import AttributeProto, TensorProto, helper
from onnx.utils import Extractor

from tools.static_onnx import specialize_static_shapes


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ancestors(outputs, producers, graph_inputs, *, stops=()):
    """Return required graph inputs, stopping at explicit branch boundary values."""
    pending, visited, required = list(outputs), set(), set()
    stops = set(stops)
    while pending:
        value = pending.pop()
        if value in visited or not value:
            continue
        visited.add(value)
        if value in stops:
            continue
        if value in graph_inputs:
            required.add(value)
        elif value in producers:
            pending.extend(producers[value].input)
    return required


def _contracts(values):
    types = {TensorProto.FLOAT: "float32", TensorProto.FLOAT16: "float16",
             TensorProto.DOUBLE: "float64", TensorProto.INT32: "int32",
             TensorProto.INT64: "int64", TensorProto.INT16: "int16",
             TensorProto.INT8: "int8", TensorProto.UINT8: "uint8", TensorProto.BOOL: "bool"}
    result = []
    for value in values:
        tensor = value.type.tensor_type
        if tensor.elem_type not in types or not tensor.HasField("shape"):
            raise ValueError(f"Branch tensor requires a known tensor type/shape: {value.name}")
        shape = [dim.dim_value if dim.HasField("dim_value") else None for dim in tensor.shape.dim]
        if any(size is None or size < 1 for size in shape):
            raise ValueError(f"Branch tensor requires fixed positive dimensions: {value.name}: {shape}")
        result.append({"name": value.name, "shape": shape, "dtype": types[tensor.elem_type]})
    return result


def _operators(graph):
    counts = Counter()
    for node in graph.node:
        counts[node.op_type] += 1
        for attribute in node.attribute:
            nested = ([attribute.g] if attribute.type == AttributeProto.GRAPH else
                      attribute.graphs if attribute.type == AttributeProto.GRAPHS else ())
            for child in nested:
                counts.update(_operators(child))
    return dict(counts)


def _rename_values(graph, replacements):
    existing = {value for node in graph.node for value in (*node.input, *node.output)}
    existing.update(value.name for value in (*graph.input, *graph.output, *graph.initializer))
    for old, new in replacements.items():
        if new in existing and new != old:
            raise ValueError(f"Cannot name branch boundary {new!r}: source tensor already uses that name")
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = replacements.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = replacements.get(name, name)
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                _rename_values(attribute.g, replacements)
            elif attribute.type == AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    _rename_values(child, replacements)
    for value in (*graph.input, *graph.output, *graph.value_info, *graph.initializer):
        value.name = replacements.get(value.name, value.name)


def _fusion_inputs(model, producers):
    fusion = next((node for node in model.graph.node
                   if node.op_type == "Conv" and "fuser" in node.name.split("/")), None)
    if fusion is None or not fusion.input:
        raise ValueError("Cannot find the first fuser Conv; explicit cuts still require an identifiable fusion boundary")
    concat = producers.get(fusion.input[0])
    if concat is None or concat.op_type != "Concat" or len(concat.input) != 2:
        raise ValueError("The first fuser Conv must consume a two-input BEV Concat")
    axis = next((attribute.i for attribute in concat.attribute if attribute.name == "axis"), None)
    if axis != 1:
        raise ValueError("The BEV fusion boundary must concatenate along channel axis 1")
    return fusion, concat


def _identify_cut(value, producers, graph_inputs):
    producer = producers.get(value)
    prefix = producer.name if producer is not None else value
    lineage = _ancestors([value], producers, graph_inputs)
    camera = "images" in lineage
    lidar = (not camera and any(name in ("points", "point_indices", "voxel_features", "lidar_bev", "dense_lookup")
                               or name.startswith("neighbors_") for name in lineage))
    # Prefixes confirm, rather than override, known source-input lineage.
    if camera and not prefix.startswith(("/lidar/", "/sparse/")):
        return "camera"
    if lidar and not prefix.startswith("/camera/"):
        return "lidar"
    return None


def split_model(source: Path, output_dir: Path, *, camera_output=None, lidar_output=None):
    """Extract camera, LiDAR and fusion models into a new output directory.

    Returns a persisted manifest with absolute ONNX paths and fixed input/output
    contracts. All original graph outputs remain fusion outputs. Calibration and
    runtime scheduling belong to the caller; successful splitting is not HBM
    compilation or quantized-accuracy verification.
    """
    source = Path(source).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite existing split directory: {output_dir}")
    if source.suffix.lower() != ".onnx":
        raise ValueError("source must be an existing .onnx file")
    if (camera_output is None) != (lidar_output is None):
        raise ValueError("camera_output and lidar_output must be provided together")
    fingerprint = _sha256(source)
    model = onnx.load(str(source))
    onnx.checker.check_model(model, full_check=True)
    _contracts(model.graph.input)
    model, specialization = specialize_static_shapes(model)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    producers = {value: node for node in model.graph.node for value in node.output}
    input_names = [value.name for value in model.graph.input]
    graph_inputs = set(input_names)
    fusion, concat = _fusion_inputs(model, producers)
    if camera_output is None:
        identified = {_identify_cut(value, producers, graph_inputs): value for value in concat.input}
        if set(identified) != {"camera", "lidar"}:
            raise ValueError("Ambiguous BEV branches; provide both camera_output and lidar_output tensor names")
        camera_output, lidar_output = identified["camera"], identified["lidar"]
    if not isinstance(camera_output, str) or not isinstance(lidar_output, str) or camera_output == lidar_output:
        raise ValueError("Camera/LiDAR cut points must be distinct tensor names")
    if set(concat.input) != {camera_output, lidar_output}:
        raise ValueError("Explicit cuts must be exactly the two BEV inputs consumed by the first fuser Concat")
    cuts = {"camera": camera_output, "lidar": lidar_output}
    values = {value.name: value for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    if any(name not in values for name in cuts.values()):
        raise ValueError("BEV cut point shape information is missing after ONNX shape inference")
    camera_spec, lidar_spec = _contracts([values[camera_output], values[lidar_output]])
    if (len(camera_spec["shape"]) != 4 or len(lidar_spec["shape"]) != 4
            or camera_spec["shape"][0] != lidar_spec["shape"][0]
            or camera_spec["shape"][2:] != lidar_spec["shape"][2:]):
        raise ValueError("Camera/LiDAR cuts must be matching NCHW BEV feature maps")
    output_names = [value.name for value in model.graph.output]
    extra = _ancestors(output_names, producers, graph_inputs, stops=cuts.values())
    if extra:
        raise ValueError(f"Fusion has additional source-input dependencies outside BEV cuts: {sorted(extra)}")
    required = {name: _ancestors([cut], producers, graph_inputs) for name, cut in cuts.items()}
    if not all(required.values()):
        raise ValueError("Each encoder branch must depend on runtime sensor/plan inputs")
    source_metadata = {item.key: item.value for item in model.metadata_props}
    manifest = {"schema": "bevfusion-split-onnx-v1", "source_onnx": str(source),
                "source_sha256": fingerprint, "specialization": specialization,
                "cuts": cuts, "fusion_concat_order": list(concat.input), "fuser_node": fusion.name,
                "branches": {}, "execution_order": ["camera", "lidar", "fusion"],
                "quantized_accuracy_verified": False, "hbm_compiled": False,
                "task": source_metadata.get("task"), "source_unchanged": True}
    # Validate all three extracted graphs before creating files, so malformed
    # boundaries fail without leaving a misleading partial split bundle.
    extracted = {}
    extractor = Extractor(model)
    for name in ("camera", "lidar", "fusion"):
        names = ([item for item in input_names if item in required[name]] if name != "fusion"
                 else [camera_output, lidar_output])
        outputs = [cuts[name]] if name != "fusion" else output_names
        branch = extractor.extract_model(names, outputs)
        branch.graph.name = f"bevfusion_{name}"
        _rename_values(branch.graph, {cut: f"{kind}_bev" for kind, cut in cuts.items()
                                     if name == "fusion" or kind == name})
        metadata = {key: value for key, value in source_metadata.items()
                    if key in ("task", "classes", "pool_backend", "pool_num_points", "calibration_mode",
                               "fixed_calibration", "box_origin", "box_format_version", "box_columns",
                               "checkpoint_decoded_z_origin", "score_threshold", "grid")}
        metadata.update(branch=name, scope="branch", source_sha256=fingerprint,
                        source_lidar_backend=source_metadata.get("lidar_backend", "unknown"),
                        source_onnx=str(source))
        helper.set_model_props(branch, metadata)
        onnx.checker.check_model(branch, full_check=True)
        counts = _operators(branch.graph)
        manifest["branches"][name] = {"onnx": str(output_dir / name / "model.onnx"),
            "inputs": _contracts(branch.graph.input), "outputs": _contracts(branch.graph.output),
            "operator_counts": counts,
            "unverified_target_operators": {op: counts[op] for op in ("Loop", "Unique", "If", "Scan", "NonZero") if counts.get(op)}}
        extracted[name] = branch
    _contracts(model.graph.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    whole_path = output_dir / "whole_model.onnx"
    onnx.save(model, str(whole_path))
    manifest["whole_model"] = str(whole_path)
    for name, branch in extracted.items():
        path = Path(manifest["branches"][name]["onnx"])
        path.parent.mkdir()
        onnx.save(branch, str(path))
        manifest["branches"][name]["file_bytes"] = path.stat().st_size
    manifest["source_unchanged"] = _sha256(source) == fingerprint
    if not manifest["source_unchanged"]:
        raise RuntimeError("Source ONNX changed during splitting")
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest
