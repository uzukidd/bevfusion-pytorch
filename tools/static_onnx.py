"""Prove and remove static conditional-Squeeze guards from exported ONNX.

Legacy PyTorch emits If(Shape(x)[axis] == 1, Squeeze(x), Identity(x)).
This pass evaluates only small constant/shape expressions, never sensor values
or learned activations. Unknown conditions are left intact and reported. It is
not a substitute for auditing the remaining target operator support.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import onnx
from onnx import AttributeProto, numpy_helper


_SMALL = 256


def _attributes(node):
    return {attribute.name: attribute for attribute in node.attribute}


def _shapes(model):
    # This pass specializes top-level guards only. Subgraph-local tensor names
    # may shadow outer names and must never contribute shape facts here.
    graph = model.graph
    result = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor = value.type.tensor_type
        if tensor.HasField("shape"):
            result[value.name] = tuple(d.dim_value if d.HasField("dim_value") else None for d in tensor.shape.dim)
    for value in graph.initializer:
        result[value.name] = tuple(value.dims)
    return result


def _small_tensor(tensor):
    if math.prod(tensor.dims) > _SMALL:
        return None
    return numpy_helper.to_array(tensor)


class _ShapeValues:
    def __init__(self, graph, shapes):
        self.shapes = shapes
        self.producers = {output: node for node in graph.node for output in node.output}
        self.values = {value.name: small for value in graph.initializer if (small := _small_tensor(value)) is not None}
        self.visiting = set()

    def get(self, name):
        if name in self.values:
            return self.values[name]
        if name in self.visiting or name not in self.producers:
            return None
        self.visiting.add(name)
        try:
            value = self._compute(self.producers[name])
            if value is not None:
                value = np.asarray(value)
                if value.size <= _SMALL:
                    self.values[name] = value
                    return value
            return None
        except (ValueError, TypeError, IndexError, KeyError, ZeroDivisionError, OverflowError):
            return None
        finally:
            self.visiting.discard(name)

    def _compute(self, node):
        attrs = _attributes(node)
        op = node.op_type
        if node.domain not in ("", "ai.onnx"):
            return None
        if op == "Constant":
            if "value" in attrs:
                return _small_tensor(attrs["value"].t)
            for key, field in (("value_int", "i"), ("value_ints", "ints"), ("value_float", "f"), ("value_floats", "floats")):
                if key in attrs:
                    return np.asarray(getattr(attrs[key], field))
            return None
        if op == "Shape":
            shape = self.shapes.get(node.input[0])
            if shape is None:
                return None
            start = attrs["start"].i if "start" in attrs else 0
            end = attrs["end"].i if "end" in attrs else len(shape)
            shape = shape[start:end]
            return np.asarray(shape, dtype=np.int64) if all(value is not None for value in shape) else None
        # Resolve a known selected dimension even when other dimensions remain
        # symbolic; do not substitute concrete sample activation shapes.
        if op == "Gather" and (attrs["axis"].i if "axis" in attrs else 0) == 0:
            source = self.producers.get(node.input[0])
            indices = self.get(node.input[1])
            if source is not None and source.op_type == "Shape" and indices is not None:
                shape = self.shapes.get(source.input[0])
                if shape is not None:
                    shape_attrs = _attributes(source)
                    start = shape_attrs["start"].i if "start" in shape_attrs else 0
                    end = shape_attrs["end"].i if "end" in shape_attrs else len(shape)
                    selected = np.asarray(np.take(np.asarray(shape[start:end], dtype=object), indices.astype(np.int64), axis=0))
                    if all(value is not None for value in selected.reshape(-1)):
                        return np.asarray(selected, dtype=np.int64)
        values = [self.get(name) if name else None for name in node.input]
        if any(value is None for name, value in zip(node.input, values) if name):
            return None
        if op == "Identity":
            return values[0]
        if op == "Gather":
            return np.take(values[0], values[1].astype(np.int64), axis=attrs["axis"].i if "axis" in attrs else 0)
        if op == "Concat":
            return np.concatenate(values, axis=attrs["axis"].i)
        if op == "Cast":
            dtype = onnx.helper.tensor_dtype_to_np_dtype(attrs["to"].i)
            return values[0].astype(dtype)
        if op == "Unsqueeze":
            axes = values[1].reshape(-1).tolist() if len(values) > 1 else list(attrs["axes"].ints)
            result = values[0]
            rank = result.ndim + len(axes)
            for axis in sorted(axis % rank for axis in axes):
                result = np.expand_dims(result, axis)
            return result
        if op == "Squeeze":
            axes = tuple(values[1].reshape(-1).tolist()) if len(values) > 1 else tuple(attrs["axes"].ints) if "axes" in attrs else None
            return np.squeeze(values[0], axis=axes)
        if op == "Reshape":
            shape = values[1].astype(np.int64).reshape(-1).tolist()
            if not (attrs["allowzero"].i if "allowzero" in attrs else 0):
                shape = [values[0].shape[index] if size == 0 else size for index, size in enumerate(shape)]
            return values[0].reshape(shape)
        binary = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Equal": np.equal,
                  "Greater": np.greater, "Less": np.less, "GreaterOrEqual": np.greater_equal,
                  "LessOrEqual": np.less_equal, "And": np.logical_and, "Or": np.logical_or}
        if op in binary:
            return binary[op](values[0], values[1])
        if op == "Div":
            if np.any(values[1] == 0):
                return None
            if values[0].dtype.kind in "iu":
                quotient, remainder = np.divmod(values[0], values[1])
                # ONNX integer division truncates towards zero. Avoid converting
                # INT64 shape values to float, which could lose proof precision.
                if values[0].dtype.kind == "i":
                    quotient = quotient + (((values[0] < 0) != (values[1] < 0)) & (remainder != 0))
                return quotient.astype(values[0].dtype)
            return np.divide(values[0], values[1])
        if op == "Not":
            return np.logical_not(values[0])
        if op in ("ReduceProd", "ReduceSum"):
            axes = tuple(values[1].reshape(-1).tolist()) if len(values) > 1 else tuple(attrs["axes"].ints) if "axes" in attrs else None
            if axes == ():
                if "noop_with_empty_axes" in attrs and attrs["noop_with_empty_axes"].i:
                    return values[0]
                axes = None
            return (np.prod if op == "ReduceProd" else np.sum)(values[0], axis=axes, keepdims=bool(attrs["keepdims"].i) if "keepdims" in attrs else True)
        if op == "Slice":
            starts, ends = values[1].reshape(-1), values[2].reshape(-1)
            axes = values[3].reshape(-1) if len(values) > 3 and values[3] is not None else np.arange(len(starts))
            steps = values[4].reshape(-1) if len(values) > 4 and values[4] is not None else np.ones(len(starts), dtype=np.int64)
            slices = [slice(None)] * values[0].ndim
            for start, end, axis, step in zip(starts, ends, axes, steps):
                slices[int(axis)] = slice(int(start), int(end), int(step))
            return values[0][tuple(slices)]
        return None


def _squeeze_guard(node):
    if node.domain not in ("", "ai.onnx"):
        return None
    branches = {attr.name: attr.g for attr in node.attribute if attr.type == AttributeProto.GRAPH}
    if set(branches) != {"then_branch", "else_branch"} or any(graph.input for graph in branches.values()):
        return None
    then, otherwise = branches["then_branch"], branches["else_branch"]
    if any(child.domain not in ("", "ai.onnx") for branch in branches.values() for child in branch.node):
        return None
    meaningful = [child for child in then.node if child.op_type != "Constant"]
    if len(meaningful) != 1 or meaningful[0].op_type != "Squeeze" or len(otherwise.node) != 1 or otherwise.node[0].op_type != "Identity":
        return None
    squeeze, identity = meaningful[0], otherwise.node[0]
    if squeeze.input[0] != identity.input[0] or len(then.output) != 1 or len(otherwise.output) != 1 or len(node.output) != 1:
        return None
    if then.output[0].name != squeeze.output[0] or otherwise.output[0].name != identity.output[0]:
        return None
    return branches


def _clear_intermediate_shapes(graph):
    del graph.value_info[:]
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                _clear_intermediate_shapes(attr.g)


def specialize_static_shapes(model):
    """Return (copied model, report) after eliminating provable Squeeze Ifs.

    All top-level input dimensions must be fixed and positive. Unknown internal
    shapes remain symbolic, and unproven/other If forms are retained, allowing
    the caller's deployment audit to reject rather than guess their semantics.
    Learned weights and data-path operations are neither evaluated nor folded.
    """
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("Expected an ONNX ModelProto")
    for value in model.graph.input:
        tensor = value.type.tensor_type
        if not tensor.HasField("shape") or any(not dim.HasField("dim_value") or dim.dim_value < 1 for dim in tensor.shape.dim):
            raise ValueError("Static specialization requires fixed positive graph input dimensions")
    try:
        working = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except (onnx.shape_inference.InferenceError, RuntimeError):
        # An unreachable Squeeze on a non-unit dimension can make inference
        # reject an otherwise removable guard. Graph input/initializer shapes
        # still provide sound facts for the small shape evaluator.
        working = copy.deepcopy(model)
    shapes = _shapes(working)
    evaluator = _ShapeValues(working.graph, shapes)
    report = {"scope": "top_level_conditional_squeeze_guards",
              "if_before": sum(node.op_type == "If" for node in working.graph.node),
              "if_removed": 0, "if_remaining": 0, "specialized": [], "unresolved_if": []}
    names = {name for node in working.graph.node for name in (*node.input, *node.output)}
    names.update(node.name for node in working.graph.node if node.name)
    names.update(value.name for value in working.graph.initializer)
    def unique(base):
        value, counter = base, 0
        while value in names:
            counter += 1
            value = base + "_" + str(counter)
        names.add(value)
        return value
    rewritten = []
    for node in working.graph.node:
        if node.op_type != "If":
            rewritten.append(node)
            continue
        branches = _squeeze_guard(node)
        condition = evaluator.get(node.input[0]) if branches is not None and node.input else None
        if condition is None or condition.size != 1 or condition.dtype.kind != "b":
            report["unresolved_if"].append(node.name or node.output[0])
            rewritten.append(node)
            continue
        taken = "then_branch" if bool(condition.reshape(-1)[0]) else "else_branch"
        branch = branches[taken]
        mapping = {name: unique((node.name or node.output[0]) + "/static/" + name)
                   for child in branch.node for name in child.output}
        mapping.update({value.name: unique((node.name or node.output[0]) + "/static/" + value.name)
                        for value in branch.initializer})
        for initializer in branch.initializer:
            cloned = copy.deepcopy(initializer)
            cloned.name = mapping[initializer.name]
            working.graph.initializer.append(cloned)
        for child in branch.node:
            cloned = copy.deepcopy(child)
            cloned.name = unique((node.name or node.output[0]) + "/static/" + (child.name or child.op_type))
            for index, name in enumerate(cloned.input):
                cloned.input[index] = mapping.get(name, name)
            for index, name in enumerate(cloned.output):
                cloned.output[index] = mapping[name]
            rewritten.append(cloned)
        rewritten.append(onnx.helper.make_node("Identity", [mapping.get(branch.output[0].name, branch.output[0].name)],
                                                list(node.output), name=unique((node.name or node.output[0]) + "/static/output")))
        report["specialized"].append({"node": node.name or node.output[0], "branch": taken,
                                      "proof": "small constant/known-shape condition", "condition": bool(condition.reshape(-1)[0])})
        report["if_removed"] += 1
    del working.graph.node[:]
    working.graph.node.extend(rewritten)
    report["if_remaining"] = report["if_before"] - report["if_removed"]
    if report["if_removed"]:
        _clear_intermediate_shapes(working.graph)
        working = onnx.shape_inference.infer_shapes(working, strict_mode=True, data_prop=True)
    onnx.checker.check_model(working, full_check=True)
    return working, report


__all__ = ["specialize_static_shapes"]
