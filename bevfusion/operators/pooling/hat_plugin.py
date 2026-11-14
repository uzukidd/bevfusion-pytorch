"""Horizon-plugin LSS sampling with the existing fixed-slot geometry and weights.

The optional plugin is loaded on execution. This uses HAT's pixel-offset sampler
and quantization modules, not HAT's different BEV-to-camera geometry selection.
"""
from __future__ import annotations

import importlib
import math
import sys

import torch
from torch import nn

from .hat import HATLSSPool


class _HorizonRuntime(nn.Module):
    def __init__(self, grid_scale):
        super().__init__()
        try:
            hnn = importlib.import_module("horizon_plugin_pytorch.nn")
            quantization = importlib.import_module("horizon_plugin_pytorch.quantization")
            functional = importlib.import_module("horizon_plugin_pytorch.nn.quantized")
            self.feature_sampler = hnn.GridSample(mode="bilinear", padding_mode="zeros", align_corners=False)
            self.depth_sampler = hnn.GridSample(mode="bilinear", padding_mode="zeros", align_corners=False)
            self.feature_quant = quantization.QuantStub(scale=grid_scale)
            self.depth_quant = quantization.QuantStub(scale=grid_scale)
            self.arithmetic = functional.FloatFunctional()
        except (ImportError, OSError, AttributeError) as exc:
            raise RuntimeError("hat_plugin requires a compatible horizon_plugin_pytorch installation; "
                               "use its supported Linux/PyTorch environment") from exc

    def set_qconfig(self):
        quantization = importlib.import_module("horizon_plugin_pytorch.quantization")
        dtype = importlib.import_module("horizon_plugin_pytorch.dtype")
        self.qconfig = quantization.get_default_qat_qconfig()
        coordinate_qconfig = quantization.get_default_qat_qconfig(
            dtype=dtype.qint16, activation_qkwargs={"saturate": True})
        self.feature_quant.qconfig = coordinate_qconfig
        self.depth_quant.qconfig = coordinate_qconfig
        self.arithmetic.qconfig = self.qconfig


class HATPluginLSSPool(HATLSSPool):
    """HAT plugin arithmetic over HATLSSPool's unchanged first-K paged slots.

    ``runtime_factory(grid_scale)`` is a test-only injection hook returning an
    nn.Module with feature/depth_sampler, feature/depth_quant and arithmetic
    (mul/add) modules; production callers should leave it unset. Geometry buffers
    are runtime plans, not learned checkpoint parameters. Call set_qconfig before
    plugin prepare/export; constructing this class alone does not perform QAT.
    """
    def __init__(self, coordinates, grid_shape, channels, *, num_points=10,
                 source_page_shape=(512, 1024), grid_scale=1 / 16, runtime_factory=None):
        if isinstance(grid_scale, bool) or not isinstance(grid_scale, (int, float)):
            raise ValueError("grid_scale must be a finite positive power of two")
        if not math.isfinite(grid_scale) or grid_scale <= 0 or not math.log2(grid_scale).is_integer():
            raise ValueError("grid_scale must be a finite positive power of two")
        if runtime_factory is not None and not callable(runtime_factory):
            raise ValueError("runtime_factory must be callable")
        super().__init__(coordinates, grid_shape, channels, num_points=num_points,
                         source_page_shape=source_page_shape)
        self.grid_scale = float(grid_scale)
        self._runtime_factory = runtime_factory
        self._runtime = None
        self._qconfig_requested = False
        _, oh, ow = self.sample_shape
        y, x = torch.meshgrid(torch.arange(oh, dtype=torch.float64, device=coordinates.device),
                              torch.arange(ow, dtype=torch.float64, device=coordinates.device), indexing="ij")
        base = torch.stack((x, y), dim=-1)
        largest = 0.0
        for kind, pages in (("feature", self._feature_pages), ("depth", self._depth_pages)):
            for slot in range(self.num_points):
                for page, (_, _, ph, pw) in enumerate(pages):
                    normalized = getattr(self, f"{kind}_grid_{slot}_{page}").double()
                    factor = normalized.new_tensor((pw / 2, ph / 2))
                    offset = ((normalized + 1) * factor - .5 - base).float()
                    largest = max(largest, float(offset.abs().max()))
                    scaled = offset.double() / self.grid_scale
                    if (scaled < -32768).any() or (scaled > 32767).any():
                        raise ValueError("Pixel-offset geometry exceeds INT16 grid_scale range; "
                                         "select a larger grid_scale rather than clipping coordinates")
                    if not torch.equal(scaled, scaled.round()):
                        raise ValueError("grid_scale cannot represent pixel-offset geometry exactly")
                    self.register_buffer(f"{kind}_offset_{slot}_{page}", offset)
        self.max_abs_offset = largest
        self.eval()

    def _get_runtime(self):
        if self._runtime is None:
            factory = self._runtime_factory or _HorizonRuntime
            runtime = factory(self.grid_scale)
            if not isinstance(runtime, nn.Module):
                raise TypeError("runtime_factory must return an nn.Module")
            runtime = runtime.to(device=self.slot_mask_0.device).eval()
            self._runtime = runtime
        return self._runtime

    def set_qconfig(self):
        """Configure INT16 coordinates; plugin preparation remains an explicit step."""
        self._qconfig_requested = True
        runtime = self._get_runtime()
        if not hasattr(runtime, "set_qconfig"):
            raise RuntimeError("Plugin runtime does not support set_qconfig")
        runtime.set_qconfig()
        self.qconfig = runtime.qconfig
        return self

    def _sample_plugin(self, pages, kind, slot, runtime):
        sampler = getattr(runtime, f"{kind}_sampler")
        quant = getattr(runtime, f"{kind}_quant")
        result = None
        for page, values in enumerate(pages):
            # Plugin warp normalizes by source size-1, so singleton dimensions
            # must be zero-padded. Absolute source pixels/offsets stay unchanged.
            height, width = values.shape[-2:]
            if height < 2 or width < 2:
                values = torch.nn.functional.pad(values, (0, max(0, 2 - width), 0, max(0, 2 - height)))
            sampled = sampler(values, quant(getattr(self, f"{kind}_offset_{slot}_{page}")))
            sampled = runtime.arithmetic.mul(sampled, getattr(self, f"{kind}_mask_{slot}_{page}"))
            result = sampled if result is None else runtime.arithmetic.add(result, sampled)
        return result

    def forward(self, depth, features):
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError("hat_plugin requires Horizon HBIR export; ordinary ONNX export is unsupported")
        exporter = sys.modules.get("horizon_plugin_pytorch.quantization.hbdk4")
        exporting_hbir = exporter is not None and exporter.is_exporting()
        if not (torch.jit.is_tracing() or exporting_hbir):
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("HATPluginLSSPool is inference-only; use eval() and inference_mode()")
            if not isinstance(depth, torch.Tensor) or not isinstance(features, torch.Tensor):
                raise ValueError("depth/features must be tensors")
            if tuple(depth.shape) != self.depth_shape or tuple(features.shape) != self.feature_shape:
                raise ValueError(f"Expected depth {self.depth_shape} and features {self.feature_shape}")
            if depth.dtype != torch.float32 or features.dtype != torch.float32:
                raise ValueError("HATPluginLSSPool requires float32 inputs")
            if depth.device != features.device or depth.device != self.slot_mask_0.device:
                raise ValueError("Inputs and geometry buffers must share a device")
            if not torch.isfinite(depth).all() or not torch.isfinite(features).all():
                raise ValueError("Inputs must be finite")
        if not self.selected_points:
            return features.new_zeros(self.output_shape)
        runtime = self._get_runtime()
        packed_features = self._pack(features.permute(0, 2, 1, 3, 4), self._feature_pages)
        packed_depth = self._pack(depth.unsqueeze(1), self._depth_pages)
        output = None
        for slot in range(self.num_points):
            contribution = runtime.arithmetic.mul(self._sample_plugin(packed_features, "feature", slot, runtime),
                                                  self._sample_plugin(packed_depth, "depth", slot, runtime))
            contribution = runtime.arithmetic.mul(contribution, getattr(self, f"slot_mask_{slot}"))
            output = contribution if output is None else runtime.arithmetic.add(output, contribution)
        nx, ny, nz = self.grid_shape
        return output.reshape(self.depth_shape[0], self.channels, nz, nx, ny).permute(0, 2, 1, 3, 4).reshape(self.output_shape)

    def statistics(self):
        result = super().statistics()
        result.update(backend="hat_plugin", implementation="horizon_plugin_pytorch.GridSample/FloatFunctional",
                      coordinate_format="pixel_offset_xy", grid_scale=self.grid_scale,
                      max_abs_offset=self.max_abs_offset, export_route="HBIR",
                      minimum_plugin_source_shape=[2, 2],
                      plugin_source_shapes={
                          "features": [[self.depth_shape[0], self.channels, max(2, p[2]), max(2, p[3])]
                                       for p in self._feature_pages],
                          "depth": [[self.depth_shape[0], 1, max(2, p[2]), max(2, p[3])]
                                    for p in self._depth_pages]},
                      runtime_loaded=self._runtime is not None, qconfig_configured=self._qconfig_requested,
                      quantized_accuracy_verified=False, bpu_compiled=False)
        return result


__all__ = ["HATPluginLSSPool"]
