"""Resolved inference configuration for the supported nuScenes detection checkpoint."""
from copy import deepcopy
from pathlib import Path

from .detection_head import OBJECT_CLASSES

# The original convfuser.yaml is a leaf override; these are its inference bases.
DETECTION_CONFIG = {
    "image_size": [256, 704],
    "voxel_size": [0.075, 0.075, 0.2],
    "point_cloud_range": [-54., -54., -5., 54., 54., 3.],
    "object_classes": list(OBJECT_CLASSES),
    "model": {
        "encoders": {
            "camera": {
                "backbone": {"type": "SwinTransformer"},
                "neck": {"type": "GeneralizedLSSFPN"},
                "vtransform": {"type": "DepthLSSTransform", "out_channels": 80,
                               "xbound": [-54., 54., .3], "ybound": [-54., 54., .3],
                               "zbound": [-10., 10., 20.], "dbound": [1., 60., .5],
                               "downsample": 2},
            },
            "lidar": {
                "voxelize": {"max_num_points": 10, "max_voxels": [120000, 160000]},
                "backbone": {"type": "SparseEncoder", "in_channels": 5,
                             "sparse_shape": [1440, 1440, 41]},
            },
        },
        "fuser": {"type": "ConvFuser", "in_channels": [80, 256], "out_channels": 256},
        "decoder": {"backbone": {"type": "SECOND"}, "neck": {"type": "SECONDFPN"}},
        "heads": {"object": {
            "type": "TransFusionHead", "in_channels": 512, "hidden_channel": 128,
            "num_classes": 10, "num_proposals": 200, "num_decoder_layers": 1,
            "num_heads": 8, "nms_kernel_size": 3, "ffn_channel": 256,
            "test_cfg": {"grid_size": [1440, 1440, 41], "out_size_factor": 8,
                         "nms_type": None},
            # This released checkpoint's legacy training pipeline retained the
            # nuScenes centre Z in a nominally bottom-origin box. The legacy
            # coder therefore returns centre Z; normalize it at decoding time.
            "bbox_coder": {"post_center_range": [-61.2, -61.2, -10., 61.2, 61.2, 10.],
                           "score_threshold": 0.0, "decoded_z_origin": "center"},
        }},
    },
}


def load_detection_config(value=None):
    """Merge a convfuser leaf or resolved dict into the checkpoint's known bases.

    This builds the supported Swin/DepthLSS/TransFusion architecture, rather than
    importing OpenMMLab registries or evaluating executable ${...} expressions.
    Unknown or incompatible architecture settings raise instead of being ignored.
    """
    if isinstance(value, (str, Path)):
        import yaml
        with Path(value).open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    if value is not None and not isinstance(value, dict):
        raise ValueError("Detection config must be a YAML mapping or a dict")
    result = deepcopy(DETECTION_CONFIG)
    adjustable = {
        "model.heads.object.num_proposals",
        "model.heads.object.bbox_coder.score_threshold",
        "model.heads.object.bbox_coder.post_center_range",
        "model.heads.object.bbox_coder.decoded_z_origin",
    }

    def merge(target, source, prefix=""):
        for key, item in source.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in target:
                raise ValueError(f"Unsupported detection config field: {path}")
            if isinstance(target[key], dict):
                if not isinstance(item, dict):
                    raise ValueError(f"Expected mapping for detection config field: {path}")
                merge(target[key], item, path)
            else:
                comparable = list(item) if isinstance(item, tuple) else item
                if path not in adjustable and comparable != target[key]:
                    raise ValueError(f"Detection checkpoint requires {path}={target[key]!r}, got {item!r}")
                target[key] = deepcopy(comparable)

    merge(result, value or {})
    head = result["model"]["heads"]["object"]
    if not isinstance(head["num_proposals"], int) or not 1 <= head["num_proposals"] <= 10 * 180 * 180:
        raise ValueError("num_proposals must be a positive integer within the BEV heatmap size")
    coder = head["bbox_coder"]
    if coder["decoded_z_origin"] not in ("center", "bottom"):
        raise ValueError("decoded_z_origin must be center or bottom")
    if not isinstance(coder["score_threshold"], (int, float)) or not 0 <= coder["score_threshold"] <= 1:
        raise ValueError("score_threshold must be in [0,1]")
    bounds = coder["post_center_range"]
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 6
            or any(not isinstance(x, (int, float)) or not float('-inf') < x < float('inf') for x in bounds)
            or any(bounds[i] > bounds[i + 3] for i in range(3))):
        raise ValueError("post_center_range must contain six finite ordered bounds")
    return result
