"""Optional PNG/BIN readers for tensor-only BEVFusion.

English: reads one synchronized frame; calibration must match the six PNGs.
中文：读取同一参考帧的六相机 PNG 和原始点云 BIN；标定必须与图像对应。
NumPy and Pillow are used only by this optional file adapter, not model forward.
"""
import json
from pathlib import Path

import torch

from .model import _camera_matrix

DEFAULT_CAMERA_NAMES = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


def load_calibration(path):
    """Read original-image K and camera-to-reference-LiDAR from JSON or PT/PTH.

    Required keys: camera_intrinsics and camera2lidar. Optional camera_names and
    image_files are ordered lists of six strings. A saved complete model's
    calibration dictionary or an existing input tensor dictionary is accepted.
    Image augmentation from tensor files is intentionally not reused for PNGs.
    """
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in (".pt", ".pth"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("Calibration must be .json, .pt or .pth")
    if isinstance(payload, dict) and isinstance(payload.get("calibration"), dict):
        payload = payload["calibration"]
    if not isinstance(payload, dict) or not {"camera_intrinsics", "camera2lidar"}.issubset(payload):
        raise ValueError("Calibration must contain camera_intrinsics and camera2lidar; "
                         "an original official weight checkpoint has no sensor calibration")
    result = {
        "camera_intrinsics": _camera_matrix(payload["camera_intrinsics"], "camera_intrinsics", ((3, 3), (4, 4))),
        "camera2lidar": _camera_matrix(payload["camera2lidar"], "camera2lidar", ((4, 4),)),
    }
    if any(result[key] is None for key in result):
        raise ValueError("Calibration matrices cannot be null")
    for key in ("camera_names", "image_files"):
        if key in payload:
            values = payload[key]
            if (not isinstance(values, (list, tuple)) or len(values) != 6
                    or any(not isinstance(value, str) or not value.strip() for value in values)
                    or len(set(values)) != 6):
                raise ValueError(f"Calibration {key} must contain six distinct nonempty strings")
            result[key] = list(values)
    return result


def _file(root, value, suffix):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != suffix:
        raise ValueError(f"Expected a {suffix} file: {path}")
    return path


def read_pngs(paths):
    """Six ordered PNG paths -> uint8 RGB Tensor[6,3,H,W], no normalization."""
    import numpy as np
    from PIL import Image

    paths = list(paths)
    if len(paths) != 6:
        raise ValueError("Exactly six PNG paths are required, in calibration order")
    images = []
    expected = None
    for path in paths:
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        if expected is None:
            expected = array.shape
        elif array.shape != expected:
            raise ValueError(f"All six cameras must share H/W; {path} has {array.shape}, expected {expected}")
        images.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    return torch.stack(images)


def read_lidar_bin(path, *, point_dim=5, fifth_column="ring"):
    """Little-endian float32 BIN -> float32 Tensor[P,5].

    point_dim=4: XYZ/intensity, append zero time lag.
    point_dim=5, fifth_column='ring': XYZ/intensity/ring (nuScenes raw format),
    replace ring by zero time lag. 'time-lag' preserves an existing seconds column.
    No stride guessing: both four and five columns can divide the same file size.
    中文：默认按 nuScenes 五列读取，当前帧 ring 列清零；四列需显式指定。
    """
    import numpy as np

    if point_dim not in (4, 5):
        raise ValueError("point_dim must be 4 or 5; specify the actual BIN record width")
    if fifth_column not in ("ring", "time-lag"):
        raise ValueError("fifth_column must be ring or time-lag")
    path = Path(path).expanduser().resolve(strict=True)
    if path.stat().st_size % (4 * point_dim):
        raise ValueError(f"BIN byte count is not divisible by float32 x {point_dim}: {path}")
    array = np.fromfile(path, dtype="<f4").reshape(-1, point_dim)
    if not np.isfinite(array).all():
        raise ValueError(f"Point cloud contains NaN or infinity: {path}")
    points = torch.from_numpy(array.copy())
    if point_dim == 4:
        points = torch.cat((points, torch.zeros((len(points), 1), dtype=torch.float32)), dim=1)
    elif fifth_column == "ring":
        points[:, 4] = 0.0
    return points


def load_samples(sample_dir="data/samples", *, calibration=None, image_files=None,
                 lidar_file=None, point_dim=5, fifth_column="ring"):
    """Read six PNGs + one BIN into the dictionary consumed by test.py.

    If present, sample_dir/calibration.json is loaded automatically. Otherwise
    an original checkpoint needs an explicit calibration argument. Calibration
    can also be bound in a complete model, in which case matrices may be omitted.
    PNG order is calibration.camera_names, or DEFAULT_CAMERA_NAMES. For arbitrary
    filenames pass image_files (six ordered paths) or calibration.image_files.
    Paths in image_files/lidar_file resolve relative to sample_dir; an explicitly
    provided calibration path resolves from the process working directory.
    Returns images, points, camera_names, file provenance and optional K/extrinsics.
    """
    root = Path(sample_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Sample root must be a directory: {root}")
    if calibration is None and (root / "calibration.json").is_file():
        calibration = root / "calibration.json"
    matrices = load_calibration(calibration) if calibration is not None else {}
    names = matrices.get("camera_names")
    requested = image_files if image_files is not None else matrices.get("image_files")
    if requested is not None:
        requested = list(requested)
        if len(requested) != 6:
            raise ValueError("image_files must contain exactly six paths in calibration order")
        paths = [_file(root, value, ".png") for value in requested]
        if names is None:
            names = [path.stem for path in paths]
    else:
        names = list(names or DEFAULT_CAMERA_NAMES)
        available = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".png"]
        if len(available) != 6:
            raise ValueError(f"Expected six PNG files in {root}, found {len(available)}")
        by_stem = {path.stem.casefold(): path for path in available}
        if len(by_stem) != 6 or any(name.casefold() not in by_stem for name in names):
            raise ValueError("Cannot infer camera order. Name PNGs using the six calibration camera_names "
                             "(or DEFAULT_CAMERA_NAMES), or pass --image-files in calibration order")
        paths = [by_stem[name.casefold()].resolve() for name in names]
    if len(set(paths)) != 6:
        raise ValueError("The six cameras must use six distinct PNG files")
    if lidar_file is None:
        candidates = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".bin"]
        if len(candidates) != 1:
            raise ValueError(f"Expected one BIN file in {root}, found {len(candidates)}; pass --lidar-file")
        lidar_path = candidates[0].resolve()
    else:
        lidar_path = _file(root, lidar_file, ".bin")
    sample = {
        "images": read_pngs(paths),
        "points": read_lidar_bin(lidar_path, point_dim=point_dim, fifth_column=fifth_column),
        "camera_names": names,
        "image_paths": [str(path) for path in paths],
        "lidar_path": str(lidar_path),
        "point_format": {"input_columns": point_dim, "fifth_column": fifth_column,
                         "output_columns": ["x", "y", "z", "intensity", "time_lag_seconds"]},
    }
    sample.update({key: matrices[key] for key in ("camera_intrinsics", "camera2lidar") if key in matrices})
    return sample
