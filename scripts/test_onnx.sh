#!/usr/bin/env bash
# ONNX Runtime inference / ONNX Runtime 推理。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_path="${BEVFUSION_PYTHON:-$HOME/miniconda3/envs/vla-supervised/bin/python}"
prefix="$(dirname -- "$(dirname -- "$python_path")")"
if [[ ! -x "$python_path" || ! -d "$prefix/conda-meta" || "${prefix##*/}" == base || "${prefix##*/}" == miniconda3 ]]; then
    printf 'Set BEVFUSION_PYTHON to an existing non-base Conda interpreter.\n' >&2
    exit 1
fi
exec "$python_path" "$script_dir/../test_onnx.py" "$@"
