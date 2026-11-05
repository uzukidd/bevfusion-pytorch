#!/usr/bin/env bash
# Native Linux/macOS optional entry; Windows uses export_onnx.bat directly.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_path="${BEVFUSION_PYTHON:-$HOME/miniconda3/envs/vla-supervised/bin/python}"
if [[ ! -x "$python_path" || ! -d "$(dirname -- "$(dirname -- "$python_path")")/conda-meta" ]]; then
    printf 'Set BEVFUSION_PYTHON to an existing non-base Conda interpreter.\n' >&2
    exit 1
fi
exec "$python_path" "$script_dir/../tools/export_onnx.py" "$@"
