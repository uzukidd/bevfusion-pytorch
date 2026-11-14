#!/usr/bin/env bash
# Export four fallback LiDAR stages using the existing WSL/Linux Conda environment.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_path="${BEVFUSION_PYTHON:-$HOME/miniconda3/envs/bevfusion-pytorch/bin/python}"
if [[ ! -x "$python_path" || ! -d "$(dirname -- "$(dirname -- "$python_path")")/conda-meta" ]]; then
    printf 'Set BEVFUSION_PYTHON to an existing non-base Conda interpreter.\n' >&2
    exit 1
fi
exec "$python_path" "$script_dir/../tools/export_hybrid_stages.py" "$@"
