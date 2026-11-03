#!/usr/bin/env bash
# Install selected operators as pip packages / 安装所选算子为 pip 包。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_path="${BEVFUSION_PYTHON:-${CONDA_PREFIX:-$HOME/miniconda3/envs/vla-supervised}/bin/python}"
[[ -x "$python_path" ]] || { printf 'Set BEVFUSION_PYTHON to a named Conda Python.\n' >&2; exit 1; }
"$python_path" -c "import sys; from pathlib import Path; p=Path(sys.prefix); ok=(p/'conda-meta').is_dir() and p.name.lower() not in ('base','root','miniconda','miniconda3','anaconda','anaconda3','miniforge3'); sys.exit(0 if ok else 'Select a non-base Conda environment.')"
case "${1:-all}" in
    all) operators=(bev_pool voxel spconv) ;;
    voxel|spconv|bev_pool) operators=("$1") ;;
    *) printf 'Usage: install_ops.sh [all|voxel|spconv|bev_pool]\n' >&2; exit 1 ;;
esac
for operator in "${operators[@]}"; do
    printf 'Installing %s with %s\n' "$operator" "$python_path"
    # Keep the target environment's existing PyTorch and runtime dependencies.
    "$python_path" -m pip install --no-build-isolation --no-deps "$script_dir/../ops/$operator"
done
