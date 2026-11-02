#!/usr/bin/env bash
# Minimal inference + Matplotlib / 最小推理与 Matplotlib 可视化。
# Reuse an existing named Conda environment; never use base/system Python.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
env_name="${BEVFUSION_CONDA_ENV:-vla-supervised}"
case "${env_name,,}" in
    base|root|default|*/*|*\\*) printf 'Select an existing non-base Conda environment.\n' >&2; exit 1 ;;
esac
if [[ -n "${BEVFUSION_PYTHON:-}" ]]; then
    python_path="$BEVFUSION_PYTHON"
    prefix="$(dirname -- "$(dirname -- "$python_path")")"
    if [[ "$python_path" != /* || ! -x "$python_path" || ! -d "$prefix/conda-meta" || "${prefix##*/}" != "$env_name" ]]; then
        printf 'BEVFUSION_PYTHON must point to the selected named Conda environment.\n' >&2
        exit 1
    fi
    exec "$python_path" test.py "$@"
fi
for root in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/miniconda3 /opt/conda; do
    prefix="$root/envs/$env_name"
    if [[ -x "$prefix/bin/python" && -d "$prefix/conda-meta" ]]; then
        exec "$prefix/bin/python" test.py "$@"
    fi
done
conda_exe="${CONDA_EXE:-$(type -P conda || true)}"
if [[ -n "$conda_exe" ]]; then
    exec "$conda_exe" run --no-capture-output -n "$env_name" python test.py "$@"
fi
printf 'Named Conda environment not found. Set BEVFUSION_PYTHON to its absolute Python path.\n' >&2
exit 1
