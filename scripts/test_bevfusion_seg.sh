#!/usr/bin/env bash
# Segmentation inference using an existing non-base Conda environment.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."
entry=test_seg.py
env_name="${BEVFUSION_CONDA_ENV:-bevfusion-pytorch}"
case "${env_name,,}" in
    base|root|default|*/*|*\\*) printf 'Select an existing non-base named Conda environment.\n' >&2; exit 1 ;;
esac
run_python() {
    local python_path="$1"
    shift
    if [[ "$python_path" != /* || ! -x "$python_path" ]]; then
        printf 'BEVFUSION_PYTHON must be an absolute executable path.\n' >&2
        exit 1
    fi
    "$python_path" -c "import pathlib,sys; p=pathlib.Path(sys.prefix); sys.exit(0 if (p/'conda-meta').is_dir() and not (p/'pkgs').is_dir() and p.name.lower() not in ('base','root','default') else 1)" || {
        printf 'Select an existing non-base Conda Python.\n' >&2; exit 1;
    }
    exec "$python_path" "$entry" "$@"
}
if [[ -n "${BEVFUSION_PYTHON:-}" ]]; then
    run_python "$BEVFUSION_PYTHON" "$@"
fi
if [[ -z "${BEVFUSION_CONDA_ENV:-}" && -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-base}" != base && "${CONDA_DEFAULT_ENV:-root}" != root && "${CONDA_DEFAULT_ENV:-default}" != default && -x "$CONDA_PREFIX/bin/python" ]]; then
    run_python "$CONDA_PREFIX/bin/python" "$@"
fi
for root in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/miniconda3 /opt/conda; do
    if [[ -x "$root/envs/$env_name/bin/python" && -d "$root/envs/$env_name/conda-meta" ]]; then
        run_python "$root/envs/$env_name/bin/python" "$@"
    fi
done
conda_exe="${CONDA_EXE:-$(type -P conda || true)}"
if [[ -n "$conda_exe" ]]; then
    exec "$conda_exe" run --no-capture-output -n "$env_name" python "$entry" "$@"
fi
printf 'Named Conda environment not found. Set BEVFUSION_PYTHON or BEVFUSION_CONDA_ENV.\n' >&2
exit 1
