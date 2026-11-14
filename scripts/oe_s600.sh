#!/usr/bin/env bash
# Run toolchain commands in the selected Conda environment without Docker.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
python_path="${BEVFUSION_HBM_PYTHON:-${BEVFUSION_WSL_PYTHON:-/home/administrator/miniconda3/envs/bevfusion-pytorch/bin/python}}"
if [[ ! -x "$python_path" || ! -d "$(dirname -- "$(dirname -- "$python_path")")/conda-meta" ]]; then
    printf 'Set BEVFUSION_HBM_PYTHON to the bevfusion-pytorch Conda interpreter.\n' >&2
    exit 1
fi
if [[ -n "${OE_S600_IMAGE:-}" ]]; then
    printf 'OE_S600_IMAGE is no longer supported; commands run in bevfusion-pytorch.\n' >&2
    exit 1
fi
compiler_bin="$(dirname -- "$python_path")"
if [[ $# -eq 0 ]]; then
    set -- bash --noprofile --norc
fi
loader_tunables="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.rtld.execstack=2"
cd -- "$project_dir"
exec env "PATH=$compiler_bin:$PATH" "CONDA_PREFIX=$(dirname -- "$compiler_bin")" \
    "GLIBC_TUNABLES=$loader_tunables" "$@"
