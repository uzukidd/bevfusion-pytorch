#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_path="${BEVFUSION_PYTHON:-$HOME/miniconda3/envs/bevfusion-pytorch/bin/python}"
if [[ ! -x "$python_path" || ! -d "$(dirname -- "$(dirname -- "$python_path")")/conda-meta" ]]; then
    printf 'Set BEVFUSION_PYTHON to an existing non-base Linux Conda interpreter with Horizon plugin.\n' >&2
    exit 1
fi
loader_tunables="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.rtld.execstack=2"
exec env "GLIBC_TUNABLES=$loader_tunables" "$python_path" "$script_dir/../tools/probe_hybrid_gather.py" "$@"
