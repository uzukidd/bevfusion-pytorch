#!/usr/bin/env bash
# Use the supported Linux Horizon-plugin Conda environment; changes affect this process only.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_path="${BEVFUSION_PYTHON:-$HOME/miniconda3/envs/bevfusion-pytorch/bin/python}"
if [[ ! -x "$python_path" || ! -d "$(dirname -- "$(dirname -- "$python_path")")/conda-meta" ]]; then
    printf 'Set BEVFUSION_PYTHON to an existing non-base Linux Conda interpreter with Horizon plugin.\n' >&2
    exit 1
fi
# New glibc versions require this loader option for the bundled plugin's executable-stack ELF flag.
loader_tunables="${GLIBC_TUNABLES:+$GLIBC_TUNABLES:}glibc.rtld.execstack=2"
exec env "GLIBC_TUNABLES=$loader_tunables" "$python_path" "$script_dir/../tools/export_hat_plugin_camera.py" "$@"
