#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
exec bash "$project_root/scripts/run_python.sh" python "$project_root/scripts/train_bevfusion_surface_head.py" "$@"
