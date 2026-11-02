#!/usr/bin/env bash
# Compare standalone and legacy assembly / 新接口与旧流程对照。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_python.sh" python scripts/verify_bevfusion_minimal.py "$@"
