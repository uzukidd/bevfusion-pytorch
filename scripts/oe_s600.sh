#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
image="${OE_S600_IMAGE:-registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0}"
if ! command -v docker >/dev/null 2>&1; then
    printf 'Docker is unavailable. Run this script in Linux/WSL with Docker configured.\n' >&2
    exit 1
fi
if ! docker info >/dev/null; then
    printf 'Docker is inaccessible. Start its service and check Docker socket permissions.\n' >&2
    exit 1
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
    printf 'OE image is not installed: %s\nObtain the matching image using the official S600 toolchain instructions.\n' "$image" >&2
    exit 1
fi
terminal_flags=()
if [[ -t 0 && -t 1 ]]; then
    terminal_flags=(-it)
fi
if [[ $# -eq 0 ]]; then
    set -- bash
fi
exec docker run --rm "${terminal_flags[@]}" \
    --mount "type=bind,source=$project_dir,target=/workspace" \
    --workdir /workspace "$image" "$@"
