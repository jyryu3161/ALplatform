#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_path="${1:-$repo_dir/examples/replay.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
python -m active_learning_docking run --config "$config_path" "$@"
