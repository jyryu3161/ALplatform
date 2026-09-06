#!/usr/bin/env bash
# Backwards-compatible entry point for the unified uv installer.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$repo_dir/install.sh" "$@"
