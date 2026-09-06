#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -m venv "$repo_dir/.venv"
"$repo_dir/.venv/bin/python" -m pip install -e "${repo_dir}[docking,tuning,dev]"
echo "Activate with: source $repo_dir/.venv/bin/activate"
