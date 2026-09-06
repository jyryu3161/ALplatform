#!/usr/bin/env bash
# Source this script after creating .venv; it does not alter shell startup files.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$repo_dir/.venv/bin/activate" ]]; then
  echo 'Create the environment first: python -m venv .venv' >&2
  return 1 2>/dev/null || exit 1
fi
source "$repo_dir/.venv/bin/activate"
