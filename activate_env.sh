#!/usr/bin/env bash
# Source after install.sh; an optional argument selects another environment.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ald_env_dir="${1:-$repo_dir/.venv-ald}"
if [[ $# -eq 0 && ! -f "$ald_env_dir/bin/activate" && -f "$repo_dir/.venv/bin/activate" ]]; then
  ald_env_dir="$repo_dir/.venv"
fi
if [[ ! -f "$ald_env_dir/bin/activate" ]]; then
  echo 'Create the environment first: bash install.sh' >&2
  return 1 2>/dev/null || exit 1
fi
source "$ald_env_dir/bin/activate"
unset ald_env_dir
