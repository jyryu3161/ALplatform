#!/usr/bin/env bash
# Install the tested Linux environment without conda or a system Python.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
uv_version=0.11.7
uv_sha256=6681d691eb7f9c00ac6a3af54252f7ab29ae72f0c8f95bdc7f9d1401c23ea868
python_version=3.11.13
venv_dir="$repo_dir/.venv-ald"
torch_backend=cpu
offline=false
verify_args=()
bootstrap_dir=""

usage() {
  cat <<'EOF'
Usage: bash install.sh [options]

Install all seven models, docking, receptor preparation, tuning and test tools.
Requires Linux x86_64 with glibc >=2.28. CPU is the default.

  --venv PATH                 Environment directory (default: repo/.venv-ald)
  --torch-backend cpu|cu128   PyTorch 2.8.0 build; cu128 requires usable GPU 0/1
  --test                      Also run the full suite with real docking tests
  --test-models               Also test public pretrained weights; implies --test
  --offline                   Use cached uv, Python, packages and model weights
  -h, --help                  Show this help

Every installation checks native imports and runs a small three-model replay.
--test-models downloads about 7.2 GB of public weights on first use
(Nori, TabLDM, TabFM and the 44 MB TabPFN v2 regression checkpoint).
EOF
}

fail() { printf 'Installation failed: %s\n' "$*" >&2; exit 1; }
cleanup() { if [[ -n "$bootstrap_dir" ]]; then rm -rf -- "$bootstrap_dir"; fi; }
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv|--torch-backend)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
      if [[ "$1" == --venv ]]; then venv_dir="$2"; else torch_backend="$2"; fi
      shift 2 ;;
    --test|--test-models) verify_args+=("$1"); shift ;;
    --offline) offline=true; verify_args+=(--offline); shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1 (see --help)" ;;
  esac
done

[[ "$torch_backend" == cpu || "$torch_backend" == cu128 ]] || fail "Choose --torch-backend cpu or cu128"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail "The tested installer requires Linux x86_64 (including WSL2)"
glibc_version="$(getconf GNU_LIBC_VERSION 2>/dev/null)" || fail "glibc >=2.28 is required; musl/Alpine is unsupported"
glibc_version="${glibc_version#glibc }"
IFS=. read -r glibc_major glibc_minor _ <<< "$glibc_version"
[[ "$glibc_major" =~ ^[0-9]+$ && "$glibc_minor" =~ ^[0-9]+$ ]] || fail "Cannot determine glibc version"
(( glibc_major > 2 || (glibc_major == 2 && glibc_minor >= 28) )) || fail "glibc >=2.28 is required (found $glibc_version)"
[[ ! -L "$venv_dir" ]] || fail "Environment path must not be a symlink: $venv_dir"
venv_dir="$(realpath -m -- "$venv_dir")"
[[ "$venv_dir" != *$'\n'* && "$repo_dir" != *$'\n'* ]] || fail "Paths containing newlines are unsupported"
owner_file="$venv_dir/.ald-install-owner"
env_python="$venv_dir/bin/python"
existing=false
if [[ -e "$venv_dir" ]]; then
  [[ -d "$venv_dir" && -f "$owner_file" && -f "$venv_dir/pyvenv.cfg" && -x "$env_python" ]] ||
    fail "Refusing to modify an existing unmanaged/broken environment: $venv_dir. Choose a new --venv path."
  [[ "$(cat -- "$owner_file")" == "$repo_dir" ]] || fail "Environment belongs to a different checkout; choose a new --venv path"
  existing=true
fi

# Ignore activated Python environments and implicit destructive uv settings.
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV UV_SYSTEM_PYTHON UV_VENV_CLEAR UV_VENV_SEED
unset UV_CONSTRAINT UV_OVERRIDE UV_EXCLUDE UV_REQUIRE_HASHES UV_NO_VERIFY_HASHES
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export UV_CONCURRENT_BUILDS=2 UV_CONCURRENT_INSTALLS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=2 MAKEFLAGS=-j2
uv_args=(--no-config)
if "$offline"; then uv_args+=(--offline); fi

check_environment() {
  "$env_python" -I - "$python_version" "$venv_dir" <<'PY'
import pathlib
import sys

expected, target = sys.argv[1:]
if sys.version.split()[0] != expected or pathlib.Path(sys.prefix) != pathlib.Path(target):
    raise SystemExit(f"Expected isolated Python {expected} at {target}; choose a new --venv path")
configuration = (pathlib.Path(target) / "pyvenv.cfg").read_text().lower()
if "include-system-site-packages = false" not in configuration or sys.prefix == sys.base_prefix:
    raise SystemExit("System-site Python environments are unsupported; choose a new --venv path")
if (pathlib.Path(sys.base_prefix) / "conda-meta").exists():
    raise SystemExit("Use a uv-managed Python environment, not a conda interpreter")
PY
}
if "$existing"; then check_environment; fi

uv_cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/active-learning-docking/uv/$uv_version"
matching_uv() {
  local reported
  reported="$("$1" --version)" || return 1
  [[ "$reported" == "uv $uv_version" || "$reported" == "uv $uv_version "* ]]
}
uv_bin="$(command -v uv || true)"
if [[ -n "$uv_bin" ]] && matching_uv "$uv_bin"; then
  uv_bin="$(realpath -- "$uv_bin")"
elif [[ -x "$uv_cache_dir/uv" ]] && matching_uv "$uv_cache_dir/uv"; then
  uv_bin="$uv_cache_dir/uv"
else
  "$offline" && fail "uv $uv_version is not available locally; bootstrap once without --offline"
  command -v git >/dev/null || fail "git is required for the pinned Xiaomi TabLDM package"
  command -v sha256sum >/dev/null || fail "sha256sum is required to verify uv"
  bootstrap_dir="$(mktemp -d)"
  archive="$bootstrap_dir/uv.tar.gz"
  url="https://github.com/astral-sh/uv/releases/download/$uv_version/uv-x86_64-unknown-linux-gnu.tar.gz"
  printf 'Downloading verified uv %s...\n' "$uv_version"
  if command -v curl >/dev/null; then
    curl --fail --location --silent --show-error --retry 2 --connect-timeout 20 \
      --speed-limit 1024 --speed-time 60 --max-time 600 "$url" --output "$archive"
  elif command -v wget >/dev/null; then
    wget --https-only --tries=3 --timeout=60 --quiet --output-document="$archive" "$url"
  else
    fail "Install curl or wget to bootstrap uv"
  fi
  printf '%s  %s\n' "$uv_sha256" "$archive" | sha256sum --check --status || fail "uv archive SHA256 mismatch"
  tar -xzf "$archive" -C "$bootstrap_dir"
  mkdir -p -- "$uv_cache_dir"
  install -m 755 "$bootstrap_dir/uv-x86_64-unknown-linux-gnu/uv" "$uv_cache_dir/uv"
  uv_bin="$uv_cache_dir/uv"
fi
command -v git >/dev/null || fail "git is required for the pinned Xiaomi TabLDM package"

cd -- "$repo_dir"
lock_file="$repo_dir/requirements/install-$torch_backend.lock"
[[ -f "$lock_file" ]] || fail "Missing lock file: $lock_file"
printf 'Using uv %s, Python %s and PyTorch backend %s\n' "$uv_version" "$python_version" "$torch_backend"
if ! "$existing"; then
  "$uv_bin" "${uv_args[@]}" python install --no-bin "$python_version"
  "$uv_bin" "${uv_args[@]}" venv --no-project --managed-python --python "$python_version" "$venv_dir"
  printf '%s\n' "$repo_dir" > "$owner_file"
  check_environment
fi

# Seed locked build backends before building editable/pure-Python sources.
build_tools=()
while IFS= read -r requirement; do
  case "$requirement" in
    setuptools==*|wheel==*|packaging==*|hatchling==*|pathspec==*|pluggy==*|tomlkit==*|trove-classifiers==*)
      build_tools+=("$requirement") ;;
  esac
done < "$lock_file"
[[ ${#build_tools[@]} -eq 8 ]] || fail "Lock is missing pinned build backends"
"$uv_bin" "${uv_args[@]}" pip install --python "$env_python" --no-deps "${build_tools[@]}"
"$uv_bin" "${uv_args[@]}" pip install --python "$env_python" --no-deps --no-build-isolation \
  --torch-backend "$torch_backend" \
  --only-binary vina,openmm,gemmi,torch,numpy,scipy,scikit-learn,rdkit \
  --requirement "$lock_file"
"$uv_bin" "${uv_args[@]}" pip check --python "$env_python"
"$uv_bin" "${uv_args[@]}" pip freeze --python "$env_python" > "$venv_dir/installed-freeze.txt"
"$env_python" "$repo_dir/scripts/verify_install.py" --torch-backend "$torch_backend" "${verify_args[@]}"
printf '\nInstallation and verification succeeded. Activate with:\n  source %q/bin/activate\n' "$venv_dir"
