#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Install the local random-walk stack into this repo's .venv.

By default this script:
  1. Rebuilds random-walks into build/lib/librandom_walk.so.
  2. Copies that shared library into randomwalks/.
  3. Editable-installs sibling CMA repos with --no-deps.

Options:
  --install-self              Also install this repo in editable mode.
  --skip-random-walks-build   Do not run cmake --build or copy librandom_walk.so.
  --packages "a b c"          Override sibling package directory names.
  --venv PATH                 Use a different virtualenv path.
  -h, --help                  Show this help.
USAGE
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
parent_dir="$(dirname -- "$repo_root")"
venv_dir="$repo_root/.venv"
build_random_walks=1
install_self=0
packages=(environmentcma hmmcma segmentationcma kernelcma)

echo "Fetch submodule updates" >&2
git submodule update --init --recursive --remote --merge

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-self)
            install_self=1
            shift
            ;;
        --skip-random-walks-build)
            build_random_walks=0
            shift
            ;;
        --packages)
            if [[ $# -lt 2 ]]; then
                echo "error: --packages expects a quoted, space-separated list" >&2
                exit 2
            fi
            read -r -a packages <<< "$2"
            shift 2
            ;;
        --venv)
            if [[ $# -lt 2 ]]; then
                echo "error: --venv expects a path" >&2
                exit 2
            fi
            venv_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

python_bin="$venv_dir/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "error: expected virtualenv python at $python_bin" >&2
    exit 1
fi

ensure_module() {
    local module="$1"
    local requirement="$2"

    if ! "$python_bin" -c "import ${module}" >/dev/null 2>&1; then
        "$python_bin" -m pip install "$requirement"
    fi
}

ensure_module setuptools "setuptools>=64"
ensure_module wheel wheel
# Native sibling packages (currently hmmcma) also use scikit-build-core and
# pybind11. These must be present because editable installs below deliberately
# disable build isolation.
ensure_module scikit_build_core "scikit-build-core>=0.10"
ensure_module pybind11 "pybind11>=2.13"

if [[ "$build_random_walks" -eq 1 ]]; then
    cmake --build "$repo_root/build" --target random_walk

    shared_lib="$repo_root/build/lib/librandom_walk.so"
    if [[ ! -f "$shared_lib" ]]; then
        echo "error: build finished, but $shared_lib was not found" >&2
        exit 1
    fi

    cp "$shared_lib" "$repo_root/randomwalks/"
    echo "moved shared lib to randomwalks package root" >&2
fi

if [[ "$install_self" -eq 1 ]]; then
    "$python_bin" -m pip install --no-deps --no-build-isolation --editable "$repo_root"
fi

for package in "${packages[@]}"; do
    package_dir="$parent_dir/$package"
    if [[ ! -f "$package_dir/pyproject.toml" ]]; then
        echo "error: expected package checkout with pyproject.toml at $package_dir" >&2
        exit 1
    fi

    "$python_bin" -m pip install --no-deps --no-build-isolation --editable "$package_dir"
done

echo "Installed local editable packages into $venv_dir"
