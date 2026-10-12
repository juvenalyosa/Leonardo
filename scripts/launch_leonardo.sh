#!/usr/bin/env bash
set -euo pipefail

# Launch PyMOL from Leonardo's conda environment and make sure the startup
# loader is present.  This is the command-line equivalent of
# "Launch Leonardo.command".

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/launch_leonardo.sh [pymol arguments]

The launcher:
  1. finds Conda/Miniforge/Miniconda,
  2. checks that protliginteract-full exists,
  3. installs the PyMOL startup loader for Leonardo,
  4. starts PyMOL from the correct environment.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  local prefix
  for prefix in \
    "$HOME/miniforge3" \
    "$HOME/mambaforge" \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "/opt/homebrew/Caskroom/miniforge/base" \
    "/opt/anaconda3"; do
    if [[ -f "$prefix/etc/profile.d/conda.sh" ]]; then
      export PATH="$prefix/bin:$PATH"
      # shellcheck disable=SC1091
      source "$prefix/etc/profile.d/conda.sh"
      return 0
    fi
  done

  return 1
}

if ! source_conda; then
  cat >&2 <<'EOF'
ERROR: Conda was not found.

Run "Install Leonardo.command" first, or install Miniforge/Miniconda and then
run ./scripts/install_leonardo.sh.
EOF
  exit 1
fi

env_name="$(awk -F': *' '/^name:/{print $2; exit}' environments/environment.yml)"
if [[ -z "${env_name:-}" ]]; then
  echo "ERROR: could not parse environment name from environments/environment.yml" >&2
  exit 1
fi

if ! conda run -n "$env_name" python -c "import sys; print(sys.prefix)" >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: Conda environment "$env_name" is not installed.

Run "Install Leonardo.command" first.
EOF
  exit 1
fi

"$ROOT_DIR/scripts/install_pymol_loader.sh" --quiet

if [[ "$(uname -s)" == "Darwin" && -z "${PROTLIGINTERACT_OPENMM_PLATFORM:-}" ]]; then
  export PROTLIGINTERACT_OPENMM_PLATFORM=CPU
fi

export PYTHONPATH="$ROOT_DIR/legacy:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec conda run -n "$env_name" --no-capture-output pymol "$@"
