#!/usr/bin/env bash
set -euo pipefail

# macOS-friendly installer used by "Install Leonardo.command".
# It handles the two tasks that usually block non-computational users:
# finding/installing Conda and choosing the correct Leonardo install mode.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/install_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Leonardo macOS installer"
echo "Repository: $ROOT_DIR"
echo "Log file:   $LOG_FILE"
echo

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: this double-click installer is for macOS."
  echo "Use ./scripts/install_leonardo.sh on Linux or another Unix shell."
  exit 1
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

install_miniforge() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to download Miniforge automatically."
    return 1
  fi

  local arch installer url target tmp answer
  arch="$(uname -m)"
  case "$arch" in
    arm64)
      installer="Miniforge3-MacOSX-arm64.sh"
      ;;
    x86_64)
      installer="Miniforge3-MacOSX-x86_64.sh"
      ;;
    *)
      echo "ERROR: unsupported macOS architecture: $arch"
      return 1
      ;;
  esac

  target="${LEONARDO_CONDA_PREFIX:-$HOME/miniforge3}"
  if [[ -e "$target" && ! -f "$target/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: $target exists but does not look like a Conda installation."
    return 1
  fi

  if [[ -f "$target/etc/profile.d/conda.sh" ]]; then
    export PATH="$target/bin:$PATH"
    # shellcheck disable=SC1091
    source "$target/etc/profile.d/conda.sh"
    return 0
  fi

  echo "Conda was not found."
  echo "Leonardo can install Miniforge into:"
  echo "  $target"
  read -r -p "Install Miniforge there now? [Y/n] " answer
  answer="${answer:-Y}"
  case "$answer" in
    Y|y|YES|Yes|yes)
      ;;
    *)
      echo "Install cancelled. Install Miniforge/Miniconda manually, then rerun this installer."
      return 1
      ;;
  esac

  url="https://github.com/conda-forge/miniforge/releases/latest/download/$installer"
  tmp="${TMPDIR:-/tmp}/$installer"
  echo "Downloading: $url"
  curl -L --fail "$url" -o "$tmp"
  echo "Installing Miniforge into: $target"
  bash "$tmp" -b -p "$target"
  export PATH="$target/bin:$PATH"
  # shellcheck disable=SC1091
  source "$target/etc/profile.d/conda.sh"
}

if ! source_conda; then
  install_miniforge
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: Conda is still unavailable after setup."
  exit 1
fi

echo "Conda executable: $(command -v conda)"
conda --version
echo

echo "Installing Leonardo with locked versions first."
echo "If the locked environment is not compatible with this Mac, the installer will offer a flexible solve."
echo

if "$ROOT_DIR/scripts/install_leonardo.sh" --locked; then
  echo
  echo "Leonardo locked installation completed."
else
  echo
  echo "The locked installation failed on this host."
  read -r -p "Try the flexible environments/environment.yml installation instead? [Y/n] " answer
  answer="${answer:-Y}"
  case "$answer" in
    Y|y|YES|Yes|yes)
      "$ROOT_DIR/scripts/install_leonardo.sh" --flexible
      ;;
    *)
      echo "Install stopped after locked-environment failure."
      exit 1
      ;;
  esac
fi

cat <<EOF

Leonardo is installed.

Double click this file to launch:
  $ROOT_DIR/installers/shortcuts/Launch Leonardo.command

Inside PyMOL open:
  Plugins > Leonardo

Install log:
  $LOG_FILE
EOF
