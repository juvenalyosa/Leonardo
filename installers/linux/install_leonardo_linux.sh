#!/usr/bin/env bash
set -euo pipefail

# Linux installer used by "Install Leonardo Linux.sh".
# It mirrors the macOS flow: find or install Miniforge, then run the main
# Leonardo installer with the locked environment first and a flexible fallback.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/install_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install_linux_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Leonardo Linux installer"
echo "Repository: $ROOT_DIR"
echo "Log file:   $LOG_FILE"
echo

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: this installer is for Linux. Use the macOS or Windows installer for other systems."
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
    "/opt/conda" \
    "/opt/miniforge3" \
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
    x86_64|amd64)
      installer="Miniforge3-Linux-x86_64.sh"
      ;;
    aarch64|arm64)
      installer="Miniforge3-Linux-aarch64.sh"
      ;;
    *)
      echo "ERROR: unsupported Linux architecture: $arch"
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
echo "If the locked environment is not compatible with this Linux host, the installer will offer a flexible solve."
echo

if "$ROOT_DIR/install_leonardo.sh" --locked; then
  echo
  echo "Leonardo locked installation completed."
else
  echo
  echo "The locked installation failed on this host."
  read -r -p "Try the flexible environment.yml installation instead? [Y/n] " answer
  answer="${answer:-Y}"
  case "$answer" in
    Y|y|YES|Yes|yes)
      "$ROOT_DIR/install_leonardo.sh" --flexible
      ;;
    *)
      echo "Install stopped after locked-environment failure."
      exit 1
      ;;
  esac
fi

cat <<EOF

Leonardo is installed.

Launch with:
  $ROOT_DIR/launch_leonardo.sh

Inside PyMOL open:
  Plugins > Leonardo

Install log:
  $LOG_FILE
EOF
