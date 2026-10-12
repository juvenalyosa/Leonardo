#!/usr/bin/env bash
# Kept for existing links and shortcuts: the installer for macOS and Linux is now
# installers/install_leonardo.sh (installs conda, the environment, Leonardo, the PyMOL loader,
# the MODELLER license and the Leonardo shortcuts). This runs it on this source folder.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT_DIR/installers/install_leonardo.sh" --source "$ROOT_DIR" "$@"
