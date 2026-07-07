#!/usr/bin/env bash
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v clear >/dev/null 2>&1; then
  clear
fi

cat <<'EOF'
Leonardo Linux installer
========================

This installer creates the conda environment, installs Leonardo, registers the
PyMOL startup loader, runs smoke checks and rebuilds the PyMOL plugin ZIP.

The first installation can take a long time because Conda must download
scientific packages such as PyMOL, OpenMM, AmberTools, Psi4, RDKit and MOPAC.

EOF

"$ROOT_DIR/installers/linux/install_leonardo_linux.sh"
status=$?

echo
if [[ "$status" -eq 0 ]]; then
  echo "Installation finished successfully."
  echo "Launch with: $ROOT_DIR/launch_leonardo.sh"
else
  echo "Installation failed with exit code $status."
  echo "Check the newest log in:"
  echo "  $ROOT_DIR/install_logs"
fi

echo
if [[ -t 0 ]]; then
  read -r -p "Press Return to close this window. "
fi
exit "$status"
