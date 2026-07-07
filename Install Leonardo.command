#!/usr/bin/env bash
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clear

cat <<'EOF'
Leonardo installer
==================

This window will install the conda environment, Leonardo source package,
PyMOL startup loader and PyMOL plugin ZIP.

The first installation can take a long time because Conda must download
scientific packages such as PyMOL, OpenMM, AmberTools, Psi4, RDKit and MOPAC.

EOF

"$ROOT_DIR/installers/macos/install_leonardo_macos.sh"
status=$?

echo
if [[ "$status" -eq 0 ]]; then
  echo "Installation finished successfully."
  echo "Next: double click 'Launch Leonardo.command'."
else
  echo "Installation failed with exit code $status."
  echo "Check the newest log in:"
  echo "  $ROOT_DIR/install_logs"
fi

echo
read -r -p "Press Return to close this window. "
exit "$status"
