#!/usr/bin/env bash
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clear

cat <<'EOF'
Launching Leonardo
==================

PyMOL will open from the Leonardo conda environment.  Inside PyMOL, use:
  Plugins > Leonardo

EOF

"$ROOT_DIR/launch_leonardo.sh"
status=$?

echo
if [[ "$status" -ne 0 ]]; then
  echo "Launch failed with exit code $status."
  echo "If Leonardo was not installed yet, double click 'Install Leonardo.command' first."
  echo
  read -r -p "Press Return to close this window. "
fi

exit "$status"
