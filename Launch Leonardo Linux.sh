#!/usr/bin/env bash
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v clear >/dev/null 2>&1; then
  clear
fi

cat <<'EOF'
Launching Leonardo on Linux
===========================

PyMOL will open from the Leonardo conda environment.  Inside PyMOL, use:
  Plugins > Leonardo

EOF

"$ROOT_DIR/launch_leonardo.sh"
status=$?

echo
if [[ "$status" -ne 0 ]]; then
  echo "Launch failed with exit code $status."
  echo "If Leonardo was not installed yet, run 'Install Leonardo Linux.sh' first."
  echo
  if [[ -t 0 ]]; then
    read -r -p "Press Return to close this window. "
  fi
fi

exit "$status"
