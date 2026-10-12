#!/usr/bin/env bash
# Double-click installer for macOS (published with every Leonardo release).
# The first time, macOS asks for confirmation: right-click (or Control-click) this file,
# choose "Open", then "Open" again.
clear
cat <<'EOF'
Leonardo installer for macOS
============================

This window installs Leonardo and everything it needs (PyMOL, OpenMM, AmberTools,
Psi4, MOPAC, MODELLER, ...). It takes 20-60 minutes and about 10 GB of disk.
Keep the Mac connected to power and the internet. You can use the Mac meanwhile.

EOF
VERSION="@LEONARDO_VERSION@"
if [[ "$VERSION" == @*@ ]]; then
  url="https://github.com/juvenalyosa/Leonardo/releases/latest/download/install_leonardo.sh"
else
  url="https://github.com/juvenalyosa/Leonardo/releases/download/$VERSION/install_leonardo.sh"
fi
script="$(mktemp -d)/install_leonardo.sh"
if curl -fsSL --retry 3 "$url" -o "$script"; then
  bash "$script"
  status=$?
else
  echo "Could not download the installer. Check the internet connection and try again."
  status=1
fi
echo
if [[ "$status" -eq 0 ]]; then
  echo "Done. Open 'Leonardo' from your Desktop or Applications."
else
  echo "The installation stopped (see the messages above). Running this file again continues it."
fi
read -r -p "Press Return to close this window. " _
exit "$status"
