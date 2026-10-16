#!/usr/bin/env bash
# Build dist/Leonardo-Installer-<version>.zip: a self-contained installer for macOS and Linux.
# It holds the Leonardo source (git archive of HEAD) and double-click/one-command launchers that
# run installers/install_leonardo.sh on it. Only the scientific packages are downloaded (public
# conda channels), so it works while the GitHub repository is private.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
version="$(awk -F'"' '/^version *=/{print $2; exit}' pyproject.toml)"
name="Leonardo-Installer-${version}"
out_dir="$ROOT_DIR/dist"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
bundle="$work/$name"
mkdir -p "$bundle/leonardo" "$out_dir"

git archive --format=tar HEAD | tar -x -C "$bundle/leonardo"

cat > "$bundle/Install Leonardo (macOS).command" <<'EOF'
#!/usr/bin/env bash
# The first time, macOS asks for confirmation: right-click this file > Open > Open.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clear
echo "Leonardo installer for macOS"
echo "============================"
echo
echo "This installs Leonardo and everything it needs. It takes 20-60 minutes and about 10 GB."
echo "Keep the Mac on power and connected to the internet."
echo
bash "$here/leonardo/installers/install_leonardo.sh" --source "$here/leonardo" --copy
status=$?
echo
[[ "$status" -eq 0 ]] && echo "Done. Open 'Leonardo' from your Desktop or Applications." \
                      || echo "The installation stopped (see above). Double-click this file again to continue."
read -r -p "Press Return to close this window. " _
exit "$status"
EOF

cat > "$bundle/install-leonardo-linux.sh" <<'EOF'
#!/usr/bin/env bash
# Linux: open a terminal in this folder and run:  bash install-leonardo-linux.sh
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Leonardo installer for Linux (20-60 minutes, about 10 GB)"
bash "$here/leonardo/installers/install_leonardo.sh" --source "$here/leonardo" --copy "$@"
EOF

cat > "$bundle/READ ME FIRST - LEEME PRIMERO.txt" <<EOF
Leonardo ${version} - installation / instalación
================================================

ENGLISH

macOS (Apple Silicon or Intel)
  1. Double-click "Install Leonardo (macOS).command".
     The first time macOS blocks it: right-click (Control-click) the file, choose "Open",
     then "Open" again.
  2. A Terminal window installs everything (20-60 minutes, about 10 GB). When it asks for
     the MODELLER key, paste it or just press Return to add it later.
  3. Open "Leonardo" from your Desktop or Applications, then in PyMOL: Plugins > Leonardo.

Linux (x86_64, e.g. Ubuntu, Fedora)
  1. Open a terminal in this folder (right-click > "Open in Terminal").
  2. Run:   bash install-leonardo-linux.sh
  3. Open "Leonardo" from the applications menu or the Desktop.

MODELLER license (homology modeling)
  Free for academic use: https://salilab.org/modeller/registration.html
  The key arrives by e-mail. Paste it during the installation, or later open the
  "Leonardo MODELLER License" app (macOS) / menu entry (Linux).

After installing, this folder can be deleted. To update, run the installer of the new version.


ESPAÑOL

macOS (Apple Silicon o Intel)
  1. Haga doble clic en "Install Leonardo (macOS).command".
     La primera vez macOS lo bloquea: clic derecho (Control-clic) sobre el archivo, elija
     "Abrir" y luego "Abrir" otra vez.
  2. Una ventana de Terminal instala todo (20-60 minutos, unos 10 GB). Cuando pida la clave
     de MODELLER, péguela o pulse Return para agregarla después.
  3. Abra "Leonardo" desde el Escritorio o Aplicaciones y en PyMOL: Plugins > Leonardo.

Linux (x86_64, p. ej. Ubuntu, Fedora)
  1. Abra una terminal en esta carpeta (clic derecho > "Abrir en terminal").
  2. Ejecute:   bash install-leonardo-linux.sh
  3. Abra "Leonardo" desde el menú de aplicaciones o el Escritorio.

Licencia de MODELLER (modelado por homología)
  Gratuita para uso académico: https://salilab.org/modeller/registration.html
  La clave llega por correo. Péguela durante la instalación, o después abra la app
  "Leonardo MODELLER License" (macOS) / la entrada del menú (Linux).

Después de instalar puede borrar esta carpeta. Para actualizar, ejecute el instalador de la
nueva versión.

------------------------------------------------------------------------------------------
Leonardo is built using PyMOL(TM)*: it installs Open-Source PyMOL (Copyright (C) Schrodinger,
LLC) unmodified from conda-forge and runs as a plugin in it. Leonardo is an independent project,
not a PyMOL product distributed by Schrodinger, LLC. PyMOL's copyright and trademark notices:
leonardo/THIRD_PARTY_LICENSES.md.
* PyMOL is a trademark of Schrodinger, LLC.
EOF

chmod +x "$bundle/Install Leonardo (macOS).command" "$bundle/install-leonardo-linux.sh"
rm -f "$out_dir/$name.zip"
( cd "$work" && zip -qry "$out_dir/$name.zip" "$name" )
( cd "$out_dir" && shasum -a 256 "$name.zip" > "$name.zip.sha256" )
echo "Created $out_dir/$name.zip ($(du -h "$out_dir/$name.zip" | cut -f1))"
