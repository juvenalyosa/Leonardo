#!/usr/bin/env bash
# Leonardo one-step installer for macOS and Linux.
#
# For users who do not install scientific software by hand. One command installs:
#   * Miniforge (conda), if no conda is found,
#   * the "protliginteract-full" environment with every third-party program Leonardo uses
#     (open-source PyMOL, OpenMM, AmberTools, Psi4, MOPAC, MODELLER, RDKit, Open Babel,
#     smina, fpocket, PACKMOL-Memgen, BLAST, HMMER, Clustal Omega, MAFFT, ...),
#   * Leonardo itself and its PyMOL loader (Plugin > Leonardo),
#   * the MODELLER license key (asked once; it can be added later),
#   * a "Leonardo" app/menu entry and a "Leonardo MODELLER License" helper.
#
# Usage (Terminal):
#   curl -fsSL https://github.com/juvenalyosa/Leonardo/releases/latest/download/install_leonardo.sh | bash
# or, after downloading this file:
#   bash install_leonardo.sh
#
# Options:
#   --yes                  answer "yes" to every question (unattended install)
#   --version TAG          Leonardo release to install (default: the release this file came with)
#   --source DIR           install from a local Leonardo source tree instead of downloading
#   --copy                 with --source: copy it into ~/.leonardo (installer bundles; the unzipped
#                          folder can then be deleted). Without --copy the folder is used in place.
#   --modeller-key KEY     MODELLER license key (or set KEY_MODELLER)
#   --skip-env             do not create/update the conda environment (repair shortcuts only)
#   --no-shortcuts         do not create the app / menu entries
#   --no-ml                skip the leonardo-ml environment (OpenMM-ML / ANI potentials, ~150 MB)
#   -h, --help             this help
#
# Environment variables: KEY_MODELLER, LEONARDO_HOME (default ~/.leonardo),
# LEONARDO_CONDA_PREFIX (default ~/miniforge3 when conda must be installed).

set -euo pipefail

LEONARDO_REPO="juvenalyosa/Leonardo"
LEONARDO_VERSION_DEFAULT="@LEONARDO_VERSION@"      # filled in by the release workflow
ENV_NAME="protliginteract-full"
MIN_FREE_GB=12

ASSUME_YES=0
VERSION=""
SOURCE_DIR=""
COPY_SOURCE=0
SKIP_ENV=0
SHORTCUTS=1
WITH_ML=1
MODELLER_KEY="${KEY_MODELLER:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y) ASSUME_YES=1; shift ;;
    --version) VERSION="${2:-}"; shift 2 ;;
    --source) SOURCE_DIR="${2:-}"; shift 2 ;;
    --copy) COPY_SOURCE=1; shift ;;
    --modeller-key) MODELLER_KEY="${2:-}"; shift 2 ;;
    --skip-env) SKIP_ENV=1; shift ;;
    --no-shortcuts) SHORTCUTS=0; shift ;;
    --no-ml) WITH_ML=0; shift ;;
    -h|--help)
      if [[ -f "${BASH_SOURCE[0]:-}" ]]; then sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      else echo "See https://github.com/juvenalyosa/Leonardo/blob/main/docs/INSTALL.md"; fi
      exit 0 ;;
    *) echo "Unknown option: $1 (use --help)"; exit 2 ;;
  esac
done

LEONARDO_HOME="${LEONARDO_HOME:-$HOME/.leonardo}"
APP_LINK="$LEONARDO_HOME/app"                    # stable path to the installed source
BIN_DIR="$LEONARDO_HOME/bin"
LOG_DIR="$LEONARDO_HOME/install_logs"
mkdir -p "$LOG_DIR" "$BIN_DIR"
LOG_FILE="$LOG_DIR/install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ------------------------------------------------------------------ helpers
bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { echo; bold "==> $*"; }
warn() { printf '\033[33mWARNING:\033[0m %s\n' "$*"; }
die() {
  printf '\033[31mERROR:\033[0m %s\n' "$*"
  echo
  echo "Nothing you need is lost. Fix the problem above and run the installer again;"
  echo "it continues where it stopped. Full log: $LOG_FILE"
  exit 1
}
# questions are read from the terminal even when the script itself comes through a pipe (curl | bash)
ask_yes() {
  local prompt="$1" answer
  if [[ "$ASSUME_YES" -eq 1 ]]; then return 0; fi
  if [[ ! -r /dev/tty ]]; then return 0; fi
  read -r -p "$prompt [Y/n] " answer </dev/tty || answer=""
  [[ -z "$answer" || "$answer" =~ ^[YySs] ]]
}
ask_text() {
  local prompt="$1" answer=""
  if [[ "$ASSUME_YES" -eq 1 || ! -r /dev/tty ]]; then echo ""; return 0; fi
  read -r -p "$prompt " answer </dev/tty || answer=""
  echo "$answer"
}

OS="$(uname -s)"
ARCH="$(uname -m)"

bold "Leonardo installer"
echo "System:   $OS $ARCH"
echo "Log file: $LOG_FILE"

# ------------------------------------------------------------------ 1. system checks
step "1/7 Checking this computer"
case "$OS/$ARCH" in
  Darwin/arm64|Darwin/x86_64|Linux/x86_64) ;;
  Linux/aarch64|Linux/arm64)
    die "Linux on ARM is not supported: several required programs (MODELLER, Psi4, MOPAC) have no ARM Linux builds." ;;
  *) die "Unsupported system: $OS $ARCH. Leonardo supports macOS (Apple Silicon or Intel) and Linux (x86_64)." ;;
esac
command -v curl >/dev/null 2>&1 || die "curl is required (Linux: sudo apt install curl  or  sudo dnf install curl)."
command -v tar >/dev/null 2>&1 || die "tar is required."

free_gb="$(df -Pk "$HOME" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
echo "Free disk space in your home folder: ${free_gb} GB (Leonardo needs about 10 GB)"
if [[ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]]; then
  warn "less than ${MIN_FREE_GB} GB free; the installation may run out of space."
  ask_yes "Continue anyway?" || die "Free some disk space and run the installer again."
fi

if [[ "$OS" == "Linux" ]]; then
  # PyMOL's window needs the usual desktop graphics libraries (present on desktop installs)
  missing_libs=()
  # ldconfig is in /sbin, often not on a normal user's PATH (Debian/Ubuntu): without it every library looked missing
  LDCONFIG="$(command -v ldconfig 2>/dev/null || true)"
  for candidate in /sbin/ldconfig /usr/sbin/ldconfig; do
    [[ -z "$LDCONFIG" && -x "$candidate" ]] && LDCONFIG="$candidate"
  done
  for lib in libGL.so.1 libEGL.so.1 libxkbcommon-x11.so.0 libfontconfig.so.1 libdbus-1.so.3 libXrender.so.1; do
    [[ -n "$LDCONFIG" ]] || break                    # cannot tell: let PyMOL report a missing library itself
    # no `grep -q`: it stops reading early and, with pipefail, ldconfig's SIGPIPE marks the library missing
    "$LDCONFIG" -p 2>/dev/null | grep -F "$lib" >/dev/null || missing_libs+=("$lib")
  done
  if [[ "${#missing_libs[@]}" -gt 0 ]]; then
    warn "graphics libraries PyMOL needs are missing: ${missing_libs[*]}"
    if command -v apt-get >/dev/null 2>&1; then
      pkgs="libgl1 libegl1 libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 libxrender1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-cursor0"
      echo "They can be installed with:  sudo apt-get install -y $pkgs"
      if ask_yes "Install them now (asks for your password)?"; then
        sudo apt-get update && sudo apt-get install -y $pkgs || warn "could not install them; PyMOL may not open."
      fi
    elif command -v dnf >/dev/null 2>&1; then
      pkgs="mesa-libGL mesa-libEGL libxkbcommon-x11 fontconfig dbus-libs libXrender xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil xcb-util-cursor"
      echo "They can be installed with:  sudo dnf install -y $pkgs"
      if ask_yes "Install them now (asks for your password)?"; then
        sudo dnf install -y $pkgs || warn "could not install them; PyMOL may not open."
      fi
    else
      warn "install your distribution's OpenGL, EGL and xkbcommon-x11 packages before opening Leonardo."
    fi
  fi
fi

# ------------------------------------------------------------------ 2. conda
step "2/7 Finding or installing conda (Miniforge)"
find_conda() {
  local candidate
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null)" && [[ -n "$CONDA_BASE" ]] && return 0
  fi
  for candidate in "${LEONARDO_CONDA_PREFIX:-}" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" \
                   "$HOME/anaconda3" "/opt/homebrew/Caskroom/miniforge/base" "/opt/miniforge3" "/opt/anaconda3"; do
    if [[ -n "$candidate" && -x "$candidate/bin/conda" ]]; then
      CONDA_BASE="$candidate"
      return 0
    fi
  done
  return 1
}
CONDA_BASE=""
if find_conda; then
  echo "Using conda at: $CONDA_BASE"
else
  target="${LEONARDO_CONDA_PREFIX:-$HOME/miniforge3}"
  echo "Conda was not found. Leonardo will install Miniforge (free, about 300 MB) into:"
  echo "  $target"
  ask_yes "Install Miniforge now?" || die "Leonardo needs conda. Install Miniforge from https://conda-forge.org/download/ and run the installer again."
  case "$OS/$ARCH" in
    Darwin/arm64) mf="Miniforge3-MacOSX-arm64.sh" ;;
    Darwin/x86_64) mf="Miniforge3-MacOSX-x86_64.sh" ;;
    Linux/x86_64) mf="Miniforge3-Linux-x86_64.sh" ;;
  esac
  tmp="$(mktemp -d)/$mf"
  curl -fL --retry 3 "https://github.com/conda-forge/miniforge/releases/latest/download/$mf" -o "$tmp" \
    || die "could not download Miniforge (check your internet connection)."
  bash "$tmp" -b -p "$target" || die "the Miniforge installation failed."
  rm -f "$tmp"
  CONDA_BASE="$target"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
CONDA="$CONDA_BASE/bin/conda"
"$CONDA" --version

# ------------------------------------------------------------------ 3. Leonardo source
step "3/7 Getting Leonardo"
if [[ -z "$SOURCE_DIR" ]]; then
  here=""
  if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then     # not when piped (curl | bash)
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
  if [[ -n "$here" && -f "$here/../legacy/code_v2.py" && -z "$VERSION" ]]; then
    SOURCE_DIR="$(cd "$here/.." && pwd)"           # run from a Leonardo checkout
  fi
fi
if [[ -n "$SOURCE_DIR" ]]; then
  [[ -f "$SOURCE_DIR/legacy/code_v2.py" ]] || die "$SOURCE_DIR is not a Leonardo source folder."
  SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
  if [[ "$COPY_SOURCE" -eq 1 ]]; then
    src_version="$(awk -F'"' '/^version *=/{print $2; exit}' "$SOURCE_DIR/pyproject.toml")"
    INSTALL_DIR="$LEONARDO_HOME/app-v${src_version:-local}"
    if [[ "$INSTALL_DIR" != "$SOURCE_DIR" ]]; then
      rm -rf "$INSTALL_DIR.tmp"
      cp -R "$SOURCE_DIR" "$INSTALL_DIR.tmp" || die "could not copy Leonardo into $LEONARDO_HOME."
      rm -rf "$INSTALL_DIR"
      mv "$INSTALL_DIR.tmp" "$INSTALL_DIR"
    fi
    echo "Leonardo ${src_version} copied to: $INSTALL_DIR"
  else
    INSTALL_DIR="$SOURCE_DIR"
    echo "Installing from the local source: $INSTALL_DIR"
  fi
else
  if [[ -z "$VERSION" ]]; then
    VERSION="$LEONARDO_VERSION_DEFAULT"
    if [[ "$VERSION" == @*@ ]]; then              # not stamped by a release: newest release (pre-releases too)
      VERSION="$(curl -fsSL "https://api.github.com/repos/$LEONARDO_REPO/releases?per_page=1" 2>/dev/null \
                 | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1 || true)"
      VERSION="${VERSION:-main}"
    fi
  fi
  INSTALL_DIR="$LEONARDO_HOME/app-$VERSION"
  url="https://github.com/$LEONARDO_REPO/archive/${VERSION}.tar.gz"
  [[ "$VERSION" == v* ]] && url="https://github.com/$LEONARDO_REPO/archive/refs/tags/${VERSION}.tar.gz"
  echo "Downloading Leonardo $VERSION ..."
  tmpdir="$(mktemp -d)"
  curl -fL --retry 3 "$url" -o "$tmpdir/leonardo.tar.gz" \
    || die "could not download Leonardo $VERSION (if the GitHub repository is private, use the Leonardo-Installer zip instead)."
  tar -xzf "$tmpdir/leonardo.tar.gz" -C "$tmpdir" || die "the Leonardo download is damaged; run the installer again."
  extracted="$(find "$tmpdir" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  [[ -f "$extracted/legacy/code_v2.py" ]] || die "the download does not contain Leonardo."
  rm -rf "$INSTALL_DIR"
  mv "$extracted" "$INSTALL_DIR"
  rm -rf "$tmpdir"
  echo "Leonardo files: $INSTALL_DIR"
fi
ln -sfn "$INSTALL_DIR" "$APP_LINK"

# MODELLER key already saved in the existing environment (empty if none or the XXXX placeholder); never printed
saved_modeller_key() {
  # conda run does not pass stdin, and activation scripts print to stdout: the script goes in -c and the
  # key comes back on a marked line
  local code='import glob, re, sys
for path in glob.glob(sys.prefix + "/lib/modeller-*/modlib/modeller/config.py"):
    m = re.search(r"^license\s*=\s*r?[\x27\x22]([^\x27\x22]+)[\x27\x22]", open(path).read(), re.M)
    if m and m.group(1).strip() and m.group(1).strip() != "XXXX":
        print("LEONARDO_SAVED_KEY=" + m.group(1).strip())
        break'
  "$CONDA" run -n "$ENV_NAME" python -c "$code" 2>/dev/null | sed -n 's/^LEONARDO_SAVED_KEY=//p' | head -n 1 || true
}

# ------------------------------------------------------------------ 4. MODELLER key
step "4/7 MODELLER license key"
cat <<'EOF'
MODELLER (homology modeling) is free for academic use but needs a personal license key.
Get one at https://salilab.org/modeller/registration.html (it arrives by e-mail).
You can paste it now, or press Return to skip and add it later with
"Leonardo MODELLER License" (everything else works without it).
EOF
if [[ -z "$MODELLER_KEY" ]]; then
  MODELLER_KEY="$(saved_modeller_key)"           # an update keeps the key of the existing environment
  [[ -n "$MODELLER_KEY" ]] && echo "Using the MODELLER key saved in your Leonardo environment."
fi
if [[ -z "$MODELLER_KEY" ]]; then
  MODELLER_KEY="$(ask_text "MODELLER license key (Return to skip):")"
fi
MODELLER_KEY="$(printf '%s' "$MODELLER_KEY" | tr -d '[:space:]')"
if [[ -n "$MODELLER_KEY" ]]; then
  export KEY_MODELLER="$MODELLER_KEY"            # read by the modeller package when it is installed
  echo "MODELLER key received."
else
  echo "Skipped: MODELLER can be activated later."
fi

# ------------------------------------------------------------------ 5. environment
if [[ "$SKIP_ENV" -eq 0 ]]; then
  step "5/7 Installing the scientific software (20-60 minutes, about 8 GB; keep the computer awake)"
  echo "Environment: $ENV_NAME (PyMOL, OpenMM, AmberTools, Psi4, MOPAC, MODELLER, RDKit, ...)"
  echo "Note: this sets conda's channel_priority to 'strict', which the Leonardo environment needs."
  ( cd "$INSTALL_DIR" && PATH="$CONDA_BASE/bin:$PATH" LEONARDO_WITH_ML="$WITH_ML" bash scripts/install_envs.sh --flexible ) \
    || die "installing the scientific software failed (see the messages above)."
  step "Installing Leonardo into the environment"
  "$CONDA" run -n "$ENV_NAME" python -m pip install --no-deps --no-build-isolation -e "$INSTALL_DIR" \
    || die "could not install Leonardo into the environment."
else
  step "5/7 Skipping the environment (--skip-env)"
fi
# activation scripts can print to stdout under conda run: the prefix comes on a marked line
ENV_PREFIX="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print("LEONARDO_PREFIX=" + sys.prefix)' 2>/dev/null \
  | sed -n 's/^LEONARDO_PREFIX=//p' | head -n 1 || true)"
[[ -n "$ENV_PREFIX" ]] || die "the $ENV_NAME environment is not installed."

# ------------------------------------------------------------------ helper scripts
# Qt 6 (PySide6) when the environment has it (decided once here, not at every start); PyQt5 otherwise
QT_API_LINE="# Qt binding: PyMOL's default"
if "$CONDA" run -n "$ENV_NAME" python -c "import PySide6" >/dev/null 2>&1; then
  QT_API_LINE='export QT_API="${QT_API:-pyside6}"    # Qt 6'
fi
cat > "$BIN_DIR/leonardo" <<EOF
#!/usr/bin/env bash
# Start PyMOL with Leonardo (generated by the Leonardo installer).
export PYTHONPATH="$APP_LINK/legacy:$APP_LINK/src:$APP_LINK\${PYTHONPATH:+:\$PYTHONPATH}"
$QT_API_LINE
# the OpenMM platform is chosen by Leonardo's hardware benchmark (run in a child process); set
# LEONARDO_OPENMM_PLATFORM=CPU to force the CPU
exec "$CONDA" run -n "$ENV_NAME" --no-capture-output pymol "\$@"
EOF

cat > "$BIN_DIR/leonardo-modeller-license" <<EOF
#!/usr/bin/env bash
# Save the MODELLER license key in Leonardo's environment (generated by the Leonardo installer).
set -uo pipefail
CONDA="$CONDA"
ENV_NAME="$ENV_NAME"
EOF
cat >> "$BIN_DIR/leonardo-modeller-license" <<'EOF'
use_dialogs() {  # dialogs only when started from the app/menu entry (no terminal, not the installer)
  [[ -z "${LEONARDO_NO_DIALOG:-}" && ! -t 0 && ! -t 1 ]]
}
message() {     # message TEXT : a dialog when started from the app, else the terminal
  if ! use_dialogs; then
    echo "$1"
    return
  fi
  if [[ "$(uname -s)" == "Darwin" ]] && command -v osascript >/dev/null 2>&1; then
    osascript -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with title \"Leonardo\"" >/dev/null 2>&1 && return
  elif command -v zenity >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    zenity --info --title="Leonardo" --text="$1" 2>/dev/null && return
  fi
  echo "$1"
}
key="${1:-}"
if [[ -z "$key" ]]; then
  prompt="Paste your MODELLER license key.\n\nFree academic keys: https://salilab.org/modeller/registration.html"
  if use_dialogs && [[ "$(uname -s)" == "Darwin" ]] && command -v osascript >/dev/null 2>&1; then
    key="$(osascript -e "text returned of (display dialog \"$prompt\" default answer \"\" with title \"Leonardo - MODELLER license\")" 2>/dev/null || true)"
  elif use_dialogs && command -v zenity >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    key="$(zenity --entry --title="Leonardo - MODELLER license" --text="$(printf '%b' "$prompt")" 2>/dev/null || true)"
  else
    printf '%b\n' "$prompt"
    read -r -p "Key: " key
  fi
fi
key="$(printf '%s' "$key" | tr -d '[:space:]')"
[[ -n "$key" ]] || { message "No key entered: nothing was changed."; exit 1; }
config="$("$CONDA" run -n "$ENV_NAME" python -c "import glob,sys; c=sorted(glob.glob(sys.prefix+'/lib/modeller-*/modlib/modeller/config.py')); print('LEONARDO_CONFIG=' + (c[-1] if c else ''))" 2>/dev/null | sed -n 's/^LEONARDO_CONFIG=//p' | head -n 1)"
[[ -n "$config" && -f "$config" ]] || { message "MODELLER is not installed in the Leonardo environment. Run the Leonardo installer again."; exit 1; }
install_dir="$(dirname "$(dirname "$(dirname "$config")")")"
printf "install_dir = r'%s'\nlicense = r'%s'\n" "$install_dir" "$key" > "$config"
if "$CONDA" run -n "$ENV_NAME" python -c "import modeller" >/dev/null 2>&1; then
  message "MODELLER is activated. You can now build homology models in Leonardo."
else
  message "The key was saved but MODELLER rejected it. Check that it was copied exactly (Salilab e-mail) and run this again."
  exit 1
fi
EOF
chmod +x "$BIN_DIR/leonardo" "$BIN_DIR/leonardo-modeller-license"

if [[ -n "$MODELLER_KEY" ]]; then
  step "Activating MODELLER"
  LEONARDO_NO_DIALOG=1 "$BIN_DIR/leonardo-modeller-license" "$MODELLER_KEY" || warn "MODELLER did not accept the key; run 'Leonardo MODELLER License' later."
fi

# ------------------------------------------------------------------ 6. PyMOL plugin
step "6/7 Adding Leonardo to PyMOL (Plugin > Leonardo)"
startup="$HOME/.pymol/startup"
mkdir -p "$startup"
if [[ -d "$startup/Leonardo" ]]; then
  # a copy installed earlier through PyMOL's Plugin Manager would show a second, older Leonardo
  backup="$HOME/.pymol/startup_backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$backup"
  mv "$startup/Leonardo" "$backup/"
  echo "Moved an older Plugin-Manager copy of Leonardo to: $backup"
fi
( cd "$INSTALL_DIR" && LEONARDO_LOADER_ROOT="$APP_LINK" bash scripts/install_pymol_loader.sh --quiet ) \
  || die "could not install the PyMOL loader."
echo "PyMOL loader: $startup/leonardo.py"

# ------------------------------------------------------------------ 7. shortcuts
if [[ "$SHORTCUTS" -eq 1 ]]; then
  step "7/7 Creating the Leonardo shortcuts"
  if [[ "$OS" == "Darwin" ]]; then
    make_app() {   # make_app "Name" script
      local app="$HOME/Applications/$1.app"
      rm -rf "$app"
      mkdir -p "$app/Contents/MacOS"
      cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>$1</string>
  <key>CFBundleDisplayName</key><string>$1</string>
  <key>CFBundleIdentifier</key><string>org.leonardo.$(echo "$1" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>NSHumanReadableCopyright</key><string>Leonardo (MIT license). Built using PyMOL(TM). PyMOL is a trademark of Schrodinger, LLC.</string>
</dict></plist>
PLIST
      if [[ -f "$APP_LINK/assets/logo.png" ]] && command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
        local iconset
        iconset="$(mktemp -d)/AppIcon.iconset"
        mkdir -p "$iconset" "$app/Contents/Resources"
        for size in 16 32 128 256 512; do
          sips -z "$size" "$size" "$APP_LINK/assets/logo.png" --out "$iconset/icon_${size}x${size}.png" >/dev/null 2>&1 || true
          sips -z $((size * 2)) $((size * 2)) "$APP_LINK/assets/logo.png" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null 2>&1 || true
        done
        iconutil -c icns "$iconset" -o "$app/Contents/Resources/AppIcon.icns" >/dev/null 2>&1 || true
      fi
      printf '#!/usr/bin/env bash\nexec "%s"\n' "$2" > "$app/Contents/MacOS/launcher"
      chmod +x "$app/Contents/MacOS/launcher"
      echo "Created: $app"
    }
    mkdir -p "$HOME/Applications"
    make_app "Leonardo" "$BIN_DIR/leonardo"
    make_app "Leonardo MODELLER License" "$BIN_DIR/leonardo-modeller-license"
    if [[ -d "$HOME/Desktop" ]]; then
      ln -sfn "$HOME/Applications/Leonardo.app" "$HOME/Desktop/Leonardo.app"
      echo "Desktop shortcut: $HOME/Desktop/Leonardo.app"
    fi
  else
    apps="$HOME/.local/share/applications"
    mkdir -p "$apps"
    cat > "$apps/leonardo.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Leonardo
Comment=Molecular modeling suite. Built using PyMOL(TM); PyMOL is a trademark of Schrodinger, LLC.
Exec="$BIN_DIR/leonardo"
Icon=$APP_LINK/assets/logo.png
Terminal=false
Categories=Science;Chemistry;Education;
DESK
    cat > "$apps/leonardo-modeller-license.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Leonardo MODELLER License
Comment=Activate MODELLER for Leonardo
Exec="$BIN_DIR/leonardo-modeller-license"
Terminal=$(command -v zenity >/dev/null 2>&1 && echo false || echo true)
Categories=Science;
DESK
    chmod +x "$apps/leonardo.desktop" "$apps/leonardo-modeller-license.desktop"
    echo "Menu entries: $apps/leonardo.desktop"
    desktop_dir="$(command -v xdg-user-dir >/dev/null 2>&1 && xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
    if [[ -d "$desktop_dir" ]]; then
      cp "$apps/leonardo.desktop" "$desktop_dir/"
      chmod +x "$desktop_dir/leonardo.desktop"
      command -v gio >/dev/null 2>&1 && gio set "$desktop_dir/leonardo.desktop" metadata::trusted true 2>/dev/null || true
      echo "Desktop shortcut: $desktop_dir/leonardo.desktop"
    fi
  fi
else
  step "7/7 Skipping shortcuts (--no-shortcuts)"
fi

# ------------------------------------------------------------------ C++ extension (optional)
step "Building the cryptic-pocket C++ extension"
PYTHONPATH="$APP_LINK/src" "$CONDA" run -n "$ENV_NAME" python -m leonardo.cryptic_build \
  || warn "the C++ extension was not built; cryptic-pocket analysis will use its numpy version (same results, slower)."

# ------------------------------------------------------------------ check
step "Final check"
if "$CONDA" run -n "$ENV_NAME" python -c "import sys; sys.path[:0]=['$APP_LINK/legacy','$APP_LINK/src','$APP_LINK']; import ProtLigInteract, pymol, openmm, rdkit; print('Leonardo', ProtLigInteract.__version__, 'ready')"; then
  :
else
  die "the final check failed."
fi
# what PyMOL itself sees at start: its plugin search must find the Leonardo loader (Plugin > Leonardo)
# (one line: conda run does not pass arguments that contain line breaks)
plugin_check="$("$CONDA" run -n "$ENV_NAME" python -c 'import pymol.plugins as P; P.initialize(-2); print("LEONARDO_PLUGIN=" + ("found" if "leonardo" in P.plugins else "missing"))' 2>/dev/null | sed -n 's/^LEONARDO_PLUGIN=//p' | head -n 1 || true)"
if [[ "$plugin_check" == "found" ]]; then
  echo "PyMOL finds Leonardo: Plugin > Leonardo"
else
  warn "PyMOL does not list the Leonardo plugin (check: ${plugin_check:-no answer}). Look at ~/.pymolpluginsrc.py and ~/.pymol/startup/leonardo.py, or write to the support address in README.md."
fi
PYTHONPATH="$APP_LINK/src:$APP_LINK/legacy" "$CONDA" run -n "$ENV_NAME" python -m leonardo.report_scientific_readiness \
  || warn "some optional workflows report missing pieces (see above)."
# older copies of Leonardo left by previous installations (the environment points at the current one)
for old in "$LEONARDO_HOME"/app-*; do
  [[ -d "$old" && "$old" != "$INSTALL_DIR" ]] && rm -rf "$old" && echo "Removed an older copy: $old"
done
modeller_state="not activated yet (open 'Leonardo MODELLER License')"
"$CONDA" run -n "$ENV_NAME" python -c "import modeller" >/dev/null 2>&1 && modeller_state="activated"

echo
bold "Leonardo is installed."
if [[ "$OS" == "Darwin" ]]; then
  echo "  Open it: double-click 'Leonardo' on your Desktop or in Applications (~/Applications)."
else
  echo "  Open it: 'Leonardo' in your applications menu or on your Desktop."
fi
echo "  Then, inside PyMOL: Plugin > Leonardo"
echo "  MODELLER: $modeller_state"
echo "  From a terminal: $BIN_DIR/leonardo"
echo "  To update later, run this installer again."
echo "  Log: $LOG_FILE"
