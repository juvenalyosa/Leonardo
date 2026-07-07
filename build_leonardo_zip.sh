#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_NAME="${1:-Leonardo.zip}"
DIST_DIR="$ROOT_DIR/dist"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/leonardo_pkg.XXXXXX")"
PKG_DIR="$STAGE_DIR/Leonardo"
TMP_ZIP="$STAGE_DIR/$OUT_NAME"
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1783382400}"  # 2026-07-07T00:00:00Z

cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

mkdir -p "$PKG_DIR"

copy_if_exists() {
  local src="$1"
  if [ -e "$ROOT_DIR/$src" ]; then
    cp -R "$ROOT_DIR/$src" "$PKG_DIR/"
  fi
}

copy_if_exists "__init__.py"
copy_if_exists "code_v2.py"
copy_if_exists "background_tasks.py"
copy_if_exists "bioseq_runtime.py"
copy_if_exists "dock_runtime.py"
copy_if_exists "smina_runtime.py"
copy_if_exists "task_monitor_runtime.py"
copy_if_exists "redock_runtime.py"
copy_if_exists "vs_runtime.py"
copy_if_exists "leadopt_runtime.py"
copy_if_exists "abfe_runtime.py"
copy_if_exists "alphafold_runtime.py"
copy_if_exists "openmm_dynamics_runtime.py"
copy_if_exists "openmm_qmmm_runtime.py"
copy_if_exists "small_molecule_runtime.py"
copy_if_exists "modeler_runtime.py"
copy_if_exists "qmmm_mopac_runtime.py"
copy_if_exists "psi4_cluster_runtime.py"
copy_if_exists "psi4_runtime.py"
copy_if_exists "scientific_readiness_runtime.py"
copy_if_exists "report_scientific_readiness.py"
copy_if_exists "structure_fetch_runtime.py"
copy_if_exists "interaction_typing_runtime.py"
copy_if_exists "interaction_runtime.py"
copy_if_exists "trajectory_interaction_runtime.py"
copy_if_exists "ligand_canvas_runtime.py"
copy_if_exists "ligand_vib_refine_runtime.py"
copy_if_exists "hand_gesture_runtime.py"
copy_if_exists "abfe_deploy_models.json"
copy_if_exists "mutator.py"
copy_if_exists "genmodels.py"
copy_if_exists "protliginteract.ui"
copy_if_exists "logo.png"
copy_if_exists "README.md"
copy_if_exists "LICENSE"
copy_if_exists "CITATION.cff"
copy_if_exists "environment.yml"
copy_if_exists "environment-protliginteract-full.lock.yml"
copy_if_exists "environment-optional.yml"
copy_if_exists "install_envs.sh"
copy_if_exists "install_leonardo.sh"
copy_if_exists "install_pymol_loader.sh"
copy_if_exists "launch_leonardo.sh"
copy_if_exists "build_leonardo_zip.sh"
copy_if_exists "Install Leonardo.command"
copy_if_exists "Launch Leonardo.command"
copy_if_exists "Install Leonardo Linux.sh"
copy_if_exists "Launch Leonardo Linux.sh"
copy_if_exists "Install Leonardo Windows.bat"
copy_if_exists "Launch Leonardo Windows.bat"
copy_if_exists "installers"
copy_if_exists "pyproject.toml"
copy_if_exists "requirements.txt"
copy_if_exists "ENV_SETUP.md"
copy_if_exists "ARCHITECTURE.md"
copy_if_exists "CONTRIBUTING.md"

if [ -d "$ROOT_DIR/screenshots" ]; then
  mkdir -p "$PKG_DIR/screenshots"
  find "$ROOT_DIR/screenshots" -type f ! -name "*.md" -exec cp {} "$PKG_DIR/screenshots/" \;
fi

(
  cd "$STAGE_DIR"
  python - "$TMP_ZIP" "$SOURCE_DATE_EPOCH" <<'PY'
import hashlib
import os
import stat
import sys
import time
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
source_date_epoch = int(sys.argv[2])
root = Path("Leonardo")
timestamp = time.gmtime(source_date_epoch)[:6]

def entry_mode(path: Path) -> int:
    if path.is_dir():
        return 0o755
    current = path.stat().st_mode
    if current & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        return 0o755
    return 0o644

def add_entry(zf: zipfile.ZipFile, path: Path) -> None:
    arcname = path.as_posix()
    if path.is_dir():
        arcname = arcname.rstrip("/") + "/"
    info = zipfile.ZipInfo(arcname, timestamp)
    mode = entry_mode(path)
    info.external_attr = (mode << 16) | (0x10 if path.is_dir() else 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    if path.is_dir():
        zf.writestr(info, b"")
    else:
        with path.open("rb") as handle:
            zf.writestr(info, handle.read())

paths = [root]
paths.extend(sorted(root.rglob("*"), key=lambda p: p.as_posix()))
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for path in paths:
        if path.name in {".DS_Store", "__pycache__"}:
            continue
        if "__pycache__" in path.parts:
            continue
        add_entry(zf, path)

digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
print(f"sha256  {digest}  {zip_path.name}")
PY
)

mkdir -p "$DIST_DIR"
mv "$TMP_ZIP" "$DIST_DIR/$OUT_NAME"
python - "$DIST_DIR/$OUT_NAME" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
PY
echo "Created $DIST_DIR/$OUT_NAME"
