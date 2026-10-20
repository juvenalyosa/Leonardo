#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_NAME="${1:-Leonardo.zip}"
DIST_DIR="$ROOT_DIR/dist"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/leonardo_pkg.XXXXXX")"
PKG_DIR="$STAGE_DIR/Leonardo"
TMP_ZIP="$STAGE_DIR/$OUT_NAME"
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1783468800}"  # 2026-07-08T00:00:00Z

cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

mkdir -p "$PKG_DIR"

# Canonical package namespace.
if [ -d "$ROOT_DIR/src/leonardo" ]; then
  mkdir -p "$PKG_DIR/leonardo"
  cp -R "$ROOT_DIR/src/leonardo/"* "$PKG_DIR/leonardo/"
  # the C++ extension is built on each computer (python -m leonardo.cryptic_build), never shipped
  rm -f "$PKG_DIR/leonardo/"_cryptic*.so "$PKG_DIR/leonardo/"_cryptic*.pyd
fi

# Benchmark command helpers imported by the installed ABFE probe module.
mkdir -p "$PKG_DIR/leonardo_benchmarks/abfe/ptb3_ben"
cp "$ROOT_DIR/benchmarks/__init__.py" "$PKG_DIR/leonardo_benchmarks/__init__.py"
cp "$ROOT_DIR/benchmarks/abfe/__init__.py" "$PKG_DIR/leonardo_benchmarks/abfe/__init__.py"
cp "$ROOT_DIR/benchmarks/abfe/ptb3_ben/"*.py "$PKG_DIR/leonardo_benchmarks/abfe/ptb3_ben/"

# PyMOL still expects a plugin package entrypoint and historical top-level
# module names inside the installable ZIP. Their sources live outside the
# repository root so the checkout remains easy to navigate.
cp "$ROOT_DIR/packaging/pymol/__init__.py" "$PKG_DIR/__init__.py"
find "$ROOT_DIR/legacy" -maxdepth 1 -type f -name "*.py" -exec cp {} "$PKG_DIR/" \;

if [ -d "$ROOT_DIR/assets" ]; then
  cp -R "$ROOT_DIR/assets" "$PKG_DIR/assets"
fi

copy_if_exists() {
  local src="$1"
  if [ -e "$ROOT_DIR/$src" ]; then
    cp -R "$ROOT_DIR/$src" "$PKG_DIR/"
  fi
}

copy_if_exists "README.md"
copy_if_exists "LICENSE"
copy_if_exists "THIRD_PARTY_LICENSES.md"
copy_if_exists "CITATION.cff"
copy_if_exists "environments"
copy_if_exists "scripts"
copy_if_exists "cpp"
copy_if_exists "installers"
if [ -d "$ROOT_DIR/installers/shortcuts" ]; then
  find "$ROOT_DIR/installers/shortcuts" -maxdepth 1 -type f -exec cp {} "$PKG_DIR/" \;
fi
copy_if_exists "pyproject.toml"
copy_if_exists "requirements.txt"
copy_if_exists "docs"
copy_if_exists "CONTRIBUTING.md"

if [ -d "$ROOT_DIR/docs/screenshots" ]; then
  mkdir -p "$PKG_DIR/docs/screenshots"
  find "$ROOT_DIR/docs/screenshots" -type f ! -name "*.md" -exec cp {} "$PKG_DIR/docs/screenshots/" \;
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
