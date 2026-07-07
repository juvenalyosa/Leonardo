#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_NAME="${1:-Leonardo.zip}"
DIST_DIR="$ROOT_DIR/dist"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/leonardo_pkg.XXXXXX")"
PKG_DIR="$STAGE_DIR/Leonardo"
TMP_ZIP="$STAGE_DIR/$OUT_NAME"

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
  zip -rq "$TMP_ZIP" "Leonardo"
)

mkdir -p "$DIST_DIR"
mv "$TMP_ZIP" "$DIST_DIR/$OUT_NAME"
echo "Created $DIST_DIR/$OUT_NAME"
