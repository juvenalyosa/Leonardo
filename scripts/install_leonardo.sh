#!/usr/bin/env bash
set -euo pipefail

# One-command Leonardo installer for reviewers and end users.
# It delegates environment solving to scripts/install_envs.sh, then installs the source
# tree in editable mode and runs a short validation suite.  The script avoids
# GUI automation: users still install dist/Leonardo.zip through PyMOL's plugin
# manager after the ZIP is built.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WITH_OPTIONAL=0
RECREATE=0
RUN_FULL_TESTS=0
RUN_PAPER_ASSETS=0
SKIP_ZIP=0
INSTALL_PYMOL_LOADER=1
ENV_SPEC_MODE="auto"
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/install_leonardo.sh [options]

Options:
  --with-optional   Also create/update the optional ML/ColabFold environment
  --recreate        Recreate conda environments from scratch
  --locked          Install the main environment from the locked version file
  --flexible        Install the main environment from environments/environment.yml
  --full-tests      Run the complete pytest suite after installation
  --paper-assets    Regenerate manuscript figures/tables after installation
  --check-only      Validate installer inputs and print the planned actions only
  --no-pymol-loader Do not install the PyMOL startup loader
  --skip-zip        Do not rebuild dist/Leonardo.zip
  -h, --help        Show this help

Default behavior:
  1. create/update protliginteract-full, preferring the locked version file
  2. install this repository with python -m pip install -e . --no-build-isolation
  3. install a PyMOL startup loader so Plugin > Leonardo appears
  4. run import/readiness/plugin-packaging smoke checks
  5. run focused or full pytest checks
  6. rebuild dist/Leonardo.zip for PyMOL Plugin Manager
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-optional)
      WITH_OPTIONAL=1
      shift
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --locked)
      ENV_SPEC_MODE="locked"
      shift
      ;;
    --flexible)
      ENV_SPEC_MODE="flexible"
      shift
      ;;
    --full-tests)
      RUN_FULL_TESTS=1
      shift
      ;;
    --paper-assets)
      RUN_PAPER_ASSETS=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --skip-zip)
      SKIP_ZIP=1
      shift
      ;;
    --no-pymol-loader)
      INSTALL_PYMOL_LOADER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

env_name="$(awk -F': *' '/^name:/{print $2; exit}' environments/environment.yml)"
if [[ -z "${env_name:-}" ]]; then
  echo "ERROR: could not parse environment name from environments/environment.yml" >&2
  exit 1
fi

if [[ "$RUN_PAPER_ASSETS" -eq 1 ]]; then
  RUN_FULL_TESTS=1
fi

install_args=()
if [[ "$WITH_OPTIONAL" -eq 1 ]]; then
  install_args+=(--with-optional)
fi
if [[ "$RECREATE" -eq 1 ]]; then
  install_args+=(--recreate)
fi
case "$ENV_SPEC_MODE" in
  locked)
    install_args+=(--locked)
    ;;
  flexible)
    install_args+=(--flexible)
    ;;
  auto)
    ;;
  *)
    echo "ERROR: invalid environment spec mode: $ENV_SPEC_MODE" >&2
    exit 2
    ;;
esac

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "Leonardo installer check-only mode"
  echo "Repository: $ROOT_DIR"
  echo "Environment: $env_name"
  echo "Environment spec mode: $ENV_SPEC_MODE"
  echo "Conda on PATH: $(command -v conda >/dev/null 2>&1 && command -v conda || echo 'not found')"
  required_files=(
    environments/environment.yml
    environments/environment-protliginteract-full.lock.yml
    pyproject.toml
    scripts/install_envs.sh
    scripts/install_pymol_loader.sh
    scripts/build_leonardo_zip.sh
    scripts/launch_leonardo.sh
    legacy/code_v2.py
    legacy/ProtLigInteract.py
    packaging/pymol/__init__.py
    src/leonardo
  )
  missing=0
  for file in "${required_files[@]}"; do
    if [[ -e "$file" ]]; then
      echo "OK: $file"
    else
      echo "MISSING: $file"
      missing=1
    fi
  done
  echo "Planned actions without --check-only:"
  echo "  install/update conda environment"
  echo "  pip install -e . --no-build-isolation"
  echo "  install PyMOL startup loader: $INSTALL_PYMOL_LOADER"
  echo "  run full tests: $RUN_FULL_TESTS"
  echo "  regenerate manuscript assets: $RUN_PAPER_ASSETS"
  echo "  build plugin ZIP: $((1 - SKIP_ZIP))"
  exit "$missing"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found on PATH. Install Miniforge/Mambaforge/Miniconda first." >&2
  exit 1
fi

echo "[1/6] Preparing conda environment: $env_name"
"$ROOT_DIR/scripts/install_envs.sh" "${install_args[@]}"

echo "[2/6] Installing Leonardo source in editable mode"
conda run -n "$env_name" python -m pip install -e . --no-build-isolation

echo "[3/6] Installing PyMOL startup loader"
if [[ "$INSTALL_PYMOL_LOADER" -eq 1 ]]; then
  "$ROOT_DIR/scripts/install_pymol_loader.sh" --quiet
else
  echo "Skipping PyMOL startup loader"
fi

echo "[4/6] Running smoke validation"
conda run -n "$env_name" python -c "import code_v2, interaction_runtime, redock_runtime, openmm_dynamics_runtime, psi4_runtime, qmmm_mopac_runtime; print('leonardo imports: OK')"
conda run -n "$env_name" leonardo-readiness --strict

if [[ "$RUN_FULL_TESTS" -eq 1 ]]; then
  echo "[5/6] Running full pytest suite"
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n "$env_name" python -m pytest -q --junitxml=draft/results/pytest_validation.xml
else
  echo "[5/6] Running focused smoke tests"
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n "$env_name" python -m pytest -q \
    tests/test_plugin_packaging.py \
    tests/test_runtime_modules.py \
    tests/test_interaction_runtime.py \
    tests/test_redock_runtime_samples.py \
    tests/test_openmm_dynamics_runtime.py \
    --junitxml=draft/results/pytest_smoke.xml
fi

if [[ "$RUN_PAPER_ASSETS" -eq 1 ]]; then
  echo "[extra] Regenerating BMC manuscript assets"
  conda run -n "$env_name" python draft/scripts/generate_bmc_validation_assets.py
fi

if [[ "$SKIP_ZIP" -eq 0 ]]; then
  echo "[6/6] Building PyMOL plugin ZIP"
  "$ROOT_DIR/scripts/build_leonardo_zip.sh"
else
  echo "[6/6] Skipping ZIP build"
fi

cat <<EOF
[DONE] Leonardo installation workflow completed.

Launch options:
  Double click: $ROOT_DIR/installers/shortcuts/Launch Leonardo.command
  Terminal:     $ROOT_DIR/scripts/launch_leonardo.sh

Inside PyMOL:
  Plugin > Leonardo

Plugin ZIP is also available for PyMOL Plugin Manager:
  $ROOT_DIR/dist/Leonardo.zip
EOF
