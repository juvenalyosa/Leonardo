#!/usr/bin/env bash
set -euo pipefail

# ProtLigInteract environment installer/updater.
# - Installs/updates main env from environments/environment.yml
# - Enforces a Psi4-compatible stack (avoids libint ABI segfaults)
# - Validates Psi4 with a real SCF calculation
# - Optionally installs/updates optional env from environments/environment-optional.yml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAIN_YAML="${MAIN_YAML:-environments/environment.yml}"
MAIN_LOCK_YAML="${MAIN_LOCK_YAML:-environments/environment-protliginteract-full.lock.yml}"
OPTIONAL_YAML="${OPTIONAL_YAML:-environments/environment-optional.yml}"
WITH_OPTIONAL=0
RECREATE=0
RUN_PSI4_SMOKE=0
ENV_SPEC_MODE="${ENV_SPEC_MODE:-auto}"
TMP_FILES=()

cleanup() {
  local tmp
  if [[ "${#TMP_FILES[@]}" -eq 0 ]]; then
    return 0
  fi
  for tmp in "${TMP_FILES[@]}"; do
    if [[ -n "$tmp" && -e "$tmp" ]]; then
      rm -f "$tmp"
    fi
  done
  return 0
}
trap cleanup EXIT

usage() {
  cat <<'EOF'
Usage:
  ./scripts/install_envs.sh [--with-optional] [--recreate] [--psi4-smoke] [--locked|--flexible]

Options:
  --with-optional   Also install/update environments/environment-optional.yml
  --recreate        Remove env(s) before creating
  --psi4-smoke      Run psi4 --test smoke after installation
  --locked          Use environments/environment-protliginteract-full.lock.yml exactly enough
                    for reviewer/user reproduction on the same platform
  --flexible        Use environments/environment.yml and let conda solve compatible versions
  -h, --help        Show this help

Default:
  auto: use the lockfile when present; fall back to environments/environment.yml if the
  locked solve is not usable on this host.
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
    --psi4-smoke)
      RUN_PSI4_SMOKE=1
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$MAIN_YAML" ]]; then
  echo "ERROR: missing $MAIN_YAML in $SCRIPT_DIR" >&2
  exit 1
fi

if [[ "$ENV_SPEC_MODE" != "auto" && "$ENV_SPEC_MODE" != "locked" && "$ENV_SPEC_MODE" != "flexible" ]]; then
  echo "ERROR: invalid ENV_SPEC_MODE=$ENV_SPEC_MODE" >&2
  exit 2
fi

main_env="$(awk -F': *' '/^name:/{print $2; exit}' "$MAIN_YAML")"
if [[ -z "${main_env:-}" ]]; then
  echo "ERROR: could not parse env name from $MAIN_YAML" >&2
  exit 1
fi

optional_env=""
if [[ -f "$OPTIONAL_YAML" ]]; then
  optional_env="$(awk -F': *' '/^name:/{print $2; exit}' "$OPTIONAL_YAML")"
fi

env_exists() {
  local env_name="$1"
  conda run -n "$env_name" python -c "import sys; print(sys.prefix)" >/dev/null 2>&1
}

create_or_update_env() {
  local env_name="$1"
  local yaml_file="$2"
  local recreate="$3"
  if [[ "$recreate" -eq 1 ]] && env_exists "$env_name"; then
    echo "[INFO] Removing existing env: $env_name"
    conda env remove -n "$env_name" -y >/dev/null
  fi
  if env_exists "$env_name"; then
    echo "[INFO] Updating env: $env_name from $yaml_file"
    conda env update -n "$env_name" -f "$yaml_file" --prune
  else
    echo "[INFO] Creating env: $env_name from $yaml_file"
    conda env create -f "$yaml_file"
  fi
}

sanitize_lockfile() {
  local input_yaml="$1"
  local output_yaml="$2"
  # conda env export writes an absolute prefix that should not be reused on
  # another machine.  It may also capture the editable Leonardo install as a
  # pip package; the repository is installed explicitly after env creation.
  grep -v -E '^(prefix:|[[:space:]]+- leonardo-pymol-suite==)' "$input_yaml" > "$output_yaml"
}

prepare_main_yaml() {
  local mode="$1"
  local selected_yaml="$MAIN_YAML"
  local selected_label="$MAIN_YAML"
  local using_lock=0

  case "$mode" in
    locked)
      if [[ ! -f "$MAIN_LOCK_YAML" ]]; then
        echo "ERROR: --locked requested but $MAIN_LOCK_YAML is missing" >&2
        exit 1
      fi
      selected_yaml="$MAIN_LOCK_YAML"
      selected_label="$MAIN_LOCK_YAML"
      using_lock=1
      ;;
    auto)
      if [[ -f "$MAIN_LOCK_YAML" ]]; then
        selected_yaml="$MAIN_LOCK_YAML"
        selected_label="$MAIN_LOCK_YAML"
        using_lock=1
      fi
      ;;
    flexible)
      selected_yaml="$MAIN_YAML"
      selected_label="$MAIN_YAML"
      ;;
  esac

  if [[ "$using_lock" -eq 1 ]]; then
    local tmp_yaml
    tmp_yaml="$(mktemp "${TMPDIR:-/tmp}/leonardo_env_lock.XXXXXX.yml")"
    TMP_FILES+=("$tmp_yaml")
    sanitize_lockfile "$selected_yaml" "$tmp_yaml"
    selected_yaml="$tmp_yaml"
  fi

  MAIN_YAML_FOR_INSTALL="$selected_yaml"
  MAIN_YAML_INSTALL_LABEL="$selected_label"
  MAIN_YAML_USING_LOCK="$using_lock"
}

enforce_psi4_stack() {
  local env_name="$1"
  echo "[INFO] Enforcing Psi4-compatible stack in $env_name"
  # Remove possible pip wheels that can shadow conda Psi4 binaries.
  conda run -n "$env_name" python -m pip uninstall -y psi4 psi4-core >/dev/null 2>&1 || true
  conda install -n "$env_name" -y -c conda-forge \
    "psi4=1.9.*" \
    "libint=2.9.*" \
    "pcmsolver=1.2.3" \
    "qcelemental>=0.28,<1.0" \
    "qcengine>=0.30,<1.0"
}

enforce_rdkit_numpy_compat() {
  local env_name="$1"
  echo "[INFO] Enforcing RDKit/NumPy ABI compatibility in $env_name"
  # Pip packages can overwrite Conda-managed NumPy files while leaving Conda's
  # package metadata unchanged. Remove those files through pip first, then
  # force exact Conda builds back into place so a mixed NumPy 1.x/2.x tree
  # cannot survive an earlier incompatible pip resolution.
  conda run -n "$env_name" python -m pip uninstall -y numpy pandas >/dev/null 2>&1 || true
  conda install -n "$env_name" -y -c conda-forge --force-reinstall \
    "numpy=1.26.4" \
    "scipy" \
    "rdkit>=2023.09" \
    "pandas<3"
}

validate_main_env() {
  local env_name="$1"
  echo "[INFO] Validating core imports in $env_name"
  conda run -n "$env_name" python -c "import numpy, rdkit, openmm, qcelemental, qcengine, meeko, pymol; print('core imports: OK')"
  conda run -n "$env_name" python -c "import numpy, rdkit; print('numpy', numpy.__version__); print('rdkit', rdkit.__version__)"
  echo "[INFO] Validating required external tools in $env_name"
  conda run -n "$env_name" python -c "import shutil,sys; req=['antechamber','parmchk2','tleap','smina','fpocket','psi4','obabel','reduce','blastp','hmmscan','clustalo','mafft']; miss=[x for x in req if shutil.which(x) is None]; print('missing_tools', miss); sys.exit(1 if miss else 0)"
  echo "[INFO] Validating Psi4 runtime in $env_name"
  conda run -n "$env_name" python -c "import psi4,glob,sys; print('psi4', psi4.__version__); print(glob.glob(sys.prefix + '/lib/libint2*.dylib'))"
  conda run -n "$env_name" psi4 --version
  conda run -n "$env_name" python -c "import psi4; mol=psi4.geometry('0 1\nO\nH 1 0.96\nH 1 0.96 2 104.5\n'); e=psi4.energy('scf/sto-3g', molecule=mol); print('psi4_scf_ok', e)"
  if [[ "$RUN_PSI4_SMOKE" -eq 1 ]]; then
    echo "[INFO] Running Psi4 smoke tests (can take ~1 minute)"
    conda run -n "$env_name" psi4 --test smoke
  fi
}

echo "[INFO] Setting strict channel priority"
conda config --set channel_priority strict >/dev/null

prepare_main_yaml "$ENV_SPEC_MODE"
echo "[INFO] Main environment spec: $MAIN_YAML_INSTALL_LABEL"

if [[ "$ENV_SPEC_MODE" == "auto" && "$MAIN_YAML_USING_LOCK" -eq 1 ]]; then
  if ! create_or_update_env "$main_env" "$MAIN_YAML_FOR_INSTALL" "$RECREATE"; then
    echo "[WARN] Locked environment could not be created/updated on this host."
    echo "[WARN] Falling back to flexible solve from $MAIN_YAML"
    create_or_update_env "$main_env" "$MAIN_YAML" "$RECREATE"
  fi
else
  create_or_update_env "$main_env" "$MAIN_YAML_FOR_INSTALL" "$RECREATE"
fi
enforce_psi4_stack "$main_env"
enforce_rdkit_numpy_compat "$main_env"
validate_main_env "$main_env"

if [[ "$WITH_OPTIONAL" -eq 1 ]]; then
  if [[ ! -f "$OPTIONAL_YAML" ]]; then
    echo "ERROR: --with-optional requested but $OPTIONAL_YAML not found" >&2
    exit 1
  fi
  if [[ -z "${optional_env:-}" ]]; then
    echo "ERROR: could not parse env name from $OPTIONAL_YAML" >&2
    exit 1
  fi
  create_or_update_env "$optional_env" "$OPTIONAL_YAML" "$RECREATE"
  echo "[INFO] Optional env ready: $optional_env"
fi

echo "[DONE] Main env ready: $main_env"
if [[ "$WITH_OPTIONAL" -eq 1 ]]; then
  echo "[DONE] Optional env ready: $optional_env"
fi
