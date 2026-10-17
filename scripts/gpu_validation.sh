#!/usr/bin/env bash
# GPU validation of a Leonardo install (Linux with an NVIDIA/AMD GPU, or macOS).
#
# 1. Install Leonardo with the normal installer (the full environment, not --skip-env):
#      bash install-leonardo-linux.sh --yes
# 2. Run this script from the installed source (or a clone):
#      bash ~/.leonardo/app/scripts/gpu_validation.sh 2>&1 | tee gpu_validation.log
#
# It prints the GPU and driver, checks OpenMM's force agreement between platforms, re-detects the hardware
# Leonardo will use, and replays two packs on the GPU: examples/films/03_molecular_dynamics (implicit-solvent
# MD, statistical checks) and examples/gpu_validation (explicit NPT MD with a ligand and a metal site,
# minimization, ABFE-lite with 3 replicas, a protein on gold), which also check that every step ran on a GPU.
# About 30-60 minutes on a recent GPU. Send gpu_validation.log back.
set -u

LEONARDO_HOME="${LEONARDO_HOME:-$HOME/.leonardo}"
APP="${LEONARDO_APP:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_NAME="${LEONARDO_ENV:-protliginteract-full}"
CONDA="${CONDA_EXE:-}"
for c in "$CONDA" "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
         "/opt/anaconda3/bin/conda" "/opt/miniforge3/bin/conda" "$(command -v conda 2>/dev/null)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CONDA="$c"; break; fi
done
[ -x "$CONDA" ] || { echo "conda not found: set CONDA_EXE"; exit 1; }
# activation scripts of some packages print to stdout: read a marked line, not the last one
PY="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print("LEONARDO_PY=" + sys.executable)' \
      | sed -n 's/^LEONARDO_PY=//p' | tail -1)"
[ -x "$PY" ] || { echo "python of the '$ENV_NAME' environment not found (set LEONARDO_ENV)"; exit 1; }
echo "== Leonardo GPU validation  $(date '+%F %T')"
echo "app: $APP   env: $ENV_NAME   python: $PY"

echo; echo "== GPU and driver"
(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv) || true
(command -v rocm-smi >/dev/null && rocm-smi --showproductname) || true
[ "$(uname)" = "Darwin" ] && system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|Metal" || true

echo; echo "== OpenMM platforms and force agreement"
"$PY" -m openmm.testInstallation

echo; echo "== Hardware profile Leonardo will use (fresh benchmark)"
PYTHONPATH="$APP/legacy:$APP/src" "$PY" - <<'EOF'
from leonardo import hardware_runtime as h, abfe_runtime as a
p = h.detect(benchmark=True, refresh=True)
print("benchmark:", p.get("openmm_benchmark"))
print("md platform:", p.get("md_platform"))
for name in ("CUDA", "HIP", "OpenCL", "Metal"):
    if name in p.get("openmm_platforms", []):
        print(f"{name} context properties:", a.openmm_gpu_properties(name))
EOF

export QT_API=pyside6 QT_QPA_PLATFORM=offscreen LEONARDO_NO_HARDWARE_DETECTION=1
STATUS=0
for pack in examples/films/03_molecular_dynamics examples/gpu_validation; do
  echo; echo "== replay verify $pack  $(date '+%T')"
  (cd "$APP" && PYTHONPATH="$APP/legacy:$APP/src" "$PY" -m leonardo.replay verify "$pack") || STATUS=1
done

echo; echo "== $( [ $STATUS = 0 ] && echo 'GPU VALIDATION PASSED' || echo 'GPU VALIDATION: SOME CHECKS FAILED' )  $(date '+%T')"
exit $STATUS
