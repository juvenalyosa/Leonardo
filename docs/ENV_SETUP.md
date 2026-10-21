# Leonardo Environment Setup

Leonardo is the suite name. The conda environment names below keep the legacy `protliginteract-*` prefix to avoid breaking existing local setups and scripts.

This project is best installed with a **two-environment strategy**:

- `protliginteract-full` (required): complete stable Leonardo stack (PyMOL, OpenMM, AmberTools, Psi4, RDKit, Meeko, MODELLER, local bioinformatics CLIs, and test/dev helpers)
- `protliginteract-optional` (optional): heavy/legacy extras (ColabFold/JAX/DeepChem, AutoDockTools scripts)

This keeps the plugin stable and avoids dependency conflicts (especially `numpy`/`jax`/`tensorflow`).

The main `environments/environment.yml` now covers the full stable suite:
- core Leonardo runtime and GUI dependencies
- docking / redocking / virtual-screening tools
- OpenMM / ABFE / QM-MM / Psi4 workflows
- MODELLER homology modeling
- local sequence-analysis CLIs
- regression-test and formatting tools (`pytest`, `black`, `ruff`, `pre-commit`)

The main environment now includes local bioinformatics CLI tools used by the
new **Sequence Bioinformatics** panel:
- `blastp` (BLAST+)
- `hmmscan` (HMMER)
- `clustalo` (Clustal Omega)
- `mafft`

## 0) macOS Double-Click Installer (recommended for end users)

From Finder:

1. Double click `installers/shortcuts/Install Leonardo.command`.
2. If Conda is missing, allow the installer to install Miniforge in `~/miniforge3`.
3. When installation finishes, double click `installers/shortcuts/Launch Leonardo.command`.
4. In PyMOL, open `Plugin > Leonardo`.

The macOS installer first tries the pinned
`environments/environment-protliginteract-full.lock.yml` file. If that exact environment is
not usable on the current Mac, it offers a fallback solve from `environments/environment.yml`.

The installer also writes `~/.pymol/startup/leonardo.py`; this is only a small
loader that registers Leonardo from the installed source tree when PyMOL starts.
Remove it with:

```bash
./scripts/install_pymol_loader.sh --remove
```

## 1) One-Command Shell Installer

From repo root:

```bash
./scripts/install_leonardo.sh
```

Installer preflight without environment creation or PyMOL changes:

```bash
./scripts/install_leonardo.sh --check-only --skip-zip --no-pymol-loader
```

Main + optional envs:

```bash
./scripts/install_leonardo.sh --with-optional
```

What this script does:
- applies strict conda channel priority
- creates/updates env(s), preferring the locked main environment when available
- enforces a stable Psi4 stack (`psi4=1.9.*`, `libint=2.9.*`, `pcmsolver=1.2.3`)
- validates core imports, `psi4 --version`, and a real Psi4 SCF run
- installs Leonardo in editable mode
- installs the PyMOL startup loader
- rebuilds `dist/Leonardo.zip`

Optional deeper Psi4 check:

```bash
./scripts/install_envs.sh --psi4-smoke
```

## 2) Linux and Windows Installers

Linux native:

```bash
./installers/linux/install_leonardo_linux.sh
```

or:

```bash
./installers/shortcuts/"Install Leonardo Linux.sh"
```

Windows is provided through WSL2 helper scripts, not as a native full scientific stack. Treat this route as pending independent clean-machine validation:

```powershell
wsl --install -d Ubuntu
```

After Ubuntu is configured, double click:

```text
installers\shortcuts\Install Leonardo Windows.bat
installers\shortcuts\Launch Leonardo Windows.bat
```

On Windows 11, PyMOL GUI should open through WSLg. On Windows 10, start an
external X server before launching PyMOL from WSL.

The current `environments/environment-protliginteract-full.lock.yml` was generated on macOS.
Linux and WSL installations therefore may use the automatic fallback to
`environments/environment.yml` until platform-specific lockfiles are generated.

## 3) Create the Main Environment (manual route)

From repo root:

```bash
conda config --set channel_priority strict
conda env remove -n protliginteract-full -y 2>/dev/null || true
conda env create -f environments/environment.yml
conda activate protliginteract-full
```

## 4) Validate Main Environment

```bash
PYTHONPATH=. python report_scientific_readiness.py --strict
python -c "import numpy, rdkit, openmm, psi4, qcelemental, qcengine, meeko, pymol; print('core imports: OK')"
which antechamber parmchk2 tleap reduce obabel smina fpocket psi4 blastp hmmscan clustalo mafft
python -c "import psi4; mol=psi4.geometry('0 1\nO\nH 1 0.96\nH 1 0.96 2 104.5\n'); print('psi4_scf_ok', psi4.energy('scf/sto-3g', molecule=mol))"
```

Expected:
- readiness report shows no workflows in `missing_required`
- Python imports succeed.
- binaries resolve to `.../envs/protliginteract-full/bin/...`.

## 4.1) Plugin Regression Tests

From repo root:

```bash
conda activate protliginteract-full
PYTHONPATH=. pytest -q
```

Expected:
- full suite passes in the current repo state (`531 passed` at the time this guide was updated)

## 5) Install Optional Extras (separate env, recommended)

```bash
conda env remove -n protliginteract-optional -y 2>/dev/null || true
conda env create -f environments/environment-optional.yml
conda activate protliginteract-optional
```

## 6) Wrapper for ColabFold (so plugin can call it externally)

Create wrapper:

```bash
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/colabfold_batch_pli" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec conda run -n protliginteract-optional --no-capture-output colabfold_batch "$@"
EOF
chmod +x "$HOME/.local/bin/colabfold_batch_pli"
```

Ensure PATH:

```bash
grep -q 'HOME/.local/bin' "$HOME/.zshrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
source "$HOME/.zshrc"
colabfold_batch_pli --help
```

In plugin AlphaFold panel, set executable path to:

`/Users/<your-user>/.local/bin/colabfold_batch_pli`

## 7) Plugin Runtime Rule

Always launch PyMOL from the **main env**:

```bash
conda activate protliginteract-full
pymol
```

The plugin remains a single plugin; optional tools are invoked as external processes.

The equivalent launcher is:

```bash
./scripts/launch_leonardo.sh
```

## 8) Update Environments

Main:

```bash
conda env update -n protliginteract-full -f environments/environment.yml --prune
```

Optional:

```bash
conda env update -n protliginteract-optional -f environments/environment-optional.yml --prune
```

## 9) Common Issues

### A) Psi4 error: `Library not loaded: @rpath/libint2.2.dylib`

```bash
conda activate protliginteract-full
python -m pip uninstall -y psi4 psi4-core
conda install -y -c conda-forge "psi4=1.9.*" "libint=2.9.*" "pcmsolver=1.2.3"
python -c "import psi4,glob,sys; print(psi4.__version__); print(glob.glob(sys.prefix + '/lib/libint2*.dylib'))"
python -c "import psi4; mol=psi4.geometry('0 1\nO\nH 1 0.96\nH 1 0.96 2 104.5\n'); print(psi4.energy('scf/sto-3g', molecule=mol))"
```

### B) NumPy/RDKit ABI mismatch (`rdkit` built for NumPy 1.x, runtime has NumPy 2.x)

Do not install ColabFold/DeepChem/JAX in `protliginteract-full`.

Repair:

```bash
conda activate protliginteract-full
conda install -y -c conda-forge "numpy>=1.26,<2" "rdkit>=2023.09" mdtraj qcelemental
pip check
```

### C) `Cannot uninstall jaxlib ... no RECORD file`

```bash
conda activate protliginteract-full
conda remove -y jax jaxlib || true
python -m pip install -U pip setuptools wheel
```

Then install ColabFold/JAX only in `protliginteract-optional`.

### D) AmberTools binaries not found in plugin

Confirm:

```bash
conda activate protliginteract-full
which antechamber parmchk2 tleap reduce
```

If missing, reinstall:

```bash
conda install -y -c conda-forge ambertools
```

### E) AutoDockTools scripts missing (`prepare_receptor4.py`, etc.)

On `osx-arm64` these legacy packages/scripts are frequently unavailable in conda.
The plugin can still run with `obabel` + `smina` paths (no AutoDockTools scripts).

If you must use them, install from a compatible x86_64 environment (often via Rosetta) and point the plugin to those script paths manually.

### F) Choosing the OpenMM platform (GPU or CPU)

Leonardo benchmarks the OpenMM platforms once (in a child process, so a crashing driver cannot take
PyMOL down) and runs MD, ABFE-lite, membranes and surfaces on the fastest one: CUDA on NVIDIA, HIP on
AMD, OpenCL on Apple Silicon and other GPUs. Every result reports the platform it ran on.

- NVIDIA and AMD GPUs run in **mixed precision**. Apple GPUs have no 64-bit arithmetic, so their
  OpenCL rejects mixed precision; Leonardo detects this and runs them in **single precision**.
- To force a platform, set `LEONARDO_OPENMM_PLATFORM` (or the older `PROTLIGINTERACT_OPENMM_PLATFORM`)
  to `CUDA`, `HIP`, `OpenCL`, `Metal` or `CPU` before starting PyMOL. A platform that is not installed
  is reported in Leonardo's debug log and the next one is used.

```bash
conda activate protliginteract-full
export LEONARDO_OPENMM_PLATFORM=CPU      # e.g. if a GPU driver misbehaves
pymol
```

To check a GPU install end to end, run `bash scripts/gpu_validation.sh` (see the script header).

## 10) Files

- Main env: `environments/environment.yml`
- Locked main env: `environments/environment-protliginteract-full.lock.yml`
- Optional env: `environments/environment-optional.yml`
- Installer script: `scripts/install_envs.sh`
- Full installer: `scripts/install_leonardo.sh`
- macOS double-click installer: `installers/shortcuts/Install Leonardo.command`
- macOS double-click launcher: `installers/shortcuts/Launch Leonardo.command`
- Linux installer: `installers/shortcuts/Install Leonardo Linux.sh`
- Linux launcher: `installers/shortcuts/Launch Leonardo Linux.sh`
- Windows WSL installer: `installers/shortcuts/Install Leonardo Windows.bat`
- Windows WSL launcher: `installers/shortcuts/Launch Leonardo Windows.bat`
- This guide: `docs/ENV_SETUP.md`
