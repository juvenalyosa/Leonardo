# Leonardo

<p align="center">
  <img src="logo.png" alt="Leonardo logo" width="220">
</p>

Leonardo is a PyMOL-based workbench for structural bioinformatics, molecular
modeling, docking, molecular dynamics, quantum chemistry, and reproducible
protein–ligand workflows.

Current version: **0.2.0a5** (`2026-07-08-r5`)<br>
Status: **pre-release for testing and manuscript verification**

Author: Juvenal Yosa, PhD<br>
Support and bug reports: `thinkai.col@gmail.com`

## Main capabilities

- Protein–ligand and protein–nucleic-acid interaction analysis
- Pocket detection, docking, redocking, and virtual screening
- Ligand sourcing, editing, protonation, charge assignment, and preparation
- Hit-to-lead ranking and optional OpenMM refinement
- OpenMM molecular dynamics and guided-exit diagnostic workflows
- Psi4 and MOPAC quantum-chemistry workflows
- Optional OpenMM/MOPAC QM/MM helper workflows
- AlphaFold/ColabFold result generation and inspection
- MODELLER-based protein repair and mutation modeling
- Sequence analysis for simulation planning
- Project saving, task monitoring, plotting, and report generation

Leonardo orchestrates established scientific engines; it does not replace
method-specific validation. Production free-energy calculations, docking
benchmarks, QM accuracy, protonation choices, and simulation convergence remain
the user's responsibility.

## Requirements

### Supported systems

- macOS
- Linux
- Windows 11 through WSL2/WSLg (experimental)

### Required host software

- Conda, Miniforge, or Mambaforge
- Git
- A Qt-enabled PyMOL installation (provided by the main Conda environment)
- Enough disk space for Conda environments and scientific output files

CPU execution is supported. An NVIDIA GPU is optional and mainly useful for
local ColabFold or accelerated OpenMM workloads.

## Environments

Leonardo deliberately uses two Conda environments:

- `protliginteract-full`: required environment for PyMOL and the main scientific stack
- `protliginteract-optional`: optional environment for ColabFold/JAX, DeepChem, CatBoost, and XGBoost

Always launch Leonardo/PyMOL from `protliginteract-full`.

## Recommended installation

Clone the repository and enter it:

```bash
git clone https://github.com/juvenalyosa/Leonardo.git
cd Leonardo
```

The repository is private, so GitHub authentication must already be configured.

Run the main installer:

```bash
./install_leonardo.sh
```

The installer:

1. creates or updates `protliginteract-full`;
2. installs Leonardo in editable mode;
3. validates core dependencies;
4. installs the PyMOL startup loader;
5. builds `dist/Leonardo.zip`.

Launch Leonardo:

```bash
./launch_leonardo.sh
```

Then open `Plugins > Leonardo` in PyMOL.

### Install optional ML/ColabFold tools

```bash
./install_leonardo.sh --with-optional
```

To recreate both environments from scratch:

```bash
./install_leonardo.sh --recreate --with-optional
```

## Platform shortcuts

### macOS

Double-click:

1. `Install Leonardo.command`
2. `Launch Leonardo.command`

### Linux

```bash
./installers/linux/install_leonardo_linux.sh
./launch_leonardo.sh
```

### Windows through WSL2

Install WSL2 Ubuntu first:

```powershell
wsl --install -d Ubuntu
```

Then use:

1. `Install Leonardo Windows.bat`
2. `Launch Leonardo Windows.bat`

Windows 11 uses WSLg for the PyMOL interface. Windows 10 requires an external
X server.

## Manual installation

Create the main environment:

```bash
conda env create -f environment.yml
conda activate protliginteract-full
python -m pip install -e . --no-build-isolation
./install_pymol_loader.sh
./build_leonardo_zip.sh
```

Create the optional environment only when needed:

```bash
conda env create -f environment-optional.yml
```

Update existing environments:

```bash
conda env update -n protliginteract-full -f environment.yml --prune
conda env update -n protliginteract-optional -f environment-optional.yml --prune
```

The macOS lockfile `environment-protliginteract-full.lock.yml` provides a more
strictly pinned installation. On other platforms, use `environment.yml`.

## ColabFold wrapper

PyMOL stays in the main environment while ColabFold runs in the optional one.
Create a wrapper if local ColabFold is required:

```bash
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/colabfold_batch_pli" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec conda run -n protliginteract-optional --no-capture-output colabfold_batch "$@"
EOF
chmod +x "$HOME/.local/bin/colabfold_batch_pli"
```

Add `~/.local/bin` to `PATH` and select `colabfold_batch_pli` in Leonardo's
AlphaFold settings.

## Validate the installation

Activate the main environment:

```bash
conda activate protliginteract-full
```

Run the readiness report:

```bash
leonardo-readiness --strict
```

Run lint and tests:

```bash
make lint
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Validate without modifying environments or PyMOL startup files:

```bash
./install_leonardo.sh --check-only --skip-zip --no-pymol-loader
```

## Install through PyMOL's Plugin Manager

The startup loader is recommended. Alternatively:

1. Build the plugin archive:

   ```bash
   ./build_leonardo_zip.sh
   ```

2. Launch PyMOL from `protliginteract-full`.
3. Open `Plugin > Plugin Manager > Install New Plugin`.
4. Select `dist/Leonardo.zip`.

## Repository structure

```text
Leonardo/
├── src/leonardo/       Canonical Python runtime package
├── tests/              Unit, integration, and packaging tests
├── installers/         Platform-specific installers
├── benchmark_data/     Small reproducible scientific fixtures
├── qm/                 Quantum-workflow examples and validation
├── docs/               Maintainer documentation
├── draft/              Manuscript and validation artifacts
├── dist/               Generated PyMOL release archive
├── code_v2.py          PyMOL/Qt controller and UI orchestration
└── *_runtime.py        Legacy compatibility import aliases
```

New runtime implementation belongs under `src/leonardo/`. Root-level runtime
modules are compatibility aliases for existing PyMOL installations and external
scripts. See [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md) and
[ARCHITECTURE.md](ARCHITECTURE.md).

## Development

Use the main environment:

```bash
conda activate protliginteract-full
python -m pip install -e . --no-build-isolation
make lint
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Before publishing a release:

```bash
make reviewer-check
```

Runtime modules should not import Qt or PyMOL. UI-specific work remains in
`code_v2.py` until it is migrated into smaller UI modules.

## Common problems

### Psi4 fails with `libint` or `pcmsolver` errors

Reinstall the tested stack:

```bash
conda activate protliginteract-full
conda install -y -c conda-forge "psi4=1.9.*" "libint=2.9.*" "pcmsolver=1.2.3"
```

### RDKit, pandas, SciPy, or NumPy ABI errors

Keep NumPy below version 2 and reinstall through Conda:

```bash
conda install -y -c conda-forge "numpy=1.26.4" "pandas<3" scipy "rdkit>=2023.09"
```

### AmberTools executables are missing

```bash
conda install -y -c conda-forge ambertools
which antechamber parmchk2 tleap reduce
```

### PyMOL opens without the Leonardo menu

```bash
conda activate protliginteract-full
./install_pymol_loader.sh
./launch_leonardo.sh
```

### OpenMM is unstable on macOS

Force the CPU platform:

```bash
export PROTLIGINTERACT_OPENMM_PLATFORM=CPU
./launch_leonardo.sh
```

## Scientific and licensing notes

Leonardo is an orchestration and visualization layer. Its outputs depend on the
selected inputs, protonation states, force fields, sampling, engine versions,
and convergence criteria.

Third-party engines may have separate licenses or usage restrictions. See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff).

Leonardo source code is distributed under the [MIT License](LICENSE).
