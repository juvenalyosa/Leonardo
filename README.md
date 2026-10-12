# Leonardo

<p align="center">
  <img src="assets/logo.png" alt="Leonardo logo" width="220">
</p>

Leonardo is a PyMOL workbench for structural bioinformatics, docking,
molecular dynamics, quantum chemistry, and reproducible protein–ligand
workflows.

Current version: **0.2.0a7** (`2026-10-01-r2`)<br>
Status: **pre-release for testing and manuscript verification**

## Install (end users)

Download **`Leonardo-Installer-<version>.zip`** from the release page, unzip it and:

- **macOS:** double-click `Install Leonardo (macOS).command` (the first time: right-click > Open).
- **Linux:** in that folder run `bash install-leonardo-linux.sh`.

It installs everything (conda, PyMOL, OpenMM, AmberTools, Psi4, MOPAC, MODELLER, ...), asks for
the MODELLER license key, and creates a **Leonardo** app/menu entry. Step-by-step guide in
English and Spanish: [docs/INSTALL.md](docs/INSTALL.md).

## Install (developers)

Requirements: macOS or Linux (Windows 11 through WSL2/WSLg), Git, about 10 GB of disk.
An NVIDIA GPU is optional; CPU execution is supported.

Authenticate with GitHub first because this repository is private, then clone it and run the
same installer on your working copy (it installs Miniforge if no conda is found):

```bash
git clone https://github.com/juvenalyosa/Leonardo.git
cd Leonardo
bash installers/install_leonardo.sh
```

Leonardo uses `protliginteract-full` (required) and, for local ColabFold and ML tools,
`protliginteract-optional`. The previous developer workflow (editable install, smoke tests and
`dist/Leonardo.zip`) is still available:

```bash
./scripts/install_leonardo.sh                 # add --with-optional / --recreate as needed
```

## Launch

```bash
./scripts/launch_leonardo.sh
```

In PyMOL, open `Plugins > Leonardo`.

## Platform shortcuts

- macOS: double-click `installers/shortcuts/Install Leonardo.command`, then
  `installers/shortcuts/Launch Leonardo.command`.
- Linux: run `installers/shortcuts/Install Leonardo Linux.sh`, then
  `installers/shortcuts/Launch Leonardo Linux.sh`.
- Windows 11: install WSL2 Ubuntu with `wsl --install -d Ubuntu`, then use the
  two `.bat` files in `installers/shortcuts/`.

Windows 10 needs an external X server because it does not include WSLg.

## Manual installation

```bash
conda env create -f environments/environment.yml
conda activate protliginteract-full
python -m pip install -e . --no-build-isolation
./scripts/install_pymol_loader.sh
./scripts/build_leonardo_zip.sh
```

To create the optional environment:

```bash
conda env create -f environments/environment-optional.yml
```

The macOS lockfile at
`environments/environment-protliginteract-full.lock.yml` provides stricter
version pinning. Other platforms should normally use
`environments/environment.yml`.

## Validate

Activate the main environment and run:

```bash
conda activate protliginteract-full
leonardo-readiness --strict
make lint
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Check installer inputs without changing environments or user files:

```bash
./scripts/install_leonardo.sh --check-only --skip-zip --no-pymol-loader
```

## Install with PyMOL Plugin Manager

The startup loader is recommended. To install a standalone plugin ZIP instead:

```bash
./scripts/build_leonardo_zip.sh
```

Then open PyMOL from `protliginteract-full`, select
`Plugin > Plugin Manager > Install New Plugin`, and choose
`dist/Leonardo.zip`.

## Repository layout

```text
Leonardo/
├── src/leonardo/        Canonical runtime package
├── legacy/              PyMOL UI and historical import compatibility
├── packaging/pymol/     PyMOL plugin entry point
├── assets/              UI, images, and bundled JSME files
├── tests/               Automated test suite
├── benchmarks/          Scientific fixtures and validation workflows
├── examples/            Small input files and runnable examples
├── environments/        Conda specifications and lockfile
├── scripts/             Build, install, launch, and maintenance commands
├── installers/          Platform-specific installers and shortcuts
├── docs/                Architecture and maintainer documentation
├── draft/               Manuscript and generated validation artifacts
└── dist/                Reproducible PyMOL release archive
```

New runtime implementation belongs in `src/leonardo/`. The files in `legacy/`
preserve existing PyMOL and external-script imports while the UI is migrated
into smaller modules. See
[Repository layout](docs/REPOSITORY_LAYOUT.md) and
[Architecture](docs/ARCHITECTURE.md).

## Development

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

## Scientific and licensing notes

Leonardo orchestrates established scientific engines; it does not replace
method-specific validation. Results depend on inputs, protonation states,
force fields, sampling, engine versions, and convergence criteria.

Third-party engines may have separate licenses or usage restrictions. See
[Third-party licenses](THIRD_PARTY_LICENSES.md).

Citation metadata is provided in [CITATION.cff](CITATION.cff). Leonardo source
code is distributed under the [MIT License](LICENSE).

Author: Juvenal Yosa, PhD<br>
Support and bug reports: `thinkai.col@gmail.com`
