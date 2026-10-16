# Repository layout

Leonardo uses a canonical `src/leonardo/` package. The PyMOL plugin and older
third-party scripts can still import compatibility aliases installed from
`legacy/`.

## Top-level areas

| Area | Contents | Policy |
| --- | --- | --- |
| `src/leonardo/` | Runtime implementations and package configuration | Canonical production package; internal imports must be relative |
| `legacy/` | PyMOL UI controller and historical top-level aliases | Compatibility only; reusable implementation belongs under `src/leonardo/` |
| `packaging/pymol/` | PyMOL plugin entry point | Packaged into the root of the plugin ZIP |
| `assets/` | Qt UI, images, and icons | Runtime resources |
| `tests/` | Unit, integration, and packaging tests | Every runtime change should add or update a test |
| `benchmarks/` | Scientific fixtures, QM checks, ABFE and redocking validation | Inputs and metadata only; generated outputs go under `draft/` or a user-selected output directory |
| `examples/` | Small example inputs and logs | Keep examples minimal and documented |
| `scripts/` | Build, installation, launch, and maintenance commands | Commands resolve the repository root themselves |
| `environments/` | Conda specifications and lockfiles | Reproducibility contract |
| `installers/` | Platform-specific installers and shortcuts | Thin wrappers around `scripts/install_leonardo.sh` |
| `docs/` | Maintainer and architecture documentation | Design decisions and migration notes live here |
| `dist/` | Release ZIP and checksum | Generated; rebuild with `./scripts/build_leonardo_zip.sh` |
| `draft/` | Manuscript/validation artifacts | Generated or review-specific; never import from runtime code |

## Rules

1. New reusable scientific logic belongs in a `*_runtime.py` module.
2. UI/PyMOL/Qt compatibility code stays in `legacy/code_v2.py`; runtime modules must not import Qt or PyMOL.
3. Tests use repository fixtures and temporary output directories.
4. Environment files are the reproducibility contract; document non-obvious pins.

## Migration plan

Runtime implementations have moved to `src/leonardo/`. Aliases in `legacy/`
remain temporarily because existing PyMOL ZIPs, external scripts, and tests use
the historical import paths. Command-line entry points are available under
`leonardo.*`; operational commands live in `scripts/`.
