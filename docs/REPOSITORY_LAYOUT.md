# Repository layout

Leonardo uses a canonical `src/leonardo/` package. The PyMOL plugin and older
third-party scripts can still import root-level compatibility aliases.

## Top-level areas

| Area | Contents | Policy |
| --- | --- | --- |
| `src/leonardo/` | Runtime implementations and package configuration | Canonical production package; internal imports must be relative |
| `*.py` (root) | PyMOL entry points, CLI launchers, and compatibility aliases | Compatibility only; reusable implementation belongs under `src/leonardo/` |
| `tests/` | Unit, integration, and packaging tests | Every runtime change should add or update a test |
| `qm/`, `test_abfe_3ptb_ben/`, `redocking_real_*`, `benchmark_data/` | Small scientific fixtures and reproducible examples | Inputs and metadata only; generated outputs go under `draft/` or a user-selected output directory |
| `installers/` | Platform-specific installers | Thin wrappers around the root installer |
| `jsme/` | Bundled JSME web asset | Third-party asset; do not edit without updating its license notice |
| `docs/` | Maintainer and architecture documentation | Design decisions and migration notes live here |
| `dist/` | Release ZIP and checksum | Generated; rebuild with `./build_leonardo_zip.sh` |
| `draft/` | Manuscript/validation artifacts | Generated or review-specific; never import from runtime code |

## Rules

1. New reusable scientific logic belongs in a `*_runtime.py` module.
2. UI/PyMOL/Qt code stays in `code_v2.py`; runtime modules must not import Qt or PyMOL.
3. Tests use repository fixtures and temporary output directories.
4. Environment files are the reproducibility contract; document non-obvious pins.

## Migration plan

Runtime implementations have moved to `src/leonardo/`. Root-level aliases must
remain temporarily because existing PyMOL ZIPs, external scripts, and tests use
the historical import paths. Command-line entry points are available under
`leonardo.*`, while root scripts remain stable launch targets.
