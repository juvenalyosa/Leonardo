# Repository layout

Leonardo is currently in a compatibility-first transition. The PyMOL plugin
still loads Python modules from the repository root, so runtime modules remain
there until the package migration is complete.

## Top-level areas

| Area | Contents | Policy |
| --- | --- | --- |
| `*.py` (root) | PyMOL entry points and runtime modules | Production code; keep UI in `code_v2.py` and `ProtLigInteract.py` compatibility glue thin |
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

The migration has started with `src/leonardo/` and canonical namespace entry
points for `smina_runtime`, `vs_runtime`, `leadopt_runtime`,
`interaction_typing_runtime`, `alphafold_runtime`, `psi4_runtime`,
`abfe_runtime`, `redock_runtime`, `openmm_dynamics_runtime`,
`openmm_qmmm_runtime`, `modeler_runtime`, and `bioseq_runtime`.
The same namespace also covers structure fetching, small molecules, ligand
editing/refinement, interaction and trajectory analysis, task monitoring,
docking orchestration, mutation, QM/MM, and scientific-readiness helpers.
Root-level modules remain compatibility implementations for now. Each
subsequent runtime module will move only after its import and packaging tests
pass.

Command-line entry points are also available under `leonardo.*`; root scripts
remain stable launch targets for existing installers.
