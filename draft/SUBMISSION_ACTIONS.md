# Leonardo BMC submission actions

This file tracks manuscript blockers that require external assets, public release
metadata or additional benchmark runs.  The current manuscript draft compiles and
the local workflow-verification assets regenerate, but these items should be completed
before journal submission.

## Public release metadata

- Status: repository created; permanent archive pending.
- Reviewer repository: `https://github.com/juvenalyosa/Leonardo` (currently
  private).
- Grant reviewer/editor access while private, or make the repository public for
  submission.
- Archive the exact submission release in Zenodo or an equivalent permanent
  software archive and replace the placeholder archive line with the DOI.

## UI evidence

- Status: partially closed.  `assets/leonardo.png` is included as the suite-level UI
  figure.
- Capture real Leonardo docs/screenshots, not mockups:
  - interaction-analysis or docking workflow window with object choices visible,
  - one result window or report popup showing parsed outputs.
- Place PNG docs/screenshots in `docs/screenshots/`.
- Add one manuscript figure or supplementary figure only after the real
  docs/screenshots are available.

## Interaction-analysis benchmark

- Status: pending optional dependency and expanded dataset.
- `draft/scripts/run_interaction_external_benchmark.py` writes a reviewer-facing
  status/comparison table.  It currently records missing PLIP/ProLIF rather than
  fabricating agreement values.
- Install or add a reproducible optional dependency path for PLIP or ProLIF.
- Compare Leonardo contact calls against PLIP/ProLIF for 3PTB/BEN.
- Add 3-5 additional protein-ligand complexes covering interaction classes not
  present in 3PTB/BEN, especially halogen bonds, pi-stacking, cation-pi contacts
  and metal coordination.
- Report agreement by interaction class and keep all disagreement cases in a
  reviewer-visible CSV table.

## Pocket/redocking verification

- Status: pocket overlap closed for one crystallographic case; broader benchmark
  remains optional.
- Pocket docking now includes a 1STP/biotin fpocket crystal-site overlap table:
  `draft/results/pocket_1stp_crystal_overlap.csv`.
- Keep smina internal RMSD separate from template/crystal RMSD in all tables and
  captions.
- Either add a small multi-target redocking table or keep the claim explicitly
  limited to the current single-target workflow check.

## Free-energy and reaction-path scope

- Status: manuscript scope is controlled; production benchmarks remain future
  work.
- Keep the historical ABFE-lite module framed as a guided-exit diagnostic
  workflow unless a real minimal ABFE calculation with standard-state/restraint
  corrections and uncertainty is added.
- Keep the current MOPAC result described as a synthetic capped-cluster
  reaction.  A protein-derived enzymatic example requires a real PDB-derived
  microcluster with residue provenance, caps and endpoint-connectivity checks.
- Keep the OpenMM alanine-dipeptide result described as an internal backend and
  plotting-stability check unless a replicated, uncertainty-calibrated CV/FES
  example is generated.
