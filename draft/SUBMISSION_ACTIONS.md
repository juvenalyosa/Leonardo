# Leonardo BMC submission actions

This file tracks manuscript blockers that require external assets, public release
metadata or additional benchmark runs.  The current manuscript draft compiles and
the local workflow-verification assets regenerate, but these items should be completed
before journal submission.

## Public release metadata

- Create the public repository or anonymous reviewer repository.
- Archive the exact submission release in Zenodo or an equivalent permanent
  software archive.
- Replace the pending availability lines in `draft/manuscript/main.tex` with the
  public URL, archive DOI and peer-review access location.

## UI evidence

- Capture real Leonardo screenshots, not mockups:
  - main PyMOL/Leonardo launcher or module menu,
  - interaction-analysis or docking workflow window with object choices visible,
  - one result window or report popup showing parsed outputs.
- Place PNG screenshots in `screenshots/`.
- Add one manuscript figure or supplementary figure only after the real
  screenshots are available.

## Interaction-analysis benchmark

- Install or add a reproducible optional dependency path for PLIP or ProLIF.
- Compare Leonardo contact calls against PLIP/ProLIF for 3PTB/BEN.
- Add 3-5 additional protein-ligand complexes covering interaction classes not
  present in 3PTB/BEN, especially halogen bonds, pi-stacking, cation-pi contacts
  and metal coordination.
- Report agreement by interaction class and keep all disagreement cases in a
  reviewer-visible CSV table.

## Pocket/redocking verification

- For pocket docking, add crystal-ligand overlap or binding-site residue overlap
  for the selected pocket.
- Keep smina internal RMSD separate from template/crystal RMSD in all tables and
  captions.
- Either add a small multi-target redocking table or keep the claim explicitly
  limited to the current single-target workflow check.

## Free-energy and reaction-path scope

- Keep ABFE-lite as a guided-exit diagnostic workflow unless a real minimal ABFE
  calculation with standard-state/restraint corrections and uncertainty is added.
- Keep the current MOPAC result described as a synthetic capped-cluster
  reaction.  A protein-derived enzymatic example requires a real PDB-derived
  microcluster with residue provenance, caps and endpoint-connectivity checks.
- Keep the OpenMM alanine-dipeptide result described as an internal backend and
  plotting-stability check unless a replicated, uncertainty-calibrated CV/FES
  example is generated.
