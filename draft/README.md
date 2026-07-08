# Leonardo BMC Bioinformatics Draft

This directory contains the current publication draft and reproducible assets.

## Main files

- `manuscript/main.tex`: Springer Nature/BMC LaTeX manuscript draft.
- `manuscript/references.bib`: verified bibliography entries.
- `references/reference_verification.md`: DOI/URL verification log.
- `scripts/generate_bmc_validation_assets.py`: reproducible workflow-check figure/table generator.
- `results/workflow_verification_summary.json`: numerical summary used in the manuscript.
- `results/artifact_manifest.csv`: file-level byte counts and SHA-256 checksums for manuscript assets.
- `results/release_manifest.csv`: short reviewer-release manifest covering ZIP, PDF, licenses, local JSME assets, environment files and required fixtures. The GitHub release workflow writes the published manifest after tag checkout so the release asset can include the exact source commit.
- `results/pytest_validation.xml`: repository test report used by the manuscript.
- `figures/figure_*.pdf` and `figures/figure_*.png`: manuscript figures.
- `templates/springer-nature-latex-template.zip`: official Springer Nature LaTeX package downloaded for this draft.

## Current Workflow-Check Assets

- Figure 1: Leonardo workflow architecture and contribution boundary.
- Figure 2: real Leonardo suite interface screenshot.
- Figure 3: pipeline inventory and workflow-evidence summary.
- Figure 4: module evidence matrix for adapter modules and native pipelines.
- Figure 5: pocket-docking workflow check with fpocket/smina outputs.
- Figure 6: PyMOL-rendered 3PTB/benzamidine protein-ligand interaction scene.
- Figure 7: 3PTB/benzamidine interaction-contact analysis.
- Figure 8: single-target 1STP/biotin redocking workflow check.
- Figure 9: local ligand-library QC descriptor analysis.
- Figure 10: OpenMM alanine-dipeptide CV diagnostic-grid backend and plotting check.
- Figure 11: guided-exit state-gating diagnostic using the historical ABFE-lite module.
- Figure 12: MOPAC PM7 synthetic capped-cluster QST2/QST3/TS/IRC example.
- Figure 13: Psi4/MOPAC QST2/QST3/IRC reaction-path workflow check.

## Items still needed before submission

- Final author names, affiliations and contact email.
- Reviewer repository URL: `https://github.com/juvenalyosa/Leonardo` (currently private; grant reviewer access or make public at submission).
- Archived DOI, ideally Zenodo, for the exact final submission release.
- Confirmed license and non-academic restrictions.
- Additional real workflow screenshots for supplementary material, especially object-selection and result/report pop-up windows.
- Broader external-user testing or benchmark expansion if we want stronger claims than "software platform/workflow verification."

## Reproduction

From the repository root, the preferred reviewer command is:

```bash
./install_leonardo.sh --paper-assets
```

A non-mutating installer preflight is also available:

```bash
./install_leonardo.sh --check-only --skip-zip --no-pymol-loader
```

Inside an already installed Leonardo environment, the compact reviewer workflow is:

```bash
make reviewer-check
```

The manuscript figures can also be regenerated directly:

```bash
conda activate protliginteract-full
python -m pip install -e . --no-build-isolation
python draft/scripts/generate_bmc_validation_assets.py
```
