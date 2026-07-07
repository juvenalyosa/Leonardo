# Leonardo BMC Bioinformatics Draft

This directory contains the current publication draft and reproducible assets.

## Main files

- `manuscript/main.tex`: Springer Nature/BMC LaTeX manuscript draft.
- `manuscript/references.bib`: verified bibliography entries.
- `references/reference_verification.md`: DOI/URL verification log.
- `scripts/generate_bmc_validation_assets.py`: reproducible workflow-check figure/table generator.
- `results/workflow_verification_summary.json`: numerical summary used in the manuscript.
- `results/artifact_manifest.csv`: file-level byte counts and SHA-256 checksums for manuscript assets.
- `results/pytest_validation.xml`: repository test report used by the manuscript.
- `figures/figure_*.pdf` and `figures/figure_*.png`: manuscript figures.
- `templates/springer-nature-latex-template.zip`: official Springer Nature LaTeX package downloaded for this draft.

## Current Workflow-Check Assets

- Figure 1: Leonardo workflow architecture and contribution boundary.
- Figure 2: pipeline comparison statistics.
- Figure 3: module evidence matrix for adapter modules and native pipelines.
- Figure 4: 3PTB/benzamidine interaction-contact analysis.
- Figure 5: single-target 1STP/biotin redocking workflow check.
- Figure 6: local ligand-library QC descriptor analysis.
- Figure 7: OpenMM alanine-dipeptide CV/FES backend and plotting check.
- Figure 8: publication-oriented MOPAC PM7 synthetic capped-cluster QST2/QST3/TS/IRC example.
- Figure 9: Psi4/MOPAC QST2/QST3/IRC reaction-path workflow check.

## Items still needed before submission

- Final author names, affiliations and contact email.
- Public repository URL.
- Archived DOI, ideally Zenodo.
- Confirmed license and non-academic restrictions.
- Screenshots of the professional Leonardo UI for supplementary material.
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

The manuscript figures can also be regenerated directly inside the validated environment:

```bash
conda activate protliginteract-full
python -m pip install -e . --no-build-isolation
python draft/scripts/generate_bmc_validation_assets.py
```
