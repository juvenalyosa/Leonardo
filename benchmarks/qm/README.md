# QM smoke tests

This folder provides terminal-level smoke tests for the Psi4 workflow logic before plugin wiring.

## Run

From repository root:

```bash
python3 benchmarks/qm/qm_smoke_tests.py --out-dir benchmarks/qm/results
```

If `psi4` is installed and on `PATH`, a real minimal Psi4 optimization test is executed.
If not, runtime Psi4 is marked as skipped while the non-Psi4 logic tests still run.

To force skipping runtime Psi4 even when installed:

```bash
python3 benchmarks/qm/qm_smoke_tests.py --skip-psi4-runtime --out-dir benchmarks/qm/results
```

## What is tested

- ESP grid generation (`psi4_generate_esp_grid` behavior)
- Psi4 script generation for operations:
  - optimize
  - frequencies
  - stationary
  - ts
  - irc
  - esp_fit
- MOL2 charge rewriting (`psi4_write_charged_mol2` behavior)
- Optional real Psi4 execution (small water optimize)

## Outputs

- Artifacts and generated inputs in `benchmarks/qm/results/`
- Machine-readable summary JSON in `benchmarks/qm/results/qm_smoke_summary.json`

## Real-ligand ESP-fit test

Run on the bundled example ligand with Psi4 and write a charged MOL2:

```bash
/opt/anaconda3/bin/conda run -n bgt-openmm82 \
  python benchmarks/qm/test_esp_fit_real_ligand.py \
  --ligand examples/3ptb_ben/BEN.sdf \
  --out-dir benchmarks/qm/esp_real \
  --method hf \
  --basis '6-31G*'
```

Outputs are written under `benchmarks/qm/esp_real/<ligand>_espfit/` including:
- `ligand_espfit.mol2`
- `espfit_report.json`
- `psi4_output.out`
- `psi4_result.json`
