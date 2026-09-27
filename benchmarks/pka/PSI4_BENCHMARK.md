# Psi4 HF/3-21g (PCM water, PM7/COSMO geometries) pKa benchmark

Calibration `psi4-2026.09.27`, written by `benchmarks/pka/run_pka_benchmark.py`.

Experimental values: IUPAC Digitized pKa Dataset v2.4 (Perrin 1965/1972, Serjeant & Dempsey
1979), CC BY-NC 4.0 - "Reproduced by permission of International Union of Pure and Applied
Chemistry". The dataset is downloaded at run time and not redistributed; this report only
contains aggregate statistics.

Molecules: 327 (218 calibration, 109 held-out test), 10 conformers per state. Calibration: robust (Theil-Sen) per-class slope,
shrinkage pseudo-count 20 (leave-one-out), fitted on 185 in-domain calibration molecules;
out-of-domain flags: acyl_substituted, cyclic_imine, empirical_only, ortho_heteroatom, polyaza_ring, resonance_acceptor, tertiary_aniline, unusual_group.

## Held-out test set

| Model | n | MAE | RMSE | within 1 unit |
|---|---|---|---|---|
| Psi4 HF/3-21g uncalibrated | n=109 | 3.71 | 4.92 | 26% |
| Psi4 HF/3-21g, global robust slope | n=109 | 1.63 | 2.08 | 40% |
| Psi4 HF/3-21g, per-class least squares (all molecules) | n=109 | 1.52 | 2.15 | 46% |
| Psi4 HF/3-21g, robust per-class slope with shrinkage (all classes) | n=109 | 1.53 | 2.02 | 40% |
| **Shipped: Psi4 HF/3-21g, PM7 where Psi4 HF/3-21g is not recommended** | n=109 | 0.90 | 1.27 | 66% |
| Empirical (class-typical pKa) | n=109 | 1.37 | 1.77 | 50% |
| Psi4 HF/3-21g (all classes), in-domain sites | n=95 | 1.39 | 1.74 | 41% |
| Psi4 HF/3-21g (all classes), out-of-domain sites (warned) | n=14 | 2.51 | 3.37 | 36% |
| Psi4 HF/3-21g (all classes), any warning flag | n=24 | 2.36 | 3.04 | 29% |
| Psi4 HF/3-21g (all classes), no warning flag | n=85 | 1.30 | 1.62 | 44% |

## Per class

`cal n`: in-domain molecules the slope was fitted on (of `train n`). Psi4 HF/3-21g vs PM7 is
decided on the calibration set (leave-one-out MAE), never on the test set. Classes with
fewer than 5 test molecules are not statistically meaningful.

| Class | train n | cal n | slope | LOO MAE Psi4 HF/3-21g / PM7 | Psi4 HF/3-21g recommended | test Psi4 HF/3-21g: n, MAE, RMSE, within 1 | test empirical MAE |
|---|---|---|---|---|---|---|---|
| amidine | 19 | 17 | 0.35 | 1.74 / 1.10 | **no** | n=10, 1.29, 1.39, 30% | 1.49 |
| amine1 | 20 | 20 | 0.23 | 1.36 / 0.55 | **no** | n=10, 1.49, 1.60, 20% | 1.27 |
| amine2 | 19 | 19 | 0.19 | 1.25 / 0.97 | **no** | n=10, 1.70, 2.00, 30% | 1.29 |
| amine3 | 19 | 19 | 0.30 | 1.69 / 1.55 | **no** | n=10, 2.15, 2.52, 20% | 1.23 |
| aniline | 20 | 8 | 0.13 | 2.54 / 1.28 | **no** | n=10, 1.69, 1.84, 20% | 1.38 |
| carboxylic_aliphatic | 20 | 20 | 0.23 | 1.08 / 1.10 | yes | n=10, 0.94, 1.05, 50% | 0.74 |
| carboxylic_aromatic | 20 | 20 | 0.22 | 0.57 / 0.43 | **no** | n=10, 0.69, 0.89, 70% | 0.57 |
| guanidine | 8 | 3 | 0.38 | 6.24 / 3.63 | **no** | n=4, 4.89, 5.61, 0% | 3.56 |
| imidazole | 20 | 19 | 0.37 | 1.47 / 0.59 | **no** | n=10, 1.02, 1.25, 60% | 1.84 |
| phenol | 20 | 13 | 0.18 | 1.53 / 0.55 | **no** | n=10, 2.52, 2.91, 30% | 2.15 |
| pyridine | 20 | 15 | 0.34 | 0.85 / 0.45 | **no** | n=10, 0.98, 1.28, 70% | 1.14 |
| thiol_aliphatic | 8 | 8 | 0.27 | 1.10 / 0.85 | **no** | n=3, 0.81, 0.94, 67% | 1.05 |
| thiol_aromatic | 5 | 4 | 0.25 | 0.22 / 0.35 | yes | n=2, 0.22, 0.24, 100% | 0.52 |

## Per warning flag (test set)

| Flag | n | MAE | RMSE | within 1 |
|---|---|---|---|---|
| acyl_substituted | n=2 | 6.28 | 6.88 | 0% |
| cyclic_imine | n=2 | 1.05 | 1.21 | 50% |
| ortho_heteroatom | n=2 | 1.64 | 2.00 | 50% |
| polyaza_ring | n=2 | 2.22 | 2.29 | 0% |
| resonance_acceptor | n=6 | 2.48 | 2.89 | 33% |
| tertiary_amine | n=10 | 2.15 | 2.52 | 20% |
| tertiary_aniline | n=2 | 0.47 | 0.49 | 100% |

## Shipped Psi4 HF/3-21g vs the other method's shipped model (same test molecules)

Other method: `pm7_benchmark_details.json`.

| Class | n | Psi4 HF/3-21g MAE | other MAE |
|---|---|---|---|
| amidine | 10 | 0.50 | 0.50 |
| amine1 | 10 | 0.39 | 0.39 |
| amine2 | 10 | 0.85 | 0.85 |
| amine3 | 10 | 1.23 | 1.23 |
| aniline | 10 | 0.99 | 0.99 |
| carboxylic_aliphatic | 10 | 0.94 | 0.87 |
| carboxylic_aromatic | 10 | 0.28 | 0.28 |
| guanidine | 4 | 2.25 | 2.25 |
| imidazole | 10 | 0.92 | 0.92 |
| phenol | 10 | 1.12 | 1.12 |
| pyridine | 10 | 1.28 | 1.28 |
| thiol_aliphatic | 3 | 1.32 | 1.32 |
| thiol_aromatic | 2 | 0.22 | 0.60 |
| **all** | 109 | 0.90 | 0.90 |
