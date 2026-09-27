# PM7/COSMO pKa benchmark

Calibration `pm7-2026.09.27`, written by `benchmarks/pka/run_pka_benchmark.py`.

Experimental values: IUPAC Digitized pKa Dataset v2.4 (Perrin 1965/1972, Serjeant & Dempsey
1979), CC BY-NC 4.0 - "Reproduced by permission of International Union of Pure and Applied
Chemistry". The dataset is downloaded at run time and not redistributed; this report only
contains aggregate statistics.

Molecules: 330 (221 calibration, 109 held-out test), 10 conformers per state. Calibration: robust (Theil-Sen) per-class slope,
shrinkage pseudo-count 5 (leave-one-out), fitted on 188 in-domain calibration molecules;
out-of-domain flags: acyl_substituted, cyclic_imine, empirical_only, ortho_heteroatom, polyaza_ring, resonance_acceptor, tertiary_aniline, unusual_group.

## Held-out test set

| Model | n | MAE | RMSE | within 1 unit |
|---|---|---|---|---|
| PM7 uncalibrated | n=109 | 1.46 | 2.06 | 49% |
| PM7, global robust slope | n=109 | 1.00 | 1.45 | 66% |
| PM7, per-class least squares (all molecules) | n=109 | 1.04 | 1.48 | 63% |
| PM7, robust per-class slope with shrinkage (all classes) | n=109 | 0.99 | 1.44 | 66% |
| **Shipped: PM7, empirical where PM7 is not recommended** | n=109 | 0.90 | 1.27 | 67% |
| Empirical (class-typical pKa) | n=109 | 1.37 | 1.77 | 50% |
| shipped model, in-domain sites | n=95 | 0.88 | 1.30 | 69% |
| shipped model, out-of-domain sites (warned) | n=14 | 1.72 | 2.17 | 43% |
| shipped model, any warning flag | n=24 | 1.89 | 2.39 | 42% |
| shipped model, no warning flag | n=85 | 0.73 | 1.02 | 73% |

## Per class

`cal n`: in-domain molecules the slope was fitted on (of `train n`). PM7 vs empirical is
decided on the calibration set (leave-one-out MAE), never on the test set. Classes with
fewer than 5 test molecules are not statistically meaningful.

| Class | train n | cal n | slope | LOO MAE PM7 / empirical | PM7 recommended | test PM7: n, MAE, RMSE, within 1 | test empirical MAE |
|---|---|---|---|---|---|---|---|
| amidine | 20 | 18 | 0.57 | 1.11 / 1.68 | yes | n=10, 0.50, 0.70, 70% | 1.49 |
| amine1 | 20 | 20 | 0.68 | 0.55 / 1.25 | yes | n=10, 0.39, 0.57, 80% | 1.27 |
| amine2 | 20 | 20 | 0.77 | 0.93 / 1.02 | yes | n=10, 0.85, 1.00, 70% | 1.29 |
| amine3 | 20 | 20 | 0.87 | 1.64 / 1.53 | **no** | n=10, 2.12, 2.67, 40% | 1.23 |
| aniline | 20 | 8 | 0.41 | 1.28 / 2.08 | yes | n=10, 0.99, 1.42, 70% | 1.38 |
| carboxylic_aliphatic | 20 | 20 | 0.62 | 1.10 / 1.33 | yes | n=10, 0.87, 1.00, 60% | 0.74 |
| carboxylic_aromatic | 20 | 20 | 0.56 | 0.43 / 0.74 | yes | n=10, 0.28, 0.35, 100% | 0.57 |
| guanidine | 8 | 3 | 0.81 | 3.63 / 4.97 | yes | n=4, 2.25, 3.09, 50% | 3.56 |
| imidazole | 20 | 19 | 0.98 | 0.59 / 1.86 | yes | n=10, 0.91, 1.28, 80% | 1.84 |
| phenol | 20 | 13 | 0.38 | 0.55 / 1.81 | yes | n=10, 1.12, 1.28, 50% | 2.15 |
| pyridine | 20 | 15 | 0.80 | 0.45 / 1.37 | yes | n=10, 1.28, 1.65, 50% | 1.14 |
| thiol_aliphatic | 8 | 8 | 1.07 | 0.85 / 0.99 | yes | n=3, 1.32, 1.45, 33% | 1.05 |
| thiol_aromatic | 5 | 4 | 0.67 | 0.35 / 0.50 | yes | n=2, 0.60, 0.61, 100% | 0.52 |

## Per warning flag (test set)

| Flag | n | MAE | RMSE | within 1 |
|---|---|---|---|---|
| acyl_substituted | n=2 | 2.51 | 3.43 | 50% |
| cyclic_imine | n=2 | 0.22 | 0.23 | 100% |
| ortho_heteroatom | n=2 | 2.38 | 2.43 | 0% |
| polyaza_ring | n=2 | 1.20 | 1.38 | 50% |
| resonance_acceptor | n=6 | 1.74 | 1.89 | 33% |
| tertiary_amine | n=10 | 2.12 | 2.67 | 40% |
| tertiary_aniline | n=2 | 2.93 | 2.93 | 0% |
