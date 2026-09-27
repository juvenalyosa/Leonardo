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
| **PM7, robust per-class slope with shrinkage (shipped)** | n=109 | 0.99 | 1.44 | 66% |
| Empirical (class-typical pKa) | n=109 | 1.37 | 1.77 | 50% |
| shipped model, in-domain sites | n=95 | 0.88 | 1.30 | 69% |
| shipped model, out-of-domain sites (warned) | n=14 | 1.72 | 2.17 | 43% |
| shipped model, any warning flag | n=24 | 1.89 | 2.39 | 42% |
| shipped model, no warning flag | n=85 | 0.73 | 1.02 | 73% |

## Per class (shipped model, test set)

| Class | train n | slope | test n | MAE | RMSE | within 1 |
|---|---|---|---|---|---|---|
| amidine | 20 | 0.57 | n=10 | 0.50 | 0.70 | 70% |
| amine1 | 20 | 0.68 | n=10 | 0.39 | 0.57 | 80% |
| amine2 | 20 | 0.77 | n=10 | 0.85 | 1.00 | 70% |
| amine3 | 20 | 0.87 | n=10 | 2.12 | 2.67 | 40% |
| aniline | 20 | 0.41 | n=10 | 0.99 | 1.42 | 70% |
| carboxylic_aliphatic | 20 | 0.62 | n=10 | 0.87 | 1.00 | 60% |
| carboxylic_aromatic | 20 | 0.56 | n=10 | 0.28 | 0.35 | 100% |
| guanidine | 8 | 0.81 | n=4 | 2.25 | 3.09 | 50% |
| imidazole | 20 | 0.98 | n=10 | 0.91 | 1.28 | 80% |
| phenol | 20 | 0.38 | n=10 | 1.12 | 1.28 | 50% |
| pyridine | 20 | 0.80 | n=10 | 1.28 | 1.65 | 50% |
| thiol_aliphatic | 8 | 1.07 | n=3 | 1.32 | 1.45 | 33% |
| thiol_aromatic | 5 | 0.67 | n=2 | 0.60 | 0.61 | 100% |

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
