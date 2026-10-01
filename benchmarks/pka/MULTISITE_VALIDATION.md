# Two-site molecules: macroscopic pKa and charge state

Written by `benchmarks/pka/run_multisite_validation.py`. Experimental values: IUPAC Digitized
pKa Dataset v2.4 (Perrin 1965/1972, Serjeant & Dempsey 1979), CC BY-NC 4.0 - "Reproduced by
permission of International Union of Pure and Applied Chemistry"; downloaded at run time,
aggregate statistics only.

412 molecules with two ionizable sites of QM classes, none used for calibration (the
calibration is fitted on single-site molecules). Predicted and experimental macroscopic pKa
values are compared in ascending order. The experimental charge at pH 7.4 is +(number of basic sites) minus the number of experimental pKa values below 7.4.
Molecules with an experimental pKa within 0.5 of that pH (mixed charge states) are left out
of the charge column.

The site-coupling term (`pka_runtime.SITE_COUPLING`, one strength and distance law per pair
type) was fitted on half of these molecules (split by InChIKey hash); **the held-out half is
the validation**, the full set is shown for reference.

| Method | Molecules | Category | n | macro pKa MAE | RMSE | within 1 unit | charge at pH 7.4 right | time per molecule |
|---|---|---|---|---|---|---|---|---|
| empirical | held-out | diacid | 82 | 1.07 | 1.62 | 63% | 65/75 (87%) | 0.0 s |
| empirical | held-out | dibase | 56 | 0.94 | 1.37 | 67% | 42/45 (93%) | 0.0 s |
| empirical | held-out | acid+base | 79 | 1.17 | 1.48 | 56% | 69/74 (93%) | 0.0 s |
| empirical | held-out | **all** | 217 | 1.07 | 1.51 | 61% | 176/194 (91%) | 0.0 s |
| empirical | all | diacid | 150 | 1.00 | 1.45 | 62% | 116/129 (90%) | 0.0 s |
| empirical | all | dibase | 108 | 0.87 | 1.30 | 71% | 74/80 (92%) | 0.0 s |
| empirical | all | acid+base | 154 | 1.17 | 1.50 | 57% | 135/143 (94%) | 0.0 s |
| empirical | all | **all** | 412 | 1.03 | 1.43 | 63% | 325/352 (92%) | 0.0 s |
| pm7 | held-out | diacid | 82 | 1.12 | 1.50 | 53% | 66/75 (88%) | 2.0 s |
| pm7 | held-out | dibase | 56 | 1.01 | 1.80 | 72% | 37/45 (82%) | 2.0 s |
| pm7 | held-out | acid+base | 79 | 0.86 | 1.19 | 72% | 73/74 (99%) | 2.0 s |
| pm7 | held-out | **all** | 217 | 1.00 | 1.48 | 65% | 176/194 (91%) | 2.0 s |
| pm7 | all | diacid | 150 | 1.09 | 1.42 | 54% | 117/129 (91%) | 2.0 s |
| pm7 | all | dibase | 108 | 0.99 | 1.76 | 72% | 68/80 (85%) | 2.0 s |
| pm7 | all | acid+base | 154 | 0.75 | 1.05 | 75% | 140/143 (98%) | 2.0 s |
| pm7 | all | **all** | 412 | 0.94 | 1.40 | 67% | 325/352 (92%) | 2.0 s |

Categories: diacid (two acidic sites), dibase (two basic sites), acid+base (one of each;
zwitterions such as amino acids when the acid is the stronger one).

## Before the coupling term (same held-out molecules)

Added by hand from runs of the previous runtime (commit 08ad0d5); keep it when regenerating
the table above.

Previous runtime: empirical sites independent (no interaction between charged sites); PM7
with the energies of every microstate (its dianions over-couple: a diacid's second pKa
came out 3-4 units too high, e.g. glutaric acid 2.8 / 8.1 vs exp 4.3 / 5.4); charge from the
single most populated microstate (piperazine came out +2 at pH 7.4 because its +1
population is split over two equivalent microstates).

| Method | Category | n | macro pKa MAE | within 1 unit | charge at pH 7.4 right |
|---|---|---|---|---|---|
| empirical | diacid | 82 | 1.15 | 59% | 66/75 (88%) |
| empirical | dibase | 56 | 1.89 | 39% | 23/45 (51%) |
| empirical | acid+base | 79 | 1.27 | 51% | 69/74 (93%) |
| empirical | **all** | 217 | 1.38 | 51% | 158/194 (81%) |
| pm7 | diacid | 82 | 1.24 | 54% | 59/75 (79%) |
| pm7 | dibase | 56 | 1.05 | 71% | 36/45 (80%) |
| pm7 | acid+base | 79 | 0.93 | 61% | 73/74 (99%) |
| pm7 | **all** | 217 | 1.08 | 61% | 168/194 (87%) |

Parameters: `benchmarks/pka/fit_site_coupling.py --method empirical|pm7` (fit half only).
Rerunning the PM7 fit reproduces the anion-anion and cation-anion parameters; for two cationic
sites the optimum is flat (7.75/d shipped vs 13.5/d^1.5 on a rerun: equal at d = 3, held-out
MAE 0.980 vs 0.985, same charge states), because PM7 energies differ slightly between runs.
Not covered: coupling through conjugated/aromatic paths (e.g. two ring nitrogens of a
pyrimidine) is much stronger than the distance law and is under-estimated; groups the module
does not model (aromatic NH, amide NH, enols) were excluded from the set.
