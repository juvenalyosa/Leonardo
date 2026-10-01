# Psi4 cheap-level pilot (negative result)

Question: do cheap Psi4 single points (PCM water, CPCM/GePol UFF area 1.0) on the lowest
PM7/COSMO conformer beat the shipped PM7/COSMO model? Plan target for Psi4: MAE <= 0.8.

Molecules: 60 in-domain molecules from the **calibration** split of the PM7 benchmark
(5 per class, 3-4 for guanidine/thiol_aromatic/amine1/amine3; the test split was not used),
same isodesmic references, 10 ETKDG/MMFF conformers optimized with PM7/COSMO. Calibration:
robust per-class slope with shrinkage (pseudo-count 5), leave-one-out. Experimental values:
IUPAC Digitized pKa Dataset v2.4 (CC BY-NC 4.0, not redistributed; aggregate statistics only).

| Model | LOO MAE | uncalibrated MAE | wall time (10 cores, 5 x 2-thread Psi4) |
|---|---|---|---|
| PM7/COSMO (shipped) | **0.92** | 1.18 | - |
| Psi4 HF/3-21G | 1.32 | 3.23 | 16 min |
| Psi4 B3LYP/6-31G* | 1.25 | 2.70 | 47 min |

Acids: PM7 0.70, HF/3-21G 0.89, B3LYP/6-31G* 0.85. Bases: PM7 1.07, HF/3-21G 1.61,
B3LYP/6-31G* 1.52. Psi4 was worse than PM7 in 9-10 of 13 classes; its wins (amine2,
carboxylic_aromatic, thiol_aromatic) are 0.1-0.2 units at n <= 5.

Conclusion: at these levels Psi4 is slower and less accurate than PM7. It is kept as an
**experimental** method: the full benchmark (`PSI4_BENCHMARK.md`, HF/3-21G, 327 molecules,
held-out test set) recommends Psi4 only for aliphatic carboxylic acids and aromatic thiols,
and the runtime uses PM7 for every other class, so the shipped Psi4 model is on par with PM7
(test MAE 0.90 both) at a much higher cost.
