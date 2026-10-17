# Enhanced sampling and OpenMM-ML through Leonardo's assistant tool

Runs every method of `run_enhanced_sampling` (`action: run`) on real systems, as the assistant does:

1. Alanine dipeptide (amber14 + GBn2, 300 K), φ torsion: umbrella sampling (24 windows, MBAR), well-tempered
   metadynamics (1 ns), GaMD and aMD (1 ns, dual boost, cumulant reweighting). Each must find the C7eq/αR basin
   at φ ≈ -1.25 rad.
2. Trypsin-benzamidine (PDB 3PTB): a short umbrella sampling of the protein-ligand distance (benzamidine
   parameterized with GAFF2, calcium site restrained). This checks the set-up on a complex, not a PMF.
3. Alanine dipeptide with ANI-2x (OpenMM-ML, separate leonardo-ml environment), 5 ps: temperature and stability.

```
python -m leonardo.replay verify examples/enhanced_sampling
```

About 20 minutes on a laptop GPU. The runs are stochastic and short: the checks are basins and stability. The
longer cross-check of the four profiles is `scripts/enhanced_sampling_validation.py` (docs/ENHANCED_SAMPLING.md).
