# Cryptic pockets

Cryptic pockets are sites that the structure does not show: they open when the protein moves. Leonardo uses
its own method, physics first with a light calibration, and the grid work is in C++.

1. **Motions.** An anisotropic network model (ANM) is built on the Cα atoms, with a 15 Å cutoff and a sparse
   Hessian. Its 30 lowest non-trivial modes are the cheap, collective motions of the fold. Zheng and
   co-workers showed that sampling along the low modes reshapes cryptic sites into detectable pockets.
2. **Ensemble.** 24 conformations are built as random thermal combinations of those modes (amplitudes
   ∝ 1/√λ), each scaled to a 3 Å Cα RMSD. Every atom of a residue moves with its Cα.
3. **Pockets in each conformation (C++).** The grid uses 1 Å spacing.
   - A grid point is a pocket point when a water-sized probe fits there (≥ 3 Å from every atom centre)
     and at least 70% of 30 rays meet protein within 10 Å.
   - Connected points form pockets, scored by volume × burial.
   - Each residue gets the score of the best pocket it lines (one of its atoms within 4 Å).
4. **Score.** The per-residue features are:
   - the pocket volume lined in the ensemble;
   - how often a pocket is next to the residue;
   - the residue's normal-mode mobility.

   A logistic model with these three weights gives each residue a probability. It was fitted on the
   CryptoBench training folds only (`src/leonardo/cryptic_pocket_model.json`).
5. **Sites.** Called residues are grouped into ranked sites. A residue is called when it is above 0.175, or in
   its protein's top 10%, and never below 0.05. Each site gets the pocket volume next to its residues in the
   structure and averaged over the conformations:
   - **forms with motion:** under 60 Å³ in the structure, or the volume at least doubles;
   - **grows with motion:** +30%;
   - **already open:** anything else.

## Use

- **Window:** **Cryptic Pockets...** in the main window. Choose an object (and chains), then click
  *Find cryptic pockets*. The table lists the sites; click one to show it.
- **Assistant:** `find_cryptic_pockets`. For example, *"Find cryptic pockets in this apo structure and show
  me the best one."*
- **Output:** the object is coloured by probability (blue → red), and the sites are selections
  `<object>_cryptic_<rank>`.
- **Speed:** about 1 s for 300 residues with the C++ extension, and about 15× slower with numpy.

## Benchmark: CryptoBench

The benchmark is CryptoBench (Škrhák et al., *Bioinformatics* 2025). Its test split has 222 apo structures
and 74,046 residues. The protocol is the benchmark's own:
- every observed residue of the apo chain is scored;
- a residue is positive when it lines the cryptic site of any of its holo structures;
- AUC and AUPRC are computed over all residues pooled.

Only the four training folds were used to choose parameters and fit the weights; the test split was scored
once. Reproduce it with `scripts/benchmark_cryptobench.py` (`features`, `fit`, `test`).

| Method | Test set | AUC | AUPRC |
|---|---|---|---|
| Leonardo: pockets in the structure only (no dynamics) | CB full, 222 | 0.739 | 0.145 |
| Leonardo: ensemble pocket score (physics, no fitting) | CB full, 222 | 0.797 | 0.162 |
| **Leonardo: cryptic score (calibrated, 3 weights)** | CB full, 222 | **0.807** | **0.179** |
| PocketMiner (graph neural network) * | CB-PM subset | 0.76 | 0.19 |
| P2Rank on apo * | CB-P2RANK-apo subset | 0.81 | 0.21 |
| pLM-NN (ESM2-3B embeddings, trained on CryptoBench) * | CB full | 0.86 | 0.36 |

\* Numbers published by the CryptoBench authors. Their subsets are not published as lists, so PocketMiner and
P2Rank were scored on fewer structures than Leonardo.

**Sites.** A predicted site counts as found when it shares at least half of its residues, or a third of the
true site's, with the true site (and at least 3 residues). On that test:
- the true site is Leonardo's first site in **34%** of the 222 structures;
- it is among the first three in **53%**.

**What the numbers say:**
- The motions add about 0.06 AUC over the static structure.
- Leonardo matches P2Rank and beats PocketMiner on AUC.
- Its precision (AUPRC) is below the machine-learning methods. The best of them, pLM-NN, learns from a
  3-billion-parameter protein language model.
- Leonardo's advantage is that each prediction has a physical reason: which pocket grows, and by how many Å³,
  along which motions.

**Limitations:**
- Many CryptoBench sites are already partly open in the apo structure; most correct sites are labelled
  *already open*.
- Side-chain rotations are not sampled: every atom moves with its Cα. A site that opens by a side-chain flip
  shows only through the backbone motion around it.
- Probabilities depend on the protein's size; the ranking within a protein matters, not the absolute value.
- These are predictions. Confirm a site with MD (enhanced sampling) or docking.

## C++ extension

The source is `cpp/cryptic/pocket_grid.cpp`; it uses pybind11 and the C++ standard library only. The
installer compiles it on each computer with `python -m leonardo.cryptic_build`, which needs a C++17 compiler
(Xcode Command Line Tools on macOS, g++ on Linux). Without it, Leonardo uses the numpy version of the same
algorithm; the tests check that both give the same pockets.

Sources:
- PocketMiner: Meller et al., *Nat. Commun.* 14, 1177 (2023).
- Normal-mode-guided sampling of cryptic sites: Kumar, Zheng et al.
- Cooperative changes in solvent exposure: Porter et al., *Biophys. J.* 116, 818 (2019).
- CryptoBench: Škrhák et al., *Bioinformatics* 41, btae745 (2025).
