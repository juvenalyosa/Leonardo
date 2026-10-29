# Cryptic pockets

Cryptic pockets are sites that the structure does not show: they open when the protein moves. Leonardo uses
**Crypsis 0.2** (<https://github.com/juvenalyosa/crypsis>), which grew out of Leonardo's original method and is
copied into `src/leonardo/crypsis`. The method is physics first, and the grid work is in C++.

1. **Motions.** An anisotropic network model (ANM) is built on the Cα atoms, with a 15 Å cutoff and a sparse
   Hessian. Its 30 lowest non-trivial modes are the cheap, collective motions of the fold. A Gaussian network
   model gives the slow and fast fluctuations and the hinges, where parts of the protein move against each other.
2. **Ensembles.** Each conformation is a random thermal combination of the modes (amplitudes ∝ 1/√λ). There are
   three ensembles of 24 conformations:
   - scaled to a Cα RMSD of 3 Å;
   - scaled to 5 Å;
   - 3 Å with the side chains cut back to Cβ, because many cryptic pockets are hidden by a few side chains.
3. **Pockets in each conformation (C++).** The grid uses 1 Å spacing.
   - A grid point is a pocket point when a water-sized probe fits there (≥ 3 Å from every atom centre) and at
     least 70% of 30 rays meet protein within 10 Å.
   - Connected points form pockets, scored by volume × burial.
   - Each residue gets the score of the best pocket it lines (one of its atoms within 4 Å).
4. **Descriptors.** Each residue gets 49 descriptors:
   - pockets in the structure and across each ensemble;
   - pockets with side chains cut back;
   - solvent exposure and depth;
   - hinges and fluctuations;
   - backbone geometry;
   - B-factor and amino-acid propensity.
5. **Models.** Gradient-boosted trees were fitted on the CryptoBench training folds only:

   | Model | Inputs | CryptoBench test AUC / AUPRC | Memory |
   |---|---|---|---|
   | `hybrid` | + ESM-2 650M embeddings | 0.889 / 0.388 | ~5.4 GB |
   | `hybrid`, ESM-2 150M | + ESM-2 150M embeddings | 0.884 / 0.368 | ~1.4 GB |
   | `physics` | the 49 descriptors | 0.864 / 0.312 | < 0.3 GB |
   | `v1` | Leonardo's original 3-weight logistic model | 0.807 / 0.179 | < 0.3 GB |

   The installer adds fair-esm (the ESM-2 code) and downloads the weights of both language models (~3.2 GB;
   `--no-esm-weights` skips this, and the first hybrid run then downloads them). The default, `auto`, takes the
   most accurate model the computer can run:
   - the 650M hybrid from 16 GB of memory;
   - the 150M hybrid from 8 GB;
   - the physics model otherwise.

6. **Sites.** The top 10% of residues are grouped into ranked sites. Each site gets the pocket volume next to its
   residues in the structure and averaged over the conformations:
   - **forms with motion:** under 60 Å³ in the structure, or the volume at least doubles;
   - **grows with motion:** +30%;
   - **already open:** anything else.

   Each site also gets a docking box.

## Use

- **Window:** **Cryptic Pockets...** in the main window. Choose an object, the chains and the model (preselected
  for the computer), then click
  *Find cryptic pockets*. The table lists the sites; click one to show it, and *Use as docking box* centres the
  docking box on it.
- **Assistant:** `find_cryptic_pockets`, with `model="auto"` by default, or `physics`, `hybrid` (and
  `esm="150M"`) or `v1`. For example, *"Find cryptic pockets in this apo structure and show me the best one."*
- **Output:** the object is coloured by probability (blue → red), and the sites are selections
  `<object>_cryptic_<rank>`.
- **Speed** (with the C++ extension, on a 2018 laptop CPU):
  - physics model: 6–30 s for 130–970 residues;
  - hybrid model: about 10–20 s more;
  - with numpy instead of the C++ extension, about 15× slower.

## Benchmark: CryptoBench

The benchmark is CryptoBench (Škrhák et al., *Bioinformatics* 2025). Its test split has 222 apo structures and
74,046 residues. The protocol is the benchmark's own:
- every observed residue of the apo chain is scored;
- a residue is positive when it lines the cryptic site of any of its holo structures;
- AUC and AUPRC are computed over all residues pooled, with 95% confidence intervals from resampling proteins.

Only the four training folds were used to choose and fit the models; the test split was scored once. The Crypsis
repository rebuilds every model and number from the public data (`scripts/cryptobench.py`, `BENCHMARK.md`).

| Method | AUC [95% CI] | AUPRC [95% CI] | True site ranked 1st / in top 3 |
|---|---|---|---|
| **Crypsis 0.2 hybrid (ESM-2 650M)** | **0.889** [0.871–0.905] | **0.388** [0.344–0.433] | **51% / 64%** |
| Crypsis 0.2 physics | 0.864 [0.843–0.882] | 0.312 [0.274–0.354] | 41% / 55% |
| Leonardo v1 (3-weight logistic) | 0.807 | 0.179 | 34% / 53% (earlier site definition) |
| P2Rank 2.5.1 (run by us) | 0.788 [0.769–0.805] | 0.191 [0.163–0.226] | |
| PocketMiner (run by us) | 0.762 [0.738–0.783] | 0.173 [0.149–0.202] | |
| Seq2Pocket (fine-tuned ESM-2 3B; published) | 0.894 | 0.475 | |

**What the numbers say:**
- The physics model alone clearly beats P2Rank and PocketMiner.
- A fine-tuned 3-billion-parameter language model (Seq2Pocket) is still more precise. Its AUPRC is 0.475.
- Crypsis's advantage is that each prediction has a physical reason: which pocket grows, and by how many Å³,
  along which motions.

**Limitations:**
- Sites that open only through partial unfolding, large side-chain rotations or a rearrangement absent from the
  apo structure are missed. Examples are the PTP1B helix α7, the HIV-1 RT NNRTI pocket and the p38α lipid pocket.
- The ensembles come from an elastic network, not molecular dynamics.
- Probabilities rank residues within a protein. Compare sites within one structure.
- The sign of each normal mode is arbitrary and the LAPACK build decides it, so another computer or environment
  can sample a different but equivalent ensemble. In Leonardo's environment, for example, 3 of lysozyme's 30 modes
  come out flipped compared with the environment that trained the models. Probabilities can then differ by up to
  ~0.1, while the sites stay the same.
- These are predictions. Confirm a site with MD (enhanced sampling) or docking.

## C++ extension

The source is `cpp/cryptic/pocket_grid.cpp`; it uses pybind11 and the C++ standard library only. The installer
compiles it on each computer with `python -m leonardo.cryptic_build` (into `src/leonardo/_cryptic*.so`, where
Leonardo's copy of Crypsis looks for it). This needs a C++17 compiler: Xcode Command Line Tools on macOS, g++ on
Linux. Without it, Leonardo uses the numpy version of the same algorithm; the tests check that both give the same
pockets.

Sources:
- Crypsis: <https://github.com/juvenalyosa/crypsis>.
- CryptoBench: Škrhák et al., *Bioinformatics* 41, btae745 (2025).
- PocketMiner: Meller et al., *Nat. Commun.* 14, 1177 (2023).
- Seq2Pocket: Škrhák et al., *bioRxiv* (2026), doi:10.64898/2026.01.28.702257.
- Normal-mode-guided sampling of cryptic sites: Kumar, Zheng et al.
- Cooperative changes in solvent exposure: Porter et al., *Biophys. J.* 116, 818 (2019).
