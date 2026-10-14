# Replay pack: film 06, a new covalent lock (protein G B1 disulfide design)

Re-run this calculation without the AI assistant and check that you get the same answer:

```bash
python -m leonardo.replay verify examples/films/06_protein_stability_design
```

It takes about 4 minutes on a laptop. It needs no API key.

## What is in the pack

| File | Content |
|---|---|
| `1PGA.pdb` | Protein G B1 domain (RCSB PDB 1PGA) |
| `pipeline.json` | Load the structure, then `design_disulfides`, refining the 12 best candidates (seed 20260915) |
| `expected.json` | Reference values with tolerances |
| `prompt.md` | The plain-language request behind the pipeline |

## How `design_disulfides` works

1. **Candidates.** It looks for residue pairs whose backbone fits a disulfide: Cα–Cα 4.0–7.5 Å, Cβ–Cβ 3.2–5.2 Å,
   not Gly or Pro, and at least 3 residues apart. Candidates are ranked by how close they are to natural
   disulfides.
2. **Build.** Both residues become Cys and the S–S bond is added. The model is refined with Amber14 while the
   backbone is held in place. This step is reproducible.
3. **Filters,** from natural disulfide geometry: S–S 1.95–2.15 Å, C–S–S angles 90–120°, |χ3| 65–115°, and Cα RMS
   shift below 0.7 Å. Passing designs are ranked by ||χ3| − 90°|.
4. **Rigidity.** A Gaussian network model of the Cα atoms is computed with and without the new cross-link.

## Reference result

18 candidate pairs were found and the 12 best were refined. Nine pass the filters; three fail on χ3 (−62°,
−144° and −135°).

| Best design | S–S | χ3 | Cα shift | Relative motion of the pair |
|---|---|---|---|---|
| **K10C/E56C** | 2.05 Å | 89° | 0.19 Å | −51% (network model) |

The tool reaches the same design as the film's scripted run (K10C/E56C, −51%) on its own. This is a structural
design and rigidity model, not a measured thermal stability.

## How reproducible it is

The refinement is seeded and runs on one CPU thread with deterministic forces, so the same input gives the same
designs.
