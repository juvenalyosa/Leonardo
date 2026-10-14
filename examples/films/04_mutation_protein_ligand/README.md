# Replay pack: film 04, trypsin D189A, changing the recognition pocket

Re-run this calculation without the AI assistant and check that you get the same answer:

```bash
python -m leonardo.replay verify examples/films/04_mutation_protein_ligand
```

It takes about 40 seconds on a laptop. It needs no API key, but MODELLER needs its (free academic) license.

## What is in the pack

| File | Content |
|---|---|
| `3PTB.pdb` | Trypsin with its crystal benzamidine (RCSB PDB 3PTB) |
| `pipeline.json` | Load the structure. Build the D189A mutant with `model_mutant` (MODELLER, 3 models, seed 1). Then `run_redocking` of benzamidine into wild type and mutant: three repeats from seed 20260915, exhaustiveness 16, pH 7.4 |
| `expected.json` | Reference values with tolerances |
| `prompt.md` | The plain-language request behind the pipeline |

## Reference result

| | Wild type | D189A |
|---|---|---|
| Median top-1 score | −6.11 kcal/mol | −5.37 kcal/mol |
| Median top-1 RMSD to the crystal pose | 0.39 Å | 0.84 Å |

Removing the Asp189 carboxylate, which forms the salt bridge with the amidinium, weakens the score by
0.75 kcal/mol and loosens the pose. The mutant model stays within 0.09 Å (Cα) of the wild type. Vina scores
are scoring-function values, not binding free energies.

## How reproducible it is (measured, macOS arm64)

Three runs, MODELLER included, gave identical values to the last digit.

## Compared with the film

The film was rendered from a scripted run of the same engines, with a different preparation (an Amber-built
mutant, smina called directly): −5.92 → −5.38 kcal/mol. Leonardo's tools give −6.11 → −5.37 kcal/mol. The
mutant agrees closely; the wild type differs by the receptor and ligand preparation, as in pack 01.
