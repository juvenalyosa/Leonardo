# Replay pack: film 01, trypsin + benzamidine redocking

Re-run this calculation without the AI assistant and check that you get the same answer:

```bash
python -m leonardo.replay verify examples/films/01_molecular_docking
```

It takes about 30 seconds on a laptop. It needs no API key. It prints PASS or FAIL for each value and writes
`last_report.json`, which records your machine, Leonardo version and engine versions.

## What is in the pack

| File | Content |
|---|---|
| `3PTB.pdb` | Trypsin with its crystal benzamidine (RCSB PDB 3PTB) |
| `pipeline.json` | The exact Leonardo tool calls: load the structure, then `run_redocking` with exhaustiveness 16, three repeats, seed 20260915 (repeat *i* uses seed + *i*) and pH 7.4 |
| `expected.json` | The reference values and their tolerances |
| `prompt.md` | The plain-language request behind the pipeline |

## Reference result

All three repeats recover the crystal pose: the top-1 pose is 0.387 Å from the crystal and scores −6.11 kcal/mol.
The Vina score is a scoring-function value, not a binding free energy.

## How reproducible it is (measured, macOS arm64, smina 2020.12.10)

| Test | Runs | Result |
|---|---|---|
| Same pipeline, same seed | 5 | Every value identical to the reference, to the last digit |
| Same pipeline, 8 other seeds | 24 repeats | Top-1 RMSD 0.385–0.387 Å, score −6.117 to −6.111 kcal/mol, 3/3 successes every time |

The tolerances in `expected.json` (±0.15 Å, ±0.1 kcal/mol) are much wider than the spread measured here. They
leave room for other machines and engine builds, which have not been measured yet. The success count must
match exactly.

## Note on the film's numbers

The film itself was rendered from a direct smina run
(`scripts/prepare_real_cinematic_data.py`: the ligand built from SMILES, an Amber-prepared receptor, and the
same seed). That run gives −5.96 kcal/mol and 0.40 Å, and it replays exactly too
(`artifacts/reproducibility/film01_docking/`). This pack instead uses Leonardo's own redocking tool, which
prepares the ligand and receptor in its own way, so its numbers differ slightly: −6.11 kcal/mol and 0.39 Å.
Both runs find the same binding mode.
