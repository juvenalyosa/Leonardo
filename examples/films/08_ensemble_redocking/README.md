# Replay pack: film 08, P450cam + camphor ensemble redocking

Re-run this calculation without the AI assistant and check that you get the same answer:

```bash
python -m leonardo.replay verify examples/films/08_ensemble_redocking
```

It takes about 7 minutes on a laptop. Reproducible Monte Carlo sampling runs on one CPU thread, so it is slower
than the default. It needs no API key.

## What is in the pack

| File | Content |
|---|---|
| `2CPP.pdb` | Cytochrome P450cam with its heme and crystal camphor (RCSB PDB 2CPP) |
| `pipeline.json` | Load the structure, then `run_redocking` with the heme kept and a Monte Carlo receptor ensemble. The ensemble is the best model plus 3 side-chain conformations at 300 K, within 6 Å of the ligand, with seed 1 in reproducible mode. Docking uses smina/Vina with exhaustiveness 8 and seed 1001 |
| `expected.json` | Reference values with tolerances; ensemble values are checked without depending on member order |
| `prompt.md` | The plain-language request behind the pipeline |

## Reference result

All four receptor conformations recover the crystal pose. The top-1 RMSD is 0.36–0.56 Å and the best score is
−7.81 kcal/mol. The heme is kept in every member. Vina scores are not binding free energies.

## How reproducible it is (measured, macOS arm64)

Replays are identical to the reference to the last digit, ensemble and docking alike.

Before this pack was made, ensemble redocking ignored the requested docking seed: it always ran smina with a
random seed. As a result, two runs differed by about 0.01 Å. This was fixed: in ensemble mode the seed is now
kept, with one docking run per member.

The numbers shown in film 08 come from an earlier panel run made with the same tool, but without the
reproducible ensemble options (`artifacts/mc_redock_panel`, 0.48–0.61 Å). They differ slightly from this pack's
reference, for that reason.
