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

| Part | Result over 3 runs |
|---|---|
| Monte Carlo ensemble | Identical: acceptance 0.314, side-chain shifts 0.00, 0.44, 0.47 and 0.59 Å |
| Docking into the members | Within about 0.01 Å and 0.004 kcal/mol (best member 0.363, 0.374 and 0.372 Å), and 4/4 successes every time |

The docking step is not yet bit-identical in ensemble mode, unlike the single-receptor pack
`01_molecular_docking`. The cause has not been found yet. The differences are far inside the tolerances
(±0.15 Å, ±0.1 kcal/mol) and do not change the result.

The numbers shown in film 08 come from an earlier panel run made with the same tool, but without the
reproducible ensemble options (`artifacts/mc_redock_panel`, 0.48–0.61 Å). They differ slightly from this pack's
reference, for that reason.
