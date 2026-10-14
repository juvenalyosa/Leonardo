# Replay pack: film 03, apo trypsin molecular dynamics

Re-run this calculation without the AI assistant and check that you get a statistically equivalent answer:

```bash
python -m leonardo.replay verify examples/films/03_molecular_dynamics
```

It takes about 2 minutes on a laptop with a GPU (OpenCL/Metal), and longer on CPU only. It needs no API key.

## What is in the pack

| File | Content |
|---|---|
| `3PTB.pdb` | Trypsin (RCSB PDB 3PTB); benzamidine, waters and calcium are left out of the simulation |
| `pipeline.json` | Load the structure. Then `run_md`: Amber14, implicit solvent, 300 K, 2 fs, 10 ps heating + 20 ps equilibration + 100 ps production, 500 frames. Then `trajectory_rmsd_rmsf` and `trajectory_distance` (Ser195 OG – His57 NE2) |
| `expected.json` | Reference values with tolerances, plus the spread measured over three runs |
| `prompt.md` | The plain-language request behind the pipeline |

## Why this pack checks averages, not frames

A molecular dynamics trajectory is chaotic: tiny numerical differences, such as GPU summation order or a
different machine, grow until two runs follow different paths. Two runs never give the same frames. They do
give the same statistics, so this pack checks averages within tolerances set from repeated runs.

## Reference result and measured spread (three runs, macOS arm64, OpenCL)

| Value | Reference | Range over 3 runs | Tolerance |
|---|---|---|---|
| Final Cα RMSD | 1.29 Å | 1.05–1.31 Å | ±0.5 Å |
| Maximum Cα RMSD | 1.32 Å | 1.32–1.48 Å | ±0.5 Å; must also stay ≤ 2 Å (fold kept) |
| Mean Cα RMSF | 0.66 Å | 0.66–0.67 Å | ±0.15 Å |
| Ser195 OG – His57 NE2, mean | 3.82 Å | 3.78–4.05 Å | ±0.4 Å |
| Same distance, standard deviation | 0.44 Å | 0.44–0.49 Å | ±0.2 Å |
| Frames with that distance ≤ 3.5 Å | 22% | 14–30% | ±25 points |

The film's scripted run (12 ps, CPU) gave 4.02 ± 0.37 Å for the same distance, consistent with this range.
100 ps in implicit solvent is a short relaxation, not converged sampling.
