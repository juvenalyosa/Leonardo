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

## Reference result and measured spread (8 runs, macOS arm64, OpenCL)

| Value | Mean of 8 runs (reference) | SD | Range | Tolerance |
|---|---|---|---|---|
| Final Cα RMSD | 1.10 Å | 0.16 | 0.79–1.31 Å | ±0.5 Å |
| Maximum Cα RMSD | 1.31 Å | 0.09 | 1.14–1.48 Å | ±0.3 Å; must also stay ≤ 2 Å (fold kept) |
| Mean Cα RMSF | 0.66 Å | 0.03 | 0.60–0.73 Å | ±0.12 Å |
| Ser195 OG – His57 NE2, mean | 3.98 Å | 0.16 | 3.78–4.23 Å | ±0.5 Å |
| Same distance, standard deviation | 0.46 Å | 0.04 | 0.40–0.51 Å | ±0.15 Å |
| Frames with that distance ≤ 3.5 Å | 22% (12 runs) | 11 | 6–44% | ±33 points |

Tolerances are about three standard deviations of the runs. They were widened twice as more runs came in. With
3 runs a fourth fell just outside, and the H-bond fraction, the noisiest number here, needed 12 runs: one gave 44%.

The film's scripted run (12 ps, CPU) gave 4.02 ± 0.37 Å for the same distance, consistent with this range.
100 ps in implicit solvent is a short relaxation, not converged sampling.
