# GPU validation

How to check that every OpenMM workflow in Leonardo runs on the GPU, and what was checked for 0.2.0a13.

```bash
bash scripts/gpu_validation.sh 2>&1 | tee gpu_validation.log     # after a full install
```

The script prints the GPU and driver, runs OpenMM's cross-platform force check, re-detects the hardware Leonardo
will use (benchmark and context precision), and replays `examples/films/03_molecular_dynamics` and
`examples/gpu_validation`. The second pack fails unless every step ran on a GPU platform (CUDA, HIP, OpenCL or
Metal).

## Results

| Workflow | Apple M4, OpenCL (2026-10-17) | NVIDIA, CUDA |
|---|---|---|
| OpenMM force agreement (Reference/CPU/GPU) | pass (median 7e-6) | not yet run |
| Platform picked by the benchmark | OpenCL, 4.9x the CPU | |
| Context precision | single (Apple GPUs reject mixed) | mixed expected |
| MD, implicit water (film 03 pack, 8 statistical checks) | pass, OpenCL | |
| MD, explicit water NPT, ligand + Ca site (0.02 ns) | pass, OpenCL; 300.7 K, 1.03 g/mL, 93 ns/day (27k atoms) | |
| Minimization | pass, OpenCL | |
| ABFE-lite, trypsin-benzamidine, 3 replicas x 1 ns | pass, OpenCL; 2 of 3 replicas completed exit/return cycles, ΔG ≈ -54 kJ/mol (qualitative) | |
| ABFE-lite defaults, trypsin-benzamidine, 3 replicas x 3 ns | runs, OpenCL (21 min, process tree ≤ 1.7 GB, wired memory flat); ΔG not accurate: selected -62, bias-reweighted standard-state -33.7, population -2.2 to -3.0, exp-avg +0.7 kJ/mol vs ≈ -27 measured; six more runs (film pack 11): selected -53.8 to -79.2, reweighted -19.6 to -25.7, population -2.2 to -4.1 | not yet run |
| Protein on Au(111), build + minimization | pass, OpenCL single | |
| Umbrella sampling (alanine φ, MBAR) | pass, OpenCL | |
| Well-tempered metadynamics | pass, OpenCL | |
| GaMD / aMD (dual boost, reweighting) | pass, OpenCL | |
| OpenMM-ML, ANI-2x (leonardo-ml env) | pass, CPU (openmm-torch has no OpenCL kernels) | CUDA expected |
| Enhanced-sampling cross-check (alanine φ, 1 ns each) | umbrella/metaD/GaMD/aMD agree within 0.8-1.7 kJ/mol (docs/ENHANCED_SAMPLING.md) | |
| Membrane build + equilibration | runs on the GPU (same code path as surfaces); full run not repeated here | |
| MC side-chain ensemble | CPU by design (reproducible); GPU optional with CPU fallback | |
| Lead-opt refinement (OpenMM + PyMBAR) | fixed for Apple GPUs (always failed before); not rerun end to end | |

ABFE-lite is a steered guided-exit protocol: its ΔG is an apparent value, and one replica can stall with the
ligand against the protein surface (no estimate from that replica). Use 3 or more replicas.

**ABFE-lite is experimental.** It runs safely on a 16 GB Mac GPU, but its estimators disagree with each other
and with experiment by tens of kJ/mol (row above); every report, agent card and the GUI say so. It needs a
multi-target calibration on a dedicated GPU before its ΔG can be quoted.  On Apple OpenCL the OpenMM Context
runs in a child process recycled at 1.2 GB (Apple's driver leaks per kernel launch and frees only at process
exit); `LEONARDO_OPENMM_CONTEXT_PROCESS=0/1` turns this off/on.

## Fixed while validating

- Apple GPUs have no 64-bit floats; their OpenCL rejects mixed precision. Contexts asking for it failed (lead-opt
  refinement always failed on M-series Macs) or fell back silently. A cached probe now picks mixed where the
  device supports it and single otherwise.
- `LEONARDO_OPENMM_PLATFORM` was honoured only in part (on macOS a GPU request ended on the CPU); an unavailable
  request is now logged.
- ABFE-lite ran only its first replica: the next ones started from the first replica's trajectory and failed the
  chain-gap check.
- ABFE-lite read the complex at the viewer's frame: after looking at an MD trajectory (frame 500) a single-state
  complex had no atoms there.
- Membranes and surfaces never tried HIP or Metal and set no precision; they now report platform and precision.
  ABFE-lite reports its platform. The MC ensemble falls back to the CPU if the GPU does not start.

## Not in this version

Alchemical free energies (FEP, thermodynamic integration).
