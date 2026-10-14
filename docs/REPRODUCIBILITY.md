# Reproducing Leonardo's calculations

Leonardo's AI assistant turns a request into a sequence of tool calls. A language model is not deterministic:
the same prompt can lead to slightly different tool calls on different runs, for example a different
exhaustiveness or a different site radius. Reproducibility therefore does not come from the prompt. It comes
from the **recorded tool calls**.

Every tool call the assistant makes is recorded with its exact arguments, seeds included. A finished protocol can
be saved as a **pipeline**, and a pipeline can be replayed without the AI.

## Replay packs

A replay pack is a folder with:

- the input files;
- `pipeline.json`, the exact tool calls;
- `expected.json`, the reference values with tolerances;
- `prompt.md`, the plain-language request.

To check a pack on your own machine:

```bash
python -m leonardo.replay verify examples/films/01_molecular_docking
```

The command re-runs the pipeline headless, without the AI and without an API key. It compares every value with
the reference, prints PASS or FAIL, and writes a report with your machine, Leonardo version and engine versions.
`python -m leonardo.replay record <pack>` writes new reference values after a deliberate change.

| Pack | Calculation | Status |
|---|---|---|
| [`examples/films/01_molecular_docking`](../examples/films/01_molecular_docking) | Trypsin + benzamidine redocking (smina/Vina) | Verified: same seed gives identical values; 8 other seeds stay within 0.002 Å and 0.006 kcal/mol |
| [`examples/films/03_molecular_dynamics`](../examples/films/03_molecular_dynamics) | Apo trypsin MD (OpenMM, implicit solvent) | Verified statistically: 8 runs, references = means, tolerances ≈ 3 SD (Ser195–His57 3.98 ± 0.16 Å) |
| [`examples/films/04_mutation_protein_ligand`](../examples/films/04_mutation_protein_ligand) | Trypsin D189A: MODELLER mutant + redocking into wild type and mutant | Verified: identical values across runs |
| [`examples/films/08_ensemble_redocking`](../examples/films/08_ensemble_redocking) | P450cam + camphor, Monte Carlo receptor ensemble (heme kept) | Verified: identical values across runs |
| [`examples/films/02_dhla_reaction_qmmm`](../examples/films/02_dhla_reaction_qmmm) | DhlA SN2 step, QM/MM transfer scan + transition state (MOPAC PM7 / Amber14) | Verified: identical values across runs (barrier 8.0 kcal/mol, one imaginary mode) |
| Film 05 | DhlA W125A, QM/MM | In progress: the one-coordinate scan lets chloride dissociate without forming the C–O bond in the mutant; the reaction path method is being revised |
| Films 06–07 | Disulfide design | No Leonardo tool yet |

## What "the same result" means for each engine

| Engine | What to expect | Check |
|---|---|---|
| smina/Vina docking | Same seed: identical on the same build (measured). Other seeds: tiny spread for a well-defined pocket (measured for trypsin) | Values within tolerance, same success verdict |
| Monte Carlo receptor ensembles | Identical with `reproducible=True` (CPU, one thread). Measured and covered by a test | Exact |
| Molecular dynamics (OpenMM) | Trajectories diverge between machines and on GPUs; averages do not | Statistics within a confidence interval |
| Quantum chemistry (Psi4, MOPAC) | Deterministic to about 1e-6 | Energies within 0.01 kcal/mol |

The tolerances in each pack are set from repeated runs. So far they have been measured only on macOS (Apple
Silicon); results from other machines are welcome.
