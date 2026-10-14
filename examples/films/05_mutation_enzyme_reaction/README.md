# Replay pack: film 05, DhlA wild type vs W125A, the reaction barrier

Re-run this calculation without the AI assistant and check that you get the same answer:

```bash
python -m leonardo.replay verify examples/films/05_mutation_enzyme_reaction
```

It takes about 5 minutes on a laptop. It needs no API key.

## What is in the pack

| File | Content |
|---|---|
| `2DHC.pdb` | Haloalkane dehalogenase DhlA with 1,2-dichloroethane bound (RCSB PDB 2DHC) |
| `pipeline.json` | Two `run_qmmm` `reaction_path` runs, one on the wild type and one on W125A, with `model_mutant` (`method="local"`) in between |
| `expected.json` | Reference values with tolerances |
| `prompt.md` | The plain-language request behind the pipeline |

**The reaction:** the first step of DhlA. Asp124 OD2 attacks C1 of dichloroethane and CL1 leaves (SN2).

**The `reaction_path` workflow,** run for each enzyme:

1. The reactant and product minima are defined by the bonds that form and break.
2. A climbing-image NEB path connects them, a QST2-type path.
3. The transition state is refined, and its frequencies must show exactly one imaginary mode.
4. The IRC from the transition state must connect that reactant to that product.

**Model:** MOPAC PM7 for the dichloroethane and the whole Asp124 residue (20 atoms, charge −1), embedded in
the fixed Amber14 protein.

**The mutant is built locally:** only residue 125 is rebuilt, and every other atom keeps its exact position,
substrate and crystal waters included. The two QM/MM environments therefore differ only at the mutation. An
earlier version built the mutant with MODELLER, which also moves charged surface side chains by up to 6 Å. In an
electrostatic embedding without solvent, that alone shifted the reactant energy by 124 kJ/mol. The path then
went wrong: chloride left without the C–O bond forming.

## Reference result

| | Wild type | W125A |
|---|---|---|
| Barrier (from the IRC reactant minimum) | 33.8 kJ/mol = **8.08 kcal/mol** | 43.2 kJ/mol = **10.32 kcal/mol** |
| Transition state | one imaginary mode, 404i cm⁻¹ | one imaginary mode, 425i cm⁻¹ |
| IRC connects reactant and product | yes | yes |
| Reaction energy | −101.1 kJ/mol | −81.7 kJ/mol |

Removing Trp125, part of the halide-binding site, **raises the barrier by 2.24 kcal/mol** and makes the reaction
less exothermic. Losing the stabilization of the leaving chloride gives exactly this effect. The film's scripted
run gave 8.41 → 11.13 kcal/mol. These are potential energies with the protein fixed, not free energies.

## How reproducible it is (measured, macOS arm64)

A replay gave identical values to the last digit.
