# Replay pack: film 02, where two bonds exchange inside an enzyme (DhlA, QM/MM)

Re-run this calculation without the AI assistant and check that you get the same answer:

```bash
python -m leonardo.replay verify examples/films/02_dhla_reaction_qmmm
```

It takes about 70 seconds on a laptop. It needs no API key.

## What is in the pack

| File | Content |
|---|---|
| `2DHC.pdb` | Haloalkane dehalogenase DhlA with 1,2-dichloroethane bound (RCSB PDB 2DHC) |
| `pipeline.json` | Load the structure, then `run_qmmm` with `operation="reaction_path"` |
| `expected.json` | Reference values with tolerances |
| `prompt.md` | The plain-language request behind the pipeline |

**The reaction:** the first step of DhlA. Asp124 OD2 attacks C1 of dichloroethane and CL1 leaves (SN2).

**The `reaction_path` workflow:**

1. The reactant and product minima are defined by the bonds that form (OD2–C1) and break (C1–CL1).
2. A climbing-image NEB path connects them, a QST2-type path.
3. The transition state is refined with the dimer method, and its frequencies must show exactly one imaginary
   mode.
4. The IRC runs from the transition state in both directions. It takes steepest-descent steps in mass-weighted
   coordinates and halves the step when the energy rises. It must connect that reactant to that product.

**Model:** MOPAC PM7 for the dichloroethane and the whole Asp124 residue: 20 atoms, charge −1. Leonardo puts
the full residue of every reacting atom in the QM region. This is embedded in the fixed Amber14 protein, crystal
waters included.

**Saved geometries:** the run writes `irc_path.npz` (the IRC geometries, completed at both ends by the
optimized minima) and `qm_atoms.json`. Film 02 is rendered from these files.

## Reference result

| | Value |
|---|---|
| Barrier (from the IRC reactant minimum) | 33.8 kJ/mol = **8.08 kcal/mol** |
| Transition state | exactly one imaginary mode, 403.7i cm⁻¹ |
| Reaction energy (covalent alkyl-enzyme + chloride) | −101.1 kJ/mol = −24.2 kcal/mol |
| Along the IRC | C–O 3.07 → 1.46 Å forms while C–Cl 1.87 → 3.69 Å breaks |

These are potential energies with the protein fixed, not free energies. The film's original scripted run gave
8.41 kcal/mol and 425.5i cm⁻¹. An earlier version of this pack used a relaxed transfer scan and gave
8.0 kcal/mol. `run_qmmm` `reaction_scan` with `scan_atom_c` still offers that scan as a faster estimate.

## How reproducible it is (measured, macOS arm64)

Replays give identical values to the last digit. Two fixes were needed to get there:

- **Hydrogens:** Leonardo adds them to the MM environment reproducibly, with a fixed seed and the Reference
  platform.
- **Barrier definition:** the barrier is measured from the reactant minimum.
