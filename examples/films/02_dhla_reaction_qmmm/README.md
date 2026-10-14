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
| `pipeline.json` | Load the structure, then `run_qmmm` (`reaction_scan`) |
| `expected.json` | Reference values with tolerances |
| `prompt.md` | The plain-language request behind the pipeline |

The `run_qmmm` step uses:

- **Reaction:** the first step of DhlA. Asp124 OD2 attacks C1 of dichloroethane and CL1 leaves (SN2).
- **QM region:** dichloroethane plus the Asp124 side chain (20 atoms with hydrogens and link atom), charge −1,
  MOPAC PM7.
- **MM environment:** Amber14, fixed, with electrostatic embedding. The crystal waters are kept.
- **Scan:** a relaxed scan of the transfer coordinate ξ = d(CL1–C1) − d(C1–OD2), from −1.6 to +1.6 Å in 13
  points.
- **Transition state:** the highest point is refined, and the frequencies are checked.

## Reference result

| | Value |
|---|---|
| Barrier (refined transition state, from the reactant minimum) | 33.5 kJ/mol = 8.0 kcal/mol |
| Transition-state imaginary modes | exactly one, 404.6i cm⁻¹ |
| Reaction energy (covalent alkyl-enzyme + chloride) | −69.4 kJ/mol |
| Along the path | C–O 3.44 → 1.49 Å forms while C–Cl 1.85 → 3.10 Å breaks |

These are potential energies with the protein fixed, not free energies. The film's scripted run, using a
different path method, gave 8.41 kcal/mol and 425.5i cm⁻¹.

## How reproducible it is (measured, macOS arm64)

Three runs gave identical values to the last digit.

Two fixes were needed to get there:

- **Hydrogens:** Leonardo now adds them to the MM environment reproducibly, with a fixed seed and the Reference
  platform. Before that, the barrier varied by 0.3 kJ/mol between runs and the product energy by 18 kJ/mol.
- **Barrier definition:** the barrier is now measured from the reactant minimum of the scan, not from its first
  point. Here the first point lies 3 kJ/mol above the minimum.
