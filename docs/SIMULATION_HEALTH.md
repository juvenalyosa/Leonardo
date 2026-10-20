# Simulation health

A simulation that finishes is not always a simulation that means something. Leonardo checks the system
before it simulates it and watches the run while it goes, so a result that is not physical is never reported
as a success.

Every finding has a level:

| Level | Meaning | What happens |
|---|---|---|
| **red** | the simulation would be wrong | the assistant refuses to run it (the user can override); the GUI asks first; the run stops |
| **yellow** | suspicious, or real only in some cases | the result carries the warning and a lower confidence |
| **info** | worth knowing | reported |

Each result ends with a **confidence**: *high* (nothing found), *medium* (yellow findings) or *low* (red
findings, or the run was stopped), with its reasons.

## Before simulating

Checks on the structure (the input of MD, minimization, QM/MM):

| Check | Level | Why |
|---|---|---|
| Chain bonded across a missing segment (C–N > 4 Å) | red | OpenMM joins the residues on both sides with one stretched peptide bond, which pulls the ends together |
| Residues with missing heavy atoms | red | the force field has no template: the residue is dropped (a hole) or the build fails |
| Atoms on top of each other (< 1 Å) | red | duplicated atoms: both alternate locations, a copied chain or ligand |
| Chain breaks not bonded | yellow | free termini where the protein is continuous |
| Steric clashes (< 2 Å) | yellow | minimization usually relieves them |
| Cys pairs at disulfide distance without a bond | yellow | two thiols pressed together |
| D-amino acids under L names | yellow | usually a modelling error |
| Non-proline cis peptides | yellow | rare in real proteins (~0.03 %) |

Checks on the built OpenMM system:

| Check | Level | Why |
|---|---|---|
| Time step beyond what the constraints carry (2.5 fs; 4.5 fs with heavier hydrogens) | red | the fastest vibrations are not integrated; energy leaks in |
| Non-integer net charge | red | broken charges in a residue or ligand |
| Box shorter than twice the cutoff | red | an atom sees its own image |
| Charged periodic box | yellow | PME's neutralizing background distorts energies |
| Solute closer to its image than one cutoff | yellow | it interacts with its own copy |
| Very high salt (> 1 M) | yellow | |

Time steps between 2.5 and 4.5 fs get hydrogen mass repartitioning (3 amu) automatically.

## During the run

`MDMonitor` reads the state between chunks of steps (about every ps, using the same reads that already drain
the GPU queue):

- **Stops the run** on NaN energies or coordinates, on forces above 10⁵ kJ/mol/nm after minimization, and on a
  temperature far above the thermostat. It names the atoms that were moving fastest and saves the frame where
  it stopped. The trajectory up to there is kept.
- **Flags** a Cα RMSD above 5 Å from the minimized structure, a ligand or cofactor more than 6 Å from where it
  started (after fitting the protein), a density outside 0.90–1.12 g/mL in production, the solute approaching its
  periodic image, a mean temperature off the set point, and energy drift in NVE.

All thresholds are in `simulation_health_runtime.THRESHOLDS`.

## From the assistant

- `check_simulation_system` runs the structure checks on an object without changing it.
- `run_md` and `minimize_structure` run them first and refuse on red. `allow_unsafe=true` overrides this, but
  only with the user's explicit OK.
- MD results carry `health`: the confidence, the findings with their fixes, and statistics (production
  temperature, maximum Cα RMSD and ligand drift, density, the largest force after minimization).

Most red findings are fixed by **Prepare Protein** ([PROTEIN_PREP.md](PROTEIN_PREP.md)).
