"""
Receptor ensembles by Monte Carlo sampling of the binding-site side chains (replaces the MODELLER
multi-model ensembles for ensemble redocking).

Algorithm: Monte Carlo with minimization (Li & Scheraga, PNAS 1987) in side-chain torsion space.
  1. The receptor (best model) is repaired and protonated (PDBFixer, pH 7), and described with
     Amber14 + GB/OBC2 implicit solvent (non-periodic cutoff 1.2 nm).
  2. Only side-chain atoms of the binding-site residues move; every other atom is frozen (mass 0),
     so the fold and the backbone are kept.
  3. Each move picks a site residue and one of its chi angles and rotates it by a Gaussian step
     (sigma = step_deg) or, with probability rotamer_jump_prob, by +/-120 degrees (rotamer jump);
     the structure is then locally minimized.
  4. Metropolis criterion on the minimized potential energy at temperature T:
     accept if dE <= 0 or rand < exp(-dE / kT).
  5. After a burn-in, the current state is saved every `steps_per_structure` moves until
     `n_structures` distinct conformations (site heavy-atom RMSD > 0.25 A to all saved ones) exist.

Run in a child process: ``python mc_ensemble_runtime.py args.json`` writes PDB files and
``mc_ensemble.json`` (energies, acceptance, RMSD to the start) into ``out_dir``.
"""

import json
import math
import os
import sys
import time

import numpy as np

KB_KJ = 0.0083144626          # kJ/mol/K
BACKBONE = {"N", "CA", "C", "O", "H", "HA", "HA2", "HA3", "OXT", "H1", "H2", "H3"}
_RES_ALIAS = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "CYM": "CYS"}
CHI = {
    "ARG": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
    "ASN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "GLU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")],
    "HIS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "LYS": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "MET": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
}


def _rotate(pos, idx, a, b, angle_rad):
    """Rotate positions[idx] about the axis a->b by angle (Rodrigues)."""
    axis = b - a
    axis = axis / np.linalg.norm(axis)
    v = pos[idx] - b
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rot = v * c + np.cross(axis, v) * s + np.outer(v @ axis, axis) * (1 - c)
    out = pos.copy()
    out[idx] = rot + b
    return out


def _downstream(residue_atoms, bonds, b, c):
    """Atoms of the residue on c's side of the b-c bond."""
    nbr = {i: set() for i in residue_atoms}
    for i, j in bonds:
        if i in nbr and j in nbr:
            nbr[i].add(j)
            nbr[j].add(i)
    seen, stack = {c}, [c]
    while stack:
        k = stack.pop()
        for n in nbr[k]:
            if n != b and n not in seen:
                seen.add(n)
                stack.append(n)
    seen.discard(c)
    return sorted(seen)


def site_residues_near(pdb_path, ligand_xyz_ang, radius_ang):
    """(chain, resnum) of protein residues with any heavy atom within radius of the ligand atoms."""
    lig = np.asarray(ligand_xyz_ang, dtype=float).reshape(-1, 3)
    out = set()
    for line in open(pdb_path):
        if line.startswith("ATOM") and line[76:78].strip() != "H":
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if np.min(np.linalg.norm(lig - xyz, axis=1)) <= radius_ang:
                out.add((line[21].strip(), line[22:27].strip()))
    return sorted(out)


def sample_ensemble(pdb_in, out_dir, n_structures=10, site_residues=None, temperature_k=300.0,
                    steps_per_structure=20, step_deg=40.0, rotamer_jump_prob=0.3, seed=1, platform="",
                    burn_in=None, max_steps=None, progress=print):
    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer

    t0 = time.time()
    rng = np.random.default_rng(int(seed))
    os.makedirs(out_dir, exist_ok=True)
    fixer = PDBFixer(filename=pdb_in)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}                       # no loop building here: MODELLER/repair does that
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    top, pos0 = fixer.topology, fixer.positions
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    system = ff.createSystem(top, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.2 * unit.nanometer,
                             constraints=None)
    # PyMOL gives a blank chain as '' and OpenMM as ' ': compare stripped
    wanted = {(str(c).strip(), str(r).strip()) for c, r in (site_residues or [])}
    bonds = [(b.atom1.index, b.atom2.index) for b in top.bonds()]
    moves, movable = [], set()
    for res in top.residues():
        chain = str(res.chain.id).strip()
        key = (chain, str(res.id).strip() + str(getattr(res, "insertionCode", "") or "").strip())
        if wanted and key not in wanted and (chain, str(res.id).strip()) not in wanted:
            continue
        name = _RES_ALIAS.get(res.name, res.name)
        atoms = {a.name: a.index for a in res.atoms()}
        if name not in CHI or res.name == "CYX":
            continue
        ids = [a.index for a in res.atoms()]
        for chi in CHI[name]:
            if all(n in atoms for n in chi):
                down = _downstream(ids, bonds, atoms[chi[1]], atoms[chi[2]])
                if down:
                    moves.append({"residue": f"{name}{res.id}{res.chain.id}", "b": atoms[chi[1]], "c": atoms[chi[2]],
                                  "atoms": down})
        movable.update(a.index for a in res.atoms() if a.name not in BACKBONE)
    if not moves:
        raise RuntimeError("no rotatable side chains in the selected binding site")
    for i in range(system.getNumParticles()):
        if i not in movable:
            system.setParticleMass(i, 0.0)
    names = [p.getName() for p in (mm.Platform.getPlatform(k) for k in range(mm.Platform.getNumPlatforms()))]
    plat = mm.Platform.getPlatformByName(platform if platform in names else "CPU")
    ctx = mm.Context(system, mm.VerletIntegrator(0.001 * unit.picoseconds), plat)
    ctx.setPositions(pos0)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 500)

    def state():
        st = ctx.getState(getEnergy=True, getPositions=True)
        return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))

    e_cur, x_cur = state()
    x_start, e_start = x_cur.copy(), e_cur
    heavy = [a.index for a in top.atoms() if a.index in movable and a.element is not None and a.element.symbol != "H"]
    kt = KB_KJ * float(temperature_k)
    burn_in = int(burn_in if burn_in is not None else steps_per_structure)
    max_steps = int(max_steps or (burn_in + n_structures * steps_per_structure * 6))
    saved, accepted, step = [], 0, 0
    frames_x = []

    def site_rmsd(a, b):
        return float(np.sqrt(((a[heavy] - b[heavy]) ** 2).sum(axis=1).mean()) * 10.0)

    while len(saved) < int(n_structures) and step < max_steps:
        step += 1
        mv = moves[int(rng.integers(len(moves)))]
        if rng.random() < float(rotamer_jump_prob):
            ang = math.radians(float(rng.choice([-120.0, 120.0])))
        else:
            ang = math.radians(float(rng.normal(0.0, float(step_deg))))
        trial = _rotate(x_cur, mv["atoms"], x_cur[mv["b"]], x_cur[mv["c"]], ang)
        ctx.setPositions(trial * unit.nanometer)
        mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 100)
        e_new, x_new = state()
        de = e_new - e_cur
        if de <= 0 or rng.random() < math.exp(-de / kt):
            e_cur, x_cur = e_new, x_new
            accepted += 1
        if step > burn_in and (step - burn_in) % int(steps_per_structure) == 0:
            if all(site_rmsd(x_cur, f) > 0.25 for f in frames_x):
                k = len(saved) + 1
                path = os.path.join(out_dir, f"mc_{k:03d}.pdb")
                with open(path, "w") as fh:
                    app.PDBFile.writeFile(top, x_cur * unit.nanometer, fh, keepIds=True)
                frames_x.append(x_cur.copy())
                saved.append({"path": path, "step": step, "energy_kj_mol": round(e_cur, 2),
                              "relative_energy_kj_mol": round(e_cur - e_start, 2),
                              "site_rmsd_to_start_A": round(site_rmsd(x_cur, x_start), 2)})
                progress(f"PROGRESS {len(saved)}/{n_structures} structures (step {step})")
    start_path = os.path.join(out_dir, "mc_000_start.pdb")
    with open(start_path, "w") as fh:
        app.PDBFile.writeFile(top, x_start * unit.nanometer, fh, keepIds=True)
    result = {"ok": bool(saved), "structures": saved, "start_structure": start_path,
              "start_energy_kj_mol": round(e_start, 2), "steps": step, "acceptance": round(accepted / max(1, step), 3),
              "moves": len(moves), "site_residues": sorted({m["residue"] for m in moves}),
              "movable_atoms": len(movable), "temperature_k": float(temperature_k), "platform": plat.getName(),
              "seconds": round(time.time() - t0, 1),
              "method": "Monte Carlo with minimization of binding-site side-chain torsions "
                        "(Amber14/GB-OBC2, Metropolis at T)"}
    if not saved:
        result["error"] = "no structure was accepted: raise the temperature or the number of steps"
    return result


def main(args_path):
    args = json.load(open(args_path))
    res = sample_ensemble(**args, progress=lambda text: print(text, flush=True))
    with open(os.path.join(args["out_dir"], "mc_ensemble.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("DONE", json.dumps({"ok": res["ok"], "n": len(res["structures"])}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
