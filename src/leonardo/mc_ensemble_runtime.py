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
  5. After a burn-in, the current state is saved once at least `steps_per_structure` moves have passed
     since the last saved structure and it differs from all saved ones (site heavy-atom RMSD > 0.25 A),
     until `n_structures` conformations exist or `max_steps` is reached (``produced`` < ``requested``
     is reported with a warning).
  The preparation (missing atoms, hydrogens) is seeded. With ``reproducible=True`` the sampling runs on
  one CPU thread with deterministic forces, so the same seed gives exactly the same ensemble; otherwise
  (GPU, several threads) parallel sums differ in the last digits between runs and the Metropolis chain
  diverges: ensembles are statistically equivalent, not identical.

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


_WATER = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SOL"}
OBSTACLE_SIGMA_NM = 0.30         # cofactor/ion atoms held fixed: excluded volume for the moving side chains
OBSTACLE_EPS_KJ = 0.40


def _hetero_records(pdb_path, exclude=(), drop_from_lines=()):
    """(HETATM lines except ``drop_from_lines`` residues, [(element, xyz_nm)] of non-water heavy hetero atoms
    not in ``exclude``)."""
    skip = {(str(c).strip(), str(r).strip(), str(n).strip().upper()) for c, r, n in (exclude or [])}
    drop = {(str(c).strip(), str(r).strip(), str(n).strip().upper()) for c, r, n in (drop_from_lines or [])}
    lines, atoms = [], []
    with open(pdb_path) as fh:
        for ln in fh:
            if not ln.startswith("HETATM"):
                continue
            resn, chain, resi = ln[17:20].strip().upper(), ln[21].strip(), ln[22:27].strip()
            if (chain, resi, resn) not in drop:
                lines.append(ln if ln.endswith("\n") else ln + "\n")
            if resn in _WATER or (chain, resi, resn) in skip:
                continue
            el = (ln[76:78].strip() or ln[12:16].strip()[:1]).capitalize()
            if el in ("H", "D", ""):
                continue
            atoms.append((el, [float(ln[30:38]) / 10, float(ln[38:46]) / 10, float(ln[46:54]) / 10]))
    return lines, atoms


def _obstacle_xml(elements):
    from openmm.app import Element

    out = ["<ForceField>", " <AtomTypes>"]
    for el in elements:
        mass = Element.getBySymbol(el).mass._value
        out.append(f'  <Type name="mcobs-{el}" class="mcobs{el}" element="{el}" mass="{mass:.3f}"/>')
    out += [" </AtomTypes>", " <Residues>"]
    for el in elements:
        out.append(f'  <Residue name="OBS_{el}"><Atom name="{el.upper()}" type="mcobs-{el}"/></Residue>')
    out += [" </Residues>", ' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">']
    for el in elements:
        out.append(f'  <Atom type="mcobs-{el}" charge="0" sigma="{OBSTACLE_SIGMA_NM}" epsilon="{OBSTACLE_EPS_KJ}"/>')
    out += [" </NonbondedForce>", "</ForceField>"]
    return "\n".join(out) + "\n"


def sample_ensemble(pdb_in, out_dir, n_structures=10, site_residues=None, temperature_k=300.0,
                    steps_per_structure=20, step_deg=40.0, rotamer_jump_prob=0.3, seed=1, platform="",
                    burn_in=None, max_steps=None, progress=print, exclude_hetero=(), keep_hetero_records=True,
                    reproducible=False):
    """
    Monte Carlo receptor ensemble (see the module docstring). Hetero groups of ``pdb_in`` (cofactors,
    metal ions; not waters and not ``exclude_hetero`` = [(chain, resi, resn)], e.g. the ligand being
    redocked) stay fixed during sampling as uncharged excluded-volume atoms, so side chains do not
    collapse into their space, and their original records are written back into every member
    (``keep_hetero_records``) so the receptor preparation keeps or strips them as for the rigid run.
    """
    import io

    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer

    t0 = time.time()
    rng = np.random.default_rng(int(seed))
    os.makedirs(out_dir, exist_ok=True)
    fixer = PDBFixer(filename=pdb_in)
    # modified residues (MSE, CSO, phosphoresidues...) become their standard parent instead of being removed
    # with the heterogens, which would cut the chain
    fixer.findNonstandardResidues()
    # only residues peptide-bonded into a chain are replaced; a free modified amino acid in the crystal (e.g.
    # an MSE ligand) has no template as a lone residue and stays a hetero group
    fpos = np.array(fixer.positions.value_in_unit(unit.nanometer))

    def _atom(res, name):
        return next((a.index for a in res.atoms() if a.name == name), None)

    def _linked(res):
        residues = list(res.chain.residues())
        k = residues.index(res)
        for other, mine, theirs in ((residues[k - 1] if k > 0 else None, "N", "C"),
                                    (residues[k + 1] if k + 1 < len(residues) else None, "C", "N")):
            if other is None:
                continue
            i, j = _atom(res, mine), _atom(other, theirs)
            if i is not None and j is not None and np.linalg.norm(fpos[i] - fpos[j]) < 0.2:
                return True
        return False

    fixer.nonstandardResidues = [(r, std) for r, std in fixer.nonstandardResidues if _linked(r)]
    replaced = [f"{r.name}{r.id}->{std}" for r, std in fixer.nonstandardResidues]
    # they are HETATM records in the PDB but part of the chain: not obstacles, not written back as hetero
    in_chain = [(str(r.chain.id).strip(), str(r.id).strip(), str(r.name).strip().upper())
                for r, _std in fixer.nonstandardResidues]
    het_lines, obstacles = _hetero_records(pdb_in, list(exclude_hetero or []) + in_chain, drop_from_lines=in_chain)
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}                       # no loop building here: MODELLER/repair does that
    fixer.findMissingAtoms()
    # reproducible preparation: OpenMM places missing atoms and hydrogens with Python's ``random`` and
    # relaxes them on the default (parallel, non-deterministic) platform; seed it and use Reference
    import random

    random.seed(int(seed))
    fixer.addMissingAtoms(seed=int(seed))
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    hmod = app.Modeller(fixer.topology, fixer.positions)
    hmod.addHydrogens(ff, pH=7.0, platform=mm.Platform.getPlatformByName("Reference"))
    prot_top, prot_pos = hmod.topology, hmod.positions
    n_prot = prot_top.getNumAtoms()
    if obstacles:
        # an obstacle on top of a protein atom (a modified residue missed above, alternate locations) would
        # add a huge constant repulsion that drowns every energy difference: such atoms are not obstacles
        px = np.array(prot_pos.value_in_unit(unit.nanometer))
        ox = np.array([x for _el, x in obstacles])
        dmin = np.min(np.linalg.norm(ox[:, None, :] - px[None, :, :], axis=2), axis=1)
        obstacles = [o for o, d in zip(obstacles, dmin) if d > 0.1]
    if obstacles:
        elements = sorted({el for el, _x in obstacles})
        ff.loadFile(io.StringIO(_obstacle_xml(elements)))
        otop = app.Topology()
        ochain = otop.addChain("Z")
        for el, _x in obstacles:
            ores = otop.addResidue(f"OBS_{el}", ochain)
            otop.addAtom(el.upper(), app.Element.getBySymbol(el), ores)
        mod = app.Modeller(prot_top, prot_pos)
        mod.add(otop, unit.Quantity([mm.Vec3(*x) for _el, x in obstacles], unit.nanometer))
        top, pos0 = mod.topology, mod.positions
    else:
        top, pos0 = prot_top, prot_pos
    system = ff.createSystem(top, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.2 * unit.nanometer,
                             constraints=None)
    # PyMOL gives a blank chain as '' and OpenMM as ' ': compare stripped
    wanted = {(str(c).strip(), str(r).strip()) for c, r in (site_residues or [])}
    bonds = [(b.atom1.index, b.atom2.index) for b in top.bonds()]
    moves, movable = [], set()
    for res in prot_top.residues():
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
    if reproducible:
        platform = "CPU"
    plat = mm.Platform.getPlatformByName(platform if platform in names else "CPU")
    # parallel reductions (GPU, CPU threads) change energies in the last digits from run to run, and the
    # Metropolis decisions with them; one deterministic CPU thread makes the chain exactly reproducible
    props = {"DeterministicForces": "true"} if "DeterministicForces" in plat.getPropertyNames() else {}
    if reproducible and "Threads" in plat.getPropertyNames():
        props["Threads"] = "1"
    ctx = mm.Context(system, mm.VerletIntegrator(0.001 * unit.picoseconds), plat, props)
    ctx.setPositions(pos0)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 500)

    def state():
        st = ctx.getState(getEnergy=True, getPositions=True)
        return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))

    e_cur, x_cur = state()
    x_start, e_start = x_cur.copy(), e_cur
    heavy = [a.index for a in prot_top.atoms() if a.index in movable and a.element is not None
             and a.element.symbol != "H"]

    def write_member(path, x_nm):
        """Protein of the state (with its CONECT records: disulfide bonds) + the input's original HETATM
        records renumbered after the protein (obstacles themselves are not written)."""
        buf = io.StringIO()
        app.PDBFile.writeFile(prot_top, x_nm[:n_prot] * unit.nanometer, buf, keepIds=True)
        lines = buf.getvalue().splitlines(True)
        atoms_part = [ln for ln in lines if not ln.startswith(("END", "CONECT"))]
        conect = [ln for ln in lines if ln.startswith("CONECT")]
        serial = max([int(ln[6:11]) for ln in atoms_part if ln.startswith(("ATOM", "HETATM"))] or [0])
        with open(path, "w") as fh:
            fh.writelines(atoms_part)
            if keep_hetero_records:
                for ln in het_lines:
                    serial += 1
                    fh.write(ln[:6] + f"{serial % 100000:5d}" + ln[11:])
            fh.writelines(conect)
            fh.write("END\n")
    kt = KB_KJ * float(temperature_k)
    burn_in = int(burn_in if burn_in is not None else steps_per_structure)
    max_steps = int(max_steps or (burn_in + int(n_structures) * int(steps_per_structure) * 10))
    saved, accepted, step = [], 0, 0
    last_save = burn_in
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
        # save as soon as enough moves have passed since the last structure (decorrelation) AND the current
        # state is new; waiting for exact multiples of steps_per_structure wasted the steps in between
        if step > burn_in and step - last_save >= int(steps_per_structure):
            if all(site_rmsd(x_cur, f) > 0.25 for f in frames_x):
                last_save = step
                k = len(saved) + 1
                path = os.path.join(out_dir, f"mc_{k:03d}.pdb")
                write_member(path, x_cur)
                frames_x.append(x_cur.copy())
                saved.append({"path": path, "step": step, "energy_kj_mol": round(e_cur, 2),
                              "relative_energy_kj_mol": round(e_cur - e_start, 2),
                              "site_rmsd_to_start_A": round(site_rmsd(x_cur, x_start), 2)})
                progress(f"PROGRESS {len(saved)}/{n_structures} structures (step {step})")
    start_path = os.path.join(out_dir, "mc_000_start.pdb")
    write_member(start_path, x_start)
    result = {"ok": bool(saved), "structures": saved, "start_structure": start_path,
              "requested": int(n_structures), "produced": len(saved), "max_steps": max_steps,
              "start_energy_kj_mol": round(e_start, 2), "steps": step, "acceptance": round(accepted / max(1, step), 3),
              "moves": len(moves), "site_residues": sorted({m["residue"] for m in moves}),
              "fixed_hetero_atoms": len(obstacles), "replaced_nonstandard": replaced,
              "movable_atoms": len(movable), "temperature_k": float(temperature_k), "platform": plat.getName(),
              "reproducible": bool(reproducible),
              "seconds": round(time.time() - t0, 1),
              "method": "Monte Carlo with minimization of binding-site side-chain torsions "
                        "(Amber14/GB-OBC2, Metropolis at T)"}
    if not saved:
        result["error"] = "no structure was accepted: raise the temperature or the number of steps"
    elif len(saved) < int(n_structures):
        result["warning"] = (f"only {len(saved)} of {int(n_structures)} distinct conformations after {step} Monte "
                             f"Carlo steps: the site is rigid at {float(temperature_k):.0f} K; raise the temperature "
                             "or the site radius for a broader ensemble")
    return result


def main(args_path):
    args = json.load(open(args_path))
    res = sample_ensemble(**args, progress=lambda text: print(text, flush=True))
    with open(os.path.join(args["out_dir"], "mc_ensemble.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("DONE", json.dumps({"ok": res["ok"], "n": len(res["structures"])}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
