"""
Simulation health: is the system about to be simulated physically right, and does the run stay right?

Two layers, both Qt-free:

  * ``structure_findings`` / ``system_findings`` -- checks before any step is taken.  The first one
    needs only a topology and positions (the input structure), the second the OpenMM System built from
    it (charge, time step, box).  Each finding is a dict
        {"level": "red" | "yellow" | "info", "code", "title", "detail", "where": [...], "fix"}
    "red" means the simulation would be wrong (a stretched bond across a gap, atoms on top of each other,
    residues that the force field would drop, a time step the constraints cannot carry), not merely
    unusual: D-residues, cis peptides, a charged box, metals are "yellow".
  * ``MDMonitor`` -- reads the Context between chunks of steps (the reads that ``openmm_step_synced``
    already does to drain the GPU queue, so no extra round trips) and stops a run that is blowing up
    (NaN, huge forces, temperature far above the thermostat) naming the residue responsible, and notes
    what makes a finished run suspicious (ligand leaving its pocket, unfolding, density off, energy drift
    in NVE, the solute touching its periodic image).

``summarize`` turns the findings into a confidence label (high / medium / low) with its reasons.
All thresholds are in ``THRESHOLDS`` so they can be read and overridden.
"""

import math

import numpy as np

THRESHOLDS = {
    "overlap_A": 1.0,               # two heavy atoms closer than this: duplicated atoms / altlocs (red)
    "clash_A": 2.0,                 # non-bonded heavy atoms closer than this (yellow; minimization relieves it)
    "gap_bond_A": 2.5,              # a peptide C-N longer than this is a chain break
    "gap_bond_red_A": 4.0,          # bonded across a break this long: a stretched spring in the simulation
    "cis_omega_deg": 30.0,          # |omega| below this is a cis peptide
    "max_timestep_hbonds_fs": 2.5,  # with H-bond constraints and normal hydrogen masses
    "max_timestep_flexible_fs": 1.0,
    "max_timestep_hmr_fs": 4.5,
    "ion_molar_high": 1.0,
    "force_warn_kj_mol_nm": 1.0e4,  # max |F| on an atom after minimization
    "force_fatal_kj_mol_nm": 1.0e5,
    "temp_hot_frac": 0.15,          # mean T this far above the set point (yellow)
    "temp_fatal_frac": 0.6,         # instantaneous T this far above the set point: blowing up (stop)
    "ligand_drift_A": 6.0,          # ligand centre moved this far from its starting place (yellow)
    "ca_rmsd_A": 5.0,               # Calpha RMSD from the minimized structure (yellow)
    "density_low": 0.90,            # g/mL, explicit solvent NPT
    "density_high": 1.12,
    "nve_drift_kj_mol_ns_dof": 0.05,
}

WATER = {"HOH", "WAT", "TIP3", "TIP", "SOL", "T3P", "T4P", "T4E", "SPC", "H2O", "DOD"}
IONS = {"NA", "CL", "K", "MG", "CA", "ZN", "LI", "RB", "CS", "BR", "IOD", "F", "NA+", "CL-", "K+"}
METALS = {"ZN", "MG", "CA", "FE", "FE2", "CU", "CU1", "MN", "CO", "NI", "CD", "HG", "NA", "K", "LI", "MO", "W"}

# heavy atoms of the standard residues (without OXT); protonation variants share them
_HEAVY = {
    "ALA": "N CA C O CB", "ARG": "N CA C O CB CG CD NE CZ NH1 NH2", "ASN": "N CA C O CB CG OD1 ND2",
    "ASP": "N CA C O CB CG OD1 OD2", "CYS": "N CA C O CB SG", "GLN": "N CA C O CB CG CD OE1 NE2",
    "GLU": "N CA C O CB CG CD OE1 OE2", "GLY": "N CA C O", "HIS": "N CA C O CB CG ND1 CD2 CE1 NE2",
    "ILE": "N CA C O CB CG1 CG2 CD1", "LEU": "N CA C O CB CG CD1 CD2", "LYS": "N CA C O CB CG CD CE NZ",
    "MET": "N CA C O CB CG SD CE", "PHE": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ",
    "PRO": "N CA C O CB CG CD", "SER": "N CA C O CB OG", "THR": "N CA C O CB OG1 CG2",
    "TRP": "N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2", "TYR": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH",
    "VAL": "N CA C O CB CG1 CG2",
}
_ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "ASH": "ASP",
            "GLH": "GLU", "LYN": "LYS", "CYX": "CYS", "CYM": "CYS"}
HEAVY_ATOMS = {k: set(v.split()) for k, v in _HEAVY.items()}
AMINO = set(HEAVY_ATOMS) | set(_ALIASES)


def _parent(resname):
    return _ALIASES.get(resname, resname)


def finding(level, code, title, detail, where=(), fix=""):
    return {"level": level, "code": code, "title": title, "detail": detail, "where": list(where)[:12], "fix": fix}


def residue_label(res):
    chain = getattr(res.chain, "id", "") or ""
    return f"{chain}:{res.name}{res.id}" if chain.strip() else f"{res.name}{res.id}"


def _positions_A(positions):
    """OpenMM Quantity / list of Vec3 (nm) or an (N, 3) array in nm -> (N, 3) array in Angstrom."""
    if hasattr(positions, "value_in_unit"):
        from openmm import unit

        positions = positions.value_in_unit(unit.nanometer)
    return np.asarray([[float(c) for c in p] for p in positions], dtype=float).reshape(-1, 3) * 10.0


def _element(atom):
    el = getattr(atom, "element", None)
    sym = getattr(el, "symbol", "") if el is not None else ""
    if sym:
        return sym.upper()
    return "".join(c for c in atom.name if c.isalpha())[:1].upper()


def _neighbor_pairs(xyz, cutoff):
    try:
        from scipy.spatial import cKDTree

        return cKDTree(xyz).query_pairs(cutoff, output_type="ndarray")
    except Exception:
        out = []
        for i in range(len(xyz)):
            d = np.linalg.norm(xyz[i + 1:] - xyz[i], axis=1)
            out.extend((i, i + 1 + int(j)) for j in np.nonzero(d < cutoff)[0])
        return np.asarray(out, dtype=int).reshape(-1, 2)


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1) or 1.0)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return math.degrees(math.atan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


# ---------------------------------------------------------------------------------------------------- structure

def structure_findings(topology, positions, thresholds=None):
    """Checks on the structure alone (input PDB or the Modeller before the System is built)."""
    th = dict(THRESHOLDS, **(thresholds or {}))
    xyz = _positions_A(positions)
    atoms = list(topology.atoms())
    out = []
    out += _check_residues(topology, xyz, th)
    out += _check_contacts(topology, atoms, xyz, th)
    return out


def _check_residues(topology, xyz, th):
    out = []
    missing, d_res, cis, gaps_red, gaps = [], [], [], [], []
    bonded = {frozenset((a.index, b.index)) for a, b in topology.bonds()}
    for chain in topology.chains():
        # amino acids, standard or modified (SEP, MSE...): anything with a peptide backbone
        residues = [r for r in chain.residues() if r.name in AMINO or {"N", "CA", "C"} <= {a.name for a in r.atoms()}]
        for i, res in enumerate(residues):
            names = {a.name: a.index for a in res.atoms()}
            parent = _parent(res.name)
            lacking = HEAVY_ATOMS[parent] - set(names) if parent in HEAVY_ATOMS else set()
            if lacking:
                # a residue at a chain end may lack only its terminal atoms in some files
                missing.append(f"{residue_label(res)} ({', '.join(sorted(lacking)[:4])})")
            if parent in HEAVY_ATOMS and parent != "GLY" and {"N", "CA", "C", "CB"} <= set(names):
                n, ca, c, cb = (xyz[names[k]] for k in ("N", "CA", "C", "CB"))
                # L-amino acids: (N-CA) . ((C-CA) x (CB-CA)) is positive
                if np.dot(n - ca, np.cross(c - ca, cb - ca)) < 0:
                    d_res.append(residue_label(res))
            if i + 1 < len(residues):
                nxt = residues[i + 1]
                nn = {a.name: a.index for a in nxt.atoms()}
                if "C" in names and "N" in nn:
                    d = float(np.linalg.norm(xyz[names["C"]] - xyz[nn["N"]]))
                    if d > th["gap_bond_A"]:
                        label = f"{residue_label(res)}-{residue_label(nxt)} ({d:.1f} A)"
                        if frozenset((names["C"], nn["N"])) in bonded and d > th["gap_bond_red_A"]:
                            gaps_red.append(label)
                        else:
                            gaps.append(label)
                    elif ("CA" in names and "CA" in nn and _parent(nxt.name) != "PRO"):
                        w = _dihedral(xyz[names["CA"]], xyz[names["C"]], xyz[nn["N"]], xyz[nn["CA"]])
                        if abs(w) < th["cis_omega_deg"]:
                            cis.append(f"{residue_label(res)}-{residue_label(nxt)} (omega {w:.0f})")
    if missing:
        out.append(finding("red", "missing_atoms", "Residues with missing heavy atoms",
                           f"{len(missing)} residue(s) lack side-chain or backbone atoms; the force field has no "
                           "template for them, so they would be removed (a hole in the protein) or the build fails",
                           missing, "prepare_protein completes them (Prepare Protein...)"))
    if gaps_red:
        out.append(finding("red", "bonded_gap", "Chain bonded across a gap",
                           f"{len(gaps_red)} missing segment(s) where the residues on both sides would be joined "
                           "by one stretched peptide bond: the simulation pulls the ends together",
                           gaps_red, "prepare_protein builds the missing loop or ends the chain at the gap"))
    if gaps:
        out.append(finding("yellow", "chain_break", "Chain breaks",
                           f"{len(gaps)} break(s) in the chain (missing residues); the ends are free termini",
                           gaps, "build the missing residues with prepare_protein if they matter"))
    if d_res:
        out.append(finding("yellow", "d_residue", "Inverted chirality (D-amino acids)",
                           f"{len(d_res)} residue(s) named as L-amino acids have D geometry: usually a modelling "
                           "error, unless the peptide really contains D-residues", d_res,
                           "rebuild those side chains (model_mutant to the same residue) or check the source"))
    if cis:
        out.append(finding("yellow", "cis_peptide", "Cis peptide bonds (not before Pro)",
                           f"{len(cis)} non-proline cis peptide(s): rare in real proteins (~0.03%), often a "
                           "model or refinement error", cis, "check against the electron density / source"))
    return out


def _check_contacts(topology, atoms, xyz, th):
    heavy = [a for a in atoms if _element(a) != "H"]
    if len(heavy) < 2:
        return []
    idx = np.array([a.index for a in heavy])
    pairs = _neighbor_pairs(xyz[idx], th["clash_A"])
    if not len(pairs):
        return []
    nbrs = {}
    for a, b in topology.bonds():
        nbrs.setdefault(a.index, set()).add(b.index)
        nbrs.setdefault(b.index, set()).add(a.index)
    overlaps, clashes, ss = [], [], []
    for p, q in pairs:
        i, j = int(idx[p]), int(idx[q])
        d = float(np.linalg.norm(xyz[i] - xyz[j]))
        # bonded or 1-3 pairs are not contacts, except closer than any heavy-atom bond (C#O is 1.13 A): a
        # bond read from CONECT records that PyMOL made by distance between two overlapping atoms
        if (j in nbrs.get(i, ()) or nbrs.get(i, set()) & nbrs.get(j, set())) and d >= 0.9:
            continue
        ai, aj = atoms[i], atoms[j]
        if ai.residue.name.upper() in WATER and aj.residue.name.upper() in WATER:
            continue                                   # crystal waters are rebuilt or dropped
        if {ai.name, aj.name} == {"C", "N"} and ai.residue.chain is aj.residue.chain \
                and abs(ai.residue.index - aj.residue.index) == 1:
            continue                                   # peptide bond to a residue OpenMM does not know (SEP...)
        if _element(ai) in METALS or _element(aj) in METALS:
            continue                                   # metal coordination is short by nature
        if ai.residue is aj.residue and d >= th["overlap_A"]:
            continue
        label = f"{residue_label(ai.residue)} {ai.name} - {residue_label(aj.residue)} {aj.name} ({d:.2f} A)"
        if ai.name == "SG" and aj.name == "SG" and ai.residue.name in AMINO and aj.residue.name in AMINO:
            ss.append(label)
        elif d < th["overlap_A"]:
            overlaps.append(label)
        else:
            clashes.append((d, label))
    out = []
    if overlaps:
        out.append(finding("red", "overlapping_atoms", "Atoms on top of each other",
                           f"{len(overlaps)} pair(s) of heavy atoms closer than {th['overlap_A']} A: duplicated "
                           "atoms (both alternate locations kept, a copied chain or ligand)", overlaps,
                           "keep one alternate location (prepare_protein) or remove the duplicate"))
    if ss:
        out.append(finding("yellow", "unbonded_disulfide", "Cysteine pairs not bonded",
                           f"{len(ss)} Cys pair(s) at disulfide distance without an S-S bond: they would be "
                           "simulated as two thiols pressed together", ss,
                           "prepare_protein (or the PDB's SSBOND records) makes the disulfide"))
    if clashes:
        clashes.sort()
        out.append(finding("yellow", "clashes", "Steric clashes",
                           f"{len(clashes)} non-bonded heavy-atom contact(s) under {th['clash_A']} A; minimization "
                           "should relieve them, otherwise they start the run with huge forces",
                           [c[1] for c in clashes], "check the worst ones; prepare_protein then minimize"))
    return out


# ---------------------------------------------------------------------------------------------------- system

def _net_charge(system):
    import openmm

    for f in system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            from openmm import unit

            return sum(f.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                       for i in range(f.getNumParticles()))
    return None


def degrees_of_freedom(system):
    import openmm

    n = sum(1 for i in range(system.getNumParticles())
            if system.getParticleMass(i).value_in_unit(openmm.unit.dalton) > 0)
    dof = 3 * n - system.getNumConstraints()
    if any(isinstance(f, openmm.CMMotionRemover) for f in system.getForces()):
        dof -= 3
    return max(1, dof)


def system_findings(system, topology, positions, *, timestep_fs=2.0, mode="implicit", cutoff_nm=1.0,
                    thresholds=None):
    """Checks that need the built OpenMM System: net charge, time step against constraints and
    hydrogen masses, box size against the cutoff and the solute, ion concentration."""
    from openmm import unit

    th = dict(THRESHOLDS, **(thresholds or {}))
    out = []
    q = _net_charge(system)
    if q is not None:
        if abs(q - round(q)) > 0.01:
            out.append(finding("red", "fractional_charge", "Non-integer net charge",
                               f"total charge {q:+.3f} e is not an integer: some residue or ligand has broken "
                               "charges (wrong template, partial charges that do not add up)", (),
                               "re-parameterize the ligand/cofactor; check its formal charge"))
        elif abs(q) >= 0.5:
            if system.usesPeriodicBoundaryConditions():
                out.append(finding("yellow", "charged_box", "Charged periodic box",
                                   f"net charge {q:+.0f} e: PME adds a uniform neutralizing background, which "
                                   "distorts energies and free energies", (), "add counter-ions to neutralize"))
            else:
                out.append(finding("info", "net_charge", "Net charge",
                                   f"net charge {q:+.0f} e (non-periodic, no correction needed)"))
    # water hydrogens are rigid (and not repartitioned): only the solute's hydrogens set the limit
    masses_h = [system.getParticleMass(a.index).value_in_unit(unit.dalton) for a in topology.atoms()
                if _element(a) == "H" and a.residue.name.upper() not in WATER]
    hmr = bool(masses_h) and min(masses_h) > 1.5
    constrained = system.getNumConstraints() > 0
    limit = (th["max_timestep_hmr_fs"] if hmr and constrained else th["max_timestep_hbonds_fs"] if constrained
             else th["max_timestep_flexible_fs"])
    if float(timestep_fs) > limit + 1e-6:
        why = ("hydrogens are light and only H-bonds are constrained" if constrained and not hmr
               else "no bonds are constrained" if not constrained else "even with heavier hydrogens")
        out.append(finding("red", "timestep", "Time step too long",
                           f"{timestep_fs:g} fs with {why}: the fastest vibrations are not integrated, energy "
                           f"leaks in and the run heats up or explodes (limit here {limit:g} fs)", (),
                           f"use timestep_fs <= {limit:g}" + ("" if hmr else " (4 fs needs hydrogen mass "
                                                                         "repartitioning)")))
    xyz = _positions_A(positions)
    if system.usesPeriodicBoundaryConditions():
        box = np.array([[v.value_in_unit(unit.angstrom) for v in vec]
                        for vec in system.getDefaultPeriodicBoxVectors()])
        lengths = np.diag(box)
        cutoff_A = float(cutoff_nm) * 10.0
        if lengths.min() / 2.0 < cutoff_A:
            out.append(finding("red", "box_cutoff", "Box smaller than twice the cutoff",
                               f"shortest box edge {lengths.min():.1f} A < 2 x cutoff ({2 * cutoff_A:.1f} A): an "
                               "atom would see its own image", (), "increase the solvent padding"))
        solute = [a.index for a in topology.atoms() if a.residue.name.upper() not in WATER | IONS]
        if solute:
            span = _diameter(xyz[solute])
            if lengths.min() - span < cutoff_A:
                out.append(finding("yellow", "box_small", "Solute may touch its periodic image",
                                   f"the solute spans {span:.1f} A and the box edge is {lengths.min():.1f} A: "
                                   f"less than one cutoff ({cutoff_A:.0f} A) of water between copies when it "
                                   "rotates", (), "raise the padding to >= 10-12 A"))
        n_water = sum(1 for r in topology.residues() if r.name.upper() in WATER)
        n_ion = sum(1 for r in topology.residues() if r.name.upper() in {"NA", "CL", "K", "NA+", "CL-", "K+"})
        if n_water > 50 and n_ion:
            molar = n_ion / 2.0 / (n_water / 55.5)
            if molar > th["ion_molar_high"]:
                out.append(finding("yellow", "ions_high", "Very high salt",
                                   f"~{molar:.2f} M salt ({n_ion} ions in {n_water} waters)", (),
                                   "check ionic_strength"))
    return out


def _diameter(points):
    """Largest distance between two of the points (on the convex hull, so it stays cheap)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import pdist

        if len(pts) > 4:
            pts = pts[ConvexHull(pts).vertices]
        return float(pdist(pts).max())
    except Exception:
        return float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))) * 2.0


# ---------------------------------------------------------------------------------------------------- monitor

class SimulationHealthError(RuntimeError):
    def __init__(self, item):
        super().__init__(item["detail"])
        self.finding = item


class MDMonitor:
    """Watch a running OpenMM Context.  ``check(stage, setpoint_K)`` after each chunk of steps; raises
    SimulationHealthError when the run is no longer physical.  ``findings`` keeps the warnings."""

    def __init__(self, context, system, topology, *, temperature_K=300.0, ensemble="nvt", ligand_atoms=None,
                 cutoff_nm=1.0, timestep_fs=2.0, thresholds=None):
        from openmm import unit

        self.context, self.topology = context, topology
        self.th = dict(THRESHOLDS, **(thresholds or {}))
        self.temperature_K = float(temperature_K)
        self.ensemble = str(ensemble or "nvt").lower()
        self.cutoff_A = float(cutoff_nm) * 10.0
        self.timestep_fs = float(timestep_fs)
        self.dof = degrees_of_freedom(system)
        self.periodic = bool(system.usesPeriodicBoundaryConditions())
        self.atoms = list(topology.atoms())
        self.ca = np.array([a.index for a in self.atoms if a.name == "CA" and a.residue.name in AMINO], dtype=int)
        # each ligand / cofactor on its own: (label, heavy-atom indices)
        self.ligands = ([("ligand", np.array(sorted(ligand_atoms), dtype=int))] if ligand_atoms is not None
                        else self._guess_ligands())
        self.solute = np.array([a.index for a in self.atoms if a.residue.name.upper() not in WATER | IONS], dtype=int)
        self.masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                                for i in range(system.getNumParticles())])
        self.ref = None
        self.findings, self.samples = [], []
        self._seen = set()
        self._energy = []                  # (time ps, total kJ/mol) in NVE production
        self._steps = 0
        self.max_force = None
        self._last_fast = []

    def _guess_ligands(self):
        out = []
        for res in self.topology.residues():
            name = res.name.upper()
            if name in AMINO or name in WATER or name in IONS or name in METALS:
                continue
            atoms = list(res.atoms())
            if {"N", "CA", "C"} <= {a.name for a in atoms} or {"P", "C1'"} <= {a.name for a in atoms}:
                continue                           # modified amino acid / nucleotide: part of a chain
            heavy = [a.index for a in atoms if _element(a) != "H"]
            if len(heavy) >= 6:
                out.append((residue_label(res), np.array(heavy, dtype=int)))
        return out[:20]

    def _warn(self, level, code, title, detail, where=(), fix=""):
        if code in self._seen:
            return
        self._seen.add(code)
        self.findings.append(finding(level, code, title, detail, where, fix))

    def _fatal(self, code, title, detail, where=(), fix=""):
        item = finding("red", code, title, detail, where, fix)
        self.findings.append(item)
        raise SimulationHealthError(item)

    def _residue_of(self, index):
        return residue_label(self.atoms[int(index)].residue) if 0 <= int(index) < len(self.atoms) else str(index)

    def set_reference(self):
        """Take the current (minimized) positions as the reference and look at the forces."""
        from openmm import unit

        st = self.context.getState(getPositions=True, getForces=True, getEnergy=True)
        self.ref = np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        f = np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not np.all(np.isfinite(f)) or not math.isfinite(e):
            self._fatal("nan_start", "Energy is not a number before dynamics",
                        "the minimized structure has NaN/inf energy or forces: overlapping atoms or broken "
                        "parameters", (), "check the structure (overlapping atoms) and the ligand parameters")
        norms = np.linalg.norm(f, axis=1)
        worst = np.argsort(norms)[::-1][:5]
        self.max_force = float(norms[worst[0]]) if len(norms) else 0.0
        where = [f"{self._residue_of(i)} {self.atoms[i].name} ({norms[i]:.0f} kJ/mol/nm)" for i in worst
                 if norms[i] > self.th["force_warn_kj_mol_nm"]]
        if self.max_force > self.th["force_fatal_kj_mol_nm"]:
            self._fatal("huge_force", "Huge forces after minimization",
                        f"max force {self.max_force:.2e} kJ/mol/nm: minimization did not relieve a clash, or a "
                        "ligand/residue has wrong parameters; dynamics would explode", where,
                        "look at these atoms; minimize longer or fix their geometry/parameters")
        if where:
            self._warn("yellow", "large_force", "Large forces after minimization",
                       f"max force {self.max_force:.0f} kJ/mol/nm (well-minimized systems stay below ~1000)", where,
                       "minimize more before dynamics")

    def check(self, stage, setpoint_K=None, steps=0):
        from openmm import unit

        self._steps += int(steps or 0)
        st = self.context.getState(getEnergy=True, getPositions=True, getVelocities=True,
                                   enforcePeriodicBox=False)
        pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        xyz = np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        target = float(setpoint_K if setpoint_K is not None else self.temperature_K)
        if not (math.isfinite(pe) and math.isfinite(ke)) or not np.all(np.isfinite(xyz)):
            self._fatal("nan", "The simulation exploded (NaN)",
                        f"energy or coordinates became NaN during {stage}", self._fastest_atoms(st),
                        "usually a clash, a too-long time step or bad ligand parameters; minimize longer, "
                        "shorten the time step, re-check the ligand")
        self._last_fast = self._fastest_atoms(st)
        temp = 2.0 * ke / (self.dof * 0.0083144626)
        sample = {"stage": stage, "time_ps": round(self._steps * self.timestep_fs / 1000.0, 3),
                  "T_K": round(temp, 1), "setpoint_K": round(target, 1), "PE_kJ_mol": round(pe, 1)}
        if self.ensemble != "nve" and temp > target * (1.0 + self.th["temp_fatal_frac"]) + 20.0:
            self._fatal("runaway_temperature", "Temperature running away",
                        f"T = {temp:.0f} K against a thermostat at {target:.0f} K during {stage}: energy is "
                        "pouring in faster than the thermostat removes it (instability)", self._fastest_atoms(st),
                        "shorten the time step, minimize longer, check the fastest atoms' residues")
        if self.ref is not None:
            if len(self.ca) >= 3:
                rmsd = _kabsch_rmsd(self.ref[self.ca], xyz[self.ca])
                sample["ca_rmsd_A"] = round(rmsd, 2)
                if rmsd > self.th["ca_rmsd_A"]:
                    self._warn("yellow", "unfolding", "Large structural drift",
                               f"Calpha RMSD {rmsd:.1f} A from the minimized structure during {stage}: unfolding "
                               "or a domain moving; real only if the protein is known to be flexible", (),
                               "check missing residues, protonation and the solvent model (implicit solvent "
                               "unfolds faster)")
            drifts = [(_ligand_drift(self.ref, xyz, idx, self.ca), label) for label, idx in self.ligands
                      if len(idx) >= 3]
            if drifts:
                drift, label = max(drifts)
                sample["ligand_drift_A"] = round(drift, 2)
                if drift > self.th["ligand_drift_A"]:
                    self._warn("yellow", "ligand_left", "Ligand left its starting site",
                               f"{label} moved {drift:.1f} A from its starting position (fitted on the "
                               f"protein) during {stage}", (label,),
                               "if unexpected: check its protonation/tautomer and charge, the pose, and waters "
                               "bridging it in the crystal")
        if self.periodic:
            vecs = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
            lengths = np.diag(np.asarray(vecs))
            vol_A3 = float(np.prod(lengths))
            dens = float(self.masses.sum()) * 1.66054 / vol_A3
            sample["density_g_ml"] = round(dens, 4)
            if stage == "production" and not (self.th["density_low"] < dens < self.th["density_high"]):
                self._warn("yellow", "density", "Density off",
                           f"density {dens:.3f} g/mL during production (water-protein systems sit near 1.0-1.05)",
                           (), "equilibrate longer under NPT; check that the box is filled with water")
            if len(self.solute):
                s = xyz[self.solute]
                span = s.max(axis=0) - s.min(axis=0)
                gap = float(np.min(lengths - span))
                if gap < self.cutoff_A:
                    self._warn("yellow", "image_contact", "Solute close to its periodic image",
                               f"only {max(gap, 0):.1f} A between the solute and its periodic copy along a box "
                               f"axis (cutoff {self.cutoff_A:.0f} A)", (), "use a larger box (padding)")
        if self.ensemble == "nve" and stage == "production":
            self._energy.append((self._steps * self.timestep_fs / 1000.0, pe + ke))
        self.samples.append(sample)
        if len(self.samples) > 4000:
            self.samples = self.samples[::2]
        return sample

    def worst_forces(self, positions=None, n=5):
        """The atoms with the largest forces (at ``positions``, e.g. the starting structure, if given)."""
        from openmm import unit

        if positions is not None:
            self.context.setPositions(positions)
        f = self.context.getState(getForces=True).getForces(asNumpy=True)
        norms = np.linalg.norm(np.asarray(f.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)), axis=1)
        norms = np.where(np.isfinite(norms), norms, np.inf)
        return [f"{self._residue_of(i)} {self.atoms[i].name} ({norms[i]:.0e} kJ/mol/nm)"
                for i in np.argsort(norms)[::-1][:n]]

    def explain_minimization_failure(self, exc, start_positions=None, known=()):
        """A finding for a minimization that ended in NaN: the worst forces of the starting structure and
        the red pre-checks that explain it."""
        try:
            where = self.worst_forces(start_positions)
        except Exception:
            where = []
        causes = [f["title"].lower() for f in known if f.get("level") == "red"]
        item = finding("red", "minimization_nan", "Minimization exploded (NaN)",
                       f"the energy became NaN while minimizing ({str(exc)[:80]})"
                       + (f"; likely cause: {', '.join(causes)}" if causes else
                          "; the largest starting forces are on the atoms below"),
                       where, "fix the structure first (prepare_protein), then check these atoms' residues and "
                       "ligand parameters")
        self.findings.append(item)
        return item

    def explain_exception(self, exc, stage, known=()):
        """A finding for an exception raised while stepping (OpenMM stops with 'Particle coordinate is NaN'),
        naming the atoms that were moving fastest at the last healthy check (or, before any check, those with
        the largest forces in the minimized structure) and the red pre-checks that explain it."""
        text = str(exc)
        if "nan" in text.lower():
            where, how = self._last_fast, "the fastest at the last check"
            if not where and self.ref is not None:
                try:
                    where, how = self.worst_forces(self.ref / 10.0), "those with the largest forces after minimization"
                except Exception:
                    where = []
            causes = [f["title"].lower() for f in known if f.get("level") == "red"]
            item = finding("red", "nan", "The simulation exploded (NaN)",
                           f"coordinates became NaN during {stage} (after {self._steps * self.timestep_fs / 1000.0:.1f}"
                           " ps)" + (f"; likely cause: {', '.join(causes)}" if causes else "")
                           + (f"; the atoms below were {how}" if where else ""),
                           where, "usually a clash, a too-long time step or bad ligand parameters; "
                           "minimize longer, shorten the time step, re-check those residues")
        else:
            item = finding("red", "step_error", "The simulation stopped with an error",
                           f"during {stage}: {text[:300]}", self._last_fast, "")
        self.findings.append(item)
        return item

    def _fastest_atoms(self, state, n=4):
        from openmm import unit

        try:
            v = np.asarray(state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))
            v = np.where(np.isfinite(v), v, 1e9)
            speed = np.linalg.norm(v, axis=1)
            order = np.argsort(speed)[::-1][:n]
            return [f"{self._residue_of(i)} {self.atoms[i].name}" for i in order]
        except Exception:
            return []

    def finish(self):
        """Close the run: temperature against the set point per stage, NVE drift.  Returns the report."""
        prod = [s for s in self.samples if s["stage"] == "production"]
        if prod and self.ensemble != "nve":
            tmean = float(np.mean([s["T_K"] for s in prod]))
            setp = float(np.mean([s["setpoint_K"] for s in prod]))
            if abs(tmean - setp) > self.th["temp_hot_frac"] * setp:
                self._warn("yellow", "temperature_off", "Temperature off the set point",
                           f"mean production T {tmean:.0f} K vs thermostat {setp:.0f} K", (),
                           "check the time step and the thermostat friction")
        if len(self._energy) >= 4:
            t = np.array([e[0] for e in self._energy])
            e = np.array([e[1] for e in self._energy])
            if t[-1] - t[0] > 0:
                slope = float(np.polyfit(t, e, 1)[0]) * 1000.0          # kJ/mol/ns
                per_dof = abs(slope) / self.dof
                if per_dof > self.th["nve_drift_kj_mol_ns_dof"]:
                    self._warn("yellow", "energy_drift", "Energy not conserved (NVE)",
                               f"total energy drifts {slope:+.0f} kJ/mol/ns ({per_dof:.3f} per degree of "
                               "freedom): the integration is not accurate", (),
                               "shorten the time step or tighten the constraint tolerance")
        return self.report()

    def report(self):
        prod = [s for s in self.samples if s["stage"] == "production"]
        stats = {}
        if prod:
            stats["production_T_K"] = round(float(np.mean([s["T_K"] for s in prod])), 1)
            for key in ("ca_rmsd_A", "ligand_drift_A", "density_g_ml"):
                vals = [s[key] for s in prod if key in s]
                if vals:
                    stats[f"max_{key}" if key != "density_g_ml" else "mean_density_g_ml"] = round(
                        float(max(vals) if key != "density_g_ml" else np.mean(vals)), 3)
        if self.max_force is not None:
            stats["max_force_after_minimization_kj_mol_nm"] = round(self.max_force, 1)
        return {"checks": len(self.samples), "stats": stats, "findings": list(self.findings),
                "samples": self.samples[-200:]}


def _kabsch_rmsd(ref, cur):
    a = ref - ref.mean(axis=0)
    b = cur - cur.mean(axis=0)
    u, s, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    e = (np.sum(a * a) + np.sum(b * b) - 2.0 * np.sum(s)) / len(a)
    return float(math.sqrt(max(e, 0.0)))


def _fit(ref, cur):
    """Rotation + translation that puts ``cur`` on ``ref``."""
    rc, cc = ref.mean(axis=0), cur.mean(axis=0)
    u, _s, vt = np.linalg.svd((cur - cc).T @ (ref - rc))
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return rot, cc, rc


def _ligand_drift(ref, cur, ligand, ca):
    if len(ca) >= 3:
        rot, cc, rc = _fit(ref[ca], cur[ca])
        lig = (cur[ligand] - cc) @ rot + rc
    else:
        lig = cur[ligand]
    return float(np.linalg.norm(lig.mean(axis=0) - ref[ligand].mean(axis=0)))


# ---------------------------------------------------------------------------------------------------- summary

def summarize(findings, stopped=None):
    """{"confidence": high|medium|low, "reasons": [...], "red": n, "yellow": n}"""
    findings = list(findings or [])
    red = [f for f in findings if f.get("level") == "red"]
    yellow = [f for f in findings if f.get("level") == "yellow"]
    if stopped or red:
        conf = "low"
    elif yellow:
        conf = "medium"
    else:
        conf = "high"
    reasons = [f"{f['title']}: {f['detail']}" for f in red + yellow][:8]
    return {"confidence": conf, "red": len(red), "yellow": len(yellow), "reasons": reasons}


def format_findings(findings, limit=8):
    lines = []
    for f in list(findings or [])[:limit]:
        where = f" [{'; '.join(f['where'][:3])}]" if f.get("where") else ""
        fix = f" -> {f['fix']}" if f.get("fix") else ""
        lines.append(f"{f['level'].upper()} {f['title']}: {f['detail']}{where}{fix}")
    return lines


def check_pdb(path, thresholds=None):
    """structure_findings on a PDB file (what the assistant runs before launching MD)."""
    from openmm import app

    pdb = app.PDBFile(str(path))
    return structure_findings(pdb.topology, pdb.positions, thresholds)
