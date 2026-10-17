"""Membrane systems: orient a membrane protein, embed it in a lipid bilayer, check and equilibrate it.

* Orientation: the OPM database (Orientations of Proteins in Membranes) gives structures already placed
  with the bilayer normal along z and the hydrophobic thickness; without OPM, transmembrane segments
  predicted from the sequence define the membrane axis and the hydrophobic centre.
* Bilayer: OpenMM ``Modeller.addMembrane`` (pre-equilibrated POPC, POPE, DLPC, DLPE, DMPC, DOPC, DPPC
  patches) on top of the component library (cofactors, ions, ligands keep their parameters).
  Mixtures and other lipids ("POPC:CHL1" 3:1, bacterial "POPE:POPG" 3:1, asymmetric "POPC//POPE:POPS")
  are packed around the protein by PACKMOL-Memgen (CHARMM lipid names, which both CHARMM36 and Amber
  lipid17 templates use); water and ions are then added by Leonardo so charges come from the force field.
* Checks: area per lipid, phosphate-phosphate thickness, and whether the transmembrane part of the
  protein sits in the hydrophobic core.
* Equilibration: semi-isotropic MonteCarloMembraneBarostat with protein restraints released in stages.
"""

import json
import math
import os
import re
import shutil
import sys
import time

import numpy as np

LIPIDS = ("POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC")
OPM_URL = "https://opm-assets.storage.googleapis.com/pdb/{code}.pdb"
# experimental references (area per lipid nm^2, P-P thickness nm) at ~303 K
REFERENCE = {"POPC": (0.643, 3.70), "POPE": (0.565, 3.95), "DLPC": (0.632, 3.08), "DLPE": (0.51, 3.30),
             "DMPC": (0.606, 3.53), "DOPC": (0.674, 3.67), "DPPC": (0.631, 3.80)}


# --------------------------------------------------------------------------- orientation


def fetch_opm(pdb_id, out_path, timeout=30):
    """Download the OPM-oriented structure; returns {"path", "half_thickness_nm"} or None."""
    import urllib.request

    code = str(pdb_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-z]{4}", code):
        return None
    try:
        with urllib.request.urlopen(OPM_URL.format(code=code), timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    if "ATOM" not in text:
        return None
    half = None
    m = re.search(r"1/2 of bilayer thickness:\s*([0-9.]+)", text)
    if m:
        half = float(m.group(1)) / 10.0
    lines = [ln for ln in text.splitlines() if not (ln.startswith(("ATOM", "HETATM")) and ln[17:20] == "DUM")]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"path": out_path, "half_thickness_nm": half, "source": "OPM"}


def _chain_sequences(topology):
    three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
             "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
             "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H",
             "HSE": "H", "HSP": "H", "CYX": "C", "CYM": "C", "MSE": "M"}
    out = []
    for chain in topology.chains():
        residues = [r for r in chain.residues() if r.name in three]
        if residues:
            out.append((chain, residues, "".join(three[r.name] for r in residues)))
    return out


def orient_by_tm_segments(topology, positions_nm):
    """
    Rotate/translate a membrane protein so its transmembrane helices run along z and their centre is
    at z=0 (the bilayer centre), from TM segments predicted on the sequence. Returns (positions_nm, info).
    """
    from .bioseq_runtime import predict_tm_segments

    ca = []
    segments = []
    for chain, residues, seq in _chain_sequences(topology):
        pred = predict_tm_segments(seq)
        rows = pred.get("segments", pred) if isinstance(pred, dict) else pred
        for seg in list(rows or []):
            start, end = int(seg["start"]), int(seg["end"])
            segments.append(f"{chain.id}:{residues[start - 1].id}-{residues[end - 1].id}")
            for r in residues[max(0, start - 1):end]:
                for a in r.atoms():
                    if a.name == "CA":
                        ca.append(positions_nm[a.index])
    if len(ca) < 10:
        raise ValueError("no transmembrane segments predicted: this does not look like a membrane protein "
                         "(give an OPM-oriented structure or orient it yourself)")
    X = np.array(ca)
    center = X.mean(axis=0)
    _u, _s, vt = np.linalg.svd(X - center)
    axis = vt[0]
    if axis[2] < 0:
        axis = -axis
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, z)
    c = float(np.dot(axis, z))
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    P = np.array(positions_nm)
    new = (P - center) @ R.T
    return [tuple(p) for p in new], {"source": "predicted TM segments", "segments": segments,
                                      "half_thickness_nm": None}


# --------------------------------------------------------------------------- analysis


def membrane_metrics(topology, positions_nm, box_nm, lipid="POPC", lipid_names=None):
    """Area per lipid, P-P thickness and lipid count of a built bilayer (+ reference values).
    With ``lipid_names`` (mixtures) every lipid molecule counts, sterols included."""
    if lipid_names:
        wanted = _lipid_name_set(lipid_names)
        lipid_res = [r for r in topology.residues() if r.name in wanted]
    else:
        lipid_res = [r for r in topology.residues() if _is_lipid_residue(r)]
    phos = [positions_nm[a.index] for r in lipid_res for a in r.atoms() if a.element is not None
            and a.element.symbol == "P"]
    if lipid_names:
        n_lipids = len(lipid_res)
    else:
        n_lipids = sum(1 for r in lipid_res if any(a.element is not None and a.element.symbol == "P"
                                                   for a in r.atoms()))
    out = {"lipid": lipid, "n_lipids": int(n_lipids)}
    if phos and box_nm is not None:
        z = np.array([p[2] for p in phos])
        # leaflets split at the largest gap in z (not the median: leaflets can hold different numbers of
        # lipids - asymmetric mixtures, molecules removed after packing)
        zs = np.sort(z)
        if len(zs) > 1:
            k = int(np.argmax(np.diff(zs)))
            split = 0.5 * (zs[k] + zs[k + 1])
        else:
            split = float(zs[0])
        up, low = z[z > split], z[z <= split]
        mid = 0.5 * (float(up.mean()) + float(low.mean())) if len(up) and len(low) else float(np.median(z))
        out["thickness_PP_nm"] = round(float(up.mean() - low.mean()), 3) if len(up) and len(low) else None
        box_area = float(box_nm[0] * box_nm[1])
        prot_area = _protein_cross_section_nm2(topology, positions_nm, mid, box_nm, lipid_names=lipid_names)
        out["protein_cross_section_nm2"] = round(prot_area, 2)
        out["area_per_lipid_nm2"] = round((box_area - prot_area) / max(1, n_lipids / 2.0), 4)
        out["membrane_center_z_nm"] = round(mid, 3)
    ref = REFERENCE.get(str(lipid).upper())
    if ref:
        out["reference_area_per_lipid_nm2"], out["reference_thickness_PP_nm"] = ref
    return out


def _protein_cross_section_nm2(topology, positions_nm, center_z, box_nm, slab_nm=0.5, radius_nm=0.2, grid_nm=0.1,
                              lipid_names=()):
    """Area the protein occupies in the bilayer plane: occupancy grid of protein heavy atoms in a central slab."""
    skip = {"HOH", "WAT", "NA", "CL", "SOD", "CLA", "POT", "K"} | _lipid_name_set(lipid_names)
    pts = [positions_nm[a.index] for a in topology.atoms()
           if a.residue.name not in skip and not _is_lipid_residue(a.residue) and a.element is not None
           and a.element.symbol != "H" and abs(positions_nm[a.index][2] - center_z) <= slab_nm]
    if not pts:
        return 0.0
    lx, ly = float(box_nm[0]), float(box_nm[1])
    nx, ny = max(1, int(lx / grid_nm)), max(1, int(ly / grid_nm))
    occ = np.zeros((nx, ny), dtype=bool)
    r_cells = int(math.ceil(radius_nm / grid_nm))
    offs = [(i, j) for i in range(-r_cells, r_cells + 1) for j in range(-r_cells, r_cells + 1)
            if (i * grid_nm) ** 2 + (j * grid_nm) ** 2 <= radius_nm ** 2]
    for x, y, _z in pts:
        cx, cy = int((x % lx) / grid_nm), int((y % ly) / grid_nm)
        for i, j in offs:
            occ[(cx + i) % nx, (cy + j) % ny] = True
    return float(occ.sum()) * grid_nm * grid_nm


def _lipid_name_set(lipid_names):
    """Lipid residue names plus their 3-character PDB forms (PDB files truncate POPC -> POP, CHL1 -> CHL)."""
    names = set(lipid_names or ())
    return names | {n[:3] for n in names}


def _is_lipid_residue(res):
    names = {a.name for a in res.atoms()}
    return str(res.name).upper() in {"POP", "POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC", "PC", "PE",
                                      "PA", "OL", "MY", "LA", "ST", "CHL1", "CHOL", "CHL", "PSM", "SSM"} or \
        ("P" in names and len(names) > 40)


def parse_lipid_spec(lipid, ratio=""):
    """'POPC:CHL1' + '3:1' -> ('POPC:CHL1', '3:1', ['POPC', 'CHL1']); '//' separates upper//lower leaflets."""
    spec = re.sub(r"\s+", "", str(lipid or "POPC").upper())
    if re.search(r"(?<!/)/(?!/)", spec):
        raise ValueError("separate leaflets with '//' (upper//lower), mix lipids with ':'")
    names = [x for x in re.split(r"[:/]+", spec) if x]
    if not names:
        raise ValueError("empty lipid specification")
    ratio = re.sub(r"\s+", "", str(ratio or ""))
    if not ratio:
        ratio = "//".join(":".join("1" for _ in leaf.split(":")) for leaf in spec.split("//"))
    if [len(leaf.split(":")) for leaf in spec.split("//")] != [len(leaf.split(":")) for leaf in ratio.split("//")]:
        raise ValueError(f"lipid ratio {ratio!r} does not match the lipids {spec!r}")
    for x in re.split(r"[:/]+", ratio):
        if x and float(x) <= 0:
            raise ValueError("lipid ratios must be positive")
    return spec, ratio, list(dict.fromkeys(names))


def is_mixture(lipid):
    spec = str(lipid or "").upper()
    return ":" in spec or "/" in spec or (spec not in LIPIDS and bool(spec))


def _packmol_memgen():
    """(executable, environment) for PACKMOL-Memgen, which looks for packmol under $AMBERHOME/bin."""
    import shutil

    exe = os.path.join(os.path.dirname(sys.executable), "packmol-memgen")
    if not os.path.isfile(exe):
        exe = shutil.which("packmol-memgen")
    if not exe:
        return None, None
    env = dict(os.environ)
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(exe)))
    if not os.path.isfile(os.path.join(env.get("AMBERHOME", ""), "bin", "packmol")) and \
            os.path.isfile(os.path.join(prefix, "bin", "packmol")):
        env["AMBERHOME"] = prefix
    env["PATH"] = os.path.dirname(exe) + os.pathsep + env.get("PATH", "")
    return exe, env


_WATER_NAMES = {"TIP3", "WAT", "HOH", "SOL", "TIP3P"}
_ION_NAMES = {"NA+": "NA", "CL-": "CL", "K+": "K", "SOD": "NA", "CLA": "CL", "POT": "K"}


def pack_lipid_mixture(modeller, forcefield, family, lipid, ratio="", padding_nm=1.0, water_nm=1.5,
                       ionic_strength_molar=0.15, out_dir=".", nloop=20, nloop_all=100, seed=None, progress=None,
                       max_attempts=2):
    """
    Pack a (mixed) bilayer around the prepared, oriented solute (bilayer centre z=0, normal z) with
    PACKMOL-Memgen, add it to ``modeller`` with template bonds, then neutralize and add salt by replacing
    waters. Returns {"lipid_residue_names", "counts", "ions", "packmol_seconds"}.
    """
    import subprocess

    import openmm as mm
    import openmm.app as app
    from openmm import unit

    spec, ratio, names = parse_lipid_spec(lipid, ratio)
    missing = [n for n in names if n not in forcefield._templates]
    if missing:
        common = ("POPC", "POPE", "POPG", "POPS", "POPA", "DOPC", "DOPE", "DOPS", "DPPC", "DMPC", "DLPC", "DSPC",
                  "CHL1", "PSM", "SSM")
        have = [n for n in common if n in forcefield._templates]
        raise ValueError(f"no {family} template for {', '.join(missing)} (CHARMM lipid names; available here: "
                         f"{', '.join(have)})")
    exe, env = _packmol_memgen()
    if exe is None:
        raise RuntimeError("PACKMOL-Memgen is not installed (conda install -c conda-forge ambertools)")
    solute = os.path.join(out_dir, "packmol_solute.pdb")
    with open(solute, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)
    out_pdb = os.path.join(out_dir, "packmol_bilayer.pdb")
    has_solute = any(True for _ in modeller.topology.atoms())
    cmd = [exe, "--lipids", spec, "--ratio", ratio, "--dist_wat", f"{10 * water_nm:.1f}", "--nocounter",
           "--charmm", "--overwrite", "--noprogress", "-o", out_pdb]
    if has_solute:
        cmd += ["--pdb", solute, "--preoriented", "--notprotonate", "--nottrim", "--keepligs",
                "--dist", f"{10 * padding_nm:.1f}"]
    else:
        cmd += ["--distxy_fix", f"{10 * max(padding_nm, 3.0):.1f}"]
    t0 = time.time()
    clashes, attempts = None, 0
    best = None                                    # (clashes, saved PDB) of the cleanest attempt
    for attempt in range(max(1, int(max_attempts))):
        attempts = attempt + 1
        # each retry gives PACKMOL more optimization loops and re-places badly packed molecules, which is
        # what large proteins and sterol-rich mixtures need (a new seed alone rarely helps)
        scale = attempt + 1
        run_cmd = cmd + ["--nloop", str(max(10, int(nloop)) * scale),
                         "--nloop_all", str(max(10, int(nloop_all)) * scale)]
        if seed is None or attempt > 0:
            run_cmd += ["--random"]
        if attempt > 0:
            run_cmd += ["--movebadrandom"]
        if callable(progress):
            progress(f"packing {spec} ({ratio}) with PACKMOL-Memgen (minutes)"
                     + (f", attempt {attempts} with {scale}x optimization loops (overlapping molecules in the "
                        "previous one)" if attempt else ""))
        for stale in (out_pdb, out_pdb + "_FORCED"):
            if os.path.exists(stale):
                os.remove(stale)
        run = subprocess.run(run_cmd, cwd=out_dir, env=env, capture_output=True, text=True)
        if not os.path.isfile(out_pdb):
            tail = (run.stdout + run.stderr)[-800:]
            raise RuntimeError(f"PACKMOL-Memgen failed: {tail}")
        packed = app.PDBFile(out_pdb)
        top, pos = packed.topology, packed.positions.value_in_unit(unit.nanometer)
        # PACKMOL often stops short of full convergence on large systems: hydrogen contacts are pushed apart
        # later (soft push-off); heavy atoms inside each other (e.g. a tail through a sterol ring) are not,
        # so those trigger a repack
        pairs = _clash_pairs(top, np.array([tuple(p) for p in pos]), cutoff_nm=0.06, heavy_only=True)
        clashes = len(pairs)
        if best is None or clashes < best[0]:
            keep = os.path.join(out_dir, "packmol_bilayer_best.pdb")
            shutil.copyfile(out_pdb, keep)
            if os.path.exists(out_pdb + "_FORCED"):
                shutil.copyfile(out_pdb + "_FORCED", keep + "_FORCED")
            elif os.path.exists(keep + "_FORCED"):
                os.remove(keep + "_FORCED")
            best = (clashes, keep)
        if clashes == 0:
            break
    if best is not None and best[0] < clashes:
        # a later attempt was worse: go back to the cleanest packing
        shutil.copyfile(best[1], out_pdb)
        if os.path.exists(best[1] + "_FORCED"):
            shutil.copyfile(best[1] + "_FORCED", out_pdb + "_FORCED")
        elif os.path.exists(out_pdb + "_FORCED"):
            os.remove(out_pdb + "_FORCED")
        packed = app.PDBFile(out_pdb)
        top, pos = packed.topology, packed.positions.value_in_unit(unit.nanometer)
        pairs = _clash_pairs(top, np.array([tuple(p) for p in pos]), cutoff_nm=0.06, heavy_only=True)
        clashes = len(pairs)
    removed_overlapping = []
    if clashes:
        # still overlapping (small or sterol-rich boxes): drop one molecule of each overlapping pair - water
        # first, else the smaller one; never the solute (its copy is replaced below anyway)
        residues = list(top.residues())
        solute = {r.index for r in residues if r.name not in set(names) and r.name.upper() not in _WATER_NAMES}
        drop = set()
        for i, j in pairs:
            if i in drop or j in drop:
                continue
            cand = [k for k in (i, j) if k not in solute]
            if not cand:
                continue
            cand.sort(key=lambda k: (residues[k].name.upper() not in _WATER_NAMES, len(list(residues[k].atoms()))))
            drop.add(cand[0])
        removed_overlapping = sorted(residues[k].name for k in drop)
        edit = app.Modeller(top, packed.positions)
        edit.delete([residues[k] for k in drop])
        top, packed_positions = edit.topology, edit.positions
        pos = packed_positions.value_in_unit(unit.nanometer)
    else:
        packed_positions = packed.positions
    # PACKMOL-Memgen may re-centre the solute: map its copy back onto the prepared coordinates
    ref = [modeller.positions[a.index].value_in_unit(unit.nanometer) for a in modeller.topology.atoms()
           if a.name == "CA"]
    got = [pos[a.index] for a in top.atoms() if a.name == "CA" and a.residue.name not in _WATER_NAMES]
    shift = mm.Vec3(0, 0, 0)
    if ref and len(ref) == len(got):
        shift = (np.mean(np.array([tuple(v) for v in ref]), axis=0) - np.mean(np.array([tuple(v) for v in got]),
                                                                              axis=0))
        shift = mm.Vec3(*[float(x) for x in shift])
    elif ref:
        raise RuntimeError(f"PACKMOL-Memgen changed the solute ({len(got)} vs {len(ref)} C-alpha atoms)")
    lipid_names = set(names)
    keep = app.Modeller(top, packed_positions)
    keep.delete([r for r in top.residues() if r.name not in lipid_names and r.name.upper() not in _WATER_NAMES])
    ktop = keep.topology
    for res in ktop.residues():
        if res.name.upper() in _WATER_NAMES:
            res.name = "HOH"
            for a in res.atoms():
                a.name = {"OH2": "O", "OW": "O", "HW1": "H1", "HW2": "H2"}.get(a.name, a.name)
    counts = {}
    for res in ktop.residues():
        counts[res.name] = counts.get(res.name, 0) + 1
        tmpl = forcefield._templates.get(res.name)
        if res.name == "HOH" or tmpl is None:
            continue
        by = {a.name: a for a in res.atoms()}
        for i, j in tmpl.bonds:
            x, y = by.get(tmpl.atoms[i].name), by.get(tmpl.atoms[j].name)
            if x is not None and y is not None:
                ktop.addBond(x, y)
    water = [r for r in ktop.residues() if r.name == "HOH"]
    for r in water:
        names_w = {a.name for a in r.atoms()}
        if names_w != {"O", "H1", "H2"}:
            raise RuntimeError(f"unexpected water atoms {sorted(names_w)} in the PACKMOL-Memgen output")
    kpos = [p + shift * unit.nanometer for p in keep.positions] if has_solute else keep.positions
    modeller.add(ktop, kpos)
    allpos = np.array([tuple(p) for p in modeller.positions.value_in_unit(unit.nanometer)])
    # coordinate extents, as PACKMOL-Memgen's own tleap setup does: lipid tails reach past the packing
    # regions, so a box from those regions makes tails overlap their periodic images. The small lateral
    # gaps this leaves (area per lipid overestimated) close in the NPT equilibration.
    box = allpos.max(axis=0) - allpos.min(axis=0) + 0.1
    modeller.topology.setPeriodicBoxVectors([mm.Vec3(box[0], 0, 0), mm.Vec3(0, box[1], 0), mm.Vec3(0, 0, box[2])])
    ions = add_ions_by_water_replacement(modeller, forcefield, family, ionic_strength_molar, lipid_names,
                                         seed=seed)
    return {"lipid_spec": spec, "lipid_ratio": ratio, "lipid_residue_names": sorted(lipid_names),
            "packing_attempts": attempts, "overlapping_pairs": int(clashes or 0),
            "removed_overlapping": removed_overlapping,
            "counts": {k: v for k, v in counts.items() if k in lipid_names}, "n_waters": counts.get("HOH", 0),
            "ions": ions, "packmol_seconds": round(time.time() - t0, 1),
            "packmol_converged": not os.path.exists(out_pdb + "_FORCED")}


def _clash_pairs(topology, positions_nm, cutoff_nm=0.08, heavy_only=False):
    """Residue-index pairs with atoms closer than ``cutoff_nm``, periodic in x/y/z with the box from the
    extents. With ``heavy_only`` only non-hydrogen atoms count (true interpenetration, which minimization
    cannot undo; H-H contacts are resolved by the soft push-off before minimization)."""
    from scipy.spatial import cKDTree

    atoms = [a for a in topology.atoms()
             if not heavy_only or (a.element is not None and a.element.symbol != "H")]
    if not atoms:
        return []
    X = positions_nm[[a.index for a in atoms]].copy()
    lo = X.min(axis=0)
    box = X.max(axis=0) - lo + 0.1
    X = X - lo
    tree = cKDTree(np.mod(X, box), boxsize=box)
    res = [a.residue.index for a in atoms]
    return sorted({tuple(sorted((res[i], res[j]))) for i, j in tree.query_pairs(cutoff_nm) if res[i] != res[j]})


def soft_push_off(system, positions, max_iterations=2000, d0_nm=0.22, k_kj_nm2=20000.0):
    """
    Separate overlapping atoms before a real minimization: bonded terms are kept, every nonbonded force is
    replaced by a capped repulsion k*(d0 - r)^2 for r < d0 (same exclusions). Bounded forces let the
    minimizer untangle the near-coincident atoms PACKMOL can leave, which would otherwise give Lennard-Jones
    energies of ~1e11 kJ/mol and stall the real minimization. Returns the new positions.
    """
    import openmm as mm
    from openmm import unit

    soft = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    exclusions = set()
    for i in reversed(range(soft.getNumForces())):
        f = soft.getForce(i)
        if isinstance(f, mm.NonbondedForce):
            for k in range(f.getNumExceptions()):
                a, b, *_ = f.getExceptionParameters(k)
                exclusions.add((min(a, b), max(a, b)))
            soft.removeForce(i)
        elif isinstance(f, mm.CustomNonbondedForce):
            for k in range(f.getNumExclusions()):
                a, b = f.getExclusionParticles(k)
                exclusions.add((min(a, b), max(a, b)))
            soft.removeForce(i)
        elif isinstance(f, mm.CustomBondForce) and "^12" in f.getEnergyFunction():
            soft.removeForce(i)                  # CHARMM LJ 1-4 pairs (Lennard-Jones, not bounded)
    rep = mm.CustomNonbondedForce("k_soft*step(d_soft-r)*(d_soft-r)^2")
    rep.addGlobalParameter("k_soft", float(k_kj_nm2))
    rep.addGlobalParameter("d_soft", float(d0_nm))
    rep.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    rep.setCutoffDistance(float(d0_nm) + 0.05)
    for _ in range(soft.getNumParticles()):
        rep.addParticle([])
    for a, b in exclusions:
        rep.addExclusion(a, b)
    soft.addForce(rep)
    integ = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = None
    for plat in _platform_order():
        try:
            context = mm.Context(soft, integ, mm.Platform.getPlatformByName(plat))
            break
        except Exception:
            integ = mm.VerletIntegrator(0.001 * unit.picoseconds)
            continue
    if context is None:
        context = mm.Context(soft, integ)
    context.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(context, 10.0, int(max_iterations))
    return context.getState(getPositions=True).getPositions()


def _severe_clashes(topology, positions_nm, cutoff_nm=0.08):
    return len(_clash_pairs(topology, positions_nm, cutoff_nm))


def lipid_residue_templates(topology, forcefield, lipid_names):
    """Explicit templates for lipids (CHARMM36 has look-alikes, e.g. CHL1 vs CLOL)."""
    return {r: r.name for r in topology.residues() if r.name in lipid_names and r.name in forcefield._templates}


def add_ions_by_water_replacement(modeller, forcefield, family, ionic_strength_molar, lipid_names, seed=None,
                                  min_dist_nm=0.5):
    """Neutralize and add NaCl like Modeller.addSolvent: replace waters away from solute and lipids."""
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    top = modeller.topology
    system = forcefield.createSystem(top, residueTemplates=lipid_residue_templates(top, forcefield, lipid_names))
    charge = 0.0
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            charge = sum(force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                         for i in range(force.getNumParticles()))
    waters = [r for r in top.residues() if r.name == "HOH"]
    n_pairs = int(round(float(ionic_strength_molar) * len(waters) / 55.4))
    q = int(round(charge))
    n_na, n_cl = n_pairs + max(0, -q), n_pairs + max(0, q)
    if n_na + n_cl == 0:
        return {"NA": 0, "CL": 0, "charge_before": round(charge, 3)}
    pos = np.array([tuple(p) for p in modeller.positions.value_in_unit(unit.nanometer)])
    other = np.array([pos[a.index] for a in top.atoms() if a.residue.name != "HOH"])
    cands = []
    for r in waters:
        o = next(a for a in r.atoms() if a.name == "O")
        d = np.min(np.linalg.norm(other - pos[o.index], axis=1)) if len(other) else 9.9
        if d >= min_dist_nm:
            cands.append((r, pos[o.index]))
    if len(cands) < n_na + n_cl:
        raise RuntimeError(f"not enough bulk water to place {n_na + n_cl} ions")
    rng = np.random.default_rng(seed)
    chosen = []
    for idx in rng.permutation(len(cands)):   # keep ions >= 0.5 nm apart
        r, p = cands[idx]
        if all(np.linalg.norm(p - q_) >= min_dist_nm for _r, q_ in chosen):
            chosen.append((r, p))
        if len(chosen) == n_na + n_cl:
            break
    if len(chosen) < n_na + n_cl:
        raise RuntimeError("could not place the ions")
    na, cl = ("SOD", "CLA") if family == "charmm36" else ("NA", "CL")
    ion_top = app.Topology()
    chain = ion_top.addChain()
    ion_pos = []
    for k, (_r, p) in enumerate(chosen):
        name, elem = (na, app.element.sodium) if k < n_na else (cl, app.element.chlorine)
        res = ion_top.addResidue(name, chain)
        ion_top.addAtom(name, elem, res)
        ion_pos.append(mm.Vec3(*[float(x) for x in p]))
    box = top.getPeriodicBoxVectors()
    modeller.delete([r for r, _p in chosen])
    modeller.add(ion_top, unit.Quantity(ion_pos, unit.nanometer))
    modeller.topology.setPeriodicBoxVectors(box)
    return {"NA": n_na, "CL": n_cl, "charge_before": round(charge, 3)}


def protein_in_core(topology, positions_nm, center_z_nm, half_thickness_nm=1.5):
    """Fraction of predicted-TM Calpha atoms inside the hydrophobic slab (a sanity check of the orientation)."""
    try:
        from .bioseq_runtime import predict_tm_segments
    except Exception:
        return None
    inside = total = 0
    for _chain, residues, seq in _chain_sequences(topology):
        pred = predict_tm_segments(seq)
        rows = pred.get("segments", pred) if isinstance(pred, dict) else pred
        for seg in list(rows or []):
            start, end = int(seg.get("start")), int(seg.get("end"))
            for r in residues[max(0, start - 1):end]:
                for a in r.atoms():
                    if a.name == "CA":
                        total += 1
                        inside += abs(positions_nm[a.index][2] - center_z_nm) <= half_thickness_nm
    return None if total == 0 else round(inside / total, 3)


# --------------------------------------------------------------------------- build


def build_membrane_system(pdb_path, lipid="POPC", family="amber14", ph=7.0, padding_nm=1.0,
                          ionic_strength_molar=0.15, orientation="auto", pdb_id="", out_dir=None,
                          minimize_steps=500, keep_waters=False, remove_resn=(), progress=None,
                          lipid_ratio="", water_nm=1.5, seed=None):
    """
    Oriented membrane protein + components (component library) + bilayer + water + ions, minimized
    and checked. orientation: "opm" (download by pdb_id), "tm" (predicted TM segments), "as_is"
    (already along z, bilayer centre at z=0) or "auto" (OPM if pdb_id, else TM). ``lipid`` is one of
    LIPIDS (OpenMM patch) or a PACKMOL-Memgen mixture such as "POPC:CHL1" with ``lipid_ratio`` "3:1".
    """
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    from . import component_library_runtime as cl

    def emit(msg):
        if callable(progress):
            progress(msg)

    lipid = str(lipid or "POPC").upper().replace(" ", "")
    mixture = is_mixture(lipid)
    if mixture:
        lipid, lipid_ratio, _names = parse_lipid_spec(lipid, lipid_ratio)
    out_dir = out_dir or os.path.join(cl._cache_dir(), "membranes", time.strftime("mem_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    report = {"lipid": lipid, "family": family, "ok": False, "out_dir": out_dir}
    if mixture:
        report["lipid_ratio"] = lipid_ratio
    src = pdb_path
    orient_info = {}
    mode = orientation
    if mode == "auto":
        mode = "opm" if pdb_id else "tm"
    if mode == "opm":
        emit("downloading the OPM orientation")
        got = fetch_opm(pdb_id, os.path.join(out_dir, f"{pdb_id.lower()}_opm.pdb"))
        if got is None:
            report["notes"] = [f"OPM has no entry for {pdb_id}; orienting from predicted TM segments"]
            mode = "tm"
        else:
            src, orient_info = got["path"], got
    pdb = app.PDBFile(src)
    positions = pdb.positions
    if mode == "tm":
        emit("orienting from predicted transmembrane segments")
        pos_nm, orient_info = orient_by_tm_segments(pdb.topology, [tuple(p.value_in_unit(unit.nanometer))
                                                                    for p in pdb.positions])
        positions = unit.Quantity([mm.Vec3(*p) for p in pos_nm], unit.nanometer)
    report["orientation"] = {k: v for k, v in orient_info.items() if k != "path"}
    oriented = os.path.join(out_dir, "oriented_input.pdb")
    with open(oriented, "w") as fh:
        app.PDBFile.writeFile(pdb.topology, positions, fh, keepIds=True)
    emit("assigning component parameters")
    prep = cl.prepare_system_for_solvation(oriented, family=family, ph=ph, keep_waters=keep_waters,
                                           remove_resn=remove_resn)
    if not prep["ok"]:
        report["error"] = prep["error"]
        report["plan"] = prep.get("plan")
        return report
    modeller, ff = prep["modeller"], prep["forcefield"]
    report.update({k: prep[k] for k in ("files", "protonation_variants", "gaff_components",
                                        "covalent_components", "chain_breaks") if k in prep})
    emit(f"embedding in a {lipid} bilayer (minutes)")
    t0 = time.time()
    last_error = None
    lipid_names = None
    if mixture:
        try:
            packed = pack_lipid_mixture(modeller, ff, family, lipid, lipid_ratio, padding_nm=padding_nm,
                                        water_nm=water_nm, ionic_strength_molar=ionic_strength_molar,
                                        out_dir=out_dir, seed=seed, progress=emit)
        except Exception as exc:
            report["error"] = f"membrane build failed: {exc}"
            return report
        lipid_names = packed["lipid_residue_names"]
        report["packing"] = packed
        report["build_platform"] = "PACKMOL-Memgen"
    for plat in ([] if mixture else _platform_order()):
        try:
            modeller.addMembrane(ff, lipidType=lipid, membraneCenterZ=0 * unit.nanometer,
                                 minimumPadding=float(padding_nm) * unit.nanometer,
                                 ionicStrength=float(ionic_strength_molar) * unit.molar,
                                 platform=mm.Platform.getPlatformByName(plat))
            report["build_platform"] = plat
            break
        except Exception as exc:          # OpenCL on macOS can give NaN with CHARMM36: retry on CPU
            last_error = exc
            modeller = prep["rebuild"]()
    else:
        if not mixture:
            report["error"] = f"membrane build failed: {last_error}"
            return report
    report["build_seconds"] = round(time.time() - t0, 1)
    emit("creating the system")
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds,
                             residueTemplates=lipid_residue_templates(modeller.topology, ff, lipid_names or ()))
    box = modeller.topology.getUnitCellDimensions().value_in_unit(unit.nanometer)
    pos_nm = [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]
    report["metal_restraints"] = cl.add_coordination_restraints(system, modeller.topology, pos_nm)
    report["box_nm"] = [round(float(x), 3) for x in box]
    report["n_atoms"] = system.getNumParticles()
    charge = 0.0
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            charge = sum(force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                         for i in range(system.getNumParticles()))
    report["total_charge"] = round(charge, 3)
    if mixture:
        emit("separating close contacts left by PACKMOL")
        modeller.positions = soft_push_off(system, modeller.positions)
    emit("minimizing")
    integ = mm.LangevinMiddleIntegrator(303 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = _simulation(app, mm, modeller.topology, system, integ)
    report.update(simulation_platform(sim))
    sim.context.setPositions(modeller.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    sim.minimizeEnergy(maxIterations=int(minimize_steps))
    state = sim.context.getState(getEnergy=True, getPositions=True)
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    report["energy_start_kj"], report["energy_minimized_kj"] = round(e0, 1), round(e1, 1)
    pos_nm = [tuple(p) for p in state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)]
    metrics = membrane_metrics(modeller.topology, pos_nm, box, lipid=lipid, lipid_names=lipid_names)
    report["membrane"] = metrics
    half = orient_info.get("half_thickness_nm") or 1.5
    report["tm_calpha_in_core_fraction"] = protein_in_core(modeller.topology, pos_nm,
                                                           metrics.get("membrane_center_z_nm", 0.0), half)
    pdb_out = os.path.join(out_dir, "membrane_system_minimized.pdb")
    with open(pdb_out, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, state.getPositions(), fh, keepIds=True)
    xml_out = os.path.join(out_dir, "membrane_system.xml")
    with open(xml_out, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(system))
    report.update(pdb=pdb_out, system_xml=xml_out)
    if lipid_names:        # PDB truncates 4-letter lipid names; mmCIF keeps them (for reuse with force fields)
        cif_out = os.path.join(out_dir, "membrane_system_minimized.cif")
        with open(cif_out, "w") as fh:
            app.PDBxFile.writeFile(modeller.topology, state.getPositions(), fh, keepIds=True)
        report.update(cif=cif_out, lipid_residue_names=list(lipid_names))
    warnings = []
    if abs(charge - round(charge)) > 0.01:
        warnings.append(f"non-integer total charge {charge:.3f}")
    if not math.isfinite(e1) or e1 > e0 + max(50.0, 0.02 * abs(e0)) or e1 > 0:
        # a solvated system always minimizes to a negative potential energy: a positive one means overlaps
        # the minimizer could not resolve
        warnings.append("minimization was not stable" + (" (positive energy: unresolved overlaps)" if e1 > 0 else ""))
    apl, ref_apl = metrics.get("area_per_lipid_nm2"), metrics.get("reference_area_per_lipid_nm2")
    removed = report.get("packing", {}).get("removed_overlapping") if mixture else None
    if removed:
        warnings.append(f"{len(removed)} molecule(s) removed because PACKMOL left them overlapping ("
                        + ", ".join(sorted(set(removed))) + "): the box is small for this mixture; a larger "
                        "padding gives a cleaner packing")
    if mixture and not report.get("packing", {}).get("packmol_converged", True):
        warnings.append("PACKMOL did not fully converge (a *_FORCED structure was used): check for clashes or "
                        "rebuild with more loops/another seed; minimization may not remove every overlap")
    if mixture:
        warnings.append("PACKMOL places lipids without pre-equilibration and the box comes from the coordinate "
                        "extents (small vacuum gaps at the lateral faces, area per lipid overestimated): equilibrate "
                        "(equilibrate_membrane_system, several ns for sterol-rich mixtures) before production; no "
                        "experimental area reference is used for mixtures")
    elif apl and ref_apl and abs(apl - ref_apl) / ref_apl > 0.15:
        warnings.append(f"area per lipid {apl} nm2 vs {ref_apl} experimental: expected right after insertion "
                        "(gaps next to the protein); equilibrate before production")
    frac = report["tm_calpha_in_core_fraction"]
    if frac is not None and frac < 0.6:
        warnings.append(f"only {frac:.0%} of the predicted TM residues lie in the hydrophobic core: check the "
                        "orientation (use an OPM structure)")
    report["warnings"] = warnings
    report["ok"] = not any("charge" in w or "stable" in w for w in warnings)
    with open(os.path.join(out_dir, "membrane_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


def _platform_order():
    """GPUs first (these jobs run in their own process, so a driver crash cannot take PyMOL down), the platform
    asked for in PROTLIGINTERACT_OPENMM_PLATFORM / LEONARDO_OPENMM_PLATFORM before everything else. HIP and Metal
    were missing: AMD GPUs without OpenCL ran on the CPU."""
    import openmm as mm

    try:
        from .abfe_runtime import _canonical_openmm_platform_name, openmm_platform_override
    except ImportError:
        from abfe_runtime import _canonical_openmm_platform_name, openmm_platform_override
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    wanted = _canonical_openmm_platform_name(openmm_platform_override())
    order = ([wanted] if wanted in names else []) + [p for p in ("CUDA", "HIP", "Metal", "OpenCL", "CPU")
                                                     if p in names and p != wanted]
    return order or ["Reference"]


def _gpu_properties(plat):
    try:
        from .abfe_runtime import openmm_gpu_properties
    except ImportError:
        from abfe_runtime import openmm_gpu_properties
    return openmm_gpu_properties(plat)[0]


def _simulation(app, mm, topology, system, integrator):
    """A Simulation on the first platform that starts (mixed precision on GPUs that support it). The platform
    that ran is sim.context.getPlatform().getName()."""
    for plat in _platform_order():
        for props in ([_gpu_properties(plat)] if _gpu_properties(plat) else []) + [{}]:
            try:
                return app.Simulation(topology, system, integrator, mm.Platform.getPlatformByName(plat), props)
            except Exception:
                continue
    return app.Simulation(topology, system, integrator)


def simulation_platform(sim):
    """{"platform": "CUDA", "precision": "mixed"} of a running Simulation, for reports."""
    try:
        plat = sim.context.getPlatform()
        prec = plat.getPropertyValue(sim.context, "Precision") if "Precision" in plat.getPropertyNames() else ""
        return {"platform": plat.getName(), "precision": prec}
    except Exception:
        return {"platform": "", "precision": ""}


# --------------------------------------------------------------------------- equilibration


def equilibrate_membrane(pdb_path, system_xml, out_dir, stages_ps=(50, 50, 100), restraint_k=(1000, 400, 50),
                         temperature_k=303.15, pressure_bar=1.0, timestep_fs=2.0, report_every_ps=10.0,
                         progress=None, lipid_names=None, lipid_label=None):
    """
    Staged NPT equilibration of a membrane system: heavy-atom protein restraints (kJ/mol/nm^2) released
    stage by stage, semi-isotropic MonteCarloMembraneBarostat (xy coupled, z free), Langevin thermostat.
    Writes a trajectory, a CSV of area per lipid / thickness / box, and the final structure.
    """
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    pdb = app.PDBFile(pdb_path)
    with open(system_xml) as fh:
        system = mm.XmlSerializer.deserialize(fh.read())
    lipid_set = _lipid_name_set(lipid_names)
    protein = [a.index for a in pdb.topology.atoms() if a.residue.name not in ("HOH", "NA", "CL", "SOD", "CLA", "POT")
               and a.residue.name not in lipid_set and not _is_lipid_residue(a.residue)
               and a.element is not None and a.element.symbol != "H"]
    rest = mm.CustomExternalForce("0.5*k_rest*periodicdistance(x, y, z, x0, y0, z0)^2")
    rest.addGlobalParameter("k_rest", float(restraint_k[0]))
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    ref = pdb.positions.value_in_unit(unit.nanometer)
    for i in protein:
        rest.addParticle(i, list(ref[i]))
    system.addForce(rest)
    system.addForce(mm.MonteCarloMembraneBarostat(
        pressure_bar * unit.bar, 0 * unit.bar * unit.nanometer, temperature_k * unit.kelvin,
        mm.MonteCarloMembraneBarostat.XYIsotropic, mm.MonteCarloMembraneBarostat.ZFree, 100))
    integ = mm.LangevinMiddleIntegrator(temperature_k * unit.kelvin, 1 / unit.picosecond, timestep_fs * unit.femtosecond)
    sim = _simulation(app, mm, pdb.topology, system, integ)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin)
    os.makedirs(out_dir, exist_ok=True)
    dcd = os.path.join(out_dir, "membrane_equilibration.dcd")
    # a run shorter than one report interval would leave an empty trajectory (the tool then cannot load it)
    total_steps = sum(int(ps * 1000 / timestep_fs) for ps in stages_ps)
    every = max(1, min(int(report_every_ps * 1000 / timestep_fs), total_steps))
    sim.reporters.append(app.DCDReporter(dcd, every))
    rows = []
    lipid = lipid_label or next((r.name for r in pdb.topology.residues() if _is_lipid_residue(r)), "POPC")
    for stage, (ps, k) in enumerate(zip(stages_ps, restraint_k), start=1):
        sim.context.setParameter("k_rest", float(k))
        n = int(ps * 1000 / timestep_fs)
        done = 0
        while done < n:
            chunk = min(every, n - done)
            sim.step(chunk)
            done += chunk
            st = sim.context.getState(getPositions=True, getEnergy=True)
            box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
            dims = (float(box[0][0]), float(box[1][1]), float(box[2][2]))
            pos = [tuple(p) for p in st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)]
            m = membrane_metrics(pdb.topology, pos, dims, lipid=lipid, lipid_names=lipid_names)
            rows.append({"stage": stage, "restraint_k": k, "time_ps": round(sum(stages_ps[:stage - 1]) + done * timestep_fs / 1000, 2),
                         "box_x_nm": round(dims[0], 3), "box_z_nm": round(dims[2], 3),
                         "area_per_lipid_nm2": m.get("area_per_lipid_nm2"), "thickness_PP_nm": m.get("thickness_PP_nm"),
                         "potential_kj": round(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), 1)})
            if callable(progress):
                progress(f"stage {stage}/{len(stages_ps)}: {rows[-1]['time_ps']} ps, APL {rows[-1]['area_per_lipid_nm2']}")
    csv = os.path.join(out_dir, "membrane_equilibration.csv")
    with open(csv, "w") as fh:
        fh.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r.values()) + "\n")
    final = os.path.join(out_dir, "membrane_equilibrated.pdb")
    st = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(final, "w") as fh:
        app.PDBFile.writeFile(pdb.topology, st.getPositions(), fh, keepIds=True)
    last = rows[-1]
    ref = REFERENCE.get(lipid.upper(), (None, None))
    return {"ok": True, **simulation_platform(sim), "trajectory": dcd, "csv": csv, "final_pdb": final,
            "frames": len(rows),
            "final_area_per_lipid_nm2": last["area_per_lipid_nm2"], "final_thickness_PP_nm": last["thickness_PP_nm"],
            "reference_area_per_lipid_nm2": ref[0], "reference_thickness_PP_nm": ref[1],
            "total_ps": sum(stages_ps)}


def main(argv=None):
    """CLI for background jobs: python -m leonardo.membrane_runtime args.json (build|equilibrate)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    with open(argv[0]) as fh:
        args = json.load(fh)
    action = args.pop("action")
    out_json = args.pop("result_json")

    def progress(msg):
        print("PROGRESS " + str(msg), flush=True)

    try:
        if action == "build":
            res = build_membrane_system(progress=progress, **args)
        else:
            res = equilibrate_membrane(progress=progress, **args)
    except Exception as exc:
        import traceback

        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
