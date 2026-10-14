"""Materials and hybrid bio-inorganic systems: metal surfaces/nanoparticles, graphene, protein-on-surface
systems for MD (e.g. a biosensor: an enzyme on Au(111)) and QM clusters cut from them for MOPAC/Psi4.

Force field for the inorganic part:
* fcc metals: INTERFACE-FF 12-6 Lennard-Jones (Heinz, Vaia, Farmer, Naik, J. Phys. Chem. C 2008, 112,
  17281), fitted to surface and interface energies, compatible with CHARMM/Amber/OPLS combination
  rules; metal atoms carry no charge (no image-charge polarization in this model).
* graphene/graphite carbon: Steele LJ (sigma 3.40 A, epsilon/k 28 K), uncharged.
* silica: hydroxylated alpha-quartz (001) (geminal Q2 silanols, 9.6 OH/nm2, the fully hydroxylated,
  neutral surface) with ClayFF (Cygan, Liang, Kalinichev, J. Phys. Chem. B 2004, 108, 1255): point
  charges + 12-6 LJ, charge-neutral by construction (every Si has four O; each O is bridging or a silanol).
  Deprotonated silanols (SiO-, present at neutral pH on real silica) are not modelled.
* titania: rutile TiO2 (110) geometry (stoichiometric, bridging-O rows) for QM clusters; for MM a
  parameter set must be supplied (``oxide_params``) - none is bundled.
The inorganic atoms are frozen (mass 0) in MD: a rigid substrate, the usual choice for adsorption.
"""

import json
import math
import os
import time

import numpy as np

# symbol: (sigma nm, epsilon kJ/mol, lattice constant A)  -- INTERFACE-FF (Heinz 2008), r0 = 2^(1/6) sigma
METAL_LJ = {
    "Au": (0.26290, 22.1334, 4.080), "Ag": (0.26326, 19.0790, 4.090), "Al": (0.26059, 16.8197, 4.050),
    "Cu": (0.23305, 19.7485, 3.615), "Ni": (0.22736, 23.6396, 3.520), "Pb": (0.31760, 12.2591, 4.950),
    "Pd": (0.25114, 25.7316, 3.890), "Pt": (0.25346, 32.6352, 3.920),
}
CARBON_LJ = (0.34000, 0.2328)          # Steele graphite
# ClayFF type: (element, residue, atom name, charge e, sigma nm, epsilon kJ/mol); sigma = R0 / 2^(1/6)
CLAYFF = {"st": ("Si", "QSI", "SI", 2.1, 0.33020, 7.7007e-06), "ob": ("O", "QOB", "OB", -1.05, 0.31655, 0.65019),
          "oh": ("O", "QOH", "OH", -0.95, 0.31655, 0.65019), "ho": ("H", "QOH", "HO", 0.425, 0.1, 0.0)}
OXIDES = {"SiO2": "001", "TiO2": "110"}          # material: facet built
MATERIALS = sorted(METAL_LJ) + ["graphene"] + list(OXIDES)
FACETS = ("111", "100", "110", "001")
OXIDE_RESNAMES = {"QSI", "QOB", "QOH", "MTI", "MO"}


def _resname(symbol):
    return "GRC" if symbol == "graphene" else ("M" + symbol.upper())[:3]


def _quartz_001(size_nm, layers):
    """Hydroxylated alpha-quartz (001) slab: `layers` c repeats, both faces with geminal silanols."""
    from ase.build import make_supercell
    from ase.spacegroup import crystal

    q = crystal(["Si", "O"], basis=[(0.4697, 0.0, 2 / 3), (0.4135, 0.2669, 0.1191 + 2 / 3)], spacegroup=154,
                cellpar=[4.916, 4.916, 5.405, 90, 90, 120])
    ortho = make_supercell(q, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])          # 4.916 x 8.515 x 5.405 A
    ortho.wrap()
    nx = max(1, int(math.ceil(size_nm[0] * 10 / ortho.cell[0, 0])))
    ny = max(1, int(math.ceil(size_nm[1] * 10 / ortho.cell[1, 1])))
    nz = max(1, int(layers)) + 1
    bulk = ortho.repeat((nx, ny, nz))
    lx, ly = float(bulk.cell[0, 0]), float(bulk.cell[1, 1])
    X = bulk.get_positions()
    sym = np.array(bulk.get_chemical_symbols())
    # Si layers are 1.80 A apart (z = 0, 1.80, 3.60 ...): cut half-way between layers
    lo, hi = 0.9, 0.9 + 5.405 * (nz - 1)
    si = np.where((sym == "Si") & (X[:, 2] > lo) & (X[:, 2] < hi))[0]
    ox = np.where(sym == "O")[0]

    def mic(d):
        d = d.copy()
        d[..., 0] -= lx * np.round(d[..., 0] / lx)
        d[..., 1] -= ly * np.round(d[..., 1] / ly)
        return d

    d = np.linalg.norm(mic(X[ox][:, None, :] - X[si][None, :, :]), axis=2)       # (nO, nSi)
    nbr = d < 1.75
    keep_o = ox[nbr.any(axis=1)]
    n_si_of_o = nbr[nbr.any(axis=1)].sum(axis=1)
    if not np.all(nbr.sum(axis=0) == 4):
        raise RuntimeError("quartz slab construction failed (Si without four O)")
    mid = 0.5 * (lo + hi)
    symbols, types, pos, bonds = [], [], [], []
    for i in si:
        symbols.append("Si"), types.append("st"), pos.append(X[i])
    for o, n in zip(keep_o, n_si_of_o):
        if n == 2:
            symbols.append("O"), types.append("ob"), pos.append(X[o])
            continue
        s_idx = si[np.argmin(np.linalg.norm(mic(X[si] - X[o]), axis=1))]
        u = mic(X[s_idx] - X[o])
        u /= np.linalg.norm(u)
        normal = np.array([0.0, 0.0, 1.0 if X[o][2] > mid else -1.0])
        perp = normal - (normal @ u) * u
        perp /= np.linalg.norm(perp)
        theta = math.radians(109.47)
        h = X[o] + 1.0 * (math.cos(theta) * u + math.sin(theta) * perp)
        symbols.append("O"), types.append("oh"), pos.append(X[o])
        symbols.append("H"), types.append("ho"), pos.append(h)
        bonds.append((len(pos) - 2, len(pos) - 1))
    pos = np.array(pos) / 10.0
    pos[:, 0] %= lx / 10
    pos[:, 1] %= ly / 10
    pos[:, 2] -= pos[:, 2].max()                    # outermost silanol H at z = 0
    n_oh = types.count("oh")
    return {"symbols": symbols, "types": types, "bonds": bonds, "positions_nm": pos,
            "cell_nm": [lx / 10, ly / 10], "kind": "slab", "material": "SiO2", "facet": "001",
            "layers": int(layers), "n_atoms": len(symbols), "thickness_nm": round(float(-pos[:, 2].min()), 3),
            "silanols_per_nm2": round(n_oh / 2 / (lx * ly / 100), 2),
            "charges": [CLAYFF[t][3] for t in types], "forcefield": "ClayFF (Cygan 2004)"}


def _rutile_110(size_nm, layers):
    """Stoichiometric rutile TiO2 (110) slab (bridging-O rows on top), orthogonal cell."""
    from ase.build import surface
    from ase.spacegroup import crystal

    r = crystal(["Ti", "O"], basis=[(0, 0, 0), (0.3053, 0.3053, 0)], spacegroup=136,
                cellpar=[4.594, 4.594, 2.959, 90, 90, 90])
    slab = surface(r, (1, 1, 0), max(1, int(layers)), vacuum=None, periodic=False)
    cell = slab.cell.lengths()
    nx = max(1, int(math.ceil(size_nm[0] * 10 / cell[0])))
    ny = max(1, int(math.ceil(size_nm[1] * 10 / cell[1])))
    slab = slab.repeat((nx, ny, 1))
    pos = slab.get_positions() / 10.0
    pos[:, 2] -= pos[:, 2].max()
    lx, ly = float(slab.cell[0, 0]) / 10, float(slab.cell[1, 1]) / 10
    pos[:, 0] %= lx
    pos[:, 1] %= ly
    return {"symbols": slab.get_chemical_symbols(), "positions_nm": pos, "cell_nm": [lx, ly], "kind": "slab",
            "material": "TiO2", "facet": "110", "layers": int(layers), "n_atoms": len(slab),
            "thickness_nm": round(float(-pos[:, 2].min()), 3), "polymorph": "rutile"}


# --------------------------------------------------------------------------- builders


def build_material(material="Au", kind="slab", facet="111", size_nm=(4.0, 4.0), layers=4, shells=3):
    """
    ASE model of a material: an orthogonal periodic slab (top surface at z=0) or a nanoparticle
    (icosahedral, centred at the origin). Returns {"symbols", "positions_nm" (n,3), "cell_nm", "kind", ...}.
    """
    from ase import build as ab
    from ase.cluster import Icosahedron

    if material not in MATERIALS:
        raise ValueError(f"material must be one of {', '.join(MATERIALS)}")
    if material in OXIDES:
        if kind != "slab":
            raise ValueError("oxides are built as surface slabs")
        if facet not in (None, "", OXIDES[material], "111"):      # "111" = the generic default
            raise ValueError(f"{material} is built as the {OXIDES[material]} surface")
        return (_quartz_001 if material == "SiO2" else _rutile_110)(size_nm, layers)
    if kind not in ("slab", "nanoparticle"):
        raise ValueError("kind must be 'slab' or 'nanoparticle'")
    if kind == "slab" and material != "graphene" and str(facet) not in ("111", "100", "110"):
        raise ValueError(f"{material} slabs are built as the 111, 100 or 110 facet")
    if kind == "nanoparticle":
        if material == "graphene":
            raise ValueError("nanoparticles are for metals")
        atoms = Icosahedron(material, noshells=int(shells), latticeconstant=METAL_LJ[material][2])
        pos = atoms.get_positions() / 10.0
        pos -= pos.mean(axis=0)
        return {"symbols": atoms.get_chemical_symbols(), "positions_nm": pos, "cell_nm": None, "kind": kind,
                "material": material, "n_atoms": len(atoms),
                "diameter_nm": round(float(2 * np.linalg.norm(pos, axis=1).max()), 3)}
    if material == "graphene":
        a = 2.46
        nx = max(1, int(round(size_nm[0] * 10 / a)))
        ny = max(1, int(round(size_nm[1] * 10 / (a * math.sqrt(3)))))
        atoms = ab.graphene(a=a, size=(nx, 2 * ny, 1), vacuum=None)
        atoms = _orthogonalize_graphene(atoms)
        layers = 1
    else:
        a = METAL_LJ[material][2]
        fn = {"111": ab.fcc111, "100": ab.fcc100, "110": ab.fcc110}[str(facet)]
        # repeat lengths of the orthogonal cell (fcc111 needs an even repeat along y)
        cell_1 = fn(material, size=(1, 2, int(layers)), a=a, orthogonal=True).cell.lengths()
        cell_1 = (cell_1[0], cell_1[1] / 2.0)
        nx = max(2, int(math.ceil(size_nm[0] * 10 / cell_1[0])))
        ny = max(2, int(math.ceil(size_nm[1] * 10 / cell_1[1])))
        if str(facet) == "111" and ny % 2:
            ny += 1                                  # orthogonal fcc111 needs an even repeat along y
        atoms = fn(material, size=(nx, ny, int(layers)), a=a, orthogonal=True, vacuum=None)
    pos = atoms.get_positions() / 10.0
    pos[:, 2] -= pos[:, 2].max()                     # top layer at z = 0
    cell = atoms.cell.lengths() / 10.0
    return {"symbols": atoms.get_chemical_symbols() if material != "graphene" else ["C"] * len(atoms),
            "positions_nm": pos, "cell_nm": [float(cell[0]), float(cell[1])], "kind": "slab",
            "material": material, "facet": facet if material != "graphene" else "0001", "layers": int(layers),
            "n_atoms": len(atoms), "thickness_nm": round(float(-pos[:, 2].min()), 3)}


def _orthogonalize_graphene(atoms):
    """Rectangular graphene sheet: wrap the hexagonal cell into an orthogonal one (same density)."""
    import numpy as np
    from ase import Atoms

    cell = atoms.cell.array
    a1, a2 = cell[0], cell[1]
    lx = float(np.linalg.norm(a1))
    ly = float(abs(a2[1]))
    pos = atoms.get_positions().copy()
    pos[:, 0] = pos[:, 0] % lx
    pos[:, 1] = pos[:, 1] % ly
    keep, seen = [], set()
    for i, p in enumerate(pos):
        key = (round(p[0], 2) % round(lx, 2), round(p[1], 2) % round(ly, 2))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    out = Atoms("C" * len(keep), positions=pos[keep], cell=[lx, ly, 10.0], pbc=[True, True, False])
    return out


def _parse_oxide_params(params):
    """{"Ti": [charge, sigma_nm, epsilon_kJ], "O": [...]} (or dicts with charge/sigma_nm/epsilon_kj)."""
    out = {}
    for el, v in dict(params or {}).items():
        if isinstance(v, dict):
            v = (v.get("charge"), v.get("sigma_nm"), v.get("epsilon_kj"))
        q, sig, eps = (float(x) for x in v)
        if sig <= 0 or eps < 0:
            raise ValueError(f"bad Lennard-Jones parameters for {el}")
        out[str(el).capitalize()] = (q, sig, eps)
    return out


def _oxide_forcefield_xml(material, model, params=None):
    from openmm.app import Element

    if material == "SiO2":
        rows = {t: CLAYFF[t] for t in set(model["types"])}
        lines = ["<ForceField>", " <AtomTypes>"]
        for t, (el, _rn, _an, _q, _s, _e) in sorted(rows.items()):
            lines.append(f'  <Type name="clay-{t}" class="clay{t}" element="{el}" '
                         f'mass="{Element.getBySymbol(el).mass._value:.4f}"/>')
        lines += [" </AtomTypes>", " <Residues>",
                  '  <Residue name="QSI"><Atom name="SI" type="clay-st" charge="2.1"/></Residue>',
                  '  <Residue name="QOB"><Atom name="OB" type="clay-ob" charge="-1.05"/></Residue>',
                  '  <Residue name="QOH"><Atom name="OH" type="clay-oh" charge="-0.95"/>'
                  '<Atom name="HO" type="clay-ho" charge="0.425"/><Bond atomName1="OH" atomName2="HO"/></Residue>',
                  " </Residues>", " <HarmonicBondForce>",
                  '  <Bond class1="clayoh" class2="clayho" length="0.1" k="463700.0"/>',
                  " </HarmonicBondForce>",
                  ' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">',
                  '  <UseAttributeFromResidue name="charge"/>']
        for t, (_el, _rn, _an, _q, sig, eps) in sorted(rows.items()):
            lines.append(f'  <Atom type="clay-{t}" sigma="{sig:.5f}" epsilon="{eps:.6g}"/>')
        lines += [" </NonbondedForce>", "</ForceField>"]
        return "\n".join(lines) + "\n"
    prm = _parse_oxide_params(params)
    elems = sorted(set(model["symbols"]))
    missing = [e for e in elems if e not in prm]
    if missing:
        raise ValueError(f"no bundled MM parameters for {material}: pass oxide_params with charge, sigma_nm and "
                         f"epsilon_kj for {', '.join(elems)} from a published force field (e.g. Matsui-Akaogi-derived "
                         "or Brandt-Lyubartsev TiO2 sets); the QM route (cut_qm_cluster + MOPAC) needs none")
    counts = {e: model["symbols"].count(e) for e in elems}
    total = sum(prm[e][0] * n for e, n in counts.items())
    if abs(total) > 1e-3:
        raise ValueError(f"these oxide charges give a {total:+.3f} e slab; they must make it neutral")
    lines = ["<ForceField>", " <AtomTypes>"]
    for e in elems:
        lines.append(f'  <Type name="ox-{e}" class="ox{e}" element="{e}" mass="{Element.getBySymbol(e).mass._value:.4f}"/>')
    lines += [" </AtomTypes>", " <Residues>"]
    for e in elems:
        lines.append(f'  <Residue name="{_resname(e)}"><Atom name="{e.upper()}" type="ox-{e}" '
                     f'charge="{prm[e][0]:.6f}"/></Residue>')
    lines += [" </Residues>", ' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">',
              '  <UseAttributeFromResidue name="charge"/>']
    for e in elems:
        lines.append(f'  <Atom type="ox-{e}" sigma="{prm[e][1]:.5f}" epsilon="{prm[e][2]:.6g}"/>')
    lines += [" </NonbondedForce>", "</ForceField>"]
    return "\n".join(lines) + "\n"


def material_forcefield_xml(material, symbols, model=None, params=None):
    """ForceField XML with one single-atom residue template per element of the material (oxides: ClayFF
    silica with silanol residues, or user parameters)."""
    from openmm.app import Element

    if material in OXIDES:
        return _oxide_forcefield_xml(material, model, params)
    lines = ["<ForceField>", " <AtomTypes>"]
    elems = sorted(set(symbols))
    for e in elems:
        lines.append(f'  <Type name="mat-{e}" class="mat{e}" element="{e}" mass="{Element.getBySymbol(e).mass._value:.4f}"/>')
    lines += [" </AtomTypes>", " <Residues>"]
    for e in elems:
        rn = _resname("graphene" if material == "graphene" else e)
        lines.append(f'  <Residue name="{rn}"><Atom name="{e.upper()}" type="mat-{e}" charge="0.0"/></Residue>')
    lines += [" </Residues>", ' <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">',
              '  <UseAttributeFromResidue name="charge"/>']
    for e in elems:
        sigma, eps = CARBON_LJ if material == "graphene" else METAL_LJ[e][:2]
        lines.append(f'  <Atom type="mat-{e}" sigma="{sigma:.5f}" epsilon="{eps:.5f}"/>')
    lines += [" </NonbondedForce>", "</ForceField>"]
    return "\n".join(lines) + "\n"


def material_topology(model):
    """OpenMM Topology/positions for a material (chain 'Z', one residue per atom)."""
    import openmm.app as app
    from openmm import Vec3, unit

    top = app.Topology()
    chain = top.addChain("Z")
    pos = []
    if model.get("types"):                          # ClayFF silica: silanol O-H pairs are one residue
        partner = {o: h for o, h in model["bonds"]}
        skip = set(partner.values())
        n = 0
        for i, t in enumerate(model["types"]):
            if i in skip:
                continue
            n += 1
            el, rn, an = CLAYFF[t][:3]
            res = top.addResidue(rn, chain, str(n))
            o = top.addAtom(an, app.Element.getBySymbol(el), res)
            pos.append(Vec3(*[float(x) for x in model["positions_nm"][i]]))
            if i in partner:
                h = top.addAtom("HO", app.Element.getBySymbol("H"), res)
                top.addBond(o, h)
                pos.append(Vec3(*[float(x) for x in model["positions_nm"][partner[i]]]))
        return top, unit.Quantity(pos, unit.nanometer)
    for i, (e, p) in enumerate(zip(model["symbols"], model["positions_nm"])):
        rn = _resname("graphene" if model["material"] == "graphene" else e)
        res = top.addResidue(rn, chain, str(i + 1))
        top.addAtom(e.upper(), app.Element.getBySymbol(e), res)
        pos.append(Vec3(*[float(x) for x in p]))
    return top, unit.Quantity(pos, unit.nanometer)


def write_material_pdb(model, path):
    import openmm.app as app

    top, pos = material_topology(model)
    if model.get("cell_nm"):
        from openmm import unit

        top.setUnitCellDimensions(unit.Quantity((model["cell_nm"][0], model["cell_nm"][1], 5.0), unit.nanometer))
    with open(path, "w") as fh:
        app.PDBFile.writeFile(top, pos, fh, keepIds=True)
    return path


# --------------------------------------------------------------------------- protein on a surface


def _rotation_to(v, target):
    v = np.asarray(v, float) / (np.linalg.norm(v) or 1.0)
    t = np.asarray(target, float) / (np.linalg.norm(target) or 1.0)
    c = float(np.dot(v, t))
    ax = np.cross(v, t)
    s = float(np.linalg.norm(ax))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def _select_atoms(topology, spec):
    """Atoms of residue specs like 'A:57', '57', 'A:57-60' or residue names 'HIS' (comma/space separated)."""
    out = []
    tokens = [t for t in str(spec or "").replace(",", " ").split() if t]
    for tok in tokens:
        chain, _, rng = tok.rpartition(":")
        for res in topology.residues():
            if chain and res.chain.id != chain:
                continue
            if rng.replace("-", "").lstrip("-").isdigit() or ("-" in rng and rng.split("-")[0].isdigit()):
                if "-" in rng and not rng.startswith("-"):
                    lo, hi = (int(x) for x in rng.split("-", 1))
                    ok = res.id.lstrip("-").isdigit() and lo <= int(res.id) <= hi
                else:
                    ok = res.id == rng
            else:
                ok = str(res.name).upper() == rng.upper()
            if ok:
                out.extend(a for a in res.atoms())
    return out


def build_protein_surface_system(pdb_path, material="Au", facet="111", layers=4, family="amber14", ph=7.0,
                                 gap_nm=0.3, toward_surface="", away_from_surface="", lateral_padding_nm=1.0,
                                 water_above_nm=1.5, ionic_strength_molar=0.15, anchor_cys="", out_dir=None,
                                 minimize_steps=500, keep_waters=False, remove_resn=(), progress=None,
                                 oxide_params=None):
    """
    Protein (component library) + frozen material slab + water/ions in a periodic box, minimized and
    analysed: contacts with the surface, protein-surface Lennard-Jones adsorption energy, height of
    the chosen site. ``toward_surface``/``away_from_surface`` (residue specs) orient the protein, e.g.
    keep an enzyme's active site facing the solution. ``anchor_cys`` adds Cys SG - surface restraints
    (chemisorbed thiolate, 0.24-0.26 nm to the nearest metal atoms) as a non-bonded anchoring model.
    SiO2 uses ClayFF hydroxylated quartz (001); TiO2 rutile (110) needs ``oxide_params``.
    """
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    from . import component_library_runtime as cl

    def emit(msg):
        if callable(progress):
            progress(msg)

    out_dir = out_dir or os.path.join(cl._cache_dir(), "materials", time.strftime("surf_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    if family != "amber14":
        # the substrate LJ sits in the NonbondedForce (Amber style); CHARMM36 in OpenMM keeps LJ in a separate
        # LennardJonesForce with 1.0/1.0 1-4 scaling, so these substrate files do not combine with it
        raise ValueError("protein-surface systems are built with amber14 (the substrate force fields here "
                         "use Amber-style nonbonded terms)")
    if material not in OXIDES and str(facet) not in ("111", "100", "110"):
        raise ValueError(f"{material} slabs are built as the 111, 100 or 110 facet")
    if material in OXIDES:
        facet = OXIDES[material]
        if anchor_cys:
            raise ValueError("anchor_cys models a thiolate-metal bond: not for oxide surfaces")
        if material == "TiO2":          # fail before the (slow) protein preparation
            _oxide_forcefield_xml(material, {"symbols": ["Ti", "O", "O"]}, oxide_params)
    report = {"material": material, "facet": facet, "ok": False, "out_dir": out_dir, "warnings": []}
    emit("assigning component parameters")
    prep = cl.prepare_system_for_solvation(pdb_path, family=family, ph=ph, keep_waters=keep_waters,
                                           remove_resn=remove_resn)
    if not prep["ok"]:
        report["error"] = prep["error"]
        return report
    modeller, ff = prep["modeller"], prep["forcefield"]
    # ---- orientation
    P = np.array([tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions])
    com = P.mean(axis=0)
    R = np.eye(3)
    if toward_surface or away_from_surface:
        sel = _select_atoms(modeller.topology, toward_surface or away_from_surface)
        if not sel:
            raise ValueError(f"no residues match {toward_surface or away_from_surface!r}")
        site = P[[a.index for a in sel]].mean(axis=0)
        R = _rotation_to(site - com, (0, 0, -1) if toward_surface else (0, 0, 1))
    P = (P - com) @ R.T
    heavy = [a.index for a in modeller.topology.atoms() if a.element is not None and a.element.symbol != "H"]
    extent = P[heavy].max(axis=0) - P[heavy].min(axis=0)
    # ---- slab
    size = (float(extent[0] + 2 * lateral_padding_nm), float(extent[1] + 2 * lateral_padding_nm))
    emit(f"building {material}({facet}) slab")
    slab = build_material(material, "slab", facet, size_nm=size, layers=layers)
    lx, ly = slab["cell_nm"]
    # protein: centred in xy over the slab, lowest heavy atom gap_nm above the top layer
    P[:, 0] += lx / 2.0 - P[heavy, 0].mean()
    P[:, 1] += ly / 2.0 - P[heavy, 1].mean()
    P[:, 2] += gap_nm - P[heavy, 2].min()
    modeller.positions = unit.Quantity([mm.Vec3(*p) for p in P], unit.nanometer)
    mat_xml = os.path.join(out_dir, f"{material}_{'clayff' if material == 'SiO2' else 'interface_ff'}.xml")
    with open(mat_xml, "w") as fh:
        fh.write(material_forcefield_xml(material, slab["symbols"], model=slab, params=oxide_params))
    ff.loadFile(mat_xml)
    mtop, mpos = material_topology(slab)
    n_before = modeller.topology.getNumAtoms()
    modeller.add(mtop, mpos)
    mat_range = range(n_before, n_before + mtop.getNumAtoms())    # addSolvent keeps existing atoms first
    zmin = float(slab["positions_nm"][:, 2].min())
    height = float(P[heavy, 2].max()) + water_above_nm
    lz = height - zmin + 0.3
    box = (lx, ly, lz)
    # shift everything so the slab bottom sits at z = 0.15 nm in the box
    shift = 0.15 - zmin
    X = np.array([tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]) + np.array([0, 0, shift])
    modeller.positions = unit.Quantity([mm.Vec3(*p) for p in X], unit.nanometer)
    surface_top = shift
    emit("adding water and ions")
    modeller.addSolvent(ff, boxSize=mm.Vec3(*box) * unit.nanometer,
                        ionicStrength=float(ionic_strength_molar) * unit.molar)
    emit("creating the system")
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds)
    mat_idx = list(mat_range)
    for i in mat_idx:
        system.setParticleMass(i, 0.0)              # rigid substrate
    mat_set = set(mat_idx)
    for k in reversed(range(system.getNumConstraints())):      # silanol O-H: both frozen, no constraint
        i, j, _d = system.getConstraintParameters(k)
        if i in mat_set or j in mat_set:
            system.removeConstraint(k)
    pos_nm = [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]
    if material == "graphene":
        _exclude_graphene_neighbours(system, pos_nm, mat_idx, (lx, ly))
    mat_resn = {a.residue.name for a in modeller.topology.atoms() if a.index in mat_set}
    report["metal_restraints"] = cl.add_coordination_restraints(system, modeller.topology, pos_nm,
                                                                skip_resn=mat_resn)
    anchors = []
    if anchor_cys:
        anchors = _add_thiol_anchors(system, modeller.topology, pos_nm, anchor_cys, mat_idx)
    report["anchors"] = anchors
    emit("minimizing")
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    from .membrane_runtime import _simulation

    sim = _simulation(app, mm, modeller.topology, system, integ)
    sim.context.setPositions(modeller.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    sim.minimizeEnergy(maxIterations=int(minimize_steps))
    state = sim.context.getState(getEnergy=True, getPositions=True)
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    pos_nm = [tuple(p) for p in state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)]
    report.update(energy_start_kj=round(e0, 1), energy_minimized_kj=round(e1, 1),
                  n_atoms=system.getNumParticles(), box_nm=[round(x, 3) for x in box],
                  slab={k: slab[k] for k in ("material", "facet", "layers", "n_atoms", "thickness_nm")},
                  orientation=({"toward_surface": toward_surface} if toward_surface else
                               {"away_from_surface": away_from_surface} if away_from_surface else "as loaded"))
    report["surface_analysis"] = surface_contacts(modeller.topology, pos_nm, system, mat_idx, surface_top,
                                                  site=toward_surface or away_from_surface)
    charge = 0.0
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            charge = sum(force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                         for i in range(system.getNumParticles()))
    report["total_charge"] = round(charge, 3)
    pdb_out = os.path.join(out_dir, "surface_system_minimized.pdb")
    with open(pdb_out, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, state.getPositions(), fh, keepIds=True)
    xml_out = os.path.join(out_dir, "surface_system.xml")
    with open(xml_out, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(system))
    if material == "SiO2":
        notes = ["ClayFF (Cygan 2004) hydroxylated alpha-quartz (001), geminal silanols "
                 f"{slab.get('silanols_per_nm2')} OH/nm2, neutral: real silica at pH 7 carries some SiO- "
                 "(about -0.1 to -0.5 e/nm2), not modelled", "substrate frozen (mass 0), silanol H included"]
        report["slab"]["silanols_per_nm2"] = slab.get("silanols_per_nm2")
    elif material == "TiO2":
        notes = ["rutile (110) with user-supplied charges/LJ (oxide_params)", "substrate frozen (mass 0)"]
    else:
        notes = ["INTERFACE-FF 12-6 LJ for the metal (no image charges / polarization)", "substrate frozen (mass 0)"]
    report.update(pdb=pdb_out, system_xml=xml_out, material_xml=mat_xml,
                  model_notes=notes + (["Cys-metal anchoring by restraints (not a bonded Au-S model)"]
                                       if anchors else []))
    if abs(charge - round(charge)) > 0.01:
        report["warnings"].append(f"non-integer total charge {charge:.3f}")
    if not math.isfinite(e1) or e1 > e0 + max(50.0, 0.02 * abs(e0)):
        report["warnings"].append("minimization was not stable")
    report["ok"] = not report["warnings"]
    with open(os.path.join(out_dir, "surface_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


def _exclude_graphene_neighbours(system, pos_nm, mat_idx, cell_xy, cutoff_nm=0.3):
    """
    The Steele carbon LJ describes the sheet-adsorbate interaction; the sheet itself is covalent and has no
    bonded terms here, so its C-C neighbours (0.142/0.246/0.284 nm, far inside sigma = 0.34 nm) would add
    ~1e7 kJ/mol of LJ repulsion to every reported energy. They are frozen: exclude them (minimum image in xy).
    """
    import openmm as mm

    P = np.array([pos_nm[i] for i in mat_idx], dtype=float)
    cell = np.array([cell_xy[0], cell_xy[1]], dtype=float)
    pairs = []
    for a in range(len(P) - 1):
        d = P[a + 1:] - P[a]
        d[:, :2] -= cell * np.round(d[:, :2] / cell)
        for b in np.nonzero(np.einsum("ij,ij->i", d, d) < cutoff_nm ** 2)[0]:
            pairs.append((mat_idx[a], mat_idx[a + 1 + int(b)]))
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            for i, j in pairs:
                f.addException(i, j, 0.0, 0.1, 0.0, replace=True)
        elif isinstance(f, mm.CustomNonbondedForce):
            for i, j in pairs:
                f.addExclusion(i, j)
    return len(pairs)


def _add_thiol_anchors(system, topology, pos_nm, spec, mat_idx, r_lo=0.24, r_hi=0.26, k=50000.0):
    """Flat-bottom restraints from each chosen Cys SG to its three nearest surface atoms (hollow site)."""
    import openmm as mm

    sg = [a for a in _select_atoms(topology, spec) if a.name == "SG"]
    if not sg:
        raise ValueError(f"no Cys SG in {spec!r} to anchor")
    M = np.array([pos_nm[i] for i in mat_idx])
    force = mm.CustomBondForce("0.5*leo_anchor_k*(max(0, r-leo_anchor_rhi)^2 + max(0, leo_anchor_rlo-r)^2)")
    force.addGlobalParameter("leo_anchor_k", k)
    force.addGlobalParameter("leo_anchor_rlo", r_lo)
    force.addGlobalParameter("leo_anchor_rhi", r_hi)
    out = []
    pairs = []
    for a in sg:
        d = np.linalg.norm(M - np.array(pos_nm[a.index]), axis=1)
        for j in np.argsort(d)[:3]:
            force.addBond(a.index, mat_idx[int(j)], [])
            pairs.append((a.index, mat_idx[int(j)]))
        out.append(f"{a.residue.name}{a.residue.id}{a.residue.chain.id}:SG, {d.min() * 10:.1f} A from the surface")
    # a chemisorbed S-Au contact is a bond: no Lennard-Jones repulsion between the anchored pairs
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            for i, j in pairs:
                f.addException(i, j, 0.0, 0.1, 0.0, replace=True)
        elif isinstance(f, mm.CustomNonbondedForce):
            existing = {tuple(sorted(f.getExclusionParticles(k))) for k in range(f.getNumExclusions())}
            for i, j in pairs:
                if tuple(sorted((i, j))) not in existing:
                    f.addExclusion(i, j)
    force.setName("LeonardoThiolAnchor")
    system.addForce(force)
    return out


def surface_contacts(topology, pos_nm, system, mat_idx, surface_top_z, site="", contact_nm=0.45, cutoff_nm=1.2):
    """Residues touching the surface, protein-surface LJ energy (LB mixing, as OpenMM), site height."""
    import openmm as mm
    from openmm import unit

    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    mat = set(mat_idx)
    skip = {"HOH", "WAT", "NA", "CL", "SOD", "CLA", "POT"}
    prot = [a for a in topology.atoms() if a.index not in mat and a.residue.name not in skip]
    X = np.array(pos_nm)
    M = X[list(mat_idx)]
    sig_m = np.array([nb.getParticleParameters(i)[1].value_in_unit(unit.nanometer) for i in mat_idx])
    eps_m = np.array([nb.getParticleParameters(i)[2].value_in_unit(unit.kilojoule_per_mole) for i in mat_idx])
    q_m = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in mat_idx])
    charged = bool(np.any(np.abs(q_m) > 1e-6))
    e_lj, e_coul, contacts, min_d = 0.0, 0.0, {}, 1e9
    for a in prot:
        d = np.linalg.norm(M - X[a.index], axis=1)
        near = d < cutoff_nm
        if not near.any():
            continue
        q, s, e = nb.getParticleParameters(a.index)
        s = s.value_in_unit(unit.nanometer)
        e = e.value_in_unit(unit.kilojoule_per_mole)
        sij = 0.5 * (s + sig_m[near])
        eij = np.sqrt(e * eps_m[near])
        sr6 = (sij / d[near]) ** 6
        e_lj += float(np.sum(4 * eij * (sr6 * sr6 - sr6)))
        if charged:
            e_coul += float(138.935458 * q.value_in_unit(unit.elementary_charge) * np.sum(q_m[near] / d[near]))
        dm = float(d.min())
        min_d = min(min_d, dm)
        if dm < contact_nm and a.element is not None and a.element.symbol != "H":
            key = f"{a.residue.name}{a.residue.id}{a.residue.chain.id}"
            contacts[key] = min(contacts.get(key, 9.0), round(dm * 10, 2))
    out = {"protein_surface_LJ_kj_mol": round(e_lj, 1), "min_distance_A": round(min_d * 10, 2),
           "contact_residues": dict(sorted(contacts.items(), key=lambda kv: kv[1])[:40]),
           "n_contact_residues": len(contacts)}
    if charged:      # plain Coulomb within the cutoff, no solvent screening: a trend, not a binding energy
        out["protein_surface_coulomb_unscreened_kj_mol"] = round(e_coul, 1)
    if site:
        sel = _select_atoms(topology, site)
        if sel:
            z = X[[a.index for a in sel], 2] - surface_top_z
            out["site_height_above_surface_nm"] = round(float(z.mean()), 3)
    return out


# --------------------------------------------------------------------------- QM clusters


def cut_qm_cluster(pdb_path, center_spec, radius_nm=0.6, material_resnames=None, out_xyz=None):
    """
    QM cluster around an adsorption site: the whole residues of the adsorbate/protein within
    ``radius_nm`` of the centre atoms plus the substrate atoms within ``radius_nm`` (metal clusters for
    MOPAC PM7 / Psi4 with def2 ECPs). Writes XYZ; returns atoms, composition and notes.
    """
    import openmm.app as app
    from openmm import unit

    pdb = app.PDBFile(pdb_path)
    X = np.array([tuple(p.value_in_unit(unit.nanometer)) for p in pdb.positions])
    centre = _select_atoms(pdb.topology, center_spec)
    if not centre:
        raise ValueError(f"no atoms match {center_spec!r}")
    C = X[[a.index for a in centre]]
    mat_names = {x.upper() for x in (material_resnames or [])} or \
        {_resname(e) for e in METAL_LJ} | {"GRC"} | OXIDE_RESNAMES
    keep = set()
    skip = {"HOH", "WAT", "NA", "CL", "SOD", "CLA", "POT"}
    for res in pdb.topology.residues():
        atoms = list(res.atoms())
        d = min(float(np.linalg.norm(C - X[a.index], axis=1).min()) for a in atoms)
        if d > radius_nm:
            continue
        if res.name in mat_names:
            keep.update(a.index for a in atoms)
        elif res.name not in skip:
            keep.update(a.index for a in atoms)
    atoms = [a for a in pdb.topology.atoms() if a.index in keep]
    comp = {}
    for a in atoms:
        comp[a.element.symbol] = comp.get(a.element.symbol, 0) + 1
    out_xyz = out_xyz or os.path.splitext(pdb_path)[0] + "_qm_cluster.xyz"
    with open(out_xyz, "w") as fh:
        fh.write(f"{len(atoms)}\nQM cluster around {center_spec}\n")
        for a in atoms:
            x, y, z = X[a.index] * 10
            fh.write(f"{a.element.symbol:2s} {x:12.5f} {y:12.5f} {z:12.5f}\n")
    notes = ["cut protein residues are whole residues (no link atoms): check the electron count and charge"]
    if any(r.name in {"QSI", "QOB", "QOH", "MTI", "MO"} for r in {a.residue for a in atoms}):
        notes.append("oxide cluster: cut Si-O / Ti-O bonds leave dangling bonds; saturate them with H (Si-OH) or "
                     "use a stoichiometric, charge-balanced cluster before MOPAC")
    if any(e in METAL_LJ for e in comp):
        notes.append("metal cluster: MOPAC PM7 has parameters for these metals; Psi4 needs basis sets with ECPs "
                     "(def2-SVP/TZVP) — use small clusters (tens of atoms)")
    return {"xyz": out_xyz, "n_atoms": len(atoms), "composition": comp, "notes": notes}


def main(argv=None):
    """CLI for background jobs: python -m leonardo.materials_runtime args.json."""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    with open(argv[0]) as fh:
        args = json.load(fh)
    args.pop("action", None)
    out_json = args.pop("result_json")

    def progress(msg):
        print("PROGRESS " + str(msg), flush=True)

    try:
        res = build_protein_surface_system(progress=progress, **args)
    except Exception as exc:
        import traceback

        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
