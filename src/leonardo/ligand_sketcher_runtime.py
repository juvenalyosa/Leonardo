"""Interactive 2D ligand sketcher (Maestro/ChemDraw-style) for the Ligand Editor.

Three layers:

* pure RDKit edit operations on a heavy-atom molecule with a 2D conformer
  (placement of new atoms at the best free angle, ring attach/fuse, bond order
  cycling, charges, wedge/hash stereo, descriptors) - headless-testable;
* ``SketcherCanvas``: skeletal-formula rendering plus mouse/keyboard input;
* ``LigandSketcherWidget``: tool palette, element strip, property panel,
  undo/redo, and callback attributes the host dialog wires up.
"""

import math

try:
    from pymol.Qt import QtCore, QtGui, QtWidgets
except Exception:  # pragma: no cover - fallback for non-PyMOL contexts
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    except Exception:  # pragma: no cover
        from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - RDKit is required by the editor
    Chem = None
    AllChem = None


BOND_LEN = 1.5  # RDKit depiction bond length (model units)

ELEMENT_COLORS = {
    "C": "#1F2933",
    "H": "#52606D",
    "N": "#1D4ED8",
    "O": "#DC2626",
    "S": "#B7791F",
    "P": "#C05621",
    "F": "#15803D",
    "CL": "#15803D",
    "BR": "#9A3412",
    "I": "#6D28D9",
    "B": "#B45309",
    "SI": "#475569",
}

PALETTE_ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]

PERIODIC_ROWS = [
    ["H", "He"],
    ["Li", "Be", "B", "C", "N", "O", "F", "Ne"],
    ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"],
    ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
    ["Ga", "Ge", "As", "Se", "Br", "Kr"],
    ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd"],
    ["In", "Sn", "Sb", "Te", "I", "Xe"],
    ["Cs", "Ba", "Pt", "Au", "Hg", "Tl", "Pb", "Bi"],
]

# (button label, tooltip, SMILES with one [*] attachment point)
FRAGMENT_LIBRARY = [
    ("Me", "Methyl", "[*]C"),
    ("Et", "Ethyl", "[*]CC"),
    ("iPr", "Isopropyl", "[*]C(C)C"),
    ("tBu", "tert-Butyl", "[*]C(C)(C)C"),
    ("cPr", "Cyclopropyl", "[*]C1CC1"),
    ("OH", "Hydroxyl", "[*]O"),
    ("OMe", "Methoxy", "[*]OC"),
    ("SH", "Thiol", "[*]S"),
    ("NH2", "Amino", "[*]N"),
    ("NHMe", "Methylamino", "[*]NC"),
    ("NMe2", "Dimethylamino", "[*]N(C)C"),
    ("NHAc", "Acetamido", "[*]NC(C)=O"),
    ("F", "Fluoro", "[*]F"),
    ("Cl", "Chloro", "[*]Cl"),
    ("Br", "Bromo", "[*]Br"),
    ("CF3", "Trifluoromethyl", "[*]C(F)(F)F"),
    ("CN", "Nitrile", "[*]C#N"),
    ("NO2", "Nitro", "[*][N+](=O)[O-]"),
    ("CHO", "Formyl", "[*]C=O"),
    ("COOH", "Carboxylic acid", "[*]C(=O)O"),
    ("COOMe", "Methyl ester", "[*]C(=O)OC"),
    ("CONH2", "Carboxamide", "[*]C(N)=O"),
    ("SO2Me", "Methylsulfonyl", "[*]S(C)(=O)=O"),
    ("SO2NH2", "Sulfonamide", "[*]S(N)(=O)=O"),
    ("Ph", "Phenyl", "[*]c1ccccc1"),
    ("Bn", "Benzyl", "[*]Cc1ccccc1"),
    ("Py", "4-Pyridyl", "[*]c1ccncc1"),
    ("Morph", "Morpholino", "[*]N1CCOCC1"),
    ("Pip", "Piperazinyl", "[*]N1CCNCC1"),
    ("Tz", "1H-Tetrazol-5-yl", "[*]c1nn[nH]n1"),
]

# (tool key, tooltip)
RING_TOOLS = [
    ("ring3", "Cyclopropane"),
    ("ring4", "Cyclobutane"),
    ("ring5", "Cyclopentane"),
    ("ring6", "Cyclohexane"),
    ("benzene", "Benzene"),
    ("ring7", "Cycloheptane"),
]


class SketchError(ValueError):
    """An edit that would produce an invalid molecule."""


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _bond_type(order):
    key = str(order or "single").strip().lower()
    return {
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
        "aromatic": Chem.BondType.AROMATIC,
    }.get(key, Chem.BondType.SINGLE)


def bond_order_key(bond):
    bt = bond.GetBondType()
    if bt == Chem.BondType.DOUBLE:
        return "double"
    if bt == Chem.BondType.TRIPLE:
        return "triple"
    if bt == Chem.BondType.AROMATIC:
        return "aromatic"
    return "single"


def normalize_symbol(symbol):
    raw = str(symbol or "").strip()
    if not raw:
        return "C"
    return raw[0].upper() + raw[1:].lower()


def atomic_number(symbol):
    sym = normalize_symbol(symbol)
    try:
        z = int(Chem.GetPeriodicTable().GetAtomicNumber(sym))
    except Exception:
        z = 0
    if z <= 0:
        raise SketchError(f"Unknown element: {sym}")
    return z


def conformer_is_3d(mol, tol=1e-3):
    if mol is None or int(mol.GetNumConformers()) <= 0:
        return False
    conf = mol.GetConformer()
    for idx in range(int(mol.GetNumAtoms())):
        if abs(float(conf.GetAtomPosition(idx).z)) > tol:
            return True
    return False


def conformer_is_degenerate(mol):
    if mol is None or int(mol.GetNumConformers()) <= 0:
        return True
    n = int(mol.GetNumAtoms())
    if n <= 1:
        return False
    conf = mol.GetConformer()
    xs = [float(conf.GetAtomPosition(i).x) for i in range(n)]
    ys = [float(conf.GetAtomPosition(i).y) for i in range(n)]
    return (max(xs) - min(xs) < 1e-6) and (max(ys) - min(ys) < 1e-6)


def ensure_sketch_coords(mol):
    """Return ``mol`` with a usable flat 2D conformer (recomputed if 3D or missing)."""
    if mol is None:
        return None
    if int(mol.GetNumConformers()) <= 0 or conformer_is_3d(mol) or conformer_is_degenerate(mol):
        AllChem.Compute2DCoords(mol)
    return mol


def strip_hydrogens(mol):
    """Heavy-atom copy (also drops H atoms that only define double-bond stereo)."""
    out = Chem.Mol(mol)
    try:
        params = Chem.RemoveHsParameters()
        params.removeDefiningBondStereo = True
        return Chem.RemoveHs(out, params)
    except Exception:
        try:
            return Chem.RemoveHs(out)
        except Exception:
            return out


def atom_xy(mol, idx):
    pos = mol.GetConformer().GetAtomPosition(int(idx))
    return float(pos.x), float(pos.y)


def set_atom_xy(mol, idx, x, y):
    conf = mol.GetConformer()
    try:
        from rdkit.Geometry import Point3D

        conf.SetAtomPosition(int(idx), Point3D(float(x), float(y), 0.0))
    except Exception:  # pragma: no cover - older RDKit
        pos = conf.GetAtomPosition(int(idx))
        pos.x, pos.y, pos.z = float(x), float(y), 0.0
        conf.SetAtomPosition(int(idx), pos)


def all_positions(mol):
    if mol is None or int(mol.GetNumConformers()) <= 0:
        return {}
    return {i: atom_xy(mol, i) for i in range(int(mol.GetNumAtoms()))}


def _crowding(point, positions, exclude=()):
    best = None
    skip = set(int(i) for i in exclude)
    for idx, (x, y) in positions.items():
        if int(idx) in skip:
            continue
        d = math.hypot(point[0] - x, point[1] - y)
        if best is None or d < best:
            best = d
    return 1e9 if best is None else best


def _crowding_penalty(point, positions, exclude=()):
    skip = set(int(i) for i in exclude)
    total = 0.0
    for idx, (x, y) in positions.items():
        if int(idx) in skip:
            continue
        d2 = (point[0] - x) ** 2 + (point[1] - y) ** 2
        total += 1.0 / max(d2, 1e-3)
    return total


def free_direction(mol, idx, order="single"):
    """Unit vector pointing to the best free position around atom ``idx``."""
    idx = int(idx)
    positions = all_positions(mol)
    x0, y0 = positions[idx]
    atom = mol.GetAtomWithIdx(idx)
    nbrs = [int(nb.GetIdx()) for nb in atom.GetNeighbors()]
    if not nbrs:
        return math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    angles = []
    for j in nbrs:
        xj, yj = positions[j]
        angles.append(math.atan2(yj - y0, xj - x0))
    candidates = []
    if len(nbrs) == 1:
        a = angles[0]
        existing = mol.GetBondBetweenAtoms(idx, nbrs[0])
        linear = str(order).lower() == "triple" or (
            existing is not None and existing.GetBondType() == Chem.BondType.TRIPLE
        )
        if linear:
            candidates = [a + math.pi]
        else:
            candidates = [a + math.radians(120.0), a - math.radians(120.0)]
    else:
        srt = sorted(a % (2.0 * math.pi) for a in angles)
        gaps = []
        for k, a in enumerate(srt):
            nxt = srt[(k + 1) % len(srt)] + (2.0 * math.pi if k + 1 == len(srt) else 0.0)
            gaps.append((nxt - a, a))
        gaps.sort(reverse=True)
        widest = gaps[0][0]
        for gap, start in gaps:
            if gap < widest - 1e-6:
                break
            candidates.append(start + 0.5 * gap)
    # Least crowded candidate (zigzag falls out of this), nudged away from collisions.
    best = None
    for ang in candidates:
        for delta in (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0):
            a = ang + math.radians(delta)
            pt = (x0 + BOND_LEN * math.cos(a), y0 + BOND_LEN * math.sin(a))
            clear = _crowding(pt, positions, exclude=[idx] + nbrs)
            penalty = _crowding_penalty(pt, positions, exclude=[idx]) + 0.004 * abs(delta)
            if clear < 0.8 * BOND_LEN:
                penalty += 10.0
            if best is None or penalty < best[0]:
                best = (penalty, a)
            if delta == 0.0 and clear >= 0.8 * BOND_LEN:
                break
    a = best[1]
    return math.cos(a), math.sin(a)


def new_atom_position(mol, idx, order="single"):
    x0, y0 = atom_xy(mol, idx)
    ux, uy = free_direction(mol, idx, order=order)
    return x0 + BOND_LEN * ux, y0 + BOND_LEN * uy


def snapped_position(anchor_xy, target_xy, step_deg=30.0):
    ax, ay = anchor_xy
    ang = math.atan2(target_xy[1] - ay, target_xy[0] - ax)
    step = math.radians(step_deg)
    ang = round(ang / step) * step
    return ax + BOND_LEN * math.cos(ang), ay + BOND_LEN * math.sin(ang)


def find_atom_near(mol, xy, tol=0.35 * BOND_LEN, exclude=()):
    best = None
    skip = set(int(i) for i in exclude)
    for idx, (x, y) in all_positions(mol).items():
        if idx in skip:
            continue
        d = math.hypot(x - xy[0], y - xy[1])
        if d <= tol and (best is None or d < best[0]):
            best = (d, idx)
    return None if best is None else int(best[1])


# --------------------------------------------------------------------------
# Edit plumbing
# --------------------------------------------------------------------------


def _empty_rw():
    rw = Chem.RWMol()
    rw.AddConformer(Chem.Conformer(0), assignId=True)
    return rw


def _relax_h(atom):
    try:
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
        atom.UpdatePropertyCache(strict=False)
    except Exception:
        pass


def _clear_aromatic(mol):
    changed = False
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            atom.SetIsAromatic(False)
            changed = True
    for bond in mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
            changed = True
    return changed


def begin_edit(mol):
    """Kekulized editable copy of ``mol`` (or a new empty molecule)."""
    if mol is None or int(mol.GetNumAtoms()) == 0:
        return _empty_rw()
    base = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(base)
    except Exception:
        base = Chem.Mol(mol)
        _clear_aromatic(base)
        Chem.SanitizeMol(base)
    try:
        Chem.Kekulize(base, clearAromaticFlags=True)
    except Exception:
        pass
    if int(base.GetNumConformers()) <= 0:
        AllChem.Compute2DCoords(base)
    return Chem.RWMol(base)


def _friendly_error(err):
    txt = str(err or "").strip()
    low = txt.lower()
    if "valence" in low:
        import re

        m = re.search(r"atom\s*#\s*(\d+)\s+([A-Za-z]+)", txt)
        if m:
            return f"{m.group(2)} (atom {m.group(1)}) would exceed its allowed valence."
        return "That change would exceed an atom's allowed valence."
    if "kekulize" in low:
        return "Aromatic ring cannot be kekulized; check ring atoms and hydrogens."
    return txt or "Invalid molecule."


def finish_edit(rw, touched=()):
    """Sanitize an edited molecule, salvaging manual-H and aromatic leftovers."""
    mol = rw.GetMol() if hasattr(rw, "GetMol") else Chem.Mol(rw)
    for idx in touched:
        if 0 <= int(idx) < int(mol.GetNumAtoms()):
            _relax_h(mol.GetAtomWithIdx(int(idx)))
    if int(mol.GetNumAtoms()) == 0:
        return mol
    try:
        out = Chem.Mol(mol)
        Chem.SanitizeMol(out)
        return out
    except Exception as first:
        salvage = Chem.Mol(mol)
        for atom in salvage.GetAtoms():
            if atom.GetAtomicNum() != 1 and atom.GetFormalCharge() == 0:
                _relax_h(atom)
        _clear_aromatic(salvage)
        try:
            Chem.SanitizeMol(salvage)
            return salvage
        except Exception:
            raise SketchError(_friendly_error(first))


# --------------------------------------------------------------------------
# Edit operations (all return a new sanitized molecule)
# --------------------------------------------------------------------------


def add_atom(mol, anchor, symbol="C", order="single", xy=None):
    """Grow a new atom from ``anchor`` (or an isolated atom when anchor is None)."""
    z = atomic_number(symbol)
    rw = begin_edit(mol)
    if anchor is None:
        pos = xy if xy is not None else (0.0, 0.0)
        j = int(rw.AddAtom(Chem.Atom(z)))
        set_atom_xy(rw, j, pos[0], pos[1])
        return finish_edit(rw, touched=[j]), j
    i = int(anchor)
    if xy is None:
        xy = new_atom_position(rw, i, order=order)
    existing = find_atom_near(rw, xy, exclude=[i])
    if existing is not None and rw.GetBondBetweenAtoms(i, existing) is None:
        rw.AddBond(i, int(existing), _bond_type(order))
        return finish_edit(rw, touched=[i, existing]), int(existing)
    j = int(rw.AddAtom(Chem.Atom(z)))
    set_atom_xy(rw, j, xy[0], xy[1])
    rw.AddBond(i, j, _bond_type(order))
    return finish_edit(rw, touched=[i, j]), j


def add_chain_bond(mol, xy, order="single"):
    """Place a new two-atom C-C fragment starting at ``xy``."""
    rw = begin_edit(mol)
    i = int(rw.AddAtom(Chem.Atom(6)))
    set_atom_xy(rw, i, xy[0], xy[1])
    ang = math.radians(30.0)
    j = int(rw.AddAtom(Chem.Atom(6)))
    set_atom_xy(rw, j, xy[0] + BOND_LEN * math.cos(ang), xy[1] + BOND_LEN * math.sin(ang))
    rw.AddBond(i, j, _bond_type(order))
    return finish_edit(rw, touched=[i, j]), j


def connect_atoms(mol, i, j, order="single"):
    i, j = int(i), int(j)
    if i == j:
        raise SketchError("Pick two different atoms.")
    rw = begin_edit(mol)
    bond = rw.GetBondBetweenAtoms(i, j)
    if bond is None:
        rw.AddBond(i, j, _bond_type(order))
    else:
        bond.SetBondType(_bond_type(order))
    return finish_edit(rw, touched=[i, j])


def set_bond_order(mol, i, j, order):
    rw = begin_edit(mol)
    bond = rw.GetBondBetweenAtoms(int(i), int(j))
    if bond is None:
        raise SketchError("Those atoms are not bonded.")
    bond.SetBondType(_bond_type(order))
    bond.SetBondDir(Chem.BondDir.NONE)
    return finish_edit(rw, touched=[i, j])


def cycle_bond_order(mol, i, j):
    """single -> double -> triple -> single (aromatic bonds become double first)."""
    bond = mol.GetBondBetweenAtoms(int(i), int(j))
    if bond is None:
        raise SketchError("Those atoms are not bonded.")
    kek = begin_edit(mol)
    cur = bond_order_key(kek.GetBondBetweenAtoms(int(i), int(j)))
    nxt = {"single": "double", "double": "triple", "triple": "single"}.get(cur, "double")
    try:
        return set_bond_order(mol, i, j, nxt)
    except SketchError:
        if nxt == "triple":
            return set_bond_order(mol, i, j, "single")
        raise


def set_element(mol, idx, symbol):
    z = atomic_number(symbol)
    rw = begin_edit(mol)
    atom = rw.GetAtomWithIdx(int(idx))
    atom.SetAtomicNum(z)
    atom.SetFormalCharge(0)
    return finish_edit(rw, touched=[idx])


def change_charge(mol, idx, delta):
    rw = begin_edit(mol)
    atom = rw.GetAtomWithIdx(int(idx))
    atom.SetFormalCharge(int(atom.GetFormalCharge()) + int(delta))
    return finish_edit(rw, touched=[idx])


def delete_atoms(mol, indices):
    idxs = sorted({int(i) for i in indices}, reverse=True)
    if not idxs:
        return mol
    rw = begin_edit(mol)
    touched = set()
    for i in idxs:
        for nb in rw.GetAtomWithIdx(i).GetNeighbors():
            touched.add(int(nb.GetIdx()))
    for i in idxs:
        rw.RemoveAtom(i)
    remap = []
    for t in touched:
        if t in idxs:
            continue
        shift = sum(1 for d in idxs if d < t)
        remap.append(t - shift)
    return finish_edit(rw, touched=remap)


def delete_bond(mol, i, j):
    rw = begin_edit(mol)
    if rw.GetBondBetweenAtoms(int(i), int(j)) is None:
        raise SketchError("Those atoms are not bonded.")
    rw.RemoveBond(int(i), int(j))
    return finish_edit(rw, touched=[i, j])


def move_atoms(mol, indices, dx, dy):
    """In-place translation of atoms in the 2D conformer (no sanitization needed)."""
    for idx in indices:
        x, y = atom_xy(mol, idx)
        set_atom_xy(mol, idx, x + dx, y + dy)
    return mol


def clean_2d(mol):
    out = Chem.Mol(mol)
    AllChem.Compute2DCoords(out)
    return out


# ------------------------------ rings --------------------------------------


def ring_spec(tool_key):
    key = str(tool_key or "").strip().lower()
    if key == "benzene":
        return 6, True
    if key.startswith("ring"):
        try:
            return max(3, min(8, int(key[4:]))), False
        except Exception:
            return 6, False
    return 6, False


def _polygon(center, radius, start_angle, n, step_sign=1.0):
    pts = []
    for k in range(n):
        a = start_angle + step_sign * 2.0 * math.pi * k / float(n)
        pts.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)))
    return pts


def _circumradius(n, side=BOND_LEN):
    return side / (2.0 * math.sin(math.pi / float(n)))


def ring_vertices_free(n, center):
    return _polygon(center, _circumradius(n), math.pi / 2.0 + (math.pi / n if n % 2 == 0 else 0.0), n)


def ring_vertices_attached(mol, anchor, n, spiro=False):
    """Vertices for a ring hanging off ``anchor`` (first vertex bonded to it, or the anchor itself)."""
    x0, y0 = atom_xy(mol, anchor)
    ux, uy = free_direction(mol, anchor)
    r = _circumradius(n)
    if spiro:
        first = (x0, y0)
    else:
        first = (x0 + BOND_LEN * ux, y0 + BOND_LEN * uy)
    center = (first[0] + r * ux, first[1] + r * uy)
    start = math.atan2(first[1] - center[1], first[0] - center[0])
    return _polygon(center, r, start, n)


def ring_vertices_fused(mol, a, b, n):
    """Vertices (starting a, b, ...) for a ring sharing bond a-b, on its emptier side."""
    ax, ay = atom_xy(mol, a)
    bx, by = atom_xy(mol, b)
    side = max(math.hypot(bx - ax, by - ay), 1e-6)
    mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
    nx, ny = -(by - ay) / side, (bx - ax) / side
    score = 0.0
    for idx, (x, y) in all_positions(mol).items():
        if idx in (int(a), int(b)):
            continue
        d = math.hypot(x - mx, y - my)
        if d < 3.5 * BOND_LEN:
            score += ((x - mx) * nx + (y - my) * ny) / max(d, 1e-3)
    if score > 0:
        nx, ny = -nx, -ny
    apothem = side / (2.0 * math.tan(math.pi / float(n)))
    r = side / (2.0 * math.sin(math.pi / float(n)))
    center = (mx + apothem * nx, my + apothem * ny)
    ang_a = math.atan2(ay - center[1], ax - center[0])
    ang_b = math.atan2(by - center[1], bx - center[0])
    step = 2.0 * math.pi / float(n)
    diff = (ang_b - ang_a + math.pi) % (2.0 * math.pi) - math.pi
    sign = 1.0 if diff > 0 else -1.0
    del step
    return _polygon(center, r, ang_a, n, step_sign=sign)


def _build_ring(mol, vertices, fixed=None, aromatic=False, link_from=None):
    """Create ring atoms at ``vertices`` (reusing existing atoms on top of a vertex)."""
    rw = begin_edit(mol)
    fixed = dict(fixed or {})
    ids = []
    for k, xy in enumerate(vertices):
        if k in fixed:
            ids.append(int(fixed[k]))
            continue
        near = find_atom_near(rw, xy) if int(rw.GetNumAtoms()) else None
        if near is not None and near not in ids:
            ids.append(int(near))
            continue
        j = int(rw.AddAtom(Chem.Atom(6)))
        set_atom_xy(rw, j, xy[0], xy[1])
        ids.append(j)
    n = len(ids)
    ring_bonds = []
    for k in range(n):
        i, j = ids[k], ids[(k + 1) % n]
        if rw.GetBondBetweenAtoms(i, j) is None:
            rw.AddBond(i, j, Chem.BondType.SINGLE)
            ring_bonds.append((i, j, True))
        else:
            ring_bonds.append((i, j, False))
    if link_from is not None and rw.GetBondBetweenAtoms(int(link_from), ids[0]) is None:
        rw.AddBond(int(link_from), ids[0], Chem.BondType.SINGLE)
    touched = list(ids) + ([int(link_from)] if link_from is not None else [])
    if not aromatic:
        return finish_edit(rw, touched=touched)
    base = rw.GetMol()
    attempts = []
    # 1) Let RDKit kekulize an aromatic-flagged ring.
    arom = Chem.RWMol(base)
    for i in ids:
        arom.GetAtomWithIdx(i).SetIsAromatic(True)
    for i, j, _new in ring_bonds:
        bd = arom.GetBondBetweenAtoms(i, j)
        bd.SetBondType(Chem.BondType.AROMATIC)
        bd.SetIsAromatic(True)
    attempts.append(arom)
    # 2) Explicit alternating kekule patterns on the new bonds only.
    for offset in (0, 1):
        alt = Chem.RWMol(base)
        for k, (i, j, new) in enumerate(ring_bonds):
            if new and (k + offset) % 2 == 0:
                alt.GetBondBetweenAtoms(i, j).SetBondType(Chem.BondType.DOUBLE)
        attempts.append(alt)
    for cand in attempts:
        try:
            return finish_edit(cand, touched=touched)
        except SketchError:
            continue
    return finish_edit(base, touched=touched)


def attach_ring(mol, anchor, tool_key="benzene", spiro=False):
    n, aromatic = ring_spec(tool_key)
    verts = ring_vertices_attached(mol, anchor, n, spiro=spiro)
    if spiro:
        return _build_ring(mol, verts, fixed={0: int(anchor)}, aromatic=aromatic)
    return _build_ring(mol, verts, aromatic=aromatic, link_from=int(anchor))


def fuse_ring(mol, a, b, tool_key="benzene"):
    n, aromatic = ring_spec(tool_key)
    verts = ring_vertices_fused(mol, a, b, n)
    return _build_ring(mol, verts, fixed={0: int(a), 1: int(b)}, aromatic=aromatic)


def place_ring(mol, center, tool_key="benzene"):
    n, aromatic = ring_spec(tool_key)
    return _build_ring(mol, ring_vertices_free(n, center), aromatic=aromatic)


# ------------------------------ fragments ----------------------------------


def _fragment_with_coords(frag_smiles):
    frag = Chem.MolFromSmiles(str(frag_smiles or ""))
    if frag is None:
        raise SketchError(f"Invalid fragment SMILES: {frag_smiles}")
    dummies = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        raise SketchError("Fragment needs exactly one [*] attachment point.")
    dummy = int(dummies[0])
    nbs = list(frag.GetAtomWithIdx(dummy).GetNeighbors())
    if len(nbs) != 1:
        raise SketchError("Fragment attachment point must have one neighbor.")
    AllChem.Compute2DCoords(frag)
    return frag, dummy, int(nbs[0].GetIdx())


def attach_fragment(mol, anchor, frag_smiles):
    """Attach a [*]-fragment to ``anchor``, oriented along its free direction."""
    frag, dummy, fanchor = _fragment_with_coords(frag_smiles)
    base = begin_edit(mol)
    if anchor is None:
        target_dummy, target_dir = (0.0, 0.0), (1.0, 0.0)
    else:
        target_dummy = atom_xy(base, anchor)
        target_dir = free_direction(base, anchor)
    dx0, dy0 = atom_xy(frag, dummy)
    fx, fy = atom_xy(frag, fanchor)
    src = math.atan2(fy - dy0, fx - dx0)
    dst = math.atan2(target_dir[1], target_dir[0])
    others = all_positions(base)
    best = None
    for mirror in (1.0, -1.0):
        rot = dst - (mirror * src)
        coords = {}
        for a in frag.GetAtoms():
            ix = int(a.GetIdx())
            x, y = atom_xy(frag, ix)
            x, y = x - dx0, (y - dy0) * mirror
            coords[ix] = (
                target_dummy[0] + x * math.cos(rot) - y * math.sin(rot),
                target_dummy[1] + x * math.sin(rot) + y * math.cos(rot),
            )
        clash = sum(
            max(0.0, BOND_LEN - _crowding(xy, others)) for ix, xy in coords.items() if ix != dummy
        )
        if best is None or clash < best[0]:
            best = (clash, coords)
    coords = best[1]
    rw = Chem.RWMol(base)
    mapping = {}
    for a in frag.GetAtoms():
        ix = int(a.GetIdx())
        if ix == dummy:
            continue
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetIsAromatic(a.GetIsAromatic())
        na.SetNumExplicitHs(a.GetNumExplicitHs())
        j = int(rw.AddAtom(na))
        set_atom_xy(rw, j, coords[ix][0], coords[ix][1])
        mapping[ix] = j
    for bd in frag.GetBonds():
        i, j = int(bd.GetBeginAtomIdx()), int(bd.GetEndAtomIdx())
        if dummy in (i, j):
            continue
        rw.AddBond(mapping[i], mapping[j], bd.GetBondType())
        if bd.GetIsAromatic():
            rw.GetBondBetweenAtoms(mapping[i], mapping[j]).SetIsAromatic(True)
    touched = [mapping[fanchor]]
    if anchor is not None:
        rw.AddBond(int(anchor), mapping[fanchor], Chem.BondType.SINGLE)
        touched.append(int(anchor))
    return finish_edit(rw, touched=touched), mapping[fanchor]


# ------------------------------ stereo -------------------------------------


def display_wedges(mol):
    """{(begin, end): 'wedge'|'hash'} derived from chiral tags and 2D coordinates."""
    out = {}
    if mol is None or int(mol.GetNumAtoms()) == 0 or int(mol.GetNumConformers()) <= 0:
        return out
    try:
        tmp = Chem.Mol(mol)
        Chem.WedgeMolBonds(tmp, tmp.GetConformer())
        for bd in tmp.GetBonds():
            d = bd.GetBondDir()
            if d == Chem.BondDir.BEGINWEDGE:
                out[(int(bd.GetBeginAtomIdx()), int(bd.GetEndAtomIdx()))] = "wedge"
            elif d == Chem.BondDir.BEGINDASH:
                out[(int(bd.GetBeginAtomIdx()), int(bd.GetEndAtomIdx()))] = "hash"
    except Exception:
        return {}
    return out


def stereo_labels(mol):
    """{atom_idx: 'R'|'S'|'?'} for (possible) tetrahedral stereocenters."""
    out = {}
    if mol is None or int(mol.GetNumAtoms()) == 0:
        return out
    try:
        tmp = Chem.Mol(mol)
        Chem.AssignStereochemistry(tmp, cleanIt=True, force=True)
        for idx, label in Chem.FindMolChiralCenters(tmp, includeUnassigned=True, useLegacyImplementation=False):
            out[int(idx)] = str(label)
    except Exception:
        return {}
    return out


def potential_stereocenters(mol):
    return set(stereo_labels(mol).keys())


def set_wedge(mol, begin, end, kind="wedge"):
    """Define the stereocenter at ``begin`` with a wedge/hash bond toward ``end``."""
    begin, end = int(begin), int(end)
    if mol.GetBondBetweenAtoms(begin, end) is None:
        raise SketchError("Those atoms are not bonded.")
    current = display_wedges(mol)
    rw = begin_edit(mol)
    bond = rw.GetBondBetweenAtoms(begin, end)
    if bond.GetBondType() != Chem.BondType.SINGLE:
        raise SketchError("Wedge/hash bonds must be single bonds.")
    if int(bond.GetBeginAtomIdx()) != begin:
        rw.RemoveBond(begin, end)
        rw.AddBond(begin, end, Chem.BondType.SINGLE)
    for (b0, b1), k in current.items():
        if b0 == begin:
            continue
        bd = rw.GetBondBetweenAtoms(b0, b1)
        if bd is not None and int(bd.GetBeginAtomIdx()) == b0:
            bd.SetBondDir(Chem.BondDir.BEGINWEDGE if k == "wedge" else Chem.BondDir.BEGINDASH)
    for bd in rw.GetAtomWithIdx(begin).GetBonds():
        if int(bd.GetBeginAtomIdx()) == begin:
            bd.SetBondDir(Chem.BondDir.NONE)
    bond = rw.GetBondBetweenAtoms(begin, end)
    bond.SetBondDir(Chem.BondDir.BEGINWEDGE if kind == "wedge" else Chem.BondDir.BEGINDASH)
    out = finish_edit(rw)
    Chem.AssignChiralTypesFromBondDirs(out, out.GetConformer().GetId())
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    for bd in out.GetBonds():
        bd.SetBondDir(Chem.BondDir.NONE)
    return out


def invert_center(mol, idx):
    out = Chem.Mol(mol)
    atom = out.GetAtomWithIdx(int(idx))
    if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
        raise SketchError("Atom has no defined stereo; draw a wedge or hash bond first.")
    atom.InvertChirality()
    Chem.AssignStereochemistry(out, cleanIt=True, force=True)
    return out


def set_center_cip(mol, idx, label):
    """Give stereocenter ``idx`` the CIP label 'R' or 'S' (or 'none' to clear it)."""
    want = str(label or "").strip().upper()
    idx = int(idx)
    if want in ("NONE", "", "?", "UNDEFINED"):
        out = Chem.Mol(mol)
        out.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.AssignStereochemistry(out, cleanIt=True, force=True)
        return out
    if want not in ("R", "S"):
        raise SketchError("Stereo label must be R, S or none.")
    if idx not in potential_stereocenters(mol):
        raise SketchError(f"Atom {idx + 1} is not a stereocenter.")
    for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
        out = Chem.Mol(mol)
        out.GetAtomWithIdx(idx).SetChiralTag(tag)
        Chem.AssignStereochemistry(out, cleanIt=True, force=True)
        if stereo_labels(out).get(idx) == want:
            return out
    raise SketchError(f"Could not assign {want} to atom {idx + 1}.")


def resolve_group(name_or_smiles):
    """[*]-SMILES for a group label/name from FRAGMENT_LIBRARY, or a [*]-SMILES as given."""
    raw = str(name_or_smiles or "").strip()
    key = raw.lower().replace("-", "").replace(" ", "")
    for label, name, smi in FRAGMENT_LIBRARY:
        if key in (label.lower(), name.lower().replace("-", "").replace(" ", "")):
            return smi
    if "*" in raw and Chem.MolFromSmiles(raw) is not None:
        return raw
    known = ", ".join(label for label, _n, _s in FRAGMENT_LIBRARY)
    raise SketchError(f"Unknown group {raw!r}. Use one of: {known}; or a SMILES with one [*] attachment point.")


# ------------------------------ properties ---------------------------------


def molecule_summary(mol):
    info = {
        "atoms": 0,
        "formula": "",
        "mw": 0.0,
        "charge": 0,
        "clogp": 0.0,
        "tpsa": 0.0,
        "hbd": 0,
        "hba": 0,
        "rotb": 0,
        "rings": 0,
        "fragments": 0,
        "stereo_undefined": 0,
        "lipinski_violations": 0,
        "smiles": "",
    }
    if mol is None or int(mol.GetNumAtoms()) == 0:
        return info
    try:
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

        info["atoms"] = int(mol.GetNumHeavyAtoms())
        info["formula"] = str(rdMolDescriptors.CalcMolFormula(mol))
        info["mw"] = float(Descriptors.MolWt(mol))
        info["charge"] = int(Chem.GetFormalCharge(mol))
        info["clogp"] = float(Crippen.MolLogP(mol))
        info["tpsa"] = float(rdMolDescriptors.CalcTPSA(mol))
        info["hbd"] = int(Lipinski.NumHDonors(mol))
        info["hba"] = int(Lipinski.NumHAcceptors(mol))
        info["rotb"] = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        info["rings"] = int(rdMolDescriptors.CalcNumRings(mol))
        info["fragments"] = len(Chem.GetMolFrags(mol))
        info["stereo_undefined"] = sum(1 for v in stereo_labels(mol).values() if v == "?")
        info["lipinski_violations"] = int(
            (info["mw"] > 500) + (info["clogp"] > 5) + (info["hbd"] > 5) + (info["hba"] > 10)
        )
        info["smiles"] = str(Chem.MolToSmiles(mol, isomericSmiles=True))
    except Exception:
        pass
    return info


def formula_html(formula):
    out = []
    for ch in str(formula or ""):
        out.append(f"<sub>{ch}</sub>" if ch.isdigit() else ch)
    return "".join(out).replace("</sub><sub>", "")


# --------------------------------------------------------------------------
# Rendering data
# --------------------------------------------------------------------------


def atom_label_parts(atom, show_carbon=False):
    """(symbol, hcount, charge) for a label, or None for a hidden skeletal carbon."""
    sym = str(atom.GetSymbol())
    charge = int(atom.GetFormalCharge())
    try:
        hcount = int(atom.GetTotalNumHs())
    except Exception:
        hcount = 0
    iso = int(atom.GetIsotope())
    degree = int(atom.GetDegree())
    if sym == "C" and not show_carbon and charge == 0 and iso == 0 and degree > 0 and atom.GetNumRadicalElectrons() == 0:
        return None
    return sym, hcount, charge


def build_render_model(mol, show_carbons=False):
    """Kekulized depiction data cached per molecule change."""
    data = {"bonds": [], "labels": {}, "wedges": {}, "cip": {}, "ring_center": {}}
    if mol is None or int(mol.GetNumAtoms()) == 0:
        return data
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=False)
    except Exception:
        pass
    pos = all_positions(mol)
    ring_info = mol.GetRingInfo()
    rings = [tuple(r) for r in ring_info.AtomRings()]
    for bd in kek.GetBonds():
        a, b = int(bd.GetBeginAtomIdx()), int(bd.GetEndAtomIdx())
        center = None
        best_ring = None
        for ring in rings:
            if a in ring and b in ring:
                if best_ring is None or len(ring) < len(best_ring) or (
                    len(ring) == 6 and len(best_ring) != 6
                ):
                    best_ring = ring
        if best_ring is not None:
            xs = [pos[i][0] for i in best_ring]
            ys = [pos[i][1] for i in best_ring]
            center = (sum(xs) / len(xs), sum(ys) / len(ys))
        data["bonds"].append({"a": a, "b": b, "order": bond_order_key(bd), "ring_center": center})
    for atom in mol.GetAtoms():
        parts = atom_label_parts(atom, show_carbon=show_carbons)
        if parts is not None:
            data["labels"][int(atom.GetIdx())] = parts
    data["wedges"] = display_wedges(mol)
    data["cip"] = stereo_labels(mol)
    return data


# --------------------------------------------------------------------------
# Icons
# --------------------------------------------------------------------------

ICON_INK = "#243447"


def _icon_pixmap(draw_fn, size=28):
    ratio = 2.0
    try:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            ratio = max(2.0, float(app.devicePixelRatio()))
    except Exception:
        pass
    pix = QtGui.QPixmap(int(size * ratio), int(size * ratio))
    pix.fill(QtCore.Qt.transparent)
    try:
        pix.setDevicePixelRatio(ratio)
    except Exception:
        pass
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    draw_fn(p, float(size))
    p.end()
    return QtGui.QIcon(pix)


def _pen(color=ICON_INK, width=1.8):
    pen = QtGui.QPen(QtGui.QColor(color), width)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    return pen


def tool_icon(key):
    key = str(key)

    def line(p, s, off):
        p.drawLine(QtCore.QPointF(s * 0.2 + off[0], s * 0.8 + off[1]), QtCore.QPointF(s * 0.8 + off[0], s * 0.2 + off[1]))

    def draw(p, s):
        p.setPen(_pen())
        if key == "select":
            poly = QtGui.QPolygonF(
                [
                    QtCore.QPointF(s * 0.28, s * 0.16),
                    QtCore.QPointF(s * 0.28, s * 0.80),
                    QtCore.QPointF(s * 0.44, s * 0.64),
                    QtCore.QPointF(s * 0.56, s * 0.88),
                    QtCore.QPointF(s * 0.66, s * 0.83),
                    QtCore.QPointF(s * 0.54, s * 0.59),
                    QtCore.QPointF(s * 0.76, s * 0.59),
                ]
            )
            p.setBrush(QtGui.QColor("#FFFFFF"))
            p.drawPolygon(poly)
        elif key == "erase":
            p.save()
            p.translate(s * 0.5, s * 0.5)
            p.rotate(-40)
            p.setBrush(QtGui.QColor("#F9A8B4"))
            p.drawRoundedRect(QtCore.QRectF(-s * 0.32, -s * 0.15, s * 0.64, s * 0.30), 3, 3)
            p.setBrush(QtGui.QColor("#FFFFFF"))
            p.drawRoundedRect(QtCore.QRectF(-s * 0.32, -s * 0.15, s * 0.24, s * 0.30), 3, 3)
            p.restore()
        elif key == "single":
            line(p, s, (0, 0))
        elif key == "double":
            line(p, s, (-2.2, -2.2))
            line(p, s, (2.2, 2.2))
        elif key == "triple":
            line(p, s, (-3.4, -3.4))
            line(p, s, (0, 0))
            line(p, s, (3.4, 3.4))
        elif key == "wedge":
            p.setBrush(QtGui.QColor(ICON_INK))
            poly = QtGui.QPolygonF(
                [QtCore.QPointF(s * 0.2, s * 0.8), QtCore.QPointF(s * 0.74, s * 0.16), QtCore.QPointF(s * 0.84, s * 0.26)]
            )
            p.drawPolygon(poly)
        elif key == "hash":
            for k in range(6):
                t = 0.12 + 0.16 * k
                cx, cy = s * (0.2 + 0.6 * t / 0.92), s * (0.8 - 0.6 * t / 0.92)
                half = 1.0 + 5.0 * t
                p.drawLine(QtCore.QPointF(cx - half * 0.7, cy - half * 0.7), QtCore.QPointF(cx + half * 0.7, cy + half * 0.7))
        elif key in ("charge_plus", "charge_minus"):
            p.setBrush(QtGui.QColor("#FFFFFF"))
            p.drawEllipse(QtCore.QPointF(s * 0.5, s * 0.5), s * 0.3, s * 0.3)
            p.drawLine(QtCore.QPointF(s * 0.36, s * 0.5), QtCore.QPointF(s * 0.64, s * 0.5))
            if key == "charge_plus":
                p.drawLine(QtCore.QPointF(s * 0.5, s * 0.36), QtCore.QPointF(s * 0.5, s * 0.64))
        elif key == "chain":
            pts = [QtCore.QPointF(s * x, s * y) for x, y in ((0.12, 0.62), (0.34, 0.40), (0.56, 0.62), (0.78, 0.40))]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        elif key.startswith("ring") or key == "benzene":
            n, aromatic = ring_spec(key)
            r = s * 0.36
            start = -math.pi / 2.0 + (math.pi / n if n % 2 == 0 else 0.0)
            pts = [
                QtCore.QPointF(s * 0.5 + r * math.cos(start + 2 * math.pi * k / n), s * 0.53 + r * math.sin(start + 2 * math.pi * k / n))
                for k in range(n)
            ]
            p.drawPolygon(QtGui.QPolygonF(pts))
            if aromatic:
                inner = [QtCore.QPointF(s * 0.5 + 0.72 * (q.x() - s * 0.5), s * 0.53 + 0.72 * (q.y() - s * 0.53)) for q in pts]
                for k in (0, 2, 4):
                    p.drawLine(inner[k], inner[(k + 1) % n])
        elif key == "invert":
            p.drawArc(QtCore.QRectF(s * 0.2, s * 0.2, s * 0.6, s * 0.6), 30 * 16, 280 * 16)
            p.setBrush(QtGui.QColor(ICON_INK))
            p.drawPolygon(
                QtGui.QPolygonF(
                    [QtCore.QPointF(s * 0.78, s * 0.22), QtCore.QPointF(s * 0.80, s * 0.42), QtCore.QPointF(s * 0.62, s * 0.36)]
                )
            )

    return _icon_pixmap(draw)


def action_icon(key):
    key = str(key)

    def draw(p, s):
        p.setPen(_pen(width=1.7))
        if key in ("undo", "redo"):
            sign = 1.0 if key == "undo" else -1.0
            path = QtGui.QPainterPath()
            x0 = s * (0.5 - sign * 0.28)
            path.moveTo(QtCore.QPointF(x0, s * 0.42))
            path.cubicTo(
                QtCore.QPointF(s * 0.5, s * 0.20),
                QtCore.QPointF(s * (0.5 + sign * 0.42), s * 0.40),
                QtCore.QPointF(s * (0.5 + sign * 0.18), s * 0.78),
            )
            p.drawPath(path)
            p.drawLine(QtCore.QPointF(x0, s * 0.42), QtCore.QPointF(x0 + sign * s * 0.02, s * 0.20))
            p.drawLine(QtCore.QPointF(x0, s * 0.42), QtCore.QPointF(x0 + sign * s * 0.22, s * 0.46))
        elif key == "clean":
            p.drawLine(QtCore.QPointF(s * 0.2, s * 0.8), QtCore.QPointF(s * 0.62, s * 0.38))
            for cx, cy, r in ((0.72, 0.26, 0.07), (0.50, 0.20, 0.045), (0.80, 0.50, 0.045)):
                p.drawLine(QtCore.QPointF(s * (cx - r), s * cy), QtCore.QPointF(s * (cx + r), s * cy))
                p.drawLine(QtCore.QPointF(s * cx, s * (cy - r)), QtCore.QPointF(s * cx, s * (cy + r)))
        elif key == "fit":
            for sx, sy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                x = s * (0.2 if sx == 0 else 0.8)
                y = s * (0.2 if sy == 0 else 0.8)
                dx = s * 0.16 * (1 if sx == 0 else -1)
                dy = s * 0.16 * (1 if sy == 0 else -1)
                p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + dx, y))
                p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x, y + dy))
            p.drawEllipse(QtCore.QPointF(s * 0.5, s * 0.5), s * 0.12, s * 0.12)
        elif key == "clear":
            p.drawRoundedRect(QtCore.QRectF(s * 0.28, s * 0.32, s * 0.44, s * 0.50), 2, 2)
            p.drawLine(QtCore.QPointF(s * 0.22, s * 0.32), QtCore.QPointF(s * 0.78, s * 0.32))
            p.drawLine(QtCore.QPointF(s * 0.42, s * 0.22), QtCore.QPointF(s * 0.58, s * 0.22))
            p.drawLine(QtCore.QPointF(s * 0.42, s * 0.44), QtCore.QPointF(s * 0.42, s * 0.72))
            p.drawLine(QtCore.QPointF(s * 0.58, s * 0.44), QtCore.QPointF(s * 0.58, s * 0.72))
        elif key == "import":
            p.drawLine(QtCore.QPointF(s * 0.5, s * 0.16), QtCore.QPointF(s * 0.5, s * 0.60))
            p.drawLine(QtCore.QPointF(s * 0.34, s * 0.45), QtCore.QPointF(s * 0.5, s * 0.60))
            p.drawLine(QtCore.QPointF(s * 0.66, s * 0.45), QtCore.QPointF(s * 0.5, s * 0.60))
            p.drawLine(QtCore.QPointF(s * 0.2, s * 0.66), QtCore.QPointF(s * 0.2, s * 0.82))
            p.drawLine(QtCore.QPointF(s * 0.2, s * 0.82), QtCore.QPointF(s * 0.8, s * 0.82))
            p.drawLine(QtCore.QPointF(s * 0.8, s * 0.82), QtCore.QPointF(s * 0.8, s * 0.66))
        elif key == "hydrogen":
            f = QtGui.QFont()
            f.setPixelSize(int(s * 0.5))
            f.setBold(True)
            p.setFont(f)
            p.drawText(QtCore.QRectF(0, 0, s * 0.8, s), QtCore.Qt.AlignCenter, "H")
            p.drawText(QtCore.QRectF(s * 0.46, 0, s * 0.5, s * 0.6), QtCore.Qt.AlignCenter, "+")
        elif key == "copy":
            p.drawRoundedRect(QtCore.QRectF(s * 0.34, s * 0.30, s * 0.42, s * 0.50), 3, 3)
            p.drawLine(QtCore.QPointF(s * 0.24, s * 0.66), QtCore.QPointF(s * 0.24, s * 0.20))
            p.drawLine(QtCore.QPointF(s * 0.24, s * 0.20), QtCore.QPointF(s * 0.60, s * 0.20))

    return _icon_pixmap(draw, size=22)


# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------


class SketcherCanvas(QtWidgets.QWidget):
    """Skeletal-formula view with direct manipulation; delegates edits to its controller."""

    HOVER = QtGui.QColor(37, 99, 235, 60)
    SELECT = QtGui.QColor(37, 99, 235, 45)
    BOND_INK = QtGui.QColor("#1F2933")

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.scale = 32.0
        self.offset = QtCore.QPointF(0.0, 0.0)
        self._fitted_once = False
        self.hover_atom = None
        self.hover_bond = None
        self.selection = set()
        self._press = None
        self._press_pos = None
        self._press_button = None
        self._press_atom = None
        self._press_bond = None
        self._dragging = False
        self._drag_last = None
        self._band = None
        self._pan_last = None
        self._cursor = None
        self._ghost = None
        self._flash = None
        self.show_atom_numbers = False
        self.show_stereo = True
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(420, 360)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    # ---------------- coordinates -----------------
    def to_screen(self, x, y):
        return QtCore.QPointF(self.offset.x() + x * self.scale, self.offset.y() - y * self.scale)

    def to_model(self, pt):
        return ((pt.x() - self.offset.x()) / self.scale, (self.offset.y() - pt.y()) / self.scale)

    def fit_view(self, max_scale=38.0):
        mol = self.controller.mol
        pos = all_positions(mol)
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        if not pos:
            self.scale = 32.0
            self.offset = QtCore.QPointF(w / 2.0, h / 2.0)
            self.update()
            return
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        span_x = max(max(xs) - min(xs), BOND_LEN)
        span_y = max(max(ys) - min(ys), BOND_LEN)
        pad = 60.0
        self.scale = max(8.0, min(max_scale, (w - 2 * pad) / span_x, (h - 2 * pad) / span_y))
        cx = 0.5 * (max(xs) + min(xs))
        cy = 0.5 * (max(ys) + min(ys))
        self.offset = QtCore.QPointF(w / 2.0 - cx * self.scale, h / 2.0 + cy * self.scale)
        self.update()

    def ensure_visible(self):
        """Pan (never zoom) just enough to keep newly drawn atoms on screen."""
        pos = all_positions(self.controller.mol)
        if not pos:
            return
        margin = 30.0
        dx = dy = 0.0
        for x, y in pos.values():
            p = self.to_screen(x, y)
            if p.x() < margin:
                dx = max(dx, margin - p.x())
            if p.x() > self.width() - margin:
                dx = min(dx, self.width() - margin - p.x())
            if p.y() < margin:
                dy = max(dy, margin - p.y())
            if p.y() > self.height() - margin:
                dy = min(dy, self.height() - margin - p.y())
        if dx or dy:
            self.offset = QtCore.QPointF(self.offset.x() + dx, self.offset.y() + dy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._fitted_once:
            self.fit_view()
            self._fitted_once = True

    # ---------------- hit testing -----------------
    def atom_at(self, pt, radius=11.0):
        best = None
        for idx, (x, y) in all_positions(self.controller.mol).items():
            q = self.to_screen(x, y)
            d = math.hypot(q.x() - pt.x(), q.y() - pt.y())
            if d <= radius and (best is None or d < best[0]):
                best = (d, idx)
        return None if best is None else int(best[1])

    def bond_at(self, pt, radius=7.0):
        mol = self.controller.mol
        if mol is None:
            return None
        pos = all_positions(mol)
        best = None
        for bd in mol.GetBonds():
            a, b = int(bd.GetBeginAtomIdx()), int(bd.GetEndAtomIdx())
            p1 = self.to_screen(*pos[a])
            p2 = self.to_screen(*pos[b])
            vx, vy = p2.x() - p1.x(), p2.y() - p1.y()
            vv = vx * vx + vy * vy
            if vv < 1e-6:
                continue
            t = max(0.0, min(1.0, ((pt.x() - p1.x()) * vx + (pt.y() - p1.y()) * vy) / vv))
            d = math.hypot(pt.x() - (p1.x() + t * vx), pt.y() - (p1.y() + t * vy))
            if d <= radius and 0.12 < t < 0.88 and (best is None or d < best[0]):
                best = (d, (a, b))
        return None if best is None else best[1]

    # ---------------- input -----------------
    def _pos(self, event):
        try:
            return QtCore.QPointF(event.localPos())
        except Exception:
            return QtCore.QPointF(event.pos())

    def mousePressEvent(self, event):
        self.setFocus()
        pt = self._pos(event)
        self._press = pt
        self._press_button = event.button()
        self._dragging = False
        self._press_atom = self.atom_at(pt)
        self._press_bond = None if self._press_atom is not None else self.bond_at(pt)
        if event.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
            self._pan_last = pt
        event.accept()

    def mouseMoveEvent(self, event):
        pt = self._pos(event)
        self._cursor = pt
        if self._pan_last is not None:
            d = pt - self._pan_last
            self.offset = self.offset + d
            self._pan_last = pt
            self._dragging = True
            self.update()
            return
        if self._press is not None and not self._dragging:
            if math.hypot(pt.x() - self._press.x(), pt.y() - self._press.y()) > 4.0:
                self._dragging = True
                self._drag_last = self._press
                self.controller.canvas_drag_started(self._press_atom, self._press_bond)
        if self._press is not None and self._dragging:
            self._handle_drag(pt)
        else:
            atom = self.atom_at(pt)
            bond = None if atom is not None else self.bond_at(pt)
            if atom != self.hover_atom or bond != self.hover_bond:
                self.hover_atom, self.hover_bond = atom, bond
                self._ghost = self.controller.ghost_for(atom, bond)
        self.update()

    def _handle_drag(self, pt):
        tool = self.controller.tool
        if tool == "select":
            if self._press_atom is not None or self._press_bond is not None:
                prev = self.to_model(self._drag_last)
                cur = self.to_model(pt)
                self.controller.move_by(self._drag_group(), cur[0] - prev[0], cur[1] - prev[1])
                self._drag_last = pt
            else:
                self._band = QtCore.QRectF(self._press, pt).normalized()
        elif self._press_atom is not None and self.controller.tool_draws_bonds():
            anchor = atom_xy(self.controller.mol, self._press_atom)
            target_atom = self.atom_at(pt)
            if target_atom is not None and target_atom != self._press_atom:
                self._ghost = {"segments": [(anchor, atom_xy(self.controller.mol, target_atom))], "label": None}
            else:
                end = snapped_position(anchor, self.to_model(pt))
                self._ghost = {"segments": [(anchor, end)], "label": self.controller.ghost_label()}

    def _drag_group(self):
        if self._press_atom is not None:
            if self._press_atom in self.selection:
                return sorted(self.selection)
            return [self._press_atom]
        if self._press_bond is not None:
            a, b = self._press_bond
            if a in self.selection and b in self.selection:
                return sorted(self.selection)
            return [a, b]
        return []

    def mouseReleaseEvent(self, event):
        pt = self._pos(event)
        mods = event.modifiers()
        shift = bool(mods & QtCore.Qt.ShiftModifier)
        if self._pan_last is not None:
            self._pan_last = None
            self._press = None
            if not self._dragging and event.button() == QtCore.Qt.RightButton:
                self.controller.canvas_context_menu(self.atom_at(pt), self.bond_at(pt), self.mapToGlobal(pt.toPoint()))
            self._dragging = False
            return
        if self._press is None:
            return
        tool = self.controller.tool
        if self._dragging:
            if tool == "select":
                if self._band is not None:
                    picked = set()
                    for idx, (x, y) in all_positions(self.controller.mol).items():
                        if self._band.contains(self.to_screen(x, y)):
                            picked.add(idx)
                    self.selection = (self.selection | picked) if shift else picked
                    self.controller.selection_changed()
                self.controller.canvas_drag_finished()
            elif self._press_atom is not None and self.controller.tool_draws_bonds():
                target_atom = self.atom_at(pt)
                if target_atom is not None and target_atom != self._press_atom:
                    self.controller.connect_to(self._press_atom, target_atom)
                else:
                    end = snapped_position(atom_xy(self.controller.mol, self._press_atom), self.to_model(pt))
                    self.controller.grow_to(self._press_atom, end)
        else:
            self.controller.canvas_click(self._press_atom, self._press_bond, self.to_model(pt), shift=shift)
        self._press = None
        self._band = None
        self._dragging = False
        self.hover_atom = self.atom_at(pt)
        self.hover_bond = None if self.hover_atom is not None else self.bond_at(pt)
        self._ghost = self.controller.ghost_for(self.hover_atom, self.hover_bond)
        self.update()

    def leaveEvent(self, event):
        self.hover_atom = None
        self.hover_bond = None
        self._ghost = None
        self._cursor = None
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        try:
            delta = event.angleDelta().y()
        except Exception:
            delta = 0
        if not delta:
            return
        factor = 1.0015 ** delta
        new_scale = max(6.0, min(160.0, self.scale * factor))
        factor = new_scale / self.scale
        try:
            anchor = QtCore.QPointF(event.position())
        except Exception:
            anchor = QtCore.QPointF(event.pos())
        self.offset = QtCore.QPointF(
            anchor.x() - (anchor.x() - self.offset.x()) * factor,
            anchor.y() - (anchor.y() - self.offset.y()) * factor,
        )
        self.scale = new_scale
        self.update()

    def keyPressEvent(self, event):
        if self.controller.canvas_key(event, self.hover_atom, self.hover_bond):
            event.accept()
            return
        super().keyPressEvent(event)

    def refresh_hover(self):
        if self._cursor is None:
            self._ghost = None
            return
        self.hover_atom = self.atom_at(self._cursor)
        self.hover_bond = None if self.hover_atom is not None else self.bond_at(self._cursor)
        self._ghost = self.controller.ghost_for(self.hover_atom, self.hover_bond)

    def flash_atoms(self, indices):
        self._flash = set(int(i) for i in indices)
        QtCore.QTimer.singleShot(900, self._clear_flash)
        self.update()

    def _clear_flash(self):
        self._flash = None
        self.update()

    # ---------------- painting -----------------
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor("#FFFFFF"))
        self._paint_grid(p)
        mol = self.controller.mol
        if mol is None or int(mol.GetNumAtoms()) == 0:
            self._paint_empty(p)
            p.end()
            return
        render = self.controller.render
        pos = {i: self.to_screen(*xy) for i, xy in all_positions(mol).items()}
        font_px = max(10, min(22, int(self.scale * 0.46)))
        font = QtGui.QFont(self.font())
        font.setPixelSize(font_px)
        font.setWeight(QtGui.QFont.DemiBold)
        metrics = QtGui.QFontMetricsF(font)
        radius = {idx: 0.62 * font_px for idx in render.get("labels", {})}
        self._paint_highlights(p, pos, font_px)
        self._paint_bonds(p, render, pos, radius)
        self._paint_labels(p, mol, render, pos, font, metrics)
        self._paint_extras(p, mol, render, pos, font_px)
        self._paint_ghost(p)
        if self._band is not None:
            p.setPen(QtGui.QPen(QtGui.QColor("#2563EB"), 1, QtCore.Qt.DashLine))
            p.setBrush(QtGui.QColor(37, 99, 235, 25))
            p.drawRect(self._band)
        p.end()

    def _paint_grid(self, p):
        step = BOND_LEN * self.scale
        if step < 14:
            return
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#E4E9F0"))
        ox = self.offset.x() % step
        oy = self.offset.y() % step
        y = oy
        while y < self.height():
            x = ox
            while x < self.width():
                p.drawEllipse(QtCore.QPointF(x, y), 1.0, 1.0)
                x += step
            y += step

    def _paint_empty(self, p):
        rect = QtCore.QRectF(self.rect())
        p.setPen(QtGui.QColor("#52606D"))
        f = QtGui.QFont(self.font())
        f.setPixelSize(17)
        f.setWeight(QtGui.QFont.DemiBold)
        p.setFont(f)
        p.drawText(rect.adjusted(0, -60, 0, -60), QtCore.Qt.AlignCenter, "Start drawing")
        f.setPixelSize(13)
        f.setWeight(QtGui.QFont.Normal)
        p.setFont(f)
        p.setPen(QtGui.QColor("#7B8794"))
        p.drawText(
            rect.adjusted(40, 10, -40, 10),
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap,
            "Click anywhere with a bond, ring, or element tool.\n"
            "Type or paste a SMILES on the right, or use Import to load a file or the active ligand.",
        )

    def _paint_highlights(self, p, pos, font_px):
        r = max(9.0, 0.55 * font_px + 4.0)
        p.setPen(QtCore.Qt.NoPen)
        if self.selection:
            p.setBrush(self.SELECT)
            for idx in self.selection:
                if idx in pos:
                    p.drawEllipse(pos[idx], r + 2, r + 2)
        if self._flash:
            p.setBrush(QtGui.QColor(220, 38, 38, 70))
            for idx in self._flash:
                if idx in pos:
                    p.drawEllipse(pos[idx], r + 5, r + 5)
        if self.hover_atom is not None and self.hover_atom in pos:
            color = QtGui.QColor(220, 38, 38, 70) if self.controller.tool == "erase" else self.HOVER
            p.setBrush(color)
            p.drawEllipse(pos[self.hover_atom], r + 3, r + 3)
        if self.hover_bond is not None:
            a, b = self.hover_bond
            if a in pos and b in pos:
                color = QtGui.QColor(220, 38, 38, 80) if self.controller.tool == "erase" else QtGui.QColor(37, 99, 235, 70)
                pen = QtGui.QPen(color, max(8.0, 0.3 * self.scale))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                p.setPen(pen)
                p.drawLine(pos[a], pos[b])
                p.setPen(QtCore.Qt.NoPen)

    def _trimmed(self, p1, p2, r1, r2):
        dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return p1, p2
        ux, uy = dx / d, dy / d
        r1 = min(r1, 0.45 * d)
        r2 = min(r2, 0.45 * d)
        return QtCore.QPointF(p1.x() + ux * r1, p1.y() + uy * r1), QtCore.QPointF(p2.x() - ux * r2, p2.y() - uy * r2)

    def _paint_bonds(self, p, render, pos, radius):
        width = max(1.4, min(3.0, self.scale * 0.055))
        gap = max(3.5, self.scale * 0.16)
        pen = QtGui.QPen(self.BOND_INK, width)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        wedges = render.get("wedges", {})
        for bd in render.get("bonds", []):
            a, b = bd["a"], bd["b"]
            if a not in pos or b not in pos:
                continue
            p1, p2 = self._trimmed(pos[a], pos[b], radius.get(a, 0.0), radius.get(b, 0.0))
            dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            nx, ny = -dy / d, dx / d
            order = bd["order"]
            wedge = wedges.get((a, b)) or None
            if wedge is None and (b, a) in wedges:
                wedge = wedges[(b, a)]
                p1, p2 = p2, p1
                dx, dy, nx, ny = -dx, -dy, -nx, -ny
            p.setPen(pen)
            if wedge == "wedge":
                w = max(4.0, self.scale * 0.17)
                poly = QtGui.QPolygonF(
                    [p1, QtCore.QPointF(p2.x() + nx * w, p2.y() + ny * w), QtCore.QPointF(p2.x() - nx * w, p2.y() - ny * w)]
                )
                p.setBrush(self.BOND_INK)
                p.drawPolygon(poly)
                p.setBrush(QtCore.Qt.NoBrush)
                continue
            if wedge == "hash":
                w = max(4.0, self.scale * 0.17)
                steps = max(4, int(d / max(4.0, self.scale * 0.13)))
                thin = QtGui.QPen(self.BOND_INK, max(1.1, width * 0.75))
                p.setPen(thin)
                for k in range(1, steps + 1):
                    t = k / float(steps)
                    cx, cy = p1.x() + dx * t, p1.y() + dy * t
                    p.drawLine(QtCore.QPointF(cx + nx * w * t, cy + ny * w * t), QtCore.QPointF(cx - nx * w * t, cy - ny * w * t))
                continue
            if order == "single":
                p.drawLine(p1, p2)
            elif order == "triple":
                p.drawLine(p1, p2)
                for s in (1, -1):
                    p.drawLine(
                        QtCore.QPointF(p1.x() + s * nx * gap, p1.y() + s * ny * gap),
                        QtCore.QPointF(p2.x() + s * nx * gap, p2.y() + s * ny * gap),
                    )
            else:
                center = bd.get("ring_center")
                side = None
                if center is not None:
                    cpt = self.to_screen(*center)
                    mx, my = 0.5 * (p1.x() + p2.x()), 0.5 * (p1.y() + p2.y())
                    side = 1.0 if ((cpt.x() - mx) * nx + (cpt.y() - my) * ny) > 0 else -1.0
                elif a not in radius and b not in radius:
                    side = self._substituent_side(a, b, pos, nx, ny)
                if side is None:
                    for s in (0.5, -0.5):
                        p.drawLine(
                            QtCore.QPointF(p1.x() + s * nx * gap, p1.y() + s * ny * gap),
                            QtCore.QPointF(p2.x() + s * nx * gap, p2.y() + s * ny * gap),
                        )
                else:
                    p.drawLine(p1, p2)
                    shrink = 0.14 * d
                    ux, uy = dx / d, dy / d
                    s1 = 0.0 if a in radius else shrink
                    s2 = 0.0 if b in radius else shrink
                    p.drawLine(
                        QtCore.QPointF(p1.x() + ux * s1 + side * nx * gap, p1.y() + uy * s1 + side * ny * gap),
                        QtCore.QPointF(p2.x() - ux * s2 + side * nx * gap, p2.y() - uy * s2 + side * ny * gap),
                    )

    def _substituent_side(self, a, b, pos, nx, ny):
        mol = self.controller.mol
        score = 0.0
        count = 0
        mx, my = 0.5 * (pos[a].x() + pos[b].x()), 0.5 * (pos[a].y() + pos[b].y())
        for idx in (a, b):
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                j = int(nb.GetIdx())
                if j in (a, b) or j not in pos:
                    continue
                score += (pos[j].x() - mx) * nx + (pos[j].y() - my) * ny
                count += 1
        if count == 0:
            return None
        return 1.0 if score > 0 else -1.0

    def _paint_labels(self, p, mol, render, pos, font, metrics):
        small = QtGui.QFont(font)
        small.setPixelSize(max(8, int(font.pixelSize() * 0.68)))
        small_m = QtGui.QFontMetricsF(small)
        for idx, (sym, hcount, charge) in render.get("labels", {}).items():
            if idx not in pos:
                continue
            c = pos[idx]
            color = QtGui.QColor(ELEMENT_COLORS.get(sym.upper(), "#334E68"))
            atom = mol.GetAtomWithIdx(idx)
            left = False
            if hcount:
                sx = sum(pos[int(nb.GetIdx())].x() - c.x() for nb in atom.GetNeighbors() if int(nb.GetIdx()) in pos)
                left = sx > 0.5
            sym_w = metrics.horizontalAdvance(sym) if hasattr(metrics, "horizontalAdvance") else metrics.width(sym)
            h = metrics.height()
            sym_rect = QtCore.QRectF(c.x() - sym_w / 2.0, c.y() - h / 2.0, sym_w, h)
            label_right = sym_rect.right()
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor("#FFFFFF"))
            p.drawEllipse(c, 0.55 * h, 0.55 * h)
            p.setFont(font)
            p.setPen(color)
            p.drawText(sym_rect, QtCore.Qt.AlignCenter, sym)
            if hcount:
                h_w = metrics.horizontalAdvance("H") if hasattr(metrics, "horizontalAdvance") else metrics.width("H")
                n_txt = str(hcount) if hcount > 1 else ""
                n_w = (small_m.horizontalAdvance(n_txt) if hasattr(small_m, "horizontalAdvance") else small_m.width(n_txt)) if n_txt else 0.0
                if left:
                    hx = sym_rect.left() - h_w - n_w
                else:
                    hx = sym_rect.right()
                    label_right = hx + h_w + n_w
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor("#FFFFFF"))
                p.drawRect(QtCore.QRectF(hx, sym_rect.top() + 2, h_w + n_w, h - 2))
                p.setPen(color)
                p.setFont(font)
                p.drawText(QtCore.QRectF(hx, sym_rect.top(), h_w, h), QtCore.Qt.AlignCenter, "H")
                if n_txt:
                    p.setFont(small)
                    p.drawText(QtCore.QRectF(hx + h_w, sym_rect.top() + h * 0.35, n_w + 2, h * 0.8), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, n_txt)
            if charge:
                txt = ("+" if charge > 0 else "−") if abs(charge) == 1 else f"{abs(charge)}{'+' if charge > 0 else chr(0x2212)}"
                p.setFont(small)
                p.setPen(color)
                p.drawText(
                    QtCore.QRectF(label_right, sym_rect.top() - h * 0.38, 26, h * 0.8),
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                    txt,
                )

    def _paint_extras(self, p, mol, render, pos, font_px):
        tiny = QtGui.QFont(self.font())
        tiny.setPixelSize(max(8, int(font_px * 0.58)))
        p.setFont(tiny)
        if self.show_stereo:
            for idx, lab in render.get("cip", {}).items():
                if idx not in pos:
                    continue
                c = pos[idx]
                txt = f"({lab})" if lab in ("R", "S") else "?"
                color = QtGui.QColor("#7C3AED") if lab in ("R", "S") else QtGui.QColor("#D97706")
                ux, uy = _widest_gap_direction(
                    [pos[int(nb.GetIdx())] - c for nb in mol.GetAtomWithIdx(idx).GetNeighbors() if int(nb.GetIdx()) in pos]
                )
                dist = 0.45 * self.scale
                tx, ty = c.x() + ux * dist, c.y() + uy * dist
                p.setPen(color)
                p.drawText(QtCore.QRectF(tx - 20, ty - 8, 40, 16), QtCore.Qt.AlignCenter, txt)
        if self.show_atom_numbers:
            p.setPen(QtGui.QColor("#9AA5B1"))
            for idx, c in pos.items():
                p.drawText(QtCore.QRectF(c.x() - 30, c.y() - font_px * 1.25, 26, 14), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, str(idx + 1))

    def _paint_ghost(self, p):
        g = self._ghost
        if not g:
            return
        color = QtGui.QColor(37, 99, 235, 150)
        pen = QtGui.QPen(color, max(1.4, self.scale * 0.05))
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        for a, b in g.get("segments", []):
            p.drawLine(self.to_screen(*a), self.to_screen(*b))
        label = g.get("label")
        if label and g.get("segments"):
            end = self.to_screen(*g["segments"][-1][1])
            f = QtGui.QFont(self.font())
            f.setPixelSize(max(10, int(self.scale * 0.42)))
            f.setWeight(QtGui.QFont.DemiBold)
            p.setFont(f)
            p.setPen(color)
            p.drawText(QtCore.QRectF(end.x() - 20, end.y() - 12, 40, 24), QtCore.Qt.AlignCenter, label)


# --------------------------------------------------------------------------
# Composite widget
# --------------------------------------------------------------------------

STYLE = """
QWidget#LigandSketcher { background: #F4F6F9; }
QFrame#SketchCard { background: #FFFFFF; border: 1px solid #DDE3EA; border-radius: 8px; }
QFrame#CanvasFrame { background: #FFFFFF; border: 1px solid #CBD2D9; border-radius: 8px; }
QLabel#CardTitle { color: #1E3A5F; font-weight: 600; font-size: 12px; letter-spacing: 0.3px; }
QLabel#PropKey { color: #7B8794; font-size: 11px; }
QLabel#PropVal { color: #1F2933; font-size: 12px; font-weight: 600; }
QLabel#Hint { color: #7B8794; font-size: 11px; }
QToolButton#Tool {
    background: transparent; border: 1px solid transparent; border-radius: 6px;
    min-width: 34px; min-height: 34px; max-width: 34px; max-height: 34px;
}
QToolButton#Tool:hover { background: #EEF2F7; border-color: #D5DDE7; }
QToolButton#Tool:checked { background: #DCE8F7; border-color: #1E3A5F; }
QToolButton#Elem {
    background: #FFFFFF; border: 1px solid #DDE3EA; border-radius: 6px;
    min-width: 30px; min-height: 28px; font-weight: 700; font-size: 13px;
}
QToolButton#Elem:hover { background: #EEF2F7; }
QToolButton#Elem:checked { background: #DCE8F7; border: 1px solid #1E3A5F; }
QToolButton#Frag {
    background: #FFFFFF; border: 1px solid #DDE3EA; border-radius: 5px;
    padding: 3px 4px; font-size: 11px; color: #243447; min-width: 44px;
}
QToolButton#Frag:hover { background: #EEF2F7; }
QToolButton#Frag:checked { background: #DCE8F7; border: 1px solid #1E3A5F; }
QToolButton#Action {
    background: #FFFFFF; border: 1px solid #DDE3EA; border-radius: 6px;
    padding: 4px 8px; color: #243447; font-size: 12px;
}
QToolButton#Action:hover { background: #EEF2F7; }
QToolButton#Action:disabled { color: #B0B8C1; }
QToolButton#Action::menu-indicator { image: none; }
QLineEdit#Smiles {
    background: #FFFFFF; border: 1px solid #CBD2D9; border-radius: 6px; padding: 5px 7px;
    font-family: Menlo, Consolas, monospace; font-size: 12px;
}
QLineEdit#Smiles:focus { border-color: #1E3A5F; }
QPushButton#Primary {
    background: #1E3A5F; color: #FFFFFF; border: none; border-radius: 7px;
    padding: 9px 14px; font-weight: 600; font-size: 13px;
}
QPushButton#Primary:hover { background: #274B78; }
QPushButton#Primary:disabled { background: #9AA5B1; }
QCheckBox { color: #3E4C59; font-size: 12px; }
QLabel#Status { font-size: 12px; padding: 2px 4px; }
QScrollArea#PanelScroll, QWidget#PanelInner { background: transparent; }
"""


class LigandSketcherWidget(QtWidgets.QWidget):
    """Maestro-style 2D sketcher. Host wires the ``on_*`` callback attributes."""

    MAX_HISTORY = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LigandSketcher")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.mol = None
        self.render = build_render_model(None)
        self.tool = "single"
        self.element = "C"
        self.fragment = None
        self._undo = []
        self._redo = []
        self._drag_snapshot = None
        # Host callbacks
        self.on_molecule_changed = None
        self.on_build_3d = None
        self.on_import_active = None
        self.on_import_file = None
        self.on_paste_clipboard = None
        self.on_open_web_drawer = None
        self.on_protonate = None
        self._build_ui()
        self.setStyleSheet(STYLE)
        self._set_tool("single")
        self._refresh_panel()

    def keyPressEvent(self, event):
        # Keep Return (from the SMILES field) and Esc from reaching the host
        # dialog, where they would click its default button or close it.
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Escape):
            if event.key() == QtCore.Qt.Key_Escape:
                self.canvas.selection = set()
                self._set_tool("select")
                self.canvas.setFocus()
            event.accept()
            return
        super().keyPressEvent(event)

    def sizeHint(self):
        # The canvas and the scrolling side panel absorb extra space, so report a
        # compact hint; host dialogs then never need an outer scroll area.
        return QtCore.QSize(960, 560)

    # ------------------------------------------------------------------ UI
    def _card(self, title=None):
        frame = QtWidgets.QFrame()
        frame.setObjectName("SketchCard")
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        if title:
            lbl = QtWidgets.QLabel(title.upper())
            lbl.setObjectName("CardTitle")
            lay.addWidget(lbl)
        return frame, lay

    def _action_button(self, text, icon_key, tip, slot, shortcut_text=""):
        btn = QtWidgets.QToolButton()
        btn.setObjectName("Action")
        btn.setText(text)
        btn.setIcon(action_icon(icon_key))
        btn.setIconSize(QtCore.QSize(18, 18))
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon if text else QtCore.Qt.ToolButtonIconOnly)
        btn.setToolTip(tip + (f"  ({shortcut_text})" if shortcut_text else ""))
        if slot is not None:
            btn.clicked.connect(slot)
        return btn

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)

        # ---- top action bar
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(6)
        self.btn_import = self._action_button("Import", "import", "Load a ligand", None)
        self.btn_import.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self.btn_import)
        self._menu_actions = {}
        for key, text in (
            ("active", "Active ligand"),
            ("file", "From file (SDF, MOL, MOL2, PDB, SMILES)..."),
            ("clipboard", "Paste SMILES from clipboard"),
            (None, None),
            ("ketcher", "Draw in Ketcher (web)..."),
            ("local_jsme", "Draw in JSME (local)..."),
            ("jsme", "Draw in JSME (web)..."),
        ):
            if key is None:
                menu.addSeparator()
                continue
            act = menu.addAction(text)
            act.triggered.connect(lambda _c=False, k=key: self._on_import(k))
            self._menu_actions[key] = act
        self.btn_import.setMenu(menu)
        bar.addWidget(self.btn_import)
        bar.addSpacing(6)
        self.btn_undo = self._action_button("", "undo", "Undo", self.undo, "Ctrl+Z")
        self.btn_redo = self._action_button("", "redo", "Redo", self.redo, "Ctrl+Shift+Z")
        bar.addWidget(self.btn_undo)
        bar.addWidget(self.btn_redo)
        bar.addSpacing(6)
        bar.addWidget(self._action_button("Clean", "clean", "Clean up the 2D layout", self.clean_layout))
        bar.addWidget(self._action_button("Fit", "fit", "Fit the molecule in the view", self.fit_view))
        bar.addWidget(self._action_button("Clear", "clear", "Clear the canvas", self.clear))
        bar.addSpacing(6)
        self.btn_protonate = self._action_button(
            "Protonate", "hydrogen", "Set protonation/charges at the workflow pH", self._on_protonate
        )
        bar.addWidget(self.btn_protonate)
        bar.addStretch(1)
        self.btn_view = QtWidgets.QToolButton()
        self.btn_view.setObjectName("Action")
        self.btn_view.setText("View")
        self.btn_view.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        vmenu = QtWidgets.QMenu(self.btn_view)
        self.act_numbers = vmenu.addAction("Atom numbers")
        self.act_numbers.setCheckable(True)
        self.act_numbers.toggled.connect(self._on_view_changed)
        self.act_stereo = vmenu.addAction("Stereo labels (R/S)")
        self.act_stereo.setCheckable(True)
        self.act_stereo.setChecked(True)
        self.act_stereo.toggled.connect(self._on_view_changed)
        self.act_carbons = vmenu.addAction("Show carbon labels")
        self.act_carbons.setCheckable(True)
        self.act_carbons.toggled.connect(self._on_view_changed)
        self.btn_view.setMenu(vmenu)
        bar.addWidget(self.btn_view)
        root.addLayout(bar)

        # ---- element strip
        self.tool_group = QtWidgets.QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self._tool_buttons = {}
        estrip = QtWidgets.QHBoxLayout()
        estrip.setSpacing(4)
        self._elem_buttons = {}
        for sym in PALETTE_ELEMENTS:
            btn = QtWidgets.QToolButton()
            btn.setObjectName("Elem")
            btn.setText(sym)
            btn.setCheckable(True)
            btn.setStyleSheet(f"QToolButton#Elem {{ color: {ELEMENT_COLORS.get(sym.upper(), '#243447')}; }}")
            hot = {"Cl": "L", "Br": "B"}.get(sym, sym)
            btn.setToolTip(f"{sym}: click an atom to change it, click empty space to place it  (key {hot})")
            btn.clicked.connect(lambda _c=False, s=sym: self.set_element_tool(s))
            self.tool_group.addButton(btn)
            self._elem_buttons[sym] = btn
            estrip.addWidget(btn)
        self.btn_periodic = QtWidgets.QToolButton()
        self.btn_periodic.setObjectName("Elem")
        self.btn_periodic.setText("…")
        self.btn_periodic.setToolTip("Other elements (periodic table)")
        self.btn_periodic.clicked.connect(self._open_periodic_table)
        estrip.addWidget(self.btn_periodic)
        estrip.addSpacing(12)
        self.lbl_tool_hint = QtWidgets.QLabel("")
        self.lbl_tool_hint.setObjectName("Hint")
        estrip.addWidget(self.lbl_tool_hint, 1)
        root.addLayout(estrip)

        # ---- body: tools | canvas | panel
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(8)
        tools_frame, tools_lay = self._card()
        tools_lay.setContentsMargins(5, 6, 5, 6)
        tools_lay.setSpacing(2)
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(3)
        tool_specs = [
            ("select", "Select / move  (Esc)\nDrag atoms or bonds to move; drag on empty space to box-select."),
            ("erase", "Erase  (Del)\nClick an atom or bond to delete it."),
            ("single", "Single bond  (1)\nClick an atom to grow, click a bond to cycle its order, drag to draw."),
            ("double", "Double bond  (2)"),
            ("triple", "Triple bond  (3)"),
            ("chain", "Chain\nClick an atom to grow a carbon; drag to point it."),
            ("wedge", "Wedge bond (up)\nClick a bond at its stereocenter end; click again to flip."),
            ("hash", "Hashed bond (down)\nClick a bond at its stereocenter end; click again to flip."),
            ("charge_plus", "Increase charge  (+)"),
            ("charge_minus", "Decrease charge  (−)"),
            ("invert", "Invert stereocenter\nClick a defined stereocenter to flip R/S."),
        ]
        row = col = 0
        for key, tip in tool_specs:
            self._add_tool_button(grid, key, tip, row, col)
            col += 1
            if col == 2:
                row, col = row + 1, 0
        if col:
            row, col = row + 1, 0
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color: #E4E9F0;")
        grid.addWidget(sep, row, 0, 1, 2)
        row += 1
        for key, name in RING_TOOLS:
            tip = f"{name}\nClick an atom to attach, a bond to fuse, empty space to place. Shift+click an atom for spiro."
            self._add_tool_button(grid, key, tip, row, col)
            col += 1
            if col == 2:
                row, col = row + 1, 0
        tools_lay.addLayout(grid)
        tools_lay.addStretch(1)
        body.addWidget(tools_frame, 0)

        canvas_frame = QtWidgets.QFrame()
        canvas_frame.setObjectName("CanvasFrame")
        cf = QtWidgets.QVBoxLayout(canvas_frame)
        cf.setContentsMargins(1, 1, 1, 1)
        self.canvas = SketcherCanvas(self, canvas_frame)
        cf.addWidget(self.canvas)
        body.addWidget(canvas_frame, 1)

        panel = QtWidgets.QVBoxLayout()
        panel.setSpacing(8)
        props, props_lay = self._card("Molecule")
        self.lbl_formula = QtWidgets.QLabel("—")
        self.lbl_formula.setStyleSheet("font-size: 16px; font-weight: 600; color: #1F2933;")
        self.lbl_formula.setTextFormat(QtCore.Qt.RichText)
        props_lay.addWidget(self.lbl_formula)
        pgrid = QtWidgets.QGridLayout()
        pgrid.setHorizontalSpacing(10)
        pgrid.setVerticalSpacing(3)
        self._prop_labels = {}
        for k, (key, title) in enumerate(
            (
                ("mw", "MW"),
                ("charge", "Charge"),
                ("clogp", "cLogP"),
                ("tpsa", "TPSA"),
                ("hbd", "HBD"),
                ("hba", "HBA"),
                ("rotb", "Rot. bonds"),
                ("atoms", "Heavy atoms"),
            )
        ):
            kl = QtWidgets.QLabel(title)
            kl.setObjectName("PropKey")
            vl = QtWidgets.QLabel("—")
            vl.setObjectName("PropVal")
            r, c = divmod(k, 2)
            pgrid.addWidget(kl, 2 * r, c)
            pgrid.addWidget(vl, 2 * r + 1, c)
            self._prop_labels[key] = vl
        props_lay.addLayout(pgrid)
        self.lbl_flags = QtWidgets.QLabel("")
        self.lbl_flags.setWordWrap(True)
        self.lbl_flags.setTextFormat(QtCore.Qt.RichText)
        props_lay.addWidget(self.lbl_flags)
        panel.addWidget(props)

        smi_card, smi_lay = self._card("SMILES")
        srow = QtWidgets.QHBoxLayout()
        srow.setSpacing(4)
        self.edit_smiles = QtWidgets.QLineEdit()
        self.edit_smiles.setObjectName("Smiles")
        self.edit_smiles.setPlaceholderText("Type or paste SMILES, press Enter")
        self.edit_smiles.returnPressed.connect(self._on_smiles_entered)
        srow.addWidget(self.edit_smiles, 1)
        self.btn_copy = self._action_button("", "copy", "Copy SMILES", self._copy_smiles)
        srow.addWidget(self.btn_copy)
        smi_lay.addLayout(srow)
        panel.addWidget(smi_card)

        frag_card, frag_lay = self._card("Groups")
        fgrid = QtWidgets.QGridLayout()
        fgrid.setSpacing(3)
        self._frag_buttons = {}
        for k, (label, name, smi) in enumerate(FRAGMENT_LIBRARY):
            btn = QtWidgets.QToolButton()
            btn.setObjectName("Frag")
            btn.setText(label)
            btn.setCheckable(True)
            btn.setToolTip(f"{name}  {smi.replace('[*]', '-')}\nClick an atom to attach it.")
            btn.clicked.connect(lambda _c=False, s=smi, n=name: self.set_fragment_tool(s, n))
            self.tool_group.addButton(btn)
            self._frag_buttons[smi] = btn
            r, c = divmod(k, 5)
            fgrid.addWidget(btn, r, c)
        frag_lay.addLayout(fgrid)
        panel.addWidget(frag_card)
        panel.addStretch(1)
        panel.setContentsMargins(0, 0, 4, 0)
        panel_inner = QtWidgets.QWidget()
        panel_inner.setObjectName("PanelInner")
        panel_inner.setLayout(panel)
        panel_scroll = QtWidgets.QScrollArea()
        panel_scroll.setObjectName("PanelScroll")
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel_scroll.setWidget(panel_inner)
        panel_scroll.setMinimumHeight(120)
        panel_scroll.setProperty("leonardoVerticalOnly", True)

        build_card, build_lay = self._card("3D ligand")
        self.cb_include_h = QtWidgets.QCheckBox("Add hydrogens at workflow pH")
        self.cb_include_h.setChecked(True)
        self.cb_relax = QtWidgets.QCheckBox("Relax geometry (MMFF, UFF fallback)")
        self.cb_relax.setChecked(True)
        build_lay.addWidget(self.cb_include_h)
        build_lay.addWidget(self.cb_relax)
        self.btn_build = QtWidgets.QPushButton("Build 3D && Use as Ligand")
        self.btn_build.setObjectName("Primary")
        self.btn_build.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_build.setAutoDefault(False)
        self.btn_build.setDefault(False)
        self.btn_build.clicked.connect(self._on_build)
        build_lay.addWidget(self.btn_build)
        panel_w = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(panel_w)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(8)
        side.addWidget(panel_scroll, 1)
        side.addWidget(build_card, 0)
        panel_w.setFixedWidth(310)
        body.addWidget(panel_w, 0)
        root.addLayout(body, 1)

        # ---- status line
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setObjectName("Status")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        for seq, fn in (
            ("Ctrl+Z", self.undo),
            ("Ctrl+Shift+Z", self.redo),
            ("Ctrl+Y", self.redo),
            ("Ctrl+A", self.select_all),
        ):
            sc = _make_shortcut(seq, self, fn)
            if sc is not None:
                sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)

    def _add_tool_button(self, grid, key, tip, row, col):
        btn = QtWidgets.QToolButton()
        btn.setObjectName("Tool")
        btn.setCheckable(True)
        btn.setIcon(tool_icon(key))
        btn.setIconSize(QtCore.QSize(24, 24))
        btn.setToolTip(tip)
        btn.clicked.connect(lambda _c=False, k=key: self._set_tool(k))
        self.tool_group.addButton(btn)
        self._tool_buttons[key] = btn
        grid.addWidget(btn, row, col)

    # --------------------------------------------------------------- tools
    TOOL_HINTS = {
        "select": "Drag to move atoms; drag on empty space to box-select. Del removes the selection.",
        "erase": "Click an atom or bond to delete it.",
        "single": "Click an atom to grow, a bond to cycle single/double/triple, empty space to start. Drag to draw.",
        "double": "Click an atom to grow a double bond, or a bond to make it double.",
        "triple": "Click an atom to grow a triple bond, or a bond to make it triple.",
        "chain": "Click an atom to extend the chain; drag to aim.",
        "wedge": "Click a bond near its stereocenter to wedge it; click again to flip.",
        "hash": "Click a bond near its stereocenter to hash it; click again to flip.",
        "charge_plus": "Click an atom to raise its formal charge.",
        "charge_minus": "Click an atom to lower its formal charge.",
        "invert": "Click a stereocenter to invert it.",
    }

    def _set_tool(self, key):
        self.tool = str(key)
        btn = self._tool_buttons.get(self.tool)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        if self.tool.startswith("ring") or self.tool == "benzene":
            name = dict(RING_TOOLS).get(self.tool, "Ring")
            hint = f"{name}: click an atom to attach, a bond to fuse, empty space to place. Shift+click = spiro."
        else:
            hint = self.TOOL_HINTS.get(self.tool, "")
        self.lbl_tool_hint.setText(hint)
        cursor = QtCore.Qt.ArrowCursor if self.tool == "select" else QtCore.Qt.CrossCursor
        self.canvas.setCursor(cursor)
        self.canvas.refresh_hover()
        self.canvas.update()

    def set_element_tool(self, symbol):
        sym = normalize_symbol(symbol)
        self.element = sym
        self.tool = "element"
        btn = self._elem_buttons.get(sym)
        if btn is not None:
            btn.setChecked(True)
        else:
            checked = self.tool_group.checkedButton()
            if checked is not None:
                self.tool_group.setExclusive(False)
                checked.setChecked(False)
                self.tool_group.setExclusive(True)
        self.lbl_tool_hint.setText(
            f"{sym}: click an atom to change it (click again to grow a {sym}), empty space to place, drag to draw."
        )
        self.canvas.setCursor(QtCore.Qt.CrossCursor)
        self.canvas.refresh_hover()
        self.canvas.update()

    def set_fragment_tool(self, smiles, name=""):
        self.fragment = str(smiles)
        self.tool = "fragment"
        btn = self._frag_buttons.get(self.fragment)
        if btn is not None:
            btn.setChecked(True)
        self.lbl_tool_hint.setText(f"{name or 'Group'}: click an atom to attach it, or empty space to place it.")
        self.canvas.setCursor(QtCore.Qt.CrossCursor)
        self.canvas.refresh_hover()
        self.canvas.update()

    def tool_draws_bonds(self):
        return self.tool in ("single", "double", "triple", "chain", "element")

    def _tool_order(self):
        return {"double": "double", "triple": "triple"}.get(self.tool, "single")

    def ghost_label(self):
        if self.tool == "element" and self.element != "C":
            return self.element
        return None

    def ghost_for(self, atom, bond):
        """Preview geometry for the current tool at the hovered atom/bond."""
        mol = self.mol
        if mol is None:
            return None
        try:
            if atom is not None:
                if self.tool in ("single", "double", "triple", "chain") or (
                    self.tool == "element" and mol.GetAtomWithIdx(atom).GetSymbol() == self.element
                ):
                    start = atom_xy(mol, atom)
                    end = new_atom_position(mol, atom, order=self._tool_order())
                    return {"segments": [(start, end)], "label": self.ghost_label()}
                if self.tool == "fragment":
                    start = atom_xy(mol, atom)
                    end = new_atom_position(mol, atom)
                    return {"segments": [(start, end)], "label": None}
                if self.tool.startswith("ring") or self.tool == "benzene":
                    n, _arom = ring_spec(self.tool)
                    verts = ring_vertices_attached(mol, atom, n)
                    segs = [(atom_xy(mol, atom), verts[0])]
                    segs += [(verts[k], verts[(k + 1) % n]) for k in range(n)]
                    return {"segments": segs, "label": None}
            if bond is not None and (self.tool.startswith("ring") or self.tool == "benzene"):
                n, _arom = ring_spec(self.tool)
                verts = ring_vertices_fused(mol, bond[0], bond[1], n)
                return {"segments": [(verts[k], verts[k + 1]) for k in range(1, n - 1)] + [(verts[-1], verts[0])], "label": None}
        except Exception:
            return None
        return None

    # ------------------------------------------------------- model / history
    def _snapshot(self):
        return None if self.mol is None else Chem.Mol(self.mol)

    def _push_undo(self):
        self._undo.append(self._snapshot())
        if len(self._undo) > self.MAX_HISTORY:
            self._undo = self._undo[-self.MAX_HISTORY:]
        self._redo = []

    def _apply(self, new_mol, message=None, notify=True, record=True):
        if record:
            self._push_undo()
        self.mol = new_mol
        self._model_changed(notify=notify)
        if message:
            self.show_status(message, "ok")

    def _model_changed(self, notify=True):
        self.render = build_render_model(self.mol, show_carbons=self.act_carbons.isChecked())
        n = 0 if self.mol is None else int(self.mol.GetNumAtoms())
        self.canvas.selection = {i for i in self.canvas.selection if i < n}
        self.canvas.refresh_hover()
        self.canvas.ensure_visible()
        self.canvas.update()
        self._refresh_panel()
        if notify and callable(self.on_molecule_changed):
            try:
                self.on_molecule_changed(self.current_mol())
            except Exception:
                pass

    def _edit(self, fn, *args, message=None, **kwargs):
        try:
            result = fn(*args, **kwargs)
        except SketchError as err:
            self.show_status(str(err), "error")
            return None
        except Exception as err:  # RDKit edge cases
            self.show_status(_friendly_error(err), "error")
            return None
        mol = result[0] if isinstance(result, tuple) else result
        self._apply(mol, message=message)
        return result

    def current_mol(self):
        return None if self.mol is None else Chem.Mol(self.mol)

    def set_molecule(self, mol, record=True, fit=None):
        """Load a molecule from the host (file, SMILES, protonation, 3D build)."""
        if mol is None:
            new = None
        else:
            new = strip_hydrogens(mol)
            same_graph = (
                self.mol is not None
                and int(self.mol.GetNumAtoms()) == int(new.GetNumAtoms())
                and all(
                    self.mol.GetAtomWithIdx(i).GetAtomicNum() == new.GetAtomWithIdx(i).GetAtomicNum()
                    for i in range(int(new.GetNumAtoms()))
                )
            )
            if same_graph and (conformer_is_3d(new) or int(new.GetNumConformers()) <= 0):
                new.RemoveAllConformers()
                conf = Chem.Conformer(self.mol.GetConformer())
                new.AddConformer(conf, assignId=True)
                if fit is None:
                    fit = False
            else:
                ensure_sketch_coords(new)
        if record:
            self._push_undo()
        self.mol = new
        self.canvas.selection = set()
        self._model_changed(notify=False)
        if fit is None or fit:
            self.canvas.fit_view()

    def undo(self):
        if not self._undo:
            self.show_status("Nothing to undo.", "info")
            return
        self._redo.append(self._snapshot())
        self.mol = self._undo.pop()
        self._model_changed()
        self.show_status("Undone.", "info")

    def redo(self):
        if not self._redo:
            self.show_status("Nothing to redo.", "info")
            return
        self._undo.append(self._snapshot())
        self.mol = self._redo.pop()
        self._model_changed()
        self.show_status("Redone.", "info")

    def clear(self):
        if self.mol is None:
            return
        self._apply(None, message="Canvas cleared (Ctrl+Z to restore).")
        self.canvas.fit_view()

    def clean_layout(self):
        if self.mol is None:
            return
        self._edit(clean_2d, self.mol, message="2D layout cleaned.")
        self.canvas.fit_view()

    def fit_view(self):
        self.canvas.fit_view()

    def select_all(self):
        if self.mol is not None:
            self.canvas.selection = set(range(int(self.mol.GetNumAtoms())))
            self._set_tool("select")
            self.canvas.update()

    def selection_changed(self):
        n = len(self.canvas.selection)
        if n:
            self.show_status(f"{n} atom(s) selected. Drag to move, Del to delete.", "info")

    # ------------------------------------------------------ canvas callbacks
    def canvas_drag_started(self, atom, bond):
        if self.tool == "select" and (atom is not None or bond is not None):
            self._drag_snapshot = self._snapshot()

    def canvas_drag_finished(self):
        if self._drag_snapshot is not None:
            self._undo.append(self._drag_snapshot)
            self._redo = []
            self._drag_snapshot = None
            self._model_changed()

    def move_by(self, indices, dx, dy):
        if self.mol is None or not indices:
            return
        move_atoms(self.mol, indices, dx, dy)
        self.render = build_render_model(self.mol, show_carbons=self.act_carbons.isChecked())

    def _grow_symbol(self):
        return self.element if self.tool == "element" else "C"

    def grow_to(self, anchor, xy):
        sym = self._grow_symbol()
        if sym == "H":
            self.show_status("Hydrogens are implicit; use Protonate or the charge tools.", "info")
            return
        self._edit(add_atom, self.mol, anchor, sym, self._tool_order(), xy=xy)

    def connect_to(self, a, b):
        self._edit(connect_atoms, self.mol, a, b, self._tool_order(), message="Bond added.")

    def canvas_click(self, atom, bond, xy, shift=False):
        tool = self.tool
        mol = self.mol
        if tool == "select":
            if atom is not None:
                if shift:
                    self.canvas.selection ^= {atom}
                else:
                    self.canvas.selection = {atom}
            elif bond is not None:
                self.canvas.selection = set(bond) if not shift else self.canvas.selection | set(bond)
            else:
                self.canvas.selection = set()
            self.canvas.update()
            return
        if tool == "erase":
            if atom is not None:
                self._edit(delete_atoms, mol, [atom])
            elif bond is not None:
                self._edit(delete_bond, mol, bond[0], bond[1])
            return
        if tool in ("single", "double", "triple", "chain"):
            order = self._tool_order()
            if atom is not None:
                self._edit(add_atom, mol, atom, "C", order)
            elif bond is not None:
                if tool == "single":
                    self._edit(cycle_bond_order, mol, bond[0], bond[1])
                elif tool == "chain":
                    pass
                else:
                    self._edit(set_bond_order, mol, bond[0], bond[1], order)
            else:
                if tool == "chain":
                    self._edit(add_atom, mol, None, "C", xy=xy)
                else:
                    self._edit(add_chain_bond, mol, xy, order)
            return
        if tool == "element":
            sym = self.element
            if atom is not None:
                if sym == "H":
                    self.show_status("Hydrogens are implicit; use Protonate or the charge tools.", "info")
                    return
                if mol.GetAtomWithIdx(atom).GetSymbol() == sym:
                    self._edit(add_atom, mol, atom, sym)
                else:
                    self._edit(set_element, mol, atom, sym)
            elif bond is None:
                if sym == "H":
                    self.show_status("Hydrogens are implicit; draw heavy atoms.", "info")
                    return
                self._edit(add_atom, mol, None, sym, xy=xy)
            return
        if tool in ("charge_plus", "charge_minus"):
            if atom is not None:
                self._edit(change_charge, mol, atom, 1 if tool == "charge_plus" else -1)
            return
        if tool in ("wedge", "hash"):
            if bond is not None:
                self._apply_wedge(bond, xy, tool)
            elif atom is not None:
                self._edit(invert_center, mol, atom, message="Stereocenter inverted.")
            return
        if tool == "invert":
            if atom is not None:
                self._edit(invert_center, mol, atom, message="Stereocenter inverted.")
            return
        if tool == "fragment" and self.fragment:
            if atom is not None:
                self._edit(attach_fragment, mol, atom, self.fragment)
            elif bond is None:
                result = self._edit(attach_fragment, mol, None, self.fragment)
                if result is not None:
                    self._place_new_atoms_at(mol, xy)
            return
        if tool.startswith("ring") or tool == "benzene":
            if atom is not None:
                self._edit(attach_ring, mol, atom, tool, spiro=bool(shift))
            elif bond is not None:
                self._edit(fuse_ring, mol, bond[0], bond[1], tool)
            else:
                self._edit(place_ring, mol, xy, tool)
            return

    def _place_new_atoms_at(self, old_mol, xy):
        start = 0 if old_mol is None else int(old_mol.GetNumAtoms())
        idxs = list(range(start, int(self.mol.GetNumAtoms())))
        if not idxs:
            return
        x0, y0 = atom_xy(self.mol, idxs[0])
        move_atoms(self.mol, idxs, xy[0] - x0, xy[1] - y0)
        self._model_changed()

    def _apply_wedge(self, bond, xy, tool):
        mol = self.mol
        a, b = bond
        centers = potential_stereocenters(mol)
        da = math.hypot(atom_xy(mol, a)[0] - xy[0], atom_xy(mol, a)[1] - xy[1])
        db = math.hypot(atom_xy(mol, b)[0] - xy[0], atom_xy(mol, b)[1] - xy[1])
        order = [a, b] if da <= db else [b, a]
        begin = next((x for x in order if x in centers), None)
        if begin is None:
            self.show_status("Neither end of that bond is a stereocenter.", "error")
            return
        end = b if begin == a else a
        kind = "wedge" if tool == "wedge" else "hash"
        current = self.render.get("wedges", {}).get((begin, end))
        if current == kind:
            kind = "hash" if kind == "wedge" else "wedge"
        self._edit(set_wedge, mol, begin, end, kind)

    def canvas_key(self, event, atom, bond):
        key = event.key()
        text = str(event.text() or "")
        mods = event.modifiers()
        if mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier):
            return False
        if key in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            if self.canvas.selection:
                self._edit(delete_atoms, self.mol, sorted(self.canvas.selection))
                self.canvas.selection = set()
            elif atom is not None:
                self._edit(delete_atoms, self.mol, [atom])
            elif bond is not None:
                self._edit(delete_bond, self.mol, bond[0], bond[1])
            return True
        if key == QtCore.Qt.Key_Escape:
            self.canvas.selection = set()
            self._set_tool("select")
            return True
        if text in ("+", "=") and atom is not None:
            self._edit(change_charge, self.mol, atom, 1)
            return True
        if text in ("-", "_") and atom is not None:
            self._edit(change_charge, self.mol, atom, -1)
            return True
        if text in ("1", "2", "3"):
            order = {"1": "single", "2": "double", "3": "triple"}[text]
            if bond is not None:
                self._edit(set_bond_order, self.mol, bond[0], bond[1], order)
            elif atom is not None:
                self._edit(add_atom, self.mol, atom, "C", order)
            else:
                self._set_tool(order)
            return True
        hot = {"c": "C", "n": "N", "o": "O", "s": "S", "p": "P", "f": "F", "h": "H", "i": "I", "l": "Cl", "b": "Br"}
        sym = hot.get(text.lower())
        if sym is not None:
            if atom is not None:
                if sym == "H":
                    return True
                self._edit(set_element, self.mol, atom, sym)
            else:
                self.set_element_tool(sym)
            return True
        return False

    def canvas_context_menu(self, atom, bond, global_pos):
        if self.mol is None or (atom is None and bond is None):
            return
        menu = QtWidgets.QMenu(self)
        if atom is not None:
            for sym in ("C", "N", "O", "S", "F", "Cl"):
                menu.addAction(f"Change to {sym}", lambda s=sym: self._edit(set_element, self.mol, atom, s))
            menu.addSeparator()
            menu.addAction("Charge +1", lambda: self._edit(change_charge, self.mol, atom, 1))
            menu.addAction("Charge −1", lambda: self._edit(change_charge, self.mol, atom, -1))
            if self.mol.GetAtomWithIdx(atom).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                menu.addAction("Invert stereocenter", lambda: self._edit(invert_center, self.mol, atom))
            menu.addSeparator()
            menu.addAction("Delete atom", lambda: self._edit(delete_atoms, self.mol, [atom]))
        else:
            a, b = bond
            for order in ("single", "double", "triple"):
                menu.addAction(f"{order.title()} bond", lambda o=order: self._edit(set_bond_order, self.mol, a, b, o))
            menu.addSeparator()
            menu.addAction("Delete bond", lambda: self._edit(delete_bond, self.mol, a, b))
        _exec_menu(menu, global_pos)

    # ------------------------------------------------------------ panel / io
    def _refresh_panel(self):
        info = molecule_summary(self.mol)
        empty = self.mol is None or int(self.mol.GetNumAtoms()) == 0
        self.lbl_formula.setText("—" if empty else formula_html(info["formula"]))
        vals = {
            "mw": f"{info['mw']:.1f}",
            "charge": f"{info['charge']:+d}" if info["charge"] else "0",
            "clogp": f"{info['clogp']:.2f}",
            "tpsa": f"{info['tpsa']:.1f}",
            "hbd": str(info["hbd"]),
            "hba": str(info["hba"]),
            "rotb": str(info["rotb"]),
            "atoms": str(info["atoms"]),
        }
        for key, lbl in self._prop_labels.items():
            lbl.setText("—" if empty else vals[key])
        flags = []
        if not empty:
            if info["fragments"] > 1:
                flags.append(("#B45309", f"⚠ {info['fragments']} disconnected fragments"))
            if info["stereo_undefined"]:
                flags.append(("#B45309", f"⚠ {info['stereo_undefined']} undefined stereocenter(s)"))
            if info["lipinski_violations"]:
                flags.append(("#B45309", f"Lipinski: {info['lipinski_violations']} violation(s)"))
            else:
                flags.append(("#15803D", "✓ Lipinski rule of five"))
        self.lbl_flags.setText("<br>".join(f"<span style='color:{c}; font-size:11px'>{t}</span>" for c, t in flags))
        if not self.edit_smiles.hasFocus():
            self.edit_smiles.setText("" if empty else info["smiles"])
        self.btn_undo.setEnabled(bool(self._undo))
        self.btn_redo.setEnabled(bool(self._redo))
        self.btn_build.setEnabled(not empty)
        self.btn_protonate.setEnabled(not empty)

    def show_status(self, text, kind="info"):
        txt = str(text or "").strip()
        if txt.lower().startswith("status:"):
            txt = txt[7:].strip()
        colors = {"ok": "#15803D", "error": "#B91C1C", "info": "#3E4C59", "warn": "#B45309"}
        dots = {"ok": "●", "error": "●", "info": "○", "warn": "●"}
        color = colors.get(kind, colors["info"])
        self.lbl_status.setText(
            f"<span style='color:{color}'>{dots.get(kind, '')}</span>&nbsp;"
            f"<span style='color:{'#B91C1C' if kind == 'error' else '#3E4C59'}'>{_html_escape(txt)}</span>"
        )

    def _on_view_changed(self, *_args):
        self.canvas.show_atom_numbers = self.act_numbers.isChecked()
        self.canvas.show_stereo = self.act_stereo.isChecked()
        self.render = build_render_model(self.mol, show_carbons=self.act_carbons.isChecked())
        self.canvas.update()

    def _on_smiles_entered(self):
        txt = str(self.edit_smiles.text() or "").strip()
        if not txt:
            return
        mol = Chem.MolFromSmiles(txt)
        if mol is None:
            self.show_status("That SMILES could not be parsed.", "error")
            return
        self.set_molecule(mol, record=True, fit=True)
        self.edit_smiles.clearFocus()
        self._refresh_panel()
        if callable(self.on_molecule_changed):
            self.on_molecule_changed(self.current_mol())
        self.show_status("Molecule loaded from SMILES.", "ok")

    def _copy_smiles(self):
        info = molecule_summary(self.mol)
        if not info["smiles"]:
            self.show_status("Nothing to copy.", "info")
            return
        try:
            QtWidgets.QApplication.clipboard().setText(info["smiles"])
        except Exception:
            pass
        self.show_status("SMILES copied to the clipboard.", "ok")

    def _on_import(self, key):
        handlers = {
            "active": self.on_import_active,
            "file": self.on_import_file,
            "clipboard": self.on_paste_clipboard,
        }
        fn = handlers.get(key)
        if key in ("ketcher", "local_jsme", "jsme"):
            fn = (lambda k=key: self.on_open_web_drawer(k)) if callable(self.on_open_web_drawer) else None
        if key == "clipboard" and not callable(fn):
            fn = self._paste_clipboard_default
        if callable(fn):
            fn()

    def _paste_clipboard_default(self):
        try:
            txt = str(QtWidgets.QApplication.clipboard().text() or "").strip()
        except Exception:
            txt = ""
        mol = Chem.MolFromSmiles(txt.split()[0]) if txt else None
        if mol is None:
            self.show_status("The clipboard does not contain a valid SMILES.", "error")
            return
        self.set_molecule(mol, record=True, fit=True)
        if callable(self.on_molecule_changed):
            self.on_molecule_changed(self.current_mol())

    def _on_protonate(self):
        if callable(self.on_protonate):
            self.on_protonate()

    def _on_build(self):
        if self.mol is None:
            return
        frags = molecule_summary(self.mol)["fragments"]
        if frags > 1:
            self.show_status(
                f"The sketch has {frags} disconnected fragments; delete the extra pieces before building the ligand.",
                "error",
            )
            return
        if callable(self.on_build_3d):
            self.on_build_3d()

    def _open_periodic_table(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Choose element")
        dlg.setStyleSheet(STYLE)
        lay = QtWidgets.QGridLayout(dlg)
        lay.setSpacing(3)
        for r, row in enumerate(PERIODIC_ROWS):
            for c, sym in enumerate(row):
                btn = QtWidgets.QToolButton()
                btn.setObjectName("Elem")
                btn.setText(sym)
                btn.setStyleSheet(f"QToolButton#Elem {{ color: {ELEMENT_COLORS.get(sym.upper(), '#243447')}; min-width: 34px; }}")
                btn.clicked.connect(lambda _c=False, s=sym: (self.set_element_tool(s), dlg.accept()))
                lay.addWidget(btn, r, c)
        _exec_dialog(dlg)


def _widest_gap_direction(vectors):
    """Unit vector bisecting the widest angular gap between screen-space bond vectors."""
    angles = sorted(math.atan2(v.y(), v.x()) % (2.0 * math.pi) for v in vectors if abs(v.x()) + abs(v.y()) > 1e-6)
    if not angles:
        return 0.0, 1.0
    if len(angles) == 1:
        a = angles[0] + math.pi
        return math.cos(a), math.sin(a)
    best = None
    for k, a in enumerate(angles):
        nxt = angles[(k + 1) % len(angles)] + (2.0 * math.pi if k + 1 == len(angles) else 0.0)
        if best is None or nxt - a > best[0]:
            best = (nxt - a, a + 0.5 * (nxt - a))
    return math.cos(best[1]), math.sin(best[1])


def _html_escape(txt):
    return str(txt).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _make_shortcut(seq, parent, fn):
    cls = getattr(QtWidgets, "QShortcut", None) or getattr(QtGui, "QShortcut", None)
    if cls is None:
        return None
    sc = cls(QtGui.QKeySequence(seq), parent)
    sc.activated.connect(fn)
    return sc


def _exec_menu(menu, pos):
    if hasattr(menu, "exec_"):
        return menu.exec_(pos)
    return menu.exec(pos)


def _exec_dialog(dlg):
    if hasattr(dlg, "exec_"):
        return dlg.exec_()
    return dlg.exec()
