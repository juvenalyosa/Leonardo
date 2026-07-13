import math

AA_PROTEIN_RESN = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "GLH", "HID", "HIE", "HIP", "LYN", "CYM", "CYX", "MSE",
}

BACKBONE_ATOM_NAMES = {
    "N", "CA", "C", "O", "OXT", "H", "HN", "HA", "HA2", "HA3", "HT1", "HT2", "HT3",
}

CAP_ATOM_NAMES = BACKBONE_ATOM_NAMES.union({
    "CB", "HB", "HB1", "HB2", "HB3",
})


def _as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int_or_none(value):
    try:
        return int(str(value).strip())
    except Exception:
        return None


def parse_bond_change_pairs(text):
    raw = str(text or "").strip()
    if not raw:
        return []
    out = []
    seen = set()
    chunks = []
    cur = ""
    for ch in raw:
        if ch in ",;":
            if cur.strip():
                chunks.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        chunks.append(cur.strip())
    for tok in chunks:
        t = str(tok).replace(":", "-").replace(" ", "")
        if "-" not in t:
            continue
        p = t.split("-", 1)
        if len(p) != 2:
            continue
        a = _to_int_or_none(p[0])
        b = _to_int_or_none(p[1])
        if a is None or b is None:
            continue
        if a <= 0 or b <= 0 or a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append((int(a), int(b)))
    return out


def parse_atom_index_list(text):
    """Parse user-facing atom-index lists such as "12, 45 80-82"."""
    raw = str(text or "").strip()
    if not raw:
        return []
    out = []
    seen = set()
    tokens = raw.replace(";", ",").replace("+", ",").replace(" ", ",").split(",")
    for tok in tokens:
        t = str(tok or "").strip()
        if not t:
            continue
        if "-" in t:
            parts = t.split("-", 1)
            a = _to_int_or_none(parts[0])
            b = _to_int_or_none(parts[1])
            if a is None or b is None:
                continue
            for idx in range(min(a, b), max(a, b) + 1):
                if idx > 0 and idx not in seen:
                    seen.add(int(idx))
                    out.append(int(idx))
            continue
        idx = _to_int_or_none(t)
        if idx is not None and idx > 0 and idx not in seen:
            seen.add(int(idx))
            out.append(int(idx))
    return out


def covalent_radius(element):
    e = str(element or "").strip().upper()
    table = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "P": 1.07,
        "S": 1.05,
        "CL": 1.02,
        "BR": 1.20,
        "I": 1.39,
    }
    if e in table:
        return float(table[e])
    if e and e[0] in table:
        return float(table[e[0]])
    return 0.77


def atom_xyz(atom):
    a = dict(atom or {})
    return (
        _as_float(a.get("x", 0.0), 0.0),
        _as_float(a.get("y", 0.0), 0.0),
        _as_float(a.get("z", 0.0), 0.0),
    )


def distance_atoms(a, b):
    ax, ay, az = atom_xyz(a)
    bx, by, bz = atom_xyz(b)
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _atom_index_map(atoms):
    out = {}
    for k, a in enumerate(list(atoms or []), start=1):
        idx = _to_int_or_none((a or {}).get("index", k))
        if idx is None:
            continue
        out[int(idx)] = int(k - 1)
    return out


def _norm3(v):
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]) + float(v[2]) * float(v[2]))


def _unit3(v):
    n = _norm3(v)
    if n <= 1.0e-12:
        return [1.0, 0.0, 0.0], 0.0
    return [float(v[0]) / n, float(v[1]) / n, float(v[2]) / n], float(n)


def adjust_atoms_for_bond_changes(
    atoms,
    form_pairs=None,
    break_pairs=None,
    iterations=24,
    form_scale=1.07,
    break_extra=1.20,
):
    src = [dict(a or {}) for a in list(atoms or [])]
    if not src:
        return [], "empty atoms"
    idx_map = _atom_index_map(src)
    form_pairs = list(form_pairs or [])
    break_pairs = list(break_pairs or [])
    if not form_pairs and not break_pairs:
        return src, "no bond edits"
    coords = []
    for a in src:
        coords.append([
            float((a or {}).get("x", 0.0) or 0.0),
            float((a or {}).get("y", 0.0) or 0.0),
            float((a or {}).get("z", 0.0) or 0.0),
        ])
    n_iter = int(max(1, iterations))
    for _ in range(n_iter):
        for a_idx, b_idx in list(form_pairs or []):
            ia = idx_map.get(int(a_idx))
            ib = idx_map.get(int(b_idx))
            if ia is None or ib is None:
                continue
            va = coords[ia]
            vb = coords[ib]
            dvec = [float(vb[0]) - float(va[0]), float(vb[1]) - float(va[1]), float(vb[2]) - float(va[2])]
            u, dist = _unit3(dvec)
            ra = covalent_radius((src[ia] or {}).get("element", "C"))
            rb = covalent_radius((src[ib] or {}).get("element", "C"))
            target = float(max(1.10, float(form_scale) * (ra + rb)))
            delta = float(target - dist)
            step = 0.28 * delta
            for j in range(3):
                sh = 0.5 * step * float(u[j])
                va[j] -= sh
                vb[j] += sh
        for a_idx, b_idx in list(break_pairs or []):
            ia = idx_map.get(int(a_idx))
            ib = idx_map.get(int(b_idx))
            if ia is None or ib is None:
                continue
            va = coords[ia]
            vb = coords[ib]
            dvec = [float(vb[0]) - float(va[0]), float(vb[1]) - float(va[1]), float(vb[2]) - float(va[2])]
            u, dist = _unit3(dvec)
            ra = covalent_radius((src[ia] or {}).get("element", "C"))
            rb = covalent_radius((src[ib] or {}).get("element", "C"))
            target = float(max(dist, ra + rb + float(max(0.5, break_extra))))
            delta = float(target - dist)
            if delta <= 0.0:
                continue
            step = 0.20 * delta
            for j in range(3):
                sh = 0.5 * step * float(u[j])
                va[j] -= sh
                vb[j] += sh
    out = []
    for i, a in enumerate(src):
        aa = dict(a)
        aa["x"] = float(coords[i][0])
        aa["y"] = float(coords[i][1])
        aa["z"] = float(coords[i][2])
        out.append(aa)
    note = (
        f"product guess updated with form={len(form_pairs)} break={len(break_pairs)} "
        f"(iters={n_iter})"
    )
    return out, note


def atom_residue_key(atom):
    a = dict(atom or {})
    chain = str(a.get("chain", "") or "").strip()
    resi = str(a.get("resi", "") or "").strip()
    resn = str(a.get("resn", "") or "").strip().upper()
    return (chain, resi, resn)


def _residue_sort_key(residue_key):
    chain, resi, resn = tuple(residue_key or ("", "", ""))
    ri = _to_int_or_none(resi)
    if ri is None:
        return (str(chain), 1, str(resi), str(resn))
    return (str(chain), 0, int(ri), str(resn))


def _residue_numeric(residue_key):
    try:
        return _to_int_or_none(tuple(residue_key or ("", "", ""))[1])
    except Exception:
        return None


def _atom_identity(atom):
    a = dict(atom or {})
    idx = _to_int_or_none(a.get("index"))
    if idx is not None and idx > 0:
        return ("index", int(idx))
    x, y, z = atom_xyz(a)
    return (
        "atom",
        str(a.get("chain", "") or "").strip(),
        str(a.get("resi", "") or "").strip(),
        str(a.get("resn", "") or "").strip().upper(),
        str(a.get("name", "") or "").strip().upper(),
        round(x, 4),
        round(y, 4),
        round(z, 4),
    )


def is_backbone_atom(atom):
    a = dict(atom or {})
    name = str(a.get("name", "") or "").strip().upper()
    resn = str(a.get("resn", "") or "").strip().upper()
    return bool(name in BACKBONE_ATOM_NAMES and resn in AA_PROTEIN_RESN)


def _residue_atom_map(atoms):
    out = {}
    for a in list(atoms or []):
        rk = atom_residue_key(a)
        out.setdefault(rk, []).append(dict(a or {}))
    return out


def _protein_residue_keys_by_chain(atoms):
    chain_map = {}
    for a in list(atoms or []):
        rk = atom_residue_key(a)
        chain, resi, resn = rk
        if resn not in AA_PROTEIN_RESN:
            continue
        ri = _to_int_or_none(resi)
        if ri is None:
            continue
        chain_map.setdefault(chain, set()).add((int(ri), rk))
    return {
        chain: [rk for _ri, rk in sorted(rows, key=lambda x: (int(x[0]), str(x[1][1]), str(x[1][2])))]
        for chain, rows in chain_map.items()
    }


def _boundary_residue_keys(atoms):
    """Return first/last residue keys for every selected protein segment."""
    out = set()
    for _chain, keys in _protein_residue_keys_by_chain(atoms).items():
        rows = [(int(_residue_numeric(k)), k) for k in keys if _residue_numeric(k) is not None]
        if not rows:
            continue
        rows_s = sorted(rows, key=lambda x: int(x[0]))
        start_key = rows_s[0][1]
        prev_i = int(rows_s[0][0])
        prev_key = rows_s[0][1]
        for ri, rk in rows_s[1:]:
            if int(ri) != int(prev_i) + 1:
                out.add(start_key)
                out.add(prev_key)
                start_key = rk
            prev_i = int(ri)
            prev_key = rk
        out.add(start_key)
        out.add(prev_key)
    return out


def select_cluster_atoms_around_seeds(atoms, seed_indices=None, radius=5.0, expand_residues=True):
    """Build a cluster from seed atom indices and a radial cutoff.

    This backend helper mirrors the GUI "around atom(s)" mode but remains
    independent from PyMOL, so extraction behavior can be regression-tested.
    """
    rows = [dict(a or {}) for a in list(atoms or [])]
    seeds = set(int(x) for x in list(seed_indices or []) if int(x) > 0)
    rad = float(max(0.1, _as_float(radius, 5.0)))
    if not rows:
        return [], {"seed_count": 0, "within_atom_count": 0, "residue_count": 0, "radius": float(rad)}
    idx_map = _atom_index_map(rows)
    seed_atoms = [rows[idx_map[i]] for i in sorted(seeds) if i in idx_map]
    if not seed_atoms:
        return [], {"seed_count": 0, "within_atom_count": 0, "residue_count": 0, "radius": float(rad)}
    within_ids = set()
    selected_residues = set()
    for atom in rows:
        if any(distance_atoms(atom, seed) <= rad for seed in seed_atoms):
            within_ids.add(_atom_identity(atom))
            selected_residues.add(atom_residue_key(atom))
    if bool(expand_residues):
        out = [dict(a or {}) for a in rows if atom_residue_key(a) in selected_residues]
    else:
        out = [dict(a or {}) for a in rows if _atom_identity(a) in within_ids]
    return out, {
        "seed_count": int(len(seed_atoms)),
        "within_atom_count": int(len(within_ids)),
        "residue_count": int(len(selected_residues)),
        "radius": float(rad),
    }


def _find_atom_by_name(residue_atoms, name):
    target = str(name or "").strip().upper()
    for a in list(residue_atoms or []):
        if str((a or {}).get("name", "") or "").strip().upper() == target:
            return dict(a or {})
    return None


def _neighbor_residue_key(residue_key, context_chain_keys, direction):
    chain = tuple(residue_key or ("", "", ""))[0]
    keys = list((context_chain_keys or {}).get(chain, []) or [])
    try:
        pos = keys.index(tuple(residue_key))
    except Exception:
        return None
    nxt = int(pos) + int(direction)
    if 0 <= nxt < len(keys):
        return tuple(keys[nxt])
    return None


def _cap_position(anchor_atom, partner_atom=None, fallback_atom=None, bond_length=1.09):
    ax, ay, az = atom_xyz(anchor_atom)
    if partner_atom:
        px, py, pz = atom_xyz(partner_atom)
        vec = (px - ax, py - ay, pz - az)
    elif fallback_atom:
        fx, fy, fz = atom_xyz(fallback_atom)
        vec = (ax - fx, ay - fy, az - fz)
    else:
        vec = (1.0, 0.0, 0.0)
    unit, norm = _unit3(vec)
    if norm <= 1.0e-8:
        unit = [1.0, 0.0, 0.0]
    return (
        ax + float(bond_length) * float(unit[0]),
        ay + float(bond_length) * float(unit[1]),
        az + float(bond_length) * float(unit[2]),
    )


def add_peptide_link_h_caps(cluster_atoms, context_atoms=None, cap_n_term=True, cap_c_term=True):
    """Add link hydrogens where selected protein residues are truncated.

    Full ACE/NME cap construction is fragile for arbitrary PyMOL selections.
    Link hydrogens are deterministic and use the missing peptide-bond direction
    from the full source object when that context is available.
    """
    rows = [dict(a or {}) for a in list(cluster_atoms or [])]
    if not rows:
        return [], {"cap_count": 0, "cap_atom_indices": [], "capped_residue_keys": []}
    ctx_rows = [dict(a or {}) for a in list(context_atoms or [])] or list(rows)
    context_is_full = bool(context_atoms)
    cluster_res = _residue_atom_map(rows)
    context_res = _residue_atom_map(ctx_rows)
    cluster_keys_by_chain = _protein_residue_keys_by_chain(rows)
    context_keys_by_chain = _protein_residue_keys_by_chain(ctx_rows)
    cluster_key_set = set(k for keys in cluster_keys_by_chain.values() for k in keys)
    max_idx = 0
    for a in rows:
        idx = _to_int_or_none(a.get("index"))
        if idx is not None:
            max_idx = max(max_idx, int(idx))
    cap_rows = []
    capped_keys = set()
    for rk in sorted(cluster_key_set, key=_residue_sort_key):
        ratoms = cluster_res.get(rk, [])
        n_atom = _find_atom_by_name(ratoms, "N")
        ca_atom = _find_atom_by_name(ratoms, "CA")
        c_atom = _find_atom_by_name(ratoms, "C")
        key_source = context_keys_by_chain if context_is_full else cluster_keys_by_chain
        prev_key = _neighbor_residue_key(rk, key_source, -1)
        next_key = _neighbor_residue_key(rk, key_source, 1)
        needs_n_cap = bool(cap_n_term and n_atom and prev_key and prev_key not in cluster_key_set)
        needs_c_cap = bool(cap_c_term and c_atom and next_key and next_key not in cluster_key_set)
        if (not context_is_full) and prev_key is None and n_atom:
            needs_n_cap = bool(cap_n_term)
        if (not context_is_full) and next_key is None and c_atom:
            needs_c_cap = bool(cap_c_term)
        if needs_n_cap:
            partner = _find_atom_by_name(context_res.get(prev_key, []), "C") if prev_key else None
            x, y, z = _cap_position(n_atom, partner_atom=partner, fallback_atom=ca_atom, bond_length=1.02)
            max_idx += 1
            cap_rows.append({
                "index": int(max_idx),
                "name": "HNCP",
                "element": "H",
                "symbol": "H",
                "resn": str(rk[2]),
                "chain": str(rk[0]),
                "resi": str(rk[1]),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "is_cap": True,
                "cap_type": "N-link-H",
                "cap_anchor_index": _to_int_or_none(n_atom.get("index")),
            })
            capped_keys.add(rk)
        if needs_c_cap:
            partner = _find_atom_by_name(context_res.get(next_key, []), "N") if next_key else None
            x, y, z = _cap_position(c_atom, partner_atom=partner, fallback_atom=ca_atom, bond_length=1.09)
            max_idx += 1
            cap_rows.append({
                "index": int(max_idx),
                "name": "HCCP",
                "element": "H",
                "symbol": "H",
                "resn": str(rk[2]),
                "chain": str(rk[0]),
                "resi": str(rk[1]),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "is_cap": True,
                "cap_type": "C-link-H",
                "cap_anchor_index": _to_int_or_none(c_atom.get("index")),
            })
            capped_keys.add(rk)
    out = list(rows) + list(cap_rows)
    return out, {
        "cap_count": int(len(cap_rows)),
        "cap_atom_indices": [int(a.get("index")) for a in cap_rows],
        "capped_residue_keys": sorted(list(capped_keys), key=_residue_sort_key),
    }


def build_cluster_freeze_indices(atoms, freeze_backbone=True, freeze_caps=True):
    rows = [dict(a or {}) for a in list(atoms or [])]
    idxs = set()
    boundary_keys = set()
    if bool(freeze_backbone):
        for a in rows:
            if is_backbone_atom(a):
                idx = _to_int_or_none(a.get("index"))
                if idx is not None and idx > 0:
                    idxs.add(int(idx))
    if bool(freeze_caps):
        boundary_keys = _boundary_residue_keys(rows)
        for a in rows:
            rk = atom_residue_key(a)
            if bool((a or {}).get("is_cap", False)):
                idx = _to_int_or_none(a.get("index"))
                if idx is not None and idx > 0:
                    idxs.add(int(idx))
                continue
            if rk not in boundary_keys:
                continue
            nm = str(a.get("name", "") or "").strip().upper()
            if nm not in CAP_ATOM_NAMES:
                continue
            idx = _to_int_or_none(a.get("index"))
            if idx is not None and idx > 0:
                idxs.add(int(idx))
    info = {
        "freeze_count": int(len(idxs)),
        "boundary_residue_count": int(len(boundary_keys)),
        "boundary_residue_keys": sorted(list(boundary_keys)),
    }
    return sorted(idxs), info


def prepare_cluster_model(
    atoms,
    context_atoms=None,
    seed_indices=None,
    radius=5.0,
    expand_residues=True,
    add_caps=True,
    freeze_backbone=True,
    freeze_caps=True,
):
    """Extract, cap, and freeze a QM cluster model.

    The GUI calls this after collecting atoms from PyMOL.  Tests call it
    directly to keep the extraction/capping rules stable over time.
    """
    if seed_indices:
        selected, select_meta = select_cluster_atoms_around_seeds(
            atoms,
            seed_indices=seed_indices,
            radius=radius,
            expand_residues=expand_residues,
        )
    else:
        selected = [dict(a or {}) for a in list(atoms or [])]
        select_meta = {
            "seed_count": 0,
            "within_atom_count": int(len(selected)),
            "residue_count": int(len(set(atom_residue_key(a) for a in selected))),
            "radius": float(max(0.1, _as_float(radius, 5.0))),
        }
    prepared = list(selected)
    cap_meta = {"cap_count": 0, "cap_atom_indices": [], "capped_residue_keys": []}
    if bool(add_caps):
        prepared, cap_meta = add_peptide_link_h_caps(prepared, context_atoms=context_atoms)
    freeze_idx, freeze_meta = build_cluster_freeze_indices(
        prepared,
        freeze_backbone=freeze_backbone,
        freeze_caps=freeze_caps,
    )
    meta = {
        "input_atom_count": int(len(list(atoms or []))),
        "selected_atom_count": int(len(selected)),
        "cluster_atom_count": int(len(prepared)),
        "selection": dict(select_meta or {}),
        "caps": dict(cap_meta or {}),
        "freeze": dict(freeze_meta or {}),
        "freeze_atom_indices": list(freeze_idx or []),
    }
    return prepared, freeze_idx, meta


def build_pymol_cluster_selection(
    source_object="",
    mode="around_selection",
    seed_selection="",
    seed_indices=None,
    radius=5.0,
    expand_residues=True,
    include_waters=False,
):
    """Create PyMOL selection strings for the cluster GUI."""
    obj = str(source_object or "").strip()
    mode_key = str(mode or "around_selection").strip().lower()
    seed_expr = str(seed_selection or "").strip()
    rad = float(max(0.1, _as_float(radius, 5.0)))
    if isinstance(seed_indices, str):
        idxs = parse_atom_index_list(seed_indices)
    else:
        idxs = [int(x) for x in list(seed_indices or []) if int(x) > 0]
    obj_scope = f"({obj})" if obj else "all"
    water_filter = "" if bool(include_waters) else " and not solvent"
    if mode_key == "manual":
        expr = seed_expr or "polymer.protein"
        low = expr.lower()
        explicit = ("model " in low) or ("obj " in low) or ("sele" in low) or ("all" == low)
        core = str(expr) if explicit or not obj else f"({obj_scope}) and ({expr})"
    elif mode_key == "around_indices":
        if not idxs:
            core = ""
        else:
            seed = f"({obj_scope}) and index {'+'.join(str(i) for i in idxs)}"
            core = f"({obj_scope}) and (within {rad:.3f} of ({seed}))"
    else:
        seed = seed_expr or "sele"
        low = seed.lower()
        explicit = ("model " in low) or ("obj " in low) or ("sele" in low) or ("all" == low)
        seed_sel = str(seed) if explicit or not obj else f"({obj_scope}) and ({seed})"
        core = f"({obj_scope}) and (within {rad:.3f} of ({seed_sel}))"
    if core and water_filter:
        core = f"({core}){water_filter}"
    final_sel = f"byres ({core})" if (core and bool(expand_residues)) else str(core)
    return {
        "source_object": str(obj),
        "mode": str(mode_key),
        "radius": float(rad),
        "seed_indices": list(idxs),
        "seed_selection": str(seed_expr),
        "core_sel": str(final_sel),
        "cap_sel": "",
        "final_sel": str(final_sel),
    }


def cluster_workflow_stage_plan(do_classical=True, do_preopt=True, run_qst3=True, run_irc=True):
    stages = []
    if bool(do_classical):
        stages.extend(["mm_react", "mm_prod"])
    if bool(do_preopt):
        stages.extend(["react_preopt", "prod_preopt"])
    stages.append("qst2_ts")
    if bool(run_qst3):
        stages.append("qst3_ts")
    if bool(run_irc):
        stages.append("irc")
    return stages


__all__ = [
    "AA_PROTEIN_RESN",
    "BACKBONE_ATOM_NAMES",
    "CAP_ATOM_NAMES",
    "parse_bond_change_pairs",
    "parse_atom_index_list",
    "covalent_radius",
    "atom_xyz",
    "distance_atoms",
    "adjust_atoms_for_bond_changes",
    "atom_residue_key",
    "is_backbone_atom",
    "select_cluster_atoms_around_seeds",
    "add_peptide_link_h_caps",
    "build_cluster_freeze_indices",
    "prepare_cluster_model",
    "build_pymol_cluster_selection",
    "cluster_workflow_stage_plan",
]
