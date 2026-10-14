"""Component library: find force-field parameters for every residue of a biomolecular system.

Leonardo used to have one route for anything non-standard (antechamber -> GAFF2). This module
indexes the residue templates of the force fields that are already installed (CHARMM36 with
heme, cofactors and lipids; Amber14 + GLYCAM + lipid21 + ions) by their element-labelled
heavy-atom graph, and for each residue of a structure decides where its parameters come from:

    library   : a curated template (by name alias or by graph isomorphism)
    gaff      : organic molecule without a template -> parameterize (GAFF2/OpenFF)
    metal     : metal-containing molecule without a template (needs a metal model)
    unmatched : incomplete residue or unknown chemistry (reported with the closest template)

``prepare_modeller`` then renames residues/atoms to the chosen templates and registers
hydrogen definitions derived from the templates themselves, so ``Modeller.addHydrogens``
protonates the components exactly as the force field expects (e.g. heme with deprotonated
propionates, charge -2 in CHARMM36).
"""

import json
import math
import os
import tempfile
import threading

STANDARD_POLYMER_RESN = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN", "ACE",
    "NME", "NMA", "NH2", "DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "I", "DI",
    "MSE",   # selenomethionine: PDBFixer replaces it with MET
}
REPLACEABLE_RESIDUES = {"MSE"}
WATER_RESN = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SOL", "SPC"}
METALS = {"LI", "NA", "K", "RB", "CS", "BE", "MG", "CA", "SR", "BA", "AL", "GA", "IN", "TL", "SN", "PB", "BI",
          "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "Y", "ZR", "NB", "MO", "TC", "RU", "RH",
          "PD", "AG", "CD", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "LA", "CE", "PR", "ND", "SM",
          "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "U"}
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "F": 0.57, "CL": 1.02,
                  "BR": 1.20, "I": 1.39, "SE": 1.20, "B": 0.84, "SI": 1.11, "FE": 1.32, "ZN": 1.22, "CU": 1.32,
                  "MG": 1.41, "CO": 1.26, "NI": 1.24, "MN": 1.39, "CA": 1.76}
COORDINATION_CUTOFF_NM = 0.28

_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "component_library.json")
_USER_FILE = os.path.join(os.path.expanduser("~"), ".leonardo", "parameters", "component_library.json")


def _openmmforcefields_dir():
    try:
        import openmmforcefields

        return os.path.join(os.path.dirname(openmmforcefields.__file__), "ffxml")
    except Exception:
        return ""


def forcefield_families():
    """Families, their base files and optional extra sources (loaded only when a component needs them)."""
    ffx = _openmmforcefields_dir()
    amber = lambda name: os.path.join(ffx, "amber", name) if ffx else ""  # noqa: E731
    return {
        "amber14": {
            "label": "Amber ff14SB + TIP3P (+ GLYCAM, lipid21)",
            "base": ["amber14-all.xml"],
            # water AND the ion templates (Zn, Mg, Ca, Na, Cl...) live here, not in amber14-all
            "water": ["amber14/tip3p.xml"],
            "extras": {
                "glycam": {"files": [amber("GLYCAM_06j-1.xml")], "index_with_base": True},
                "lipid21": {"files": [amber("lipid21.xml")], "index_with_base": False},
                # phosphorylated residues only load with openmmforcefields' ff14SB, which then
                # replaces OpenMM's amber14-all as the protein force field of the system
                "phosaa": {"files": [amber("ff14SB.xml"), amber("phosaa14SB.xml"), amber("tip3p_standard.xml")],
                           "index_with_base": False, "replaces_base": True,
                           "only": ["SEP", "TPO", "PTR", "S1P", "T1P", "Y1P"]},
            },
        },
        "charmm36": {
            "label": "CHARMM36 (proteins, lipids, heme, nucleotides, cofactors) + TIP3P",
            "base": ["charmm36.xml"],
            "water": ["charmm36/water.xml"],      # TIP3P and the ions (ZN2, MG, SOD, CLA, POT...)
            "extras": {},
            # residue + patch -> registered template (OpenMM's charmm36.xml has the patches but does
            # not apply them on its own): phosphoserine and phosphotyrosine, di- and monoanionic
            "patched": [("SER", "SP2"), ("SER", "SP1"), ("TYR", "TP2"), ("TYR", "TP1")],
        },
    }


def forcefield_family_base_files(family):
    """The protein force-field file(s) of a family (water/ion files are added per component)."""
    return list(forcefield_families().get(family, {}).get("base", []))


def load_library(path=None, user_path=None):
    """Curated aliases/metadata; the user file (same format) extends or overrides entries."""
    data = {"components": {}}
    user_default = os.path.join(os.environ["LEONARDO_PARAMETERS_DIR"], "component_library.json") \
        if os.environ.get("LEONARDO_PARAMETERS_DIR") else _USER_FILE
    for p in (path or _DATA_FILE, user_path or user_default):
        if p and os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                extra = json.load(fh)
            for key, entry in dict(extra.get("components", {})).items():
                data["components"][str(key).upper()] = dict(entry)
    return data


# --------------------------------------------------------------------------- graphs


def _element_symbol(atom):
    el = getattr(atom, "element", None)
    sym = getattr(el, "symbol", "") if el is not None else ""
    return str(sym or "").upper()


def template_heavy_graph(template):
    """networkx graph of a ForceField residue template (heavy atoms, element labels)."""
    import networkx as nx

    g = nx.Graph()
    heavy = {}
    for i, a in enumerate(template.atoms):
        sym = _element_symbol(a)
        if sym and sym != "H":
            heavy[i] = sym
            g.add_node(a.name, element=sym)
    for i, j in template.bonds:
        if i in heavy and j in heavy:
            g.add_edge(template.atoms[i].name, template.atoms[j].name)
    return g


def graph_formula(g):
    counts = {}
    for _n, d in g.nodes(data=True):
        counts[d["element"]] = counts.get(d["element"], 0) + 1
    return " ".join(f"{k}{counts[k]}" for k in sorted(counts))


def template_hydrogen_parents(template):
    """[(H name, parent heavy-atom name)] of a template."""
    out = []
    for idx, a in enumerate(template.atoms):
        if _element_symbol(a) != "H":
            continue
        for i, j in template.bonds:
            if idx in (i, j):
                out.append((a.name, template.atoms[j if i == idx else i].name))
                break
    return out


def template_charge(template):
    total = 0.0
    for a in template.atoms:
        q = getattr(a, "parameters", {}).get("charge") if hasattr(a, "parameters") else None
        if q is None:
            return None
        total += float(q)
    return round(total, 3)


def residue_heavy_graph(residue, positions_nm=None):
    """Graph of a topology residue from its bonds (or distances when the file had no CONECT records)."""
    import networkx as nx

    g = nx.Graph()
    atoms = [a for a in residue.atoms() if _element_symbol(a) not in ("", "H")]
    for a in atoms:
        g.add_node(a.name, element=_element_symbol(a), index=a.index)
    names = {a.index: a.name for a in atoms}
    bonded = False
    for bond in residue.bonds() if hasattr(residue, "bonds") else []:
        a, b = bond[0], bond[1]
        if a.index in names and b.index in names and a.residue is residue and b.residue is residue:
            g.add_edge(names[a.index], names[b.index])
            bonded = True
    if not bonded and positions_nm is not None and len(atoms) > 1:
        for i, a in enumerate(atoms):
            for b in atoms[i + 1:]:
                ra = COVALENT_RADII.get(_element_symbol(a), 0.8)
                rb = COVALENT_RADII.get(_element_symbol(b), 0.8)
                d = _dist(positions_nm[a.index], positions_nm[b.index])
                if d < 0.1 * (ra + rb) * 1.15:
                    g.add_edge(a.name, b.name)
    return g


def _dist(p, q):
    return math.sqrt(sum((float(p[k]) - float(q[k])) ** 2 for k in range(3)))


def match_graphs(template_g, residue_g):
    """Mapping residue atom name -> template atom name for isomorphic heavy-atom graphs, else None."""
    from networkx.algorithms import isomorphism

    if template_g.number_of_nodes() != residue_g.number_of_nodes():
        return None
    if template_g.number_of_edges() != residue_g.number_of_edges():
        return None
    gm = isomorphism.GraphMatcher(residue_g, template_g, node_match=lambda a, b: a["element"] == b["element"])
    if not gm.is_isomorphic():
        return None
    mapping = dict(gm.mapping)
    # prefer the mapping that keeps names when several are equivalent (symmetry)
    if all(k in template_g for k in residue_g.nodes):
        same = {k: k for k in residue_g.nodes}
        if all(template_g.has_edge(same[u], same[v]) for u, v in residue_g.edges):
            return same
    return mapping


def missing_vs_template(template_g, residue_g):
    """Heavy atoms of the template the residue lacks, if the residue is a subgraph (disordered/partial)."""
    from networkx.algorithms import isomorphism

    if residue_g.number_of_nodes() >= template_g.number_of_nodes() or residue_g.number_of_nodes() == 0:
        return None
    gm = isomorphism.GraphMatcher(template_g, residue_g, node_match=lambda a, b: a["element"] == b["element"])
    if gm.subgraph_is_monomorphic():
        used = set(gm.mapping)
        return sorted(n for n in template_g.nodes if n not in used)
    return None


# --------------------------------------------------------------------------- catalog


class TemplateCatalog:
    """Templates of one force-field family, indexed by heavy-atom formula."""

    def __init__(self, family):
        import openmm.app as app

        spec = forcefield_families()[family]
        self.family = family
        self.base_files = list(spec["base"]) + list(spec.get("water", []))
        self.entries = {}
        self.by_formula = {}
        self.by_name = {}
        protein_ff = app.ForceField(*spec["base"])
        self._index(protein_ff, source="base", names=None)
        base_ff = app.ForceField(*self.base_files)
        if spec.get("water"):
            # templates only in the water file (ions) carry that file, so runs without explicit
            # water (implicit solvent, gas phase) still load it for them
            self._index(base_ff, source="water", names=set(base_ff._templates) - set(protein_ff._templates),
                        files=list(spec["water"]))
        for key, extra in spec["extras"].items():
            files = [f for f in extra["files"] if f and os.path.isfile(f)]
            if len(files) != len(extra["files"]):
                continue
            try:
                if extra.get("index_with_base"):
                    ff = app.ForceField(*self.base_files, *files)
                    new = set(ff._templates) - set(base_ff._templates)
                else:
                    ff = app.ForceField(*files)
                    new = set(extra["only"]) & set(ff._templates) if extra.get("only") else None
            except Exception:
                continue
            self._index(ff, source=key, names=new, files=files, replaces_base=bool(extra.get("replaces_base")))
        if spec.get("patched"):
            self._index_patched(base_ff, spec["patched"])

    def _index_patched(self, ff, pairs):
        """
        Register residue+patch templates through a generated residue XML (usable by any ForceField).
        OpenMM's CHARMM36 phosphorylation patches keep the hydroxyl H the original CHARMM patch
        deletes (HG1 of Ser, HH of Tyr); that H is removed here and the total charge must come out
        integer, otherwise the template is not offered.
        """
        import openmm.app as app

        made = []
        for resname, patch in pairs:
            t, p = ff._templates.get(resname), ff._patches.get(patch)
            if t is None or p is None:
                continue
            try:
                patched = p.createPatchedTemplates([t])[0]
            except Exception:
                continue
            fixed = _drop_replaced_hydroxyl_h(patched, original=t)
            if fixed is not None:
                made.append(fixed)
        if not made:
            return
        path = os.path.join(_cache_dir(), f"{self.family}_patched_residues.xml")
        with open(path, "w") as fh:
            fh.write(templates_to_xml(made))
        try:
            loaded = app.ForceField(*self.base_files, path)
        except Exception:
            return
        self._index(loaded, source="patch", names={t["name"] for t in made}, files=[path])

    def _index(self, ff, source, names=None, files=None, replaces_base=False):
        for name, t in ff._templates.items():
            if names is not None and name not in names:
                continue
            if name in self.by_name:
                continue
            g = template_heavy_graph(t)
            entry = {"name": name, "source": source, "files": list(files or []), "graph": g,
                     "formula": graph_formula(g), "charge": template_charge(t),
                     "external_bonds": len(getattr(t, "externalBonds", []) or []),
                     "hydrogens": template_hydrogen_parents(t), "n_atoms": len(t.atoms),
                     "replaces_base": bool(replaces_base)}
            self.by_name[name] = entry
            self.by_formula.setdefault(entry["formula"], []).append(entry)

    def find(self, residue_g, names=(), external_bonds=0):
        """Best template for a residue graph: aliases first, then any isomorphic template."""
        for n in names:
            e = self.by_name.get(n)
            if e is not None:
                mapping = match_graphs(e["graph"], residue_g)
                if mapping is not None:
                    return e, mapping, "alias"
        candidates = self.by_formula.get(graph_formula(residue_g), [])
        ranked = sorted(candidates, key=lambda e: (e["external_bonds"] != external_bonds, e["source"] != "base"))
        for e in ranked:
            mapping = match_graphs(e["graph"], residue_g)
            if mapping is not None:
                return e, mapping, "graph"
        return None, None, None

    def closest_partial(self, residue_g, names=()):
        for n in names:
            e = self.by_name.get(n)
            if e is not None:
                miss = missing_vs_template(e["graph"], residue_g)
                if miss:
                    return e, miss
        return None, None


def _cache_dir():
    path = os.environ.get("LEONARDO_PARAMETERS_DIR") or os.path.join(os.path.expanduser("~"), ".leonardo", "parameters")
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except OSError:
        return tempfile.gettempdir()


def _template_dict(t, skip=()):
    keep = [i for i, a in enumerate(t.atoms) if a.name not in skip]
    idx = {old: new for new, old in enumerate(keep)}
    return {"name": t.name,
            "atoms": [(t.atoms[i].name, t.atoms[i].type, float(getattr(t.atoms[i], "parameters", {}).get("charge", 0.0)))
                      for i in keep],
            "bonds": [(idx[i], idx[j]) for i, j in t.bonds if i in idx and j in idx],
            "external": [idx[i] for i in t.externalBonds if i in idx]}


def _drop_replaced_hydroxyl_h(patched, original):
    """Template dict without the original residue's H on an O that the patch bonded to P (or None)."""
    names = [a.name for a in patched.atoms]
    elem = {a.name: _element_symbol(a) for a in patched.atoms}
    nbrs = {n: set() for n in names}
    for i, j in patched.bonds:
        nbrs[names[i]].add(names[j])
        nbrs[names[j]].add(names[i])
    orig_names = {a.name for a in original.atoms}
    drop = {h for o in names if elem[o] == "O" and any(elem[x] == "P" for x in nbrs[o])
            for h in nbrs[o] if elem[h] == "H" and h in orig_names}
    d = _template_dict(patched, skip=drop)
    total = sum(q for _n, _t, q in d["atoms"])
    if abs(total - round(total)) > 0.01:
        return None
    return d


def templates_to_xml(templates):
    """ForceField <Residues> XML for residue templates (dicts from _template_dict or OpenMM templates)."""
    from xml.sax.saxutils import quoteattr

    lines = ["<ForceField>", " <Residues>"]
    for t in templates:
        d = t if isinstance(t, dict) else _template_dict(t)
        lines.append(f"  <Residue name={quoteattr(d['name'])}>")
        for name, typ, q in d["atoms"]:
            lines.append(f"   <Atom name={quoteattr(name)} type={quoteattr(typ)} charge=\"{q:.6f}\"/>")
        for i, j in d["bonds"]:
            lines.append(f"   <Bond atomName1={quoteattr(d['atoms'][i][0])} atomName2={quoteattr(d['atoms'][j][0])}/>")
        for i in d["external"]:
            lines.append(f"   <ExternalBond atomName={quoteattr(d['atoms'][i][0])}/>")
        lines.append("  </Residue>")
    lines += [" </Residues>", "</ForceField>"]
    return "\n".join(lines) + "\n"


_catalog_cache = {}
_catalog_lock = threading.Lock()


def get_catalog(family):
    with _catalog_lock:
        if family not in _catalog_cache:
            _catalog_cache[family] = TemplateCatalog(family)
        return _catalog_cache[family]


# --------------------------------------------------------------------------- resolver


def _is_polymer_residue(residue, catalog):
    name = str(residue.name).upper()
    if name in STANDARD_POLYMER_RESN:
        return True
    try:
        from openmm.app import PDBFile

        PDBFile._loadNameReplacementTables()
        if name in PDBFile._residueNameReplacements:
            return PDBFile._residueNameReplacements[name] in catalog.by_name
    except Exception:
        pass
    return False


def _is_modified_polymer_residue(residue, positions_nm):
    """Non-standard residue built into a chain (backbone atoms bonded to a neighbour), e.g. SEP, TPO, MSE."""
    names = {a.name for a in residue.atoms()}
    backbone = {"N", "CA", "C"} <= names or {"P", "O5'", "C5'"} <= names
    if not backbone:
        return False
    for bond in residue.bonds() if hasattr(residue, "bonds") else []:
        if bond[0].residue is not bond[1].residue:
            return True
    atoms = {a.name: a for a in residue.atoms()}
    for other in residue.chain.residues():
        if other is residue:
            continue
        oat = {a.name: a for a in other.atoms()}
        for x, y in (("N", "C"), ("C", "N"), ("P", "O3'")):
            if x in atoms and y in oat and _dist(positions_nm[atoms[x].index], positions_nm[oat[y].index]) < 0.2:
                return True
    return False


def _coordination(topology, positions_nm, metal_atoms):
    out = []
    for m in metal_atoms:
        for a in topology.atoms():
            if a.residue is m.residue or _element_symbol(a) in ("", "H", "C"):
                continue
            d = _dist(positions_nm[m.index], positions_nm[a.index])
            if d <= COORDINATION_CUTOFF_NM:
                out.append({"metal": f"{m.residue.name}{m.residue.id}:{m.name}",
                            "ligand_atom": f"{a.residue.name}{a.residue.id}{a.residue.chain.id}:{a.name}",
                            "distance_A": round(d * 10.0, 2)})
    return out


def resolve_components(topology, positions_nm, family="amber14", library=None, try_other_families=True):
    """
    Where each non-polymer residue gets its parameters, in ``family``. The plan also says which
    family covers the system best (curated parameters for more components); ties keep ``family``.
    """
    library = library or load_library()
    plan = _resolve(topology, positions_nm, family, library, try_other_families)
    if try_other_families and not plan["ready"]:
        best, best_score = family, _deficits(plan)
        for fam in forcefield_families():
            if fam == family:
                continue
            other = _resolve(topology, positions_nm, fam, library, False)
            if _deficits(other) < best_score:
                best, best_score = fam, _deficits(other)
        plan["recommended_family"] = best
        if best != family:
            plan["warnings"].insert(0, f"{best} has parameters for cofactors/metals that {family} would drop or "
                                       f"approximate: build the whole system in {best} (force fields are not mixed)")
    if family == "charmm36" and any(r.get("source") == "gaff" for r in plan["components"]):
        plan["warnings"].append(GAFF_IN_CHARMM_WARNING)
    return plan


GAFF_IN_CHARMM_WARNING = ("GAFF2 ligands in CHARMM36 get CHARMM's 1-4 scaling (electrostatics 1.0, LJ 1.0 instead of "
                          "GAFF's 0.833/0.5): their torsional energetics are off. Use amber14 for these ligands "
                          "(or CGenFF parameters), or treat CHARMM36 results for them as approximate")


DEFICIT_SOURCES = {"metal", "unmatched", "modified_residue", "covalent", "covalent_metal"}
CURATED_CATEGORIES = {"cofactor", "nucleotide", "lipid"}


def _deficits(plan):
    """
    Component types that would be dropped or badly parameterized in this family: no parameters at
    all (metal/unmatched/modified residue), or a curated cofactor/nucleotide/lipid falling back to
    GAFF. A ligand or additive going to GAFF is an acceptable route and does not count.
    """
    out = set()
    for r in plan["components"]:
        if r.get("buildable"):
            continue
        if r["source"] in DEFICIT_SOURCES or (r["source"] == "gaff" and r.get("category") in CURATED_CATEGORIES):
            out.add(r["resn"].upper())
    return len(out)


def covalent_links(topology, positions_nm):
    """
    Covalent bonds between a non-polymer component and another residue (covalent inhibitors, c-type
    heme thioethers, glycosylation...): from the topology bonds and, when the file had no LINK/CONECT
    records, from heavy-atom distances. Metal coordination is excluded. Returns
    {component residue: [(component atom, partner atom, distance_nm)]}.
    """
    out = {}

    def add(a, b, d):
        out.setdefault(a.residue, []).append((a, b, d))

    seen = set()
    for bond in topology.bonds():
        a, b = bond[0], bond[1]
        if a.residue is b.residue or _element_symbol(a) in METALS or _element_symbol(b) in METALS:
            continue
        d = _dist(positions_nm[a.index], positions_nm[b.index])
        limit = 0.1 * 1.25 * (COVALENT_RADII.get(_element_symbol(a), 0.8) + COVALENT_RADII.get(_element_symbol(b), 0.8))
        if d > limit:
            continue            # a CONECT written from a close contact (e.g. by PyMOL), not a covalent bond
        for x, y in ((a, b), (b, a)):
            if str(x.residue.name).upper() not in STANDARD_POLYMER_RESN | WATER_RESN:
                add(x, y, d)
                seen.add((x.index, y.index))
    comps = [r for r in topology.residues() if str(r.name).upper() not in STANDARD_POLYMER_RESN | WATER_RESN]
    if comps:
        others = [a for a in topology.atoms() if _element_symbol(a) not in ("", "H") and _element_symbol(a) not in METALS]
        for res in comps:
            mine = [a for a in res.atoms() if _element_symbol(a) not in ("", "H") and _element_symbol(a) not in METALS]
            for a in mine:
                ra = COVALENT_RADII.get(_element_symbol(a), 0.8)
                for b in others:
                    if b.residue is res or (a.index, b.index) in seen:
                        continue
                    d = _dist(positions_nm[a.index], positions_nm[b.index])
                    if d < 0.1 * 1.15 * (ra + COVALENT_RADII.get(_element_symbol(b), 0.8)):
                        add(a, b, d)
                        seen.add((a.index, b.index))
    return out


def _internal_residue(res, positions_nm):
    atoms = {a.name for a in res.atoms()}
    if not {"N", "CA", "C"} <= atoms:
        return False
    prev_c, next_n = _backbone_neighbours(res.chain.topology, res, positions_nm)
    return prev_c is not None and next_n is not None


def _single_internal_partner(link_rows, positions_nm):
    partners = {b.residue for _a, b, _d in link_rows}
    if len(partners) != 1:
        return False
    partner = next(iter(partners))
    return str(partner.name).upper() in STANDARD_POLYMER_RESN and _internal_residue(partner, positions_nm)


def _merged_template(lig, link_rows, catalog, positions_nm, meta, family):
    """
    A covalent ligand plus the protein residue it is bound to, matched together to one curated template
    (CHARMM36: LYR = lysine-retinal Schiff base, ALY, CYSF/CYSG prenylated Cys...). Returns plan fields or None.
    """
    import networkx as nx

    partners = {b.residue for _a, b, _d in link_rows}
    if len(partners) != 1:
        return None
    partner = next(iter(partners))
    if str(partner.name).upper() not in STANDARD_POLYMER_RESN:
        return None
    g = nx.Graph()
    for prefix, res in (("P:", partner), ("L:", lig)):
        sub = residue_heavy_graph(res, positions_nm)
        for n, d in sub.nodes(data=True):
            g.add_node(prefix + n, element=d["element"])
        for u, v in sub.edges:
            g.add_edge(prefix + u, prefix + v)
    for a, b, _d in link_rows:
        g.add_edge("L:" + a.name, "P:" + b.name)
    names = [str(x) for x in dict(meta.get("templates", {})).get(family, [])]
    entry, mapping, how = catalog.find(g, names=names, external_bonds=2)
    if entry is None or entry["external_bonds"] == 0:
        return None
    return {"source": "library", "family": family, "template": entry["name"], "matched_by": how,
            "template_source": entry["source"], "extra_files": entry["files"], "charge": entry["charge"],
            "category": "covalent adduct (merged residue)", "atom_map": mapping, "confidence": "high",
            "merge": {"chain": partner.chain.id, "resi": partner.id, "resn": partner.name},
            "partners": [f"{b.residue.name}{b.residue.id}{b.residue.chain.id}:{b.name}-{a.name}"
                         for a, b, _d in link_rows]}


def merge_residue_pairs(modeller, pairs):
    """
    Move the atoms of each ligand residue into its partner residue (one residue per adduct, as the
    merged template expects). pairs: [((lig chain, resi, resn), (partner chain, resi, resn))]. Returns
    {partner key: {atom name in merged residue: "L:"/"P:" original name}}.
    """
    import openmm.app as app

    lig_to_partner = {lk: pk for lk, pk in pairs}
    residues = {(r.chain.id, r.id, r.name): r for r in modeller.topology.residues()}
    ligands = {residues[lk]: residues[pk] for lk, pk in lig_to_partner.items() if lk in residues and pk in residues}
    if not ligands:
        return {}
    extra = {}
    for lig, partner in ligands.items():
        extra.setdefault(partner, []).append(lig)
    old = modeller.topology
    pos = list(modeller.positions)
    new = app.Topology()
    new.setPeriodicBoxVectors(old.getPeriodicBoxVectors())
    amap, new_pos, origin = {}, [], {}
    for chain in old.chains():
        residues_in_chain = [r for r in chain.residues() if r not in ligands]
        if not residues_in_chain:
            continue
        nc = new.addChain(chain.id)
        for res in residues_in_chain:
            nr = new.addResidue(res.name, nc, res.id, res.insertionCode)
            names = {}
            for a in res.atoms():
                amap[a] = new.addAtom(a.name, a.element, nr, a.id)
                new_pos.append(pos[a.index])
                names[a.name] = "P:" + a.name
            for lig in extra.get(res, []):
                for a in lig.atoms():
                    name = a.name if a.name not in names else a.name + "L"
                    amap[a] = new.addAtom(name, a.element, nr, a.id)
                    new_pos.append(pos[a.index])
                    names[name] = "L:" + a.name
            origin[(res.chain.id, res.id, res.name)] = names
    for bond in old.bonds():
        if bond[0] in amap and bond[1] in amap:
            new.addBond(amap[bond[0]], amap[bond[1]])
    modeller.topology = new
    modeller.positions = _as_quantity(new_pos)
    return origin


def _resolve(topology, positions_nm, family, library, try_other_families):
    comps = library.get("components", {})
    catalog = get_catalog(family)
    others = [f for f in forcefield_families() if f != family] if try_other_families else []
    rows = []
    metal_atoms = []
    links = covalent_links(topology, positions_nm)
    for res in topology.residues():
        name = str(res.name).upper()
        if name in WATER_RESN:
            continue
        if _is_polymer_residue(res, catalog):
            continue
        atoms = list(res.atoms())
        heavy = [a for a in atoms if _element_symbol(a) not in ("", "H")]
        meta = dict(comps.get(name, {}))
        row = {"resn": res.name, "chain": res.chain.id, "resi": res.id, "heavy_atoms": len(heavy),
               "category": meta.get("category", ""), "description": meta.get("description", "")}
        merged = _merged_template(res, links.get(res, []), catalog, positions_nm, meta, family) if res in links else None
        if merged is not None:
            row.update(merged)
            rows.append(row)
            continue
        if res in links and not _is_modified_polymer_residue(res, positions_nm):
            # bound to the protein: a free-molecule template (library or GAFF) would lose the bond
            partners = [f"{b.residue.name}{b.residue.id}{b.residue.chain.id}:{b.name}-{a.name}" for a, b, _d in links[res]]
            has_metal = any(_element_symbol(a) in METALS for a in heavy)
            if has_metal:
                metal_atoms.extend(a for a in heavy if _element_symbol(a) in METALS)
            row.update(source="covalent_metal" if has_metal else "covalent", partners=partners, confidence="low",
                       category=row["category"] or "covalent ligand",
                       covalent=[{"atom": a.name, "partner_resn": b.residue.name, "partner_resi": b.residue.id,
                                  "partner_chain": b.residue.chain.id, "partner_atom": b.name,
                                  "distance_A": round(d * 10.0, 2)} for a, b, d in links[res]],
                       reason=("covalently bound to " + ", ".join(partners)
                               + ("; metal-containing covalent cofactor (e.g. c-type heme) has no template here"
                                  if has_metal else "; needs a covalent (capped-fragment) template, not a free "
                                  "molecule")))
            if meta.get("notes"):
                row["notes"] = meta["notes"]
            if row["source"] == "covalent" and family == "amber14" and _cys_link(row) is not None:
                row["buildable"] = True
                row["reason"] += "; buildable: capped-fragment GAFF2 template on Cys (Amber)"
            elif (row["source"] == "covalent" and family == "amber14"
                  and _single_internal_partner(links[res], positions_nm)):
                row["buildable"] = "noncanonical"
                row["reason"] += "; buildable: residue+ligand as a GAFF2 non-canonical residue (ACE/NME caps)"
            rows.append(row)
            continue
        if _is_modified_polymer_residue(res, positions_nm):
            g_mod = residue_heavy_graph(res, positions_nm)
            aliases_mod = [str(x) for x in dict(meta.get("templates", {})).get(family, [])] + [res.name]
            entry, mapping, how = catalog.find(g_mod, names=aliases_mod, external_bonds=2)
            if entry is not None and entry["external_bonds"] > 0:
                row.update(source="library", family=family, template=entry["name"], matched_by=how,
                           template_source=entry["source"], extra_files=entry["files"], charge=entry["charge"],
                           category=row["category"] or "modified residue", atom_map=mapping, confidence="high",
                           replaces_base=entry.get("replaces_base", False))
                if meta.get("notes"):
                    row["notes"] = meta["notes"]
                rows.append(row)
                continue
            row.update(source="modified_residue", category="modified residue", confidence="low",
                       reason="non-standard residue inside a polymer chain with no residue template in this force "
                              "field (phospho/methylated/covalently modified residue): not a free-molecule GAFF case")
            # GAFF 1-4 scaling (0.833/0.5) cannot share a NonbondedForce with CHARMM36 (1.0/1.0): Amber only
            if family == "amber14" and _internal_residue(res, positions_nm):
                row["buildable"] = "noncanonical"
                row["reason"] += "; buildable as a GAFF2 non-canonical residue (ACE/NME caps)"
            rows.append(row)
            continue
        is_ion = len(heavy) == 1
        elements = {_element_symbol(a) for a in heavy}
        if elements & METALS:
            metal_atoms.extend(a for a in heavy if _element_symbol(a) in METALS)
        g = residue_heavy_graph(res, positions_nm)
        aliases = [str(x) for x in dict(meta.get("templates", {})).get(family, [])] + [res.name]
        entry, mapping, how = catalog.find(g, names=aliases)
        if entry is not None:
            row.update(source="library", family=family, template=entry["name"], matched_by=how,
                       template_source=entry["source"], extra_files=entry["files"], charge=entry["charge"],
                       category=row["category"] or ("ion" if is_ion else "component"), atom_map=mapping,
                       confidence="high" if how == "alias" or entry["name"] == res.name else "medium")
            if meta.get("notes"):
                row["notes"] = meta["notes"]
            rows.append(row)
            continue
        alt = None
        for fam in others:
            try:
                e2, m2, how2 = get_catalog(fam).find(g, names=[str(x) for x in dict(meta.get("templates", {})).get(
                    fam, [])] + [res.name])
            except Exception:
                e2 = None
            if e2 is not None:
                alt = {"family": fam, "template": e2["name"], "charge": e2["charge"]}
                break
        partial, missing = catalog.closest_partial(g, names=aliases)
        if missing:
            row.update(source="unmatched", reason=f"incomplete residue: {len(missing)} heavy atom(s) missing vs "
                       f"template {partial['name']} ({', '.join(missing[:8])}); rebuild them first",
                       confidence="low")
        elif elements & METALS:
            row.update(source="metal", reason="metal-containing molecule with no template in this force field; "
                       "needs a metal model (bonded MCPB, nonbonded, or QM/MM)", confidence="low")
        elif is_ion:
            row.update(source="unmatched", reason=f"no ion template for {name} in {family}", confidence="low")
        else:
            row.update(source="gaff", reason="organic molecule without a curated template: GAFF2/OpenFF "
                       "parameters (AM1-BCC charges) are generated for it", confidence="medium")
        if alt:
            row["available_in"] = alt
        rows.append(row)
    plan = _summarize(rows, family)
    plan["metal_coordination"] = _coordination(topology, positions_nm, metal_atoms) if metal_atoms else []
    if plan["metal_coordination"]:
        plan["warnings"].append("metal ions coordinate protein/ligand atoms; they are treated as nonbonded ions "
                                "(no metal-ligand bonds) in this model")
    return plan


def _summarize(rows, family):
    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    recommend = family
    warnings = []
    for r in rows:
        if r["source"] in ("metal", "unmatched", "modified_residue", "covalent", "covalent_metal"):
            warnings.append(f"{r['resn']} {r['chain']}{r['resi']}: {r['reason']}")
    return {"family": family, "recommended_family": recommend, "counts": counts, "components": rows,
            "warnings": warnings, "ready": all(r["source"] == "library" for r in rows)}


# --------------------------------------------------------------------------- preparation


def hydrogen_definitions_xml(entries):
    """OpenMM hydrogen-definition XML for templates (the template is the protonation authority)."""
    lines = ["<Residues>"]
    for e in entries:
        lines.append(f'  <Residue name="{e["name"]}">')
        for h, parent in e["hydrogens"]:
            lines.append(f'    <H name="{h}" parent="{parent}"/>')
        lines.append("  </Residue>")
    lines.append("</Residues>")
    return "\n".join(lines) + "\n"


def prepare_modeller(modeller, plan, family=None, renames=None):
    """
    Rename planned components to their templates (residue and heavy-atom names), drop their
    existing hydrogens, and register template-derived hydrogen definitions (OpenMM keeps them
    class-wide; a name already defined with other hydrogens is reported, not overwritten).
    The names are only needed by Modeller.addHydrogens: pass a dict as ``renames`` and call
    ``restore_names`` right after it so outputs keep the PDB names (ZN, not ZN2).
    Returns the force-field files.
    """

    family = family or plan["family"]
    catalog = get_catalog(family)
    merged_rows = [r for r in plan["components"] if r.get("merge") and r.get("source") == "library"]
    if merged_rows:
        from openmm import unit as _unit

        pos_nm = [tuple(p.value_in_unit(_unit.nanometer)) for p in modeller.positions]
        lig_atoms = {}
        for r in merged_rows:
            lig = next((x for x in modeller.topology.residues()
                        if (x.chain.id, x.id, x.name) == (r["chain"], r["resi"], r["resn"])), None)
            if lig is not None:
                lig_atoms[(r["chain"], r["resi"], r["resn"])] = lig
        # the ligand-partner bond may be missing when the file had no LINK/CONECT: add it
        for r in merged_rows:
            lig = lig_atoms.get((r["chain"], r["resi"], r["resn"]))
            if lig is None:
                continue
            links = covalent_links(modeller.topology, pos_nm).get(lig, [])
            have = {(frozenset((b[0].index, b[1].index))) for b in modeller.topology.bonds()}
            for a, b, _d in links:
                if frozenset((a.index, b.index)) not in have:
                    modeller.topology.addBond(a, b)
        origin = merge_residue_pairs(modeller, [((r["chain"], r["resi"], r["resn"]),
                                                 (r["merge"]["chain"], r["merge"]["resi"], r["merge"]["resn"]))
                                                for r in merged_rows])
        for r in merged_rows:
            key = (r["merge"]["chain"], r["merge"]["resi"], r["merge"]["resn"])
            res = next(x for x in modeller.topology.residues() if (x.chain.id, x.id, x.name) == key)
            back = {}
            names = origin.get(key, {})
            prefixed = dict(r.get("atom_map") or {})
            for a in list(res.atoms()):
                if a.element is not None and a.element.symbol == "H":
                    continue
                tmpl = prefixed.get(names.get(a.name, ""))
                if tmpl:
                    back[tmpl] = a.name
                    a.name = tmpl
            if renames is not None:
                renames[(res.chain.id, res.id, r["template"])] = {"resn": res.name, "atoms": back}
            modeller.delete([a for a in res.atoms() if a.element is not None and a.element.symbol == "H"])
            res = next(x for x in modeller.topology.residues() if (x.chain.id, x.id, x.name) == key)
            res.name = r["template"]
    wanted = {(r["chain"], r["resi"], r["resn"]): r for r in plan["components"]
              if r.get("source") == "library" and r.get("family", family) == family and not r.get("merge")}
    used, extra_files, drop, override = {}, [], [], None
    for res in modeller.topology.residues():
        row = wanted.get((res.chain.id, res.id, res.name))
        if row is None:
            continue
        mapping = dict(row.get("atom_map") or {})
        back = {}
        for a in res.atoms():
            if _element_symbol(a) == "H":
                drop.append(a)
            elif a.name in mapping:
                back[mapping[a.name]] = a.name
                a.name = mapping[a.name]
        if renames is not None:
            renames[(res.chain.id, res.id, row["template"])] = {"resn": res.name, "atoms": back}
        res.name = row["template"]
        entry = catalog.by_name[row["template"]]
        if entry["hydrogens"]:
            used[entry["name"]] = entry
        if entry.get("replaces_base"):
            override = list(entry["files"])
            continue
        for f in entry["files"]:
            if f not in extra_files:
                extra_files.append(f)
    if drop:
        modeller.delete(drop)
    _drop_unsupported_external_bonds(modeller.topology, catalog)
    for r in merged_rows:
        entry = catalog.by_name[r["template"]]
        if entry["hydrogens"]:
            used[entry["name"]] = entry
        for f in entry["files"]:
            if f not in extra_files:
                extra_files.append(f)
    if used:
        register_hydrogen_definitions(list(used.values()))
    if override:
        # the replacement set brings its own water and ion templates
        water = set(forcefield_families()[family].get("water", []))
        return list(override) + [f for f in extra_files if f not in override and f not in water]
    base = list(catalog.base_files)
    return base + [f for f in extra_files if f not in base]


_STANDARD_H_NAMES = None


def _standard_hydrogen_residues():
    """Residue names defined by OpenMM's own hydrogens.xml (never overridden)."""
    global _STANDARD_H_NAMES
    if _STANDARD_H_NAMES is None:
        import xml.etree.ElementTree as ET

        import openmm.app as app

        path = os.path.join(os.path.dirname(app.__file__), "data", "hydrogens.xml")
        _STANDARD_H_NAMES = {r.get("name") for r in ET.parse(path).getroot().iter("Residue")}
    return _STANDARD_H_NAMES


def register_hydrogen_definitions(entries):
    """
    Load template-derived hydrogen definitions (Modeller keeps them class-wide). A component name that
    another family defined differently (e.g. TEA: a GLYCAM sugar in Amber, tetraethylammonium in
    CHARMM) is re-registered for the current build; OpenMM's standard residues are never overridden.
    Returns the standard-residue names skipped because of a conflict.
    """
    import openmm.app as app

    app.Modeller._loadStandardHydrogenDefinitions()
    known = getattr(app.Modeller, "_residueHydrogens", {}) or {}
    standard = _standard_hydrogen_residues()
    todo, conflicts = [], []
    for e in entries:
        have = known.get(e["name"])
        if have is None:
            todo.append(e)
            continue
        names = {getattr(h, "name", None) for h in getattr(have, "hydrogens", [])}
        if names and names != {h for h, _p in e["hydrogens"]}:
            if e["name"] in standard:
                conflicts.append(e["name"])
            else:
                todo.append(e)
    if todo:
        path = os.path.join(tempfile.gettempdir(), "leonardo_hdefs_" + "_".join(sorted(e["name"] for e in todo))[:80]
                            + ".xml")
        with open(path, "w") as fh:
            fh.write(hydrogen_definitions_xml(todo))
        app.Modeller.loadHydrogenDefinitions(path)
    return conflicts


def restore_names(topology, renames):
    """Put back the original residue and heavy-atom names after Modeller.addHydrogens."""
    for res in topology.residues():
        info = renames.get((res.chain.id, res.id, res.name))
        if info is None:
            continue
        for a in res.atoms():
            if a.name in info["atoms"]:
                a.name = info["atoms"][a.name]
        res.name = info["resn"]


# monovalent bulk ions (crystal or added by solvation): they diffuse, they are not metal sites
BULK_ION_RESN = {"NA", "K", "LI", "RB", "CS", "SOD", "POT", "LIT", "CES", "NA+", "K+"}


def metal_sites(topology, positions_nm, cutoff_nm=COORDINATION_CUTOFF_NM, skip_resn=(), include_water=False):
    """[(metal atom, ligand atom, distance_nm)] for N/O/S atoms of other residues around metal atoms.
    Water ligands, bulk monovalent ions and ``skip_resn`` residues (e.g. a metal slab) are left out, so
    a solvated system does not get its ions pinned to their first-shell waters."""
    pos = positions_nm
    skip = {str(x).upper() for x in skip_resn or ()}
    metals = [a for a in topology.atoms() if _element_symbol(a) in METALS
              and str(a.residue.name).upper() not in BULK_ION_RESN | skip]
    ligands = [a for a in topology.atoms() if _element_symbol(a) in ("N", "O", "S")
               and str(a.residue.name).upper() not in (skip if include_water else WATER_RESN | skip)]
    out = []
    for m in metals:
        for a in ligands:
            if a.residue is m.residue:
                continue
            d = _dist(pos[m.index], pos[a.index])
            if d <= cutoff_nm:
                out.append((m, a, d))
    return out


def coordination_variants(topology, positions_nm):
    """
    Modeller.addHydrogens variants for metal ligands: Cys binding a metal through SG is a thiolate
    (CYM), His binding through NE2 carries its proton on ND1 (HID) and through ND1 on NE2 (HIE).
    """
    variants = [None] * topology.getNumResidues()
    changed = []
    for m, a, _d in metal_sites(topology, positions_nm):
        res = a.residue
        name = str(res.name).upper()
        if name in ("CYS", "CYX", "CYM") and a.name == "SG":
            variants[res.index] = "CYM"
        elif name in ("HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"):
            variants[res.index] = "HID" if a.name == "NE2" else ("HIE" if a.name == "ND1" else None)
        else:
            continue
        if variants[res.index]:
            changed.append(f"{res.name}{res.id}{res.chain.id}->{variants[res.index]} (binds {m.residue.name})")
    return variants, sorted(set(changed))


def add_coordination_restraints(system, topology, positions_nm, k_kj_nm2=50000.0, tolerance_nm=0.02, skip_resn=()):
    """
    Keep metal sites intact without a bonded metal model: a flat-bottom harmonic restraint on every
    metal-ligand distance (crystal value +/- tolerance). Covers heme Fe-His/Cys/O2, Zn-Cys/His, Mg-phosphate...
    Returns a description of each restraint.
    """
    import openmm as mm

    sites = metal_sites(topology, positions_nm, skip_resn=skip_resn)
    if not sites:
        return []
    # parameter names must not clash with other restraint forces in the same Context (e.g. a global "k")
    force = mm.CustomBondForce("0.5*leo_metal_k*(max(0, r-r0-leo_metal_tol)^2 + max(0, r0-leo_metal_tol-r)^2)")
    force.addGlobalParameter("leo_metal_k", float(k_kj_nm2))
    force.addGlobalParameter("leo_metal_tol", float(tolerance_nm))
    force.addPerBondParameter("r0")
    out = []
    for m, a, d in sites:
        force.addBond(m.index, a.index, [float(d)])
        out.append(f"{m.residue.name}{m.residue.id}:{m.name}-{a.residue.name}{a.residue.id}:{a.name} {d * 10:.2f} A")
    force.setName("LeonardoMetalCoordination")
    system.addForce(force)
    return out


def split_chains_at_breaks(topology, positions, max_link_nm=0.2):
    """
    Copy of a topology whose chains are split where consecutive polymer residues are not linked
    (C-N or O3'-P farther than max_link_nm: missing loops, cut-out sites), so each segment gets
    proper termini instead of a stretched bond across the gap. Returns (topology, breaks).
    """
    import openmm.app as app
    from openmm import unit

    pos = [p.value_in_unit(unit.nanometer) for p in positions]
    new = app.Topology()
    new.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    atom_map = {}
    breaks = []
    cut = set()
    for chain in topology.chains():
        current, prev = None, None
        for res in chain.residues():
            linked = False
            if prev is not None:
                pa = {a.name: a for a in prev.atoms()}
                ra = {a.name: a for a in res.atoms()}
                for x, y in (("C", "N"), ("O3'", "P")):
                    if x in pa and y in ra:
                        linked = _dist(pos[pa[x].index], pos[ra[y].index]) <= max_link_nm
                        if not linked:
                            breaks.append(f"{chain.id}:{prev.name}{prev.id}-{res.name}{res.id}")
                            cut.add((prev, res))
                        break
                else:
                    linked = True        # ligands/ions/waters stay in their chain
            if current is None or (prev is not None and not linked):
                current = new.addChain(chain.id)
            nr = new.addResidue(res.name, current, res.id, res.insertionCode)
            for a in res.atoms():
                atom_map[a] = new.addAtom(a.name, a.element, nr, a.id)
            prev = res
    for bond in topology.bonds():
        a, b = bond[0], bond[1]
        if (a.residue, b.residue) in cut or (b.residue, a.residue) in cut:
            continue
        new.addBond(atom_map[a], atom_map[b])
    return new, breaks


def additive_residue_names(library=None):
    """Crystallization additives (sulfate, glycerol, PEG...) that are removed before building by default."""
    comps = (library or load_library()).get("components", {})
    return {k for k, v in comps.items() if str(v.get("category", "")) == "buffer"}


def _drop_unsupported_external_bonds(topology, catalog):
    """
    Remove inter-residue bonds of components whose template has no external bonds (metal
    coordination read from LINK/CONECT records: Fe-His, Fe-O2, Zn-Cys...). These contacts stay
    as nonbonded interactions; bonded metal/heme models add explicit patches instead.
    """
    removed = []
    keep = []
    for bond in topology._bonds:
        a, b = bond[0], bond[1]
        metal_bond = _element_symbol(a) in METALS or _element_symbol(b) in METALS
        if a.residue is not b.residue and metal_bond:
            ta = catalog.by_name.get(a.residue.name)
            tb = catalog.by_name.get(b.residue.name)
            no_ext = (ta is not None and ta["external_bonds"] == 0) or (tb is not None and tb["external_bonds"] == 0)
            if no_ext:
                removed.append(f"{a.residue.name}{a.residue.id}:{a.name}-{b.residue.name}{b.residue.id}:{b.name}")
                continue
        keep.append(bond)
    topology._bonds = keep
    return removed


# --------------------------------------------------------------------------- GAFF route

CCD_URL = "https://files.rcsb.org/ligands/view/{code}_ideal.sdf"


def ccd_reference(code, fetch=True, timeout=20):
    """RDKit molecule of a PDB chemical component (bond orders), cached under ~/.leonardo/parameters/ccd."""
    from rdkit import Chem

    code = str(code or "").strip().upper()
    if not code:
        return None
    path = os.path.join(_cache_dir(), "ccd", f"{code}.sdf")
    if not os.path.isfile(path) and fetch:
        try:
            import urllib.request

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with urllib.request.urlopen(CCD_URL.format(code=code), timeout=timeout) as resp:
                data = resp.read()
            if b"M  END" in data:
                with open(path, "wb") as fh:
                    fh.write(data)
        except Exception:
            return None
    if not os.path.isfile(path):
        return None
    mol = Chem.MolFromMolFile(path, removeHs=False)
    if mol is None:
        return None
    params = Chem.RemoveHsParameters()
    params.removeDefiningBondStereo = True       # the CCD keeps imine/amidine H that define E/Z
    return Chem.RemoveHs(mol, params)


def residue_heavy_mol(residue, positions_nm):
    """RDKit molecule (single bonds) of a residue's heavy atoms with their coordinates; returns (mol, names)."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    g = residue_heavy_graph(residue, positions_nm)
    names = [n for n in g.nodes]
    rw = Chem.RWMol()
    for n in names:
        rw.AddAtom(Chem.Atom(g.nodes[n]["element"].capitalize()))
    for u, v in g.edges:
        rw.AddBond(names.index(u), names.index(v), Chem.BondType.SINGLE)
    conf = Chem.Conformer(len(names))
    for i, n in enumerate(names):
        p = positions_nm[g.nodes[n]["index"]]
        conf.SetAtomPosition(i, Point3D(10.0 * p[0], 10.0 * p[1], 10.0 * p[2]))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    return mol, names


def _bond_orders_from_geometry(mol):
    """Bond orders of a heavy-atom molecule perceived by Open Babel from its 3D coordinates (atom order kept)."""
    from openbabel import pybel
    from rdkit import Chem
    from rdkit.Chem import AllChem

    block = Chem.MolToPDBBlock(mol, flavor=4)
    ob = pybel.readstring("pdb", block)
    ob.OBMol.PerceiveBondOrders()
    # SDF keeps Open Babel's atom order (= ours) and coordinates
    ref = Chem.MolFromMolBlock(ob.write("sdf"), removeHs=True, sanitize=True)
    if ref is None or ref.GetNumAtoms() != mol.GetNumAtoms():
        raise ValueError("could not perceive bond orders from the geometry")
    if any(ref.GetAtomWithIdx(i).GetSymbol() != mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())):
        ref = AllChem.AssignBondOrdersFromTemplate(ref, mol)
    return _bond_lengths_check(ref)


# single/double thresholds (A) for non-aromatic bonds: longer "doubles" are demoted, shorter "singles" promoted
_DOUBLE_MAX = {frozenset(("C",)): 1.44, frozenset(("C", "N")): 1.40, frozenset(("C", "O")): 1.33,
               frozenset(("N",)): 1.35}
_SINGLE_MIN = {frozenset(("C",)): 1.39, frozenset(("C", "N")): 1.35, frozenset(("C", "O")): 1.30}
_MAX_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2}


def _bond_lengths_check(mol):
    """
    Correct bond orders the crystal geometry contradicts (Open Babel may read a protonated Schiff base
    C=N as an amine, or misplace double bonds in a polyene). Aromatic bonds are left alone; the change
    is kept only if the molecule still sanitizes.
    """
    from rdkit import Chem

    if mol.GetNumConformers() == 0:
        return mol
    conf = mol.GetConformer()
    rw = Chem.RWMol(mol)
    try:
        Chem.Kekulize(rw, clearAromaticFlags=False)
    except Exception:
        return mol
    changed = False

    def length(b):
        return (conf.GetAtomPosition(b.GetBeginAtomIdx()) - conf.GetAtomPosition(b.GetEndAtomIdx())).Length()

    def pair(b):
        return frozenset((b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()))

    def heavy_valence(a):
        return sum(int(round(x.GetBondTypeAsDouble())) for x in a.GetBonds())

    for b in rw.GetBonds():                                   # 1) demote long "double" bonds
        if b.GetIsAromatic() or b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        lim = _DOUBLE_MAX.get(pair(b))
        if lim and length(b) > lim:
            b.SetBondType(Chem.BondType.SINGLE)
            changed = True
    for b in sorted(rw.GetBonds(), key=length):              # 2) promote short "single" bonds, shortest first
        if b.GetIsAromatic() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        lim = _SINGLE_MIN.get(pair(b))
        if not lim or length(b) >= lim:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if all(heavy_valence(a) + 1 <= _MAX_VALENCE.get(a.GetSymbol(), 0) for a in (a1, a2)):
            if any(x.GetBondType() == Chem.BondType.DOUBLE for a in (a1, a2) for x in a.GetBonds()):
                continue                                      # already conjugated there: keep single
            b.SetBondType(Chem.BondType.DOUBLE)
            changed = True
    if not changed:
        return mol
    out = rw.GetMol()
    for a in out.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return mol


# aromatic ring bonds are 1.33-1.42 A in small-molecule structures; crystal ligands at 2-3 A
# resolution stretch them to ~1.47 (imatinib in 1IEP), a saturated C-C is 1.53
_AROMATIC_RING_BOND_MAX = 1.50
_AROMATIC_RING_BOND_MEAN_MAX = 1.47
_AROMATIC_RING_BOND_S_MAX = 1.80


def _planar_rings(mol, max_dev=0.12):
    """Atom-index tuples of the 5/6-membered rings whose atoms lie in a plane and whose bonds are
    all short enough to be aromatic (a cyclohexane or a saturated ring fails both)."""
    import numpy as np

    conf = mol.GetConformer()
    out = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6):
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if any(a.GetSymbol() not in ("C", "N", "O", "S") for a in atoms):
            continue
        if any(a.GetSymbol() == "C" and a.GetDegree() > 3 for a in atoms):
            continue
        pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                        for i in ring])
        centered = pts - pts.mean(axis=0)
        normal = np.linalg.svd(centered)[2][-1]
        if float(np.max(np.abs(centered @ normal))) > max_dev:
            continue
        lengths = []
        ok = True
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            d = (conf.GetAtomPosition(a) - conf.GetAtomPosition(b)).Length()
            if "S" in (mol.GetAtomWithIdx(a).GetSymbol(), mol.GetAtomWithIdx(b).GetSymbol()):
                ok = ok and d <= _AROMATIC_RING_BOND_S_MAX
            else:
                ok = ok and d <= _AROMATIC_RING_BOND_MAX
                lengths.append(d)
        if ok and lengths and sum(lengths) / len(lengths) <= _AROMATIC_RING_BOND_MEAN_MAX:
            out.append(tuple(ring))
    return out


def _bond_orders_from_planarity(mol):
    """
    Last-resort bond orders of a heavy-atom molecule from its 3D geometry: flat rings with short
    bonds are aromatic, short exocyclic C=O / C=N / C=C / S=O are double, very short C-C / C-N are
    triple. Pyrrole-type ring N get their H by trial until RDKit can kekulize. Open Babel misreads
    some heteroaromatics (imatinib's pyrimidine); this keeps a flat ring flat.
    """
    import itertools

    from rdkit import Chem

    rw = Chem.RWMol(mol)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
        a.SetFormalCharge(0)
    rw.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(rw)
    conf = rw.GetConformer()
    aromatic = set()
    for ring in _planar_rings(rw):
        for k in range(len(ring)):
            bond = rw.GetBondBetweenAtoms(ring[k], ring[(k + 1) % len(ring)])
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)
        for i in ring:
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
            aromatic.add(i)

    def length(b):
        return (conf.GetAtomPosition(b.GetBeginAtomIdx()) - conf.GetAtomPosition(b.GetEndAtomIdx())).Length()

    def free_valence(a):
        used = sum(1.5 if x.GetIsAromatic() else x.GetBondTypeAsDouble() for x in a.GetBonds())
        return {"C": 4, "N": 3, "O": 2, "S": 6, "P": 5}.get(a.GetSymbol(), 0) - used

    for b in sorted(rw.GetBonds(), key=length):
        if b.GetIsAromatic():
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        pair = frozenset((a1.GetSymbol(), a2.GetSymbol()))
        d = length(b)
        if pair == frozenset(("C",)) and d <= 1.24 or pair == frozenset(("C", "N")) and d <= 1.20:
            if free_valence(a1) >= 2 and free_valence(a2) >= 2:
                b.SetBondType(Chem.BondType.TRIPLE)
            continue
        limits = {frozenset(("C", "O")): 1.30, frozenset(("C", "N")): 1.32, frozenset(("C",)): 1.38,
                  frozenset(("S", "O")): 1.52, frozenset(("P", "O")): 1.55, frozenset(("N", "O")): 1.26}
        lim = limits.get(pair)
        if lim is None or d > lim:
            continue
        if a1.GetIsAromatic() and a2.GetIsAromatic():
            continue
        if any(a.GetIsAromatic() and a.GetSymbol() == "C" for a in (a1, a2)) and pair != frozenset(("C", "O")):
            continue
        if free_valence(a1) >= 1 and free_valence(a2) >= 1:
            b.SetBondType(Chem.BondType.DOUBLE)
    for a in rw.GetAtoms():
        if a.GetSymbol() == "N" and a.GetDegree() == 4:
            a.SetFormalCharge(1)             # quaternary ammonium
    candidates = [i for i in aromatic if rw.GetAtomWithIdx(i).GetSymbol() == "N"
                  and rw.GetAtomWithIdx(i).GetDegree() == 2]
    base = rw.GetMol()
    for n in range(0, min(3, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, n):
            trial = Chem.RWMol(base)
            for i in combo:
                trial.GetAtomWithIdx(i).SetNumExplicitHs(1)
            try:
                out = trial.GetMol()
                Chem.SanitizeMol(out)
                return out
            except Exception:
                continue
    raise ValueError("could not assign bond orders from the geometry")


def _geometry_contradicts(mol, crystal):
    """True when ``mol`` (bond orders from some source) leaves a flat aromatic-looking ring of
    ``crystal`` without any aromatic or double bond (Open Babel read tetrabromobenzotriazole's
    benzene ring as a tetrabromocyclohexene)."""
    from rdkit import Chem

    probe = Chem.RWMol(crystal)
    probe.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(probe)
    for ring in _planar_rings(probe):
        for i in ring:
            a = mol.GetAtomWithIdx(i)
            if a.GetSymbol() == "C" and not a.GetIsAromatic() and not any(
                    b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC) for b in a.GetBonds()):
                return True
    return False


GENERIC_LIGAND_NAMES = frozenset({"UNL", "UNK", "LIG", "MOL", "DRG", "INH", "XXX", "LG1", "L01", "UNX"})


def ligand_bond_orders_from_pdb(pdb_block, resn="", smiles_hint="", fetch_ccd=True, ccd_timeout=6):
    """
    Heavy-atom RDKit molecule of one ligand residue given as a PDB block, with bond orders.

    A PDB block carries no bond orders: RDKit's proximity bonding makes every bond single, so an
    aromatic ring is read as a cyclohexane and protonating that molecule roughly doubles the
    hydrogens (benzamidine got 17 H instead of 9, a quinazoline N became an amine). Bond orders
    come from, in order: the PDB chemical component (``resn``), ``smiles_hint``, Open Babel's
    perception from the 3D geometry. Atom order, coordinates and PDB atom names of the input are
    kept. Returns ``(mol, source)``; ``mol`` is None when nothing worked (``source`` says why).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    raw = Chem.MolFromPDBBlock(str(pdb_block or ""), removeHs=False, sanitize=False)
    if raw is None or raw.GetNumAtoms() == 0:
        return None, "RDKit could not read the ligand coordinates"
    heavy = Chem.RWMol(raw)
    for idx in sorted((a.GetIdx() for a in raw.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
        heavy.RemoveAtom(idx)
    crystal = heavy.GetMol()
    for atom in crystal.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetFormalCharge(0)
    try:
        crystal.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    n_heavy = crystal.GetNumAtoms()

    def _finish(mol, source):
        out = Chem.Mol(mol)
        for i, atom in enumerate(out.GetAtoms()):
            info = crystal.GetAtomWithIdx(i).GetPDBResidueInfo()
            if info is not None:
                atom.SetMonomerInfo(info)
        Chem.SanitizeMol(out)
        return out, source

    templates = []
    code = str(resn or "").strip().upper()
    if code in GENERIC_LIGAND_NAMES:
        code = ""                          # docking/MD placeholder names, not chemical components
    try:
        ref = ccd_reference(code, fetch=fetch_ccd, timeout=ccd_timeout) if code else None
    except Exception:
        ref = None
    if ref is not None:
        templates.append((ref, f"PDB chemical component {str(resn).strip().upper()}"))
    if smiles_hint:
        try:
            hint = Chem.MolFromSmiles(str(smiles_hint))
            if hint is not None:
                templates.append((Chem.RemoveHs(hint), "SMILES template"))
        except Exception:
            pass
    for ref, source in templates:
        ref_heavy = Chem.RemoveHs(ref, sanitize=False)
        if ref_heavy.GetNumAtoms() != n_heavy:
            continue                       # a missing/extra atom in the crystal: the template cannot map
        try:
            return _finish(AllChem.AssignBondOrdersFromTemplate(ref_heavy, crystal), source)
        except Exception:
            continue
    errors = []
    try:
        ob_mol = _bond_orders_from_geometry(crystal)
        if not _geometry_contradicts(ob_mol, crystal):
            return _finish(ob_mol, "inferred from the geometry (Open Babel)")
        errors.append("Open Babel left a flat ring saturated")
    except Exception as exc:
        errors.append(f"Open Babel: {exc}")
    try:
        return _finish(_bond_orders_from_planarity(crystal), "inferred from ring planarity and bond lengths")
    except Exception as exc:
        errors.append(f"geometry: {exc}")
    return None, "no bond orders: no matching chemical component template; " + "; ".join(errors)


def assign_ligand_chemistry(residue, positions_nm, ph=7.4, fetch_ccd=True):
    """
    Heavy-atom molecule of a ligand residue with bond orders (CCD template, else inferred from the
    geometry), protonated at ``ph`` with Leonardo's pKa model and hydrogens placed. Returns
    (rdkit mol with H, heavy-atom names, info).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol, names = residue_heavy_mol(residue, positions_nm)
    info = {"bond_orders": "", "protonation": ""}
    ref = ccd_reference(residue.name, fetch=fetch_ccd)
    assigned = None
    if ref is not None and ref.GetNumAtoms() == mol.GetNumAtoms():
        try:
            assigned = AllChem.AssignBondOrdersFromTemplate(ref, mol)
            info["bond_orders"] = "PDB chemical component dictionary"
        except Exception:
            assigned = None
    if assigned is None:
        assigned = _bond_orders_from_geometry(mol)
        info["bond_orders"] = "inferred from the crystal geometry (Open Babel)"
    Chem.SanitizeMol(assigned)
    try:
        from .pka_runtime import protonate_molecule

        prot = protonate_molecule(assigned, ph=float(ph))
        prot = prot[0] if isinstance(prot, tuple) else prot
        info["protonation"] = f"Leonardo pKa model at pH {float(ph):.1f}"
    except Exception as exc:
        prot = Chem.AddHs(assigned, addCoords=True)
        info["protonation"] = f"neutral (pKa model unavailable: {exc})"
    if prot.GetNumConformers() == 0 or any(prot.GetAtomWithIdx(i).GetSymbol() != assigned.GetAtomWithIdx(i).GetSymbol()
                                           for i in range(assigned.GetNumAtoms())):
        raise ValueError(f"protonation changed the heavy atoms of {residue.name}")
    info["formal_charge"] = int(Chem.GetFormalCharge(prot))
    info["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(prot))
    return prot, names, info


def _ligand_topology(resname, chain_id, res_id, mol, heavy_names):
    """OpenMM topology + positions (nm) for an RDKit ligand with hydrogens (heavy names kept)."""
    import openmm.app as app
    from openmm import Vec3, unit

    top = app.Topology()
    chain = top.addChain(chain_id)
    res = top.addResidue(resname, chain, res_id)
    atoms, pos, used = [], [], set(heavy_names)
    hcount = 0
    conf = mol.GetConformer()
    for i, a in enumerate(mol.GetAtoms()):
        if i < len(heavy_names):
            name = heavy_names[i]
        else:
            hcount += 1
            name = f"H{hcount}"
            while name in used:
                hcount += 1
                name = f"H{hcount}"
            used.add(name)
        atoms.append(top.addAtom(name, app.Element.getBySymbol(a.GetSymbol()), res))
        p = conf.GetAtomPosition(i)
        pos.append(Vec3(p.x, p.y, p.z) * 0.1)
    for b in mol.GetBonds():
        top.addBond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])
    return top, unit.Quantity(pos, unit.nanometer)


def _ensure_ambertools_on_path():
    """antechamber/sqm (AM1-BCC charges) live in the conda env; PyMOL launched elsewhere may lack them."""
    import shutil
    import sys

    if shutil.which("antechamber"):
        return True
    for cand in (os.path.join(sys.prefix, "bin"), os.path.join(os.environ.get("CONDA_PREFIX", ""), "bin")):
        if cand and os.path.isfile(os.path.join(cand, "antechamber")):
            os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
            os.environ.setdefault("AMBERHOME", os.path.dirname(cand))
            return True
    return False


def install_gaff_components(modeller, plan, ph=7.4, forcefield="gaff-2.11", fetch_ccd=True):
    """
    Replace every plan row routed to GAFF by its protonated, hydrogen-complete model and return a
    GAFF template generator for them (AM1-BCC charges, cached in ~/.leonardo/parameters). Returns
    (generator or None, per-component info).
    """
    _ensure_ambertools_on_path()
    from openff.toolkit import Molecule
    from openmm import unit
    from openmmforcefields.generators import GAFFTemplateGenerator

    rows = {(r["chain"], r["resi"], r["resn"]): r for r in plan["components"] if r.get("source") == "gaff"}
    if not rows:
        return None, []
    pos_nm = [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]
    molecules, info, done = {}, [], set()
    targets = [res for res in modeller.topology.residues() if (res.chain.id, res.id, res.name) in rows]
    built = []
    for res in targets:
        mol, names, meta = assign_ligand_chemistry(res, pos_nm, ph=ph, fetch_ccd=fetch_ccd)
        built.append((res, mol, names, meta))
        if res.name not in molecules:
            molecules[res.name] = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        if res.name not in done:
            done.add(res.name)
            info.append(dict(meta, resn=res.name, route=forcefield + " + AM1-BCC"))
    for res, mol, names, _meta in built:
        chain_id, res_id, resname = res.chain.id, res.id, res.name
        # Modeller.delete rebuilds the topology: residue objects from before the previous delete are
        # silently ignored, so look the residue up again in the current topology
        current = next(r for r in modeller.topology.residues() if (r.chain.id, r.id, r.name) == (chain_id, res_id,
                                                                                                    resname))
        modeller.delete([current])
        top, pos = _ligand_topology(resname, chain_id, res_id, mol, names)
        modeller.add(top, pos)
    cache = os.path.join(_cache_dir(), f"{forcefield}_am1bcc_cache.json")
    gen = GAFFTemplateGenerator(molecules=list(molecules.values()), forcefield=forcefield, cache=cache)
    return gen, info


# --------------------------------------------------------------------------- covalent (Cys) route


def _cys_link(row):
    """(ligand atom name, partner) when a covalent row is a single bond to a Cys SG (buildable), else None."""
    links = row.get("covalent") or []
    if len(links) != 1:
        return None
    link = links[0]
    if str(link["partner_resn"]).upper() in ("CYS", "CYX", "CYM") and link["partner_atom"] == "SG":
        return link
    return None


def _gaff_tables(xml_text):
    """Parse a GAFF ffxml: types, per-class bonded parameters, nonbonded, residue atoms (in molecule order)."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    types = {t.get("name"): dict(t.attrib) for t in root.find("AtomTypes")}
    bonds, angles, propers, impropers = {}, {}, {}, []
    for b in root.find("HarmonicBondForce") or []:
        k = (b.get("class1"), b.get("class2"))
        bonds[k] = bonds[k[::-1]] = dict(b.attrib)
    for a in root.find("HarmonicAngleForce") or []:
        k = (a.get("class1"), a.get("class2"), a.get("class3"))
        angles[k] = angles[k[::-1]] = dict(a.attrib)
    tor = root.find("PeriodicTorsionForce")
    for t in tor if tor is not None else []:
        if t.tag == "Proper":
            k = tuple(t.get(f"class{i}") for i in range(1, 5))
            propers[k] = propers[k[::-1]] = dict(t.attrib)
        else:
            impropers.append(dict(t.attrib))
    nb = root.find("NonbondedForce")
    lj = {a.get("class"): dict(a.attrib) for a in nb.findall("Atom")}
    res = root.find("Residues").find("Residue")
    atoms = [dict(a.attrib) for a in res.findall("Atom")]
    return {"types": types, "bonds": bonds, "angles": angles, "propers": propers, "impropers": impropers,
            "lj": lj, "atoms": atoms, "nb_attrs": dict(nb.attrib)}


def _retag(entry, new_classes, prefix):
    out = {k: v for k, v in entry.items() if not k.startswith("class")}
    for i, c in enumerate(new_classes, start=1):
        out[f"{prefix}{i}"] = c
    return out


def _openff_molecule(mol):
    """OpenFF Molecule from RDKit with stereo taken from the 3D coordinates (E/Z of C=N+ etc.)."""
    from openff.toolkit import Molecule
    from rdkit import Chem

    work = Chem.Mol(mol)
    try:
        Chem.AssignStereochemistryFrom3D(work)
    except Exception:
        pass
    try:
        off = Molecule.from_rdkit(work, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    except Exception:
        for b in work.GetBonds():
            b.SetStereo(Chem.BondStereo.STEREONONE)
            b.SetBondDir(Chem.BondDir.NONE)
        off = Molecule.from_rdkit(work, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    # the 3D coordinates carry the double-bond geometry; OpenFF's E/Z labels of C=N+(H) cannot always be
    # mapped back to RDKit (GAFF generator), so they are dropped (atom chirality is kept)
    for bond in off.bonds:
        bond._stereochemistry = None
    return off


def _residue_signature(xml):
    """Atom names/elements and bonds of the (single) residue template in a ForceField XML string: two
    copies of a component with the same signature can share one template (OpenMM refuses two
    non-identical templates matching the same residue graph)."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    types = {t.get("name"): t.get("element") for t in root.iter("Type")}
    res = next(root.iter("Residue"))
    atoms = frozenset((a.get("name"), types.get(a.get("type"))) for a in res.iter("Atom"))
    bonds = frozenset(frozenset((b.get("atomName1"), b.get("atomName2"))) for b in res.iter("Bond"))
    ext = frozenset(e.get("atomName") for e in res.iter("ExternalBond"))
    return atoms, bonds, ext


class _TemplateRegistry:
    """Per-build bookkeeping of generated templates: identical chemistry -> one shared template,
    same residue name with different chemistry -> a numbered variant."""

    def __init__(self):
        self.rows = {}              # base name -> [(signature, template name)]

    def lookup(self, base, xml):
        sig = _residue_signature(xml)
        rows = self.rows.setdefault(base, [])
        for s, tname in rows:
            if s == sig:
                return tname, None
        suffix = "" if not rows else str(len(rows) + 1)
        return None, suffix

    def add(self, base, xml, tname):
        self.rows.setdefault(base, []).append((_residue_signature(xml), tname))


def build_capped_template(mol, cap_classes, resname, keep_names, external, forcefield="gaff-2.11", tag="cov",
                          suffix=""):
    """
    Generic capped-fragment template. ``mol``: protonated molecule with conformer = the residue of interest
    plus cap atoms standing in for the neighbouring protein atoms; ``cap_classes``: {cap atom index: protein
    atom class} for every cap atom (heavy and H); ``keep_names``: {kept atom index: name} for heavy atoms
    (H get H1, H2...); ``external``: kept atom indices bonded to the protein. GAFF2 + AM1-BCC on the capped
    molecule; the cap is removed, its charge folded onto the external atoms, and every bonded term through the
    junction re-emitted with protein classes (terms made only of cap atoms belong to the protein and are not
    emitted; torsions ending on a cap atom beyond the first shell use a wildcard class). Returns
    (ForceField XML, template name, info, {mol index: kept name or None}).
    """
    from xml.sax.saxutils import quoteattr

    from openmmforcefields.generators import GAFFTemplateGenerator
    from rdkit import Chem

    _ensure_ambertools_on_path()
    cap = set(cap_classes)
    external = list(external)
    offmol = _openff_molecule(mol)
    gen = GAFFTemplateGenerator(molecules=[offmol], forcefield=forcefield,
                                cache=os.path.join(_cache_dir(), f"{forcefield}_am1bcc_cache.json"))
    tab = _gaff_tables(gen.generate_residue_template(offmol))
    cls = [tab["types"][a["type"]]["class"] for a in tab["atoms"]]
    q = [float(a["charge"]) for a in tab["atoms"]]

    def c_of(i):
        return cap_classes.get(i, cls[i])

    keep = [i for i in range(mol.GetNumAtoms()) if i not in cap]
    # cap charge -> the external atom it is bonded to
    lig_q = {i: q[i] for i in keep}
    nbr = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()] for i in range(mol.GetNumAtoms())}
    for e in external:
        cap_side = set()
        stack = [n for n in nbr[e] if n in cap]
        while stack:
            x = stack.pop()
            if x in cap_side:
                continue
            cap_side.add(x)
            stack += [n for n in nbr[x] if n in cap and n not in cap_side]
        lig_q[e] += sum(q[x] for x in cap_side)
    formal = int(round(Chem.GetFormalCharge(mol)))
    residual = formal - sum(lig_q.values())
    for e in external:
        lig_q[e] += residual / max(1, len(external))
    names, used, hn = {}, set(keep_names.values()), 0
    for i in keep:
        if i in keep_names:
            names[i] = keep_names[i]
        else:
            hn += 1
            while f"H{hn}" in used:
                hn += 1
            names[i] = f"H{hn}"
            used.add(names[i])
    tname = f"{resname}_{tag.upper()}"[:12] + suffix
    prefix = f"{tag}{resname}{suffix}-"
    types_used = sorted({cls[i] for i in keep})
    lines = ["<ForceField>", " <AtomTypes>"]
    for c in types_used:
        src = next(t for t in tab["types"].values() if t["class"] == c)
        lines.append(f'  <Type name={quoteattr(prefix + c)} class={quoteattr(c)} element={quoteattr(src["element"])} '
                     f'mass={quoteattr(src["mass"])}/>')
    lines += [" </AtomTypes>", " <Residues>", f"  <Residue name={quoteattr(tname)}>"]
    for i in keep:
        lines.append(f"   <Atom name={quoteattr(names[i])} type={quoteattr(prefix + cls[i])} charge=\"{lig_q[i]:.6f}\"/>")
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in names and j in names:
            lines.append(f"   <Bond atomName1={quoteattr(names[i])} atomName2={quoteattr(names[j])}/>")
    for e in external:
        lines.append(f"   <ExternalBond atomName={quoteattr(names[e])}/>")
    lines += ["  </Residue>", " </Residues>"]
    first_shell = {n for e in external for n in nbr[e] if n in cap}
    rows = {"bond": [], "angle": [], "proper": []}
    seen, missing = set(), []

    def add(kind, idx, table):
        if all(i in cap for i in idx):
            return True                                   # protein-internal term
        entry = table.get(tuple(cls[i] for i in idx))
        if entry is None:
            return False
        new = [c_of(i) for i in idx]
        if kind == "proper":
            for pos in (0, 3):
                if idx[pos] in cap and idx[pos] not in first_shell and idx[3 - pos] not in cap:
                    new[pos] = ""                           # any protein neighbour of that cap atom
        new = tuple(new)
        if (kind, new) in seen or (kind, new[::-1]) in seen:
            return True
        seen.add((kind, new))
        rows[kind].append(_retag(entry, new, "class"))
        return True

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if not add("bond", (i, j), tab["bonds"]):
            missing.append(("bond", i, j))
    for j in range(mol.GetNumAtoms()):
        for x in nbr[j]:
            for y in nbr[j]:
                if x < y and not add("angle", (x, j, y), tab["angles"]):
                    missing.append(("angle", x, j, y))
    for b in mol.GetBonds():
        j, k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        for i in nbr[j]:
            for m in nbr[k]:
                if len({i, j, k, m}) == 4 and not add("proper", (i, j, k, m), tab["propers"]):
                    missing.append(("proper", i, j, k, m))
    lines.append(" <HarmonicBondForce>")
    lines += [f"  <Bond {' '.join(f'{k}={quoteattr(v)}' for k, v in r.items())}/>" for r in rows["bond"]]
    lines += [" </HarmonicBondForce>", " <HarmonicAngleForce>"]
    lines += [f"  <Angle {' '.join(f'{k}={quoteattr(v)}' for k, v in r.items())}/>" for r in rows["angle"]]
    lines += [" </HarmonicAngleForce>", " <PeriodicTorsionForce>"]
    lines += [f"  <Proper {' '.join(f'{k}={quoteattr(v)}' for k, v in r.items())}/>" for r in rows["proper"]]
    cap_gaff = {cls[i] for i in cap} - {cls[i] for i in keep}
    lines += [f"  <Improper {' '.join(f'{k}={quoteattr(v)}' for k, v in r.items())}/>" for r in tab["impropers"]
              if not {r.get(f"class{n}") for n in range(1, 5)} & cap_gaff]
    lines.append(" </PeriodicTorsionForce>")
    nb = tab["nb_attrs"]
    lines.append(f' <NonbondedForce coulomb14scale={quoteattr(nb["coulomb14scale"])} lj14scale={quoteattr(nb["lj14scale"])}>')
    lines.append('  <UseAttributeFromResidue name="charge"/>')
    for c in types_used:
        lj = tab["lj"][c]
        lines.append(f'  <Atom type={quoteattr(prefix + c)} sigma={quoteattr(lj["sigma"])} epsilon={quoteattr(lj["epsilon"])}/>')
    lines += [" </NonbondedForce>", "</ForceField>"]
    info = {"template": tname, "external_atoms": [names[e] for e in external], "formal_charge": formal,
            "charge_folded_from_caps": round(sum(lig_q.values()) - sum(q[i] for i in keep) - residual, 4),
            "residual_charge_moved": round(residual, 4), "missing_terms": len(missing)}
    return "\n".join(lines) + "\n", tname, info, {i: names.get(i) for i in range(mol.GetNumAtoms())}


def build_cys_adduct_template(mol, n_lig, heavy_names, resname, sg_class="S", cb_class="2C", hb_class="H1",
                              forcefield="gaff-2.11", suffix=""):
    """Covalent ligand on a Cys SG (see build_capped_template): cap S [n_lig] and C [n_lig+1] + methyl H."""
    s_idx, c_idx = n_lig, n_lig + 1
    cap = {s_idx: sg_class, c_idx: cb_class}
    for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            cap[nb.GetIdx()] = hb_class
    attach = [nb.GetIdx() for nb in mol.GetAtomWithIdx(s_idx).GetNeighbors() if nb.GetIdx() != c_idx]
    if len(attach) != 1:
        raise ValueError("the cap sulfur must bond exactly one ligand atom")
    xml, tname, info, names = build_capped_template(mol, cap, resname, dict(enumerate(heavy_names)), attach,
                                                    forcefield=forcefield, tag="cys", suffix=suffix)
    info["attach_atom"] = names[attach[0]]
    info["charge_moved_to_attach_atom"] = round(info["charge_folded_from_caps"] + info["residual_charge_moved"], 4)
    info["names"] = [n for n in names.values() if n]
    return xml, tname, info, [names.get(i) for i in range(mol.GetNumAtoms())]


def install_covalent_cys_components(modeller, plan, ph=7.4):
    """
    Build every Cys-linked covalent ligand of the plan (Amber): capped-fragment GAFF2/AM1-BCC
    template with an external bond, partner Cys -> CYX, and the ligand-SG bond in the topology.
    Returns (extra ForceField XML files, {residue index: 'CYX'} variants, info).
    """
    import openmm.app as app
    from openmm import unit
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rows = [r for r in plan["components"] if r.get("source") == "covalent" and r.get("buildable") is True]
    if not rows:
        return [], {}, []
    catalog = get_catalog("amber14")
    base_ff = app.ForceField(*catalog.base_files)
    cyx_types = {a.name: a.type for a in base_ff._templates["CYX"].atoms}
    ff_types = base_ff._atomTypes
    sg_cls, cb_cls, hb_cls = (ff_types[cyx_types[n]].atomClass for n in ("SG", "CB", "HB2"))
    files, info, cys_residues = [], [], []
    registry = _TemplateRegistry()
    for row in rows:
        pos_nm = [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]    # re-indexed per copy
        link = _cys_link(row)
        lig = next(r for r in modeller.topology.residues()
                   if (r.chain.id, r.id, r.name) == (row["chain"], row["resi"], row["resn"]))
        cys = next(r for r in modeller.topology.residues()
                   if r.chain.id == link["partner_chain"] and r.id == link["partner_resi"]
                   and str(r.name).upper() in ("CYS", "CYX", "CYM"))
        cys_atoms = {a.name: a for a in cys.atoms()}
        heavy, names = residue_heavy_mol(lig, pos_nm)
        rw = Chem.RWMol(heavy)
        n_lig = rw.GetNumAtoms()
        s_i = rw.AddAtom(Chem.Atom("S"))
        c_i = rw.AddAtom(Chem.Atom("C"))
        rw.AddBond(names.index(link["atom"]), s_i, Chem.BondType.SINGLE)
        rw.AddBond(s_i, c_i, Chem.BondType.SINGLE)
        conf = rw.GetConformer()
        for idx, nm in ((s_i, "SG"), (c_i, "CB")):
            p = pos_nm[cys_atoms[nm].index]
            conf.SetAtomPosition(idx, Point3D(10 * p[0], 10 * p[1], 10 * p[2]))
        capped = _bond_orders_from_geometry(rw.GetMol())
        Chem.SanitizeMol(capped)
        try:
            from .pka_runtime import protonate_molecule

            prot = protonate_molecule(capped, ph=float(ph))
            prot = prot[0] if isinstance(prot, tuple) else prot
        except Exception:
            prot = Chem.AddHs(capped, addCoords=True)
        xml, tname, meta, all_names = build_cys_adduct_template(prot, n_lig, names, row["resn"], sg_cls, cb_cls, hb_cls)
        shared, suffix = registry.lookup(("cys", row["resn"]), xml)
        if suffix:
            xml, tname, meta, all_names = build_cys_adduct_template(prot, n_lig, names, row["resn"], sg_cls, cb_cls,
                                                                    hb_cls, suffix=suffix)
        if shared is None:          # identical copies (homodimers) share the first copy's template
            registry.add(("cys", row["resn"]), xml, tname)
            path = os.path.join(_cache_dir(), f"covalent_{row['resn']}_{row['chain']}{row['resi']}.xml")
            with open(path, "w") as fh:
                fh.write(xml)
            files.append(path)
        # replace the ligand by the capped model minus the cap, bonded to SG
        keep = [i for i, nm in enumerate(all_names) if nm is not None]
        top = app.Topology()
        chain = top.addChain(row["chain"])
        res = top.addResidue(row["resn"], chain, row["resi"])
        atoms = {}
        pconf = prot.GetConformer()
        pos = []
        from openmm import Vec3

        for i in keep:
            a = prot.GetAtomWithIdx(i)
            atoms[i] = top.addAtom(all_names[i], app.Element.getBySymbol(a.GetSymbol()), res)
            p = pconf.GetAtomPosition(i)
            pos.append(Vec3(p.x, p.y, p.z) * 0.1)
        for b in prot.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in atoms and j in atoms:
                top.addBond(atoms[i], atoms[j])
        modeller.delete([lig])
        modeller.add(top, unit.Quantity(pos, unit.nanometer))
        new_res = [r for r in modeller.topology.residues() if r.name == row["resn"] and r.id == row["resi"]
                   and r.chain.id == row["chain"]][-1]
        attach_atom = next(a for a in new_res.atoms() if a.name == meta["attach_atom"])
        sg_atom = next(a for r in modeller.topology.residues() if r.chain.id == link["partner_chain"]
                       and r.id == link["partner_resi"] and str(r.name).upper() in ("CYS", "CYX", "CYM")
                       for a in r.atoms() if a.name == "SG")
        modeller.topology.addBond(attach_atom, sg_atom)
        cys_residues.append((link["partner_chain"], link["partner_resi"]))
        info.append(dict(meta, resn=row["resn"], partner=f"CYS{link['partner_resi']}{link['partner_chain']}:SG",
                         route="capped fragment (CH3-S-) GAFF2 + AM1-BCC, Amber CYX link"))
    variants = {}
    for r in modeller.topology.residues():
        if (r.chain.id, r.id) in cys_residues and str(r.name).upper() in ("CYS", "CYX", "CYM"):
            variants[r.index] = "CYX"
    return files, variants, info


# --------------------------------------------------------------------------- QM-region placeholders

CCD_CIF_URL = "https://files.rcsb.org/ligands/view/{code}.cif"

# UFF Lennard-Jones (Rappe et al. 1992): x_i [A], D_i [kcal/mol]. Only the QM-MM steric term uses them.
UFF_LJ = {"H": (2.886, 0.044), "C": (3.851, 0.105), "N": (3.660, 0.069), "O": (3.500, 0.060), "F": (3.364, 0.050),
          "S": (4.035, 0.274), "P": (4.147, 0.305), "CL": (3.947, 0.227), "BR": (4.189, 0.251), "I": (4.500, 0.339),
          "SE": (4.205, 0.291), "FE": (2.912, 0.013), "ZN": (2.763, 0.124), "CU": (3.495, 0.005),
          "MN": (2.961, 0.013), "MG": (3.021, 0.111), "CO": (2.872, 0.014), "NI": (2.834, 0.015),
          "CA": (3.399, 0.238), "MO": (3.052, 0.056), "W": (2.734, 0.067), "NA": (2.983, 0.030), "K": (3.812, 0.035)}


def ccd_component(code, fetch=True, timeout=20):
    """Atoms (name, element, formal charge, ideal xyz in A) and bonds of a PDB chemical component (mmCIF, cached)."""
    code = str(code or "").strip().upper()
    path = os.path.join(_cache_dir(), "ccd", f"{code}.cif")
    if not os.path.isfile(path) and fetch:
        try:
            import urllib.request

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with urllib.request.urlopen(CCD_CIF_URL.format(code=code), timeout=timeout) as resp:
                data = resp.read()
            if b"_chem_comp_atom" in data:
                with open(path, "wb") as fh:
                    fh.write(data)
        except Exception:
            return None
    if not os.path.isfile(path):
        return None
    import gemmi

    block = gemmi.cif.read(path).sole_block()
    tab = block.find("_chem_comp_atom.", ["atom_id", "type_symbol", "charge", "pdbx_model_Cartn_x_ideal",
                                          "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal"])
    atoms = []
    for row in tab:
        def num(v):
            try:
                return float(v)
            except ValueError:
                return float("nan")
        atoms.append({"name": gemmi.cif.as_string(row[0]), "element": gemmi.cif.as_string(row[1]).upper(),
                      "charge": int(num(row[2])) if math.isfinite(num(row[2])) else 0,
                      "xyz": (num(row[3]), num(row[4]), num(row[5]))})
    bonds = [(gemmi.cif.as_string(r[0]), gemmi.cif.as_string(r[1]), gemmi.cif.as_string(r[2]).upper())
             for r in block.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order"])]
    return {"atoms": atoms, "bonds": bonds}


def _kabsch(P, Q):
    """Rotation R and translation t with R @ p + t ~ q (least squares), P/Q: (n, 3)."""
    import numpy as np

    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


XH_BOND_NM = {"C": 0.109, "N": 0.101, "O": 0.097, "S": 0.134, "P": 0.142, "B": 0.119, "SI": 0.148}


def _place_hydrogen_locally(h, parent, ideal, crystal, nbrs, elem):
    """
    H position from the CCD ideal geometry fitted on the parent's local frame (parent + heavy
    neighbours, extended to second neighbours when needed): a global fit misplaces H wherever the
    crystal conformation differs from the ideal one (rotated methyls, propionates...).
    """
    import numpy as np

    ref = [parent] + [x for x in nbrs.get(parent, []) if elem.get(x) != "H" and x in crystal]
    if len(ref) < 3:
        for x in list(ref[1:]):
            ref += [y for y in nbrs.get(x, []) if elem.get(y) != "H" and y in crystal and y not in ref]
    ref = [x for x in ref if all(map(math.isfinite, ideal[x]["xyz"]))][:6]
    P = np.array([ideal[x]["xyz"] for x in ref]) * 0.1
    Q = np.array([crystal[x] for x in ref])
    if len(ref) >= 3:
        R, t = _kabsch(P, Q)
    else:                                    # isolated fragment: translate only
        R, t = np.eye(3), Q[0] - P[0]
    pos = R @ (np.array(ideal[h]["xyz"]) * 0.1) + t
    # standard X-H length (the CCD uses X-ray-style short X-H distances)
    bond = XH_BOND_NM.get(elem.get(parent, "C"),
                          float(np.linalg.norm((np.array(ideal[h]["xyz"]) - np.array(ideal[parent]["xyz"])) * 0.1)))
    vec = pos - np.array(crystal[parent])
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        pos = np.array(crystal[parent]) + vec / norm * bond
    return tuple(float(v) for v in pos)


def complete_from_ccd(residue, positions_nm, ph=7.4, fetch=True, removed_leaving=(), external=()):
    """
    Heavy atoms of a residue (crystal coordinates) plus the hydrogens of its PDB chemical component,
    placed by fitting the CCD ideal geometry onto the crystal atoms. Carboxylic-acid H are removed
    above pH 5 and H on atoms bonded to another residue (``external``: thioether/ester links) are dropped.
    Returns {"atoms": [(name, element, xyz_nm)], "bonds": [(n1, n2)], "charge": int, "notes": [...]}.
    """
    import numpy as np

    comp = ccd_component(residue.name, fetch=fetch)
    if comp is None:
        raise ValueError(f"no chemical component definition for {residue.name}")
    crystal = {a.name: positions_nm[a.index] for a in residue.atoms() if _element_symbol(a) != "H"}
    ideal = {a["name"]: a for a in comp["atoms"]}
    common = [n for n in crystal if n in ideal and ideal[n]["element"] != "H" and all(map(math.isfinite, ideal[n]["xyz"]))]
    if len(common) < 3:
        raise ValueError(f"{residue.name}: crystal atom names do not match the chemical component")
    P = np.array([ideal[n]["xyz"] for n in common]) * 0.1
    Q = np.array([crystal[n] for n in common])
    R, t = _kabsch(P, Q)
    nbrs = {}
    for a1, a2, _o in comp["bonds"]:
        nbrs.setdefault(a1, []).append(a2)
        nbrs.setdefault(a2, []).append(a1)
    elem = {n: a["element"] for n, a in ideal.items()}
    orders = {frozenset((a1, a2)): o for a1, a2, o in comp["bonds"]}
    drop, notes = set(), []

    def fitted(name):
        parent = next(x for x in nbrs.get(name, []) if x in crystal)
        return np.array(_place_hydrogen_locally(name, parent, ideal, crystal, nbrs, elem))

    for n, partner_xyz in dict(external).items():
        # one H per link: the one pointing at the partner atom (the CCD describes the free component)
        h_on = [x for x in nbrs.get(n, []) if elem.get(x) == "H" and all(map(math.isfinite, ideal[x]["xyz"]))]
        for pxyz in partner_xyz:
            if not h_on:
                break
            best = min(h_on, key=lambda x: float(np.linalg.norm(fitted(x) - np.array(pxyz))))
            drop.add(best)
            h_on.remove(best)
    if float(ph) > 5.0:
        for n, a in ideal.items():
            if a["element"] != "O":
                continue
            hs = [x for x in nbrs.get(n, []) if elem.get(x) == "H"]
            heavy = [x for x in nbrs.get(n, []) if elem.get(x) != "H"]
            if len(hs) == 1 and len(heavy) == 1 and elem[heavy[0]] == "C":
                c = heavy[0]
                if any(elem.get(x) == "O" and orders.get(frozenset((c, x))) == "DOUB" for x in nbrs.get(c, [])):
                    drop.add(hs[0])
                    notes.append(f"{n}-{hs[0]} removed (carboxylate at pH {float(ph):.1f})")
    atoms, charge = [], 0
    for n, a in ideal.items():
        if a["element"] == "H":
            if n in drop:
                continue
            parent = [x for x in nbrs.get(n, []) if x in crystal]
            if not parent or not all(map(math.isfinite, a["xyz"])):
                continue
            xyz = _place_hydrogen_locally(n, parent[0], ideal, crystal, nbrs, elem)
        elif n in crystal:
            xyz = tuple(crystal[n])
        else:
            continue                                  # leaving group / atom absent from the crystal
        atoms.append((n, a["element"], xyz))
        charge += int(a["charge"])
    names = {n for n, _e, _x in atoms}
    # removed carboxylic H: the O becomes an anion, unless the CCD already marked that O -1 (it sometimes
    # lists both the -1 and the H, e.g. HEC)
    charge -= sum(1 for d in drop for x in nbrs.get(d, []) if elem.get(x) == "O" and int(ideal[x]["charge"]) == 0)
    bonds = [(a1, a2) for a1, a2, _o in comp["bonds"] if a1 in names and a2 in names]
    has_metal = any(e in METALS for _n, e, _x in atoms)
    if has_metal:
        notes.append("formal charge from the chemical component definition; the metal oxidation state is NOT "
                     "included and must be set for the QM region")
    return {"atoms": atoms, "bonds": bonds, "charge": int(charge), "notes": notes, "has_metal": has_metal}


# 1-4 scaling of each family: every NonbondedForce tag loaded into one ForceField must agree
FAMILY_14_SCALES = {"amber14": (0.833333, 0.5), "charmm36": (1.0, 1.0)}


def nonbonded_xml_lines(family, lj_by_type):
    """NonbondedForce (+ CHARMM LennardJonesForce) entries for extra atom types {type: (sigma nm, eps kJ/mol)},
    charges taken from the residue templates. Amber: LJ in NonbondedForce. CHARMM36 in OpenMM puts LJ in a
    LennardJonesForce (NonbondedForce epsilon 0), and every tag's 1-4 scales must match the family."""
    from xml.sax.saxutils import quoteattr

    c14, lj14 = FAMILY_14_SCALES.get(family, FAMILY_14_SCALES["amber14"])
    lines = [f' <NonbondedForce coulomb14scale="{c14}" lj14scale="{lj14}">', '  <UseAttributeFromResidue name="charge"/>']
    charmm = family == "charmm36"
    for t, (sig, eps) in lj_by_type.items():
        if charmm:
            lines.append(f'  <Atom type={quoteattr(t)} sigma="1.0" epsilon="0.0"/>')
        else:
            lines.append(f'  <Atom type={quoteattr(t)} sigma="{sig:.5f}" epsilon="{eps:.6g}"/>')
    lines.append(" </NonbondedForce>")
    if charmm:
        lines.append(' <LennardJonesForce lj14scale="1.0" useDispersionCorrection="False">')
        for t, (sig, eps) in lj_by_type.items():
            lines.append(f'  <Atom type={quoteattr(t)} sigma="{sig:.5f}" epsilon="{eps:.6g}"/>')
        lines.append(" </LennardJonesForce>")
    return lines


def qm_placeholder_xml(resname, model, external_atoms=(), family="amber14"):
    """ForceField XML: atoms of a QM-region residue with zero charge, UFF Lennard-Jones and no bonded terms.
    Atom types are prefixed with the template name so several placeholders can be loaded together."""
    from xml.sax.saxutils import quoteattr

    elems = sorted({e for _n, e, _x in model["atoms"]})
    lines = ["<ForceField>", " <AtomTypes>"]
    from openmm.app import Element

    tp = f"qmph{resname}-"
    for e in elems:
        mass = Element.getBySymbol(e.capitalize()).mass._value
        lines.append(f'  <Type name={quoteattr(tp + e)} class={quoteattr(tp + e)} element='
                     f'{quoteattr(e.capitalize())} mass="{mass:.4f}"/>')
    lines += [" </AtomTypes>", " <Residues>", f"  <Residue name={quoteattr(resname)}>"]
    for n, e, _x in model["atoms"]:
        lines.append(f'   <Atom name={quoteattr(n)} type={quoteattr(tp + e)} charge="0.0"/>')
    for a1, a2 in model["bonds"]:
        lines.append(f"   <Bond atomName1={quoteattr(a1)} atomName2={quoteattr(a2)}/>")
    for a in external_atoms:
        lines.append(f"   <ExternalBond atomName={quoteattr(a)}/>")
    lines += ["  </Residue>", " </Residues>",
              ]
    lj = {}
    for e in elems:
        x, d = UFF_LJ.get(e, (3.5, 0.1))
        lj[tp + e] = (0.1 * x / 2 ** (1.0 / 6.0), d * 4.184)
    lines += nonbonded_xml_lines(family, lj)
    lines += ["</ForceField>"]
    return "\n".join(lines) + "\n"


# "unmatched" (incomplete residues) is left out on purpose: missing atoms would leave a hole in the QM region
QM_PLACEHOLDER_SOURCES = ("metal", "covalent_metal", "covalent")

# common oxidation states and spin multiplicities (2S+1) of metal centres; the choice is the user's
METAL_STATES = {
    "FE": [{"oxidation": 2, "multiplicities": [1, 5], "note": "Fe(II): low spin (1) 6-coordinate heme; high spin (5)"},
           {"oxidation": 3, "multiplicities": [2, 6], "note": "Fe(III): low spin (2) e.g. cyt c/His-Met heme; high spin (6)"}],
    "ZN": [{"oxidation": 2, "multiplicities": [1], "note": "Zn(II), closed shell"}],
    "CU": [{"oxidation": 1, "multiplicities": [1], "note": "Cu(I)"}, {"oxidation": 2, "multiplicities": [2], "note": "Cu(II)"}],
    "MN": [{"oxidation": 2, "multiplicities": [6], "note": "Mn(II) high spin"},
           {"oxidation": 3, "multiplicities": [5], "note": "Mn(III) high spin"}],
    "CO": [{"oxidation": 2, "multiplicities": [2, 4], "note": "Co(II)"}, {"oxidation": 3, "multiplicities": [1], "note": "Co(III)"}],
    "NI": [{"oxidation": 2, "multiplicities": [1, 3], "note": "Ni(II) square planar (1) / octahedral (3)"}],
    "MG": [{"oxidation": 2, "multiplicities": [1], "note": "Mg(II)"}],
    "CA": [{"oxidation": 2, "multiplicities": [1], "note": "Ca(II)"}],
    "MO": [{"oxidation": 4, "multiplicities": [3], "note": "Mo(IV)"}, {"oxidation": 6, "multiplicities": [1], "note": "Mo(VI)"}],
}
SIDE_CHAIN_CHARGE = {"ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "CYM": -1}


def plan_qm_region(topology, positions_nm, plan, ph=7.4, fetch_ccd=True, include_metal_ions=True):
    """
    QM-region proposals, one per site: each metal cofactor / heme c / metal cluster / coordinated
    metal ion with the side chains of the residues coordinating its metal and of its covalent
    partners (cut at CA-CB and capped with link atoms), the charge breakdown and the oxidation/spin
    options to confirm with the user. Returns {"sites": [...], "warnings": [...]}.
    """
    rows = [r for r in plan["components"] if r.get("source") in QM_PLACEHOLDER_SOURCES and not r.get("buildable")]
    rows += [r for r in plan["components"] if r.get("source") == "library" and r.get("category") == "cofactor"
             and any(_element_symbol(a) in METALS for res in topology.residues()
                     if (res.chain.id, res.id, res.name) == (r["chain"], r["resi"], r["resn"]) for a in res.atoms())]
    sites_metal = metal_sites(topology, positions_nm, include_water=True)   # coordinated waters belong in QM
    if include_metal_ions:
        # catalytic/structural metal ions (Zn, Mg, Mn...) coordinated by the protein
        bound = {(m.residue.chain.id, m.residue.id, m.residue.name) for m, _a, _d in sites_metal
                 if len(list(m.residue.atoms())) == 1}
        rows += [r for r in plan["components"] if r.get("category") == "ion" and
                 (r["chain"], r["resi"], r["resn"]) in bound and r not in rows]
    out = {"sites": [], "warnings": []}
    if not rows:
        out["warnings"].append("no metal cofactor or unparameterizable component: choose the QM region by chemistry")
        return out

    def resi_key(x):
        return int("".join(c for c in str(x) if c.isdigit() or c == "-") or 0)

    for r in rows:
        res = next((x for x in topology.residues() if (x.chain.id, x.id, x.name) == (r["chain"], r["resi"], r["resn"])), None)
        if res is None:
            continue
        side = {}
        for m, a, d in sites_metal:
            if m.residue is res and a.residue is not res:
                side.setdefault((a.residue.chain.id, a.residue.id), [a.residue, []])[1].append((m, a, d))
        for c in r.get("covalent", []) or []:
            partner = next((x for x in topology.residues()
                            if x.chain.id == c["partner_chain"] and x.id == c["partner_resi"]), None)
            if partner is not None:
                side.setdefault((partner.chain.id, partner.id), [partner, []])
        site = {"component": f"{res.name} {res.chain.id}{res.id}", "resn": res.name, "chain": res.chain.id,
                "resi": res.id, "side_chains": [], "charge_terms": [], "warnings": []}
        try:
            if len(list(res.atoms())) == 1:
                organic, metals = 0, [_element_symbol(next(iter(res.atoms())))]
            else:
                model = complete_from_ccd(res, positions_nm, ph=ph, fetch=fetch_ccd)
                organic = model["charge"]
                metals = [e for _n, e, _x in model["atoms"] if e in METALS]
            site["charge_terms"].append(f"{res.name}{res.id} without metal {organic:+d}")
        except Exception as exc:
            organic, metals = None, [_element_symbol(a) for a in res.atoms() if _element_symbol(a) in METALS]
            site["warnings"].append(f"{res.name}: organic charge unknown ({exc})")
        sc_total = 0
        for (ch, ri), (pres, contacts) in sorted(side.items(), key=lambda kv: (kv[0][0], resi_key(kv[0][1]))):
            name = str(pres.name).upper()
            q = SIDE_CHAIN_CHARGE.get(name, 0)
            if name in ("CYS", "CYX", "CYM") and any(a.name == "SG" for _m, a, _d in contacts):
                q = -1                                   # thiolate bound to the metal
            why = ", ".join(f"{a.name}-{m.name} {d * 10:.2f} A" for m, a, d in contacts) or "covalent partner"
            site["side_chains"].append({"resn": pres.name, "chain": ch, "resi": ri, "why": why, "charge": q})
            sc_total += q
            if q:
                site["charge_terms"].append(f"{pres.name}{ri} {q:+d}")
        parts = [f"(chain {res.chain.id} and resi {res.id} and resn {res.name})"]
        by_chain = {}
        for (ch, ri), (pres, _c) in side.items():
            by_chain.setdefault(ch, []).append(ri)
        for ch, ris in by_chain.items():
            parts.append(f"(chain {ch} and resi {'+'.join(sorted(ris, key=resi_key))} "
                         "and not name N+CA+C+O+H+HA+H1+H2+H3+OXT)")
        site["selection"] = " or ".join(parts)
        waters = [x for x in site["side_chains"] if str(x["resn"]).upper() in WATER_RESN]
        if waters:
            site["warnings"].append(f"{len(waters)} crystal water(s) coordinate the metal: keep crystal waters "
                                    "for this QM/MM run")
        base = None if organic is None else organic + sc_total
        site["charge_without_metals"] = base
        options = []
        counts = {}
        for mt in metals:
            counts[mt] = counts.get(mt, 0) + 1
        for mt, n in counts.items():
            for o in METAL_STATES.get(mt, []):
                options.append({"metal": mt, "count": n, "oxidation": o["oxidation"], "note": o["note"],
                                "multiplicities": o["multiplicities"] if n == 1 else [],
                                "qm_charge": None if base is None else base + n * int(o["oxidation"])})
            if mt not in METAL_STATES:
                site["warnings"].append(f"no default oxidation/spin for {mt}: ask the user")
        site["charge_options"] = options
        out["sites"].append(site)
    if len(out["sites"]) > 1:
        out["warnings"].append(f"{len(out['sites'])} sites: run QM/MM on one site at a time")
    return out

def install_qm_placeholders(modeller, plan, ph=7.4, fetch_ccd=True, family="amber14"):
    """
    QM/MM route for components without MM parameters (heme c, metal clusters, exotic cofactors):
    complete them with CCD hydrogens and give them a placeholder template (no classical chemistry;
    MOPAC describes them). Cys bonded to them become CYX. They MUST be in the QM region: returns
    (files, {residue index: variant}, residue keys [(chain, resi, resn)], info).
    """
    import openmm.app as app
    from openmm import Vec3, unit

    rows = [r for r in plan["components"] if r.get("source") in QM_PLACEHOLDER_SOURCES and not r.get("buildable")]
    if not rows:
        return [], {}, [], []
    files, keys, info, cys_partners = [], [], [], set()
    registry = _TemplateRegistry()
    for row in rows:
        # every replacement below re-indexes the atoms: positions must be read again
        pos_nm = [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]
        res = next((r for r in modeller.topology.residues()
                    if (r.chain.id, r.id, r.name) == (row["chain"], row["resi"], row["resn"])), None)
        if res is None:
            continue
        links = list(row.get("covalent", []))

        def partner_atom_of(c):
            return next((a for r in modeller.topology.residues() if r.chain.id == c["partner_chain"]
                         and r.id == c["partner_resi"] for a in r.atoms() if a.name == c["partner_atom"]), None)

        external = {}
        for c in links:
            pa = partner_atom_of(c)
            if pa is not None:
                external.setdefault(c["atom"], []).append(pos_nm[pa.index])
        model = complete_from_ccd(res, pos_nm, ph=ph, fetch=fetch_ccd, external=external)
        tname = f"{row['resn']}_QM"
        xml = qm_placeholder_xml(tname, model, external_atoms=sorted(external), family=family)
        shared, suffix = registry.lookup(("qm", row["resn"]), xml)
        if suffix:
            tname = f"{row['resn']}_QM{suffix}"
            xml = qm_placeholder_xml(tname, model, external_atoms=sorted(external), family=family)
        if shared is None:          # identical copies (e.g. the hemes of cytochrome c3) share one template
            registry.add(("qm", row["resn"]), xml, tname)
            path = os.path.join(_cache_dir(), f"qm_placeholder_{row['resn']}_{row['chain']}{row['resi']}.xml")
            with open(path, "w") as fh:
                fh.write(xml)
            files.append(path)
        else:
            tname = shared
        top = app.Topology()
        chain = top.addChain(row["chain"])
        # the residue keeps its PDB name (selections, QM-region keys, outputs); OpenMM matches the
        # template by graph, not by name
        nr = top.addResidue(row["resn"], chain, row["resi"])
        by_name = {}
        pos = []
        for n, e, xyz in model["atoms"]:
            by_name[n] = top.addAtom(n, app.Element.getBySymbol(e.capitalize()), nr)
            pos.append(Vec3(*xyz))
        for a1, a2 in model["bonds"]:
            top.addBond(by_name[a1], by_name[a2])
        modeller.delete([res])
        modeller.add(top, unit.Quantity(pos, unit.nanometer))
        # the topology was rebuilt: look the partners up again
        new_res = [r for r in modeller.topology.residues() if r.name == row["resn"] and r.id == row["resi"]
                   and r.chain.id == row["chain"]][-1]
        new_atoms = {a.name: a for a in new_res.atoms()}
        partner_atoms = []
        for c in links:
            partner = partner_atom_of(c)
            if partner is None or c["atom"] not in new_atoms:
                continue
            modeller.topology.addBond(new_atoms[c["atom"]], partner)
            partner_atoms.append((c["atom"], partner))
            if str(partner.residue.name).upper() in ("CYS", "CYX", "CYM") and c["partner_atom"] == "SG":
                cys_partners.add((c["partner_chain"], c["partner_resi"]))
        keys.append((row["chain"], row["resi"], row["resn"]))
        info.append({"resn": row["resn"], "template": tname, "qm_atoms": len(model["atoms"]),
                     "organic_formal_charge": model["charge"], "has_metal": model["has_metal"],
                     "notes": model["notes"],
                     "external_bonds": [f"{a}->{p.residue.name}{p.residue.id}:{p.name}" for a, p in partner_atoms]})
    variants = {r.index: "CYX" for r in modeller.topology.residues()
                if (r.chain.id, r.id) in cys_partners and str(r.name).upper() in ("CYS", "CYX", "CYM")}
    if variants and family == "charmm36":
        # OpenMM's CHARMM36 has no Cys with a bond from SG to another residue (only the disulfide patch)
        path = os.path.join(_cache_dir(), "charmm36_thioether_cys.xml")
        with open(path, "w") as fh:
            fh.write(charmm_thioether_cys_xml())
        files.append(path)
    return files, variants, keys, info


def charmm_thioether_cys_xml(name="CYSQ"):
    """CHARMM36 Cys whose SG is bonded to another residue (heme c thioether, QM partner): the CYS
    template without HG1, its charge moved to SG (neutral residue), an external bond on SG."""
    from xml.sax.saxutils import quoteattr

    import openmm.app as app

    cys = app.ForceField("charmm36.xml")._templates["CYS"]
    hg = next(i for i, a in enumerate(cys.atoms) if a.name == "HG1")
    q_hg = cys.atoms[hg].parameters["charge"]
    lines = ["<ForceField>", " <Residues>", f"  <Residue name={quoteattr(name)}>"]
    for i, a in enumerate(cys.atoms):
        if i == hg:
            continue
        q = a.parameters["charge"] + (q_hg if a.name == "SG" else 0.0)
        lines.append(f"   <Atom name={quoteattr(a.name)} type={quoteattr(a.type)} charge=\"{q:.4f}\"/>")
    for i, j in cys.bonds:
        if hg not in (i, j):
            lines.append(f"   <Bond atomName1={quoteattr(cys.atoms[i].name)} atomName2={quoteattr(cys.atoms[j].name)}/>")
    for i in list(cys.externalBonds) + [next(k for k, a in enumerate(cys.atoms) if a.name == "SG")]:
        lines.append(f"   <ExternalBond atomName={quoteattr(cys.atoms[i].name)}/>")
    lines += ["  </Residue>", " </Residues>", "</ForceField>"]
    return "\n".join(lines) + "\n"


def prepare_system_for_solvation(pdb_path, family="amber14", ph=7.0, keep_waters=False, remove_resn=(),
                                 remove_additives=True, gaff_ligands=True, fetch_ccd=True):
    """
    Everything before solvent: clean the structure (waters/additives, chain breaks, missing atoms),
    resolve and parameterize every component (library, GAFF, covalent Cys adducts), protonate
    (metal-site variants) and return {"ok", "modeller", "forcefield", "plan", ..., "rebuild"}.
    ``rebuild()`` repeats the preparation (e.g. after a failed membrane insertion).
    """
    import openmm.app as app
    from openmm import unit

    pdb = app.PDBFile(pdb_path)
    # remove waters/additives/requested residues first, so the gaps they leave become chain ends
    removed = {x.upper() for x in remove_resn} | (additive_residue_names() if remove_additives else set())
    pre = app.Modeller(pdb.topology, pdb.positions)
    drop = [r for r in pre.topology.residues()
            if (not keep_waters and r.name.upper() in WATER_RESN) or r.name.upper() in removed]
    # hydrogens of standard residues are rebuilt with the force-field names (PyMOL/other tools name
    # them differently, e.g. N-terminal H01/H02, and the templates would not match)
    drop_h = [a for a in pre.topology.atoms() if str(a.residue.name).upper() in STANDARD_POLYMER_RESN
              and a.element is not None and a.element.symbol == "H"]
    if drop or drop_h:
        pre.delete(drop + drop_h)
    split_top, breaks = split_chains_at_breaks(pre.topology, pre.positions)
    # a lone amino acid/nucleotide left by a gap has no template (both termini at once): drop it
    lone = [r for c in split_top.chains() for r in c.residues()
            if len(list(c.residues())) == 1 and str(r.name).upper() in STANDARD_POLYMER_RESN]
    lone_ids = [f"{r.name}{r.id}{r.chain.id}" for r in lone]
    if lone:
        pruned = app.Modeller(split_top, pre.positions)
        pruned.delete(lone)
        split_top, pre = pruned.topology, pruned
    topology, positions = split_top, pre.positions
    try:
        import io

        from pdbfixer import PDBFixer

        buf = io.StringIO()
        app.PDBFile.writeFile(split_top, pre.positions, buf, keepIds=True)
        buf.seek(0)
        fixer = PDBFixer(pdbfile=buf)
        # only chemistry-preserving swaps (MSE -> MET); PDBFixer would also turn SEP/TPO/PTR into
        # SER/THR/TYR, silently removing the phosphorylation the library can parameterize
        fixer.findNonstandardResidues()
        fixer.nonstandardResidues = [(r, std) for r, std in fixer.nonstandardResidues
                                     if str(r.name).upper() in REPLACEABLE_RESIDUES]
        fixer.replaceNonstandardResidues()
        fixer.findMissingResidues()
        fixer.missingResidues = {}          # gaps become chain ends, loops are not invented
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        topology, positions = fixer.topology, fixer.positions
    except Exception:
        pass
    modeller = app.Modeller(topology, positions)
    pos_nm = [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]
    plan = resolve_components(modeller.topology, pos_nm, family=family)
    report = {"family": family, "plan": plan, "ok": False, "chain_breaks": breaks,
              "removed_isolated_residues": lone_ids}
    allowed = {"library", "gaff"} if gaff_ligands else {"library"}
    blocked = [r for r in plan["components"] if r["source"] not in allowed and not r.get("buildable")]
    if blocked:
        report["error"] = "components without parameters: " + "; ".join(
            f"{r['resn']} {r['chain']}{r['resi']} -> {r['source']}" for r in blocked)
        return report
    renames = {}
    files = prepare_modeller(modeller, plan, family, renames=renames)
    ff = app.ForceField(*files)
    gen, gaff_info = (install_gaff_components(modeller, plan, ph=ph, fetch_ccd=fetch_ccd) if gaff_ligands
                      else (None, []))
    nc_files, nc_info = install_noncanonical_residues(modeller, plan, family=family, ph=ph, renames=renames)
    for f in nc_files:
        ff.loadFile(f)
    if nc_info:
        report["noncanonical_residues"] = nc_info
        if family != "amber14":
            report.setdefault("notes", []).append("GAFF2 non-canonical residue inside a CHARMM36 protein: mixed "
                                                  "force fields at that residue")
    cov_files, cov_variants, cov_info = install_covalent_cys_components(modeller, plan, ph=ph)
    if cov_files:
        for f in cov_files:
            ff.loadFile(f)
        report["covalent_components"] = cov_info
    if gen is not None:
        ff.registerTemplateGenerator(gen.generator)
        report["gaff_components"] = gaff_info
        if family != "amber14":
            report.setdefault("notes", []).append(GAFF_IN_CHARMM_WARNING)
    variants, variant_notes = coordination_variants(
        modeller.topology, [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions])
    for idx, v in cov_variants.items():
        variants[idx] = v
    modeller.addHydrogens(ff, pH=float(ph), variants=variants)
    restore_names(modeller.topology, renames)
    report["protonation_variants"] = variant_notes
    report["modeller"], report["forcefield"], report["files"] = modeller, ff, files
    report["ok"] = True
    report["rebuild"] = lambda: prepare_system_for_solvation(
        pdb_path, family=family, ph=ph, keep_waters=keep_waters, remove_resn=remove_resn,
        remove_additives=remove_additives, gaff_ligands=gaff_ligands, fetch_ccd=fetch_ccd)["modeller"]
    return report


# --------------------------------------------------------------------------- non-canonical residues


def _backbone_neighbours(topology, res, positions_nm):
    """(previous residue C atom, next residue N atom) bonded to res.N / res.C, by bonds or distance."""
    atoms = {a.name: a for a in res.atoms()}
    prev_c = next_n = None
    for bond in topology.bonds():
        a, b = bond[0], bond[1]
        for x, y in ((a, b), (b, a)):
            if x.residue is res and y.residue is not res:
                if x.name == "N" and y.name == "C":
                    prev_c = y
                elif x.name == "C" and y.name == "N":
                    next_n = y
    if (prev_c is None or next_n is None) and "N" in atoms and "C" in atoms:
        for other in res.chain.residues():
            if other is res:
                continue
            oat = {a.name: a for a in other.atoms()}
            if prev_c is None and "C" in oat and _dist(positions_nm[oat["C"].index], positions_nm[atoms["N"].index]) < 0.2:
                prev_c = oat["C"]
            if next_n is None and "N" in oat and _dist(positions_nm[oat["N"].index], positions_nm[atoms["C"].index]) < 0.2:
                next_n = oat["N"]
    return prev_c, next_n


def _backbone_classes(family):
    import openmm.app as app

    cat = get_catalog(family)
    ff = app.ForceField(*cat.base_files)
    ala = {a.name: ff._atomTypes[a.type].atomClass for a in ff._templates["ALA"].atoms}
    return {"N": ala["N"], "H": ala.get("H", ala.get("HN")), "CA": ala["CA"], "HA": ala["HA"], "C": ala["C"],
            "O": ala["O"]}


def _protonate_lysine_schiff_base(mol, nz_index):
    """C=N-Lys Schiff bases (retinal, PLP internal aldimine) are protonated in proteins (pKa ~ 12-13)."""
    from rdkit import Chem

    nz = mol.GetAtomWithIdx(nz_index)
    double_c = [b for b in nz.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(nz).GetSymbol() == "C"]
    if not double_c or nz.GetFormalCharge() != 0 or nz.GetTotalNumHs(includeNeighbors=True) > 0:
        return mol, False
    heavy = Chem.RemoveHs(mol)
    heavy.GetAtomWithIdx(nz_index).SetFormalCharge(1)
    heavy.GetAtomWithIdx(nz_index).SetNoImplicit(False)
    Chem.SanitizeMol(heavy)
    return Chem.AddHs(heavy, addCoords=True), True


def install_noncanonical_residues(modeller, plan, family="amber14", ph=7.4, renames=None,
                                  protonate_schiff_base=True):
    """
    Covalent adducts on any residue (merged with their residue) and modified residues without a template:
    the residue is parameterized as a capped dipeptide ACE-X*-NME with GAFF2 + AM1-BCC, the caps built from
    the real neighbouring backbone atoms, and put back in place with external bonds at N and C. Returns
    (ForceField XML files, info).
    """
    from openmm import unit
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rows = [r for r in plan["components"] if r.get("buildable") == "noncanonical"]
    if not rows:
        return [], []
    bb = _backbone_classes(family)
    files, info = [], []
    registry = _TemplateRegistry()
    for row in rows:
        pos_nm = [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]
        key = (row["chain"], row["resi"], row["resn"])
        if row["source"] == "covalent":
            lig = next(r for r in modeller.topology.residues() if (r.chain.id, r.id, r.name) == key)
            have = {frozenset((b[0].index, b[1].index)) for b in modeller.topology.bonds()}
            for a, b, _d in covalent_links(modeller.topology, pos_nm).get(lig, []):
                if frozenset((a.index, b.index)) not in have:
                    modeller.topology.addBond(a, b)
            c = row["covalent"][0]
            key = (c["partner_chain"], c["partner_resi"], c["partner_resn"])
            merge_residue_pairs(modeller, [((row["chain"], row["resi"], row["resn"]), key)])
            pos_nm = [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]
        res = next(r for r in modeller.topology.residues() if (r.chain.id, r.id, r.name) == key)
        prev_c, next_n = _backbone_neighbours(modeller.topology, res, pos_nm)
        if prev_c is None or next_n is None:
            raise ValueError(f"{res.name}{res.id}: modified residues at a chain terminus are not supported")
        prev = {a.name: a for a in prev_c.residue.atoms()}
        nxt = {a.name: a for a in next_n.residue.atoms()}
        heavy = [a for a in res.atoms() if a.element is not None and a.element.symbol != "H"]
        rw = Chem.RWMol()
        idx = {}
        for a in heavy:
            idx[a] = rw.AddAtom(Chem.Atom(a.element.symbol))
        cap_atoms = [("aceC", prev["C"]), ("aceO", prev["O"]), ("aceCA", prev["CA"]), ("nmeN", nxt["N"]),
                     ("nmeCA", nxt["CA"])]
        cap_idx = {}
        for tag, a in cap_atoms:
            cap_idx[tag] = rw.AddAtom(Chem.Atom(a.element.symbol))
        for bond in modeller.topology.bonds():
            a, b = bond[0], bond[1]
            if a in idx and b in idx:
                rw.AddBond(idx[a], idx[b], Chem.BondType.SINGLE)
        ratoms = {a.name: a for a in heavy}
        for i, j in (("aceC", "aceO"), ("aceC", "aceCA"), ("nmeN", "nmeCA")):
            rw.AddBond(cap_idx[i], cap_idx[j], Chem.BondType.SINGLE)
        rw.AddBond(cap_idx["aceC"], idx[ratoms["N"]], Chem.BondType.SINGLE)
        rw.AddBond(idx[ratoms["C"]], cap_idx["nmeN"], Chem.BondType.SINGLE)
        conf = Chem.Conformer(rw.GetNumAtoms())
        for a, i in list(idx.items()) + [(a, cap_idx[t]) for t, a in cap_atoms]:
            p = pos_nm[a.index]
            conf.SetAtomPosition(i, Point3D(10 * p[0], 10 * p[1], 10 * p[2]))
        mol = rw.GetMol()
        mol.AddConformer(conf, assignId=True)
        capped = _bond_orders_from_geometry(mol)
        Chem.SanitizeMol(capped)
        try:
            from .pka_runtime import protonate_molecule

            prot = protonate_molecule(capped, ph=float(ph))
            prot = prot[0] if isinstance(prot, tuple) else prot
        except Exception:
            prot = Chem.AddHs(capped, addCoords=True)
        schiff = False
        if protonate_schiff_base and str(res.name).upper() in ("LYS", "LYN") and "NZ" in ratoms:
            prot, schiff = _protonate_lysine_schiff_base(prot, idx[ratoms["NZ"]])
        cap_classes = {cap_idx["aceC"]: bb["C"], cap_idx["aceO"]: bb["O"], cap_idx["aceCA"]: bb["CA"],
                       cap_idx["nmeN"]: bb["N"], cap_idx["nmeCA"]: bb["CA"]}
        for t in ("aceCA", "nmeCA"):
            for nb in prot.GetAtomWithIdx(cap_idx[t]).GetNeighbors():
                if nb.GetAtomicNum() == 1:
                    cap_classes[nb.GetIdx()] = bb["HA"]
        for nb in prot.GetAtomWithIdx(cap_idx["nmeN"]).GetNeighbors():
            if nb.GetAtomicNum() == 1:
                cap_classes[nb.GetIdx()] = bb["H"]
        keep_names = {i: a.name for a, i in idx.items()}
        external = [idx[ratoms["N"]], idx[ratoms["C"]]]
        xml, tname, meta, names = build_capped_template(prot, cap_classes, res.name, keep_names, external, tag="ncaa")
        shared, suffix = registry.lookup(("ncaa", res.name), xml)
        if suffix:                  # same residue name, different chemistry (e.g. two kinds of modified Lys)
            xml, tname, meta, names = build_capped_template(prot, cap_classes, res.name, keep_names, external,
                                                            tag="ncaa", suffix=suffix)
        if shared is None:
            registry.add(("ncaa", res.name), xml, tname)
            path = os.path.join(_cache_dir(), f"noncanonical_{res.name}_{res.chain.id}{res.id}.xml")
            with open(path, "w") as fh:
                fh.write(xml)
            files.append(path)
        else:
            tname = shared
        original = res.name
        _replace_residue_in_place(modeller, res, prot, names, prev_c, next_n, new_name=tname)
        if renames is not None:
            # the template name keeps Modeller.addHydrogens from applying the standard residue's H definitions
            renames[(res.chain.id, res.id, tname)] = {"resn": original, "atoms": {}}
        if schiff:
            meta["notes"] = ["Lys Schiff base (C=N) protonated (+1), as in retinal proteins and PLP enzymes; "
                             "pass protonate_schiff_base=False for a neutral imine"]
        info.append(dict(meta, residue=f"{res.name}{res.id}{res.chain.id}", source=row["source"],
                         ligand=row["resn"] if row["source"] == "covalent" else None,
                         route="ACE/NME-capped GAFF2 + AM1-BCC non-canonical residue"))
    return files, info


def _replace_residue_in_place(modeller, res, mol, names, prev_c, next_n, new_name=None):
    """Rebuild the topology with ``res`` replaced by the kept atoms of ``mol`` (same chain position)."""
    import openmm.app as app
    from openmm import Vec3, unit

    old = modeller.topology
    pos = list(modeller.positions)
    conf = mol.GetConformer()
    new = app.Topology()
    new.setPeriodicBoxVectors(old.getPeriodicBoxVectors())
    amap, new_pos, kept = {}, [], {}
    for chain in old.chains():
        nc = new.addChain(chain.id)
        for r in chain.residues():
            nr = new.addResidue(new_name if (r is res and new_name) else r.name, nc, r.id, r.insertionCode)
            if r is res:
                for i, name in names.items():
                    if name is None:
                        continue
                    a = mol.GetAtomWithIdx(i)
                    kept[i] = new.addAtom(name, app.Element.getBySymbol(a.GetSymbol()), nr)
                    p = conf.GetAtomPosition(i)
                    new_pos.append(Vec3(p.x, p.y, p.z) * 0.1)
                continue
            for a in r.atoms():
                amap[a] = new.addAtom(a.name, a.element, nr, a.id)
                new_pos.append(pos[a.index].value_in_unit(unit.nanometer))
    for bond in old.bonds():
        if bond[0] in amap and bond[1] in amap:
            new.addBond(amap[bond[0]], amap[bond[1]])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in kept and j in kept:
            new.addBond(kept[i], kept[j])
    by_name = {names[i]: kept[i] for i in kept}
    new.addBond(amap[prev_c], by_name["N"])
    new.addBond(by_name["C"], amap[next_n])
    modeller.topology = new
    modeller.positions = unit.Quantity([Vec3(*[float(x) for x in p]) for p in new_pos], unit.nanometer)


def _as_quantity(positions):
    """List of Vec3 quantities (any length unit) -> one Quantity in nm, as Modeller expects."""
    from openmm import Vec3, unit

    out = []
    for p in positions:
        v = p.value_in_unit(unit.nanometer) if hasattr(p, "value_in_unit") else p
        out.append(Vec3(float(v[0]), float(v[1]), float(v[2])))
    return unit.Quantity(out, unit.nanometer)


def build_and_validate(pdb_path, family="amber14", ph=7.0, keep_waters=False, remove_resn=(), remove_additives=True,
                       metal_restraints=True, gaff_ligands=True, fetch_ccd=True, solvate=False,
                       padding_nm=1.0, ionic_strength_molar=0.15, minimize_steps=500, out_dir=None,
                       platform="CPU"):
    """
    Build an OpenMM system for a structure with all its components and check it: every residue
    has a template, the total charge is an integer, and a short minimization is stable.
    """
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    prep = prepare_system_for_solvation(pdb_path, family=family, ph=ph, keep_waters=keep_waters,
                                        remove_resn=remove_resn, remove_additives=remove_additives,
                                        gaff_ligands=gaff_ligands, fetch_ccd=fetch_ccd)
    report = {k: v for k, v in prep.items() if k not in ("modeller", "forcefield", "rebuild")}
    report["ok"] = False
    if not prep["ok"]:
        return report
    modeller, ff, files, plan = prep["modeller"], prep["forcefield"], prep["files"], prep["plan"]
    if solvate:
        modeller.addSolvent(ff, padding=padding_nm * unit.nanometer, ionicStrength=ionic_strength_molar * unit.molar)
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                                 constraints=app.HBonds)
    else:
        # vacuum check: a cutoff keeps the minimization O(N) on large proteins
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff=1.2 * unit.nanometer, constraints=app.HBonds)
    charge = 0.0
    comp_charges = {}
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            qs = [force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                  for i in range(system.getNumParticles())]
            charge = sum(qs)
            names = {r["resn"] for r in plan["components"] if r.get("source") in ("library", "gaff")}
            names |= {c["resn"] for c in report.get("covalent_components", [])}
            for res in modeller.topology.residues():
                if res.name in names:
                    comp_charges.setdefault(f"{res.name} {res.chain.id}{res.id}",
                                            round(sum(qs[a.index] for a in res.atoms()), 3))
    if metal_restraints:
        report["metal_restraints"] = add_coordination_restraints(
            system, modeller.topology, [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions])
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    try:
        plat = mm.Platform.getPlatformByName(platform)
        sim = app.Simulation(modeller.topology, system, integ, plat)
    except Exception:
        sim = app.Simulation(modeller.topology, system, integ)
    sim.context.setPositions(modeller.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if int(minimize_steps) > 0:              # OpenMM reads maxIterations=0 as "until converged"
        sim.minimizeEnergy(maxIterations=int(minimize_steps))
    state = sim.context.getState(getEnergy=True, getPositions=True)
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    # an infinite/NaN start (two atoms placed on top of each other) is fine if minimization fixes it
    # and an already-minimized input may rise a few kJ/mol when the H-bond constraints are applied
    stable = math.isfinite(e1) and (not math.isfinite(e0) or e1 <= e0 + max(50.0, 0.02 * abs(e0)))
    report.update(ok=bool(stable and abs(charge - round(charge)) < 0.01), files=files,
                  n_atoms=system.getNumParticles(), total_charge=round(charge, 3),
                  component_charges=comp_charges, energy_start_kj=round(e0, 1), energy_minimized_kj=round(e1, 1))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        pdb_out = os.path.join(out_dir, "system_minimized.pdb")
        with open(pdb_out, "w") as fh:
            app.PDBFile.writeFile(modeller.topology, state.getPositions(), fh, keepIds=True)
        xml_out = os.path.join(out_dir, "system.xml")
        with open(xml_out, "w") as fh:
            fh.write(mm.XmlSerializer.serialize(system))
        with open(os.path.join(out_dir, "component_plan.json"), "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        report.update(pdb=pdb_out, system_xml=xml_out)
    if abs(charge - round(charge)) >= 0.01:
        report["error"] = f"non-integer total charge {charge:.3f}"
    elif not stable:
        report["error"] = f"minimization was not stable ({e0} -> {e1} kJ/mol)"
    return report
