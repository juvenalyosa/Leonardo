"""
Automatic protein preparation: from a crystal / cryo-EM / predicted structure to a complete, protonated
model ready for docking, MD or QM/MM.  Works on a PDB file, no PyMOL and no GUI.

Steps (each one reported, each one optional):

1. **Clean**: one alternate location per residue (the most occupied; ties go to A), existing hydrogens removed
   (they are rebuilt consistently), modified residues replaced by their parent (MSE -> MET, ...), waters kept
   or removed, ligands / ions / cofactors set aside and put back unchanged at the end.
2. **Missing residues** (internal gaps, optionally the missing termini): the full sequence comes from SEQRES,
   from ``sequences`` or from RCSB; loops are built with MODELLER, moving only the missing residues and one
   neighbour on each side (every other atom keeps its crystal position), or with PDBFixer when MODELLER (or its
   license) is not available.  A gap that is not filled is split into two chains with proper termini.
3. **Missing heavy atoms** (incomplete side chains): PDBFixer.
4. **Protonation at the chosen pH**: PROPKA gives the pKa of every titratable residue in its environment (Asp,
   Glu, His, Lys, Cys); a residue is protonated when its pKa is above the pH.  ``reduce`` flips Asn / Gln / His
   side chains to the orientation that makes the best hydrogen bonds and chooses the His tautomer (HID / HIE).
   Metal ligands keep the state the metal imposes (Cys thiolate, His bound through the right nitrogen),
   disulfides are CYX.  Hydrogens are then added with OpenMM (Amber names).
5. **Minimization** (optional): hydrogens and built residues relax, crystal heavy atoms are restrained.

Without PROPKA the charge states fall back to OpenMM's pH rule (standard pKa values); without ``reduce`` there
are no flips and OpenMM chooses the His tautomer.  Both are reported.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict, defaultdict

import numpy as np

WATER_RESN = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"}
METAL_RESN = {"ZN", "MG", "CA", "MN", "FE", "FE2", "FE3", "CU", "CU1", "CO", "NI", "CD", "HG", "NA", "K", "LI",
              "SR", "BA", "PT", "AU", "AG", "PB", "CS", "RB", "AL", "GA", "YB", "W", "MO"}
AMINO = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
         "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYX", "CYM", "HSD",
         "HSE", "HSP"}
NUCLEIC = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "RA", "RC", "RG", "RU"}
ONE_LETTER = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
              "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
              "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
              "SEP": "S", "TPO": "T", "PTR": "Y"}
THREE_LETTER = {v: k for k, v in ONE_LETTER.items() if k not in ("MSE", "SEP", "TPO", "PTR")}
TITRATABLE = ("ASP", "GLU", "HIS", "LYS", "CYS", "TYR")
# phosphorylated residues are kept (openmmforcefields' ff14SB + phosaa14SB have their templates); other modified
# residues are replaced by their parent, MSE -> MET silently (same chemistry), the rest with a warning
PHOSPHO = {"SEP": "SER", "TPO": "THR", "PTR": "TYR"}
PHOSPHATE_ATOMS = {"P", "O1P", "O2P", "O3P", "OP1", "OP2", "OP3"}
SILENT_REPLACEMENTS = {"MSE"}
# pKa used when PROPKA is missing (the values OpenMM's pH rule assumes)
MODEL_PKA = {"ASP": 3.9, "GLU": 4.3, "HIS": 6.0, "LYS": 10.5, "CYS": 8.3, "TYR": 10.1}


def _log(progress, msg):
    if callable(progress):
        try:
            progress(str(msg))
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------- PDB text

def _is_coord(line):
    return line.startswith("ATOM  ") or line.startswith("HETATM")


def _res_key(line):
    """(chain, resSeq + iCode) of an ATOM/HETATM line."""
    return (line[21], line[22:27].strip())


def _element(line):
    el = line[76:78].strip().upper() if len(line) >= 78 else ""
    if el:
        return el
    name = line[12:16].strip().upper()
    return re.sub(r"[^A-Z]", "", name)[:1]


def _is_hydrogen(line):
    el = _element(line)
    if el in ("H", "D"):
        return True
    if not el:
        return line[12:16].strip().upper().startswith("H")
    return False


def select_altlocs(lines):
    """
    One alternate location per residue: the one with the highest mean occupancy (ties: the first letter).
    Returns (lines without altloc letters, [{"residue", "altlocs", "kept", "occupancy"}]).
    """
    occ = defaultdict(lambda: defaultdict(list))
    resn_of = {}
    for ln in lines:
        if _is_coord(ln) and len(ln) > 16 and ln[16] not in (" ", ""):
            key = _res_key(ln)
            try:
                occ[key][ln[16]].append(float(ln[54:60]))
            except ValueError:
                occ[key][ln[16]].append(0.0)
            resn_of.setdefault((key, ln[16]), ln[17:20].strip())
    chosen = {}
    report = []
    for key, alts in occ.items():
        best = sorted(alts, key=lambda a: (-float(np.mean(alts[a])), a))[0]
        chosen[key] = best
        report.append({"residue": f"{resn_of.get((key, best), '')}{key[1]}:{key[0]}".strip(),
                       "altlocs": "".join(sorted(alts)), "kept": best,
                       "occupancy": round(float(np.mean(alts[best])), 2)})
    out = []
    for ln in lines:
        if _is_coord(ln) and len(ln) > 16 and ln[16] not in (" ", ""):
            if chosen.get(_res_key(ln)) != ln[16]:
                continue
            ln = ln[:16] + " " + ln[17:]
        out.append(ln)
    return out, sorted(report, key=lambda r: (r["residue"].split(":")[-1], _num(r["residue"])))


def _num(label):
    m = re.search(r"(-?\d+)", str(label))
    return int(m.group(1)) if m else 0


def classify_lines(lines, modified=None):
    """
    Split PDB lines into polymer (ATOM records and modified residues inside chains), waters and other hetero
    groups (ligands, ions, cofactors).  ``modified``: residue names treated as polymer (PDBFixer's list).
    """
    modified = set(modified or ())
    header, polymer, waters, hets = [], [], [], []
    for ln in lines:
        if ln.startswith("SEQRES") or ln.startswith("CRYST1"):
            header.append(ln)
            continue
        if not _is_coord(ln):
            continue
        resn = ln[17:20].strip().upper()
        if resn in WATER_RESN:
            waters.append(ln)
        elif ln.startswith("ATOM  ") or resn in modified:
            polymer.append(ln)
        else:
            hets.append(ln)
    return header, polymer, waters, hets


def _substitutions():
    try:
        from pdbfixer.pdbfixer import substitutions

        return dict(substitutions)
    except Exception:
        return {"MSE": "MET"}


def _write(path, lines, end=True):
    with open(path, "w") as fh:
        for ln in lines:
            fh.write(ln.rstrip("\n") + "\n")
        if end:
            fh.write("END\n")
    return path


def _het_groups(lines):
    groups = OrderedDict()
    for ln in lines:
        key = (ln[17:20].strip(), ln[21], ln[22:27].strip())
        groups.setdefault(key, []).append(ln)
    return [{"resn": k[0], "chain": k[1], "resi": k[2], "atoms": len(v),
             "kind": "metal/ion" if k[0].upper() in METAL_RESN else "ligand/cofactor"} for k, v in groups.items()]


# ----------------------------------------------------------------------------------------------- tools

def tool_status():
    """Which external tools this machine has: PROPKA, reduce, MODELLER (with a license), PDBFixer, OpenMM."""
    out = {}
    for mod in ("propka", "pdbfixer", "openmm"):
        try:
            __import__(mod)
            out[mod] = True
        except Exception:
            out[mod] = False
    out["reduce"] = bool(reduce_executable())
    out["modeller"] = modeller_available()
    return out


def amber_ffxml(name):
    """Path of an openmmforcefields Amber ffxml file ('' when openmmforcefields is missing)."""
    try:
        import openmmforcefields

        path = os.path.join(os.path.dirname(openmmforcefields.__file__), "ffxml", "amber", name)
        return path if os.path.isfile(path) else ""
    except Exception:
        return ""


def phospho_forcefield_files():
    """ff14SB + phosaa14SB (SEP/TPO/PTR templates), or [] when unavailable."""
    files = [amber_ffxml("ff14SB.xml"), amber_ffxml("phosaa14SB.xml")]
    return files if all(files) else []


def python_executable():
    """The environment's Python (inside the PyMOL app ``sys.executable`` can be the PyMOL launcher)."""
    for cand in (os.path.join(sys.prefix, "bin", "python"), os.path.join(sys.prefix, "python.exe")):
        if os.path.isfile(cand):
            return cand
    return sys.executable


def reduce_executable():
    for cand in (os.path.join(sys.prefix, "bin", "reduce"), os.path.join(os.path.dirname(sys.executable), "reduce"),
                 shutil.which("reduce") or ""):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return ""


_MODELLER_OK = None


def modeller_available():
    """MODELLER importable with a valid license (checked once, in a child process: a bad key aborts)."""
    global _MODELLER_OK
    if _MODELLER_OK is None:
        try:
            r = subprocess.run([python_executable(), "-c", "import modeller, sys; sys.exit(0)"], capture_output=True,
                               timeout=60)
            _MODELLER_OK = r.returncode == 0
        except Exception:
            _MODELLER_OK = False
    return _MODELLER_OK


def fetch_rcsb_sequences(pdb_id, timeout=20):
    """{auth chain id: one-letter sequence} of the polymer entities of a PDB entry (RCSB REST)."""
    import urllib.request

    pid = str(pdb_id or "").strip().upper()
    if not re.fullmatch(r"[0-9][A-Z0-9]{3}", pid):
        return {}

    def get(url):
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            return json.loads(fh.read().decode())

    out = {}
    entry = get(f"https://data.rcsb.org/rest/v1/core/entry/{pid}")
    for eid in entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", []) or []:
        ent = get(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pid}/{eid}")
        seq = (ent.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can") or "").replace("\n", "").strip()
        ptype = str(ent.get("entity_poly", {}).get("type", "")).lower()
        if not seq or "polypeptide" not in ptype:
            continue
        for ch in ent.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", []) or []:
            out[str(ch)] = seq
    return out


# ----------------------------------------------------------------------------------------------- analysis

def _fixer_from_lines(lines, sequences=None):
    """PDBFixer on polymer lines; ``sequences`` {chain: one-letter} replace / supply SEQRES."""
    from pdbfixer import PDBFixer
    from pdbfixer.pdbfixer import Sequence

    text = "".join(ln.rstrip("\n") + "\n" for ln in lines) + "END\n"
    fixer = PDBFixer(pdbfile=io.StringIO(text))
    if sequences:
        chains = {c.id: c for c in fixer.topology.chains()}
        seqs = [s for s in fixer.sequences if s.chainId not in sequences]
        for cid, seq in sequences.items():
            if cid in chains:
                seqs.append(Sequence(cid, [THREE_LETTER.get(x, "UNK") for x in str(seq).upper() if x.isalpha()]))
        fixer.sequences = seqs
    return fixer


def _chain_breaks(topology, positions_nm, max_link_nm=0.2):
    out = []
    for chain in topology.chains():
        prev = None
        for res in chain.residues():
            if prev is not None:
                pa = {a.name: a.index for a in prev.atoms()}
                ra = {a.name: a.index for a in res.atoms()}
                for x, y in (("C", "N"), ("O3'", "P")):
                    if x in pa and y in ra:
                        d = float(np.linalg.norm(positions_nm[pa[x]] - positions_nm[ra[y]]))
                        if d > max_link_nm:
                            out.append({"chain": chain.id, "after": f"{prev.name}{prev.id}",
                                        "before": f"{res.name}{res.id}", "distance_A": round(10 * d, 1)})
                        break
            prev = res
    return out


def _positions_nm(positions):
    from openmm import unit

    return np.array([[p.x, p.y, p.z] for p in positions.value_in_unit(unit.nanometer)], float)


def _sequence_mismatches(fixer):
    """Chains whose full sequence (SEQRES / given) is not the observed residues plus the missing ones, e.g. a
    mutated model or the wrong entity: filling their 'gaps' would insert residues that are not missing."""
    seqs = {s.chainId: len(s.residues) for s in getattr(fixer, "sequences", [])}
    bad = []
    for ci, chain in enumerate(fixer.topology.chains()):
        if chain.id not in seqs:
            continue
        observed = sum(1 for r in chain.residues() if r.name in AMINO or r.name in NUCLEIC)
        missing = sum(len(v) for (c, _r), v in fixer.missingResidues.items() if c == ci)
        if observed and observed + missing != seqs[chain.id]:
            bad.append(chain.id)
    return sorted(set(bad))


def _gaps_from_fixer(fixer):
    chains = list(fixer.topology.chains())
    gaps = []
    for (ci, ri), names in sorted(fixer.missingResidues.items()):
        chain = chains[ci]
        res = list(chain.residues())
        where = "N-terminus" if ri == 0 else ("C-terminus" if ri >= len(res) else "internal")
        after = f"{res[ri - 1].name}{res[ri - 1].id}" if ri > 0 else ""
        before = f"{res[ri].name}{res[ri].id}" if ri < len(res) else ""
        gaps.append({"chain": chain.id, "where": where, "after": after, "before": before, "length": len(names),
                     "sequence": "".join(ONE_LETTER.get(n, "X") for n in names), "_key": (ci, ri)})
    return gaps


def analyze_structure(pdb_path, sequences=None):
    """
    What a structure needs before simulation: alternate locations, existing hydrogens, modified residues,
    missing residues (from SEQRES or ``sequences``; chain breaks otherwise), missing heavy atoms, disulfides,
    waters, ligands and ions.
    """
    with open(pdb_path) as fh:
        lines = fh.readlines()
    _clean, altlocs = select_altlocs(lines)
    subs = _substitutions()
    header, polymer, waters, hets = classify_lines(_clean, modified=subs)
    n_h = sum(1 for ln in polymer if _is_hydrogen(ln))
    polymer = [ln for ln in polymer if not _is_hydrogen(ln)]
    fixer = _fixer_from_lines(header + polymer, sequences)
    fixer.findMissingResidues()
    gaps = _gaps_from_fixer(fixer)
    fixer.findNonstandardResidues()
    nonstandard = [f"{r.name}{r.id}:{r.chain.id}->{rep}" for r, rep in fixer.nonstandardResidues]
    fixer.findMissingAtoms()
    missing_atoms = [{"residue": f"{r.name}{r.id}:{r.chain.id}", "atoms": [a.name for a in atoms]}
                     for r, atoms in fixer.missingAtoms.items()]
    pos = _positions_nm(fixer.positions)
    breaks = _chain_breaks(fixer.topology, pos)
    known = {(g["chain"], g["after"], g["before"]) for g in gaps if g["where"] == "internal"}
    unknown_breaks = [b for b in breaks if (b["chain"], b["after"], b["before"]) not in known]
    sg = [(a, pos[a.index]) for a in fixer.topology.atoms() if a.name == "SG" and a.residue.name == "CYS"]
    ss = []
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            if float(np.linalg.norm(sg[i][1] - sg[j][1])) < 0.25:
                ss.append(f"Cys{sg[i][0].residue.id}:{sg[i][0].residue.chain.id}-"
                          f"Cys{sg[j][0].residue.id}:{sg[j][0].residue.chain.id}")
    chains = []
    for c in fixer.topology.chains():
        res = list(c.residues())
        kind = ("protein" if any(r.name in AMINO for r in res) else
                "nucleic" if any(r.name in NUCLEIC for r in res) else "other")
        chains.append({"chain": c.id, "residues": len(res), "kind": kind,
                       "first": res[0].id if res else "", "last": res[-1].id if res else ""})
    return {
        "file": os.path.abspath(pdb_path),
        "chains": chains,
        "has_seqres": bool([ln for ln in header if ln.startswith("SEQRES")]) or bool(sequences),
        "altlocs": altlocs,
        "hydrogens_present": n_h,
        "nonstandard": nonstandard,
        "gaps": [{k: v for k, v in g.items() if not k.startswith("_")} for g in gaps],
        "chain_breaks_without_sequence": unknown_breaks,
        "missing_atoms": missing_atoms,
        "disulfides": ss,
        "waters": len({(ln[21], ln[22:27]) for ln in waters}),
        "hetero_groups": _het_groups(hets),
        "tools": tool_status(),
    }


# ----------------------------------------------------------------------------------------------- gaps

_MODELLER_SCRIPT = r'''
import json, sys
from modeller import Environ, Selection, log
from modeller.automodel import AutoModel, assess, refine

args = json.load(open(sys.argv[1]))
log.minimal()
env = Environ(rand_seed=-abs(int(args.get("seed", 2)) % 49999 + 2))
env.io.atom_files_directory = ["."]
env.io.hetatm = False


class GapModel(AutoModel):
    """Only the missing residues and their neighbours move; every other atom keeps the template position."""

    def select_atoms(self):
        return Selection(*[self.residues[i] for i in args["select"]])


a = GapModel(env, alnfile="gap.ali", knowns="tmpl", sequence="target", assess_methods=(assess.DOPE,))
a.starting_model = 1
a.ending_model = int(args.get("n_models", 3))
a.md_level = refine.fast
a.make()
ok = [m for m in a.outputs if m["failure"] is None]
if not ok:
    sys.exit(2)
best = min(ok, key=lambda m: m["DOPE score"])
json.dump({"best": best["name"], "dope": best["DOPE score"],
           "models": [[m["name"], m["DOPE score"]] for m in ok]}, open("gap_result.json", "w"))
'''


def _labels_for_insert(prev_label, next_label, count, where):
    """Residue numbers for ``count`` inserted residues: free numbers between the neighbours when they exist,
    insertion codes after ``prev_label`` otherwise."""
    def split(label):
        m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", str(label).strip())
        return (int(m.group(1)), m.group(2)) if m else (None, "")

    p = split(prev_label)[0] if prev_label else None
    n = split(next_label)[0] if next_label else None
    if where == "N-terminus" and n is not None:
        return [str(n - count + i) for i in range(count)]
    if where == "C-terminus" and p is not None:
        return [str(p + 1 + i) for i in range(count)]
    if p is not None and n is not None and n - p - 1 >= count:
        return [str(p + 1 + i) for i in range(count)]
    if p is not None:
        codes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return [f"{p}{codes[i % 26]}" for i in range(count)]
    return [str(i + 1) for i in range(count)]


def build_gaps_modeller(fixer, gaps, workdir, n_models=3, seed=2, progress=None, timeout=3600):
    """
    Build the ``gaps`` (from ``_gaps_from_fixer``) of the protein chains with MODELLER.  Returns
    (pdb lines of the protein chains with crystal numbering and chain ids, built residue labels).
    """
    import openmm.app as app

    chains = list(fixer.topology.chains())
    want = defaultdict(dict)
    for g in gaps:
        ci, ri = g["_key"]
        want[ci][ri] = fixer.missingResidues[(ci, ri)]
    prot_chains = [ci for ci, c in enumerate(chains) if any(r.name in AMINO for r in c.residues())]
    os.makedirs(workdir, exist_ok=True)
    # template: protein chains only, heavy atoms, in topology order
    sub = app.Modeller(fixer.topology, fixer.positions)
    sub.delete([c for ci, c in enumerate(sub.topology.chains()) if ci not in prot_chains])
    buf = io.StringIO()
    app.PDBFile.writeFile(sub.topology, sub.positions, buf, keepIds=True)
    # MODELLER sees a phosphorylated residue as its parent; the phosphate is put back afterwards
    tmpl_lines, phospho_lines = [], defaultdict(list)
    for ln in buf.getvalue().splitlines():
        if _is_coord(ln) and ln[17:20].strip() in PHOSPHO:
            phospho_lines[(ln[21], ln[22:27].strip())].append(ln)
            if ln[12:16].strip() in PHOSPHATE_ATOMS:
                continue
            ln = "ATOM  " + ln[6:17] + f"{PHOSPHO[ln[17:20].strip()]:>3}" + ln[20:]
        tmpl_lines.append(ln)
    _write(os.path.join(workdir, "tmpl.pdb"), [ln for ln in tmpl_lines if not ln.startswith("END")])
    tmpl_seq, targ_seq, select, labels, built = [], [], [], [], []
    gi = 0
    for ci in prot_chains:
        res = list(chains[ci].residues())
        ts, ss = [], []
        for ri in range(len(res) + 1):
            ins = want[ci].get(ri)
            if ins:
                where = "N-terminus" if ri == 0 else ("C-terminus" if ri == len(res) else "internal")
                new = _labels_for_insert(res[ri - 1].id if ri > 0 else "", res[ri].id if ri < len(res) else "",
                                         len(ins), where)
                first = gi
                for name, lab in zip(ins, new):
                    ts.append("-")
                    ss.append(ONE_LETTER.get(name, "A"))
                    labels.append((chains[ci].id, lab, name))
                    built.append(f"{name}{lab}:{chains[ci].id}")
                    select.append(gi)
                    gi += 1
                if ri > 0:
                    select.append(first - 1)                     # neighbours move with the loop
                if ri < len(res):
                    select.append(gi)
            if ri < len(res):
                r = res[ri]
                ts.append(ONE_LETTER.get(r.name, "A"))
                ss.append(ONE_LETTER.get(r.name, "A"))
                labels.append((chains[ci].id, r.id, r.name))
                gi += 1
        tmpl_seq.append("".join(ts))
        targ_seq.append("".join(ss))
    with open(os.path.join(workdir, "gap.ali"), "w") as fh:
        fh.write(">P1;tmpl\nstructureX:tmpl:FIRST:@:LAST:@::::\n" + "/".join(tmpl_seq) + "*\n")
        fh.write(">P1;target\nsequence:target::::::::\n" + "/".join(targ_seq) + "*\n")
    with open(os.path.join(workdir, "gap_args.json"), "w") as fh:
        json.dump({"select": sorted(set(select)), "n_models": int(n_models), "seed": int(seed)}, fh)
    with open(os.path.join(workdir, "gap_build.py"), "w") as fh:
        fh.write(_MODELLER_SCRIPT)
    _log(progress, f"MODELLER: building {len(built)} missing residue(s) ({n_models} models)")
    r = subprocess.run([python_executable(), "gap_build.py", "gap_args.json"], cwd=workdir, capture_output=True,
                       text=True, timeout=timeout)
    result = os.path.join(workdir, "gap_result.json")
    if r.returncode != 0 or not os.path.isfile(result):
        raise RuntimeError(f"MODELLER failed: {(r.stdout + r.stderr)[-1200:]}")
    with open(result) as fh:
        meta = json.load(fh)
    # renumber the model: residues in order -> (chain, label) of the target
    out, idx, last = [], -1, None
    with open(os.path.join(workdir, meta["best"])) as fh:
        for ln in fh:
            if not ln.startswith("ATOM"):
                continue
            key = (ln[21], ln[22:27])
            if key != last:
                idx += 1
                last = key
            cid, lab, _name = labels[idx]
            m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", lab)
            num, icode = (int(m.group(1)), m.group(2)) if m else (idx + 1, "")
            out.append(f"{ln[:21]}{cid}{num:>4d}{icode or ' '}{ln[27:]}")
    if idx + 1 != len(labels):
        raise RuntimeError(f"MODELLER model has {idx + 1} residues, expected {len(labels)}")
    if phospho_lines:
        restored, done = [], set()
        for ln in out:
            key = (ln[21], ln[22:27].strip())
            if key in phospho_lines:
                if key not in done:
                    restored.extend(phospho_lines[key])
                    done.add(key)
                continue
            restored.append(ln)
        out = restored
    return out, built, {"dope": meta.get("dope"), "models": meta.get("models")}


# ----------------------------------------------------------------------------------------------- protonation

def run_reduce(pdb_in, workdir):
    """``reduce -FLIP`` on a heavy-atom file.  Returns (path with reduce's hydrogens, flips, his tautomers)."""
    exe = reduce_executable()
    if not exe:
        return None, [], {}
    out_path = os.path.join(workdir, "reduce_flip.pdb")
    r = subprocess.run([exe, "-FLIP", "-Quiet", pdb_in], capture_output=True, text=True, timeout=600)
    if not r.stdout.strip():
        return None, [], {}
    with open(out_path, "w") as fh:
        fh.write(r.stdout)
    flips = []
    pat = re.compile(r"USER  MOD\s+(?:Set\s*[\d.]+|Single)\s*:\s*(\S)?\s*(-?\d+)([A-Za-z]?)\s+(ASN|GLN|HIS)\s+\S*:"
                     r"FLIP\s+(\S+)")
    for ln in r.stdout.splitlines():
        if ln.startswith("USER  MOD") and ":FLIP" in ln:
            m = pat.search(ln)
            if m:
                flips.append({"chain": m.group(1) or "", "resi": m.group(2) + m.group(3), "resn": m.group(4),
                              "what": m.group(5)})
    his = defaultdict(set)
    for ln in r.stdout.splitlines():
        if _is_coord(ln) and ln[17:20] == "HIS" and ln[12:16].strip() in ("HD1", "HE2"):
            his[(ln[21], ln[22:27].strip())].add(ln[12:16].strip())
    taut = {}
    for key, hs in his.items():
        taut[key] = "HIP" if hs == {"HD1", "HE2"} else ("HID" if "HD1" in hs else "HIE")
    return out_path, flips, taut


def run_propka(pdb_path):
    """[{resn, chain, resi, pka, model_pka}] from PROPKA 3 (empty when PROPKA is missing or fails)."""
    try:
        import contextlib

        import propka.run
    except Exception:
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            mol = propka.run.single(pdb_path, optargs=["--quiet"], write_pka=False)
    except Exception:
        return None
    conf = mol.conformations.get("AVR") or next(iter(mol.conformations.values()))
    out = []
    for g in conf.groups:
        if g.residue_type in TITRATABLE and g.atom is not None:
            out.append({"resn": g.residue_type, "chain": str(g.atom.chain_id).strip(), "resi": str(g.atom.res_num),
                        "pka": round(float(g.pka_value), 2), "model_pka": round(float(g.model_pka), 2)})
    return out


def protonation_plan(topology, pkas, his_tautomers, ph, metal_variants=None, positions_nm=None):
    """
    Modeller.addHydrogens variants (one per residue, None = OpenMM's default) and the list of decisions.
    PROPKA decides the charge (protonated when pKa > pH), reduce the neutral His tautomer, metals override.
    """
    by_key = {}
    for p in pkas or []:
        by_key[(p["chain"], p["resi"], p["resn"])] = p
    variants = [None] * topology.getNumResidues()
    decisions = []
    ss_cys = set()
    for bond in topology.bonds():
        a, b = bond
        if a.name == "SG" and b.name == "SG":
            ss_cys.update({a.residue.index, b.residue.index})
    metal_variants = metal_variants or {}
    for res in topology.residues():
        name = res.name.upper()
        if name not in TITRATABLE:
            continue
        key = (res.chain.id, str(res.id).strip())
        p = by_key.get((key[0], key[1], name))
        pka = p["pka"] if p else MODEL_PKA[name]
        label = f"{name.capitalize()}{res.id}:{res.chain.id}"
        if (key[0], key[1]) in metal_variants:
            variants[res.index] = metal_variants[(key[0], key[1])]
            decisions.append({"residue": label, "state": variants[res.index], "pka": p and p["pka"],
                              "why": "metal ligand"})
            continue
        state = None
        if name == "ASP":
            state = "ASH" if pka > ph else "ASP"
        elif name == "GLU":
            state = "GLH" if pka > ph else "GLU"
        elif name == "LYS":
            state = "LYN" if pka < ph else "LYS"
        elif name == "CYS":
            if res.index in ss_cys:
                continue
            state = "CYM" if pka < ph else "CYS"
        elif name == "HIS":
            if pka > ph:
                state = "HIP"
            else:
                t = his_tautomers.get(key)
                state = t if t in ("HID", "HIE") else _his_tautomer_by_hbond(res, positions_nm)
        elif name == "TYR":
            if pka < ph:
                decisions.append({"residue": label, "state": "TYR", "pka": pka,
                                  "why": f"pKa {pka:.1f} < pH but Amber has no tyrosinate; kept neutral"})
            continue
        variants[res.index] = state
        default = {"ASP": "ASP", "GLU": "GLU", "LYS": "LYS", "CYS": "CYS"}.get(name)
        shifted = p is not None and abs(p["pka"] - p["model_pka"]) >= 1.0
        if state != default or shifted or name == "HIS":
            why = ("PROPKA" if p else "standard pKa") + f" {pka:.1f} vs pH {ph:g}"
            if name == "HIS" and state in ("HID", "HIE"):
                why += "; tautomer from " + ("reduce" if his_tautomers.get(key) in ("HID", "HIE") else "H-bonds")
            decisions.append({"residue": label, "state": state, "pka": pka, "why": why})
    return variants, decisions


def _his_tautomer_by_hbond(res, positions_nm):
    """HID when only ND1 has a nearby acceptor (O within 3.2 A of another residue), HIE otherwise."""
    if positions_nm is None:
        return "HIE"
    atoms = {a.name: a for a in res.atoms()}
    if "ND1" not in atoms or "NE2" not in atoms:
        return "HIE"
    pos = positions_nm
    others = [a.index for a in res.chain.topology.atoms() if a.element is not None and a.element.symbol == "O"
              and a.residue is not res]
    if not others:
        return "HIE"
    o = pos[others]

    def near(n):
        return bool(np.any(np.linalg.norm(o - pos[atoms[n].index], axis=1) < 0.32))

    return "HID" if near("ND1") and not near("NE2") else "HIE"


_PHOSPHO_HYDROGENS = """<Residues>
 <Residue name="SEP">
  <H name="H" parent="N"/><H name="H2" parent="N" terminal="N"/><H name="H3" parent="N" terminal="N"/>
  <H name="HA" parent="CA"/><H name="HB2" parent="CB"/><H name="HB3" parent="CB"/>
 </Residue>
 <Residue name="TPO">
  <H name="H" parent="N"/><H name="H2" parent="N" terminal="N"/><H name="H3" parent="N" terminal="N"/>
  <H name="HA" parent="CA"/><H name="HB" parent="CB"/>
  <H name="HG21" parent="CG2"/><H name="HG22" parent="CG2"/><H name="HG23" parent="CG2"/>
 </Residue>
 <Residue name="PTR">
  <H name="H" parent="N"/><H name="H2" parent="N" terminal="N"/><H name="H3" parent="N" terminal="N"/>
  <H name="HA" parent="CA"/><H name="HB2" parent="CB"/><H name="HB3" parent="CB"/>
  <H name="HD1" parent="CD1"/><H name="HD2" parent="CD2"/><H name="HE1" parent="CE1"/><H name="HE2" parent="CE2"/>
 </Residue>
</Residues>
"""
_PHOSPHO_H_LOADED = False


def _load_phospho_hydrogens():
    """Hydrogen definitions of SEP / TPO / PTR (the phosphate as the dianion, its state at pH > 6)."""
    global _PHOSPHO_H_LOADED
    if _PHOSPHO_H_LOADED:
        return
    import openmm.app as app

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
        fh.write(_PHOSPHO_HYDROGENS)
    try:
        app.Modeller.loadHydrogenDefinitions(fh.name)
        _PHOSPHO_H_LOADED = True
    finally:
        os.unlink(fh.name)


def _split_at_breaks(fixer):
    """(new PDBFixer whose chains are split at unlinked C-N / O3'-P, breaks, removed lone residues)."""
    import openmm.app as app

    from .component_library_runtime import split_chains_at_breaks

    top, breaks = split_chains_at_breaks(fixer.topology, fixer.positions)
    if not breaks:
        return fixer, [], []
    lone = []
    mod = app.Modeller(top, fixer.positions)
    drop = []
    for chain in mod.topology.chains():
        res = list(chain.residues())
        if len(res) == 1 and res[0].name in AMINO:
            lone.append(f"{res[0].name}{res[0].id}:{chain.id}")
            drop.append(chain)
    if drop:
        mod.delete(drop)
    buf = io.StringIO()
    app.PDBFile.writeFile(mod.topology, mod.positions, buf, keepIds=True)
    from pdbfixer import PDBFixer

    new = PDBFixer(pdbfile=io.StringIO(buf.getvalue()))
    new.missingResidues = {}
    return new, breaks, lone


_EXTRA_HYDROGENS = """<Residues>
 <Residue name="CYM">
  <H name="H" parent="N"/><H name="H2" parent="N" terminal="N"/><H name="H3" parent="N" terminal="N"/>
  <H name="HA" parent="CA"/><H name="HB2" parent="CB"/><H name="HB3" parent="CB"/>
 </Residue>
</Residues>
"""
_EXTRA_H_LOADED = False


def _load_extra_hydrogens():
    """Hydrogen definitions of the Cys thiolate (CYM): OpenMM's CYS has only the CYS / CYX variants."""
    global _EXTRA_H_LOADED
    if _EXTRA_H_LOADED:
        return
    import openmm.app as app

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
        fh.write(_EXTRA_HYDROGENS)
    try:
        app.Modeller.loadHydrogenDefinitions(fh.name)
        _EXTRA_H_LOADED = True
    finally:
        os.unlink(fh.name)


def _bond_from_templates(topology, positions, forcefield, names):
    """PDBFile only bonds residues it knows: give residues ``names`` (e.g. SEP/TPO/PTR) their internal bonds
    from the force-field template and their peptide bonds to the neighbours (C-N within 2 A)."""
    pos = _positions_nm(positions)
    have = {frozenset((b[0].index, b[1].index)) for b in topology.bonds()}

    def link(a, b):
        key = frozenset((a.index, b.index))
        if key not in have:
            topology.addBond(a, b)
            have.add(key)

    templates = getattr(forcefield, "_templates", {})
    for chain in topology.chains():
        residues = list(chain.residues())
        for i, res in enumerate(residues):
            if res.name not in names:
                continue
            atoms = {a.name: a for a in res.atoms()}
            tmpl = templates.get(res.name)
            if tmpl is not None:
                for i1, i2 in tmpl.bonds:
                    n1, n2 = tmpl.atoms[i1].name, tmpl.atoms[i2].name
                    if n1 in atoms and n2 in atoms:
                        link(atoms[n1], atoms[n2])
            for other, (x, y) in ((residues[i - 1] if i > 0 else None, ("C", "N")),
                                  (residues[i + 1] if i + 1 < len(residues) else None, ("N", "C"))):
                if other is None:
                    continue
                oa = {a.name: a for a in other.atoms()}
                if x in oa and y in atoms and np.linalg.norm(pos[oa[x].index] - pos[atoms[y].index]) < 0.2:
                    link(oa[x], atoms[y])


def covalent_links_to_hets(protein_lines, het_lines, max_A=2.05):
    """Protein heavy atoms within bonding distance of a ligand heavy atom (covalent inhibitors, adducts)."""
    lig = [ln for ln in het_lines if ln[17:20].strip().upper() not in METAL_RESN and not _is_hydrogen(ln)]
    if not lig:
        return []
    lxyz = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in lig])
    out = []
    for ln in protein_lines:
        if not _is_coord(ln) or _is_hydrogen(ln) or ln[12:16].strip() in ("C", "O", "N", "CA"):
            continue
        p = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        d = np.linalg.norm(lxyz - p, axis=1)
        k = int(np.argmin(d))
        if d[k] <= max_A:
            out.append({"chain": ln[21], "resi": ln[22:27].strip(), "resn": ln[17:20].strip(),
                        "residue": f"{ln[17:20].strip().capitalize()}{ln[22:27].strip()}:{ln[21]}",
                        "atom": ln[12:16].strip(), "het": f"{lig[k][17:20].strip()}{lig[k][22:27].strip()}",
                        "het_atom": lig[k][12:16].strip(), "distance_A": float(d[k])})
    return out


def metal_variant_map(protein_lines, het_lines):
    """{(chain, resi): variant} for Cys / His bound to a metal ion (component_library_runtime's rules)."""
    metals = [ln for ln in het_lines if ln[17:20].strip().upper() in METAL_RESN]
    if not metals:
        return {}, []
    try:
        import openmm.app as app

        from .component_library_runtime import coordination_variants

        text = "".join(ln.rstrip("\n") + "\n" for ln in protein_lines + metals) + "END\n"
        pdb = app.PDBFile(io.StringIO(text))
        variants, changed = coordination_variants(pdb.topology, _positions_nm(pdb.positions))
    except Exception:
        return {}, []
    out = {}
    for res in pdb.topology.residues():
        v = variants[res.index]
        if v:
            out[(res.chain.id, str(res.id).strip())] = v
    return out, changed


def variants_from_hydrogens(topology, positions):
    """
    Protonation states already present in a structure, read from its hydrogens (by distance, so any naming
    works): ASH / GLH (H on a carboxylate O), HID / HIE / HIP, LYN (two H on NZ), CYM (no H on a free SG).
    One entry per residue, None where the residue has no hydrogens or is not titratable.  Lets a prepared
    protein keep its states when Leonardo's MD / component paths rebuild the hydrogens.
    """
    try:
        from openmm import unit

        pos = np.array([[v.x, v.y, v.z] for v in positions.value_in_unit(unit.nanometer)], float)
    except Exception:
        pos = np.asarray(positions, float)
    out = [None] * topology.getNumResidues()
    ss = set()
    sg = [a for a in topology.atoms() if a.name == "SG"]
    for i in range(len(sg)):
        for j in range(i + 1, len(sg)):
            if np.linalg.norm(pos[sg[i].index] - pos[sg[j].index]) < 0.25:
                ss.update({sg[i].residue.index, sg[j].residue.index})
    for res in topology.residues():
        name = res.name.upper()
        if name not in ("ASP", "GLU", "HIS", "LYS", "CYS", "ASH", "GLH", "HID", "HIE", "HIP", "LYN", "CYM"):
            continue
        atoms = list(res.atoms())
        hs = [a for a in atoms if a.element is not None and a.element.symbol == "H"]
        if not hs:
            continue
        hxyz = pos[[a.index for a in hs]]

        def n_h(atom_name):
            a = next((x for x in atoms if x.name == atom_name), None)
            if a is None:
                return 0
            return int(np.sum(np.linalg.norm(hxyz - pos[a.index], axis=1) < 0.115))

        if name in ("ASP", "ASH"):
            out[res.index] = "ASH" if n_h("OD1") + n_h("OD2") else "ASP"
        elif name in ("GLU", "GLH"):
            out[res.index] = "GLH" if n_h("OE1") + n_h("OE2") else "GLU"
        elif name in ("HIS", "HID", "HIE", "HIP"):
            d, e = n_h("ND1"), n_h("NE2")
            out[res.index] = "HIP" if d and e else ("HID" if d else ("HIE" if e else None))
        elif name in ("LYS", "LYN"):
            k = n_h("NZ")
            out[res.index] = "LYN" if k == 2 else ("LYS" if k == 3 else None)
        elif name in ("CYS", "CYM") and res.index not in ss:
            out[res.index] = "CYS" if n_h("SG") else "CYM"
    return out


# ----------------------------------------------------------------------------------------------- main

def prepare_protein(pdb_in, pdb_out, *, ph=7.4, fill_gaps=True, fill_termini=False, gap_method="auto",
                    sequences=None, keep_waters=False, keep_hets=True, flips=True, use_propka=True,
                    minimize=True, n_models=3, seed=2, workdir=None, progress=None):
    """
    Prepare ``pdb_in`` and write ``pdb_out``.  ``gap_method``: auto (MODELLER, PDBFixer without it), modeller,
    pdbfixer.  ``sequences``: {chain: one-letter sequence} when the file has no SEQRES.  Returns the report.
    """
    import openmm.app as app

    work = workdir or tempfile.mkdtemp(prefix="leonardo_prep_")
    os.makedirs(work, exist_ok=True)
    report = OrderedDict(input=os.path.abspath(pdb_in), output=os.path.abspath(pdb_out), ph=float(ph),
                         steps=[], warnings=[])
    tools = tool_status()
    report["tools"] = tools

    def step(name, detail):
        report["steps"].append({"step": name, "detail": detail})
        _log(progress, f"{name}: {detail}")

    with open(pdb_in) as fh:
        lines = fh.readlines()
    # 1. clean
    lines, altlocs = select_altlocs(lines)
    report["altlocs"] = altlocs
    if altlocs:
        step("Alternate locations", f"{len(altlocs)} residue(s): kept the most occupied conformation")
    subs = _substitutions()
    header, polymer, waters, hets = classify_lines(lines, modified=subs)
    n_h = sum(1 for ln in polymer if _is_hydrogen(ln))
    polymer = [ln for ln in polymer if not _is_hydrogen(ln)]
    waters = [ln for ln in waters if not _is_hydrogen(ln)]
    if n_h:
        step("Hydrogens", f"{n_h} existing hydrogen(s) removed; all are rebuilt for pH {ph:g}")
    if not polymer:
        raise ValueError("no protein or nucleic-acid atoms in the structure")
    report["hetero_groups"] = _het_groups(hets)
    if hets and not keep_hets:
        step("Hetero groups", f"{len(report['hetero_groups'])} ligand/ion group(s) removed")
        hets = []
    elif hets:
        step("Hetero groups", f"{len(report['hetero_groups'])} ligand/ion group(s) kept unchanged "
             "(prepare ligands separately: their protonation is not changed here)")
    if not keep_waters:
        if waters:
            step("Waters", f"{len({(ln[21], ln[22:27]) for ln in waters})} water(s) removed")
        waters = []
    fixer = _fixer_from_lines(header + polymer, sequences)
    fixer.findNonstandardResidues()
    phospho_ff = phospho_forcefield_files()
    kept = [(r, rep) for r, rep in fixer.nonstandardResidues if r.name in PHOSPHO and phospho_ff]
    replaced = [(r, rep) for r, rep in fixer.nonstandardResidues if (r, rep) not in kept]
    report["nonstandard"] = [f"{r.name}{r.id}:{r.chain.id}->{rep}" for r, rep in replaced]
    report["kept_modified"] = [f"{r.name}{r.id}:{r.chain.id}" for r, _rep in kept]
    if replaced:
        fixer.nonstandardResidues = replaced
        fixer.replaceNonstandardResidues()
        step("Modified residues", "replaced by the parent residue: " + ", ".join(report["nonstandard"]))
        lost = [label for (r, _rep), label in zip(replaced, report["nonstandard"])
                if r.name not in SILENT_REPLACEMENTS]
        if lost:
            report["warnings"].append("modification removed (no template): " + ", ".join(lost)
                                      + "; keep it with build_validated_system / the component library")
    if kept:
        step("Modified residues", "kept (ff14SB phosphorylated-residue templates): "
             + ", ".join(report["kept_modified"]))
    original_heavy = {(a.residue.chain.id, str(a.residue.id).strip(), a.name)
                      for a in fixer.topology.atoms()}
    # 2. missing residues
    fixer.findMissingResidues()
    bad = _sequence_mismatches(fixer)
    if bad:
        report["warnings"].append("the full sequence does not match the structure for chain(s) "
                                  + ", ".join(bad) + "; their gaps are not filled")
        fixer.missingResidues = {k: v for k, v in fixer.missingResidues.items()
                                 if list(fixer.topology.chains())[k[0]].id not in bad}
    gaps = _gaps_from_fixer(fixer)
    report["gaps"] = [{k: v for k, v in g.items() if not k.startswith("_")} for g in gaps]
    to_fill = [g for g in gaps if fill_gaps and (g["where"] == "internal" or fill_termini)]
    method = None
    built = []
    if to_fill:
        prot_gaps = [g for g in to_fill
                     if any(r.name in AMINO for r in list(fixer.topology.chains())[g["_key"][0]].residues())]
        use_modeller = gap_method in ("auto", "modeller") and prot_gaps and tools["modeller"]
        if gap_method == "modeller" and not tools["modeller"]:
            report["warnings"].append("MODELLER is not available (or has no license): gaps built with PDBFixer")
        if use_modeller:
            try:
                model_lines, built, meta = build_gaps_modeller(fixer, prot_gaps, os.path.join(work, "modeller"),
                                                               n_models=n_models, seed=seed, progress=progress)
                method = "MODELLER"
                report["modeller"] = meta
                # nucleic-acid / other chains stay as they were
                sub = app.Modeller(fixer.topology, fixer.positions)
                sub.delete([c for c in sub.topology.chains() if any(r.name in AMINO for r in c.residues())])
                buf = io.StringIO()
                app.PDBFile.writeFile(sub.topology, sub.positions, buf, keepIds=True)
                other = [ln for ln in buf.getvalue().splitlines() if ln.startswith(("ATOM", "HETATM", "TER"))]
                fixer = _fixer_from_lines(model_lines + ["TER"] + other)
            except Exception as exc:
                report["warnings"].append(f"MODELLER failed ({str(exc)[:300]}); gaps built with PDBFixer")
                use_modeller = False
        if not use_modeller:
            keys = {g["_key"] for g in to_fill}
            fixer.missingResidues = {k: v for k, v in fixer.missingResidues.items() if k in keys}
            method = "PDBFixer"
            built = [f"{g['sequence']} ({g['chain']}: {g['after'] or 'start'}-{g['before'] or 'end'})"
                     for g in to_fill]
        else:
            fixer.missingResidues = {}
        step("Missing residues", f"{sum(g['length'] for g in to_fill)} residue(s) in {len(to_fill)} gap(s) "
             f"built with {method}")
    else:
        fixer.missingResidues = {}
        if gaps:
            step("Missing residues", f"{len(gaps)} gap(s) left open (fill_gaps off or termini only); "
                 "the chain is split there")
    report["built_residues"] = built
    report["gap_method"] = method
    # unfilled gaps: split the chain there before the missing atoms are added, so that every new terminus gets
    # its OXT / terminal hydrogens; a residue left alone between two gaps cannot be a peptide and is dropped
    if method != "PDBFixer":
        fixer, breaks, lone = _split_at_breaks(fixer)
        report["chain_breaks"] = breaks
        if breaks:
            step("Chain breaks", f"{len(breaks)} unfilled break(s): chains split with proper termini "
                 f"({', '.join(breaks[:6])}{'...' if len(breaks) > 6 else ''})")
        if lone:
            report["warnings"].append("isolated residue(s) between gaps removed: " + ", ".join(lone))
    # 3. missing heavy atoms
    fixer.findMissingAtoms()
    n_missing = sum(len(v) for v in fixer.missingAtoms.values()) + sum(len(v) for v in fixer.missingTerminals.values())
    report["missing_atoms"] = [{"residue": f"{r.name}{r.id}:{r.chain.id}", "atoms": [a.name for a in atoms]}
                               for r, atoms in fixer.missingAtoms.items()]
    fixer.addMissingAtoms(seed=int(seed))
    if n_missing:
        step("Missing atoms", f"{n_missing} heavy atom(s) added in {len(fixer.missingAtoms)} residue(s)"
             + (" and terminal OXT" if fixer.missingTerminals else ""))
    heavy_path = os.path.join(work, "heavy.pdb")
    with open(heavy_path, "w") as fh:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    # 4. flips and tautomers (reduce), pKa (PROPKA)
    taut, flip_list = {}, []
    src_for_h = heavy_path
    if flips and tools["reduce"]:
        red_path, flip_list, taut = run_reduce(heavy_path, work)
        if red_path:
            with open(red_path) as fh:
                # TER records keep the chains split at unfilled gaps
                red = [ln for ln in fh if (_is_coord(ln) and not _is_hydrogen(ln)) or ln.startswith("TER")]
            src_for_h = _write(os.path.join(work, "heavy_flipped.pdb"), red)
            step("Side-chain flips", f"{len(flip_list)} Asn/Gln/His flipped by reduce" if flip_list else
                 "no Asn/Gln/His flip needed (reduce)")
    elif flips:
        report["warnings"].append("reduce not found: no Asn/Gln/His flips; OpenMM chooses His tautomers")
    report["flips"] = flip_list
    pkas = None
    if use_propka and tools["propka"]:
        with open(src_for_h) as fh:
            prot = [ln for ln in fh if _is_coord(ln)]
        ligs = [ln for ln in hets if ln[17:20].strip().upper() not in METAL_RESN]
        pk_in = _write(os.path.join(work, "propka_in.pdb"), prot + ligs)
        pkas = run_propka(pk_in)
        if pkas is None and ligs:
            pkas = run_propka(_write(os.path.join(work, "propka_in_protein.pdb"), prot))
            report["warnings"].append("PROPKA could not read the ligands: pKa computed without them")
        if pkas is None:
            report["warnings"].append("PROPKA failed: standard pKa values used")
    elif use_propka:
        report["warnings"].append("PROPKA is not installed: standard pKa values used (OpenMM pH rule)")
    report["pka"] = pkas or []
    # 5. hydrogens (OpenMM)
    pdb = app.PDBFile(src_for_h)
    topology, positions = pdb.topology, pdb.positions
    if any(r.name in PHOSPHO for r in topology.residues()):
        _bond_from_templates(topology, positions, app.ForceField(*phospho_ff), set(PHOSPHO))
        _load_phospho_hydrogens()
    pos_nm = _positions_nm(positions)
    with open(src_for_h) as fh:
        prot_lines = [ln for ln in fh if _is_coord(ln)]
    metal_map, metal_changes = metal_variant_map(prot_lines, hets)
    links = covalent_links_to_hets(prot_lines, hets)
    report["covalent_links"] = [f"{x['residue']}:{x['atom']}-{x['het']}:{x['het_atom']} {x['distance_A']:.2f} A"
                                for x in links]
    thioethers = {(x["chain"], x["resi"]) for x in links if x["resn"] == "CYS" and x["atom"] == "SG"}
    for x in links:
        if (x["chain"], x["resi"]) not in thioethers:
            report["warnings"].append(f"{x['residue']} is covalently bound to {x['het']} through {x['atom']}: the "
                                      "hydrogens on that atom are those of the free residue; build the adduct with "
                                      "build_validated_system / the component library")
    for key in thioethers:
        metal_map[key] = "CYM"           # no HG; written as CYX below (a thioether, not a thiolate)
    variants, decisions = protonation_plan(topology, pkas, taut, float(ph), metal_map, pos_nm)
    for d in decisions:
        if d["why"] == "metal ligand" and tuple(re.match(r"^\w+?(-?\d+[A-Za-z]?):(\S*)$", d["residue"]).groups()[::-1]) \
                in thioethers:
            d["why"], d["state"] = "covalently bound to a ligand (thioether)", "CYX"
    report["protonation"] = decisions
    report["metal_ligands"] = metal_changes
    has_phospho = any(r.name in PHOSPHO for r in topology.residues())
    if has_phospho:
        ff_files = list(phospho_ff) + ([amber_ffxml("tip3p_standard.xml")] if waters else [])
    else:
        ff_files = ["amber14-all.xml"] + (["amber14/tip3pfb.xml"] if waters else [])
    report["forcefield"] = [os.path.basename(f) for f in ff_files]
    ff = app.ForceField(*ff_files)
    modeller = app.Modeller(topology, positions)
    if "CYM" in variants:
        # OpenMM's hydrogen definitions have no CYM variant: the residue is renamed (Amber has its template)
        _load_extra_hydrogens()
        for res in modeller.topology.residues():
            if variants[res.index] == "CYM":
                res.name = "CYM"
                variants[res.index] = None
    if waters:
        wpdb = app.PDBFile(io.StringIO("".join(ln.rstrip("\n") + "\n" for ln in waters) + "END\n"))
        modeller.add(wpdb.topology, wpdb.positions)
        variants = variants + [None] * wpdb.topology.getNumResidues()
    try:
        modeller.addHydrogens(ff, pH=float(ph), variants=variants)
    except ValueError as exc:
        # a residue without a force-field template (e.g. a modified residue at a cut terminus): hydrogens
        # from the geometric definitions only
        report["warnings"].append(f"force field cannot place every hydrogen ({str(exc).split('.')[0]}); "
                                  "hydrogens placed by geometry and relaxed by the minimization when possible")
        modeller.addHydrogens(None, pH=float(ph), variants=variants)
    n_hyd = sum(1 for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H")
    charged = {"ASH": 0, "GLH": 0, "HIP": 0, "LYN": 0, "CYM": 0}
    for d in decisions:
        if d["state"] in charged:
            charged[d["state"]] += 1
    source = "PROPKA pKa" if pkas else "standard pKa"
    unusual = ", ".join(f"{v} {k}" for k, v in charged.items() if v)
    step("Protonation", f"{n_hyd} hydrogens at pH {ph:g} ({source}; "
         + (unusual if unusual else "every titratable residue in its usual state") + ")")
    final_top, final_pos = modeller.topology, modeller.positions
    # 6. minimization
    if minimize:
        try:
            final_pos, mini = _minimize(final_top, final_pos, ff_files, original_heavy, bool(waters))
            report["minimization"] = mini
            step("Minimization", f"{mini['steps']}; heavy-atom shift of restrained atoms "
                 f"{mini['restrained_rmsd_A']:.2f} A (max {mini['restrained_max_A']:.2f} A)")
        except Exception as exc:
            report["warnings"].append(f"minimization skipped: {str(exc)[:300]}")
    # write: protein (+ waters) with hydrogens, then the hetero groups as they were
    for res in final_top.residues():
        if res.name == "CYM" and (res.chain.id, str(res.id).strip()) in thioethers:
            res.name = "CYX"
    buf = io.StringIO()
    app.PDBFile.writeFile(final_top, final_pos, buf, keepIds=True)
    body = [ln for ln in buf.getvalue().splitlines() if not ln.startswith(("END", "CONECT"))]
    tail = [ln.rstrip("\n") for ln in hets]
    if tail:
        body.append("TER")
    os.makedirs(os.path.dirname(os.path.abspath(pdb_out)) or ".", exist_ok=True)
    _write(pdb_out, body + tail)
    report["atoms"] = final_top.getNumAtoms()
    report["hydrogens"] = n_hyd
    report["changed_residues"] = _changed_residue_keys(report)
    with open(os.path.splitext(pdb_out)[0] + "_report.json", "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    return report


def _changed_residue_keys(report):
    """(chain, resi) groups for display: built, protonation changes, flips, completed side chains."""
    def key(label):
        m = re.match(r"^[A-Za-z]{1,4}?(-?\d+[A-Za-z]?):(\S*)$", str(label))
        return (m.group(2), m.group(1)) if m else None

    out = {"built": [], "protonation": [], "flipped": [], "completed": []}
    for b in report.get("built_residues") or []:
        k = key(b)
        if k:
            out["built"].append(k)
    for d in report.get("protonation") or []:
        if d["state"] in ("ASH", "GLH", "HIP", "LYN", "CYM"):
            k = key(d["residue"])
            if k:
                out["protonation"].append(k)
    for f in report.get("flips") or []:
        out["flipped"].append((f["chain"], f["resi"]))
    for m in report.get("missing_atoms") or []:
        k = key(m["residue"])
        if k:
            out["completed"].append(k)
    return out


def _minimize(topology, positions, ff_files, original_heavy, has_waters, k_kj_nm2=4184.0, max_iter=2000):
    """Relax hydrogens and built atoms; heavy atoms that were in the crystal are restrained (10 kcal/mol/A^2).
    Implicit solvent (GBn2) without waters, vacuum with them.  CPU platform."""
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    files = list(ff_files) + ([] if has_waters else ["implicit/gbn2.xml"])
    ff = app.ForceField(*files)
    # a 1.2 nm cutoff: GB without one scales as N^2 (100 s for 6000 atoms on a CPU) and changes nothing here
    system = ff.createSystem(topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.2 * unit.nanometer,
                             constraints=None)
    force = mm.CustomExternalForce("0.5*leo_prep_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("leo_prep_k", float(k_kj_nm2))
    for p in ("x0", "y0", "z0"):
        force.addPerParticleParameter(p)
    pos = positions.value_in_unit(unit.nanometer)
    restrained = []
    for a in topology.atoms():
        if a.element is None or a.element.symbol == "H":
            continue
        if (a.residue.chain.id, str(a.residue.id).strip(), a.name) in original_heavy:
            force.addParticle(a.index, [pos[a.index][0], pos[a.index][1], pos[a.index][2]])
            restrained.append(a.index)
    system.addForce(force)
    integ = mm.VerletIntegrator(0.001)
    platform = mm.Platform.getPlatformByName("CPU")
    ctx = mm.Context(system, integ, platform)
    ctx.setPositions(positions)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, int(max_iter))
    st = ctx.getState(getEnergy=True, getPositions=True)
    e1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    new = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    old = np.array(pos)
    d = np.linalg.norm(new[restrained] - old[restrained], axis=1) * 10.0 if restrained else np.zeros(1)
    del ctx, integ
    return st.getPositions(), {"energy_before_kj": round(e0, 1), "energy_after_kj": round(e1, 1),
                               "steps": "restrained minimization (" + ("vacuum, waters kept" if has_waters
                                                                     else "GBn2 implicit solvent") + ")",
                               "restrained_atoms": len(restrained),
                               "restrained_rmsd_A": float(np.sqrt(np.mean(d ** 2))),
                               "restrained_max_A": float(d.max())}


def launch_job(args, workdir, python=None):
    """
    Run ``prepare_protein`` in a child process (the GUI and the assistant stay responsive; a cancel kills it).
    ``args``: prepare_protein keywords plus pdb_in, pdb_out and optionally pdb_id (RCSB sequence).
    Returns (Popen, log path, result json path); the log has one ``PROGRESS ...`` line per step.
    """
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    os.makedirs(workdir, exist_ok=True)
    args = dict(args)
    args.setdefault("result_json", os.path.join(workdir, "prep_result.json"))
    args_path = os.path.join(workdir, "prep_args.json")
    with open(args_path, "w") as fh:
        json.dump(args, fh)
    log_path = os.path.join(workdir, "prep_log.txt")
    log = open(log_path, "w")
    kw = {"start_new_session": True} if os.name == "posix" else {}
    proc = subprocess.Popen([python or python_executable(), "-m", "leonardo.protein_prep_runtime", args_path],
                            cwd=workdir,
                            stdout=log, stderr=subprocess.STDOUT, env=env, **kw)
    proc._leonardo_log = log
    return proc, log_path, args["result_json"]


def job_progress(log_path):
    """Last PROGRESS line of a job log ('' before the first)."""
    try:
        with open(log_path) as fh:
            rows = [ln[9:].strip() for ln in fh if ln.startswith("PROGRESS")]
        return rows[-1] if rows else ""
    except OSError:
        return ""


def job_result(proc, log_path, result_json):
    """(report, error) of a finished job."""
    log = getattr(proc, "_leonardo_log", None)
    if log is not None and not log.closed:
        log.close()
    res = {}
    if os.path.isfile(result_json):
        with open(result_json) as fh:
            res = json.load(fh)
    if res.get("ok"):
        return res["report"], ""
    tail = ""
    try:
        with open(log_path) as fh:
            tail = fh.read()[-800:]
    except OSError:
        pass
    return None, str(res.get("error") or tail or "protein preparation failed")


def _main(args_path):
    """Background job: ``python -m leonardo.protein_prep_runtime args.json`` (PROGRESS lines, result json)."""
    with open(args_path) as fh:
        args = json.load(fh)
    result_json = args.pop("result_json")
    pdb_id = args.pop("pdb_id", "")
    try:
        if pdb_id and not args.get("sequences") and args.get("fill_gaps", True):
            with open(args["pdb_in"]) as fh:
                has_seqres = any(ln.startswith("SEQRES") for ln in fh)
            if not has_seqres:
                print(f"PROGRESS full sequence of {pdb_id} from RCSB", flush=True)
                try:
                    args["sequences"] = fetch_rcsb_sequences(pdb_id)
                except Exception as exc:
                    print(f"PROGRESS RCSB sequence unavailable ({exc}); gaps are reported, not filled", flush=True)
        pdb_in, pdb_out = args.pop("pdb_in"), args.pop("pdb_out")
        report = prepare_protein(pdb_in, pdb_out, progress=lambda m: print("PROGRESS " + m, flush=True), **args)
        out = {"ok": True, "report": report}
    except Exception as exc:
        import traceback

        traceback.print_exc()
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    with open(result_json, "w") as fh:
        json.dump(out, fh, default=str)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1]))
