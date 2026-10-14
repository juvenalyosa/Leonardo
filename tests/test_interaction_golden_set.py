"""Golden set: Leonardo's protein-ligand interaction detection on complexes whose key contacts are
documented in the literature. Each case asserts the contacts that must be found (type, residue,
atoms) and the ones that would be clearly wrong.

Structures: tests/data/interactions_golden/*.pdb (RCSB entries trimmed to residues within 12 A of the
ligand, first altloc) and examples/films (3PTB, 2CPP). Ligand bond orders come from the vendored PDB
chemical-component files (tests/data/interactions_golden/ccd), so the run is offline.

  3PTB  trypsin-benzamidine     Asp189 salt bridge (bidentate, charge-assisted), Ser190 OG, Gly219 O
                                (Marquart et al. 1983, Acta Cryst B39:480)
  1IEP  Abl-imatinib            Thr315, Met318 hinge, Glu286, Asp381 backbone, piperazinium-Ile360 O
                                (Nagar et al. 2002, Cancer Res 62:4236)
  1M17  EGFR-erlotinib          hinge Met769 N...quinazoline N1 (atom N2 in the file), 2.7 A
                                (Stamos et al. 2002, JBC 277:46265)
  2CPP  P450cam-camphor         camphor O...Tyr96 OH (Poulos et al. 1987, JMB 195:687)
  3HS4  CA II-acetazolamide     Zn-N1 (sulfonamide), N1-H...Thr199 OG1, Thr199 N...O1, Gln92 NE2
                                (Sippel et al. 2009, Acta Cryst F65:992)
  1US0  aldose reductase-IDD594 C-Br...Thr113 OG1 halogen bond 2.97 A; carboxylate to Tyr48, His110,
                                Trp111 (Howard et al. 2004, Proteins 55:792)
  1ACL  AChE-decamethonium      quaternary ammonium cation-pi with Trp84 and Trp279
                                (Harel et al. 1993, PNAS 90:9031)
"""
import os
import shutil
import tempfile
from collections import Counter

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pymol")
pytest.importorskip("rdkit")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(REPO, "tests", "data", "interactions_golden")


def _path(name):
    for p in (os.path.join(GOLDEN, name),
              os.path.join(REPO, "examples", "films", "01_molecular_docking", name),
              os.path.join(REPO, "examples", "films", "08_ensemble_redocking", name)):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(name)


@pytest.fixture(scope="module")
def leo():
    params = tempfile.mkdtemp(prefix="leo_golden_params_")
    shutil.copytree(os.path.join(GOLDEN, "ccd"), os.path.join(params, "ccd"))
    old = os.environ.get("LEONARDO_PARAMETERS_DIR")
    os.environ["LEONARDO_PARAMETERS_DIR"] = params
    from leonardo.replay import _headless_leonardo

    work = tempfile.mkdtemp(prefix="leo_golden_work_")
    app, ctx, reg = _headless_leonardo(work)
    ctx.dlg._rcsb_chemcomp_smiles = lambda *a, **k: ""          # offline: no RCSB lookups
    cache = {}

    def run(pdb, chain, resi, resn):
        key = (pdb, chain, resi, resn)
        if key not in cache:
            from pymol import cmd

            cmd.delete("all")
            obj = "g_" + os.path.splitext(os.path.basename(pdb))[0]
            reg.call("load_structure_file", {"path": _path(pdb), "object_name": obj}, ctx)
            out = reg.call("analyze_interactions", {"object": obj, "ligand_chain": chain,
                                                    "ligand_resi": resi, "ligand_resn": resn}, ctx)
            feats = ctx.dlg._cached_features
            cache[key] = {
                "out": out,
                "obj": obj,
                "records": [dict(r) for r in ctx.dlg.interactions],
                "lig_atoms": [dict(a) for a in feats.get("ligand_atoms", [])],
                "lig_h": sum(1 for a in feats.get("ligand_atoms", []) if str(a.get("element")).upper() == "H"),
                "objects_with_ligand": [o for o in cmd.get_names("objects", enabled_only=1)
                                        if cmd.get_type(o) == "object:molecule"
                                        and cmd.count_atoms(f"({o}) and resn {resn}") > 0],
                "refined": ctx.dlg._ligand_refined_mol,
            }
        return cache[key]

    yield run, ctx, reg
    if old is None:
        os.environ.pop("LEONARDO_PARAMETERS_DIR", None)
    else:
        os.environ["LEONARDO_PARAMETERS_DIR"] = old
    from pymol import cmd

    cmd.delete("all")
    shutil.rmtree(params, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)


def _res(rec):
    return f"{rec.get('prot_resn')}{rec.get('prot_resi')}"


def _find(records, itype, residue, prot_atom=None, lig_atom=None):
    out = []
    for r in records:
        if r.get("type") != itype or _res(r) != residue:
            continue
        if prot_atom and prot_atom not in str(r.get("prot_atom", "")).split(","):
            continue
        if lig_atom and lig_atom not in str(r.get("lig_atom", "")).split(","):
            continue
        out.append(r)
    return out


def _assert_has(records, itype, residue, prot_atom=None, lig_atom=None, max_dist=None):
    hits = _find(records, itype, residue, prot_atom, lig_atom)
    table = "\n".join(f"  {r['type']} {_res(r)} {r['prot_atom']}-{r['lig_atom']} {float(r['distance']):.2f}"
                      for r in records if r["type"] != "Van der Waals")
    assert hits, f"missing {itype} {residue} {prot_atom or ''}-{lig_atom or ''}; found:\n{table}"
    if max_dist is not None:
        assert min(float(h["distance"]) for h in hits) <= max_dist
    return hits


def _assert_invariants(case):
    records = case["records"]
    assert records
    lig_el = {str(a["name"]): str(a["element"]).upper() for a in case["lig_atoms"]}
    seen = Counter()
    for r in records:
        assert r.get("distance_kind"), r
        if r["type"] == "Hydrogen Bond":
            # the record distance is always donor-acceptor; H...A only when the H is known
            assert abs(float(r["distance"]) - float(r["da_distance"])) < 1e-6
            assert r["distance_kind"] == "D...A"
            assert r["hbond_mode"] in ("explicit_hydrogen", "heavy_atom_fallback")
            assert r["donor_side"] in ("protein", "ligand")
            if r["hbond_mode"] == "explicit_hydrogen":
                assert float(r["ha_distance"]) < float(r["da_distance"])
            else:
                assert r["ha_distance"] is None and float(r["distance"]) <= 3.6 + 1e-6
            seen[(r["prot_res"], r["prot_atom"], r["lig_atom"])] += 1
        if r["type"] == "Hydrophobic":
            # only carbon...carbon (no polar atom in a hydrophobic contact)
            assert all(n.strip().startswith("C") for n in str(r["prot_atom"]).split(",")), r
            assert all(lig_el.get(n.strip()) == "C" for n in str(r["lig_atom"]).split(",")), r
        if r["type"] in ("Pi-Pi Stacking", "T-Shaped Pi-Pi"):
            assert 0.0 <= float(r["details"].split("angle ")[1].split("°")[0]) <= 90.0
    # one H-bond record per atom pair (not once per direction)
    assert not [k for k, n in seen.items() if n > 1], seen
    # a salt-bridged pair is not also reported as a separate H-bond on the same group atoms
    for sb in (r for r in records if r["type"] == "Salt Bridge"):
        for pair in sb.get("hbond_pairs", []):
            assert not _find(records, "Hydrogen Bond", _res(sb), pair["prot_atom"], pair["lig_atom"])
    # the ligand is in the scene once (no refined copy next to the crystal ligand)
    assert len(case["objects_with_ligand"]) == 1, case["objects_with_ligand"]


def test_3ptb_benzamidine(leo):
    run, _ctx, _reg = leo
    case = run("3PTB.pdb", "A", "1", "BEN")
    _assert_invariants(case)
    # benzamidinium C7H9N2+: 9 H (was 17: the ring was read as cyclohexane without bond orders)
    assert case["lig_h"] == 9
    recs = case["records"]
    sb = _assert_has(recs, "Salt Bridge", "ASP189", max_dist=4.0)
    assert len(_find(recs, "Salt Bridge", "ASP189")) == 1          # was 2 (amidine split in two groups)
    pairs = {(p["prot_atom"], p["lig_atom"]) for p in sb[0]["hbond_pairs"]}
    assert len(pairs) >= 2 and "charge-assisted" in sb[0]["details"]
    _assert_has(recs, "Hydrogen Bond", "SER190", "OG", max_dist=3.2)
    _assert_has(recs, "Hydrogen Bond", "GLY219", "O", "N1", max_dist=3.0)
    assert not _find(recs, "Hydrogen Bond", "ASP189")               # folded into the salt bridge
    assert not [r for r in recs if r["type"] in ("Cation-Pi", "Metal Coordination", "Halogen Bond")]


def test_1iep_imatinib(leo):
    run, _ctx, _reg = leo
    case = run("1IEP.pdb", "A", "201", "STI")
    _assert_invariants(case)
    assert case["lig_h"] == 32                                      # C29H31N7O + piperazinium H+
    recs = case["records"]
    _assert_has(recs, "Hydrogen Bond", "THR315", "OG1", "N13", max_dist=3.1)
    _assert_has(recs, "Hydrogen Bond", "MET318", "N", "N3", max_dist=3.1)   # hinge
    _assert_has(recs, "Hydrogen Bond", "GLU286", None, "N21", max_dist=3.2)  # amide N-H
    _assert_has(recs, "Hydrogen Bond", "ASP381", "N", "O29", max_dist=3.1)   # amide C=O
    _assert_has(recs, "Hydrogen Bond", "ILE360", "O", "N51", max_dist=3.0)   # piperazinium N-H
    # the amide N21 is not a cation: no "salt bridge" with Glu286 (was reported before)
    assert not _find(recs, "Salt Bridge", "GLU286")
    cations = {str(a.get("ionic_group_id")) for a in case["lig_atoms"] if a.get("is_cation")}
    assert len(cations) == 1


def test_1m17_erlotinib_hinge(leo):
    run, _ctx, _reg = leo
    case = run("1M17.pdb", "A", "999", "AQ4")
    _assert_invariants(case)
    assert case["lig_h"] == 23                                      # C22H23N3O4, neutral
    recs = case["records"]
    hinge = _assert_has(recs, "Hydrogen Bond", "MET769", "N", "N2", max_dist=2.8)
    assert hinge[0]["donor_side"] == "protein" and hinge[0]["hbond_mode"] == "heavy_atom_fallback"
    assert not _find(recs, "Van der Waals", "MET769")               # was misclassified as vdW
    assert not [a for a in case["lig_atoms"] if a.get("is_cation")]


def test_2cpp_camphor(leo):
    run, _ctx, _reg = leo
    case = run("2CPP.pdb", "A", "422", "CAM")
    _assert_invariants(case)
    assert case["lig_h"] == 16                                      # C10H16O (was 18)
    recs = case["records"]
    tyr = _assert_has(recs, "Hydrogen Bond", "TYR96", "OH", "O", max_dist=2.8)
    assert tyr[0]["donor_side"] == "protein"                        # camphor has no donor
    assert all(r["donor_side"] == "protein" for r in recs if r["type"] == "Hydrogen Bond")


def test_3hs4_acetazolamide_zinc(leo):
    run, _ctx, _reg = leo
    case = run("3HS4.pdb", "A", "701", "AZM")
    _assert_invariants(case)
    assert case["lig_h"] == 6                                       # C4H6N4O3S2 (was 15)
    recs = case["records"]
    zn = _assert_has(recs, "Metal Coordination", "ZN301", "ZN", "N1", max_dist=2.1)
    assert zn[0]["distance_kind"] == "metal-donor atom"
    _assert_has(recs, "Hydrogen Bond", "THR199", "OG1", "N1", max_dist=3.0)
    _assert_has(recs, "Hydrogen Bond", "THR199", "N", "O1", max_dist=3.1)
    _assert_has(recs, "Hydrogen Bond", "GLN92", "NE2", "O3", max_dist=3.3)
    # the Zn-bound histidines are not H-bond partners of the Zn-bound sulfonamide N
    for his in ("HIS94", "HIS96", "HIS119"):
        assert not _find(recs, "Hydrogen Bond", his), his
    # Thr200 OG1 has one H: at most one H-bond from it
    assert len([r for r in _find(recs, "Hydrogen Bond", "THR200", "OG1") if r["donor_side"] == "protein"]) <= 1


def test_1us0_halogen_bond(leo):
    run, _ctx, _reg = leo
    case = run("1US0.pdb", "A", "320", "LDT")
    _assert_invariants(case)
    recs = case["records"]
    xb = _assert_has(recs, "Halogen Bond", "THR113", "OG1", "BR8", max_dist=3.1)
    assert xb[0]["distance_kind"] == "X...A"
    _assert_has(recs, "Hydrogen Bond", "TYR48", "OH", max_dist=3.0)
    _assert_has(recs, "Hydrogen Bond", "HIS110", "NE2", max_dist=3.0)
    _assert_has(recs, "Hydrogen Bond", "TRP111", "NE1", max_dist=3.2)
    _assert_has(recs, "Pi-Pi Stacking", "TRP111")
    # fluorine is not a sigma-hole donor
    assert not [r for r in recs if r["type"] == "Halogen Bond" and str(r["lig_atom"]).startswith("F")]


def test_1acl_decamethonium_cation_pi(leo):
    run, _ctx, _reg = leo
    case = run("1ACL.pdb", "A", "999", "DME")
    _assert_invariants(case)
    recs = case["records"]
    _assert_has(recs, "Cation-Pi", "TRP84", max_dist=5.0)
    _assert_has(recs, "Cation-Pi", "TRP279", max_dist=5.0)
    assert not [r for r in recs if r["type"] == "Hydrogen Bond"]    # quaternary N has no H


def test_input_hydrogens_are_kept_and_used(leo, tmp_path):
    """A ligand that comes with hydrogens keeps them (no replacement by placed ones) and its
    H-bonds are judged with the D-H...A angle of those hydrogens."""
    run, _ctx, _reg = leo
    from rdkit import Chem

    base = run("3PTB.pdb", "A", "1", "BEN")
    mol = base["refined"]
    assert mol is not None
    conf = mol.GetConformer()
    lines = open(_path("3PTB.pdb")).read().splitlines()
    h_lines = []
    serial = 5000
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        serial += 1
        name = f"H{len(h_lines) + 1}"
        h_lines.append(f"HETATM{serial:5d} {name:<4s} BEN A   1    {p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00 20.00           H")
    last_ben = max(i for i, l in enumerate(lines) if l.startswith("HETATM") and l[17:20] == "BEN")
    out = lines[: last_ben + 1] + h_lines + lines[last_ben + 1:]
    path = tmp_path / "3PTB_withH.pdb"
    path.write_text("\n".join(out) + "\n")
    case = run(str(path), "A", "1", "BEN")
    assert case["lig_h"] == 9
    assert not any(a.get("placed") for a in case["lig_atoms"] if str(a.get("element")).upper() == "H")
    gly = _assert_has(case["records"], "Hydrogen Bond", "GLY219", "O", "N1", max_dist=3.0)
    assert gly[0]["hbond_mode"] == "explicit_hydrogen" and gly[0]["h_placed"] is False
    _assert_has(case["records"], "Salt Bridge", "ASP189")


def test_peptide_caps_are_protein_not_ligand(leo):
    run, ctx, _reg = leo
    from pymol import cmd

    cmd.delete("all")
    rows = [
        ("ATOM", "C", "ACE", 1, (-1.2, 1.0, 0.0), "C"), ("ATOM", "O", "ACE", 1, (-1.2, 2.23, 0.0), "O"),
        ("ATOM", "CH3", "ACE", 1, (-2.5, 0.2, 0.0), "C"), ("ATOM", "N", "ALA", 2, (0.0, 0.3, 0.0), "N"),
        ("ATOM", "CA", "ALA", 2, (1.3, 1.0, 0.0), "C"), ("ATOM", "C", "ALA", 2, (2.5, 0.1, 0.0), "C"),
        ("ATOM", "O", "ALA", 2, (2.4, -1.13, 0.0), "O"), ("ATOM", "CB", "ALA", 2, (1.4, 1.9, 1.2), "C"),
        ("ATOM", "N", "NME", 3, (3.7, 0.7, 0.0), "N"), ("ATOM", "C", "NME", 3, (4.9, -0.1, 0.0), "C"),
    ]
    block = "".join(f"{rec:<6s}{i + 1:5d} {nm:<4s} {rn} A{ri:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n"
                    for i, (rec, nm, rn, ri, (x, y, z), el) in enumerate(rows))
    cmd.read_pdbstr(block, "capped")
    assert ctx.dlg._auto_detect_ligand_in_object("capped") is None   # was ('A', 1, 'ACE')
    lig = "HETATM   11  C1  LIG A 900       0.000   0.000   4.000  1.00  0.00           C\n" \
          "HETATM   12  O1  LIG A 900       0.000   1.200   4.300  1.00  0.00           O\n"
    cmd.read_pdbstr(block + lig, "capped_lig")
    assert ctx.dlg._auto_detect_ligand_in_object("capped_lig") == ("A", 900, "LIG")
    cmd.delete("all")


@pytest.mark.parametrize("pdb, resn, resi", [("3PTB.pdb", "BEN", "1"), ("1IEP.pdb", "STI", "201")])
def test_ligand_without_chemical_component_name(leo, tmp_path, pdb, resn, resi):
    """A docked or renamed ligand (UNL) has no CCD template: its bond orders come from the geometry
    (Open Babel, or ring planarity where Open Babel misreads imatinib's pyrimidine) and the result
    must match the named ligand."""
    run, _ctx, _reg = leo
    named = run(pdb, "A", resi, resn)
    text = open(_path(pdb)).read().replace(f" {resn} A", " UNL A")
    path = tmp_path / f"{resn}_as_UNL.pdb"
    path.write_text(text)
    case = run(str(path), "A", resi, "UNL")
    assert case["lig_h"] == named["lig_h"]

    def key(recs):
        return sorted((r["type"], _res(r), r["prot_atom"], r["lig_atom"]) for r in recs if r["type"] != "Van der Waals")

    assert key(case["records"]) == key(named["records"])


def test_ligand_typing_from_input_hydrogens(leo):
    """Without a refined model (trajectories), the ligand is typed from its own atoms: bond orders on
    the heavy atoms and the input hydrogens, whose count sets the formal charge."""
    _run, ctx, _reg = leo
    from rdkit import Chem

    heavy = [l for l in open(_path("3PTB.pdb")) if l.startswith("HETATM") and l[17:20] == "BEN"]
    mol = Chem.AddHs(Chem.MolFromSmiles("NC(=[NH2+])c1ccccc1"))
    template = Chem.MolFromPDBBlock("".join(heavy))
    from rdkit.Chem import AllChem

    placed = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles("NC(=[NH2+])c1ccccc1"), template)
    placed = Chem.AddHs(placed, addCoords=True)
    conf = placed.GetConformer()
    h_lines = [f"HETATM{9000 + a.GetIdx():5d}  H{k:<2d} BEN A   1    "
               f"{conf.GetAtomPosition(a.GetIdx()).x:8.3f}{conf.GetAtomPosition(a.GetIdx()).y:8.3f}"
               f"{conf.GetAtomPosition(a.GetIdx()).z:8.3f}  1.00 20.00           H\n"
               for k, a in enumerate([a for a in placed.GetAtoms() if a.GetAtomicNum() == 1], start=1)]
    assert mol is not None and len(h_lines) == 9
    chem, source = ctx.dlg._ligand_chem_from_input("".join(heavy + h_lines), "BEN")
    assert chem is not None and "input hydrogens" in source
    assert Chem.GetFormalCharge(chem) == 1
    assert sum(1 for a in chem.GetAtoms() if a.GetIsAromatic()) == 6
    assert sum(1 for a in chem.GetAtoms() if a.GetAtomicNum() == 1) == 9
