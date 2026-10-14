"""Interface interactions of macromolecular complexes: synthetic geometry per detector, then three
complexes with literature-documented interfaces (vendored, trimmed RCSB entries in tests/data/macro)."""
import os

import numpy as np
import pytest

from leonardo import macro_interaction_runtime as M

DATA = os.path.join(os.path.dirname(__file__), "data", "macro")


def at(name, resn, resi, chain, element, xyz):
    return {"name": name, "resn": resn, "resi": str(resi), "chain": chain, "element": element,
            "coord": np.array(xyz, dtype=float)}


def hexagon(names, resn, resi, chain, center, normal_axis=2, radius=1.39, elements=None):
    """Planar six-membered ring around ``center`` with its normal along x (0), y (1) or z (2)."""
    out = []
    axes = [k for k in range(3) if k != normal_axis]
    for k, name in enumerate(names):
        xyz = np.array(center, dtype=float)
        xyz[axes[0]] += radius * np.cos(np.pi / 3 * k)
        xyz[axes[1]] += radius * np.sin(np.pi / 3 * k)
        out.append(at(name, resn, resi, chain, (elements or {}).get(name, name[0]), xyz))
    return out


def types(result, t):
    return [r for r in result["interactions"] if r["type"] == t]


def pair(rec):
    return rec["a"]["residue"], rec["b"]["residue"]


# --------------------------------------------------------------------------- #
# synthetic geometry
# --------------------------------------------------------------------------- #
def ser_asp(d_da=2.8, acc_neighbor=(1.25, 0.0)):
    """Ser OG (chain A) donating along +x to Asp OD1 (chain B); CG placed relative to OD1."""
    og = np.array([1.43, 0.0, 0.0])
    od1 = og + [d_da, 0.0, 0.0]
    return [at("CB", "SER", 1, "A", "C", [0, 0, 0]), at("OG", "SER", 1, "A", "O", og),
            at("OD1", "ASP", 2, "B", "O", od1),
            at("CG", "ASP", 2, "B", "C", od1 + [acc_neighbor[0], acc_neighbor[1], 0.0])]


def test_hbond_heavy_atom_distance_and_angles():
    r = M.analyze_interface(ser_asp(), "A", "B", compute_sasa=False)
    hb = types(r, "Hydrogen Bond")
    assert len(hb) == 1
    assert hb[0]["distance_def"] == "D···A" and hb[0]["distance"] == pytest.approx(2.8)
    assert hb[0]["donor_side"] == "A" and hb[0]["mode"] == "heavy_atom" and hb[0]["angle"] == pytest.approx(180.0)
    assert hb[0]["subtype"] == "side chain–side chain"
    assert not types(M.analyze_interface(ser_asp(3.7), "A", "B", compute_sasa=False), "Hydrogen Bond")
    # acceptor antecedent folded back towards the donor: D...A-X < 90 deg
    bent = M.analyze_interface(ser_asp(2.8, (-0.8, 0.9)), "A", "B", compute_sasa=False)
    assert not types(bent, "Hydrogen Bond")


def test_hbond_explicit_hydrogen_uses_h_acceptor_geometry():
    atoms = ser_asp() + [at("HG", "SER", 1, "A", "H", [2.39, 0.0, 0.0])]
    hb = types(M.analyze_interface(atoms, "A", "B", compute_sasa=False), "Hydrogen Bond")
    assert len(hb) == 1 and hb[0]["distance_def"] == "H···A" and hb[0]["mode"] == "explicit_hydrogen"
    assert hb[0]["distance"] == pytest.approx(1.84, abs=0.01) and hb[0]["angle_def"] == "D–H···A"
    # hydrogen pointing away from the acceptor: D-H...A ~ 60 deg
    atoms = ser_asp() + [at("HG", "SER", 1, "A", "H", [1.2, 0.93, 0.0])]
    r = M.analyze_interface(atoms, "A", "B", compute_sasa=False)
    assert r["hydrogens_present"] and not types(r, "Hydrogen Bond")


def lys_glu(shift=0.0):
    return [at("CE", "LYS", 1, "A", "C", [-1.5, 0, 0]), at("NZ", "LYS", 1, "A", "N", [0, 0, 0]),
            at("OE1", "GLU", 5, "B", "O", [3.0 + shift, 1.1, 0]), at("OE2", "GLU", 5, "B", "O", [3.0 + shift, -1.1, 0]),
            at("CD", "GLU", 5, "B", "C", [3.7 + shift, 0, 0])]


def test_salt_bridge_needs_close_charged_atoms():
    sb = types(M.analyze_interface(lys_glu(), "A", "B", compute_sasa=False), "Salt Bridge")
    assert len(sb) == 1 and sb[0]["cation_group"] == "ammonium" and sb[0]["anion_group"] == "carboxylate"
    assert sb[0]["distance"] == pytest.approx(np.hypot(3.0, 1.1), abs=1e-3)
    assert not types(M.analyze_interface(lys_glu(3.0), "A", "B", compute_sasa=False), "Salt Bridge")


def test_salt_bridge_arginine_to_phosphate_is_backbone_readout():
    atoms = [at("NE", "ARG", 1, "A", "N", [0, 1.1, 0]), at("CZ", "ARG", 1, "A", "C", [0, 0, 0]),
             at("NH1", "ARG", 1, "A", "N", [1.15, -0.6, 0]), at("NH2", "ARG", 1, "A", "N", [-1.15, -0.6, 0]),
             at("P", "DG", 7, "B", "P", [1.2, -4.6, 0]), at("OP1", "DG", 7, "B", "O", [1.2, -3.1, 0]),
             at("OP2", "DG", 7, "B", "O", [2.5, -5.3, 0]), at("O5'", "DG", 7, "B", "O", [0.0, -5.4, 0])]
    r = M.analyze_interface(atoms, "A", "B", compute_sasa=False)
    sb = types(r, "Salt Bridge")
    assert len(sb) == 1 and sb[0]["anion_group"] == "phosphate"
    assert sb[0]["na_part"] == "phosphate" and sb[0]["readout"] == "phosphate backbone"
    assert r["interface_type"] == "protein–DNA"


def test_his_salt_bridge_is_conditional_without_hydrogens():
    atoms = [at("CG", "HIS", 1, "A", "C", [-1.0, 0.7, 0]), at("ND1", "HIS", 1, "A", "N", [0, 1.2, 0]),
             at("CE1", "HIS", 1, "A", "C", [0.6, 0, 0]), at("NE2", "HIS", 1, "A", "N", [0, -1.1, 0]),
             at("CD2", "HIS", 1, "A", "C", [-1.0, -0.6, 0]),
             at("OD1", "ASP", 2, "B", "O", [0.0, -3.9, 0]), at("OD2", "ASP", 2, "B", "O", [2.2, -3.9, 0]),
             at("CG", "ASP", 2, "B", "C", [1.1, -4.6, 0])]
    sb = types(M.analyze_interface(atoms, "A", "B", compute_sasa=False), "Salt Bridge")
    assert len(sb) == 1 and sb[0]["conditional"] == "His assumed protonated"


def test_hydrophobic_contacts_only_between_apolar_carbons():
    atoms = [at("CD1", "LEU", 1, "A", "C", [0, 0, 0]), at("CG1", "VAL", 9, "B", "C", [3.8, 0, 0]),
             at("CG", "ASP", 10, "B", "C", [0, 3.6, 0])]
    hy = types(M.analyze_interface(atoms, "A", "B", compute_sasa=False), "Hydrophobic")
    assert [pair(h) for h in hy] == [("LEU1:A", "VAL9:B")] and hy[0]["distance_def"] == "C···C"
    atoms[1]["coord"] = np.array([4.3, 0, 0])
    assert not types(M.analyze_interface(atoms, "A", "B", compute_sasa=False), "Hydrophobic")


PHE_RING = ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")
CYT_RING = ("N1", "C2", "N3", "C4", "C5", "C6")


def test_pi_stacking_aromatic_side_chain_on_nucleobase():
    atoms = hexagon(PHE_RING, "PHE", 56, "A", [0, 0, 0]) + hexagon(CYT_RING, "C", 10, "B", [0.5, 0.3, 3.6])
    atoms.append(at("O2'", "C", 10, "B", "O", [4.5, 4.5, 5.0]))        # RNA (2'-OH), out of the way
    r = M.analyze_interface(atoms, "A", "B", compute_sasa=False)
    st = types(r, "Pi-Pi Stacking")
    assert len(st) == 1 and pair(st[0]) == ("PHE56:A", "C10:B")
    assert st[0]["distance_def"] == "ring centroid–centroid" and st[0]["readout"] == "base (stacking)"
    assert st[0]["a"]["atom"].startswith("ring(") and st[0]["na_groove"] is None
    assert r["interface_type"] == "protein–RNA"


def test_t_shaped_pi_between_protein_rings():
    atoms = hexagon(PHE_RING, "PHE", 1, "A", [0, 0, 0]) + hexagon(PHE_RING, "TYR", 2, "B", [0, 0, 5.0], normal_axis=0)
    r = M.analyze_interface(atoms, "A", "B", compute_sasa=False)
    assert [x["type"] for x in r["interactions"] if "Pi" in x["type"]] == ["T-Shaped Pi-Pi"]


def test_cation_pi_arginine_over_a_base():
    atoms = hexagon(CYT_RING, "DC", 4, "B", [0, 0, 0]) + [
        at("CZ", "ARG", 3, "A", "C", [0.3, 0.2, 3.6]), at("NH1", "ARG", 3, "A", "N", [1.4, 0.8, 3.7]),
        at("NH2", "ARG", 3, "A", "N", [-0.8, 0.8, 3.7]), at("NE", "ARG", 3, "A", "N", [0.3, -1.1, 3.7])]
    cp = types(M.analyze_interface(atoms, "A", "B", compute_sasa=False), "Cation-Pi")
    assert len(cp) == 1 and cp[0]["cation_side"] == "A" and cp[0]["b"]["atom"].startswith("ring(")
    assert cp[0]["distance"] == pytest.approx(np.linalg.norm([0.3, 0.2, 3.6]), abs=1e-3)
    # in the ring plane: no cation-pi
    for a in atoms[6:]:
        a["coord"] = a["coord"] - [0, 0, a["coord"][2]] + [4.5, 0, 0]
    assert not types(M.analyze_interface(atoms, "A", "B", compute_sasa=False), "Cation-Pi")


def test_water_bridge_and_its_switch():
    atoms = [at("CB", "SER", 1, "A", "C", [-1.4, 0, 0]), at("OG", "SER", 1, "A", "O", [0, 0, 0]),
             at("O", "HOH", 301, "W", "O", [2.8, 0, 0]),
             at("OD1", "ASP", 2, "B", "O", [5.6, 0, 0]), at("CG", "ASP", 2, "B", "C", [6.85, 0, 0])]
    r = M.analyze_interface(atoms, "A", "B", compute_sasa=False)
    wb = types(r, "Water Bridge")
    assert len(wb) == 1 and wb[0]["via"]["residue"] == "HOH301:W" and wb[0]["optional"]
    assert not types(r, "Hydrogen Bond") and r["waters_present"]
    assert not types(M.analyze_interface(atoms, "A", "B", compute_sasa=False, include_waters=False), "Water Bridge")
    dry = M.analyze_interface([a for a in atoms if a["resn"] != "HOH"], "A", "B", compute_sasa=False)
    assert any("no crystal waters" in n for n in dry["notes"])


def test_disulfide_across_the_interface():
    atoms = [at("CB", "CYS", 5, "A", "C", [-1.5, 0.8, 0]), at("SG", "CYS", 5, "A", "S", [0, 0, 0]),
             at("SG", "CYS", 40, "B", "S", [2.04, 0, 0]), at("CB", "CYS", 40, "B", "C", [3.5, 0.8, 0])]
    ss = types(M.analyze_interface(atoms, "A", "B", compute_sasa=False), "Disulfide")
    assert len(ss) == 1 and ss[0]["distance"] == pytest.approx(2.04)


def test_metal_bridge_needs_donors_from_both_sides():
    zn = at("ZN", "ZN", 900, "Z", "ZN", [0, 0, 0])
    his = [at("CE1", "HIS", 10, "A", "C", [-2.9, 0.8, 0]), at("NE2", "HIS", 10, "A", "N", [-2.05, 0, 0]),
           at("CD2", "HIS", 10, "A", "C", [-2.9, -0.8, 0])]
    glu_b = [at("OE1", "GLU", 20, "B", "O", [2.0, 0, 0]), at("CD", "GLU", 20, "B", "C", [3.2, 0.3, 0])]
    r = M.analyze_interface(his + glu_b + [zn], "A", "B", compute_sasa=False)
    mb = types(r, "Metal Bridge")
    assert len(mb) == 1 and mb[0]["via"]["element"] == "ZN" and mb[0]["distance"] == pytest.approx(2.05)
    glu_a = [dict(a, chain="A", resi="21") for a in glu_b]
    other = [at("CA", "GLY", 50, "B", "C", [9.0, 0, 0])]
    assert not types(M.analyze_interface(his + glu_a + [zn] + other, "A", "B", compute_sasa=False), "Metal Bridge")


def test_dna_groove_and_rna_base_labels():
    assert M.dna_groove("G", "N7") == "major" and M.dna_groove("G", "O6") == "major"
    assert M.dna_groove("G", "N3") == "minor" and M.dna_groove("G", "N2") == "minor"
    assert M.dna_groove("A", "N1") is None and M.dna_groove("T", "C7") == "major" and M.dna_groove("C", "O2") == "minor"

    def guanine(resn, extra=()):
        return [at("C6", resn, 8, "B", "C", [1.2, 0, 0]), at("O6", resn, 8, "B", "O", [0, 0, 0]),
                at("C2", resn, 8, "B", "C", [0, -9, 0]), at("N3", resn, 8, "B", "N", [1.2, -9.7, 0])] + list(extra)
    arg = [at("CZ", "ARG", 1, "A", "C", [-4.0, 0.7, 0]), at("NH1", "ARG", 1, "A", "N", [-2.85, 0.0, 0])]
    asn = [at("CG", "ASN", 2, "A", "C", [2.6, -13.0, 0]), at("ND2", "ASN", 2, "A", "N", [1.9, -12.0, 0])]
    r = M.analyze_interface(guanine("DG") + arg + asn, "A", "B", compute_sasa=False)
    by_res = {h["a"]["residue"]: h for h in types(r, "Hydrogen Bond")}
    assert by_res["ARG1:A"]["readout"] == "base readout (major groove)" and by_res["ARG1:A"]["b"]["part"] == "base"
    assert by_res["ASN2:A"]["readout"] == "base readout (minor groove)" and by_res["ASN2:A"]["na_groove"] == "minor"
    rna = M.analyze_interface(guanine("G", [at("O2'", "G", 8, "B", "O", [9, 9, 9])]) + arg + asn, "A", "B",
                              compute_sasa=False)
    assert {h["readout"] for h in types(rna, "Hydrogen Bond")} == {"base"}


def test_shrake_rupley_isolated_and_buried_atoms():
    area = M.shrake_rupley([[0, 0, 0]], [1.7], probe=1.4, n_points=500)
    assert area[0] == pytest.approx(4 * np.pi * 3.1 ** 2, rel=1e-6)
    two = M.shrake_rupley([[0, 0, 0], [3.0, 0, 0]], [1.7, 1.7], probe=1.4, n_points=960)
    cap = 2 * np.pi * 3.1 * (3.1 - 1.5)          # spherical cap hidden by the neighbour
    assert two[0] == pytest.approx(4 * np.pi * 3.1 ** 2 - cap, rel=0.02) and two[0] == pytest.approx(two[1], rel=0.02)


def test_selection_grammar():
    atoms = [at("CA", "GLY", i, c, "C", [i, 0, 0]) for c in "AB" for i in range(1, 6)]
    assert len(M.select_side(atoms, "A")) == 5 and len(M.select_side(atoms, "A+B")) == 10
    assert [atoms[i]["resi"] for i in M.select_side(atoms, "A:2-3,5")] == ["2", "3", "5"]
    assert len(M.select_side(atoms, "chain B")) == 5
    assert M.select_side(atoms, [("B", "1")]) == [5]
    with pytest.raises(ValueError, match="no chain"):
        M.select_side(atoms, "Z")


def test_mmcif_reader_keeps_quoted_atom_names():
    text = """data_x
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 O "O5'" . G B 1.0 2.0 3.0 9 G P "O5'" 1
ATOM 2 N N7 . G B 4.0 5.0 6.0 9 G P N7 1
HETATM 3 O O . HOH C 7.0 8.0 9.0 501 HOH W O 1
ATOM 4 N N7 . G B 0.0 0.0 0.0 9 G P N7 2
#
"""
    atoms = M.parse_mmcif_text(text)
    assert [(a["name"], a["chain"], a["resi"], a["hetatm"]) for a in atoms] == [
        ("O5'", "P", "9", False), ("N7", "P", "9", False), ("O", "W", "501", True)]
    assert atoms[1]["coord"].tolist() == [4.0, 5.0, 6.0]


def test_default_sides_and_errors():
    atoms = ser_asp() + [at("CA", "GLY", 7, "C", "C", [40, 0, 0])]
    r = M.analyze_interface(atoms, compute_sasa=False)
    assert r["sides"]["A"]["chains"] == ["A"] and r["sides"]["B"]["chains"] == ["B"]
    with pytest.raises(ValueError, match="share"):
        M.analyze_interface(atoms, "A", "A+B", compute_sasa=False)
    with pytest.raises(ValueError):
        M.analyze_interface([at("CA", "GLY", 1, "A", "C", [0, 0, 0])], compute_sasa=False)


# --------------------------------------------------------------------------- #
# complexes with documented interfaces
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def barnase_barstar():
    return M.analyze_interface(os.path.join(DATA, "1BRS_AD.pdb"), "A", "D")


@pytest.fixture(scope="module")
def zif268():
    return M.analyze_interface(os.path.join(DATA, "1AAY.pdb"))


@pytest.fixture(scope="module")
def u1a():
    return M.analyze_interface(os.path.join(DATA, "1URN_AP.pdb"), "A", "P")


def test_barnase_barstar_1brs(barnase_barstar):
    """Buckle, Schreiber & Fersht 1994: barstar Asp39 binds barnase Arg83, Arg87 and His102; Glu76 pairs
    with Arg59, whose backbone NH binds Asp35; about 1600 A^2 buried."""
    r = barnase_barstar
    assert r["interface_type"] == "protein–protein"
    sb = {pair(x) for x in types(r, "Salt Bridge")}
    assert {("ARG83:A", "ASP39:D"), ("ARG87:A", "ASP39:D"), ("ARG59:A", "GLU76:D")} <= sb
    assert ("ARG59:A", "ASP39:D") not in sb                   # charge centres > 5.5 A apart
    hb = types(r, "Hydrogen Bond")
    hb_pairs = {(x["a"]["residue"], x["a"]["atom"], x["b"]["residue"], x["b"]["atom"]) for x in hb}
    assert ("HIS102:A", "NE2", "ASP39:D", "OD2") in hb_pairs
    assert ("HIS102:A", "ND1", "GLY31:D", "N") in hb_pairs
    bb = [x for x in hb if pair(x) == ("ARG59:A", "ASP35:D")]
    assert bb and bb[0]["subtype"] == "backbone–side chain" and bb[0]["donor_side"] == "A"
    assert all(x["distance"] <= 3.5 for x in hb)
    assert not types(r, "Metal Bridge") and not types(r, "Disulfide")
    assert 1400 <= r["sasa"]["buried_total"] <= 1800 and r["sasa"]["method"] == "shrake_rupley_numpy"
    assert {"ARG83:A", "ARG87:A", "HIS102:A", "ARG59:A"} <= set(r["interface_residues"]["A"])
    assert {"ASP39:D", "ASP35:D", "GLU76:D"} <= set(r["interface_residues"]["B"])
    assert r["hotspots"][0]["residue"] in ("ASP39:D", "HIS102:A")
    assert r["nucleic_acid"] is None
    assert types(r, "Water Bridge")                            # 1BRS has interfacial crystal waters


def test_zif268_dna_1aay(zif268):
    """Elrod-Erickson et al. 1996: Arg -1/+3/+6 of each finger read guanine O6/N7 in the major groove;
    His125/His153 contact phosphates; the Zn sites are protein-only."""
    r = zif268
    assert r["interface_type"] == "protein–DNA" and r["sides"]["B"]["chains"] == ["B", "C"]
    hb = types(r, "Hydrogen Bond")
    base = {(x["a"]["residue"], x["b"]["resn"], x["b"]["atom"]) for x in hb
            if x["readout"] == "base readout (major groove)"}
    for arg in ("ARG118:A", "ARG124:A", "ARG146:A", "ARG174:A", "ARG180:A"):
        assert any(a == arg and resn == "DG" and atom in ("O6", "N7") for a, resn, atom in base), arg
    assert ("HIS149:A", "DG", "N7") in base
    phos = {x["a"]["residue"] for x in hb + types(r, "Salt Bridge") if x["readout"] == "phosphate backbone"}
    assert {"HIS125:A", "HIS153:A", "ARG103:A", "ARG170:A"} <= phos
    assert not [x for x in hb if x["na_groove"] == "minor"]           # zinc fingers read the major groove
    na = r["nucleic_acid"]
    assert na["base_readout_specific"] >= 10 and na["backbone_specific"] >= 10
    assert na["atom_contacts_by_groove"]["major"] > 5 * na["atom_contacts_by_groove"].get("minor", 0)
    assert not types(r, "Metal Bridge")                               # Zn is coordinated by Cys/His only


def test_u1a_rna_1urn(u1a):
    """Oubridge et al. 1994: Tyr13 stacks on C10 and Phe56 on A11; Lys22 and Lys50 contact phosphates."""
    r = u1a
    assert r["interface_type"] == "protein–RNA"
    st = {pair(x) for x in types(r, "Pi-Pi Stacking")}
    assert {("TYR13:A", "C10:P"), ("PHE56:A", "A11:P")} <= st
    assert ("TYR13:A", "A11:P") not in st and ("PHE56:A", "C10:P") not in st
    sb = {(pair(x), x["readout"]) for x in types(r, "Salt Bridge")}
    assert (("LYS22:A", "A2:P"), "phosphate backbone") in sb and (("LYS50:A", "A11:P"), "phosphate backbone") in sb
    assert all(x["na_groove"] is None for x in r["interactions"])     # grooves are DNA-only
    assert r["nucleic_acid"]["base_readout_specific"] > r["nucleic_acid"]["backbone_specific"]
    assert 1300 <= r["sasa"]["buried_total"] <= 2200
