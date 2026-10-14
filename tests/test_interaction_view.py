"""Publication-style interaction views: the 3D scene composition and the 2D diagram."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("rdkit")

from leonardo import interaction_view as V  # noqa: E402


def _benzamidinium():
    """Benzamidine heavy atoms with PDB-style names and 3D coordinates (RDKit embedding, fixed seed)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("NC(=N)c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    mol = Chem.RemoveHs(mol)
    xyz = mol.GetConformer().GetPositions()
    names, counts = [], {}
    for a in mol.GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
        names.append(f"{a.GetSymbol()}{counts[a.GetSymbol()]}")
    return [(n, a.GetSymbol(), tuple(x)) for n, a, x in zip(names, mol.GetAtoms(), xyz)]


def _interactions(atoms):
    by = {n: np.array(x) for n, _e, x in atoms}
    ring = np.mean([by[f"C{i}"] for i in range(2, 8)], axis=0)
    out = []
    for itype, resn, resi, patom, latom, offset in (
            ("Salt Bridge", "ASP", "189", "OD1", "N1", (2.9, 0.0, 0.0)),
            ("Salt Bridge", "ASP", "189", "OD2", "N2", (2.9, 0.4, 0.0)),
            ("Hydrogen Bond", "SER", "190", "OG", "N2", (0.0, 3.0, 0.0)),
            ("Hydrogen Bond", "GLY", "219", "O", "N1", (0.0, -3.0, 0.0)),
            ("Hydrophobic", "VAL", "213", "CG1", "C5", (0.0, 0.0, 3.8)),
            ("Van der Waals", "GLN", "192", "OE1", "C6", (0.0, 0.0, -3.7))):
        lig = by[latom]
        out.append({"type": itype, "prot_res": f"{resn} A{resi}", "prot_resn": resn, "prot_resi": resi,
                    "prot_chain": "A", "prot_atom": patom, "lig_atom": latom,
                    "distance": float(np.linalg.norm(offset)), "prot_coord": tuple(lig + offset),
                    "lig_coord": tuple(lig)})
    out.append({"type": "Pi-Pi Stacking", "prot_res": "PHE A41", "prot_resn": "PHE", "prot_resi": "41",
                "prot_chain": "A", "prot_atom": "ring", "lig_atom": "ring", "distance": 3.9,
                "prot_coord": tuple(ring + (0.0, 0.0, 3.9)), "lig_coord": tuple(ring)})
    return out


def test_residue_labels_and_classes():
    assert V.residue_label("ASP", "189") == "Asp189"
    assert V.residue_label("ASP", "189", "B", with_chain=True) == "Asp189:B"
    assert V.residue_label("HOH", "301") == "HOH301"
    assert V.residue_label("ZN", "401") == "ZN401"
    assert V.residue_class("asp") == "negative" and V.residue_class("LYS") == "positive"
    assert V.residue_class("HOH") == "water" and V.residue_class("XYZ") == "other"


def test_residues_grouped_strongest_interaction_first():
    atoms = _benzamidinium()
    groups = V.group_by_residue(_interactions(atoms))
    keys = list(groups)
    assert keys[0] == ("A", "190", "SER") or keys[0] == ("A", "219", "GLY")     # H-bonds first
    assert len(groups[("A", "189", "ASP")]) == 2
    assert keys[-1] == ("A", "192", "GLN")                                        # contact-only last
    assert list(V.summarize(_interactions(atoms)))[:2] == ["Hydrogen Bond", "Salt Bridge"]


def test_label_is_placed_off_the_obstacles():
    taken = [(-1.0, -1.0, 1.0, 1.0)]
    xy = V.place_label((0.0, 0.0), (-1.0, 0.0), (2.0, 1.0), [(10.0, taken)])
    rect = (xy[0] - 1.0, xy[1] - 0.5, xy[0] + 1.0, xy[1] + 0.5)
    assert not V._rect_overlap(rect, taken[0])
    assert xy[0] > 0.0                       # pushed away from the side it should avoid


def test_2d_map_draws_every_residue_without_overlaps(tmp_path):
    pytest.importorskip("PIL")
    atoms = _benzamidinium()
    inter = _interactions(atoms)
    out = V.draw_2d_map(str(tmp_path / "map.png"), inter, atoms, resn="", title="BEN A1", fetch_ccd=False)
    assert os.path.isfile(out["image"]) and out["size"][0] >= 1000
    assert out["not_placed"] == [] and out["residues"] == 6
    rects = [h["rect"] for h in out["hotspots"]]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not V._rect_overlap(rects[i], rects[j])
    # the ring is drawn aromatic and every interaction maps back to its index in the input list
    assert "c1ccccc1" in out["smiles"]
    assert sorted(i for h in out["hotspots"] for i in h["interactions"]) == list(range(len(inter)))


def test_ligand_bond_orders_from_geometry_when_no_template():
    atoms = _benzamidinium()
    mol, names, info = V.ligand_molecule(atoms, resn="", fetch_ccd=False)
    from rdkit import Chem

    assert mol.GetNumAtoms() == 9 and set(names) == {n for n, _e, _x in atoms}
    assert sum(1 for b in mol.GetBonds() if b.GetIsAromatic()) == 6
    assert "geometry" in info["bond_orders"]
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)).count("=") <= 1


def test_scene_shows_one_label_per_interacting_residue():
    pytest.importorskip("pymol")
    from pymol import cmd

    cmd.delete("all")
    atoms = _benzamidinium()
    for n, e, x in atoms:
        cmd.pseudoatom("cplx", name=n, resn="BEN", resi="1", chain="A", elem=e, pos=list(x), hetatm=1)
    inter = _interactions(atoms)
    for r in inter:
        cmd.pseudoatom("cplx", name=r["prot_atom"][:4], resn=r["prot_resn"], resi=r["prot_resi"], chain="A",
                       elem="O" if r["prot_atom"].startswith("O") else "C", pos=list(r["prot_coord"]))
    cmd.pseudoatom("other", pos=[30.0, 0.0, 0.0])
    try:
        out = V.build_scene("cplx", "cplx and resn BEN", inter)
        names = cmd.get_names("objects")
        assert out["labels"] == 6 and cmd.count_atoms("iv_labels") == 6      # Asp189 twice, one label
        assert cmd.count_atoms("iv_dist") == 5                  # polar and ring interactions, not contacts
        assert {"iv_H-bond", "iv_Salt_bridge"} <= set(names)
        assert "other" not in cmd.get_names("objects", enabled_only=1)     # other objects (copies) hidden
        labels = set()
        cmd.iterate("iv_labels", "labels.add(label)", space={"labels": labels})
        assert {"Asp189", "Ser190", "Gly219", "Phe41"} <= labels
        assert cmd.get("orthoscopic") in ("on", "1", 1)
        # a second call replaces the scene instead of piling up objects
        V.build_scene("cplx", "cplx and resn BEN", inter)
        assert cmd.count_atoms("iv_labels") == 6
    finally:
        cmd.delete("all")


def test_residue_key_from_the_combined_field():
    assert V._residue_key({"prot_res": "ASP A189"}) == ("A", "189", "ASP")
    assert V._residue_key({"prot_res": "LYS B-5"}) == ("B", "-5", "LYS")
    assert V._residue_key({"prot_res": "HOH 301"}) == ("", "301", "HOH")
    assert V._residue_key({"prot_chain": "C", "prot_resi": "12", "prot_resn": "SER"}) == ("C", "12", "SER")


def test_2d_map_leaves_rdkit_depiction_settings_alone(tmp_path):
    from rdkit.Chem import rdDepictor

    before = rdDepictor.GetPreferCoordGen()
    atoms = _benzamidinium()
    V.draw_2d_map(str(tmp_path / "m.png"), _interactions(atoms), atoms, fetch_ccd=False)
    assert rdDepictor.GetPreferCoordGen() == before
