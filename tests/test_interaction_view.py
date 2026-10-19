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


def test_interface_callouts_never_overlap_and_stay_in_frame():
    # 14 residues crowded along an interface line, labels as wide as real ones, a frame that is tight for them
    rng = np.random.default_rng(3)
    anchors = [(float(x), float(y)) for x, y in zip(rng.normal(0.0, 1.5, 14), np.linspace(-9.0, 9.0, 14))]
    boxes = [V.pymol_label_box(t, 3.0) for t in ("Arg59", "His102", "Gly31", "Tyr29", "dG2:B", "Asp35",
                                                 "Lys27", "Glu60", "Arg83", "Arg87", "Asn33", "Leu34",
                                                 "Thr42", "Glu76")]
    away = [(6.0 if i % 2 else -6.0, a[1]) for i, a in enumerate(anchors)]
    frame = (20.0, 13.0)
    obstacles = [(float(x), float(y)) for x, y in zip(rng.normal(0.0, 2.0, 200), rng.uniform(-10, 10, 200))]
    spots = V.layout_callouts(anchors, boxes, away, frame, obstacles=obstacles, interactions=[(0.0, 0.0)])
    rects = [(x - b[0] / 2, y - b[1] / 2, x + b[0] / 2, y + b[1] / 2) for (x, y), b in zip(spots, boxes)]
    for i in range(len(rects)):
        assert -frame[0] <= rects[i][0] and rects[i][2] <= frame[0]
        assert -frame[1] <= rects[i][1] and rects[i][3] <= frame[1]
        for j in range(i):
            assert not V._rect_overlap(rects[i], rects[j]), (i, j)


def test_pymol_label_box_matches_measured_labels():
    # ray-traced PyMOL 3.1 labels, font 7, label_size -2, padding 0.2: 'Arg59 A' 3.60 x 1.02 A, 'W' 1.08 A wide
    w, h = V.pymol_label_box("Arg59 A", 2.0)
    assert 3.6 <= w <= 3.95 and 1.0 <= h <= 1.1
    assert 1.05 <= V.pymol_label_box("W", 2.0)[0] <= 1.25


def test_label_size_keeps_pixels_at_any_resolution():
    # a PyMOL label keeps its size in pixels: at half the image height it covers twice the Angstrom
    w1, h1 = V.pymol_label_box("Arg59", 2.0, 1300)
    w2, h2 = V.pymol_label_box("Arg59", 2.0, 650)
    assert abs(w2 / w1 - 2.0) < 1e-9 and abs(h2 / h1 - 2.0) < 1e-9
    for height in (480, 900, 1300):
        size = V.label_size_for(1.5, height)
        assert abs(V.pymol_label_box("W", size, height)[1] - 1.5) < 1e-9


def test_interface_core_picks_the_densest_region():
    # a tight cluster of six residues with many contacts and a far, sparse arm (a large interface)
    cluster = [(("A", str(i)), (float(i), 0.0, 0.0), 3) for i in range(6)]
    arm = [(("B", str(i)), (40.0 + 6.0 * i, 0.0, 0.0), 1) for i in range(8)]
    core = V.interface_core(arm + cluster, radius=12.0, max_residues=10)
    assert set(core) == {k for k, _x, _w in cluster}
    small = [(("A", str(i)), (2.0 * i, 0.0, 0.0), 1) for i in range(5)]
    assert set(V.interface_core(small, radius=16.0)) == {k for k, _x, _w in small}     # small: whole
    assert len(V.interface_core(arm + cluster, radius=200.0, max_residues=4)) == 4
    assert V.interface_core([]) == []


def test_interface_side_items_name_chains_and_kind():
    items = V.interface_side_items({"A": {"chains": ["A"], "kind": "protein", "selection": "protein"},
                                    "B": {"chains": ["B", "C"], "kind": "dna", "selection": "nucleic"}})
    assert [t for t, _rgb in items] == ["Chain A (protein)", "Chains B, C (DNA)"]
    assert items[0][1] == V.SIDE_COLORS["A"][1]


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


def test_bond_orders_by_atom_name_ignore_crystal_contacts(monkeypatch):
    """A non-bonded pair within the distance cutoff (crystal strain) must not become a bond when the PDB
    component's bond table is available: connectivity comes from the atom names."""
    atoms = _benzamidinium()
    xyz = {n: np.array(x) for n, _e, x in atoms}
    # pull N1 next to C3 (1,3 pair): the distance rule alone would bond them
    xyz["N1"] = xyz["C2"] + (xyz["N1"] - xyz["C2"]) * 0.2 + (xyz["C3"] - xyz["C2"]) * 0.75
    squeezed = [(n, e, tuple(xyz[n])) for n, e, _x in atoms]
    bonds = [("N1", "C1", "SING"), ("C1", "N2", "DOUB"), ("C1", "C2", "SING")]
    ring = ["C2", "C3", "C4", "C5", "C6", "C7"]
    bonds += [(ring[i], ring[(i + 1) % 6], "AROM") for i in range(6)]
    comp = {"atoms": [{"name": n, "element": e.upper(), "charge": 0, "xyz": x} for n, e, x in atoms],
            "bonds": bonds}
    monkeypatch.setattr("leonardo.component_library_runtime.ccd_component", lambda *a, **k: comp)
    mol, _names, info = V.ligand_molecule(squeezed, resn="BEN", fetch_ccd=False)
    assert "atom names" in info["bond_orders"]
    assert mol.GetNumBonds() == 9 and sum(1 for b in mol.GetBonds() if b.GetIsAromatic()) == 6


def _pose_like(atoms):
    """A docking pose: same molecule, atoms shuffled, names reduced to the element (repeated)."""
    order = [3, 0, 8, 5, 1, 7, 2, 6, 4]
    return [(atoms[i][1], atoms[i][1], atoms[i][2]) for i in order]


def test_template_gives_bond_orders_for_a_pose_with_repeated_names(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    template = Chem.AddHs(Chem.MolFromSmiles("NC(=N)c1ccccc1"))
    AllChem.EmbedMolecule(template, randomSeed=3)
    sdf = tmp_path / "lig.sdf"
    Chem.MolToMolFile(template, str(sdf))
    pose = _pose_like(_benzamidinium())
    mol, _names, info = V.ligand_molecule(pose, fetch_ccd=False, template=V.load_template(str(sdf)))
    assert info["bond_orders"] == "the ligand file (graph match)"
    assert sum(1 for b in mol.GetBonds() if b.GetIsAromatic()) == 6
    assert sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE) == 1
    # atom order of the pose is kept: atom 0 is still the pose's first atom (C1 of the amidine)
    assert [a.GetSymbol() for a in mol.GetAtoms()] == [e for _n, e, _x in pose]


def test_a_template_of_another_compound_is_ignored():
    from rdkit import Chem

    pose = _pose_like(_benzamidinium())
    mol, _names, info = V.ligand_molecule(pose, fetch_ccd=False, template=Chem.MolFromSmiles("CCO"))
    assert info["bond_orders"] != "the ligand file (graph match)" and mol.GetNumAtoms() == 9


def test_map_anchors_by_coordinate_when_names_repeat(tmp_path):
    atoms = _benzamidinium()
    pose = _pose_like(atoms)
    inter = _interactions(atoms)
    for r in inter:                      # the pose reports bare element names, as docked ligands do
        r["lig_atom"] = r["lig_atom"][0] if r["lig_atom"] != "ring" else "Ring"
    out = V.draw_2d_map(str(tmp_path / "pose.png"), inter, pose, fetch_ccd=False)
    assert out["residues"] == 6 and out["not_placed"] == []


def test_pymol_bonds_with_orders_become_a_template(tmp_path):
    pytest.importorskip("pymol")
    from pymol import cmd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("NC(=N)c1ccccc1"))
    AllChem.EmbedMolecule(m, randomSeed=5)
    path = tmp_path / "lig.mol"
    Chem.MolToMolFile(m, str(path))
    cmd.delete("all")
    try:
        cmd.load(str(path), "ligmol")
        tpl = V.ligand_template_from_pymol("ligmol")
        assert tpl is not None and Chem.MolToSmiles(Chem.RemoveHs(tpl)) == Chem.MolToSmiles(Chem.MolFromSmiles("NC(=N)c1ccccc1"))
        cmd.save(str(tmp_path / "lig.pdb"), "ligmol")
        cmd.delete("all")
        no_conect = [ln for ln in (tmp_path / "lig.pdb").read_text().splitlines() if not ln.startswith("CONECT")]
        (tmp_path / "bare.pdb").write_text("\n".join(no_conect) + "\n")
        cmd.load(str(tmp_path / "bare.pdb"), "ligpdb")
        assert V.ligand_template_from_pymol("ligpdb") is None          # bonds by distance: orders unknown
    finally:
        cmd.delete("all")
