import math
import os

import pytest

Chem = pytest.importorskip("rdkit.Chem")
AllChem = pytest.importorskip("rdkit.Chem.AllChem")

from leonardo import ligand_sketcher_runtime as sk  # noqa: E402


def _canon(mol):
    return Chem.MolToSmiles(Chem.Mol(mol))


def _smi(smiles):
    return Chem.CanonSmiles(smiles)


def _mol2d(smiles):
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)
    return m


def _dist(mol, i, j):
    (x1, y1), (x2, y2) = sk.atom_xy(mol, i), sk.atom_xy(mol, j)
    return math.hypot(x2 - x1, y2 - y1)


def test_chain_growth_places_zigzag_at_bond_length():
    mol, last = sk.add_chain_bond(None, (0.0, 0.0))
    mol, last = sk.add_atom(mol, last, "C")
    mol, last = sk.add_atom(mol, last, "C")
    assert _canon(mol) == "CCCC"
    for bd in mol.GetBonds():
        assert _dist(mol, bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()) == pytest.approx(sk.BOND_LEN, abs=1e-6)
    # zigzag: atoms 0 and 2 are 120 degrees apart around atom 1, and 0/3 are trans (far apart)
    assert _dist(mol, 0, 2) == pytest.approx(sk.BOND_LEN * math.sqrt(3.0), abs=1e-3)
    assert _dist(mol, 0, 3) > 3.5


def test_new_atom_opposite_triple_bond_is_linear():
    mol, last = sk.add_chain_bond(None, (0.0, 0.0), order="triple")
    x, y = sk.new_atom_position(mol, last)
    x0, y0 = sk.atom_xy(mol, 0)
    assert math.hypot(x - x0, y - y0) == pytest.approx(2.0 * sk.BOND_LEN, abs=1e-6)


def test_attach_benzene_to_methyl_gives_toluene():
    mol, _ = sk.add_atom(None, None, "C", xy=(0.0, 0.0))
    out = sk.attach_ring(mol, 0, "benzene")
    assert _canon(out) == _smi("Cc1ccccc1")


def test_fuse_benzene_on_any_benzene_bond_gives_naphthalene():
    base = _mol2d("c1ccccc1")
    for bd in base.GetBonds():
        out = sk.fuse_ring(base, bd.GetBeginAtomIdx(), bd.GetEndAtomIdx(), "benzene")
        assert _canon(out) == _smi("c1ccc2ccccc2c1")


def test_fuse_cyclohexane_on_benzene_gives_tetralin_and_ring_is_regular():
    base = _mol2d("c1ccccc1")
    out = sk.fuse_ring(base, 0, 1, "ring6")
    assert _canon(out) == _smi("c1ccc2c(c1)CCCC2")
    for bd in out.GetBonds():
        assert _dist(out, bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()) == pytest.approx(sk.BOND_LEN, abs=0.05)


def test_fused_ring_goes_to_the_empty_side():
    base = _mol2d("c1ccccc1")
    out = sk.fuse_ring(base, 0, 1, "ring5")
    cx = sum(sk.atom_xy(base, i)[0] for i in range(6)) / 6.0
    cy = sum(sk.atom_xy(base, i)[1] for i in range(6)) / 6.0
    for idx in range(6, int(out.GetNumAtoms())):
        x, y = sk.atom_xy(out, idx)
        assert math.hypot(x - cx, y - cy) > 1.6


def test_spiro_ring_shares_the_anchor_atom():
    base = _mol2d("C1CCCCC1")
    out = sk.attach_ring(base, 0, "ring5", spiro=True)
    assert _canon(out) == _smi("C1CCC2(C1)CCCCC2")


def test_place_free_ring_on_empty_canvas():
    out = sk.place_ring(None, (1.0, 2.0), "benzene")
    assert _canon(out) == "c1ccccc1"


def test_cycle_bond_order_single_double_triple_single():
    mol, _ = sk.add_chain_bond(None, (0.0, 0.0))
    seen = []
    for _ in range(3):
        mol = sk.cycle_bond_order(mol, 0, 1)
        seen.append(_canon(mol))
    assert seen == ["C=C", "C#C", "CC"]


def test_cycle_bond_order_skips_invalid_triple():
    base = _mol2d("CC(C)C")
    out = sk.cycle_bond_order(base, 0, 1)
    assert _canon(out) == _smi("C=C(C)C")
    out = sk.cycle_bond_order(out, 0, 1)
    assert _canon(out) == _smi("CC(C)C")


def test_set_element_and_charge_make_pyridinium():
    base = _mol2d("c1ccccc1")
    out = sk.set_element(base, 0, "N")
    out = sk.change_charge(out, 0, +1)
    assert _canon(out) == _smi("c1cc[nH+]cc1")


def test_invalid_valence_raises_friendly_error():
    base = _mol2d("CC(C)(C)C")
    with pytest.raises(sk.SketchError, match="valence"):
        sk.add_atom(base, 1, "C")


def test_delete_substituent_then_grow_on_aromatic_ring():
    base = _mol2d("Cc1ccccc1")
    out = sk.delete_atoms(base, [0])
    assert _canon(out) == "c1ccccc1"
    out, _ = sk.add_atom(out, 0, "O")
    assert _canon(out) == _smi("Oc1ccccc1")


def test_delete_ring_bond_opens_aromatic_ring():
    base = _mol2d("c1ccccc1")
    out = sk.delete_bond(base, 0, 1)
    assert Chem.MolFromSmiles(_canon(out)) is not None
    assert out.GetRingInfo().NumRings() == 0


def test_attach_fragments_orient_away_from_the_molecule():
    base = _mol2d("c1ccccc1")
    out, anchor = sk.attach_fragment(base, 0, "[*]C(F)(F)F")
    assert _canon(out) == _smi("FC(F)(F)c1ccccc1")
    assert _dist(out, 0, anchor) == pytest.approx(sk.BOND_LEN, abs=0.05)
    out, _ = sk.attach_fragment(out, 3, "[*][N+](=O)[O-]")
    assert _canon(out) == _smi("O=[N+]([O-])c1ccc(C(F)(F)F)cc1")
    out, _ = sk.attach_fragment(out, 1, "[*]c1ccncc1")
    assert Chem.MolFromSmiles(_canon(out)) is not None
    pos = sk.all_positions(out)
    for i in pos:
        for j in pos:
            if i < j:
                assert _dist(out, i, j) > 0.5


def test_wedge_defines_stereocenter_and_second_click_flips_it():
    base = _mol2d("CC(N)C(=O)O")
    assert sk.stereo_labels(base) == {1: "?"}
    out = sk.set_wedge(base, 1, 2, "wedge")
    first = sk.stereo_labels(out)[1]
    assert first in ("R", "S")
    flipped = sk.set_wedge(out, 1, 2, "hash")
    assert sk.stereo_labels(flipped)[1] == ("S" if first == "R" else "R")
    inverted = sk.invert_center(out, 1)
    assert sk.stereo_labels(inverted)[1] == ("S" if first == "R" else "R")


def test_wedges_are_redrawn_from_chiral_tags_after_moving_atoms():
    base = _mol2d("C[C@H](N)C(=O)O")
    wedges = sk.display_wedges(base)
    assert wedges and all(b == 1 for b, _e in wedges)
    sk.move_atoms(base, [2], 0.4, -0.3)
    assert Chem.MolToSmiles(base) == Chem.CanonSmiles("C[C@H](N)C(=O)O")


def test_edit_keeps_other_stereocenters():
    base = _mol2d("C[C@H](N)C(=O)O")
    out, _ = sk.add_atom(base, 0, "C")
    assert Chem.MolToSmiles(out) == Chem.CanonSmiles("CC[C@H](N)C(=O)O")


def test_ensure_sketch_coords_flattens_3d_conformers():
    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=7)
    m = Chem.RemoveHs(m)
    assert sk.conformer_is_3d(m)
    sk.ensure_sketch_coords(m)
    assert not sk.conformer_is_3d(m)


def test_strip_hydrogens_drops_stereo_defining_imine_h():
    m = Chem.AddHs(Chem.MolFromSmiles("N=C(N)c1ccccc1"))
    Chem.AssignStereochemistry(m, force=True)
    out = sk.strip_hydrogens(m)
    assert all(a.GetAtomicNum() != 1 for a in out.GetAtoms())


def test_molecule_summary_reports_descriptors_and_flags():
    info = sk.molecule_summary(Chem.MolFromSmiles("CC(N)C(=O)O.[Na+]"))
    assert info["formula"] == "C3H7NNaO2+"
    assert info["fragments"] == 2
    assert info["stereo_undefined"] == 1
    assert info["lipinski_violations"] == 0


def test_render_model_hides_skeletal_carbons_and_labels_heteroatoms():
    data = sk.build_render_model(_mol2d("CC(=O)[O-]"))
    assert set(data["labels"]) == {2, 3}
    assert data["labels"][3] == ("O", 0, -1)
    orders = sorted(b["order"] for b in data["bonds"])
    assert orders == ["double", "single", "single"]


def test_render_model_kekulizes_aromatic_rings_with_ring_centers():
    data = sk.build_render_model(_mol2d("c1ccccc1"))
    orders = [b["order"] for b in data["bonds"]]
    assert orders.count("double") == 3
    assert all(b["ring_center"] is not None for b in data["bonds"])


# ---------------------------------------------------------------- widget


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = sk.QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_widget_click_sequence_draws_and_undo_restores_coordinates(qapp):
    w = sk.LigandSketcherWidget()
    w.resize(1100, 720)
    changes = []
    w.on_molecule_changed = lambda m: changes.append(Chem.MolToSmiles(m) if m is not None else "")
    w._set_tool("benzene")
    w.canvas_click(None, None, (0.0, 0.0))
    assert Chem.MolToSmiles(w.mol) == "c1ccccc1"
    before = sk.all_positions(w.mol)
    w._set_tool("single")
    w.canvas_click(0, None, (0.0, 0.0))
    w.set_element_tool("O")
    w.canvas_click(6, None, (0.0, 0.0))
    assert Chem.MolToSmiles(w.mol) == Chem.CanonSmiles("Oc1ccccc1")
    assert changes[-1] == Chem.CanonSmiles("Oc1ccccc1")
    w.undo()
    w.undo()
    assert Chem.MolToSmiles(w.mol) == "c1ccccc1"
    assert sk.all_positions(w.mol) == before
    w.redo()
    assert Chem.MolToSmiles(w.mol) == "Cc1ccccc1"


def test_widget_rejected_edit_shows_error_without_changing_molecule(qapp):
    w = sk.LigandSketcherWidget()
    w.set_molecule(Chem.MolFromSmiles("CC(C)(C)C"))
    w._set_tool("single")
    w.canvas_click(1, None, (0.0, 0.0))
    assert Chem.MolToSmiles(w.mol) == "CC(C)(C)C"
    assert "valence" in w.lbl_status.text()


def test_widget_keeps_2d_layout_when_host_returns_3d_version_of_same_graph(qapp):
    w = sk.LigandSketcherWidget()
    w.set_molecule(Chem.MolFromSmiles("CCO"))
    layout = sk.all_positions(w.mol)
    m3d = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m3d, randomSeed=3)
    w.set_molecule(m3d)
    assert sk.all_positions(w.mol) == layout


def test_widget_hotkeys_change_hovered_atom_and_delete(qapp):
    QtCore, QtGui = sk.QtCore, sk.QtGui
    w = sk.LigandSketcherWidget()
    w.set_molecule(Chem.MolFromSmiles("CCC"))
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_N, QtCore.Qt.NoModifier, "n")
    assert w.canvas_key(ev, 0, None)
    assert Chem.MolToSmiles(w.mol) == "CCN"
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_2, QtCore.Qt.NoModifier, "2")
    w.canvas_key(ev, None, (1, 2))
    assert Chem.MolToSmiles(w.mol) == "C=CN"
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete, QtCore.Qt.NoModifier, "")
    w.canvas_key(ev, 2, None)
    assert Chem.MolToSmiles(w.mol) == "CN"


def test_widget_return_in_smiles_field_loads_without_building(qapp):
    QtCore, QtGui = sk.QtCore, sk.QtGui
    w = sk.LigandSketcherWidget()
    builds = []
    w.on_build_3d = lambda: builds.append(1)
    assert not w.btn_build.autoDefault()
    w.edit_smiles.setText("CCO")
    w._on_smiles_entered()
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Return, QtCore.Qt.NoModifier)
    w.keyPressEvent(ev)
    assert ev.isAccepted()
    assert Chem.MolToSmiles(w.mol) == "CCO"
    assert builds == []


def test_widget_refuses_to_build_disconnected_fragments(qapp):
    w = sk.LigandSketcherWidget()
    builds = []
    w.on_build_3d = lambda: builds.append(1)
    w.set_molecule(Chem.MolFromSmiles("CCO.CN"))
    w._on_build()
    assert builds == []
    assert "fragments" in w.lbl_status.text()
    w.set_molecule(Chem.MolFromSmiles("CCO"))
    w._on_build()
    assert builds == [1]


def test_widget_hydrogen_tool_never_puts_h_atoms_in_the_sketch(qapp):
    w = sk.LigandSketcherWidget()
    w.set_molecule(Chem.MolFromSmiles("CCO"))
    w.set_element_tool("H")
    w.canvas_click(2, None, (0.0, 0.0))
    w.canvas_click(None, None, (5.0, 5.0))
    assert Chem.MolToSmiles(w.mol) == "CCO"
