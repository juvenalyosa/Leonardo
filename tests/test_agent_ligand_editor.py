"""Assistant tools for the 2D Ligand Editor (sketch read/edit and 3D build)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
Chem = pytest.importorskip("rdkit.Chem")

from leonardo.agent.registry import ToolError  # noqa: E402
from leonardo.agent.tools import AgentContext, build_registry  # noqa: E402
from leonardo.agent.tools.ligand_editor import apply_operations, sketch_state  # noqa: E402
from leonardo.ligand_sketcher_runtime import ensure_sketch_coords  # noqa: E402

BEN = os.path.join(os.path.dirname(__file__), "..", "examples", "3ptb_ben", "BEN.sdf")


def _sketch(smiles):
    return ensure_sketch_coords(Chem.MolFromSmiles(smiles))


def test_sketch_state_numbers_atoms_from_one_with_neighbours_and_stereo():
    state = sketch_state(_sketch("CC(N)C(=O)O"))
    assert state["smiles"] == "CC(N)C(=O)O"
    assert state["atom_list"][1] == {"n": 2, "el": "C", "h": 1, "nbrs": [1, 3, 4], "stereo": "?"}
    assert state["non_single_bonds"] == [[4, 5, "double"]]
    assert state["stereo_undefined"] == 1


def test_operations_build_a_para_substituted_analog():
    mol = _sketch("c1ccccc1C(=N)N")       # benzamidine; ring atom 3 is para to ring atom 6
    new, changes = apply_operations(mol, [
        {"op": "attach_group", "atom": 3, "group": "CF3"},
        {"op": "set_element", "atom": 1, "element": "N"},
    ])
    assert Chem.MolToSmiles(new) == Chem.CanonSmiles("N=C(N)c1ccc(C(F)(F)F)cn1")
    assert changes == ["attached CF3 to atom 3", "atom 1 -> N"]


def test_operations_set_r_and_s_and_groups_by_name():
    mol = _sketch("CC(N)C(=O)O")
    r, _ = apply_operations(mol, [{"op": "set_stereo", "atom": 2, "config": "R"}])
    s, _ = apply_operations(mol, [{"op": "set_stereo", "atom": 2, "config": "S"}])
    assert Chem.MolToSmiles(r) == Chem.CanonSmiles("C[C@@H](N)C(=O)O")
    assert Chem.MolToSmiles(s) == Chem.CanonSmiles("C[C@H](N)C(=O)O")
    morph, _ = apply_operations(_sketch("c1ccccc1"), [{"op": "attach_group", "atom": 1, "group": "morpholino"}])
    assert Chem.MolToSmiles(morph) == Chem.CanonSmiles("c1ccc(N2CCOCC2)cc1")


def test_operations_are_all_or_nothing_and_name_the_failing_step():
    mol = _sketch("CCO")
    with pytest.raises(ToolError, match="operation 2 \\(add_atom\\).*valence"):
        apply_operations(mol, [
            {"op": "set_element", "atom": 3, "element": "N"},
            {"op": "add_atom", "atom": 2, "element": "C", "order": "triple"},
        ])
    with pytest.raises(ToolError, match="atom 9 does not exist"):
        apply_operations(mol, [{"op": "set_charge", "atom": 9, "charge": 1}])
    with pytest.raises(ToolError, match="Unknown group"):
        apply_operations(mol, [{"op": "attach_group", "atom": 1, "group": "unobtainium"}])


def test_ring_operations_and_charges():
    mol = _sketch("c1ccccc1")
    new, _ = apply_operations(mol, [
        {"op": "fuse_ring", "atoms": [1, 2], "ring": "benzene"},
        {"op": "set_element", "atom": 3, "element": "N"},
        {"op": "set_charge", "atom": 3, "charge": 1},
    ])
    assert Chem.MolToSmiles(new) == Chem.CanonSmiles("c1ccc2[nH+]cccc2c1")    # quinolinium


def test_tools_are_registered_with_expected_permissions():
    reg = build_registry(AgentContext())
    perms = {s.name: s.permission for s in reg.specs()}
    assert perms["ligand_sketch_state"] == "read"
    assert perms["open_ligand_editor"] == "view"
    assert perms["edit_ligand_sketch"] == "compute_light"
    assert perms["build_ligand_from_sketch"] == "compute_light"


@pytest.fixture(scope="module")
def dialog():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2

    dlg = code_v2.ProtLigInteractDialog()
    app.processEvents()
    yield dlg
    dlg.close()


def test_assistant_edits_the_visible_sketch_and_builds_the_3d_ligand(dialog):
    ctx = AgentContext(dlg=dialog)
    reg = build_registry(ctx)
    dialog._docking_ligand_file = os.path.abspath(BEN)
    state = reg.call("open_ligand_editor", {"source": "active"}, ctx)
    assert state["smiles"] == "N=C(N)c1ccccc1"
    para = [a["n"] for a in state["atom_list"] if a.get("aromatic") and a["h"] == 1][2]
    out = reg.call("edit_ligand_sketch", {"operations": [{"op": "attach_group", "atom": para, "group": "Cl"}]}, ctx)
    assert out["smiles"] == "N=C(N)c1ccc(Cl)cc1"
    assert Chem.MolToSmiles(dialog._ligand_editor_mol) == out["smiles"]       # host sees the edit
    sk = dialog.liged_sketcher
    sk.undo()
    assert Chem.MolToSmiles(sk.mol) == "N=C(N)c1ccccc1"                     # one undo step
    sk.redo()
    built = reg.call("build_ligand_from_sketch", {"add_hydrogens": True, "relax": True}, ctx)
    assert os.path.isfile(built["ligand_file"])
    assert dialog._docking_ligand_file == built["ligand_file"]
    m = next(iter(Chem.SDMolSupplier(built["ligand_file"], removeHs=False)))
    zs = [m.GetConformer().GetAtomPosition(i).z for i in range(m.GetNumAtoms())]
    assert max(zs) - min(zs) > 0.5                                            # real 3D, not the flat sketch


def test_build_refuses_disconnected_sketch(dialog):
    ctx = AgentContext(dlg=dialog)
    reg = build_registry(ctx)
    reg.call("edit_ligand_sketch", {"start_smiles": "CCO.CN", "operations": []}, ctx)
    with pytest.raises(ToolError, match="fragments"):
        reg.call("build_ligand_from_sketch", {}, ctx)
