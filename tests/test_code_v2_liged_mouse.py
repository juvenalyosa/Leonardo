import code_v2
import pytest


class _Spin:
    def __init__(self, value=0):
        self._value = int(value)

    def value(self):
        return int(self._value)

    def setValue(self, value):
        self._value = int(value)


class _Combo:
    def __init__(self, data):
        self._data = data
        self._items = [data]
        self._current = 0

    def currentData(self):
        try:
            return self._items[self._current]
        except Exception:
            return self._data

    def currentText(self):
        return str(self.currentData())

    def setCurrentText(self, value):
        self._data = value
        try:
            idx = self._items.index(value)
        except ValueError:
            self._items.append(value)
            idx = len(self._items) - 1
        self._current = idx

    def addItem(self, label, data=None):
        self._items.append(self._data if data is None else data)

    def count(self):
        return len(self._items)

    def itemData(self, idx):
        return self._items[int(idx)]

    def setCurrentIndex(self, idx):
        self._current = int(idx)


class _Mol:
    def __init__(self, n_atoms):
        self._n_atoms = int(n_atoms)

    def GetNumAtoms(self):
        return int(self._n_atoms)


def _build_dummy_dialog():
    class DummyDialog:
        pass

    for name in (
        "_liged_current_drawer_tool",
        "_liged_prepare_edit_mol",
        "_liged_current_mouse_mode",
        "_liged_mouse_mode_hint",
        "_liged_selected_element_label",
        "_liged_selected_element",
        "_liged_selected_bond_key",
        "_liged_selected_bond_label",
        "_liged_selected_bond_type",
        "_liged_selected_fragment_label",
        "_liged_clear_aromatic_annotations",
        "_liged_prepare_rw_base_for_edit",
        "_liged_finalize_rw_edit_mol",
        "_liged_relax_all_hydrogen_overrides",
        "_liged_refresh_atom_palette_buttons",
        "_liged_set_mouse_mode",
        "_on_liged_choose_element",
        "_liged_select_fragment_by_smiles",
        "_on_liged_pick_quick_fragment",
        "_liged_set_bond_from_key",
        "_liged_sync_tool_popup_from_state",
        "_liged_mouse_tool_summary",
        "_liged_refresh_mouse_guidance",
        "_liged_safe_atom_count",
        "_liged_normalize_index_state",
        "_liged_refresh_canvas_selection",
        "_on_liged_canvas_atom_click",
        "_on_liged_canvas_bond_click",
        "_on_liged_canvas_atom_extend",
        "_on_liged_canvas_hydrogen_click",
        "_liged_set_atom_model_xy",
        "_liged_adjust_hydrogen_on_atom",
        "_liged_relax_hydrogen_overrides",
        "_liged_validate_atom_index",
        "_ligand_editor_apply_rw_edit",
        "_liged_attach_atom_at_index",
        "_liged_replace_atom_at_index",
        "_liged_delete_atom_at_index",
        "_liged_add_bond_between",
        "_liged_set_bond_between",
        "_liged_delete_bond_between",
        "_liged_attach_fragment_at_index",
        "_ligand_editor_get_current_mol",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return DummyDialog()


def test_canvas_select_shift_click_sets_atom_b():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("select")
    dlg.spin_liged_atom_a = _Spin(0)
    dlg.spin_liged_atom_b = _Spin(0)
    dlg._liged_mouse_pending_atom = None
    dlg._ligand_editor_get_current_mol = lambda: _Mol(5)
    dlg._ligand_editor_set_status = lambda txt: setattr(dlg, "_status", str(txt))

    dlg._on_liged_canvas_atom_click(3, modifiers=code_v2._qt_int(code_v2.QtCore.Qt.ShiftModifier))

    assert dlg.spin_liged_atom_b.value() == 3
    assert "Shift-click sets Atom B" in dlg._status


def test_canvas_add_bond_uses_two_click_sequence():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("add_bond")
    dlg.cmb_liged_bond = _Combo("single")
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg.spin_liged_atom_a = _Spin(0)
    dlg.spin_liged_atom_b = _Spin(0)
    dlg._liged_mouse_pending_atom = None
    dlg._ligand_editor_get_current_mol = lambda: _Mol(4)
    dlg._ligand_editor_set_status = lambda txt: setattr(dlg, "_status", str(txt))
    calls = []
    dlg._liged_add_bond_between = lambda a, b: calls.append((int(a), int(b)))

    dlg._on_liged_canvas_atom_click(1)
    assert dlg._liged_mouse_pending_atom == 1
    dlg._on_liged_canvas_atom_click(3)

    assert dlg._liged_mouse_pending_atom is None
    assert calls == [(1, 3)]


def test_canvas_attach_atom_selects_new_atom_after_growth():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("attach_atom")
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_bond = _Combo("single")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg.spin_liged_atom_a = _Spin(0)
    dlg.spin_liged_atom_b = _Spin(0)
    dlg._liged_mouse_pending_atom = None
    dlg._ligand_editor_get_current_mol = lambda: _Mol(3)
    dlg._ligand_editor_set_status = lambda txt: setattr(dlg, "_status", str(txt))
    calls = []
    dlg._liged_attach_atom_at_index = lambda idx: calls.append(int(idx))

    dlg._on_liged_canvas_atom_click(1)

    assert calls == [1]


def test_canvas_bond_click_sets_existing_bond_to_selected_order():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("set_bond")
    dlg.cmb_liged_bond = _Combo("triple")
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg.spin_liged_atom_a = _Spin(0)
    dlg.spin_liged_atom_b = _Spin(0)
    dlg._liged_mouse_pending_atom = 9
    dlg._ligand_editor_set_status = lambda txt: setattr(dlg, "_status", str(txt))
    calls = []
    dlg._liged_set_bond_between = lambda a, b: calls.append((int(a), int(b)))

    dlg._on_liged_canvas_bond_click(2, 5)

    assert dlg.spin_liged_atom_a.value() == 2
    assert dlg.spin_liged_atom_b.value() == 5
    assert dlg._liged_mouse_pending_atom is None
    assert calls == [(2, 5)]


def test_canvas_bond_click_in_add_mode_upgrades_existing_bond():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("add_bond")
    dlg.cmb_liged_bond = _Combo("double")
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg.spin_liged_atom_a = _Spin(0)
    dlg.spin_liged_atom_b = _Spin(0)
    dlg._liged_mouse_pending_atom = None
    dlg._ligand_editor_set_status = lambda txt: setattr(dlg, "_status", str(txt))
    calls = []
    dlg._liged_set_bond_between = lambda a, b: calls.append((int(a), int(b)))

    dlg._on_liged_canvas_bond_click(1, 4)

    assert calls == [(1, 4)]


def test_prepare_edit_mol_strips_explicit_hydrogens_from_editor_graph():
    if not code_v2.HAS_RDKIT:
        pytest.skip("RDKit not available")

    dlg = _build_dummy_dialog()
    base = code_v2.Chem.MolFromSmiles("CCO")
    with_h = code_v2.Chem.AddHs(base)

    out = dlg._liged_prepare_edit_mol(with_h)

    assert out is not None
    assert int(with_h.GetNumAtoms()) > int(out.GetNumAtoms())
    assert int(out.GetNumAtoms()) == int(base.GetNumAtoms())


def test_safe_atom_count_uses_heavy_atom_editor_graph_when_internal_mol_has_explicit_hydrogens():
    if not code_v2.HAS_RDKIT:
        pytest.skip("RDKit not available")

    dlg = _build_dummy_dialog()
    dlg._ligand_editor_mol = code_v2.Chem.AddHs(code_v2.Chem.MolFromSmiles("CCO"))
    dlg.edit_liged_smiles = None

    assert dlg._liged_safe_atom_count() == 3


def test_current_drawer_tool_defaults_to_web_drawer_not_bkchem():
    dlg = _build_dummy_dialog()

    assert dlg._liged_current_drawer_tool() == "ketcher"


def test_current_drawer_tool_accepts_web_drawer_keys_only():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_drawer_tool = _Combo("local_jsme")
    assert dlg._liged_current_drawer_tool() == "local_jsme"
    dlg.cmb_liged_drawer_tool = _Combo("jsme")
    assert dlg._liged_current_drawer_tool() == "jsme"


def test_normalize_index_state_clamps_stale_spin_values_and_clears_pending():
    dlg = _build_dummy_dialog()
    dlg.spin_liged_atom_a = _Spin(69)
    dlg.spin_liged_atom_b = _Spin(44)
    dlg._liged_mouse_pending_atom = 69

    dlg._liged_normalize_index_state(nat=36)

    assert dlg.spin_liged_atom_a.value() == 35
    assert dlg.spin_liged_atom_b.value() == 35
    assert dlg._liged_mouse_pending_atom is None


def test_canvas_add_bond_ignores_stale_pending_index_after_molecule_shrinks():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("add_bond")
    dlg.cmb_liged_bond = _Combo("single")
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg.spin_liged_atom_a = _Spin(69)
    dlg.spin_liged_atom_b = _Spin(12)
    dlg._liged_mouse_pending_atom = 69
    dlg._ligand_editor_get_current_mol = lambda: _Mol(36)
    dlg._ligand_editor_set_status = lambda txt: setattr(dlg, "_status", str(txt))
    calls = []
    dlg._liged_add_bond_between = lambda a, b: calls.append((int(a), int(b)))

    dlg._on_liged_canvas_atom_click(5)

    assert calls == []
    assert dlg._liged_mouse_pending_atom == 5
    assert dlg.spin_liged_atom_a.value() == 5


def test_mouse_tool_summary_includes_selected_edit_parameters():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("attach_atom")
    dlg.cmb_liged_element = _Combo("Cl")
    dlg.cmb_liged_bond = _Combo("double")
    dlg.cmb_liged_fragment = _Combo("Phenyl")

    title, subtitle = dlg._liged_mouse_tool_summary()

    assert title == "Add Atom: Cl"
    assert "attach Cl" in subtitle
    assert "double bond" in subtitle


def test_canvas_same_second_bond_atom_keeps_pending_and_sets_status():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("add_bond")
    dlg.cmb_liged_bond = _Combo("single")
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg.spin_liged_atom_a = _Spin(0)
    dlg.spin_liged_atom_b = _Spin(0)
    dlg._liged_mouse_pending_atom = 2
    dlg._ligand_editor_get_current_mol = lambda: _Mol(6)
    dlg._ligand_editor_set_status = lambda txt: setattr(dlg, "_status", str(txt))
    calls = []
    dlg._liged_add_bond_between = lambda a, b: calls.append((int(a), int(b)))

    dlg._on_liged_canvas_atom_click(2)

    assert dlg._liged_mouse_pending_atom == 2
    assert calls == []
    assert "different second atom" in dlg._status


def test_mouse_tool_summary_for_hydrogen_mode_mentions_hydrogen_labels():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("attach_atom")
    dlg.cmb_liged_element = _Combo("H")
    dlg.cmb_liged_bond = _Combo("single")
    dlg.cmb_liged_fragment = _Combo("Phenyl")

    title, subtitle = dlg._liged_mouse_tool_summary()

    assert title == "Add Hydrogen"
    assert "hydrogen" in subtitle.lower()


def test_canvas_hydrogen_click_deletes_one_hydrogen_in_delete_mode():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("delete_atom")
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_bond = _Combo("single")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg.spin_liged_atom_a = _Spin(0)
    dlg.spin_liged_atom_b = _Spin(0)
    dlg._liged_mouse_pending_atom = None
    dlg._ligand_editor_set_status = lambda txt: setattr(dlg, "_status", str(txt))
    calls = []
    dlg._liged_adjust_hydrogen_on_atom = lambda idx, delta: calls.append((int(idx), int(delta)))

    dlg._on_liged_canvas_hydrogen_click(4, 2)

    assert calls == [(4, -1)]


def test_choose_element_switches_to_attach_atom_mode():
    dlg = _build_dummy_dialog()
    dlg.cmb_liged_mouse_mode = _Combo("select")
    dlg.cmb_liged_mouse_mode._items = [
        "select",
        "attach_atom",
        "replace_atom",
        "delete_atom",
        "add_bond",
        "set_bond",
        "delete_bond",
        "attach_fragment",
    ]
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_bond = _Combo("single")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg._ligand_editor_set_status = lambda txt: setattr(dlg, "_status", str(txt))

    dlg._on_liged_choose_element("O")

    assert dlg.cmb_liged_element.currentText() == "O"
    assert dlg.cmb_liged_mouse_mode.currentData() == "attach_atom"


def test_canvas_atom_extend_for_drag_attach_passes_target_coordinates():
    dlg = _build_dummy_dialog()
    calls = []
    dlg._liged_attach_atom_at_index = lambda idx, target_model_xy=None, via_drag=False: calls.append(
        (int(idx), tuple(target_model_xy or ()), bool(via_drag))
    )

    dlg._on_liged_canvas_atom_extend(2, 1.25, -0.5)

    assert calls == [(2, (1.25, -0.5), True)]


def test_prepare_rw_base_for_edit_kekulizes_aromatic_ring():
    if not code_v2.HAS_RDKIT:
        pytest.skip("RDKit not available")

    dlg = _build_dummy_dialog()
    mol = code_v2.Chem.MolFromSmiles("c1ccccc1")

    out = dlg._liged_prepare_rw_base_for_edit(mol)

    assert out is not None
    assert not any(bool(b.GetIsAromatic()) for b in out.GetBonds())
    assert any(b.GetBondType() == code_v2.Chem.BondType.DOUBLE for b in out.GetBonds())


def test_finalize_rw_edit_mol_recovers_after_breaking_aromatic_ring():
    if not code_v2.HAS_RDKIT:
        pytest.skip("RDKit not available")

    dlg = _build_dummy_dialog()
    base = code_v2.Chem.MolFromSmiles("c1ccccc1")
    rw = code_v2.Chem.RWMol(code_v2.Chem.Mol(base))
    rw.RemoveAtom(0)

    out = dlg._liged_finalize_rw_edit_mol(rw.GetMol())

    assert out is not None
    code_v2.Chem.SanitizeMol(out)
    assert int(out.GetNumAtoms()) == 5
    assert not any(bool(a.GetIsAromatic()) for a in out.GetAtoms())


def test_relax_all_hydrogen_overrides_clears_manual_h_flags():
    if not code_v2.HAS_RDKIT:
        pytest.skip("RDKit not available")

    dlg = _build_dummy_dialog()
    mol = code_v2.Chem.MolFromSmiles("Cc1ccccc1")
    atom = mol.GetAtomWithIdx(0)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(3)
    atom.UpdatePropertyCache(strict=False)

    changed = dlg._liged_relax_all_hydrogen_overrides(mol)

    assert changed is True
    assert atom.GetNoImplicit() is False
    assert int(atom.GetNumExplicitHs()) == 0


def test_delete_substituent_then_attach_atom_on_aromatic_ring_succeeds():
    if not code_v2.HAS_RDKIT:
        pytest.skip("RDKit not available")

    dlg = _build_dummy_dialog()
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_bond = _Combo("single")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg.spin_liged_atom_a = _Spin(0)
    dlg.spin_liged_atom_b = _Spin(0)
    dlg._liged_mouse_pending_atom = None
    dlg._ligand_editor_mol = code_v2.Chem.MolFromSmiles("Cc1ccccc1")
    dlg.edit_liged_smiles = None
    dlg._ligand_editor_history = []
    dlg._ligand_editor_push_history = lambda: None
    dlg._liged_refresh_canvas_selection = lambda *args, **kwargs: None
    dlg._liged_refresh_mouse_guidance = lambda *args, **kwargs: None
    dlg._ligand_editor_set_status = lambda *args, **kwargs: None
    dlg._ligand_editor_set_mol = lambda mol, source_label="": setattr(dlg, "_ligand_editor_mol", code_v2.Chem.Mol(mol)) or True

    assert dlg._liged_delete_atom_at_index(0) is True
    assert dlg._liged_attach_atom_at_index(0) is True

    out = code_v2.Chem.Mol(dlg._ligand_editor_mol)
    code_v2.Chem.SanitizeMol(out)
    assert code_v2.Chem.MolToSmiles(out, isomericSmiles=True) == "Cc1ccccc1"


def test_attach_atom_succeeds_even_if_starting_atom_has_stale_manual_hydrogen_override():
    if not code_v2.HAS_RDKIT:
        pytest.skip("RDKit not available")

    dlg = _build_dummy_dialog()
    dlg.cmb_liged_element = _Combo("C")
    dlg.cmb_liged_bond = _Combo("single")
    dlg.cmb_liged_fragment = _Combo("Phenyl")
    dlg.spin_liged_atom_a = _Spin(0)
    dlg.spin_liged_atom_b = _Spin(0)
    dlg._liged_mouse_pending_atom = None
    mol = code_v2.Chem.MolFromSmiles("c1ccccc1")
    atom0 = mol.GetAtomWithIdx(0)
    atom0.SetNoImplicit(True)
    atom0.SetNumExplicitHs(1)
    atom0.UpdatePropertyCache(strict=False)
    dlg._ligand_editor_mol = mol
    dlg.edit_liged_smiles = None
    dlg._ligand_editor_history = []
    dlg._ligand_editor_push_history = lambda: None
    dlg._liged_refresh_canvas_selection = lambda *args, **kwargs: None
    dlg._liged_refresh_mouse_guidance = lambda *args, **kwargs: None
    dlg._ligand_editor_set_status = lambda *args, **kwargs: None
    dlg._ligand_editor_set_mol = lambda mol, source_label="": setattr(dlg, "_ligand_editor_mol", code_v2.Chem.Mol(mol)) or True

    assert dlg._liged_attach_atom_at_index(0) is True

    out = code_v2.Chem.Mol(dlg._ligand_editor_mol)
    code_v2.Chem.SanitizeMol(out)
    assert int(out.GetNumAtoms()) == 7
