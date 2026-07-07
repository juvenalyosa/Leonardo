import code_v2


class _FakeCmd:
    def __init__(self):
        self.atom_counts = {}
        self.objects = set()
        self.calls = []
        self.centered = ""
        self.zoomed = None

    def count_atoms(self, selection):
        return int(self.atom_counts.get(str(selection), 0))

    def get_names(self, _kind="objects"):
        return list(self.objects)

    def set_color(self, name, rgb):
        self.calls.append(("set_color", str(name), list(rgb)))

    def set(self, setting, value, selection=None):
        self.calls.append(("set", str(setting), value, "" if selection is None else str(selection)))

    def enable(self, obj_name):
        self.calls.append(("enable", str(obj_name)))

    def center(self, selection):
        self.centered = str(selection)

    def zoom(self, selection, buffer=0):
        self.zoomed = (str(selection), buffer)


def _build_dummy_dialog():
    class DummyDialog:
        pass

    for name in (
        "_apply_dash_style",
        "_focus_interaction_by_id",
        "_highlight_interaction_visual",
        "_interaction_focus_selection",
        "_interaction_key",
        "_resolve_2d_map_interaction_id",
        "_restore_interaction_focus_style",
        "_select_table_row_for_interaction",
        "_type_color_name",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return DummyDialog()


def _interaction(itype="Hydrogen Bond", residue="ASN A123", atom="OD1"):
    return {
        "type": itype,
        "prot_res": residue,
        "prot_atom": atom,
        "lig_atom": "N1",
        "distance": 2.8,
        "details": "D...A 2.8A",
    }


def test_2d_map_focus_turns_on_interaction_without_toggling_it_off():
    dlg = _build_dummy_dialog()
    dlg.interactions = [_interaction()]
    dlg._selected_ids = set()
    dlg._focused_interaction_rid = -1
    dlg._settings = {}
    toggled = []

    def _toggle(rid):
        toggled.append(int(rid))
        dlg._selected_ids.add(int(rid))

    dlg._toggle_interaction_by_id = _toggle
    fake_cmd = _FakeCmd()
    fake_cmd.atom_counts.update({"pseudo_p_0": 1, "pseudo_l_0": 1, "intRes_0": 8})

    old_cmd = code_v2.cmd
    code_v2.cmd = fake_cmd
    try:
        assert dlg._focus_interaction_by_id(0, ensure_visible=True, update_table=True, zoom_to=True) is True
    finally:
        code_v2.cmd = old_cmd

    assert toggled == [0]
    assert dlg._selected_ids == {0}
    assert dlg._focused_interaction_rid == 0
    assert fake_cmd.zoomed is not None
    assert "pseudo_p_0" in fake_cmd.zoomed[0]
    assert ("set", "dash_radius", 0.16, "int_dash_0") in fake_cmd.calls


def test_2d_map_focus_does_not_hide_an_already_visible_interaction():
    dlg = _build_dummy_dialog()
    dlg.interactions = [_interaction()]
    dlg._selected_ids = {0}
    dlg._focused_interaction_rid = -1
    dlg._settings = {}
    toggled = []
    dlg._toggle_interaction_by_id = lambda rid: toggled.append(int(rid))
    fake_cmd = _FakeCmd()
    fake_cmd.atom_counts.update({"pseudo_p_0": 1, "pseudo_l_0": 1, "intRes_0": 8})

    old_cmd = code_v2.cmd
    code_v2.cmd = fake_cmd
    try:
        assert dlg._focus_interaction_by_id(0, ensure_visible=True, update_table=False, zoom_to=True) is True
    finally:
        code_v2.cmd = old_cmd

    assert toggled == []
    assert dlg._selected_ids == {0}
    assert dlg._focused_interaction_rid == 0
    assert fake_cmd.centered


def test_stale_2d_map_item_resolves_by_interaction_key_after_recalculation():
    dlg = _build_dummy_dialog()
    original_target = _interaction(itype="Salt Bridge", residue="ASP A45", atom="OD2")
    dlg.interactions = [_interaction(), original_target]
    map_item = {
        "rid": 0,
        "interaction_key": dlg._interaction_key(original_target),
    }

    assert dlg._resolve_2d_map_interaction_id(map_item) == 1
