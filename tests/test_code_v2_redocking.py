import code_v2


class _Combo:
    def __init__(self, text=""):
        self._text = str(text)
        self._items = []
        if self._text:
            self._items.append(self._text)
        self._signals_blocked = False

    def currentText(self):
        return str(self._text)

    def blockSignals(self, blocked):
        self._signals_blocked = bool(blocked)

    def clear(self):
        self._items = []
        self._text = ""

    def addItems(self, items):
        self._items.extend([str(item) for item in list(items or [])])

    def setCurrentText(self, text):
        self._text = str(text)
        if self._text and self._text not in self._items:
            self._items.append(self._text)


class _Check:
    def __init__(self, checked=False):
        self._checked = bool(checked)

    def isChecked(self):
        return bool(self._checked)


def _build_dummy_dialog():
    class DummyDialog:
        pass

    for name in (
        "_redock_selected_wt_object_for_base_payload",
        "_redock_selected_mutant_object_for_base_payload",
        "_redock_manual_alignment_targets",
        "_refresh_redocking_object_combos",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return DummyDialog()


def test_redock_selected_wt_object_prefers_modeler_output_when_enabled():
    dlg = _build_dummy_dialog()
    dlg.cmb_redock_wt_obj = _Combo("wt_combo")
    dlg.cb_redock_use_modeler_wt = _Check(True)
    dlg._modeler_last_model_object = "wt_model"

    assert dlg._redock_selected_wt_object_for_base_payload() == "wt_model"


def test_redock_manual_alignment_targets_falls_back_to_combo_when_modeler_output_missing():
    dlg = _build_dummy_dialog()
    dlg.cmb_redock_reference_obj = _Combo("ref_complex")
    dlg.cmb_redock_wt_obj = _Combo("wt_combo")
    dlg.cb_redock_use_modeler_wt = _Check(True)
    dlg._modeler_last_model_object = ""
    dlg.cb_redock_run_mutant = _Check(True)
    dlg.cmb_redock_mut_obj = _Combo("mut_obj")
    dlg._redock_modeler_mutation_batch_combo_label = lambda: ""

    payload = dlg._redock_manual_alignment_targets()

    assert payload["reference_obj"] == "ref_complex"
    assert payload["targets"] == [
        ("WT", "wt_combo", "cmb_redock_wt_obj"),
        ("Mutant", "mut_obj", "cmb_redock_mut_obj"),
    ]


def test_redock_refresh_object_combos_preserves_current_user_selection():
    dlg = _build_dummy_dialog()
    dlg.loaded_object = "reference_default"
    dlg._modeler_last_model_object = "mutant_default"
    dlg.cmb_redock_reference_obj = _Combo("reference_keep")
    dlg.cmb_redock_wt_obj = _Combo("wt_keep")
    dlg.cmb_redock_mut_obj = _Combo("mut_keep")
    dlg._redock_polymer_object_names = lambda: [
        "reference_default",
        "reference_keep",
        "wt_keep",
        "mut_keep",
        "mutant_default",
    ]
    dlg._candidate_structure_objects = lambda: []
    dlg._redock_modeler_mutation_batch_combo_label = lambda: ""
    dlg._on_redock_autodetect_reference_ligand = lambda: None

    dlg._refresh_redocking_object_combos()

    assert dlg.cmb_redock_reference_obj.currentText() == "reference_keep"
    assert dlg.cmb_redock_wt_obj.currentText() == "wt_keep"
    assert dlg.cmb_redock_mut_obj.currentText() == "mut_keep"
