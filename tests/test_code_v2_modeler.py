import os

import code_v2


class _FakeCmd:
    def __init__(self):
        self.saved = []
        self.fetches = []
        self.objects = {}

    def save(self, path, selection, state=1):
        self.saved.append((str(path), str(selection), int(state)))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n")

    def fetch(self, pdb_id, name=None, type="pdb", async_=0):
        obj_name = str(name or pdb_id)
        self.fetches.append((str(pdb_id), str(obj_name), str(type), int(async_)))
        self.objects[obj_name] = 42

    def count_atoms(self, selection):
        sel = str(selection or "")
        for obj_name, atom_count in list(self.objects.items()):
            if obj_name in sel:
                return int(atom_count)
        return 0

    def delete(self, obj_name):
        self.objects.pop(str(obj_name), None)

    def get_names(self, _kind="objects"):
        return list(self.objects.keys())


def _build_dummy_dialog():
    class DummyDialog:
        pass

    for name in (
        "_modeler_normalize_template_code",
        "_modeler_infer_pdb_id",
        "_modeler_resolve_local_template_object",
        "_modeler_fetch_template_object_from_rcsb",
        "_modeler_stage_local_template_pdb",
        "_modeler_stage_remote_template_pdb",
        "_modeler_stage_local_template_pdbs",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return DummyDialog()


def test_modeler_resolve_local_template_object_prefers_matching_loaded_object():
    dlg = _build_dummy_dialog()
    dlg._candidate_structure_objects = lambda: ["4xyz_model", "3ptb_loaded_chain_a"]
    dlg.pdb_path = ""

    assert dlg._modeler_resolve_local_template_object("3ptbA") == "3ptb_loaded_chain_a"
    assert dlg._modeler_resolve_local_template_object(
        "3ptbA",
        preferred_object="3ptb_preferred",
    ) == "3ptb_preferred"


def test_modeler_stage_local_template_pdbs_writes_expected_template_file(tmp_path):
    dlg = _build_dummy_dialog()
    dlg._candidate_structure_objects = lambda: ["3ptb_loaded_chain_a"]
    dlg._pymol_model_selection = lambda obj, wrap=True: f"({obj})"
    dlg.pdb_path = ""

    fake_cmd = _FakeCmd()
    old_cmd = code_v2.cmd
    code_v2.cmd = fake_cmd
    try:
        payload = dlg._modeler_stage_local_template_pdbs(
            ["3ptbA"],
            str(tmp_path),
            preferred_object="not_matching_object",
        )
    finally:
        code_v2.cmd = old_cmd

    assert payload["ok"] is True
    assert payload["skipped"] == []
    assert int(len(payload["staged"])) == 1
    staged = dict(payload["staged"][0] or {})
    assert str(staged["template_code"]) == "3ptbA"
    assert str(staged["source_object"]) == "3ptb_loaded_chain_a"
    assert str(staged["path"]) == str(tmp_path / "modeling_job" / "3ptb.pdb")
    assert os.path.isfile(staged["path"])
    assert fake_cmd.saved == [
        (str(tmp_path / "modeling_job" / "3ptb.pdb"), "(3ptb_loaded_chain_a)", 1)
    ]


def test_modeler_stage_local_template_pdbs_fetches_remote_cif_when_legacy_pdb_missing(tmp_path):
    dlg = _build_dummy_dialog()
    dlg._candidate_structure_objects = lambda: []
    dlg._pymol_model_selection = lambda obj, wrap=True: f"({obj})"
    dlg._dock_safe_obj_name = lambda text: str(text)
    dlg.pdb_path = ""

    fake_cmd = _FakeCmd()
    old_cmd = code_v2.cmd
    code_v2.cmd = fake_cmd
    try:
        payload = dlg._modeler_stage_local_template_pdbs(
            ["9gbeA"],
            str(tmp_path),
            preferred_object="",
        )
    finally:
        code_v2.cmd = old_cmd

    assert payload["ok"] is True
    assert payload["skipped"] == []
    assert int(len(payload["staged"])) == 1
    staged = dict(payload["staged"][0] or {})
    assert str(staged["template_code"]) == "9gbeA"
    assert str(staged["path"]) == str(tmp_path / "modeling_job" / "9gbe.pdb")
    assert str(staged["fetch_format"]) == "cif"
    assert os.path.isfile(staged["path"])
    assert fake_cmd.fetches[0] == ("9GBE", "_modeler_tpl_9gbe", "cif", 0)
    assert fake_cmd.saved == [
        (str(tmp_path / "modeling_job" / "9gbe.pdb"), "(_modeler_tpl_9gbe) and polymer", 1)
    ]


_RECEPTOR_3PTB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "benchmarks", "redocking", "3ptb_ben", "receptor_3ptb_no_ben_no_waters.pdb")


class _Widget:
    def __init__(self, value=None, data=None):
        self._value, self._data = value, data

    def toPlainText(self):
        return self._value

    def value(self):
        return self._value

    def currentData(self):
        return self._data

    def setPlainText(self, value):
        self._value = value

    def setValue(self, value):
        self._value = value


def _build_modeler_numbering_dialog(full_sequence=""):
    from Bio.PDB import PDBParser

    class DummyDialog:
        pass

    for name in (
        "_aa3_to_aa1",
        "_normalize_chain_id",
        "_modeler_clean_sequence",
        "_modeler_make_chain_entry",
        "_modeler_chain_entry_from_residue_records",
        "_modeler_extract_chain_data_from_biopython",
        "_modeler_prepare_base_sequence",
        "_modeler_structure_numbering_for",
        "_modeler_apply_mutations_with_start_resolution",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    dlg = DummyDialog()
    dlg._modeler_build_structure_from_object = lambda _obj: PDBParser(QUIET=True).get_structure("r", _RECEPTOR_3PTB)
    dlg._modeler_current_mode = lambda: "gap"
    dlg._modeler_sequence_prompted = False
    dlg._modeler_chain_data = dlg._modeler_extract_chain_data_from_biopython("receptor")
    dlg.cmb_modeler_chain = _Widget(data="A")
    dlg.txt_modeler_full_sequence = _Widget(full_sequence)
    dlg.spin_modeler_seq_start = _Widget(16)
    return dlg


def test_modeler_chain_analysis_keeps_insertion_codes_and_ignores_numbering_jumps():
    dlg = _build_modeler_numbering_dialog()
    data = dlg._modeler_chain_data["A"]
    # trypsin (chymotrypsinogen numbering): 223 residues, 184A/188A/221A, jumps such as 34 -> 37
    assert data["observed_len"] == 223
    assert {"184A", "188A", "221A"} <= set(data["residue_labels"])
    assert data["internal_gaps"] == [] and data["missing_total"] == 0
    assert ("34", "37") in data["numbering_jumps"]


def test_modeler_gap_mode_uses_structure_numbering_for_mutations():
    import mutator

    observed = _build_modeler_numbering_dialog()._modeler_chain_data["A"]["observed_seq"]
    # the widget is auto-filled with the observed chain: no residue may be invented
    dlg = _build_modeler_numbering_dialog(full_sequence=observed)
    seq, start, _ch, _data, _notes = dlg._modeler_prepare_base_sequence()
    assert seq == observed and start == 16

    out = dlg._modeler_apply_mutations_with_start_resolution(
        seq, ["S195A"], start_pos=start, mutator_mod=mutator, allow_auto_resolve=False)
    labels = dlg._modeler_chain_data["A"]["residue_labels"]
    i195 = labels.index("195")
    assert seq[i195] == "S"                                   # catalytic serine of trypsin
    assert out["mutated_sequence"] == seq[:i195] + "A" + seq[i195 + 1:]
