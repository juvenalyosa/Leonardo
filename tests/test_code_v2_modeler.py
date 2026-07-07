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
