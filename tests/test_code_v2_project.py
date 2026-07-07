import json
import os

import code_v2


def _build_dummy_dialog():
    class DummyDialog:
        pass

    for name in (
        "_project_root_dir",
        "_project_master_file",
        "_project_write_master_file",
        "_project_resolve_open_target",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return DummyDialog()


def test_project_write_master_file_creates_explicit_reopen_file(tmp_path):
    dlg = _build_dummy_dialog()
    dlg._project_root = str(tmp_path)
    dlg._project_name = "demo_project"
    dlg._project_state_version = 3

    state_payload = {
        "version": 3,
        "saved_at": "2026-03-31T12:34:56",
        "session_file": "session/current_scene.pse",
        "values": {"project_name": "demo_project"},
    }
    master_file = dlg._project_write_master_file(root=str(tmp_path), state_payload=state_payload)

    assert master_file == str(tmp_path / code_v2.PROJECT_MASTER_FILE_NAME)
    assert os.path.isfile(master_file)

    with open(master_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["kind"] == "leonardo_project"
    assert data["suite_name"] == code_v2.SUITE_NAME
    assert data["project_name"] == "demo_project"
    assert data["state_file"] == code_v2.PROJECT_STATE_FILE_NAME
    assert data["session_file"] == "session/current_scene.pse"


def test_project_resolve_open_target_accepts_master_file(tmp_path):
    dlg = _build_dummy_dialog()
    master_file = tmp_path / code_v2.PROJECT_MASTER_FILE_NAME
    master_file.write_text(
        json.dumps(
            {
                "kind": "leonardo_project",
                "state_file": code_v2.PROJECT_STATE_FILE_NAME,
            }
        ),
        encoding="utf-8",
    )

    resolved = dlg._project_resolve_open_target(str(master_file))

    assert resolved["ok"] is True
    assert resolved["root"] == str(tmp_path)
    assert resolved["source"] == "master_file"


def test_project_resolve_open_target_accepts_legacy_state_file(tmp_path):
    dlg = _build_dummy_dialog()
    state_file = tmp_path / code_v2.PROJECT_STATE_FILE_NAME
    state_file.write_text("{}", encoding="utf-8")

    resolved = dlg._project_resolve_open_target(str(state_file))

    assert resolved["ok"] is True
    assert resolved["root"] == str(tmp_path)
    assert resolved["source"] == "state_file"


def test_project_resolve_open_target_rejects_random_file(tmp_path):
    dlg = _build_dummy_dialog()
    other = tmp_path / "notes.json"
    other.write_text("{}", encoding="utf-8")

    resolved = dlg._project_resolve_open_target(str(other))

    assert resolved["ok"] is False
    assert code_v2.PROJECT_MASTER_FILE_NAME in resolved["error"]
