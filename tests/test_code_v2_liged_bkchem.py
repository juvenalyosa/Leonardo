import sys

import pytest

import code_v2


def _build_dummy_dialog():
    class DummyDialog:
        pass

    for name in (
        "_extract_smiles_candidate_from_text",
        "_ligand_editor_mol_from_file",
        "_liged_bkchem_draft_paths",
        "_liged_export_current_to_bkchem_draft",
        "_liged_bkchem_launch_plan",
        "_ligand_editor_get_current_mol",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return DummyDialog()


def _disable_managed_local_detection(tmp_path, monkeypatch):
    fake_code = tmp_path / "sandbox_root" / "code_v2.py"
    fake_code.parent.mkdir(parents=True, exist_ok=True)
    fake_code.write_text("# test marker\n", encoding="utf-8")
    monkeypatch.setattr(code_v2.os, "getcwd", lambda: str(tmp_path / "cwd_root"))
    monkeypatch.setattr(code_v2, "__file__", str(fake_code))


def test_bkchem_launch_plan_uses_explicit_start_script_env(tmp_path, monkeypatch):
    dlg = _build_dummy_dialog()
    start = tmp_path / "start_bkchem.py"
    start.write_text("print('bkchem')\n", encoding="utf-8")
    _disable_managed_local_detection(tmp_path, monkeypatch)
    monkeypatch.setattr(code_v2.shutil, "which", lambda name: None)
    monkeypatch.setenv("LEONARDO_BKCHEM_START", str(start))
    monkeypatch.delenv("BKCHEM_START", raising=False)
    monkeypatch.delenv("LEONARDO_BKCHEM_HOME", raising=False)
    monkeypatch.delenv("BKCHEM_HOME", raising=False)

    plan = dlg._liged_bkchem_launch_plan()

    assert plan["ok"] is True
    assert plan["source"] == "env_script"
    assert plan["path"] == str(start)
    assert plan["command"] == [sys.executable, str(start)]


def test_bkchem_launch_plan_discovers_local_clone_from_home_env(tmp_path, monkeypatch):
    dlg = _build_dummy_dialog()
    root = tmp_path / "BKChem_python_3"
    pkg = root / "bkchem"
    pkg.mkdir(parents=True)
    start = root / "start_bkchem.py"
    start.write_text("print('bkchem')\n", encoding="utf-8")
    (pkg / "bkchem.py").write_text("APP = True\n", encoding="utf-8")
    _disable_managed_local_detection(tmp_path, monkeypatch)
    monkeypatch.setattr(code_v2.shutil, "which", lambda name: None)
    monkeypatch.delenv("LEONARDO_BKCHEM_START", raising=False)
    monkeypatch.delenv("BKCHEM_START", raising=False)
    monkeypatch.setenv("LEONARDO_BKCHEM_HOME", str(root))
    monkeypatch.delenv("BKCHEM_HOME", raising=False)

    plan = dlg._liged_bkchem_launch_plan()

    assert plan["ok"] is True
    assert plan["source"] == "local_clone"
    assert plan["path"] == str(start)
    assert plan["command"] == [sys.executable, str(start)]


def test_bkchem_launch_plan_prefers_repo_managed_local_install(tmp_path, monkeypatch):
    dlg = _build_dummy_dialog()
    root = tmp_path / "repo_root"
    start = root / "external" / "BKChem_python_3" / "start_bkchem.py"
    pkg = root / "external" / "BKChem_python_3" / "bkchem" / "bkchem.py"
    py_exec = root / ".bkchem-venv" / "bin" / "python"
    pkg.parent.mkdir(parents=True, exist_ok=True)
    py_exec.parent.mkdir(parents=True, exist_ok=True)
    start.write_text("print('bkchem')\n", encoding="utf-8")
    pkg.write_text("APP = True\n", encoding="utf-8")
    py_exec.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(code_v2.os, "getcwd", lambda: str(root))
    monkeypatch.setattr(code_v2, "__file__", str(root / "code_v2.py"))
    monkeypatch.setattr(code_v2.shutil, "which", lambda name: None)

    plan = dlg._liged_bkchem_launch_plan()

    assert plan["ok"] is True
    assert plan["source"] == "managed_local"
    assert plan["path"] == str(start)
    assert plan["python"] == str(py_exec)
    assert plan["command"] == [str(py_exec), str(start)]


def test_ligand_editor_mol_from_smiles_text_file(tmp_path):
    if not code_v2.HAS_RDKIT:
        pytest.skip("RDKit not available")

    dlg = _build_dummy_dialog()
    smiles_file = tmp_path / "ligand.smi"
    smiles_file.write_text("CCO ethanol\n", encoding="utf-8")

    mol = dlg._ligand_editor_mol_from_file(str(smiles_file))

    assert mol is not None
    assert int(mol.GetNumAtoms()) == 3


def test_bkchem_draft_export_writes_mol_and_smi_files(tmp_path):
    if not code_v2.HAS_RDKIT:
        pytest.skip("RDKit not available")

    dlg = _build_dummy_dialog()
    dlg._dock_workspace_dir = lambda: str(tmp_path)
    dlg._ligand_editor_mol = code_v2.Chem.MolFromSmiles("CCO")
    dlg.edit_liged_smiles = None

    payload = dlg._liged_export_current_to_bkchem_draft()

    assert payload["ok"] is True
    assert payload["mol"].endswith("ligand_editor_draft.mol")
    assert payload["smi"].endswith("ligand_editor_draft.smi")
    assert (tmp_path / "ligands" / "bkchem_draft" / "ligand_editor_draft.mol").is_file()
    smi_path = tmp_path / "ligands" / "bkchem_draft" / "ligand_editor_draft.smi"
    assert smi_path.is_file()
    assert smi_path.read_text(encoding="utf-8").strip() == "CCO"
