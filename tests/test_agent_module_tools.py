"""The assistant sees every Leonardo module; helpers that drive the dialog headlessly."""
import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets

from leonardo.agent.registry import ToolError  # noqa: E402
from leonardo.agent.tools import AgentContext, _dialogjob, _headless, _widgets, build_registry  # noqa: E402

MODULE_TOOLS = {
    "ligands": {"set_ligand_from_smiles", "load_ligand_file", "search_pubchem", "use_pubchem_compound",
                "parametrize_ligand", "export_ligand_files", "ligand_status"},
    "screening": {"run_virtual_screening", "show_screening_hit", "run_lead_optimization"},
    "trajectory": {"trajectory_rmsd_rmsf", "trajectory_interaction_occupancy", "trajectory_interaction_energy",
                   "run_abfe"},
    "modeling": {"model_mutant", "repair_structure_gaps", "build_homology_model", "search_templates"},
    "alphafold": {"predict_structure_alphafold", "check_alphafold_result"},
    "bioseq": {"analyze_sequence", "msa_conservation"},
    "psi4": {"run_psi4"},
    "qmmm": {"run_qmmm"},
    "md": {"run_md", "minimize_structure"},
}


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_every_leonardo_module_has_tools():
    names = set(build_registry(AgentContext()).names())
    for module, tools in MODULE_TOOLS.items():
        assert tools <= names, (module, tools - names)


def test_heavy_calculations_ask_for_confirmation():
    reg = build_registry(AgentContext())
    perms = {s.name: s.permission for s in reg.specs()}
    for name in ("run_abfe", "run_qmmm", "run_psi4", "run_virtual_screening", "run_lead_optimization",
                 "predict_structure_alphafold", "model_mutant", "repair_structure_gaps", "parametrize_ligand"):
        assert perms[name] == "compute_heavy", name


def test_set_combo_by_data_text_and_rejects_unknown(app):
    class D:
        pass
    d = D()
    d.cmb = QtWidgets.QComboBox()
    d.cmb.addItem("AM1-BCC (bcc)", "bcc")
    d.cmb.addItem("Gasteiger (gas)", "gas")
    _widgets.set_combo(d, "cmb", "gas")
    assert d.cmb.currentData() == "gas"
    _widgets.set_combo(d, "cmb", "am1-bcc (BCC)")
    assert d.cmb.currentData() == "bcc"
    with pytest.raises(ToolError):
        _widgets.set_combo(d, "cmb", "resp")


def test_jsonable_drops_molecules_and_nan():
    class Mol:
        pass
    out = _widgets.jsonable({"a": float("nan"), "mol": Mol(), "_private": 1, "rows": [1.234567891, math.pi]})
    assert out == {"a": None, "rows": [1.23457, 3.14159]}


def test_headless_prompts_take_preset_values(app):
    QID = QtWidgets.QInputDialog
    _headless.install()
    _headless.begin()
    try:
        _headless.preset_prompts({"ABFE-lite Simulation Length": 0.5, "ABFE-lite Replicas": 2})
        assert QID.getDouble(None, "ABFE-lite Simulation Length", "ns", 3.0, 0.2, 200.0, 2) == (0.5, True)
        assert QID.getInt(None, "ABFE-lite Replicas", "n", 3, 1, 12, 1) == (2, True)
        assert QID.getItem(None, "Pick", "which", ["x", "y"], 1, False) == ("y", True)
        box = QtWidgets.QMessageBox()
        box.setWindowTitle("ABFE summary")
        box.setText("dG = -20 kJ/mol")
        box.exec_()
        assert any("dG = -20" in m["text"] for m in _headless.take_messages())
    finally:
        _headless.preset_prompts({})
        _headless.end()


def test_finalize_job_reads_keyword_arguments(app):
    class Dlg:
        def _finalize(self, run_result=None, operation=""):
            self.seen = (run_result, operation)

    dlg = Dlg()
    ctx = AgentContext(dlg)
    got = {}

    def start():
        dlg._finalize(run_result={"success": True}, operation="energy")

    job = _dialogjob.start_finalize_job(ctx, "t", start, "_finalize",
                                        lambda call: got.setdefault("r", call.arg(0, "run_result")) or {})
    status = ctx.jobs.status(job["job_id"])
    assert status["status"] == "done"
    assert got["r"] == {"success": True}
    assert dlg._finalize.__name__ == "_finalize"          # the original method is back
