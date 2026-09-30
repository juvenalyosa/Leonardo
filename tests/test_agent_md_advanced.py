"""Enhanced-sampling templates and MD component selection driven through Leonardo's OpenMM dialog."""
import json
import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets

from leonardo.agent.registry import ToolError, ToolRegistry  # noqa: E402
from leonardo.agent.tools import AgentContext, md, md_advanced  # noqa: E402
from leonardo.openmm_dynamics_runtime import apply_openmm_cv_preset  # noqa: E402

PDB = """\
ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C
ATOM      3  C   ALA A   1      13.149   5.892  -5.195  1.00  0.00           C
ATOM      4  O   ALA A   1      13.719   5.286  -6.105  1.00  0.00           O
ATOM      5  CB  ALA A   1      11.265   7.330  -4.368  1.00  0.00           C
ATOM      6  N   ALA A   2      13.796   6.426  -4.163  1.00  0.00           N
ATOM      7  CA  ALA A   2      15.247   6.330  -4.058  1.00  0.00           C
ATOM      8  C   ALA A   2      15.719   5.051  -3.380  1.00  0.00           C
ATOM      9  O   ALA A   2      15.037   4.476  -2.531  1.00  0.00           O
ATOM     10  CB  ALA A   2      15.803   7.541  -3.316  1.00  0.00           C
ATOM     11  N   ALA A   3      16.914   4.620  -3.768  1.00  0.00           N
ATOM     12  CA  ALA A   3      17.483   3.393  -3.221  1.00  0.00           C
ATOM     13  C   ALA A   3      18.969   3.572  -2.932  1.00  0.00           C
ATOM     14  O   ALA A   3      19.620   4.461  -3.482  1.00  0.00           O
ATOM     15  CB  ALA A   3      17.268   2.232  -4.187  1.00  0.00           C
HETATM   16  C1  BEN A 301      14.000   9.000  -2.000  1.00  0.00           C
HETATM   17  C2  BEN A 301      15.390   9.000  -2.000  1.00  0.00           C
HETATM   18  C3  BEN A 301      16.085  10.204  -2.000  1.00  0.00           C
HETATM   19  C4  BEN A 301      15.390  11.407  -2.000  1.00  0.00           C
HETATM   20  C5  BEN A 301      14.000  11.407  -2.000  1.00  0.00           C
HETATM   21  C6  BEN A 301      13.305  10.204  -2.000  1.00  0.00           C
HETATM   22  C   BEN A 301      12.000  10.204  -2.000  1.00  0.00           C
HETATM   23  N1  BEN A 301      11.300   9.000  -2.000  1.00  0.00           N
HETATM   24  N2  BEN A 301      11.300  11.407  -2.000  1.00  0.00           N
HETATM   25 ZN    ZN A 401      20.000   6.000  -1.000  1.00  0.00          ZN
HETATM   26  O   HOH A 501      22.000   2.000  -1.000  1.00  0.00           O
HETATM   27  O   HOH A 502      23.000   8.000   1.000  1.00  0.00           O
END
"""


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def dialog(app, tmp_path_factory):
    import code_v2
    from pymol import cmd

    cmd.reinitialize()
    path = tmp_path_factory.mktemp("mdadv") / "cplx.pdb"
    path.write_text(PDB)
    cmd.load(str(path), "cplx")
    dlg = code_v2.ProtLigInteractDialog()
    work = str(tmp_path_factory.mktemp("ws"))
    dlg._dock_workspace_dir = lambda: work
    app.processEvents()
    yield dlg
    dlg.close()


def _reg():
    reg = ToolRegistry()
    md_advanced.register(reg)
    return reg


# ------------------------------------------------------------------ registration / pure mapping
def test_tools_registered_with_permissions():
    perms = {s.name: s.permission for s in _reg().specs()}
    assert perms == {"run_enhanced_sampling": "compute_heavy", "list_md_components": "read",
                     "configure_md_components": "compute_light"}
    assert _reg().get("list_md_components").planning_allowed


def test_schema_rejects_out_of_range_parameters():
    reg = _reg()
    with pytest.raises(ToolError):
        reg.validate("run_enhanced_sampling", {"object": "x", "mode": "umbrella", "windows": 1})
    with pytest.raises(ToolError):
        reg.validate("run_enhanced_sampling", {"object": "x", "mode": "metadynamics", "bias_factor": 1.0})
    with pytest.raises(ToolError):
        reg.validate("run_enhanced_sampling", {"object": "x", "mode": "abfe"})
    reg.validate("run_enhanced_sampling", {"object": "x", "mode": "metadynamics", "extra_cvs": [
        {"cv_type": "dihedral", "selection_a": "a", "cv_start": -3.1, "cv_end": 3.1, "bias_width": 0.1}]})


def test_umbrella_payload_maps_parameters():
    base = apply_openmm_cv_preset("protein_ligand_distance", {"ligand": "(resn BEN)"}, workflow="umbrella")
    payload, summary = md_advanced.build_umbrella_payload(base, "cplx", {
        "cv_start": 0.3, "cv_end": 1.5, "windows": 25, "force_constant_kj_mol_nm2": 2000,
        "production_ns_per_window": 0.2, "equilibration_ps_per_window": 20, "timestep_fs": 2.0,
        "temperature_K": 310})
    w = payload["windowing"]
    assert (w["n_windows"], w["spring_k_kj_mol_nm2"], w["prod_steps"], w["equil_steps"]) == (25, 2000.0, 100000, 10000)
    assert payload["cv"]["range_nm"] == [0.3, 1.5] and payload["cv"]["umbrella"]["n_windows"] == 25
    assert payload["thermostat"] == {"temperature_k": 310.0, "timestep_fs": 2.0}
    assert summary["total_md_steps"] == 25 * 110000 and summary["total_simulated_ns"] == 5.5
    assert "warning" not in summary["overlap"]          # 0.05 nm spacing, sigma 0.036 nm


def test_umbrella_overlap_warning_and_range_checks():
    info = md_advanced._umbrella_overlap_warning(0.2, 2.2, 5, 5000, 300, "distance")
    assert "warning" in info and info["window_spacing"] == 0.5
    base = apply_openmm_cv_preset("protein_ligand_distance", {}, workflow="umbrella")
    with pytest.raises(ToolError, match="larger than"):
        md_advanced.build_umbrella_payload(base, "o", {"cv_start": 1.0, "cv_end": 0.5})
    with pytest.raises(ToolError, match="periodic"):
        md_advanced.build_umbrella_payload(base, "o", {"periodic": True})
    with pytest.raises(ToolError, match="radians"):
        md_advanced.build_umbrella_payload(base, "o", {"cv_type": "dihedral", "selection_a": "a", "selection_b": "b",
                                                       "selection_c": "c", "selection_d": "d",
                                                       "cv_start": -4.0, "cv_end": 3.0})
    with pytest.raises(ToolError, match="selection_c"):    # an angle needs three groups
        md_advanced.build_umbrella_payload(base, "o", {"cv_type": "angle", "selection_a": "a", "selection_b": "b",
                                                       "cv_start": 1.0, "cv_end": 2.0})


def test_metadynamics_payload_maps_parameters(tmp_path):
    base = apply_openmm_cv_preset("protein_ligand_distance", {"ligand": "(resn BEN)"}, workflow="metadynamics")
    payload, summary = md_advanced.build_metadynamics_payload(base, "cplx", {
        "production_ns": 0.5, "timestep_fs": 2.0, "bias_factor": 8, "hill_height_kj_mol": 1.0,
        "deposit_frequency_steps": 500, "save_frequency_steps": 1200, "bias_width": 0.05,
        "extra_cvs": [{"cv_type": "dihedral", "selection_a": "name N", "selection_b": "name CA",
                       "selection_c": "name C", "selection_d": "name O", "cv_start": -math.pi,
                       "cv_end": math.pi, "bias_width": 0.2, "periodic": True}]}, str(tmp_path))
    assert payload["total_steps"] == 250000 and payload["frequency_steps"] == 500
    assert payload["save_frequency_steps"] == 1500                 # rounded up to a multiple of frequency
    cv1, cv2 = payload["collective_variables"]
    assert cv1["metadynamics"]["bias_width"] == 0.05 and cv1["metadynamics"]["min_value"] == 0.15
    assert cv2["type"] == "dihedral" and cv2["metadynamics"]["periodic"] is True
    assert summary["hills"] == 500 and summary["delta_T_K"] == 2100.0
    with pytest.raises(ToolError, match="bias_width"):
        md_advanced.build_metadynamics_payload(base, "cplx", {"cv_start": 0.2, "cv_end": 0.4, "bias_width": 0.5},
                                               str(tmp_path))
    with pytest.raises(ToolError, match="deposit_frequency"):
        md_advanced.build_metadynamics_payload(base, "cplx", {"production_ns": 0.0002, "deposit_frequency_steps": 500},
                                               str(tmp_path))


def test_component_specifiers():
    ben = {"resn": "BEN", "chain": "A", "resi": "301", "category": "cofactor"}
    zn = {"resn": "ZN", "chain": "A", "resi": "401", "category": "ion"}
    m = md_advanced.component_matches
    p = md_advanced.parse_component_spec
    assert m(p("BEN"), ben) and not m(p("BEN"), zn)
    assert m(p("ben A:301"), ben) and not m(p("BEN B:301"), ben) and m(p("A:301"), ben)
    assert m(p("ZN 401"), zn) and m(p("ions"), zn) and not m(p("ions"), ben)
    assert m(p("ligands"), ben, "needs_parameterization") and not m(p("ligands"), ben, "standard")
    assert m(p("all"), zn)
    with pytest.raises(ToolError):
        p("  ")


# ------------------------------------------------------------------ real Leonardo dialog
def test_umbrella_template_written_through_gui_workflow(dialog):
    ctx = AgentContext(dlg=dialog)
    res = _reg().call("run_enhanced_sampling", {
        "object": "cplx", "mode": "umbrella", "cv_type": "distance", "selection_a": "resn BEN",
        "selection_b": "resi 2 and name CA", "cv_start": 0.4, "cv_end": 1.4, "windows": 21,
        "force_constant_kj_mol_nm2": 1500, "production_ns_per_window": 0.1}, ctx)
    assert "_openmm_prompt_umbrella_params" not in vars(dialog)     # the pop-up is the GUI's again
    tpl = json.load(open(res["template"]))
    assert tpl["workflow"] == "umbrella" and tpl["object"] == "cplx"
    cv = tpl["parameters"]["cv"]
    assert (cv["selection_a"], cv["selection_b"], cv["range_nm"]) == ("resn BEN", "resi 2 and name CA", [0.4, 1.4])
    assert tpl["parameters"]["windowing"]["n_windows"] == 21
    assert res["executed_production"] is False and "No production run" in res["_card"]
    assert "template" in dialog.lbl_openmm_advanced.text().lower()


def test_empty_selection_is_reported_and_gui_untouched(dialog):
    ctx = AgentContext(dlg=dialog)
    with pytest.raises(ToolError, match="match no atoms"):
        _reg().call("run_enhanced_sampling", {"object": "cplx", "mode": "umbrella", "selection_a": "resn XYZ",
                                              "selection_b": "name CA"}, ctx)
    with pytest.raises(ToolError, match="no object"):
        _reg().call("run_enhanced_sampling", {"object": "missing", "mode": "umbrella"}, ctx)
    assert "_openmm_prompt_umbrella_params" not in vars(dialog)


def test_prompt_restored_when_gui_handler_fails(dialog, monkeypatch):
    ctx = AgentContext(dlg=dialog)
    monkeypatch.setattr(dialog, "_on_openmm_run_advanced_workflow", lambda **k: 1 / 0)
    with pytest.raises(ToolError):
        _reg().call("run_enhanced_sampling", {"object": "cplx", "mode": "umbrella", "selection_a": "resn BEN",
                                              "selection_b": "name CA"}, ctx)
    assert "_openmm_prompt_umbrella_params" not in vars(dialog)


def test_default_preset_ignores_gui_combo(dialog):
    ctx = AgentContext(dlg=dialog)
    dialog.cmb_openmm_cv_preset.setCurrentIndex(dialog.cmb_openmm_cv_preset.findData("torsion_dihedral"))
    try:
        res = _reg().call("run_enhanced_sampling", {"object": "cplx", "mode": "umbrella"}, ctx)
    finally:
        dialog.cmb_openmm_cv_preset.setCurrentIndex(0)
    cv = json.load(open(res["template"]))["parameters"]["cv"]
    assert cv["type"] == "distance" and cv["preset_id"] == "protein_ligand_distance"
    assert "BEN" in cv["selection_a"] or "organic" in cv["selection_a"]


def test_metadynamics_smoke_test(dialog):
    pytest.importorskip("openmm")
    ctx = AgentContext(dlg=dialog)
    res = _reg().call("run_enhanced_sampling", {
        "object": "cplx", "mode": "metadynamics", "action": "smoke_test", "selection_a": "resn BEN",
        "selection_b": "name CA", "cv_start": 0.2, "cv_end": 1.5, "bias_width": 0.05,
        "production_ns": 0.01, "deposit_frequency_steps": 100}, ctx)
    assert res["smoke_test"]["ok"] and os.path.isfile(res["smoke_test"]["summary_json"])
    assert json.load(open(res["template"]))["parameters"]["total_steps"] == 5000
    assert "not a PMF" in res["_card"]


def test_openmm_ml_template(dialog):
    ctx = AgentContext(dlg=dialog)
    res = _reg().call("run_enhanced_sampling", {"object": "cplx", "mode": "openmm_ml", "ml_mode": "mixed",
                                                "ml_region_selection": "resn BEN", "production_ns": 0.01}, ctx)
    params = json.load(open(res["template"]))["parameters"]
    assert params["mode"] == "mixed" and params["ml_region_selection"] == "resn BEN" and params["steps"] == 10000
    with pytest.raises(ToolError, match="smoke_test"):
        _reg().call("run_enhanced_sampling", {"object": "cplx", "mode": "openmm_ml", "action": "smoke_test"}, ctx)


def test_list_and_configure_components(dialog):
    ctx = AgentContext(dlg=dialog)
    reg = _reg()
    rep = reg.call("list_md_components", {"object": "cplx"}, ctx)
    by = {r["resn"]: r for r in rep["components"]}
    assert set(by) == {"BEN", "ZN", "HOH"}
    assert by["BEN"]["ff_class"] == "needs_parameterization" and by["ZN"]["category"] == "ion"

    before = dialog._prompt_dock_hetatm_selection
    rep = reg.call("configure_md_components", {"object": "cplx", "include": ["BEN A:301"], "net_charge": 1,
                                               "charge_method": "gas"}, ctx)
    assert dialog._prompt_dock_hetatm_selection == before                # chooser restored
    inside = {r["resn"] for r in rep["components"] if r["in_md_system"]}
    assert inside == {"BEN"} and rep["selection_source"].startswith("explicit")
    assert [c["resn"] for c in dialog._openmm_selected_components("cplx")] == ["BEN"]
    assert dialog._openmm_crystal_parameterization_defaults() == {"charge_method": "gas", "atom_type": "gaff2",
                                                                  "net_charge": 1}
    assert rep["will_parameterize_at_md_time"] == ["BEN A:301"]

    rep = reg.call("configure_md_components", {"object": "cplx", "exclude": ["waters"]}, ctx)
    assert {r["resn"] for r in rep["components"] if r["in_md_system"]} == {"BEN", "ZN"}

    rep = reg.call("configure_md_components", {"object": "cplx", "include": ["none"]}, ctx)
    assert not [r for r in rep["components"] if r["in_md_system"]]
    assert dialog._openmm_selected_components("cplx") == []

    with pytest.raises(ToolError, match="no component matches"):
        reg.call("configure_md_components", {"object": "cplx", "include": ["HEM"]}, ctx)


def test_external_ligand_xml(dialog, tmp_path):
    ctx = AgentContext(dlg=dialog)
    reg = _reg()
    xml = tmp_path / "ben.xml"
    xml.write_text('<ForceField>\n <AtomTypes/>\n</ForceField>\n')
    before = vars(QtWidgets.QFileDialog).get("getOpenFileNames")
    rep = reg.call("configure_md_components", {"object": "cplx", "clear_ligand_xml": True,
                                               "ligand_xml": [str(xml)]}, ctx)
    assert vars(QtWidgets.QFileDialog).get("getOpenFileNames") is before   # file dialog restored
    assert str(xml) in rep["ligand_xml_paths"] and rep["added_xml"] == [str(xml)]
    assert str(xml) in dialog.edit_openmm_ligand_xml.text()
    bad = tmp_path / "notes.xml"
    bad.write_text("<Notes/>")
    with pytest.raises(ToolError, match="not an OpenMM"):
        reg.call("configure_md_components", {"object": "cplx", "ligand_xml": [str(bad)]}, ctx)
    with pytest.raises(ToolError, match="not found"):
        reg.call("configure_md_components", {"object": "cplx", "ligand_xml": [str(tmp_path / "none.xml")]}, ctx)
    dialog._parameterized_ligand_xml = ""
    with pytest.raises(ToolError, match="parametrize_ligand"):
        reg.call("configure_md_components", {"object": "cplx", "use_current_ligand_xml": True}, ctx)


def test_run_md_keep_switches_replace_component_choice(dialog, monkeypatch):
    ctx = AgentContext(dlg=dialog)
    _reg().call("configure_md_components", {"object": "cplx", "include": ["BEN"]}, ctx)
    monkeypatch.setattr(md, "_openmm_job", lambda *a, **k: {"job": "stub"})
    reg = ToolRegistry()
    md.register(reg)
    reg.call("run_md", {"object": "cplx"}, ctx)                   # no keep_*: the choice stays
    assert "cplx" in dialog._openmm_hetatm_selection_cache
    reg.call("run_md", {"object": "cplx", "keep_ions": True, "keep_cofactors": False}, ctx)
    assert "cplx" not in dialog._openmm_hetatm_selection_cache
    assert [c["resn"] for c in dialog._openmm_selected_components("cplx")] == ["ZN"]
