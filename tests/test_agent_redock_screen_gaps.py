"""Redocking / screening / ABFE assistant tools: HETATM choice, grid box, pose browsing, CSV, steering policy.
No docking or MD runs: Leonardo's long handlers are replaced by fakes that record what they would have used."""
import csv
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets

from leonardo.agent.registry import ToolError  # noqa: E402
from leonardo.agent.tools import AgentContext, _headless, build_registry, redocking, trajectory  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _clean_headless():
    yield
    _headless.reset()


def _props(reg, name):
    return reg.get(name).parameters["properties"]


def test_schemas_and_permissions():
    reg = build_registry(AgentContext())
    perms = {s.name: s.permission for s in reg.specs()}
    for name in ("run_redocking", "run_virtual_screening", "run_lead_optimization"):
        assert _props(reg, name)["keep_hetero"]["type"] == "array", name
        assert perms[name] == "compute_heavy"
    red = _props(reg, "run_redocking")
    assert red["grid_size_A"]["minItems"] == red["grid_size_A"]["maxItems"] == 3
    assert {"grid_center_A", "grid_padding_A"} <= set(red)
    assert perms["show_redock_pose"] == "compute_light"
    assert perms["export_redocking_csv"] == "compute_light"
    assert reg.get("show_redock_pose").parameters["required"] == ["rank"]
    assert _props(reg, "run_abfe")["steering_policy"]["enum"] == ["heuristic", "bandit"]
    assert perms["run_abfe"] == "compute_heavy"
    reg.validate("run_redocking", {"object": "x", "keep_hetero": ["HEM"], "grid_size_A": [20, 20, 22]})
    with pytest.raises(ToolError):
        reg.validate("run_redocking", {"object": "x", "grid_size_A": [20, 20]})
    with pytest.raises(ToolError):
        reg.validate("run_abfe", {"object": "x", "steering_policy": "greedy"})


def _raw(system, mode, aff, rmsd, member="", pose="redock_wt_poses", repeat=1):
    return {"system": system, "system_member": member, "mode": mode, "affinity": aff, "rmsd_ref": rmsd,
            "pose_object": pose, "state": mode, "repeat": repeat}


# every docked mode as Leonardo keeps it (_redock_raw_results); its table shows the best per system/member
RAW = [_raw("WT", 1, -9.1, 0.8), _raw("WT", 2, -8.2, 4.1), _raw("WT", 3, -8.6, 2.5),
       _raw("Mutant", 1, -7.0, 1.2, pose="redock_mut_poses"), _raw("Mutant", 2, -6.5, 2.9, pose="redock_mut_poses")]


class _Dlg:
    def __init__(self, raw, table):
        self._redock_raw_results, self._redock_results = raw, table


def test_ranked_poses_and_pick():
    table = [dict(RAW[0], rank=1), dict(RAW[3], rank=1)]
    items = redocking._ranked_poses(_Dlg(RAW, table))
    wt = [(it["rank"], it["row"]["mode"]) for it in items if it["system"] == "WT"]
    assert wt == [(1, 1), (2, 3), (3, 2)]                      # ranked by score, top = table row
    assert all(it["table_index"] == 0 for it in items if it["system"] == "WT")
    it, label = redocking._pick_pose(items, 2)
    assert label == "WT" and it["row"]["mode"] == 3
    it, label = redocking._pick_pose(items, 2, "mutant")
    assert label == "Mutant" and it["row"]["affinity"] == -6.5 and it["table_index"] == 1
    with pytest.raises(ToolError, match="ranks 1..3"):
        redocking._pick_pose(items, 5)
    with pytest.raises(ToolError, match="no redocking system"):
        redocking._pick_pose(items, 1, "Q61L")
    with pytest.raises(ToolError, match="run run_redocking"):
        redocking._pick_pose([], 1)
    # ensemble: one table row per member; default member = best-scoring one
    ens_raw = [_raw("WT", 1, -8.0, 1.0, "WT_model_001"), _raw("WT", 1, -9.5, 1.5, "WT_model_002"),
               _raw("WT", 2, -9.0, 3.0, "WT_model_002")]
    ens_table = [dict(ens_raw[0], rank=1), dict(ens_raw[1], rank=2)]
    items = redocking._ranked_poses(_Dlg(ens_raw, ens_table))
    it, _ = redocking._pick_pose(items, 1)
    assert it["member"] == "WT_model_002"
    it, _ = redocking._pick_pose(items, 2, "WT", "wt_model_002")
    assert it["row"]["affinity"] == -9.0 and it["table_index"] == 1
    with pytest.raises(ToolError, match="member"):
        redocking._pick_pose(items, 1, "WT", "WT_model_009")


def _complex(cmd):
    """A small protein with a 4-atom 'crystal ligand' LIG X900 (no files needed)."""
    cmd.reinitialize()
    cmd.fab("GASGAKLV", "prot", ss=1)
    for i, pos in enumerate(([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 6.0])):
        cmd.pseudoatom("prot", name=f"C{i + 1}", resn="LIG", resi="900", chain="X", elem="C", pos=pos, hetatm=1)


@pytest.fixture
def dlg(app, tmp_path, monkeypatch):
    import code_v2
    from pymol import cmd

    _complex(cmd)
    d = code_v2.ProtLigInteractDialog()
    app.processEvents()
    d._save_settings = lambda *a, **k: None          # never touch the user's settings file
    d._dock_workspace_dir = lambda: str(tmp_path)
    monkeypatch.setattr("leonardo.agent.presentation.save_step_scene", lambda *a, **k: {"scene": "s"})
    yield d
    d.close()


def test_run_redocking_keep_hetero_and_grid_box(dlg):
    ctx = AgentContext(dlg)
    seen = {}

    def fake_workflow():
        seen["keep"] = set(_headless._state["keep_hetero"])
        seen["active"] = _headless.active()
        fields = dlg._redock_get_reference_fields()
        center, size, _n = dlg._redock_grid_from_reference("prot", fields)
        seen["center"], seen["size"] = [float(v) for v in center], [float(v) for v in size]
        # the HETATM chooser as Leonardo calls it during redocking
        comps = [{"resn": "HEM", "category": "cofactor", "chain": "A", "resi": "1"},
                 {"resn": "XYZ", "category": "cofactor", "chain": "A", "resi": "2"},
                 {"resn": "ZN", "category": "ion", "chain": "A", "resi": "3"}]
        pick = dlg._prompt_dock_hetatm_selection("prot", comps, flow_label="redocking")
        seen["kept"] = sorted(c["resn"] for c in pick["selected"])
        dlg._redock_last_runs = {"WT": {"ok": True, "best": {"mode": 1, "affinity": -9.1, "rmsd_ref": 0.8},
                                        "top1_success": {"n_success": 1, "n": 1},
                                        "repeat_rows": [{"repeat": 1, "seed": 1001, "best_affinity": -9.1,
                                                         "best_rmsd": 0.8, "min_rmsd_all_poses": 0.5}]}}
        dlg._redock_raw_results = RAW[:3]
        dlg._populate_redocking_table(RAW[:3])

    dlg._on_run_redocking_workflow = fake_workflow
    out = build_registry(ctx).call("run_redocking", {
        "object": "prot", "ligand_chain": "X", "ligand_resi": "900", "ligand_resn": "LIG",
        "keep_hetero": ["xyz"], "grid_size_A": [21.0, 22.0, 23.5], "grid_center_A": [1.0, 2.0, 3.0]}, ctx)
    assert seen["active"] and seen["keep"] == {"XYZ"}
    assert seen["kept"] == ["HEM", "XYZ", "ZN"]                  # known cofactor + asked + ions
    assert seen["size"] == pytest.approx([21.0, 22.0, 23.5])     # the GUI's manual grid delta
    assert seen["center"] == pytest.approx([1.0, 2.0, 3.0])
    assert _headless._state["keep_hetero"] == set()              # per-run only
    assert "_redock_grid_from_reference" not in dlg.__dict__     # center override removed
    assert out["grid_box"]["size_A"] == pytest.approx([21.0, 22.0, 23.5])
    assert out["grid_box"]["center_source"] == "manual"
    assert out["kept_hetero"] == ["XYZ"]
    assert [(p["system"], p["rank"], p["mode"], p["poses_available"]) for p in out["poses"]] == [("WT", 1, 1, 3)]
    assert "manual delta" in dlg.lbl_redock_grid_box.text()


def test_run_redocking_says_when_the_requested_ensemble_was_not_used(dlg):
    ctx = AgentContext(dlg)

    def fake_workflow():
        # Leonardo fell back to the single receptor (e.g. the sampler failed) and left the reason
        dlg._redock_last_mc_ensemble_note = "WT: Monte Carlo sampling failed (no template); single receptor used."
        dlg._redock_last_runs = {"WT": {"ok": True, "best": {"mode": 1, "affinity": -9.1, "rmsd_ref": 0.8},
                                        "top1_success": {"n_success": 1, "n": 1},
                                        "repeat_rows": [{"repeat": 1, "seed": 1001, "best_affinity": -9.1,
                                                         "best_rmsd": 0.8, "min_rmsd_all_poses": 0.5}]}}
        dlg._redock_raw_results = RAW[:3]
        dlg._populate_redocking_table(RAW[:3])

    dlg._on_run_redocking_workflow = fake_workflow
    out = build_registry(ctx).call("run_redocking", {
        "object": "prot", "ligand_chain": "X", "ligand_resi": "900", "ligand_resn": "LIG",
        "ensemble_structures": 3}, ctx)
    assert out["receptor_ensemble"]["used"] is False and out["receptor_ensemble"]["requested"] == 3
    assert "sampling failed" in out["receptor_ensemble"]["reason"]
    assert "NOT used" in out["_card"] and "single-receptor result" in out["_card"]


def test_run_redocking_resolves_a_partial_ligand_residue(dlg):
    """resi without resn (or resn without resi) names the crystal ligand: it was ignored for an
    auto-detected one (resi alone) or failed in Leonardo's form (resn alone)."""
    from pymol import cmd

    cmd.pseudoatom("prot", name="C1", resn="LIG", resi="-5", chain="Y", elem="C", pos=[9.0, 9.0, 9.0], hetatm=1)
    cmd.pseudoatom("prot", name="C1", resn="ACT", resi="901", chain="X", elem="C", pos=[7.0, 0.0, 0.0], hetatm=1)
    ctx = AgentContext(dlg)
    seen = []

    def fake_workflow():
        seen.append(dlg._redock_get_reference_fields())
        dlg._redock_last_runs = {"WT": {"ok": True, "best": {"mode": 1, "affinity": -6.0},
                                        "top1_success": {"n_success": 0, "n": 0},
                                        "repeat_rows": [{"repeat": 1, "seed": 1001, "best_affinity": -6.0,
                                                         "best_rmsd": float("nan")}]}}

    dlg._on_run_redocking_workflow = fake_workflow
    reg = build_registry(ctx)
    out = reg.call("run_redocking", {"object": "prot", "ligand_chain": "Y", "ligand_resi": "-5"}, ctx)
    assert seen[-1] == ("Y", "-5", "LIG")
    assert out["crystal_ligand"] == {"chain": "Y", "resi": "-5", "resn": "LIG"}
    # no "RMSD top-1 nan" when the poses could not be matched to the crystal ligand
    assert "nan" not in out["_card"] and "RMSD unavailable" in out["_card"]
    reg.call("run_redocking", {"object": "prot", "ligand_resn": "ACT"}, ctx)
    assert seen[-1] == ("X", "901", "ACT")
    with pytest.raises(ToolError, match="no residue"):
        reg.call("run_redocking", {"object": "prot", "ligand_resn": "XYZ"}, ctx)
    n = len(seen)
    with pytest.raises(ToolError, match="ligand_chain alone"):        # was dropped for an auto-detected ligand
        reg.call("run_redocking", {"object": "prot", "ligand_chain": "Y"}, ctx)
    assert len(seen) == n


def test_residue_selection_is_one_residue():
    from pymol import cmd

    from leonardo.agent.tools.docking import _residue_selection

    cmd.reinitialize()
    for chain, resi, x in (("A", "-5", 0.0), ("A", "3", 2.0), ("B", "-5", 4.0), ("", "-5", 6.0)):
        cmd.pseudoatom("c", name="C1", resn="LIG", resi=resi, chain=chain, elem="C", pos=[x, 0.0, 0.0], hetatm=1)
    assert cmd.count_atoms(_residue_selection("c", "A", "-5", "LIG")) == 1       # not resi 3, not chain B
    assert cmd.count_atoms(_residue_selection("c", "", "-5", "LIG")) == 1        # blank chain id
    assert cmd.count_atoms(_residue_selection("c", None, "-5", "LIG")) == 3      # any chain
    cmd.reinitialize()


def test_run_redocking_padding_without_manual_size(dlg):
    ctx = AgentContext(dlg)

    def fake_workflow():
        dlg._redock_last_runs = {"WT": {"ok": True, "best": {}, "top1_success": {}}}

    dlg._on_run_redocking_workflow = fake_workflow
    out = build_registry(ctx).call("run_redocking", {
        "object": "prot", "ligand_chain": "X", "ligand_resi": "900", "ligand_resn": "LIG",
        "grid_padding_A": 5.0}, ctx)
    assert dlg.spin_redock_padding.value() == pytest.approx(5.0)
    assert out["grid_box"]["center_source"] == "crystal ligand"
    assert out["grid_box"]["center_A"] == pytest.approx([2.0, 1.0, 3.0])     # bounding-box centre
    assert out["grid_box"]["manual_size_delta_A"] == [0.0, 0.0, 0.0]


def _load_results(dlg, raw=RAW):
    dlg._redock_raw_results = [dict(r) for r in raw]
    dlg._populate_redocking_table(dlg._redock_raw_results)


def test_show_redock_pose_drives_leonardo_handlers(dlg):
    ctx = AgentContext(dlg)
    _load_results(dlg)
    calls = []

    def load(silent=False):
        rows = dlg._redock_results
        row = rows[dlg.tbl_redock_results.currentRow()]
        calls.append(("load", row["system"], row["mode"], len(rows)))
    dlg._on_redock_load_selected_pose = load

    def interactions():
        load()
        calls[-1] = ("inter",) + calls[-1][1:]
        dlg.interactions = [{"type": "hbond", "prot_res": "ASP189", "distance": 2.9},
                            {"type": "hydrophobic", "prot_res": "VAL213", "distance": 3.8}]
    dlg._on_redock_compute_interactions_for_selected = interactions
    reg = build_registry(ctx)
    out = reg.call("show_redock_pose", {"rank": 1, "system": "mutant"}, ctx)
    assert calls[-1] == ("load", "Mutant", 1, 2) and dlg.tbl_redock_results.currentRow() == 1
    assert out["system"] == "Mutant" and out["score_kcal_mol"] == -7.0
    out = reg.call("show_redock_pose", {"rank": 3, "interactions": True}, ctx)
    assert calls[-1] == ("inter", "WT", 2, 1)                     # a non-top mode, as the selected row
    assert out["mode"] == 2 and out["score_kcal_mol"] == -8.2 and out["rmsd_A"] == 4.1
    assert out["interactions_by_type"] == {"hbond": 1, "hydrophobic": 1}
    assert out["closest"][0]["residue"] == "ASP189"
    assert isinstance(dlg.tbl_redock_results, QtWidgets.QTableWidget) and len(dlg._redock_results) == 2


def test_show_redock_pose_real_preview(dlg):
    """Leonardo's own 'Load selected pose' builds the preview object from the pose object and state."""
    from pymol import cmd

    for state in (1, 2, 3):
        cmd.create("redock_wt_poses", "prot and resn LIG", 1, state)
    cmd.translate([0.0, 0.0, 9.0], "redock_wt_poses", state=2, camera=0)
    _load_results(dlg, RAW[:3])
    ctx = AgentContext(dlg)
    out = build_registry(ctx).call("show_redock_pose", {"rank": 3}, ctx)       # mode 2 = state 2
    assert out["shown_object"] == "redock_selected_pose_preview"
    z = [c[2] for c in cmd.get_coords("redock_selected_pose_preview", state=1)]
    assert min(z) == pytest.approx(9.0, abs=1e-3)
    assert dlg.tbl_redock_results.currentRow() == 0


def test_export_redocking_csv_poses_and_comparison(dlg, tmp_path):
    ctx = AgentContext(dlg)
    reg = build_registry(ctx)
    with pytest.raises(ToolError, match="run run_redocking"):
        reg.call("export_redocking_csv", {}, ctx)
    _load_results(dlg, RAW[:3])
    dlg._redock_last_runs = {"WT": {"ok": True}}
    out = reg.call("export_redocking_csv", {}, ctx)
    assert "comparison_csv" not in out and "WT-only" in out["comparison_note"]
    with open(out["poses_csv"]) as fh:
        got = list(csv.DictReader(fh))
    assert [(r["system"], r["rank"], r["mode"], r["score_kcal_mol"]) for r in got] == [
        ("WT", "1", "1", "-9.1"), ("WT", "2", "3", "-8.6"), ("WT", "3", "2", "-8.2")]
    assert os.path.dirname(out["poses_csv"]).startswith(str(tmp_path))

    getter = QtWidgets.QFileDialog.getSaveFileName
    _load_results(dlg)
    dlg._redock_last_runs = {"WT": {"ok": True}, "Mutant": {"ok": True}}
    dlg._redock_last_comparison_rows = [{"wt_label": "WT", "mutant_label": "Mutant", "mutations": "D189N",
                                         "wt_affinity_kcal": -9.1, "mut_affinity_kcal": -7.0,
                                         "delta_affinity_kcal": 2.1, "classification": "destabilizing"}]
    target = tmp_path / "cmp" / "redock_cmp"
    out = reg.call("export_redocking_csv", {"path": str(target)}, ctx)
    assert out["comparison_csv"] == str(target) + ".csv" and os.path.isfile(out["comparison_csv"])
    assert out["poses_csv"] == str(target) + "_poses.csv" and out["poses"] == 5
    with open(out["comparison_csv"]) as fh:
        assert next(csv.DictReader(fh))["mutations"] == "D189N"
    assert QtWidgets.QFileDialog.getSaveFileName == getter        # the file dialog is back


def test_abfe_steering_policy_drives_the_gui_combo(dlg):
    assert trajectory._set_abfe_policy(dlg, "bandit") == "bandit"
    assert dlg._abfe_policy_mode(source="traj") == "bandit"
    assert trajectory._set_abfe_policy(dlg, "Probe_Control") == "heuristic"
    assert dlg._abfe_policy_mode(source="traj") == "heuristic"
    if getattr(dlg, "cmb_abfe_policy_mode", None) is not None:
        assert dlg.cmb_abfe_policy_mode.currentData() == "heuristic"
    with pytest.raises(ToolError):
        trajectory._set_abfe_policy(dlg, "greedy")
    assert trajectory._abfe_policy_used(dlg, [{"policy_mode": "bandit"}]) == "bandit"
    assert trajectory._abfe_policy_used(dlg, []) == "heuristic"


def test_run_abfe_passes_the_policy_to_leonardo(dlg, app):
    from pymol import cmd

    ctx = AgentContext(dlg)
    seen = {}

    def fake_md(**kwargs):
        seen.update(kwargs)
        return {"dg_pop_kj": -20.0, "p_bound": 0.9, "kd_app_m": 1e-6}

    def fake_run(source="traj"):
        dlg._run_abfe_lite_local_md(policy_mode=dlg._abfe_policy_mode(source=source))
        raise RuntimeError("stop before the report")

    dlg._run_abfe_lite_local_md = fake_md
    dlg._run_trajectory_free_energy_calculation = fake_run
    assert "prot" in cmd.get_names("objects")
    job = build_registry(ctx).call("run_abfe", {"object": "prot", "steering_policy": "bandit"}, ctx)
    for _ in range(50):
        app.processEvents()
        status = ctx.jobs.status(job["job_id"])
        if status["status"] != "running":
            break
    assert seen["policy_mode"] == "bandit"
    assert "stop before the report" in str(status.get("error", ""))
    assert dlg._run_abfe_lite_local_md is fake_md                  # the tool's wrapper was removed


class _ScreenDlg:
    """What run_virtual_screening / run_lead_optimization touch; the screen itself only asks the
    HETATM chooser (as Leonardo's receptor preparation does) and returns one hit."""

    def __init__(self, workspace):
        self._workspace = workspace
        self._fpocket_pockets = [{"center": (0, 0, 0)}]
        self._fpocket_receptor_object = "prot"
        self._dock_hetatm_selection_cache = {"prot": ["stale"]}
        self.vs_structure_combo = QtWidgets.QComboBox()
        self.cmb_vs_pocket = QtWidgets.QComboBox()
        self.tbl_vs_hits = QtWidgets.QTableWidget(1, 1)
        self.spin_vs_exhaustiveness = QtWidgets.QSpinBox()
        self.spin_vs_exhaustiveness.setRange(1, 64)
        self.spin_vs_max_ligands = QtWidgets.QSpinBox()
        self.spin_vs_max_ligands.setRange(1, 5000)
        self.spin_vs_ph = QtWidgets.QDoubleSpinBox()
        self.spin_vs_ph.setRange(0, 14)
        self.picked = None

    def _dock_workspace_dir(self):
        return self._workspace

    def _refresh_virtual_screening_receptor_combo(self):
        self.vs_structure_combo.clear()
        self.vs_structure_combo.addItem("prot")

    def _refresh_virtual_screening_pocket_combo(self):
        self.cmb_vs_pocket.clear()
        self.cmb_vs_pocket.addItem("Pocket 1")

    def _dock_hetatm_component_cache_key(self, c):
        return c["resn"]

    def _prompt_dock_hetatm_selection(self, receptor_obj, components, flow_label="docking", cache_dict=None,
                                      allow_cached_auto=False):
        raise AssertionError("the assistant must answer the chooser")

    def _show_vs_selected_hit_preview(self, zoom=True):
        pass

    def _on_run_virtual_screening(self):
        comps = [{"resn": "HEM", "category": "cofactor"}, {"resn": "PEG", "category": "cofactor"},
                 {"resn": "SO4", "category": "cofactor"}, {"resn": "MG", "category": "ion"}]
        pick = self._prompt_dock_hetatm_selection("prot", comps, flow_label="virtual_screening")
        self.picked = sorted(c["resn"] for c in pick["selected"])
        self._vs_results = [{"rank": 1, "name": "cpd_1", "affinity": -8.0, "index": 1, "status": "ok"}]


def test_virtual_screening_keep_hetero_answers_the_chooser(app, tmp_path):
    from pymol import cmd

    cmd.reinitialize()
    cmd.fab("GASG", "prot")
    d = _ScreenDlg(str(tmp_path))
    ctx = AgentContext(d)
    job = build_registry(ctx).call("run_virtual_screening", {
        "compounds": [{"smiles": "CCO", "name": "ethanol"}], "receptor": "prot", "keep_hetero": ["so4"]}, ctx)
    assert _headless._state["keep_hetero"] == {"SO4"}          # held for the queued run
    assert "prot" not in d._dock_hetatm_selection_cache         # a stale GUI pick is not reused
    for _ in range(50):
        app.processEvents()
        status = ctx.jobs.status(job["job_id"])
        if status["status"] != "running":
            break
    assert status["status"] == "done", status
    assert d.picked == ["HEM", "MG", "SO4"]                       # PEG (not asked) removed
    assert status["result"]["kept_hetero"] == ["SO4"]
    assert _headless._state["keep_hetero"] == set()             # cleared when the job ended
