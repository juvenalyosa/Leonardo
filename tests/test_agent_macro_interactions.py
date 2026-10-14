"""Assistant tool analyze_interface on PyMOL objects (barnase-barstar, Zif268-DNA)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets

from leonardo.agent.registry import ToolError  # noqa: E402
from leonardo.agent.tools import AgentContext, build_registry  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "macro")


@pytest.fixture()
def env(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymol import cmd

    import leonardo.agent.presentation as pres

    cmd.reinitialize()
    cmd.load(os.path.join(DATA, "1BRS_AD.pdb"), "brs")
    cmd.load(os.path.join(DATA, "1AAY.pdb"), "zif")
    ctx = AgentContext(dlg=None)
    ctx.workspace = lambda sub="": str(tmp_path)
    original = pres.save_step_scene
    pres.save_step_scene = lambda *a, **k: {"scene": ""}
    yield ctx, build_registry(ctx), cmd
    pres.save_step_scene = original
    del app


def test_permission_and_registration(env):
    _ctx, reg, _cmd = env
    spec = reg.get("analyze_interface")
    assert spec is not None and spec.permission == "compute_light"
    assert spec.parameters["required"] == ["object"]


def test_default_chains_card_and_objects(env):
    ctx, reg, cmd = env
    out = reg.call("analyze_interface", {"object": "brs"}, ctx)
    assert out["interface_type"] == "protein–protein"
    assert out["sides"]["A"]["chains"] == ["A"] and out["sides"]["B"]["chains"] == ["D"]
    assert out["counts"]["Salt Bridge"] >= 3 and out["counts"]["Hydrogen Bond"] >= 10
    assert out["buried_sasa"]["method"] == "shrake_rupley_numpy" and 1400 < out["buried_sasa"]["buried_total"] < 1800
    assert "ASP39:D" in out["interface_residues"]["B"] and "Buried SASA" in out["_card"]
    sels = out["objects"]["selections"]
    assert cmd.count_atoms(sels["both"]) > 0 and cmd.count_atoms(sels["side_b"]) > 0
    assert "iface_brs_hb" in out["objects"]["dash_objects"] and "iface_brs_hb" in cmd.get_names("objects")
    assert any(r["type"] == "Salt Bridge" and r["a"] == "ARG83:A NH2" and r["b"] == "ASP39:D OD1"
               for r in out["top_contacts"])
    # a second run replaces, not duplicates, its objects
    again = reg.call("analyze_interface", {"object": "brs"}, ctx)
    assert again["objects"]["dash_objects"] == out["objects"]["dash_objects"]


def test_pymol_selections_and_pymol_area(env):
    ctx, reg, _cmd = env
    out = reg.call("analyze_interface", {"object": "brs", "selection_a": "chain A and resi 83+87",
                                          "selection_b": "D", "sasa_method": "pymol", "show": False}, ctx)
    assert set(out["interface_residues"]["A"]) == {"ARG83:A", "ARG87:A"}
    assert out["buried_sasa"]["method"] == "pymol_get_area" and out["buried_sasa"]["buried_total"] > 50
    assert out["objects"] == {}


def test_protein_dna_default_and_readout(env):
    ctx, reg, _cmd = env
    out = reg.call("analyze_interface", {"object": "zif", "include_waters": False, "show": False}, ctx)
    assert out["interface_type"] == "protein–DNA" and "Water Bridge" not in out["counts"]
    na = out["nucleic_acid"]
    assert "ARG146:A" in na["residues_by_readout"]["base readout (major groove)"]
    assert "HIS125:A" in na["residues_by_readout"]["phosphate backbone"]
    assert "base-specific" in out["_card"]


def test_errors_are_tool_errors(env):
    ctx, reg, _cmd = env
    with pytest.raises(ToolError, match="no object"):
        reg.call("analyze_interface", {"object": "nope"}, ctx)
    with pytest.raises(ToolError, match="matches no atom"):
        reg.call("analyze_interface", {"object": "brs", "selection_a": "resn XYZ", "selection_b": "D"}, ctx)
    with pytest.raises(ToolError, match="share"):
        reg.call("analyze_interface", {"object": "brs", "selection_a": "A", "selection_b": "A+D"}, ctx)
