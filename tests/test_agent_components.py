"""Assistant tools for the component library (resolve parameters, build a validated system)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
pytest.importorskip("networkx")

from leonardo.agent.registry import ToolError  # noqa: E402
from leonardo.agent.tools import AgentContext, build_registry  # noqa: E402

HEME_SITE = os.path.join(os.path.dirname(__file__), "data", "components", "1A6M_heme_site.pdb")


@pytest.fixture()
def ctx(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymol import cmd

    cmd.reinitialize()
    cmd.load(HEME_SITE, "mb")
    c = AgentContext(dlg=None)

    def ws(sub=""):
        p = tmp_path / sub
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    c.workspace = ws
    del app
    return c


def test_resolve_components_recommends_charmm_for_heme(ctx):
    reg = build_registry(ctx)
    out = reg.call("resolve_components", {"object": "mb", "family": "amber14"}, ctx)
    assert out["recommended_family"] == "charmm36" and not out["ready_for_md"]
    heme = [r for r in out["components"] if r["resn"] == "HEM"][0]
    assert heme["source"] == "metal" and heme["available_in"]["template"] == "HEME"
    out = reg.call("resolve_components", {"object": "mb", "family": "charmm36"}, ctx)
    assert out["ready_for_md"] and "HEM→HEME" in out["_card"]


def test_build_validated_system_loads_the_minimized_model(ctx):
    from pymol import cmd

    reg = build_registry(ctx)
    rep = reg.call("build_validated_system", {"object": "mb", "family": "charmm36", "minimize_steps": 50}, ctx)
    assert rep["ok"] and rep["component_charges"]["HEM A154"] == -2.0
    assert rep["object"] in cmd.get_names("objects")
    assert cmd.count_atoms(f"{rep['object']} and resn HEM* and elem H") == 30
    with pytest.raises(ToolError, match="recommended family: charmm36"):
        reg.call("build_validated_system", {"object": "mb", "family": "amber14", "minimize_steps": 10}, ctx)


def test_md_guard_refuses_to_drop_the_heme_in_amber(ctx):
    from pymol import cmd

    from leonardo.agent.tools.md import _component_guard

    with pytest.raises(ToolError, match="HEM \\(metal\\).*force_field='charmm36'"):
        _component_guard(ctx, "mb", "amber14")
    assert _component_guard(ctx, "mb", "amber14", allow_incomplete=True) is None
    assert _component_guard(ctx, "mb", "charmm36")["ready"]
    assert cmd.get("pdb_conect_all") in ("off", "0")                 # session setting restored
