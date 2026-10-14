"""analyze_interactions(object=...): the ligand residue the caller names is the one Leonardo analyzes."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pymol")

from leonardo.agent.registry import ToolError  # noqa: E402
from leonardo.agent.tools import AgentContext, build_registry  # noqa: E402


class _Field:
    def __init__(self, text=""):
        self._t = str(text)

    def setText(self, t):
        self._t = str(t)

    def text(self):
        return self._t


class _Dlg:
    def __init__(self):
        # the previous analysis (another complex) left its ligand in the fields
        self.ligand_chain, self.ligand_resi, self.ligand_resn = _Field("Z"), _Field("1"), _Field("UNL")
        self.ligand_info = ("Z", "1", "UNL")
        self.loaded_object = ""
        self.seen = None

    def _on_calc_interactions(self):
        self.seen = (self.ligand_chain.text(), self.ligand_resi.text(), self.ligand_resn.text())
        self.interactions = [{"type": "Hydrophobic", "prot_res": "ALA A1", "distance": 3.8}]

    def _on_redraw_visible_types(self):
        pass


@pytest.fixture
def cplx(monkeypatch):
    from pymol import cmd

    cmd.delete("all")
    cmd.pseudoatom("cplx", name="CA", resn="ALA", resi="1", chain="A", elem="C", pos=[0.0, 0.0, 0.0])
    cmd.pseudoatom("cplx", name="FE", resn="HEM", resi="500", chain="A", elem="Fe", pos=[3.0, 0.0, 0.0], hetatm=1)
    cmd.pseudoatom("cplx", name="C1", resn="MK1", resi="902", chain="B", elem="C", pos=[5.0, 0.0, 0.0], hetatm=1)
    monkeypatch.setattr("leonardo.agent.presentation.save_step_scene", lambda *a, **k: {"scene": "s"})
    yield cmd
    cmd.delete("all")


def _run(dlg, **args):
    ctx = AgentContext(dlg)
    return build_registry(ctx).call("analyze_interactions", dict(args, object="cplx"), ctx)


def test_resn_alone_selects_that_residue(cplx):
    # with resi blank Leonardo auto-detected the ligand and ignored ligand_resn (HEM could win)
    dlg = _Dlg()
    _run(dlg, ligand_resn="MK1")
    assert dlg.seen == ("B", "902", "MK1")


def test_resi_alone_and_unknown_residue(cplx):
    dlg = _Dlg()
    _run(dlg, ligand_chain="A", ligand_resi="500")
    assert dlg.seen == ("A", "500", "HEM")
    with pytest.raises(ToolError, match="no residue"):
        _run(_Dlg(), ligand_resn="XYZ")
    with pytest.raises(ToolError, match="ligand_chain alone"):        # was ignored (auto-detected ligand)
        _run(_Dlg(), ligand_chain="B")


def test_no_ligand_fields_clears_the_stale_ligand(cplx):
    dlg = _Dlg()
    _run(dlg)
    assert dlg.seen == ("", "", "")          # Leonardo auto-detects in this object, not the old UNL Z1


def test_negative_residue_number(cplx):
    cplx.pseudoatom("cplx", name="O1", resn="ACT", resi="-5", chain="C", elem="O", pos=[7.0, 0.0, 0.0], hetatm=1)
    dlg = _Dlg()
    _run(dlg, ligand_resi="-5")
    assert dlg.seen == ("C", "-5", "ACT")


def test_view_and_label_use_the_analyzed_copy_only(cplx):
    """Two copies of a ligand with the same resn/resi in chains E and D: the contacts drawn are those of the
    copy analyzed (the view used 'resn MK1 and resi 902', every copy in any chain), and the label names it."""
    for chain, dx in (("E", 20.0), ("D", 40.0)):
        cplx.fab("AGA", f"pep{chain}", chain=chain)
        cplx.translate([dx, 0.0, 0.0], f"pep{chain}")
        x, y, z = cplx.get_coords(f"pep{chain} and resi 2 and name CA")[0]
        cplx.pseudoatom(f"pep{chain}", name="C1", resn="MK1", resi="902", chain=chain, elem="C",
                        pos=[float(x) + 2.0, float(y), float(z)], hetatm=1)
    cplx.create("both", "cplx or pepE or pepD")
    cplx.delete("cplx or pepE or pepD")
    cplx.set_name("both", "cplx")

    class _Calc(_Dlg):
        def _on_calc_interactions(self):
            super()._on_calc_interactions()
            self.ligand_info = self.seen

    dlg = _Calc()
    out = _run(dlg, ligand_chain="E", ligand_resn="MK1")
    assert dlg.seen == ("E", "902", "MK1")
    space = {"chains": set()}
    cplx.iterate("ai_contacts", "chains.add(chain)", space=space)
    assert space["chains"] == {"E"}                  # not chain D's site around the other copy
    assert out["complex"] == "cplx MK1 E902"

    class _Auto(_Dlg):
        def _on_calc_interactions(self):
            super()._on_calc_interactions()
            self.ligand_info = ("D", "902", "MK1")      # what Leonardo's auto-detection picked

    out = _run(_Auto())
    assert out["complex"] == "cplx MK1 D902"          # was "cplx " (no ligand named)
