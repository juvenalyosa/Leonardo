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


def test_no_ligand_fields_clears_the_stale_ligand(cplx):
    dlg = _Dlg()
    _run(dlg)
    assert dlg.seen == ("", "", "")          # Leonardo auto-detects in this object, not the old UNL Z1


def test_negative_residue_number(cplx):
    cplx.pseudoatom("cplx", name="O1", resn="ACT", resi="-5", chain="C", elem="O", pos=[7.0, 0.0, 0.0], hetatm=1)
    dlg = _Dlg()
    _run(dlg, ligand_resi="-5")
    assert dlg.seen == ("C", "-5", "ACT")
