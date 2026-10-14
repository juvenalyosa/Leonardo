"""Unit checks of assistant tool helpers and the OpenMM implicit-solvent force field fix."""
import code_v2
import pytest

from leonardo.agent.tools import _headless


def _dummy(*names):
    class D:
        pass
    for n in names:
        setattr(D, n, getattr(code_v2.ProtLigInteractDialog, n))
    return D()


def test_implicit_water_uses_obc2_xml_and_no_charmm():
    dlg = _dummy("_openmm_forcefield_attempts")
    implicit = dlg._openmm_forcefield_attempts("amber14", "implicit")
    assert all("implicit/obc2.xml" in files for _label, files in implicit)
    assert not any(label.startswith("CHARMM") for label, _files in implicit)
    explicit = dlg._openmm_forcefield_attempts("amber14", "explicit")
    assert not any("implicit/obc2.xml" in files for _label, files in explicit)
    assert explicit[0][1] == ["amber14-all.xml", "amber14/tip3p.xml"]


def test_unattended_hetatm_choice_drops_the_crystal_ligand():
    class D:
        _dock_hetatm_selection_cache = {}

        def _prompt_dock_hetatm_selection(self, rec, comps, **kw):
            raise AssertionError("dialog must not open while the assistant works")

        def _dock_hetatm_component_cache_key(self, c):
            return c["resn"]
    d = D()
    _headless.auto_hetatm_selection(d)
    comps = [{"resn": "BEN", "category": "cofactor", "chain": "A", "resi": "1"},
             {"resn": "CA", "category": "ion", "chain": "A", "resi": "480"},
             {"resn": "HEM", "category": "cofactor", "chain": "A", "resi": "500"},
             {"resn": "HOH", "category": "water", "chain": "A", "resi": "600"}]
    _headless.begin()
    try:
        _headless.set_keep_hetero([])
        out = d._prompt_dock_hetatm_selection("rec", comps)
        assert [c["resn"] for c in out["selected"]] == ["CA", "HEM"]
        _headless.set_keep_hetero(["BEN"])
        out = d._prompt_dock_hetatm_selection("rec", comps)
        assert [c["resn"] for c in out["selected"]] == ["BEN", "CA", "HEM"]
        msgs = _headless.take_messages()
        assert any("removed: BEN A1" in m["text"] for m in msgs)
    finally:
        _headless.set_keep_hetero([])
        _headless.end()


def test_vs_library_keeps_compound_names(tmp_path):
    from rdkit import Chem

    path = tmp_path / "lib.sdf"
    with Chem.SDWriter(str(path)) as w:
        for name, smi in (("benzamidine", "NC(=[NH2+])c1ccccc1"), ("phenol", "Oc1ccccc1")):
            m = Chem.MolFromSmiles(smi)
            m.SetProp("_Name", name)
            w.write(m)
    dlg = _dummy("_vs_load_sdf_library", "_select_best_ligand_fragment")
    dlg._vs_library_sdf = str(path)
    entries, info = dlg._vs_load_sdf_library(str(path), 10)
    assert [e["name"] for e in entries] == ["benzamidine", "phenol"]


def test_captured_messages_belong_to_their_own_job():
    from leonardo.agent.tools import _headless

    _headless.reset()
    a = _headless.begin()
    _headless._record({"kind": "critical", "title": "A", "text": "docking failed"})
    b = _headless.begin()
    _headless._record({"kind": "information", "title": "B", "text": "md done"})
    assert [m["title"] for m in _headless.take_messages(b)] == ["B"]
    assert [m["title"] for m in _headless.take_messages(a)] == ["A", "B"]
    _headless.end(a)
    _headless.reset()                           # a reset while b runs...
    c = _headless.begin()
    _headless.end(b)                            # ...then b's late end must not end c
    assert _headless.active()
    _headless.end(c)
    assert not _headless.active()


def test_out_of_range_values_are_reported_not_clamped():
    pytest.importorskip("pymol")
    from pymol.Qt import QtWidgets

    from leonardo.agent.registry import ToolError
    from leonardo.agent.tools._widgets import set_value
    from leonardo.agent.ui.style import pill_style

    # keep a reference: an unreferenced QApplication is collected at once and the next widget aborts Python
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841

    class Dlg:
        spin = QtWidgets.QSpinBox()

    Dlg.spin.setRange(10, 5000)
    with pytest.raises(ToolError):
        set_value(Dlg, "spin", 5)
    set_value(Dlg, "spin", 20)
    assert Dlg.spin.value() == 20
    assert "rgba(31, 58, 110, 0.12)" in pill_style("#1F3A6E")


def test_describe_object_counts_only_its_own_residues_for_a_blank_chain():
    """'chain  and name CA' (blank chain id) selected the alpha carbons of every other object: an MD/SDF
    structure without chain ids was reported with the residues of the whole session."""
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    cmd.fab("ACDEFGHIK", "other", chain="A")
    cmd.fab("GAG", "nochain", chain="A")
    cmd.alter("nochain", "chain=''")
    cmd.pseudoatom("lig", name="C1", resn="UNK", resi="0", elem="C", pos=[30.0, 0.0, 0.0], hetatm=1)
    cmd.alter("lig", "chain=''")
    ctx = AgentContext()
    reg = build_registry(ctx)
    try:
        assert reg.call("describe_object", {"object": "nochain"}, ctx)["chains"] == {"-": 3}
        assert reg.call("describe_object", {"object": "other"}, ctx)["chains"] == {"A": 9}
        assert reg.call("describe_object", {"object": "lig"}, ctx)["chains"] == {"-": 0}
    finally:
        cmd.reinitialize()
