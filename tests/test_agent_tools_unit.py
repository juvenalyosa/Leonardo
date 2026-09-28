"""Unit checks of assistant tool helpers and the OpenMM implicit-solvent force field fix."""
import code_v2

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
