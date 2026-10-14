"""Assistant tools driven end to end on new inputs (1UBQ, cytochrome c): loaders, gap repair, design,
sequence analysis, charge/electron count, time estimates and structural alignment."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
cmd = pytest.importorskip("pymol").cmd

from leonardo.agent.registry import ToolError, ToolRegistry  # noqa: E402
from leonardo.agent.tools import AgentContext, bioseq, design, estimates, extras, modeling, structure  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "components")
UBQ = os.path.join(DATA, "1UBQ.pdb")
UBQ_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


class _Ctx(AgentContext):
    def __init__(self, dlg=None, root=""):
        super().__init__(dlg)
        self._root = root

    def workspace(self, sub=""):
        path = os.path.join(self._root, sub)
        os.makedirs(path, exist_ok=True)
        return path

    def scene(self, caption):
        return ""


def _reg(*modules):
    reg = ToolRegistry()
    for m in modules:
        m.register(reg)
    return reg


@pytest.fixture(scope="module")
def dlg():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2

    d = code_v2.ProtLigInteractDialog()
    app.processEvents()
    d._save_settings = lambda *a, **k: None
    yield d
    d.close()


# --------------------------------------------------------------------------- loaders
def test_loading_a_file_twice_makes_a_second_object_not_a_second_state(tmp_path):
    """load_structure_file into an existing name appended the file as state 2 of the user's object."""
    cmd.reinitialize()
    reg = _reg(structure)
    first = reg.call("load_structure_file", {"path": UBQ, "object_name": "ubq"}, _Ctx(root=str(tmp_path)))
    second = reg.call("load_structure_file", {"path": UBQ, "object_name": "ubq"}, _Ctx(root=str(tmp_path)))
    assert first["object"] == "ubq" and second["object"] == "ubq_2"
    assert cmd.count_states("ubq") == 1 and cmd.count_states("ubq_2") == 1
    assert second["states"] == 1


def test_fetching_the_same_entry_twice_works(tmp_path, monkeypatch):
    """A second fetch_structure of an entry failed (mmCIF cannot be loaded into an existing object), so a saved
    pipeline could not be re-run in the same session."""
    from leonardo import structure_fetch_runtime as sf

    cmd.reinitialize()
    cmd.load(UBQ, "_src")
    cif = str(tmp_path / "rcsb_1UBQ.cif")
    cmd.save(cif, "_src")
    cmd.delete("_src")
    monkeypatch.setattr(sf, "download_structure_file", lambda request, out_dir: {"path": cif})
    reg = _reg(structure)
    a = reg.call("fetch_structure", {"source": "rcsb", "identifier": "1UBQ"}, _Ctx(root=str(tmp_path)))
    b = reg.call("fetch_structure", {"source": "rcsb", "identifier": "1UBQ"}, _Ctx(root=str(tmp_path)))
    assert (a["object"], b["object"]) == ("rcsb_1UBQ", "rcsb_1UBQ_2")
    assert cmd.count_states("rcsb_1UBQ") == 1 and cmd.count_atoms("rcsb_1UBQ_2") == cmd.count_atoms("rcsb_1UBQ")


# --------------------------------------------------------------------------- gap repair
def _ubq_without(sel):
    cmd.reinitialize()
    cmd.load(UBQ, "_ubq")
    cmd.create("ubq_cut", f"_ubq and polymer and not ({sel})")
    cmd.delete("_ubq")


def test_gap_repair_counts_terminal_residues_once(dlg, tmp_path, monkeypatch):
    """With fill_termini the tool added the terminal residues twice to residues_filled (9 built -> 12 reported)."""
    _ubq_without("resi 1-3 or resi 30-35")
    calls = {}
    monkeypatch.setattr(modeling, "_start_modeller", lambda ctx, *a, **k: calls.update(args=a, kw=k) or {"job": 1})
    _reg(modeling).call("repair_structure_gaps", {"object": "ubq_cut", "full_sequence": UBQ_SEQ,
                                                  "fill_termini": True}, _Ctx(dlg, str(tmp_path)))
    extra = calls["kw"]["extra"]
    assert extra["residues_filled"] == 9
    assert extra["terminal_residues_added"] == {"n_term": 3, "c_term": 0}
    name, seq = calls["args"][0], calls["args"][1]
    assert name == "ubq_cut_repaired" and seq == UBQ_SEQ


def test_gap_repair_without_internal_gap_reports_missing_termini(dlg, tmp_path, monkeypatch):
    """Only termini missing and fill_termini=false: the tool said the chain had no missing residues."""
    _ubq_without("resi 1-3 or resi 75-76")
    monkeypatch.setattr(modeling, "_start_modeller", lambda *a, **k: pytest.fail("nothing to build"))
    out = _reg(modeling).call("repair_structure_gaps", {"object": "ubq_cut", "full_sequence": UBQ_SEQ},
                              _Ctx(dlg, str(tmp_path)))
    assert out["filled"] == 0
    assert out["missing_termini"] == {"n_term": 3, "c_term": 2}
    assert "fill_termini" in out["_card"]


# --------------------------------------------------------------------------- design / sequence
def test_disulfide_design_rejects_a_chain_the_protein_does_not_have(tmp_path):
    """chain='Z' ran the design on no residues and reported 'no compatible geometry' as a result."""
    cmd.reinitialize()
    cmd.load(UBQ, "ubq")
    ctx = _Ctx(root=str(tmp_path))
    with pytest.raises(ToolError, match="chain 'Z'"):
        _reg(design).call("design_disulfides", {"object": "ubq", "chain": "Z", "max_candidates": 1}, ctx)
    assert ctx.jobs.list() == []


def test_sequence_analysis_names_a_missing_chain():
    cmd.reinitialize()
    cmd.load(UBQ, "ubq")
    with pytest.raises(ToolError, match="no protein chain 'Z'"):
        bioseq._sequence_from(object="ubq", chain="Z")
    assert bioseq._sequence_from(object="ubq", chain="A") == UBQ_SEQ


# --------------------------------------------------------------------------- QM helpers / estimates / align
def test_electron_count_knows_metals():
    """Fe, Zn... were counted as carbon (cytochrome c heme: 270 electrons reported instead of 290)."""
    cmd.reinitialize()
    cmd.load(os.path.join(DATA, "1HRC_full_nowat.pdb"), "cytc")
    atoms = [{"element": a.symbol} for a in cmd.get_model("cytc and resn HEC and not hydro").atom]
    assert extras.electron_count(atoms, 0) == 34 * 6 + 4 * 7 + 4 * 8 + 26
    assert extras.electron_count([{"element": "ZN"}, {"element": "Mg"}, {"element": "Se"}], 2) == 30 + 12 + 34 - 2
    assert extras.electron_count([{"element": "Xx"}], 0) is None


def test_modeller_estimate_uses_the_size_of_the_structure():
    """model_mutant / repair_structure_gaps have no sequence argument: every estimate assumed 300 residues."""
    cmd.reinitialize()
    cmd.load(UBQ, "ubq")
    ctx = AgentContext()
    est = estimates.estimate(ctx, "model_mutant", {"object": "ubq", "num_models": 3})
    assert est["minutes"] == round(0.2 + 3 * 76 * 0.0007, 1)
    assert "seconds" in estimates.estimate(ctx, "model_mutant", {"object": "ubq", "method": "local"})["text"]
    gap = estimates.estimate(ctx, "repair_structure_gaps", {"object": "ubq", "chain": "A", "num_models": 10})
    assert gap["minutes"] == round(0.2 + 10 * 76 * 0.0007, 1)


def test_align_on_itself_is_a_clear_error():
    cmd.reinitialize()
    cmd.load(UBQ, "ubq")
    with pytest.raises(ToolError, match="both in ubq"):
        _reg(modeling).call("align_structures", {"mobile": "ubq", "target": "ubq and resi 1-40"}, _Ctx())
