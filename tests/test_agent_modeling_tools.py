"""Protein-modeling tools of the assistant: sequence cleaning, registration and structural comparison."""
import pytest

cmd = pytest.importorskip("pymol").cmd

from leonardo.agent.registry import ToolError, ToolRegistry  # noqa: E402
from leonardo.agent.tools import AgentContext, build_registry, modeling  # noqa: E402

_HELIX = "\n".join(
    f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
    for i, (x, y, z) in enumerate([(1.5 * k, 2.3 * ((k % 4) - 1.5), 1.0 * (k % 3)) for k in range(20)])
)


class _Ctx(AgentContext):
    def scene(self, caption):
        return ""


def _reg():
    reg = ToolRegistry()
    modeling.register(reg)
    return reg


def test_modeling_tools_are_registered():
    names = set(build_registry(AgentContext()).names())
    assert {"get_uniprot_sequence", "search_templates", "build_homology_model", "align_structures",
            "minimize_structure"} <= names


def test_clean_sequence_accepts_fasta_and_rejects_non_amino_acids():
    assert modeling._clean_sequence(">sp|P00533|EGFR\nMRPS GTAG\nALLA\n") == "MRPSGTAGALLA"
    with pytest.raises(ToolError):
        modeling._clean_sequence("MRPSB1")


def test_build_rejects_bad_template_codes():
    with pytest.raises(ToolError):
        _reg().call("build_homology_model", {"sequence": "A" * 30, "templates": ["1m17"], "name": "m"}, _Ctx())


def test_bioseq_object_sequence_does_not_fuse_chains():
    """analyze_sequence on a multi-chain object analyzed chains A and B joined as one protein."""
    from leonardo.agent.tools import bioseq

    cmd.reinitialize()
    cmd.fab("ACDEFGHIKL", "two", chain="A")
    cmd.fab("MNPQRSTVWY", "_b", chain="B")
    cmd.create("two", "two or _b")
    with pytest.raises(ToolError, match="chains"):
        bioseq._sequence_from(object="two")
    assert bioseq._sequence_from(object="two", chain="B") == "MNPQRSTVWY"
    cmd.fab("ACDEFGHIKLMN", "one", chain="A")
    assert bioseq._sequence_from(object="one") == "ACDEFGHIKLMN"
    cmd.fnab("GATTACA", "_dna", mode="DNA", form="B", dbl_helix=0)   # protein-DNA complex: one protein chain
    cmd.alter("_dna", "chain = 'D'")
    cmd.create("cplx", "one or _dna")
    assert bioseq._sequence_from(object="cplx") == "ACDEFGHIKLMN"


def test_mutant_of_chain_b_is_superposed_on_chain_b(tmp_path, monkeypatch):
    """The model of chain B of a homodimer was fitted on all Cα of the object and landed on chain A."""
    import json
    import os

    import numpy as np

    cmd.reinitialize()
    cmd.load(os.path.join(os.path.dirname(__file__), "data", "components", "1UBQ.pdb"), "ca")
    cmd.remove("ca and not polymer")
    cmd.create("cb", "ca")
    cmd.alter("cb", "chain = 'B'")
    cmd.translate([30, 0, 0], "cb", camera=0)
    cmd.create("dimer", "ca or cb")
    cmd.delete("ca or cb")
    cmd.create("_mdl", "dimer and chain B")         # stands in for MODELLER's model of chain B
    cmd.alter("_mdl", "chain = 'A'")
    cmd.translate([5, 40, -10], "_mdl", camera=0)
    cmd.rotate("y", 70, "_mdl", camera=0)
    run_root = tmp_path / "run"
    (run_root / "pdbs").mkdir(parents=True)
    cmd.save(str(run_root / "pdbs" / "m1.pdb"), "_mdl")
    cmd.delete("_mdl")
    (run_root / "agent_result.json").write_text(json.dumps({"best_model_path": "pdbs/m1.pdb"}))

    class _Proc:
        returncode = 0

        def poll(self):
            return 0

    class _Jobs:
        def start(self, kind, poll, cancel=None, description=""):
            return poll()

    monkeypatch.setattr(modeling.subprocess, "Popen", lambda *a, **k: _Proc())
    ctx = _Ctx()
    ctx.jobs = _Jobs()
    ctx.workspace = lambda sub="": str(tmp_path)
    out = modeling._start_modeller(ctx, "mut", "A" * 76, ["sdimB"], run_root=str(run_root), align_to="dimer",
                                   align_chain="B")
    assert out["done"] and out["result"]["rmsd_to_source_A"] < 0.1
    centre = np.array(cmd.get_coords("mut and name CA")).mean(0)
    chain_b = np.array(cmd.get_coords("dimer and chain B and name CA")).mean(0)
    assert np.linalg.norm(centre - chain_b) < 0.5


def test_align_reports_moved_residues_and_colors_them():
    cmd.reinitialize()
    cmd.read_pdbstr(_HELIX, "ref")
    cmd.read_pdbstr(_HELIX, "mob")
    cmd.alter_state(1, "mob and resi 18-20", "x = x + 4.0")      # a moved tail
    cmd.translate([10, 0, 0], "mob", camera=0)                   # plus a rigid offset the fit removes
    out = _reg().call("align_structures", {"mobile": "mob", "target": "ref", "color_threshold_A": 2.0}, _Ctx())
    assert set(out["above_threshold"]) == {"ALA18", "ALA19", "ALA20"}
    assert out["largest_displacements"][0]["d_ca_A"] == pytest.approx(4.0, abs=0.3)
    assert out["residues_compared"] == 20
    colors = {}
    cmd.iterate("ref and resi 20 and name CA", "colors['c'] = color", space={"colors": colors})
    assert colors["c"] == cmd.get_color_index("red")
