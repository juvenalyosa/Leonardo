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
