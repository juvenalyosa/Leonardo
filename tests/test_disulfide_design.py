"""Disulfide design: candidates, geometry filters, rigidity model, and one real refined design (protein G B1)."""
import os

import numpy as np
import pytest

from leonardo import disulfide_runtime as D

PGA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "films",
                   "06_protein_stability_design", "1PGA.pdb")


def test_candidates_have_disulfide_compatible_geometry_and_exclude_gly_pro():
    cands = D.find_candidates(PGA, max_candidates=50)
    assert cands and any(c["label"] == "K10C/E56C" for c in cands)
    for c in cands:
        assert D.CA_RANGE[0] <= c["ca_ca_A"] <= D.CA_RANGE[1] and D.CB_RANGE[0] <= c["cb_cb_A"] <= D.CB_RANGE[1]
        assert not {r[2] for r in c["residues"]} & {"GLY", "PRO"}
        (ch1, i1, _), (ch2, i2, _) = c["residues"]
        assert ch1 != ch2 or abs(int(i1) - int(i2)) >= 3
    scores = [c["geometry_score"] for c in cands]
    assert scores == sorted(scores)


def test_candidates_skip_bonded_cysteines_and_insertion_codes():
    # trypsin 3PTB: six native disulfides and chymotrypsin numbering (184A, 188A, 221A). Before, the candidates
    # included L163C/C182C (C182 is bonded to C168: the build fails, "1 S atom too many") and A221A pairs
    # (PDBFixer cannot address 221A: "invalid literal for int()"; with 221 next to it the number is ambiguous)
    trypsin = os.path.join(os.path.dirname(PGA), "..", "01_molecular_docking", "3PTB.pdb")
    cands = D.find_candidates(trypsin, max_candidates=500)
    assert len(cands) > 20
    used = {(c, i) for cand in cands for c, i, _n in cand["residues"]}
    for bonded in ("22", "157", "42", "58", "128", "232", "136", "201", "168", "182", "191", "220"):
        assert ("A", bonded) not in used
    assert not used & {("A", "221A"), ("A", "221"), ("A", "184"), ("A", "184A"), ("A", "188"), ("A", "188A")}


def test_filters_follow_natural_disulfide_geometry():
    good = {"ss_A": 2.05, "css_deg": [104.0, 105.0], "chi3_deg": -88.0, "ca_rmsd_A": 0.2}
    assert D.passes(good)[0]
    for change in ({"ss_A": 2.3}, {"css_deg": [85.0, 105.0]}, {"chi3_deg": 30.0}, {"ca_rmsd_A": 0.9}):
        ok, checks = D.passes(dict(good, **change))
        assert not ok and not all(checks.values())


def test_a_crosslink_reduces_the_relative_motion_of_the_pair():
    t = np.linspace(0, 6 * np.pi, 40)
    ca = np.c_[2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * t]          # a helix-like chain, 3.8 A steps
    out = D.gnm_rigidity(ca, (5, 20))
    assert out["pair_relative_motion_change_pct"] < 0 and out["mean_fluctuation_change_pct"] <= 0
    assert len(out["per_residue_fluctuation_ratio"]) == 40


def test_k10c_e56c_refines_to_a_good_disulfide(tmp_path):
    pytest.importorskip("pdbfixer")
    geom = D.build_design(PGA, [["A", "10", "LYS"], ["A", "56", "GLU"]], str(tmp_path / "k10c_e56c.pdb"))
    ok, _checks = D.passes(geom)
    assert ok and 1.95 <= geom["ss_A"] <= 2.15 and 65 <= abs(geom["chi3_deg"]) <= 115 and geom["ca_rmsd_A"] < 0.7
    text = open(tmp_path / "k10c_e56c.pdb").read()
    assert "CYX" in text or "CONECT" in text


def test_tools_expose_disulfide_design_and_the_reaction_path_operation():
    from leonardo.agent.registry import ToolRegistry
    from leonardo.agent.tools import design, qmmm

    reg = ToolRegistry()
    design.register(reg)
    qmmm.register(reg)
    specs = {s.name: s for s in reg.specs()}
    assert specs["design_disulfides"].permission == "compute_heavy"
    props = specs["run_qmmm"].parameters["properties"]
    assert "reaction_path" in props["operation"]["enum"]
    assert props["form_bonds"]["type"] == "array" and props["break_bonds"]["type"] == "array"
