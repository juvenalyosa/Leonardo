"""Hardware planning, drug-likeness filters, result sanity checks and the Monte Carlo ensemble sampler."""
import math

import numpy as np
import pytest

from leonardo import druglike_runtime as dl
from leonardo import hardware_runtime as hw
from leonardo import mc_ensemble_runtime as mc
from leonardo.agent.checks import post_checks


def test_gpu_platform_only_when_really_faster():
    prof = {"openmm_platforms": ["Reference", "CPU", "OpenCL"],
            "openmm_benchmark": {"CPU": {"steps_per_s": 3000}, "OpenCL": {"steps_per_s": 3500}}}
    assert hw.pick_md_platform(prof)["name"] == "CPU"                  # < 1.3x: not worth the driver risk
    prof["openmm_benchmark"]["OpenCL"]["steps_per_s"] = 12000
    pick = hw.pick_md_platform(prof)
    assert pick["name"] == "OpenCL" and pick["is_gpu"] and pick["speedup_vs_cpu"] == 4.0
    prof["openmm_benchmark"]["OpenCL"] = {"error": "crash"}
    assert hw.pick_md_platform(prof)["name"] == "CPU"


def test_md_estimates_scale_and_calibrate(tmp_path, monkeypatch):
    monkeypatch.setattr(hw, "CACHE_PATH", str(tmp_path / "hw.json"))
    prof = {"md_platform": {"name": "CPU", "is_gpu": False}}
    small = hw.estimate_md(3000, 1.0, "implicit", prof=prof)
    big = hw.estimate_md(6000, 1.0, "implicit", prof=prof)
    assert big["minutes"] > 2 * small["minutes"] * 0.9 and "advice" in big
    hw.record_md_speed("OpenCL", "implicit", 3000, 200.0)
    cal = hw._load_cache()["calibration"]
    assert cal["md:GPU:implicit"] == 200.0
    est = hw.estimate_md(3000, 1.0, "implicit", platform_name="OpenCL", prof=hw._load_cache())
    assert est["calibrated"] and abs(est["ns_per_day"] - 200.0) < 1e-6


def test_psi4_estimate_grows_with_basis_and_operation():
    syms = ["C"] * 7 + ["N"] * 2 + ["H"] * 9
    e1 = hw.estimate_psi4(syms, "energy", "b3lyp", "def2-svp", threads=4, prof={})
    e2 = hw.estimate_psi4(syms, "frequencies", "b3lyp", "def2-svp", threads=4, prof={})
    e3 = hw.estimate_psi4(syms, "frequencies", "b3lyp", "def2-tzvp", threads=4, prof={})
    assert e1["minutes"] < e2["minutes"] < e3["minutes"]
    assert hw.recommended_threads({"cpu": {"performance_cores": 4, "physical_cores": 10, "logical_cores": 10,
                                           "ram_gb": 16}})["psi4"] == 4


def test_druglike_rules_and_alerts():
    from rdkit import Chem

    ok = dl.properties(Chem.MolFromSmiles("NC(=[NH2+])c1ccccc1"))
    assert ok["lipinski_violations"] == [] and ok["formal_charge"] == 1
    greasy = dl.properties(Chem.MolFromSmiles("C" * 36 + "(=O)O"))
    assert set(greasy["lipinski_violations"]) == {"MW>500", "cLogP>5"}
    assert dl.passes(greasy, "lipinski")[0] is False and dl.passes(greasy, "none")[0] is True
    pains = dl.properties(Chem.MolFromSmiles("O=C1NC(=S)SC1=Cc1ccccc1"))
    assert pains["pains_alerts"] and dl.passes(pains, "lipinski", exclude_pains=True)[0] is False
    kept, rejected = dl.filter_entries([{"mol": Chem.MolFromSmiles("CCO"), "index": 1},
                                        {"mol": Chem.MolFromSmiles("C" * 36 + "(=O)O"), "index": 2}], "lipinski")
    assert [e["index"] for e in kept] == [1] and "Ro5" in rejected[0]["filter_reason"]


def test_quality_checks_flag_untrustworthy_results():
    assert post_checks("run_psi4", {"operation": "ts"}, {"freq_cm_imag": [1123.0, 250.0]})
    assert not post_checks("run_psi4", {"operation": "ts"}, {"freq_cm_imag": [1123.0, 0.0]})
    assert post_checks("run_psi4", {"operation": "stationary"}, {"freq_cm_imag": [45.0]})
    assert post_checks("run_docking", {}, {"best_score_kcal_mol": -2.1})
    assert post_checks("run_redocking", {}, {"top1_success": "1/4 repeats with RMSD <= 2 A"})
    assert post_checks("minimize_structure", {}, {"energy_kJ_mol": {"start": -100, "end": 50}})
    assert not post_checks("run_docking", {}, {"best_score_kcal_mol": -7.5})


def test_mc_torsion_move_keeps_bond_lengths():
    pos = np.array([[0.0, 0, 0], [0.15, 0, 0], [0.2, 0.14, 0], [0.35, 0.16, 0.05]])
    bonds = [(0, 1), (1, 2), (2, 3)]
    down = mc._downstream([0, 1, 2, 3], bonds, 1, 2)
    assert down == [3]
    new = mc._rotate(pos, down, pos[1], pos[2], math.radians(120))
    assert np.allclose(np.linalg.norm(new[3] - new[2]), np.linalg.norm(pos[3] - pos[2]))
    assert not np.allclose(new[3], pos[3]) and np.allclose(new[:3], pos[:3])


def test_estimated_tools_exist():
    from leonardo.agent.tools import AgentContext, build_registry
    from leonardo.agent.tools.estimates import ESTIMATED_TOOLS

    names = set(build_registry(AgentContext()).names())
    assert set(ESTIMATED_TOOLS) <= names
    assert {"system_info", "estimate_cost", "druglikeness", "generate_mc_ensemble", "set_docking_box",
            "set_flexible_residues", "run_cluster_reaction", "build_reaction_cluster"} <= names


@pytest.mark.skipif(not pytest.importorskip("pdbfixer"), reason="pdbfixer missing")
def test_mc_site_residues_from_ligand(tmp_path):
    pdb = tmp_path / "p.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                   "ATOM      2  CA  GLY A   2      20.000   0.000   0.000  1.00  0.00           C\nEND\n")
    assert mc.site_residues_near(str(pdb), [[1.0, 0, 0]], 5.0) == [("A", "1")]


def test_cluster_selection_is_valid_pymol(tmp_path):
    pymol = pytest.importorskip("pymol")
    from leonardo import psi4_cluster_runtime as pc

    cmd = pymol.cmd
    cmd.reinitialize()
    cmd.load_raw("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                 "ATOM      2  CA  GLY A   2       3.000   0.000   0.000  1.00  0.00           C\n"
                 "HETATM    3  C1  LIG A   9       1.500   0.500   0.000  1.00  0.00           C\nEND\n", "pdb", "3XYZ")
    sel = pc.build_pymol_cluster_selection("3XYZ", "around_selection", "resn LIG", radius=2.5)
    idx = pc.build_pymol_cluster_selection("3XYZ", "around_indices", seed_indices=[3], radius=2.5)
    assert cmd.count_atoms(idx["core_sel"]) == 3
    expr = sel["core_sel"]
    assert cmd.count_atoms(expr) == 3                   # both residues + the ligand, no "Malformed selection"
