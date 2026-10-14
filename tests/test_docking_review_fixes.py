import os
import shutil

import code_v2
import numpy as np
import pytest
from dock_runtime import pdb_atom_element
from smina_runtime import run_smina_repeats

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "redocking", "3ptb_ben")


def _dialog(*names):
    class DummyDialog:
        pass

    for name in names:
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return DummyDialog()


@pytest.mark.parametrize("name,resn,element", [
    (" CA ", "ILE", "C"),      # alpha carbon, not calcium
    ("CA  ", "CA", "CA"),      # calcium ion
    (" NA ", "HEM", "N"),      # heme pyrrole nitrogen, not sodium
    ("ZN  ", "ZN", "ZN"),
    (" HG ", "CYS", "H"),      # cysteine hydrogen, not mercury
    ("CL1 ", "LIG", "CL"),
    ("1HB ", "ALA", "H"),
    ("CA", "ILE", "C"),        # stripped name in a standard residue
    (" Cl ", "UNL", "CL"),     # Meeko ligand PDBQT: element-cased chlorine in column 14 (was read as carbon)
    (" Br ", "UNL", "BR"),
    (" Cl1", "UNL", "CL"),
    (" CL ", "LIG", "C"),      # upper case in column 14 stays the one-letter element
])
def test_pdb_atom_element_follows_column_alignment(name, resn, element):
    assert pdb_atom_element(name, resn) == element


def test_pdbqt_type_repair_keeps_alpha_carbons_and_real_calcium(tmp_path):
    dlg = _dialog(
        "_dock_repair_pdbqt_atom_types", "_dock_normalize_pdbqt_atom_type", "_dock_guess_element_from_atom_name",
        "_dock_pdbqt_type_token", "_dock_element_to_pdbqt_type", "_dock_valid_pdbqt_types",
    )
    rec = tmp_path / "rec.pdbqt"
    shutil.copy(os.path.join(BENCH, "receptor_3ptb_no_ben_no_waters.pdbqt"), rec)
    with open(rec, "a") as fh:
        fh.write("HETATM 9999 CA    CA A 480      10.000  10.000  10.000  1.00  0.00     2.000 Ca\n")

    def types():
        return [ln[77:79].strip() for ln in open(rec) if ln.startswith(("ATOM", "HETATM"))]

    before = types()
    assert dlg._dock_repair_pdbqt_atom_types(str(rec)) == 0
    after = types()
    # was: every alpha carbon retyped Ca; only the structural calcium and the added ion are Ca
    assert after == before and after.count("Ca") == 2


def test_pockets_rank_by_fpocket_score_not_druggability():
    dlg = _dialog("_pocket_rank_score")
    s1 = {"id": 1, "score": 0.373, "druggability_score": 0.016, "hydrophobicity_score": 17.4}
    p3 = {"id": 3, "score": 0.171, "druggability_score": 0.443}
    assert dlg._pocket_rank_score(s1)[0] == pytest.approx(0.373)       # not the hydrophobicity score
    assert dlg._pocket_rank_score(s1)[1] == pytest.approx(0.016)
    assert dlg._pocket_rank_score(s1)[0] > dlg._pocket_rank_score(p3)[0]


def test_repeat_confidence_interval_uses_student_t():
    dlg = _dialog("_dock_repeat_statistics", "_dock_repeat_confidence_label")
    st = dlg._dock_repeat_statistics([-6.0, -7.0, -8.0])
    assert st["ci_low"] == pytest.approx(-9.48, abs=0.01) and st["ci_high"] == pytest.approx(-4.52, abs=0.01)
    two = dlg._dock_repeat_statistics([-6.0, -7.0])
    assert not np.isfinite(two["ci_low"])            # no interval from two values


def _fake_smina(texts, seen):
    def run(cmdline, cwd=None, timeout=0):
        seen.append(list(cmdline))
        out = cmdline[cmdline.index("--out") + 1]
        log = cmdline[cmdline.index("--log") + 1]
        text = texts[min(len(seen) - 1, len(texts) - 1)]
        with open(log, "w") as fh:
            fh.write(text)
        if "within the search space" not in text:
            with open(out, "w") as fh:
                fh.write("MODEL 1\nREMARK VINA RESULT:    -6.1      0.000      0.000\nENDMDL\n")
        return True, text, "", 0
    return run


def test_flexible_docking_retries_with_a_larger_box_and_records_real_seed(tmp_path):
    seen = []
    texts = [
        "Using random seed: 123456\nWARNING: Could not find any conformations completely within the search space.\n",
        "Using random seed: 123456\nmode |   affinity\n1       -6.1       0.000      0.000\n",
    ]
    base = ["smina", "-r", "r.pdbqt", "-l", "l.pdbqt", "--size_x", "10", "--size_y", "12", "--size_z", "14",
            "--flexres", "A:190"]
    res = run_smina_repeats(base, str(tmp_path), 1, 0, _fake_smina(texts, seen),
                            parse_pose_scores=lambda pose, log: [{"mode": 1, "affinity": -6.1}])
    assert len(seen) == 2
    retry = seen[1]
    assert retry[retry.index("--size_x") + 1] == "16.0000" and retry[retry.index("--size_z") + 1] == "20.0000"
    row = res["repeat_rows"][0]
    assert row["ok"] and row["seed"] == 123456          # the seed smina drew, not 0


def test_pubchem_hint_does_not_replace_another_ligand():
    dlg = _dialog("_current_pubchem_smiles_hint", "_smiles_hint_matches_ligand_file")
    ben = os.path.join(BENCH, "ben_input_template.sdf")
    dlg._pubchem_hits = [{"isomeric_smiles": "CC(=O)Oc1ccccc1C(=O)O"}]
    dlg._pubchem_hint_ligand_file = "/downloads/pubchem_cid2244.sdf"
    dlg._docking_ligand_file = ben
    assert dlg._current_pubchem_smiles_hint() == ""                    # hint belongs to another file
    assert dlg._smiles_hint_matches_ligand_file("CC(=O)Oc1ccccc1C(=O)O", ben) is False
    assert dlg._smiles_hint_matches_ligand_file("NC(=N)c1ccccc1", ben) is True


def test_basic_groups_are_protonated_at_ph_7_4():
    from rdkit import Chem

    # _repair_and_protonate now delegates to the pKa module (empirical by default)
    dlg = _dialog("_repair_and_protonate", "_pka_method", "_pka_cache_dir", "_pka_backend_options")
    dlg._append_debug_log = lambda *_a: None
    for smi, charge in (("NC(=N)c1ccccc1", 1), ("CCN", 1), ("c1ccncc1", 0), ("CCC(=O)O", -1)):
        mol = dlg._repair_and_protonate(Chem.MolFromSmiles(smi), target_ph=7.4)
        assert Chem.GetFormalCharge(mol) == charge, smi
    assert dlg._last_protonation_info["method"] == "empirical"


def test_dimorphite_fallback_path_still_protonates():
    pytest.importorskip("dimorphite_dl")
    from rdkit import Chem

    dlg = _dialog("_repair_and_protonate_dimorphite", "_try_dimorphite_protonate_smiles")
    for smi, charge in (("NC(=N)c1ccccc1", 1), ("CCN", 1), ("c1ccncc1", 0), ("CCC(=O)O", -1)):
        mol = dlg._repair_and_protonate_dimorphite(Chem.MolFromSmiles(smi), target_ph=7.4)
        assert Chem.GetFormalCharge(mol) == charge, smi


def test_amidine_fallback_without_dimorphite():
    from rdkit import Chem

    dlg = _dialog("_repair_and_protonate_dimorphite")
    dlg._try_dimorphite_protonate_smiles = lambda smiles, ph: None
    for smi, charge in (("NC(=N)c1ccccc1", 1), ("CC(=O)N", 0), ("NC(=NO)c1ccccc1", 0)):
        assert Chem.GetFormalCharge(dlg._repair_and_protonate_dimorphite(Chem.MolFromSmiles(smi))) == charge, smi


def test_torsion_qc_checks_rigid_ligands(tmp_path):
    dlg = _dialog("_dock_pdbqt_torsion_qc", "_dock_read_pdbqt_torsdof")
    f = tmp_path / "rigid.pdbqt"
    f.write_text("ROOT\nATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 A\nENDROOT\nTORSDOF 5\n")
    ok, diag = dlg._dock_pdbqt_torsion_qc(str(f), {"rotb": 0, "aromatic_bonds": 6})
    assert not ok and diag["reason"] == "torsion_exceeds_expected"   # was skipped as "no reference"


def test_receptor_export_keeps_one_altloc(tmp_path):
    from pymol import cmd

    src = tmp_path / "alt.pdb"
    src.write_text(
        "ATOM      1  N   SER A 195       0.000  14.000  22.000  1.00 10.00           N\n"
        "ATOM      2  CA  SER A 195       1.000  14.000  22.000  1.00 10.00           C\n"
        "ATOM      3  OG ASER A 195       0.285  14.863  22.282  0.60 14.42           O\n"
        "ATOM      4  OG BSER A 195       1.285  14.863  22.282  0.40 14.42           O\n"
    )
    cmd.delete("altrec")
    cmd.load(str(src), "altrec")
    dlg = _dialog("_export_receptor_for_external_tools")
    dlg._pymol_model_selection = lambda obj, wrap=True: f"({obj})"
    dlg._resolve_receptor_export_state = lambda obj: 1
    out = tmp_path / "out.pdb"
    assert dlg._export_receptor_for_external_tools("altrec", str(out))
    atoms = [ln for ln in open(out) if ln.startswith("ATOM")]
    assert [ln[12:16].strip() for ln in atoms].count("OG") == 1 and all(ln[16] == " " for ln in atoms)
    cmd.delete("altrec")


def test_docking_ligand_input_is_reprepared_not_the_last_runs_pdbqt(tmp_path):
    # a docking run stores its prepared PDBQT (_docking_ligand_pdbqt); the next run used it as input,
    # so a new docking pH or PDBQT charge mode never reached the ligand
    src = tmp_path / "lig.sdf"
    src.write_text("x")
    last_run = tmp_path / "smina_job1" / "ligand_prepared.pdbqt"
    last_run.parent.mkdir()
    last_run.write_text("x")
    dlg = _dialog("_resolve_docking_ligand_input")
    dlg._docking_ligand_file = str(src)
    dlg._docking_ligand_pdbqt = str(last_run)
    assert dlg._resolve_docking_ligand_input() == (str(src), "source")
    # an explicitly exported PDBQT (Ligand Tools) is still the input
    exported = tmp_path / "ligprep.pdbqt"
    exported.write_text("x")
    dlg._ligand_source_prep_pdbqt = str(exported)
    assert dlg._resolve_docking_ligand_input() == (str(exported), "source_prep_pdbqt")
