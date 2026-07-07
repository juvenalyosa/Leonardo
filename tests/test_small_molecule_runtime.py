import pytest


rdkit = pytest.importorskip("rdkit")

from rdkit import Chem

from small_molecule_runtime import prepare_small_molecule_working_mol
from small_molecule_runtime import build_small_molecule_prep_qc
from small_molecule_runtime import small_molecule_explicit_hydrogen_count
from small_molecule_runtime import small_molecule_formal_charge
from small_molecule_runtime import summarize_pdbqt_ligand_file
from small_molecule_runtime import summarize_small_molecule_state


def _pick_largest_fragment(mol):
    frags = Chem.GetMolFrags(Chem.Mol(mol), asMols=True, sanitizeFrags=False)
    if not frags:
        return Chem.Mol(mol)
    return max(frags, key=lambda frag: (int(frag.GetNumHeavyAtoms()), int(frag.GetNumAtoms())))


def test_summarize_small_molecule_state_reports_charge_and_hydrogens():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[NH3+]"))
    summary = summarize_small_molecule_state(mol)
    assert summary["ok"] is True
    assert summary["heavy_atoms"] == 2
    assert summary["formal_charge"] == 1
    assert summary["explicit_hydrogens"] > 0


def test_prepare_small_molecule_working_mol_uses_repair_fn_for_missing_hydrogens():
    base = Chem.MolFromSmiles("CN")
    repaired = Chem.AddHs(Chem.MolFromSmiles("C[NH3+]"))

    work, info = prepare_small_molecule_working_mol(
        base,
        include_h=True,
        target_ph=7.4,
        repair_fn=lambda mol, target_ph=7.4: Chem.Mol(repaired),
        select_fragment_fn=_pick_largest_fragment,
    )

    assert info["used_repair"] is True
    assert small_molecule_formal_charge(work) == 1
    assert small_molecule_explicit_hydrogen_count(work) > 0


def test_prepare_small_molecule_working_mol_preserves_existing_hydrogenated_state():
    protonated = Chem.AddHs(Chem.MolFromSmiles("C[NH3+]"))

    def _repair_should_not_run(_mol, target_ph=7.4):
        raise AssertionError("repair_fn should not run when explicit hydrogens are already present")

    work, info = prepare_small_molecule_working_mol(
        protonated,
        include_h=True,
        target_ph=7.4,
        repair_fn=_repair_should_not_run,
        select_fragment_fn=_pick_largest_fragment,
    )

    assert info["used_repair"] is False
    assert small_molecule_formal_charge(work) == 1
    assert small_molecule_explicit_hydrogen_count(work) == small_molecule_explicit_hydrogen_count(protonated)


def test_prepare_small_molecule_working_mol_selects_main_fragment_and_can_strip_hydrogens():
    salt = Chem.AddHs(Chem.MolFromSmiles("C[NH3+].[Cl-]"))

    work, info = prepare_small_molecule_working_mol(
        salt,
        include_h=False,
        target_ph=7.4,
        select_fragment_fn=_pick_largest_fragment,
    )

    assert info["fragment_selected"] is True
    assert work.GetNumHeavyAtoms() == 2
    assert small_molecule_formal_charge(work) == 1
    assert small_molecule_explicit_hydrogen_count(work) == 0


def test_summarize_pdbqt_ligand_file_reports_charge_types_and_torsdof(tmp_path):
    pdbqt = tmp_path / "lig.pdbqt"
    pdbqt.write_text(
        "\n".join(
            [
                "ROOT",
                "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.100 C ",
                "ATOM      2  N   UNL     1       1.200   0.000   0.000  0.00  0.00    -0.100 NA",
                "ATOM      3  H   UNL     1       1.800   0.000   0.000  0.00  0.00    +0.000 HD",
                "ENDROOT",
                "TORSDOF 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_pdbqt_ligand_file(str(pdbqt))

    assert summary["ok"] is True
    assert summary["atoms"] == 3
    assert summary["heavy_atoms"] == 2
    assert summary["hydrogens"] == 1
    assert abs(float(summary["charge_sum"])) < 1.0e-6
    assert summary["has_partial_charges"] is True
    assert summary["atom_types"]["NA"] == 1
    assert summary["torsdof"] == 0


def test_build_small_molecule_prep_qc_flags_heavy_atom_loss_as_red(tmp_path):
    input_mol = Chem.MolFromSmiles("CCN")
    pdbqt = tmp_path / "bad_lig.pdbqt"
    pdbqt.write_text(
        "\n".join(
            [
                "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C ",
                "ATOM      2  N   UNL     1       1.200   0.000   0.000  0.00  0.00    +0.000 N ",
                "TORSDOF 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qc = build_small_molecule_prep_qc(
        input_mol=input_mol,
        input_file="input.sdf",
        prepared_file=str(pdbqt),
        target_ph=7.4,
    )

    assert qc["status"] == "red"
    assert qc["ok"] is False
    assert any("heavy atom count changed" in str(e) for e in qc["errors"])


def test_build_small_molecule_prep_qc_warns_for_charge_shift(tmp_path):
    input_mol = Chem.MolFromSmiles("C[NH3+]")
    pdbqt = tmp_path / "shifted_charge.pdbqt"
    pdbqt.write_text(
        "\n".join(
            [
                "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C ",
                "ATOM      2  N   UNL     1       1.200   0.000   0.000  0.00  0.00    +0.000 N ",
                "TORSDOF 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qc = build_small_molecule_prep_qc(
        input_mol=input_mol,
        input_file="input.sdf",
        prepared_file=str(pdbqt),
        target_ph=7.4,
    )

    assert qc["status"] == "yellow"
    assert qc["ok"] is True
    assert any("charge sum differs" in str(w) for w in qc["warnings"])


def test_build_small_molecule_prep_qc_blocks_unsupported_pdbqt_atom_type(tmp_path):
    input_mol = Chem.MolFromSmiles("CN")
    pdbqt = tmp_path / "bad_type.pdbqt"
    pdbqt.write_text(
        "\n".join(
            [
                "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C ",
                "ATOM      2  N   UNL     1       1.200   0.000   0.000  0.00  0.00    +0.000 XX",
                "TORSDOF 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qc = build_small_molecule_prep_qc(
        input_mol=input_mol,
        input_file="input.sdf",
        prepared_file=str(pdbqt),
    )

    assert qc["status"] == "red"
    assert any("unsupported atom types" in str(e) for e in qc["errors"])


def test_build_small_molecule_prep_qc_warns_when_pdbqt_charges_are_incomplete(tmp_path):
    input_mol = Chem.MolFromSmiles("CN")
    pdbqt = tmp_path / "missing_charge.pdbqt"
    pdbqt.write_text(
        "\n".join(
            [
                "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C ",
                "ATOM      2  N   UNL     1       1.200   0.000   0.000  0.00  0.00    BAD N ",
                "TORSDOF 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qc = build_small_molecule_prep_qc(
        input_mol=input_mol,
        input_file="input.sdf",
        prepared_file=str(pdbqt),
    )

    assert qc["status"] == "yellow"
    assert any("partial charge" in str(w) for w in qc["warnings"])


def test_build_small_molecule_prep_qc_records_protonation_repair_and_smiles_hint(tmp_path):
    input_mol = Chem.MolFromSmiles("CN")
    pdbqt = tmp_path / "repaired.pdbqt"
    pdbqt.write_text(
        "\n".join(
            [
                "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.200 C ",
                "ATOM      2  N   UNL     1       1.200   0.000   0.000  0.00  0.00    +0.800 N ",
                "ATOM      3  H   UNL     1       1.800   0.000   0.000  0.00  0.00    +0.000 HD",
                "TORSDOF 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qc = build_small_molecule_prep_qc(
        input_mol=input_mol,
        input_file="input.sdf",
        prepared_file=str(pdbqt),
        prep_info={"used_repair": True, "hydrogens_added": True},
        smiles_hint="C[NH3+]",
    )

    assert qc["status"] in {"green", "yellow"}
    assert any("protonation/repair" in str(n) for n in qc["notes"])
    assert any("SMILES" in str(n) for n in qc["notes"])
