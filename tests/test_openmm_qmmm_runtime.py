import math
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import code_v2

from openmm_qmmm_runtime import classify_residue_kind
from openmm_qmmm_runtime import classify_openmm_component_ff_class
from openmm_qmmm_runtime import build_mopac_keywords_for_operation
from openmm_qmmm_runtime import build_openmm_parameterization_audit
from openmm_qmmm_runtime import compare_openmm_system_term_summaries
from openmm_qmmm_runtime import compose_openmm_input_pdb
from openmm_qmmm_runtime import dedupe_openmm_forcefield_xml_paths
from openmm_qmmm_runtime import default_openmm_xml_selected_keys
from openmm_qmmm_runtime import export_forcefield_xml_from_amber_prmtop
from openmm_qmmm_runtime import merge_openmm_periodic_torsion_rows
from openmm_qmmm_runtime import normalize_pdb_conect_records
from openmm_qmmm_runtime import plan_openmm_staged_minimization
from openmm_qmmm_runtime import assess_ligand_topology_signature_compatibility
from openmm_qmmm_runtime import read_openmm_forcefield_residue_names
from openmm_qmmm_runtime import rewrite_pdb_component_identity
from openmm_qmmm_runtime import resolve_openmm_component_template_plan
from openmm_qmmm_runtime import sanitize_pdb_conect_for_standalone_components
from openmm_qmmm_runtime import split_openmm_parameterization_targets
from openmm_qmmm_runtime import resolve_qmmm_regions
from openmm_qmmm_runtime import run_openmm_qmmm_distance_scan_probe
from openmm_qmmm_runtime import run_openmm_qmmm_single_point_probe


def test_classify_residue_kind_basics():
    assert classify_residue_kind("ALA") == "biopolymer"
    assert classify_residue_kind("HOH") == "water"
    assert classify_residue_kind("NA") == "ion"
    assert classify_residue_kind("LIG") == "hetero"


def test_classify_residue_kind_common_md_variants_are_not_misread_as_hetero():
    for resn in ("ASH", "GLH", "HSD", "HSE", "LYN", "CYM", "CYX", "MSE", "ACE", "NME"):
        assert classify_residue_kind(resn) == "biopolymer"


def test_classify_openmm_component_ff_class_treats_ca_ion_as_standard():
    comp = {"resn": "CA", "chain": "A", "resi": "480", "category": "ion", "uid": "A|480|CA|"}
    assert classify_openmm_component_ff_class(comp) == "standard"


def test_default_openmm_xml_selected_keys_excludes_standard_ions():
    components = [
        {"resn": "CA", "chain": "A", "resi": "480", "category": "ion", "uid": "A|480|CA|"},
        {"resn": "LIG", "chain": "B", "resi": "101", "category": "cofactor", "uid": "B|101|LIG|"},
    ]
    assert default_openmm_xml_selected_keys(components) == ["B|101|LIG|"]


def test_split_openmm_parameterization_targets_skips_standard_components():
    components = [
        {"resn": "CA", "chain": "A", "resi": "480", "category": "ion", "uid": "A|480|CA|"},
        {"resn": "LIG", "chain": "B", "resi": "101", "category": "cofactor", "uid": "B|101|LIG|"},
    ]
    plan = split_openmm_parameterization_targets(components)
    assert [row["uid"] for row in plan["parameterize"]] == ["B|101|LIG|"]
    assert [row["uid"] for row in plan["skipped_standard"]] == ["A|480|CA|"]


def test_read_openmm_forcefield_residue_names(tmp_path):
    xml_path = tmp_path / "ligand_ff.xml"
    xml_path.write_text(
        "<ForceField><Residues><Residue name='BEN'><Atom name='C1' type='c'/></Residue></Residues></ForceField>",
        encoding="utf-8",
    )
    assert read_openmm_forcefield_residue_names(str(xml_path)) == ["BEN"]


def test_dedupe_openmm_forcefield_xml_paths_skips_duplicate_residue_templates(tmp_path):
    ben_a = tmp_path / "ben_a.xml"
    ben_b = tmp_path / "ben_b.xml"
    ben_a.write_text(
        "<ForceField><Residues><Residue name='BEN'><Atom name='C1' type='c'/></Residue></Residues></ForceField>",
        encoding="utf-8",
    )
    ben_b.write_text(
        "<ForceField><Residues><Residue name='BEN'><Atom name='C1' type='c'/></Residue></Residues></ForceField>",
        encoding="utf-8",
    )
    plan = dedupe_openmm_forcefield_xml_paths([str(ben_a), str(ben_b)])
    assert plan["paths"] == [str(ben_a)]
    assert len(plan["skipped"]) == 1
    assert plan["skipped"][0]["residue_templates"] == ["BEN"]


def test_dedupe_openmm_forcefield_xml_paths_keeps_distinct_residue_templates(tmp_path):
    ben = tmp_path / "ben.xml"
    lig = tmp_path / "lig.xml"
    ben.write_text(
        "<ForceField><Residues><Residue name='BEN'><Atom name='C1' type='c'/></Residue></Residues></ForceField>",
        encoding="utf-8",
    )
    lig.write_text(
        "<ForceField><Residues><Residue name='LIG'><Atom name='C1' type='c'/></Residue></Residues></ForceField>",
        encoding="utf-8",
    )
    plan = dedupe_openmm_forcefield_xml_paths([str(ben), str(lig)])
    assert plan["paths"] == [str(ben), str(lig)]
    assert plan["skipped"] == []


def test_resolve_openmm_component_template_plan_uses_forcefield_xml(tmp_path):
    ffxml = tmp_path / "ligand_openmm_ff.xml"
    ffxml.write_text(
        "<ForceField><Residues><Residue name='BEN'><Atom name='C1' type='c'/></Residue></Residues></ForceField>",
        encoding="utf-8",
    )
    components = [{"resn": "BEN", "chain": "A", "resi": "1", "category": "cofactor", "uid": "A|1|BEN|"}]
    plan = resolve_openmm_component_template_plan(
        components=components,
        xml_map={"A|1|BEN|": str(ffxml)},
        xml_kind_fn=lambda p: "forcefield" if str(p).endswith("_ff.xml") else "",
    )
    assert plan["paths"] == [str(ffxml)]
    assert plan["missing"] == []


def test_resolve_openmm_component_template_plan_converts_system_xml_to_forcefield(tmp_path):
    sysxml = tmp_path / "ligand_openmm_system.xml"
    ffxml = tmp_path / "ligand_openmm_ff.xml"
    sysxml.write_text("<System/>", encoding="utf-8")
    ffxml.write_text(
        "<ForceField><Residues><Residue name='BEN'><Atom name='C1' type='c'/></Residue></Residues></ForceField>",
        encoding="utf-8",
    )
    components = [{"resn": "BEN", "chain": "A", "resi": "1", "category": "cofactor", "uid": "A|1|BEN|"}]
    plan = resolve_openmm_component_template_plan(
        components=components,
        xml_map={"A|1|BEN|": str(sysxml)},
        xml_kind_fn=lambda p: "system" if str(p).endswith("_system.xml") else ("forcefield" if str(p).endswith("_ff.xml") else ""),
        guess_forcefield_xml_from_system_fn=lambda _p: str(ffxml),
    )
    assert plan["paths"] == [str(ffxml)]
    assert plan["missing"] == []


def test_resolve_openmm_component_template_plan_reports_missing_nonstandard_component():
    components = [{"resn": "BEN", "chain": "A", "resi": "1", "category": "cofactor", "uid": "A|1|BEN|"}]
    plan = resolve_openmm_component_template_plan(
        components=components,
        xml_map={},
        xml_kind_fn=lambda _p: "",
    )
    assert plan["paths"] == []
    assert [row["uid"] for row in plan["missing"]] == ["A|1|BEN|"]


def test_sanitize_pdb_conect_for_standalone_components_removes_external_links():
    lines = [
        "HETATM    1  C1  BEN A   1      0.000   0.000   0.000  1.00 20.00           C",
        "HETATM    2  C2  BEN A   1      1.300   0.000   0.000  1.00 20.00           C",
        "ATOM      3  CA  ALA A   2      2.600   0.000   0.000  1.00 20.00           C",
        "CONECT    1    2    3",
        "CONECT    2    1",
        "CONECT    3    1",
    ]
    out = sanitize_pdb_conect_for_standalone_components(
        lines,
        [{"chain": "A", "resi": "1", "resn": "BEN"}],
    )
    assert out["removed_links"] == 2
    assert "CONECT    1    2" in out["lines"]
    assert "CONECT    3    1" not in out["lines"]


def test_rewrite_pdb_component_identity_relabels_component_atoms():
    lines = [
        "ATOM      1  C1  LIG X 999      0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  H1  LIG X 999      0.000   0.000   1.000  1.00 20.00           H",
        "CONECT    1    2",
    ]
    out = rewrite_pdb_component_identity(lines, resn="BEN", chain="A", resi="223", hetero=True)
    assert out[0].startswith("HETATM")
    assert out[0][17:20] == "BEN"
    assert out[0][21] == "A"
    assert out[0][22:26].strip() == "223"
    assert out[2] == "CONECT    1    2"


def test_normalize_pdb_conect_records_deduplicates_targets():
    lines = [
        "ATOM      1  C1  BEN A   1      0.000   0.000   0.000  1.00 20.00           C",
        "ATOM      2  C2  BEN A   1      1.400   0.000   0.000  1.00 20.00           C",
        "ATOM      3  H1  BEN A   1      0.000   1.000   0.000  1.00 20.00           H",
        "CONECT    1    2    2    3",
        "CONECT    2    1    1",
    ]
    out = normalize_pdb_conect_records(lines)
    assert "CONECT    1    2    3" in out
    assert "CONECT    2    1" in out


def test_compose_openmm_input_pdb_renumbers_and_keeps_component_internal_conect():
    polymer = [
        "ATOM     10  N   ALA A   1      0.000   0.000   0.000  1.00 20.00           N",
        "ATOM     11  CA  ALA A   1      1.300   0.000   0.000  1.00 20.00           C",
    ]
    component = [
        "HETATM   21  C1  BEN A 480      2.600   0.000   0.000  1.00 20.00           C",
        "HETATM   22  C2  BEN A 480      3.900   0.000   0.000  1.00 20.00           C",
        "CONECT   21   22",
    ]
    out = compose_openmm_input_pdb(polymer, [component])
    lines = list(out["lines"])
    assert out["atom_count"] == 4
    assert out["conect_count"] == 1
    assert lines[0].startswith("ATOM      1")
    assert any(line.startswith("HETATM    3") for line in lines)
    assert "CONECT    3    4" in lines


def test_merge_openmm_periodic_torsion_rows_combines_duplicate_quadruplets():
    rows = [
        {"tag": "Proper", "type1": "A1", "type2": "A2", "type3": "A3", "type4": "A4", "periodicity1": "1", "phase1": "0.0", "k1": "0.5"},
        {"tag": "Proper", "type1": "A1", "type2": "A2", "type3": "A3", "type4": "A4", "periodicity1": "2", "phase1": "3.14", "k1": "1.0"},
    ]
    merged = merge_openmm_periodic_torsion_rows(rows)
    assert len(merged) == 1
    row = merged[0]
    assert row["periodicity1"] == "1"
    assert row["periodicity2"] == "2"
    assert row["k2"] == "1.0"


def test_build_openmm_parameterization_audit_reports_term_inventory_and_charge():
    audit = build_openmm_parameterization_audit(
        atom_records=[
            {"charge_e": -0.2, "sigma_nm": 0.3, "epsilon_kjmol": 0.1},
            {"charge_e": 0.1, "sigma_nm": 0.25, "epsilon_kjmol": 0.05},
            {"charge_e": 0.1, "sigma_nm": 0.28, "epsilon_kjmol": 0.0},
        ],
        residue_bond_pairs=[(0, 1), (1, 2)],
        bond_terms=[{"i": 0, "j": 1}, {"i": 1, "j": 2}],
        angle_terms=[{"i": 0, "j": 1, "k": 2}],
        torsion_terms=[{"i": 0, "j": 1, "k": 2, "l": 0, "improper": False}],
        torsion_rows=[{"tag": "Proper", "type1": "A1", "type2": "A2", "type3": "A3", "type4": "A1", "periodicity1": "1", "phase1": "0.0", "k1": "0.5"}],
    )
    assert audit["counts"]["atoms"] == 3
    assert audit["counts"]["bond_terms"] == 2
    assert audit["counts"]["angle_terms"] == 1
    assert audit["counts"]["torsion_terms"] == 1
    assert math.isclose(float(audit["charge"]["sum_e"]), 0.0, abs_tol=1.0e-12)
    assert audit["lj"]["zero_epsilon_atoms"] == 1
    assert audit["flags"]["has_bonded_terms"] is True
    assert audit["flags"]["has_lj_terms"] is True


def test_compare_openmm_system_term_summaries_accepts_full_parameter_match():
    src = {
        "particles": 2,
        "force_counts": {"NonbondedForce": 1, "HarmonicBondForce": 1, "HarmonicAngleForce": 1, "PeriodicTorsionForce": 1},
        "masses_da": [12.011, 1.008],
        "nonbonded_particles": [
            {"index": 0, "charge_e": -0.1, "sigma_nm": 0.33, "epsilon_kjmol": 0.12},
            {"index": 1, "charge_e": 0.1, "sigma_nm": 0.21, "epsilon_kjmol": 0.02},
        ],
        "nonbonded_exceptions": [
            {"a": 0, "b": 1, "chargeprod_e2": -0.008333333333, "sigma_nm": 0.25, "epsilon_kjmol": 0.01},
        ],
        "bonds": [{"a": 0, "b": 1, "length_nm": 0.109, "k_kjmol_nm2": 284512.0}],
        "angles": [{"a": 0, "b": 1, "c": 0, "angle_rad": 1.91, "k_kjmol_rad2": 350.0}],
        "torsions": [{"a": 0, "b": 1, "c": 0, "d": 1, "periodicity": 3, "phase_rad": 0.0, "k_kjmol": 1.5}],
    }
    out = compare_openmm_system_term_summaries(src, dict(src), tol=1.0e-10)
    assert out["ok"] is True
    assert out["issues"] == []
    assert math.isclose(float(out["max_abs_diff"]["nonbonded_particles"]), 0.0, abs_tol=1.0e-12)


def test_compare_openmm_system_term_summaries_flags_charge_mismatch():
    src = {
        "particles": 2,
        "force_counts": {"NonbondedForce": 1},
        "masses_da": [12.011, 1.008],
        "nonbonded_particles": [
            {"index": 0, "charge_e": -0.1, "sigma_nm": 0.33, "epsilon_kjmol": 0.12},
            {"index": 1, "charge_e": 0.1, "sigma_nm": 0.21, "epsilon_kjmol": 0.02},
        ],
        "nonbonded_exceptions": [],
        "bonds": [],
        "angles": [],
        "torsions": [],
    }
    dst = dict(src)
    dst["nonbonded_particles"] = [
        {"index": 0, "charge_e": -0.2, "sigma_nm": 0.33, "epsilon_kjmol": 0.12},
        {"index": 1, "charge_e": 0.2, "sigma_nm": 0.21, "epsilon_kjmol": 0.02},
    ]
    out = compare_openmm_system_term_summaries(src, dst, tol=1.0e-10)
    assert out["ok"] is False
    assert any("nonbonded_particles_mismatch" in issue for issue in out["issues"])


def test_assess_ligand_topology_signature_compatibility_accepts_matching_aromatic_signature():
    out = assess_ligand_topology_signature_compatibility(
        {"rotb": 0, "aromatic_bonds": 6},
        {"rotb": 0, "aromatic_bonds": 6},
    )
    assert out["ok"] is True
    assert out["issues"] == []


def test_assess_ligand_topology_signature_compatibility_rejects_lost_aromaticity():
    out = assess_ligand_topology_signature_compatibility(
        {"rotb": 0, "aromatic_bonds": 6},
        {"rotb": 2, "aromatic_bonds": 0},
    )
    assert out["ok"] is False
    assert any("aromatic_bonds_too_low" in issue for issue in out["issues"])
    assert any("unexpected_flexibility" in issue for issue in out["issues"])


def test_plan_openmm_staged_minimization_builds_heavy_atom_restraint_schedule():
    plan = plan_openmm_staged_minimization(
        atom_rows=[
            {"index": 0, "name": "N", "resn": "ALA", "element": "N"},
            {"index": 1, "name": "CA", "resn": "ALA", "element": "C"},
            {"index": 2, "name": "CB", "resn": "ALA", "element": "C"},
            {"index": 3, "name": "H", "resn": "ALA", "element": "H"},
            {"index": 4, "name": "C1", "resn": "BEN", "element": "C"},
            {"index": 5, "name": "H1", "resn": "BEN", "element": "H"},
            {"index": 6, "name": "O", "resn": "HOH", "element": "O"},
        ],
        min_iters=200,
        hydrogens_added=18,
    )
    assert plan["restrained_atom_count"] == 4
    assert plan["counts"]["polymer_backbone"] == 2
    assert plan["counts"]["polymer_sidechain"] == 1
    assert plan["counts"]["ligand_heavy"] == 1
    assert [stage["label"] for stage in plan["stages"]] == [
        "restrained_prep",
        "restrained_release",
        "unrestrained",
    ]
    assert math.isclose(float(plan["stages"][0]["scale"]), 1.0, abs_tol=1.0e-12)
    assert math.isclose(float(plan["stages"][1]["scale"]), 0.25, abs_tol=1.0e-12)
    assert math.isclose(float(plan["stages"][2]["scale"]), 0.0, abs_tol=1.0e-12)
    assert sum(int(stage["iterations"]) for stage in plan["stages"]) == 200


def test_plan_openmm_staged_minimization_handles_small_iteration_counts():
    plan = plan_openmm_staged_minimization(
        atom_rows=[
            {"index": 0, "name": "C1", "resn": "BEN", "element": "C"},
            {"index": 1, "name": "C2", "resn": "BEN", "element": "C"},
            {"index": 2, "name": "H1", "resn": "BEN", "element": "H"},
        ],
        min_iters=7,
        hydrogens_added=0,
    )
    assert plan["restrained_atom_count"] == 2
    assert sum(int(stage["iterations"]) for stage in plan["stages"]) == 7
    assert plan["stages"][0]["label"] == "restrained_prep"
    assert plan["stages"][-1]["label"] == "unrestrained"


def test_export_forcefield_xml_from_amber_prmtop_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    prmtop = root / "benchmarks/abfe/ptb3_ben" / "BEN_from_complex.prmtop"
    assert prmtop.is_file()
    out_xml = tmp_path / "BEN_from_complex_ff.xml"
    res = export_forcefield_xml_from_amber_prmtop(
        prmtop_file=str(prmtop),
        out_xml=str(out_xml),
        residue_name_hint="BEN",
    )
    assert res["ok"] is True, str(res.get("error", ""))
    assert out_xml.is_file()
    assert read_openmm_forcefield_residue_names(str(out_xml)) == ["BEN"]
    audit_path = Path(str(out_xml) + ".audit.json")
    assert audit_path.is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert int(audit["counts"]["atoms"]) == 18
    assert int(audit["counts"]["bond_terms"]) == 18
    assert int(audit["counts"]["angle_terms"]) == 27
    assert int(audit["counts"]["torsion_terms"]) == 45
    assert math.isclose(float(audit["charge"]["sum_e"]), 1.0, abs_tol=5.0e-3)
    assert bool((audit.get("roundtrip", {}) or {}).get("ok", False)) is True


def test_abfe_bonded_ff_transfer_accepts_heavy_only_shuffled_pose(tmp_path):
    class DummyDialog:
        pass

    needed = [
        "_infer_element_symbol",
        "_bond_threshold",
        "_max_covalent_neighbors",
        "_infer_local_ligand_bond_adjacency",
        "_ligand_mapping_geometry_score",
        "_map_ligand_atoms_by_graph",
        "_validate_ligand_bond_mapping_geometry",
        "_map_param_ligand_atoms_to_local",
        "_load_ligand_xml_atom_meta",
        "_inject_full_ligand_ff_terms_from_prmtop",
        "_inject_full_ligand_ff_terms_from_system_xml",
        "_export_ligand_openmm_xml_from_prmtop",
    ]
    for name in needed:
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    dlg = DummyDialog()

    root = Path(__file__).resolve().parents[1]
    prmtop = root / "benchmarks/abfe/ptb3_ben" / "BEN_from_complex.prmtop"
    pdb = root / "benchmarks/abfe/ptb3_ben" / "BEN_from_complex.pdb"
    assert prmtop.is_file()
    assert pdb.is_file()

    coords_nm = []
    local_infos = []
    for line in pdb.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_name = str(line[12:16].strip())
        element = str((line[76:78].strip() or atom_name[:1]).strip().upper())
        if element == "H" or atom_name.upper().startswith("H"):
            continue
        coords_nm.append(
            (
                float(line[30:38]) * 0.1,
                float(line[38:46]) * 0.1,
                float(line[46:54]) * 0.1,
            )
        )
        local_infos.append({"name": atom_name, "element": element})
    assert len(coords_nm) == 9
    shuffle = [3, 1, 7, 0, 8, 2, 6, 5, 4]
    coords_nm = np.asarray([coords_nm[i] for i in shuffle], dtype=float)
    local_infos = [dict(local_infos[i]) for i in shuffle]
    for idx, row in enumerate(local_infos, 1):
        row["name"] = f"X{idx}"
    local_lig = list(range(len(local_infos)))

    system = code_v2.OMM.System()
    for _ in local_lig:
        system.addParticle(12.0 * code_v2.OMM_UNIT.dalton)
    ok_prm, prm_info = dlg._inject_full_ligand_ff_terms_from_prmtop(
        system,
        local_lig,
        local_infos,
        str(prmtop),
        local_coords_nm=coords_nm,
    )
    assert ok_prm is True, str(prm_info)
    assert str(prm_info.get("mapping_mode", "")).startswith("graph_match")
    assert bool(prm_info.get("reduced_hydrogen_model", False)) is True
    assert int(prm_info.get("bond", 0) or 0) > 0

    system_xml = tmp_path / "BEN_from_complex_system.xml"
    ok_export, msg_export = dlg._export_ligand_openmm_xml_from_prmtop(str(prmtop), str(system_xml))
    assert ok_export is True, str(msg_export)
    assert system_xml.is_file()
    assert Path(str(system_xml) + ".meta.json").is_file()

    system_from_xml = code_v2.OMM.System()
    for _ in local_lig:
        system_from_xml.addParticle(12.0 * code_v2.OMM_UNIT.dalton)
    ok_xml, xml_info = dlg._inject_full_ligand_ff_terms_from_system_xml(
        system_from_xml,
        local_lig,
        local_infos,
        str(system_xml),
        local_coords_nm=coords_nm,
    )
    assert ok_xml is True, str(xml_info)
    assert str(xml_info.get("mapping_mode", "")).startswith("graph_match")
    assert bool(xml_info.get("reduced_hydrogen_model", False)) is True
    assert int(xml_info.get("bond", 0) or 0) > 0


def test_abfe_docking_pose_metadata_restore_recovers_validated_xml_mapping(tmp_path):
    class DummyDialog:
        pass

    needed = [
        "_infer_element_symbol",
        "_bond_threshold",
        "_max_covalent_neighbors",
        "_infer_local_ligand_bond_adjacency",
        "_ligand_mapping_geometry_score",
        "_map_ligand_atoms_by_graph",
        "_validate_ligand_bond_mapping_geometry",
        "_map_param_ligand_atoms_to_local",
        "_load_ligand_xml_atom_meta",
        "_inject_full_ligand_ff_terms_from_system_xml",
        "_export_ligand_openmm_xml_from_prmtop",
        "_dock_pdbqt_type_token",
        "_dock_pdbqt_atom_type_to_element",
        "_read_multimodel_pose_atom_rows",
        "_match_pose_rows_to_local_indices",
        "_normalize_abfe_local_ligand_infos_from_docking_pose",
    ]
    for name in needed:
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    dlg = DummyDialog()

    root = Path(__file__).resolve().parents[1]
    prmtop = root / "benchmarks/abfe/ptb3_ben" / "BEN_from_complex.prmtop"
    pdb = root / "benchmarks/abfe/ptb3_ben" / "BEN_from_complex.pdb"
    assert prmtop.is_file()
    assert pdb.is_file()

    pose_pdbqt = tmp_path / "docked_pose_model1.pdbqt"
    pose_lines = ["MODEL 1"]
    coords_nm = []
    exact_names = []
    for raw in pdb.read_text(encoding="utf-8").splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        atom_name = str(raw[12:16].strip())
        element = str((raw[76:78].strip() or atom_name[:1]).strip().upper())
        exact_names.append(atom_name)
        coords_nm.append(
            (
                float(raw[30:38]) * 0.1,
                float(raw[38:46]) * 0.1,
                float(raw[46:54]) * 0.1,
            )
        )
        atom_type = {"C": "C", "N": "N", "O": "O", "H": "H"}.get(element, "C")
        pose_lines.append(f"{raw[:66].ljust(66)}  0.00 {atom_type}")
    pose_lines.append("ENDMDL")
    pose_pdbqt.write_text("\n".join(pose_lines) + "\n", encoding="utf-8")

    shuffle = [8, 2, 15, 0, 11, 4, 17, 6, 1, 14, 9, 3, 12, 5, 16, 10, 7, 13]
    coords_nm = np.asarray([coords_nm[i] for i in shuffle], dtype=float)
    local_infos = [{"name": f"X{idx+1}", "element": ""} for idx in range(len(shuffle))]
    local_lig = list(range(len(local_infos)))

    system_xml = tmp_path / "BEN_from_complex_system.xml"
    ok_export, msg_export = dlg._export_ligand_openmm_xml_from_prmtop(str(prmtop), str(system_xml))
    assert ok_export is True, str(msg_export)
    assert system_xml.is_file()

    system_fail = code_v2.OMM.System()
    for _ in local_lig:
        system_fail.addParticle(12.0 * code_v2.OMM_UNIT.dalton)
    ok_before, before_info = dlg._inject_full_ligand_ff_terms_from_system_xml(
        system_fail,
        local_lig,
        local_infos,
        str(system_xml),
        local_coords_nm=coords_nm,
    )
    assert ok_before is False

    ok_norm, norm_info = dlg._normalize_abfe_local_ligand_infos_from_docking_pose(
        local_lig,
        local_infos,
        coords_nm,
        str(pose_pdbqt),
        model_index=1,
    )
    assert ok_norm is True, str(norm_info)
    assert str(norm_info.get("mode", "")) == "all_atoms"
    assert int(norm_info.get("renamed", 0) or 0) == len(local_infos)
    assert [str(row.get("name", "")) for row in local_infos] == [exact_names[i] for i in shuffle]

    system_ok = code_v2.OMM.System()
    for _ in local_lig:
        system_ok.addParticle(12.0 * code_v2.OMM_UNIT.dalton)
    ok_after, after_info = dlg._inject_full_ligand_ff_terms_from_system_xml(
        system_ok,
        local_lig,
        local_infos,
        str(system_xml),
        local_coords_nm=coords_nm,
    )
    assert ok_after is True, str(after_info)
    assert str(after_info.get("mapping_mode", "")).strip() != ""
    assert int(after_info.get("bond", 0) or 0) > 0


def test_resolve_qmmm_regions_auto_hetero_ligand():
    atom_rows = [
        {"index": 0, "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 1, "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 2, "resn": "LIG", "chain": "L", "resi": "101"},
        {"index": 3, "resn": "LIG", "chain": "L", "resi": "101"},
        {"index": 4, "resn": "HOH", "chain": "W", "resi": "1"},
    ]
    reg = resolve_qmmm_regions(atom_rows=atom_rows, n_atoms=5, config={})
    assert reg["qm_indices"] == [2, 3]
    assert set(reg["mm_indices"]) == {0, 1, 4}
    assert reg["info"]["mode"] == "auto_largest_hetero_residue"


def test_resolve_qmmm_regions_explicit_indices():
    atom_rows = [{"index": i, "resn": "ALA", "chain": "A", "resi": "1"} for i in range(6)]
    reg = resolve_qmmm_regions(
        atom_rows=atom_rows,
        n_atoms=6,
        config={"qm_indices": [0, 1], "mm_indices": [2, 3, 4, 5]},
    )
    assert reg["qm_indices"] == [0, 1]
    assert reg["mm_indices"] == [2, 3, 4, 5]
    assert reg["info"]["mode"] == "explicit_qm_indices"


def test_resolve_qmmm_regions_residue_key_selection():
    atom_rows = [
        {"index": 0, "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 1, "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 2, "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 3, "resn": "HOH", "chain": "W", "resi": "5"},
    ]
    reg = resolve_qmmm_regions(
        atom_rows=atom_rows,
        n_atoms=4,
        config={"qm_residue_keys": [{"chain": "L", "resi": "10", "resn": "LIG"}]},
    )
    assert reg["qm_indices"] == [1, 2]
    assert reg["info"]["mode"] == "residue_key_selection"


def test_resolve_qmmm_regions_atom_descriptor_selection():
    atom_rows = [
        {"index": 0, "name": "CA", "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 1, "name": "CB", "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 2, "name": "C1", "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 3, "name": "O1", "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 4, "name": "ZN", "resn": "ZN", "chain": "Z", "resi": "1"},
    ]
    reg = resolve_qmmm_regions(
        atom_rows=atom_rows,
        n_atoms=5,
        config={
            "qm_atom_descriptors": [
                {"chain": "L", "resi": "10", "resn": "LIG", "name": "C1"},
                {"chain": "L", "resi": "10", "resn": "LIG", "name": "O1"},
            ],
            "mm_atom_descriptors": [
                {"chain": "A", "resi": "1", "resn": "ALA", "name": "CA"},
                {"chain": "A", "resi": "1", "resn": "ALA", "name": "CB"},
                {"chain": "Z", "resi": "1", "resn": "ZN", "name": "ZN"},
            ],
        },
    )
    assert reg["qm_indices"] == [2, 3]
    assert reg["mm_indices"] == [0, 1, 4]
    assert reg["info"]["mode"] == "atom_descriptor_selection"
    assert int(reg["info"]["qm_atom_count"]) == 2
    assert int(reg["info"]["mm_atom_count"]) == 3


def test_resolve_qmmm_regions_auto_ligand_pocket_uses_coordinates():
    atom_rows = [
        {"index": 0, "name": "CA", "resn": "ALA", "chain": "A", "resi": "1", "element": "C", "x": 8.0, "y": 0.0, "z": 0.0},
        {"index": 1, "name": "CB", "resn": "PHE", "chain": "A", "resi": "2", "element": "C", "x": 2.4, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "C1", "resn": "LIG", "chain": "L", "resi": "10", "element": "C", "x": 0.0, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "O1", "resn": "LIG", "chain": "L", "resi": "10", "element": "O", "x": 1.0, "y": 0.0, "z": 0.0},
    ]
    reg = resolve_qmmm_regions(
        atom_rows=atom_rows,
        n_atoms=4,
        config={"auto_include_pocket": True, "qm_pocket_cutoff_ang": 3.0, "operation": "optimization"},
    )
    assert set(reg["qm_indices"]) == {1, 2, 3}
    assert reg["mm_indices"] == [0]
    assert reg["info"]["mode"] == "auto_ligand_pocket"


def test_resolve_qmmm_regions_forces_reaction_atoms_into_auto_region():
    atom_rows = [
        {"index": 0, "name": "CA", "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 1, "name": "NZ", "resn": "LYS", "chain": "A", "resi": "50"},
        {"index": 2, "name": "C1", "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 3, "name": "O1", "resn": "LIG", "chain": "L", "resi": "10"},
    ]
    reg = resolve_qmmm_regions(
        atom_rows=atom_rows,
        n_atoms=4,
        config={
            "force_qm_atom_descriptors": [
                {"chain": "A", "resi": "50", "resn": "LYS", "name": "NZ"}
            ]
        },
    )
    assert set(reg["qm_indices"]) == {1, 2, 3}
    assert reg["mm_indices"] == [0]
    assert int(reg["info"]["forced_qm_atom_count"]) == 1


def test_resolve_qmmm_regions_forces_reaction_residue_into_auto_region():
    atom_rows = [
        {"index": 0, "name": "CA", "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 1, "name": "NZ", "resn": "LYS", "chain": "A", "resi": "50"},
        {"index": 2, "name": "CE", "resn": "LYS", "chain": "A", "resi": "50"},
        {"index": 3, "name": "C1", "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 4, "name": "O1", "resn": "LIG", "chain": "L", "resi": "10"},
    ]
    reg = resolve_qmmm_regions(
        atom_rows=atom_rows,
        n_atoms=5,
        config={
            "force_qm_residue_keys": [
                {"chain": "A", "resi": "50", "resn": "LYS"}
            ]
        },
    )
    assert set(reg["qm_indices"]) == {1, 2, 3, 4}
    assert reg["mm_indices"] == [0]
    assert int(reg["info"]["forced_qm_atom_count"]) == 2


def test_build_mopac_keywords_for_operation():
    kw_probe = build_mopac_keywords_for_operation("probe", "")
    kw_opt = build_mopac_keywords_for_operation("opt", "")
    kw_thermo = build_mopac_keywords_for_operation("thermo", "")
    kw_electro = build_mopac_keywords_for_operation("electrostatic", "")
    kw_md_coupled = build_mopac_keywords_for_operation("md_coupled", "")
    assert "QMMM" in kw_probe and "AUX" in kw_probe
    assert "EF" in kw_opt and "QMMM" in kw_opt
    assert "1SCF" not in kw_opt
    assert "THERMO" in kw_thermo and "QMMM" in kw_thermo
    assert "1SCF" not in kw_thermo
    assert "GRAD" in kw_electro and "QMMM" in kw_electro
    assert "GRAD" in kw_md_coupled and "QMMM" in kw_md_coupled
    kw_custom = build_mopac_keywords_for_operation("probe", "PM6 1SCF")
    assert "PM6" in kw_custom and "QMMM" in kw_custom and "AUX" in kw_custom


def test_build_mopac_keywords_switches_stale_default_to_selected_operation():
    kw_opt = build_mopac_keywords_for_operation("optimization", "PM7 1SCF GRAD AUX QMMM")
    assert "EF" in kw_opt
    assert "THERMO" not in kw_opt

    kw_thermo = build_mopac_keywords_for_operation("thermochemistry", "PM7 1SCF GRAD AUX QMMM")
    assert "FORCE" in kw_thermo
    assert "THERMO" in kw_thermo


def test_run_openmm_qmmm_single_point_probe_smoke(tmp_path):
    xyz_full = np.array(
        [
            [0.0, 0.0, 0.0],  # protein
            [1.0, 0.0, 0.0],  # protein
            [2.0, 0.0, 0.0],  # ligand
            [3.0, 0.0, 0.0],  # ligand
        ],
        dtype=float,
    )
    atomic_numbers = [6, 6, 6, 8]
    charges = np.array([0.2, -0.2, 0.0, 0.0], dtype=float)
    atom_rows = [
        {"index": 0, "name": "CA", "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 1, "name": "CB", "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 2, "name": "C1", "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 3, "name": "O1", "resn": "LIG", "chain": "L", "resi": "10"},
    ]

    def _runner(cmdline, cwd=None, capture_output=None, text=None, timeout=None):
        assert str(cmdline[0]) == "mopac"
        assert cwd is not None
        aux = """
TOTAL_ENERGY:EV= -35.0000
ATOM_CHARGES[2]=
 0.25000000 -0.25000000
GRADIENTS:KCAL/MOL/ANGSTROM[6]=
  1.00000000 0.00000000 0.00000000
 -1.00000000 0.00000000 0.00000000
"""
        (tmp_path / "step.aux").write_text(aux, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    out = run_openmm_qmmm_single_point_probe(
        xyz_full_ang=xyz_full,
        atomic_numbers_full=atomic_numbers,
        mm_charges_e=charges,
        atom_rows=atom_rows,
        config={},
        workdir=str(tmp_path),
        runner=_runner,
    )
    assert out["ok"] is True
    assert out["qm_indices"] == [2, 3]
    assert out["mm_indices"] == [0, 1]
    qmmm = dict(out["qmmm"] or {})
    assert math.isclose(float(qmmm["energy_qm_hartree"]), -35.0 / 27.211386245988, rel_tol=1e-12)
    f_corr = np.asarray(out["force_correction_kj_mol_nm"], dtype=float)
    assert f_corr.shape == (4, 3)
    diag = dict(out["diagnostics"] or {})
    assert diag.get("third_law_ok") is True
    assert float(diag.get("net_electrostatic_force_norm", 1.0)) < 1.0e-6


def test_run_openmm_qmmm_distance_scan_probe_with_mock_step():
    xyz_full = np.array(
        [
            [0.0, 0.0, 0.0],  # protein
            [2.0, 0.0, 0.0],  # ligand atom A
            [3.0, 0.0, 0.0],  # ligand atom B (scanned)
        ],
        dtype=float,
    )
    atomic_numbers = [6, 6, 8]
    charges = np.array([0.1, 0.0, 0.0], dtype=float)
    atom_rows = [
        {"index": 0, "name": "CA", "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 1, "name": "C1", "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 2, "name": "O1", "resn": "LIG", "chain": "L", "resi": "10"},
    ]

    def _mock_step(
        xyz_full_ang,
        atomic_numbers_full,
        qm_indices,
        mm_indices,
        q_mm,
        mopac_executable="mopac",
        mopac_keywords="",
        qm_total_charge=0,
        qm_multiplicity=1,
        workdir="",
        dielectric=1.0,
        runner=None,
        timeout_s=120,
    ):
        xyz = np.asarray(xyz_full_ang, dtype=float)
        qidx = list(qm_indices)
        mmidx = list(mm_indices)
        # Energy minimum at distance 1.5 A between the 2 QM atoms.
        d = float(np.linalg.norm(xyz[qidx[1]] - xyz[qidx[0]]))
        e = float((d - 1.5) ** 2 - 10.0)
        return {
            "ok": True,
            "workdir": str(workdir or ""),
            "qm_indices": list(qidx),
            "mm_indices": list(mmidx),
            "phi_qm_au": np.zeros((len(qidx),), dtype=float),
            "q_qm": np.zeros((len(qidx),), dtype=float),
            "energy_qm_hartree": float(e),
            "forces_qm_internal_kj_mol_nm": np.zeros((len(qidx), 3), dtype=float),
            "energy_qmmm_elec_kj_mol": 0.0,
            "forces_qmmm_elec_qm_kj_mol_nm": np.zeros((len(qidx), 3), dtype=float),
            "forces_qmmm_elec_mm_kj_mol_nm": np.zeros((len(mmidx), 3), dtype=float),
        }

    out = run_openmm_qmmm_distance_scan_probe(
        xyz_full_ang=xyz_full,
        atomic_numbers_full=atomic_numbers,
        mm_charges_e=charges,
        atom_rows=atom_rows,
        scan_pair=(1, 2),
        scan_start_ang=1.0,
        scan_end_ang=2.0,
        n_points=5,
        config={},
        qmmm_step_fn=_mock_step,
    )
    assert out["ok"] is True
    assert int(out["n_points"]) == 5
    rows = list(out.get("rows", []) or [])
    assert len(rows) == 5
    best = dict(out.get("best_point", {}) or {})
    assert int(best.get("point", 0)) in (3, 2, 4)
    assert abs(float(best.get("distance_ang", 0.0)) - 1.5) <= 0.26


# ---------------------------------------------------------------------------
# Force-coupled QM/MM: the classical description of the QM region must be removed (no double counting)
# ---------------------------------------------------------------------------
def _toy_qmmm_system(q_qm=(0.4, -0.4), k_qm_bond=1000.0, k_boundary_bond=1000.0):
    """4 atoms: QM = {0, 1} bonded; MM = {2, 3} bonded; a bond, an angle and a 1-4 pair cross the boundary."""
    import pytest
    mm = pytest.importorskip("openmm")
    system = mm.System()
    for _ in range(4):
        system.addParticle(12.0)
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, k_qm_bond)        # QM-QM   -> removed
    bonds.addBond(1, 2, 0.15, k_boundary_bond)  # QM-MM   -> kept (boundary)
    bonds.addBond(2, 3, 0.15, 1000.0)           # MM-MM   -> kept
    system.addForce(bonds)
    angles = mm.HarmonicAngleForce()
    angles.addAngle(0, 1, 2, 1.9, 100.0)        # crosses the boundary -> kept
    system.addForce(angles)
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for q in (q_qm[0], q_qm[1], 0.5, -0.5):
        nb.addParticle(q, 0.3, 0.5)
    nb.addException(0, 1, 0.0, 0.3, 0.0)
    nb.addException(2, 3, 0.0, 0.3, 0.0)
    nb.addException(1, 2, 0.0, 0.3, 0.0)
    nb.addException(0, 2, q_qm[0] * 0.5 / 1.2, 0.3, 0.25)   # 1-4 style pair across the boundary
    system.addForce(nb)
    return mm, system


_TOY_XYZ = [(0.0, 0.0, 0.0), (0.17, 0.02, 0.0), (0.32, 0.14, 0.03), (0.43, 0.25, 0.07)]   # bonds a bit stretched


def _energy_forces(mm, system, xyz=_TOY_XYZ):
    import openmm.unit as u
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([mm.Vec3(*p) for p in xyz] * u.nanometer)
    st = ctx.getState(getForces=True, getEnergy=True)
    return (st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
            np.asarray(st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer)))


def test_prepare_system_for_qmmm_coupling_removes_classical_qm_terms():
    from openmm_qmmm_runtime import prepare_openmm_system_for_qmmm_coupling

    mm, system = _toy_qmmm_system()
    rep = prepare_openmm_system_for_qmmm_coupling(system, [0, 1])
    assert rep["qm_charges_zeroed"] == 2
    assert math.isclose(rep["qm_total_classical_charge_e"], 0.0, abs_tol=1e-9)
    assert rep["qm_qm_exceptions"] == 1
    assert rep["bonded_terms_removed"] == {"HarmonicBondForce": 1, "HarmonicAngleForce": 0}
    assert rep["boundary_bonded_terms_kept"] == {"HarmonicBondForce": 1, "HarmonicAngleForce": 1}
    assert rep["boundary_exceptions_uncharged"] == 2
    assert rep["unhandled_forces"] == []
    nb = [system.getForce(i) for i in range(system.getNumForces())
          if system.getForce(i).__class__.__name__ == "NonbondedForce"][0]
    for i in (0, 1):
        q, _sig, eps = nb.getParticleParameters(i)
        assert q.value_in_unit(q.unit) == 0.0 and eps.value_in_unit(eps.unit) > 0.0   # Coulomb off, LJ kept
    q2, _sig, _eps = nb.getParticleParameters(2)
    assert math.isclose(q2.value_in_unit(q2.unit), 0.5)                               # MM charges untouched


def test_prepared_system_is_independent_of_classical_qm_parameters():
    """After preparation nothing classical about the QM region may act (MOPAC provides it): changing the QM
    charges or the QM-QM bond must leave energy and forces unchanged, while boundary terms must still act."""
    from openmm_qmmm_runtime import prepare_openmm_system_for_qmmm_coupling

    results = []
    for kwargs in ({}, {"q_qm": (0.9, -0.2)}, {"k_qm_bond": 7000.0}):
        mm, system = _toy_qmmm_system(**kwargs)
        prepare_openmm_system_for_qmmm_coupling(system, [0, 1])
        results.append(_energy_forces(mm, system))
    e0, f0 = results[0]
    for e, f in results[1:]:
        assert math.isclose(e, e0, rel_tol=1e-10, abs_tol=1e-10)
        assert np.allclose(f, f0, atol=1e-8)
    # without preparation the same changes do alter the classical energy (so the test is sensitive)
    mm, a = _toy_qmmm_system()
    mm, b = _toy_qmmm_system(q_qm=(0.9, -0.2))
    assert abs(_energy_forces(mm, a)[0] - _energy_forces(mm, b)[0]) > 1.0
    # boundary terms are kept: the QM-MM bond stiffness still matters after preparation
    mm, c = _toy_qmmm_system(k_boundary_bond=7000.0)
    prepare_openmm_system_for_qmmm_coupling(c, [0, 1])
    assert abs(_energy_forces(mm, c)[0] - e0) > 1.0
