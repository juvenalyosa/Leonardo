"""Component library: curated templates for cofactors, ions and ligands found by name or chemical graph."""
import os

import pytest

app = pytest.importorskip("openmm.app")
unit = pytest.importorskip("openmm.unit")
pytest.importorskip("networkx")

from leonardo import component_library_runtime as cl  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "components")
EXAMPLE_3PTB = os.path.join(os.path.dirname(__file__), "..", "examples", "3ptb_ben", "3PTB.pdb")


def _load(path):
    pdb = app.PDBFile(path)
    return pdb.topology, [tuple(p.value_in_unit(unit.nanometer)) for p in pdb.positions]


def _by_resn(plan):
    out = {}
    for r in plan["components"]:
        out.setdefault(r["resn"], r)
    return out


def test_heme_gets_the_charmm36_template_and_amber_recommends_charmm():
    top, pos = _load(os.path.join(DATA, "1A6M_heme_site.pdb"))
    charmm = _by_resn(cl.resolve_components(top, pos, family="charmm36"))
    assert charmm["HEM"]["source"] == "library" and charmm["HEM"]["template"] == "HEME"
    assert charmm["HEM"]["charge"] == -2.0
    assert charmm["OXY"]["template"] == "O2"
    amber = cl.resolve_components(top, pos, family="amber14")
    assert _by_resn(amber)["HEM"]["source"] == "metal"
    assert amber["recommended_family"] == "charmm36"
    coord = cl.resolve_components(top, pos, family="charmm36")["metal_coordination"]
    assert any("HIS" in c["ligand_atom"] and c["distance_A"] < 2.4 for c in coord)


def test_nad_and_zinc_by_alias_and_nad_charge():
    top, pos = _load(os.path.join(DATA, "6ADH_nad_zn.pdb"))
    rows = _by_resn(cl.resolve_components(top, pos, family="charmm36"))
    assert rows["NAD"]["template"] == "NAD" and rows["NAD"]["charge"] == -1.0
    assert rows["ZN"]["template"] == "ZN2" and rows["ZN"]["charge"] == 2.0
    amber = _by_resn(cl.resolve_components(top, pos, family="amber14"))
    assert amber["ZN"]["template"] == "ZN"
    assert amber["NAD"]["source"] == "gaff" and amber["NAD"]["available_in"]["family"] == "charmm36"


def test_benzamidine_found_by_graph_as_charmm_bami():
    top, pos = _load(EXAMPLE_3PTB)
    rows = _by_resn(cl.resolve_components(top, pos, family="charmm36"))
    assert rows["BEN"]["template"] == "BAMI" and rows["BEN"]["matched_by"] == "graph"
    assert rows["BEN"]["charge"] == 1.0


def test_phosphorylated_chain_residues_are_not_sent_to_gaff():
    top, pos = _load(os.path.join(DATA, "1ATP_atp_sep.pdb"))
    rows = _by_resn(cl.resolve_components(top, pos, family="charmm36"))
    assert rows["ATP"]["template"] == "ATP" and rows["ATP"]["charge"] == -4.0
    assert rows["SEP"]["template"] == "SER-SP2"              # CHARMM phosphoserine patch
    assert rows["TPO"]["source"] == "modified_residue"       # no phosphothreonine patch in CHARMM36


def test_family_with_fewer_dropped_components_is_recommended():
    top, pos = _load(os.path.join(DATA, "1ATP_atp_sep.pdb"))
    # Amber covers TPO and Mn (ATP goes to GAFF); CHARMM would lose both
    assert cl.resolve_components(top, pos, family="charmm36")["recommended_family"] == "amber14"
    assert cl.resolve_components(top, pos, family="amber14")["recommended_family"] == "amber14"


def test_hydrogen_definitions_come_from_the_template():
    entry = cl.get_catalog("charmm36").by_name["HEME"]
    xml = cl.hydrogen_definitions_xml([entry])
    assert xml.count("<H ") == len(entry["hydrogens"]) == 30
    assert '<Residue name="HEME">' in xml


def test_partial_residue_reports_missing_atoms():
    pdb = app.PDBFile(os.path.join(DATA, "6ADH_nad_zn.pdb"))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    nad = [r for r in modeller.topology.residues() if r.name == "NAD"][0]
    modeller.delete([a for a in nad.atoms() if a.name in ("C2N", "C3N")])
    pos2 = [tuple(p.value_in_unit(unit.nanometer)) for p in modeller.positions]
    row = _by_resn(cl.resolve_components(modeller.topology, pos2, family="charmm36"))["NAD"]
    assert row["source"] == "unmatched" and "missing" in row["reason"]


def test_build_and_validate_heme_site_in_charmm36(tmp_path):
    rep = cl.build_and_validate(os.path.join(DATA, "1A6M_heme_site.pdb"), family="charmm36", minimize_steps=100,
                                out_dir=str(tmp_path))
    assert rep["ok"], rep.get("error")
    assert rep["component_charges"]["HEM A154"] == -2.0
    assert abs(rep["total_charge"] - round(rep["total_charge"])) < 1e-3
    assert rep["energy_minimized_kj"] <= rep["energy_start_kj"] + max(50, 0.02 * abs(rep["energy_start_kj"]))
    assert os.path.isfile(rep["system_xml"]) and os.path.isfile(rep["pdb"])


def test_build_refuses_when_components_lack_parameters(tmp_path):
    rep = cl.build_and_validate(os.path.join(DATA, "1A6M_heme_site.pdb"), family="amber14", minimize_steps=10)
    assert not rep["ok"] and "HEM" in rep["error"]


def test_user_library_extends_aliases(tmp_path):
    user = tmp_path / "lib.json"
    user.write_text('{"components": {"XYZ": {"category": "cofactor", "templates": {"charmm36": ["HEME"]}}}}')
    lib = cl.load_library(user_path=str(user))
    assert lib["components"]["XYZ"]["templates"]["charmm36"] == ["HEME"]
    assert "HEM" in lib["components"]


def test_recommendation_ignores_additives_and_gaff_ligands():
    top, pos = _load(EXAMPLE_3PTB)                     # benzamidine: GAFF in Amber is a fine route
    assert cl.resolve_components(top, pos, family="amber14")["recommended_family"] == "amber14"
    top, pos = _load(os.path.join(DATA, "6ADH_nad_zn.pdb"))      # NAD (cofactor) via GAFF counts
    assert cl.resolve_components(top, pos, family="amber14")["recommended_family"] == "charmm36"


def test_build_restores_pdb_names_after_template_protonation(tmp_path):
    rep = cl.build_and_validate(os.path.join(DATA, "6ADH_nad_zn.pdb"), family="charmm36", minimize_steps=20,
                                out_dir=str(tmp_path))
    assert rep["ok"], rep.get("error")
    names = {line[17:20].strip() for line in open(rep["pdb"]) if line.startswith("HETATM")}
    assert {"NAD", "ZN"} <= names and "ZN2" not in names
    assert [q for k, q in rep["component_charges"].items() if k.startswith("NAD")] == [-1.0]


# ------------------------------------------------------------------ Phase 2: modified residues, metals


def test_charmm_phospho_patches_have_integer_charges():
    cat = cl.get_catalog("charmm36")
    assert cat.by_name["SER-SP2"]["charge"] == -2.0 and cat.by_name["SER-SP1"]["charge"] == -1.0
    assert cat.by_name["TYR-TP2"]["charge"] == -2.0
    assert all(h != "HG1" for h, _p in cat.by_name["SER-SP2"]["hydrogens"])      # hydroxyl H replaced by P


def test_phosphoresidues_resolve_in_both_families():
    top, pos = _load(os.path.join(DATA, "1ATP_atp_sep.pdb"))
    charmm = _by_resn(cl.resolve_components(top, pos, family="charmm36"))
    assert charmm["SEP"]["template"] == "SER-SP2" and charmm["TPO"]["source"] == "modified_residue"
    amber = _by_resn(cl.resolve_components(top, pos, family="amber14"))
    assert amber["SEP"]["template"] == "SEP" and amber["TPO"]["template"] == "TPO"
    assert amber["SEP"]["replaces_base"]


def test_amber_phospho_build_switches_protein_forcefield_and_keeps_phosphates():
    rep = cl.build_and_validate(os.path.join(DATA, "1ATP_atp_sep.pdb"), family="amber14", remove_resn=["ATP", "MN"],
                                minimize_steps=50)
    assert rep["ok"], rep.get("error")
    assert [os.path.basename(f) for f in rep["files"]][:2] == ["ff14SB.xml", "phosaa14SB.xml"]
    charges = {k.split()[0]: q for k, q in rep["component_charges"].items()}
    assert charges == {"SEP": -2.0, "TPO": -2.0}


def test_charmm_phosphoserine_build_uses_the_patched_template():
    rep = cl.build_and_validate(os.path.join(DATA, "1ATP_atp_sep.pdb"), family="charmm36",
                                remove_resn=["MN", "TPO"], minimize_steps=50)
    assert rep["ok"], rep.get("error")
    assert {k.split()[0]: q for k, q in rep["component_charges"].items()}["SEP"] == -2.0


def test_ions_carry_the_water_file_that_defines_them():
    assert cl.get_catalog("amber14").by_name["ZN"]["files"] == ["amber14/tip3p.xml"]
    assert cl.get_catalog("charmm36").by_name["ZN2"]["files"] == ["charmm36/water.xml"]
    assert cl.forcefield_family_base_files("amber14") == ["amber14-all.xml"]


def test_zinc_site_protonation_and_restraints():
    top, pos = _load(os.path.join(DATA, "3KS3_zn_site.pdb"))
    variants, notes = cl.coordination_variants(top, pos)
    by_res = {r.id: variants[r.index] for r in top.residues() if variants[r.index]}
    assert by_res == {"94": "HID", "96": "HID", "119": "HIE"}
    rep = cl.build_and_validate(os.path.join(DATA, "3KS3_zn_site.pdb"), family="amber14", minimize_steps=50)
    assert rep["ok"], rep.get("error")
    assert len(rep["metal_restraints"]) == 3 and all("ZN" in x for x in rep["metal_restraints"])


def test_heme_axial_histidine_is_hid_and_restrained():
    rep = cl.build_and_validate(os.path.join(DATA, "1A6M_heme_site.pdb"), family="charmm36", minimize_steps=50)
    assert rep["ok"], rep.get("error")
    assert any("HIS93" in v and "HID" in v for v in rep["protonation_variants"])
    assert any("FE" in x and "NE2" in x for x in rep["metal_restraints"])


# ------------------------------------------------------------------ GAFF route for ligands without templates


def test_bond_orders_from_geometry_without_network():
    top, pos = _load(EXAMPLE_3PTB)
    ben = [r for r in top.residues() if r.name == "BEN"][0]
    mol, info_names = cl.residue_heavy_mol(ben, pos)
    assigned = cl._bond_orders_from_geometry(mol)
    from rdkit import Chem

    assert Chem.MolToSmiles(assigned) == "N=C(N)c1ccccc1"
    assert len(info_names) == 9


def test_ligand_chemistry_protonates_benzamidine_at_ph7():
    top, pos = _load(EXAMPLE_3PTB)
    ben = [r for r in top.residues() if r.name == "BEN"][0]
    mol, names, info = cl.assign_ligand_chemistry(ben, pos, ph=7.4, fetch_ccd=False)
    assert info["formal_charge"] == 1 and "NH2+" in info["smiles"]
    assert mol.GetNumAtoms() == 9 + 9                      # C7N2 + 9 H of the amidinium


def test_gaff_ligand_builds_in_amber_offline(tmp_path):
    rep = cl.build_and_validate(EXAMPLE_3PTB, family="amber14", minimize_steps=50, fetch_ccd=False)
    assert rep["ok"], rep.get("error")
    assert rep["gaff_components"][0]["resn"] == "BEN" and rep["gaff_components"][0]["formal_charge"] == 1
    assert [q for k, q in rep["component_charges"].items() if k.startswith("BEN")] == [1.0]


# ------------------------------------------------------------------ covalent components


def test_heme_c_is_refused_not_mistaken_for_heme():
    top, pos = _load(os.path.join(DATA, "1HRC_heme_c.pdb"))
    for fam in ("charmm36", "amber14"):
        plan = cl.resolve_components(top, pos, family=fam)
        row = _by_resn(plan)["HEC"]
        assert row["source"] == "covalent_metal" and not plan["ready"]
        assert {c["partner_resi"] for c in row["covalent"]} == {"14", "17"}
        assert all(r.get("template") != "HEME" for r in plan["components"])


def test_covalent_ibrutinib_is_detected_and_buildable_only_in_amber():
    top, pos = _load(os.path.join(DATA, "5P9J_ibrutinib_cys481.pdb"))
    amber = _by_resn(cl.resolve_components(top, pos, family="amber14"))["8E8"]
    assert amber["source"] == "covalent" and amber["buildable"]
    assert amber["covalent"][0]["partner_atom"] == "SG" and amber["covalent"][0]["partner_resi"] == "481"
    charmm = _by_resn(cl.resolve_components(top, pos, family="charmm36"))["8E8"]
    assert charmm["source"] == "covalent" and not charmm.get("buildable")


def test_failed_gaff_run_is_retried_then_reported():
    good = ('<ForceField><AtomTypes><Type name="ca"/></AtomTypes>'
            '<Residues><Residue name="x"><Atom name="C1" type="ca"/></Residue></Residues></ForceField>')
    bad = '<ForceField><Residues><Residue name="x"><Atom name="C1" type="ca"/></Residue></Residues></ForceField>'

    class Gen:
        def __init__(self, outputs):
            self.outputs = list(outputs)

        def generate_residue_template(self, mol):
            out = self.outputs.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

    class Mol:
        def to_smiles(self):
            return "c1ccccc1"

    assert cl._gaff_residue_xml(Gen([RuntimeError("sqm"), good]), Mol()) == good
    with pytest.raises(RuntimeError, match="not defined"):
        cl._gaff_residue_xml(Gen([RuntimeError("sqm"), bad]), Mol())


def test_antechamber_guard_repeats_a_run_that_read_zero_atoms(tmp_path):
    """AmberTools 24.8 (macOS x86_64) reads an SDF as zero atoms at random, silently: the guard runs it again."""
    import subprocess
    import sys

    from leonardo import antechamber_guard

    fake = tmp_path / "fake_antechamber.py"          # zero atoms on the first two runs, then the molecule
    fake.write_text(
        "import os, sys\n"
        "n = int(open('count').read()) if os.path.exists('count') else 0\n"
        "open('count', 'w').write(str(n + 1))\n"
        "atoms = 0 if n < 2 else 3\n"
        "open('ANTECHAMBER_AC.AC0', 'w').write('CHARGE 0\\n' + 'ATOM x\\n' * atoms)\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        "open(out, 'w').write('@<TRIPOS>MOLECULE\\nMOL\\n %d 2 1 0 0\\n' % atoms)\n")
    real = tmp_path / "real"
    real.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    real.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    (work / "in.sdf").write_text("x\n  RDKit 3D\n\n  3  2  0  0  0  0  0  0  0  0999 V2000\n")
    bin_dir = antechamber_guard.install(str(real), str(tmp_path / "guard"))
    args = ["-i", "in.sdf", "-fi", "sdf", "-o", "out.mol2", "-fo", "mol2"]
    r = subprocess.run([os.path.join(bin_dir, "antechamber")] + args, cwd=work, capture_output=True, text=True)
    assert r.returncode == 0 and (work / "count").read_text() == "3"
    assert r.stderr.count("zero atoms") == 2
    (work / "count").unlink()
    subprocess.run([os.path.join(bin_dir, "antechamber"), "-i", "in.pdb", "-fi", "pdb", "-o", "o.ac", "-fo", "ac"],
                   cwd=work, check=True)
    assert (work / "count").read_text() == "1"        # other input formats run once, untouched


def test_covalent_ibrutinib_builds_with_a_real_c_s_bond(tmp_path):
    rep = cl.build_and_validate(os.path.join(DATA, "5P9J_ibrutinib_cys481.pdb"), family="amber14",
                                minimize_steps=100, out_dir=str(tmp_path))
    assert rep["ok"], rep.get("error")
    cov = rep["covalent_components"][0]
    assert cov["attach_atom"] == "CAA" and cov["missing_terms"] == 0 and abs(cov["charge_moved_to_attach_atom"]) < 0.3
    from pymol import cmd

    cmd.delete("all")
    cmd.load(rep["pdb"], "cov")
    d = cmd.get_distance("cov and resi 481 and name SG", "cov and name CAA and not polymer")
    assert 1.74 < d < 1.94
    assert cmd.count_atoms("cov and resi 481 and name HG") == 0          # Cys481 is CYX
    assert "LeonardoMetalCoordination" not in open(rep["system_xml"]).read()
