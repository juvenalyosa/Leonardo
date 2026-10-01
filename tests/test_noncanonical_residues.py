"""Covalent adducts on any residue and modified residues: merged curated templates (CHARMM36 LYR) and
ACE/NME-capped GAFF2 non-canonical residues (Amber); bond orders from crystal geometry."""
import os

import pytest

app = pytest.importorskip("openmm.app")
unit = pytest.importorskip("openmm.unit")
Chem = pytest.importorskip("rdkit.Chem")

from leonardo import component_library_runtime as cl  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "components")


def _load(name):
    pdb = app.PDBFile(os.path.join(DATA, name))
    return pdb.topology, [tuple(p.value_in_unit(unit.nanometer)) for p in pdb.positions]


def _rows(plan):
    return {r["resn"]: r for r in plan["components"]}


def test_retinal_lysine_merges_into_charmm_lyr_and_is_noncanonical_in_amber():
    top, pos = _load("1C3W_retinal_lys216.pdb")
    charmm = _rows(cl.resolve_components(top, pos, family="charmm36"))["RET"]
    assert charmm["source"] == "library" and charmm["template"] == "LYR" and charmm["charge"] == 1.0
    assert charmm["merge"]["resn"] == "LYS" and charmm["merge"]["resi"] == "216"
    amber = _rows(cl.resolve_components(top, pos, family="amber14"))["RET"]
    assert amber["source"] == "covalent" and amber["buildable"] == "noncanonical"


def test_merged_lyr_builds_in_charmm36(tmp_path):
    rep = cl.build_and_validate(os.path.join(DATA, "1C3W_retinal_lys216.pdb"), family="charmm36",
                                minimize_steps=50, out_dir=str(tmp_path))
    assert rep["ok"], rep.get("error")
    from pymol import cmd

    cmd.delete("all")
    cmd.load(rep["pdb"], "lyr")
    assert cmd.count_atoms("lyr and resi 216 and name C15") == 1          # retinal atoms now in residue 216
    assert 1.2 < cmd.get_distance("lyr and resi 216 and name NZ", "lyr and resi 216 and name C15") < 1.45


def test_geometry_bond_orders_recover_the_protonated_schiff_base_polyene():
    top, pos = _load("1C3W_retinal_lys216.pdb")
    lys = [r for r in top.residues() if r.name == "LYS" and r.id == "216"][0]
    ret = [r for r in top.residues() if r.name == "RET"][0]
    from rdkit.Geometry import Point3D

    atoms = [a for a in list(lys.atoms()) + list(ret.atoms()) if a.element.symbol != "H"]
    rw = Chem.RWMol()
    idx = {a: rw.AddAtom(Chem.Atom(a.element.symbol)) for a in atoms}
    for res in (lys, ret):
        by = {a.name: a for a in res.atoms()}
        for u, v in cl.residue_heavy_graph(res, pos).edges:
            rw.AddBond(idx[by[u]], idx[by[v]], Chem.BondType.SINGLE)
    nz = next(a for a in lys.atoms() if a.name == "NZ")
    c15 = next(a for a in ret.atoms() if a.name == "C15")
    rw.AddBond(idx[nz], idx[c15], Chem.BondType.SINGLE)
    conf = Chem.Conformer(rw.GetNumAtoms())
    for a, i in idx.items():
        conf.SetAtomPosition(i, Point3D(*[10 * x for x in pos[a.index]]))
    mol = rw.GetMol()
    mol.AddConformer(conf)
    out = cl._bond_orders_from_geometry(mol)
    assert out.GetBondBetweenAtoms(idx[nz], idx[c15]).GetBondType() == Chem.BondType.DOUBLE
    doubles = sum(1 for b in out.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    assert doubles == 7                     # C=N + 5 polyene C=C + ring C5=C6 ... + backbone C=O
    prot, done = cl._protonate_lysine_schiff_base(Chem.AddHs(out, addCoords=True), idx[nz])
    assert done and prot.GetAtomWithIdx(idx[nz]).GetFormalCharge() == 1 and prot.GetAtomWithIdx(idx[nz]).GetTotalNumHs(includeNeighbors=True) == 1


def test_acetyl_lysine_builds_as_noncanonical_residue_in_amber(tmp_path):
    top, pos = _load("GGK_ac_GG.pdb")
    assert _rows(cl.resolve_components(top, pos, family="amber14"))["ALY"]["buildable"] == "noncanonical"
    rep = cl.build_and_validate(os.path.join(DATA, "GGK_ac_GG.pdb"), family="amber14", minimize_steps=50,
                                out_dir=str(tmp_path))
    assert rep["ok"], rep.get("error")
    nc = rep["noncanonical_residues"][0]
    assert nc["residue"] == "ALY3A" and nc["formal_charge"] == 0 and nc["missing_terms"] == 0
    assert abs(rep["total_charge"] - round(rep["total_charge"])) < 1e-3
    names = [line[17:20] for line in open(rep["pdb"]) if line.startswith(("ATOM", "HETATM"))]
    assert "ALY" in names                                   # PDB name restored after protonation


def test_charmm_aly_template_is_not_offered_because_it_lacks_the_backbone_co_bond():
    top, pos = _load("GGK_ac_GG.pdb")
    row = _rows(cl.resolve_components(top, pos, family="charmm36"))["ALY"]
    assert row["source"] == "modified_residue" and not row.get("buildable")
