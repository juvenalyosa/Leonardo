"""Monte Carlo receptor ensembles on a real protein: what moves, what is kept, reproducibility."""
import os

import numpy as np
import pytest

pytest.importorskip("pdbfixer")
app = pytest.importorskip("openmm.app")

from leonardo import mc_ensemble_runtime as mc  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "components")


def _xyz(line):
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


@pytest.fixture(scope="module")
def ubiquitin_variant(tmp_path_factory):
    """1UBQ with Met1 as an in-chain selenomethionine, a free MSE far away, and a zinc ion near Ile44."""
    lines = [ln for ln in open(os.path.join(DATA, "1UBQ.pdb")) if ln.startswith(("ATOM", "HETATM"))]
    protein = [ln for ln in lines if ln.startswith("ATOM")]
    out, met1 = [], []
    for ln in protein:
        if ln[17:20] == "MET" and ln[22:26].strip() == "1":
            name = " SE " if ln[12:16] == " SD " else ln[12:16]
            elem = "SE" if name == " SE " else ln[76:78]
            new = "HETATM" + ln[6:12] + name + ln[16:17] + "MSE" + ln[20:76] + f"{elem:>2}" + ln[78:]
            out.append(new)
            met1.append(new)
        else:
            out.append(ln)
    for ln in met1:                                     # a free selenomethionine 40 A away
        x = _xyz(ln) + np.array([40.0, 0.0, 0.0])
        out.append(ln[:22] + " 900" + ln[26:30] + f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}" + ln[54:])
    ile44 = np.array([_xyz(ln) for ln in protein if ln[17:20] == "ILE" and ln[22:26].strip() == "44"])
    zn = ile44.mean(axis=0) + np.array([0.0, 0.0, 7.0])
    out.append(f"HETATM 9999 ZN    ZN A 950    {zn[0]:8.3f}{zn[1]:8.3f}{zn[2]:8.3f}  1.00 20.00          ZN\n")
    path = tmp_path_factory.mktemp("ubq") / "ubq_variant.pdb"
    path.write_text("".join(out) + "END\n")
    site = mc.site_residues_near(str(path), ile44, 6.0)
    return str(path), site, zn


def _run(pdb, site, out_dir, seed=5):
    return mc.sample_ensemble(pdb, str(out_dir), n_structures=2, site_residues=site, steps_per_structure=3,
                              temperature_k=600.0,
                              seed=seed, platform="CPU", reproducible=True, progress=lambda *_: None)


def test_ensemble_moves_only_the_site_and_keeps_hetero_groups(ubiquitin_variant, tmp_path):
    pdb, site, zn = ubiquitin_variant
    res = _run(pdb, site, tmp_path / "a")
    assert res["ok"] and res["produced"] == res["requested"] == 2 and not res.get("warning")
    # the in-chain MSE becomes MET; the free one stays a hetero group (and an obstacle) with the zinc
    assert res["replaced_nonstandard"] == ["MSE1->MET"]
    assert res["fixed_hetero_atoms"] == 1 + 8
    from openmm import unit

    start = app.PDBFile(res["start_structure"])
    xs = np.array([tuple(p) for p in start.positions.value_in_unit(unit.nanometer)])
    site_keys = {(c.strip(), r.strip()) for c, r in site}
    for member in res["structures"]:
        text = open(member["path"]).read()
        hets = [ln for ln in text.splitlines() if ln.startswith("HETATM")]
        zn_line = next(ln for ln in hets if ln[17:20] == " ZN")
        assert np.allclose(_xyz(zn_line), zn, atol=1e-3)                  # the ion is written back, unmoved
        assert any(ln[17:20] == "MSE" and ln[22:26].strip() == "900" for ln in hets)
        assert not any(ln[17:20] == "MSE" and ln[22:26].strip() == "1" for ln in hets)
        assert any(ln.startswith("ATOM") and ln[17:20] == "MET" and ln[22:26].strip() == "1"
                   for ln in text.splitlines())
        m = app.PDBFile(member["path"])
        xm = np.array([tuple(p) for p in m.positions.value_in_unit(unit.nanometer)])
        protein = [a for a in m.topology.atoms() if a.residue.name not in ("ZN", "MSE")]
        frozen = [a.index for a in protein
                  if (a.residue.chain.id.strip(), a.residue.id.strip()) not in site_keys or a.name in mc.BACKBONE]
        assert np.abs(xm[frozen] - xs[frozen]).max() < 1.1e-4                # nm: only site side chains move
    assert 0.0 < res["acceptance"] < 1.0


def test_same_seed_gives_the_same_ensemble(ubiquitin_variant, tmp_path):
    pdb, site, _zn = ubiquitin_variant
    a = _run(pdb, site, tmp_path / "a", seed=9)
    b = _run(pdb, site, tmp_path / "b", seed=9)
    assert [s["energy_kj_mol"] for s in a["structures"]] == [s["energy_kj_mol"] for s in b["structures"]]
    assert open(a["structures"][-1]["path"]).read() == open(b["structures"][-1]["path"]).read()


def test_hetero_records_skip_waters_excluded_and_in_chain_residues(tmp_path):
    pdb = tmp_path / "h.pdb"
    pdb.write_text(
        "HETATM    1 ZN    ZN A 301       1.000   2.000   3.000  1.00  0.00          ZN\n"
        "HETATM    2  O   HOH A 401       0.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    3  C1  LIG A 501       5.000   5.000   5.000  1.00  0.00           C\n"
        "HETATM    4  N   MSE A   7       9.000   9.000   9.000  1.00  0.00           N\n")
    lines, atoms = mc._hetero_records(str(pdb), exclude=[("A", "501", "LIG"), ("A", "7", "MSE")],
                                      drop_from_lines=[("A", "7", "MSE")])
    assert [a[0] for a in atoms] == ["Zn"] and np.allclose(atoms[0][1], [0.1, 0.2, 0.3])
    assert len(lines) == 3 and not any("MSE" in ln for ln in lines)


def test_holo_option_keeps_the_ligand_space_as_a_soft_obstacle(ubiquitin_variant, tmp_path):
    pdb, site, zn = ubiquitin_variant
    # a two-carbon "ligand" next to the site; the free MSE also gives carbon obstacles (OBS_C next to LIG_C)
    lig = [zn + np.array([2.0, 0.0, 0.0]), zn + np.array([3.4, 0.0, 0.0])]
    holo = tmp_path / "holo.pdb"
    body = open(pdb).read().replace("END\n", "")
    for i, x in enumerate(lig):
        body += f"HETATM{9990 + i:5d}  C{i + 1}  LIG A 960    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00 20.00           C\n"
    holo.write_text(body + "END\n")
    res = mc.sample_ensemble(str(holo), str(tmp_path / "out"), n_structures=2, site_residues=site,
                             steps_per_structure=3, temperature_k=600.0, seed=5, platform="CPU", reproducible=True,
                             progress=lambda *_: None, exclude_hetero=[("A", "960", "LIG")], ligand_obstacle=True)
    assert res["ok"] and res["produced"] == 2
    assert res["ligand_space_kept"] and res["ligand_obstacle_atoms"] == 2
    assert res["fixed_hetero_atoms"] == 1 + 8                            # the ligand is not a cofactor obstacle
    for member in res["structures"]:
        lines = [ln for ln in open(member["path"]) if ln.startswith("HETATM") and ln[17:20] == "LIG"]
        assert len(lines) == 2 and np.allclose(_xyz(lines[0]), lig[0], atol=1e-3)
