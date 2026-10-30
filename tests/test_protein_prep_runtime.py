"""Automatic protein preparation (protein_prep_runtime): the pieces on synthetic input, then real entries
(vendored, trimmed RCSB files in tests/data/prep): lysozyme 2LZT (Glu35 pKa ~6.2), RNase A 7RSA (alternate
locations), EGFR kinase 1M17 (a 12-residue loop missing), PKA 1ATP (phosphorylated Thr197/Ser338, Mn-ATP), Zif268
1AAY (zinc sites, DNA) and the covalent ibrutinib-Cys481 pocket of 5P9J."""
import json
import os

import numpy as np
import pytest

pytest.importorskip("pdbfixer")
pytest.importorskip("openmm")

from leonardo import protein_prep_runtime as P  # noqa: E402

HERE = os.path.dirname(__file__)
PREP = os.path.join(HERE, "data", "prep")


def _atom(serial, name, resn, chain, resi, xyz, alt=" ", occ=1.0, el=""):
    el = el or name.strip()[0]
    return (f"ATOM  {serial:5d} {name:<4}{alt}{resn:>3} {chain}{resi:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"{occ:6.2f}{20.0:6.2f}          {el:>2}\n")


def _ca(path):
    out = {}
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[16] in " A":
                out[(ln[21], ln[22:27].strip())] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return out


def _residue_atoms(path, chain, resi):
    with open(path) as fh:
        return [ln[12:16].strip() for ln in fh if ln.startswith(("ATOM", "HETATM")) and ln[21] == chain
                and ln[22:27].strip() == str(resi)]


# ------------------------------------------------------------------------------------------- pieces

def test_altloc_keeps_the_most_occupied_conformation():
    lines = [_atom(1, "CA", "SER", "A", 5, (0, 0, 0)),
             _atom(2, "OG", "SER", "A", 5, (1, 0, 0), alt="A", occ=0.3),
             _atom(3, "OG", "SER", "A", 5, (0, 1, 0), alt="B", occ=0.7),
             _atom(4, "CA", "GLY", "A", 6, (3, 0, 0))]
    out, report = P.select_altlocs(lines)
    og = [ln for ln in out if ln[12:16].strip() == "OG"]
    assert len(og) == 1 and float(og[0][38:46]) == 1.0 and og[0][16] == " "
    assert report == [{"residue": "SER5:A", "altlocs": "AB", "kept": "B", "occupancy": 0.7}]


def test_inserted_residues_get_free_numbers_or_insertion_codes():
    assert P._labels_for_insert("964", "977", 12, "internal") == [str(i) for i in range(965, 977)]
    assert P._labels_for_insert("10", "12", 3, "internal") == ["10A", "10B", "10C"]     # no room: codes
    assert P._labels_for_insert("", "672", 6, "N-terminus") == [str(i) for i in range(666, 672)]
    assert P._labels_for_insert("995", "", 3, "C-terminus") == ["996", "997", "998"]


def test_protonation_rule_pka_against_ph():
    import openmm.app as app

    top = app.Topology()
    ch = top.addChain("A")
    for i, name in enumerate(["ASP", "GLU", "HIS", "LYS", "HIS"], start=1):
        top.addResidue(name, ch, str(i))
    pkas = [{"resn": "ASP", "chain": "A", "resi": "1", "pka": 7.9, "model_pka": 3.8},
            {"resn": "GLU", "chain": "A", "resi": "2", "pka": 4.1, "model_pka": 4.5},
            {"resn": "HIS", "chain": "A", "resi": "3", "pka": 7.0, "model_pka": 6.5},
            {"resn": "LYS", "chain": "A", "resi": "4", "pka": 6.5, "model_pka": 10.5},
            {"resn": "HIS", "chain": "A", "resi": "5", "pka": 5.0, "model_pka": 6.5}]
    variants, decisions = P.protonation_plan(top, pkas, {("A", "5"): "HID"}, 7.4)
    assert variants == ["ASH", "GLU", "HIE", "LYN", "HID"]
    variants, _ = P.protonation_plan(top, pkas, {}, 6.0, metal_variants={("A", "3"): "HID"})
    assert variants[2] == "HID"                                            # the metal decides
    assert variants[0] == "ASH" and variants[3] == "LYS"                    # pKa 6.5 > pH 6: still charged


# ------------------------------------------------------------------------------------------- real entries

@pytest.fixture(scope="module")
def lysozyme(tmp_path_factory):
    work = tmp_path_factory.mktemp("lyz")
    out = {}
    for ph in (7.4, 5.0):
        path = str(work / f"lyz_{ph}.pdb")
        out[ph] = (path, P.prepare_protein(os.path.join(PREP, "2LZT.pdb"), path, ph=ph, workdir=str(work / str(ph))))
    return out


def test_lysozyme_glu35_follows_its_pka(lysozyme):
    for ph, (path, report) in lysozyme.items():
        states = {d["residue"]: d for d in report["protonation"]}
        glu35 = states.get("Glu35:A")
        if report["tools"]["propka"]:
            assert 5.5 < glu35["pka"] < 7.2                                 # experiment ~6.2
            assert glu35["state"] == ("GLH" if ph == 5.0 else "GLU")
            assert ("HE2" in _residue_atoms(path, "A", 35)) == (ph == 5.0)
        assert "Asp52:A" not in states or states["Asp52:A"]["state"] == "ASP"   # pKa ~3.7: charged at both


def test_lysozyme_output_is_complete_and_close_to_the_crystal(lysozyme):
    path, report = lysozyme[7.4]
    assert report["hydrogens"] > 900
    c0, c1 = _ca(os.path.join(PREP, "2LZT.pdb")), _ca(path)
    assert set(c0) == set(c1)
    d = np.array([np.linalg.norm(c0[k] - c1[k]) for k in c0])
    assert d.max() < 0.6                                                    # restrained minimization
    with open(path) as fh:
        text = fh.read()
    assert "HOH" not in text                                               # waters removed by default
    assert "NO3" in text                                                   # hetero groups kept
    rep = json.load(open(os.path.splitext(path)[0] + "_report.json"))
    assert rep["ph"] == 7.4 and rep["steps"]


def test_prepared_states_survive_the_md_preparation(lysozyme, tmp_path):
    """Leonardo's MD / component path rebuilds hydrogens: the states set by the preparation are kept."""
    import openmm.app as app

    path, report = lysozyme[5.0]
    if not report["tools"]["propka"]:
        pytest.skip("PROPKA not installed")
    apo = tmp_path / "apo.pdb"
    with open(path) as fh, open(apo, "w") as out:
        out.writelines(ln for ln in fh if not ln.startswith("HETATM"))
    from leonardo import component_library_runtime as C

    r = C.prepare_system_for_solvation(str(apo), ph=7.0, gaff_ligands=False, fetch_ccd=False)
    assert r["ok"]
    m = r["modeller"]
    states = {f"{res.name}{res.id}": v for res, v in
              zip(m.topology.residues(), P.variants_from_hydrogens(m.topology, m.positions)) if v}
    assert states["GLU35"] == "GLH"
    pdb = app.PDBFile(path)
    assert P.variants_from_hydrogens(pdb.topology, pdb.positions).count("GLH") >= 1


def test_altlocs_and_existing_hydrogens_are_handled(tmp_path):
    report = P.prepare_protein(os.path.join(PREP, "7RSA.pdb"), str(tmp_path / "rnase.pdb"), minimize=False)
    assert len(report["altlocs"]) >= 10
    assert any(s["step"] == "Hydrogens" for s in report["steps"])           # the file has H: rebuilt
    with open(tmp_path / "rnase.pdb") as fh:
        assert not any(ln[16] not in (" ", "") for ln in fh if ln.startswith(("ATOM", "HETATM")))


def test_missing_loop_built_without_moving_the_crystal(tmp_path):
    report = P.prepare_protein(os.path.join(PREP, "1M17.pdb"), str(tmp_path / "egfr.pdb"), minimize=False,
                               workdir=str(tmp_path / "w"))
    internal = [g for g in report["gaps"] if g["where"] == "internal"]
    assert internal and internal[0]["length"] == 12 and internal[0]["sequence"] == "LPSPTDSNFYRA"
    c0, c1 = _ca(os.path.join(PREP, "1M17.pdb")), _ca(str(tmp_path / "egfr.pdb"))
    new = set(c1) - set(c0)
    assert {k[1] for k in new} == {str(i) for i in range(965, 977)}         # crystal numbering continued
    if report["gap_method"] == "MODELLER":
        # only the loop and its two neighbours (His964, Leu977) move
        d = np.array([np.linalg.norm(c0[k] - c1[k]) for k in c0 if k in c1 and k[1] not in ("964", "977")])
        assert d.max() < 0.01
    # the built loop is attached: consecutive CA 3.8 A apart through the gap (PDBFixer's, unminimized, is rough)
    seq = [c1[("A", str(i))] for i in range(963, 979)]
    gaps = [float(np.linalg.norm(a - b)) for a, b in zip(seq, seq[1:])]
    if report["gap_method"] == "MODELLER":
        assert max(gaps) < 4.3 and min(gaps) > 3.2
    else:                                   # unminimized PDBFixer loops vary run to run (CA-CA up to ~4.6 A)
        assert max(gaps) < 5.0 and any("rough backbone" in w for w in report["warnings"])
    assert ("A", "666") not in c1 and ("A", "996") not in c1                # termini not built by default


def test_gap_left_open_splits_the_chain_and_without_sequence_nothing_is_invented(tmp_path):
    src = os.path.join(PREP, "1M17.pdb")
    noseq = tmp_path / "noseq.pdb"
    with open(src) as fh, open(noseq, "w") as out:
        out.writelines(ln for ln in fh if not ln.startswith("SEQRES"))
    report = P.prepare_protein(str(noseq), str(tmp_path / "split.pdb"), minimize=False)
    assert report["built_residues"] == [] and report.get("chain_breaks")
    with open(tmp_path / "split.pdb") as fh:
        oxt = [ln for ln in fh if ln[12:16].strip() == "OXT"]
    assert len(oxt) >= 2                                                    # a C terminus before the gap too


def test_phosphorylated_residues_are_kept_and_protonated(tmp_path):
    report = P.prepare_protein(os.path.join(PREP, "1ATP.pdb"), str(tmp_path / "pka.pdb"), minimize=False)
    if not P.phospho_forcefield_files():
        pytest.skip("openmmforcefields phosaa14SB not available")
    assert sorted(report["kept_modified"]) == ["SEP338:E", "TPO197:E"]
    atoms = _residue_atoms(str(tmp_path / "pka.pdb"), "E", 197)
    assert {"P", "O1P", "O2P", "O3P", "HB", "HG21"} <= set(atoms)
    assert "HG1" not in atoms                                               # the hydroxyl carries the phosphate
    with open(tmp_path / "pka.pdb") as fh:
        text = fh.read()
    assert " MN " in text and " ATP " in text                               # Mn-ATP kept


def test_zinc_sites_and_dna(tmp_path):
    report = P.prepare_protein(os.path.join(HERE, "data", "macro", "1AAY.pdb"), str(tmp_path / "zif.pdb"),
                               minimize=False)
    states = {d["residue"]: d["state"] for d in report["protonation"]}
    for cys in (107, 112, 137, 140, 165, 168):
        assert states[f"Cys{cys}:A"] == "CYM"                               # thiolates on Zn
    for his in (125, 129, 153, 157, 181, 185):
        assert states[f"His{his}:A"] in ("HID", "HIE")
    with open(tmp_path / "zif.pdb") as fh:
        text = fh.read()
    assert " DG " in text and " ZN " in text


def test_covalent_cysteine_is_a_thioether(tmp_path):
    src = os.path.join(HERE, "data", "components", "5P9J_ibrutinib_cys481.pdb")
    report = P.prepare_protein(src, str(tmp_path / "btk.pdb"), minimize=False)
    assert any(x.startswith("Cys481:A:SG-") for x in report["covalent_links"])
    atoms = _residue_atoms(str(tmp_path / "btk.pdb"), "A", 481)
    assert "HG" not in atoms and "SG" in atoms


def test_analysis_finds_what_the_structure_needs():
    a = P.analyze_structure(os.path.join(PREP, "1M17.pdb"))
    assert [g["length"] for g in a["gaps"] if g["where"] == "internal"] == [12]
    assert {r["residue"] for r in a["altlocs"]} == {"CYS751:A", "ASP831:A"}
    assert [h["resn"] for h in a["hetero_groups"]] == ["AQ4"]
    b = P.analyze_structure(os.path.join(PREP, "1ATP.pdb"))
    assert set(b["nonstandard"]) == {"TPO197:E->THR", "SEP338:E->SER"}
