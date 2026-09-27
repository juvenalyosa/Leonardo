import math

import pytest
from pka_runtime import (
    RT_LN10,
    PkaError,
    apply_calibration,
    compute_pka,
    find_ionizable_sites,
    fit_calibration,
    macroscopic_pkas,
    mean_charge,
    microstate_free_energies,
    microstate_populations,
    protonate_molecule,
    register_backend,
    select_enumerated_sites,
    states_at_ph,
)
from rdkit import Chem
from rdkit.Chem import AllChem


def _classes(smiles):
    return [s["class"] for s in find_ionizable_sites(Chem.MolFromSmiles(smiles))]


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)O", ["carboxylic_aliphatic"]),
    ("OC(=O)c1ccccc1", ["carboxylic_aromatic"]),
    ("Oc1ccccc1", ["phenol"]),
    ("CCS", ["thiol_aliphatic"]),
    ("Sc1ccccc1", ["thiol_aromatic"]),            # not the aliphatic-thiol reference any more
    ("CCN", ["amine1"]),
    ("CCNCC", ["amine2"]),
    ("CCN(CC)CC", ["amine3"]),
    ("Nc1ccccc1", ["aniline"]),
    ("c1ccncc1", ["pyridine"]),
    ("c1c[nH]cn1", ["imidazole"]),
    ("NC(=N)c1ccccc1", ["amidine"]),
    ("NC(N)=N", ["guanidine"]),
    ("CC(=O)NC", []),                             # amide N is not a base
    ("CCO", []),                                  # alcohols are not titrated in 0-14
    ("CC(=O)Nc1ccc(O)cc1", ["phenol"]),           # anilide N is not an aniline
])
def test_site_detection(smiles, expected):
    assert _classes(smiles) == expected


def test_charged_input_is_neutralized_before_matching():
    assert _classes("CC[NH3+]") == ["amine1"] and _classes("CC(=O)[O-]") == ["carboxylic_aliphatic"]


def test_difficult_cases_are_flagged():
    nitrophenol = find_ionizable_sites(Chem.MolFromSmiles("Oc1ccc(cc1)[N+](=O)[O-]"))[0]
    assert "resonance_acceptor" in nitrophenol["flags"]
    assert "resonance_acceptor" not in find_ionizable_sites(Chem.MolFromSmiles("Oc1ccc(C)cc1"))[0]["flags"]
    assert "tertiary_amine" in find_ionizable_sites(Chem.MolFromSmiles("CN1CCOCC1"))[0]["flags"]
    assert "polyaza_ring" in find_ionizable_sites(Chem.MolFromSmiles("c1cncnc1"))[0]["flags"]


def test_single_site_populations_match_henderson_hasselbalch():
    energies = microstate_free_energies([4.76])
    for ph in (3.0, 4.76, 7.4):
        frac_protonated = microstate_populations(energies, ph)[(True,)]
        assert frac_protonated == pytest.approx(1.0 / (1.0 + 10 ** (ph - 4.76)), rel=1e-9)
    assert macroscopic_pkas(energies) == [4.76]


def test_two_identical_sites_split_macroscopic_pkas_by_log4():
    # statistical factor: pKa1 = pK - log10(2), pKa2 = pK + log10(2)
    macro = macroscopic_pkas(microstate_free_energies([4.5, 4.5]))
    assert macro == pytest.approx([4.5 - math.log10(2), 4.5 + math.log10(2)], abs=0.01)


def test_coupled_sites_use_microstate_pkas():
    # site 1 is 3 units harder to deprotonate once site 0 is deprotonated (diacid-like)
    def micro(partial, s):
        return 4.0 if s == 0 else (4.5 if partial[0] else 7.5)
    energies = microstate_free_energies([4.0, 4.5], micro_pka_fn=micro)
    macro = macroscopic_pkas(energies)
    assert macro[0] < 4.5 and macro[1] > 6.0


def test_protonation_at_ph_7_4():
    cases = {
        "NC(=N)c1ccccc1": ("NC(=[NH2+])c1ccccc1", 1),
        "CCN": ("CC[NH3+]", 1),
        "c1ccncc1": ("c1ccncc1", 0),
        "CC(=O)O": ("CC(=O)[O-]", -1),
        "NCC(=O)O": ("[NH3+]CC(=O)[O-]", 0),          # zwitterion
        "OC(=O)CCC(=O)O": ("O=C([O-])CCC(=O)[O-]", -2),
        "C[N+](C)(C)C": ("C[N+](C)(C)C", 1),            # quaternary ammonium untouched
    }
    for smi, (want, q) in cases.items():
        _mol, info = protonate_molecule(Chem.MolFromSmiles(smi), 7.4)
        assert info["smiles"] == want and info["charge"] == q, smi


def test_ph_dependence_of_imidazole():
    res = compute_pka(Chem.MolFromSmiles("c1c[nH]cn1"))
    assert mean_charge(res, 4.0) > 0.99 and mean_charge(res, 9.0) < 0.01
    top = states_at_ph(res, 6.9, min_population=0.0)
    assert top[0]["population"] == pytest.approx(0.5, abs=0.01)


def test_heavy_atom_coordinates_are_kept():
    mol = Chem.AddHs(Chem.MolFromSmiles("NC(=N)c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    heavy_in = Chem.RemoveHs(mol)
    out, info = protonate_molecule(mol, 7.4)
    assert info["charge"] == 1
    heavy_out = Chem.RemoveHs(out)
    assert heavy_out.GetNumAtoms() == heavy_in.GetNumAtoms()
    for i in range(heavy_in.GetNumAtoms()):
        a = heavy_in.GetConformer().GetAtomPosition(i)
        b = heavy_out.GetConformer().GetAtomPosition(i)
        assert heavy_in.GetAtomWithIdx(i).GetSymbol() == heavy_out.GetAtomWithIdx(i).GetSymbol()
        assert (a - b).Length() < 1e-6
    assert out.GetNumAtoms() - heavy_out.GetNumAtoms() == 9       # C6H5 + NH2 + =NH2+ hydrogens


def test_many_sites_are_capped_with_a_warning():
    sites = [{"atom": i, "class": "carboxylic_aliphatic", "kind": "acid", "empirical_pka": 4.0 + i, "empirical_sd": 1}
             for i in range(6)]
    enumerated, fixed, warnings = select_enumerated_sites(sites, focus_ph=5.0, max_sites=4)
    assert len(enumerated) == 4 and len(fixed) == 2 and warnings
    assert {s["atom"] for s in enumerated} == {0, 1, 2, 3}


def test_cache_returns_the_stored_result(tmp_path):
    class SlowQM:
        name = "slowqm"
        calibration_version = "t1"

        def site_pkas(self, parent, sites, **_):
            return [{"pka": 9.0, "sd": 0.5, "source": "slowqm"} for _s in sites]

    register_backend("slowqm", SlowQM)
    mol = Chem.MolFromSmiles("CCN")
    first = compute_pka(mol, method="slowqm", cache_dir=str(tmp_path))
    assert first["provenance"]["cache"] == "miss"
    second = compute_pka(mol, method="slowqm", cache_dir=str(tmp_path))
    assert second["provenance"]["cache"] == "hit" and second["sites"] == first["sites"]
    # the empirical method is milliseconds per molecule: nothing is written (virtual screening)
    compute_pka(Chem.MolFromSmiles("CCCN"), cache_dir=str(tmp_path / "emp"))
    assert not (tmp_path / "emp").exists()


def test_custom_backend_and_unknown_method():
    class FakeQM:
        name = "fake"

        def site_pkas(self, parent, sites, **_):
            return [{"pka": 3.0, "sd": 0.5, "source": "fake"} for _s in sites]

    register_backend("fake", FakeQM)
    _mol, info = protonate_molecule(Chem.MolFromSmiles("CCN"), 7.4, method="fake")
    assert info["method"] == "fake" and info["charge"] == 0          # pKa 3 -> neutral amine
    with pytest.raises(PkaError):
        compute_pka(Chem.MolFromSmiles("CCN"), method="no-such-method")


def test_calibration_per_class_with_global_fallback():
    rows = [{"class": "phenol", "x": x, "y": 0.5 * x} for x in (-4, -2, -1, 1, 2)]
    rows += [{"class": "pyridine", "x": x, "y": 0.9 * x} for x in (-2, -1, 1, 2, 3)]
    rows += [{"class": "thiol_aromatic", "x": 1.0, "y": 0.6}]
    cal = fit_calibration(rows, min_class_n=5)
    assert cal["classes"]["phenol"]["own"] and cal["classes"]["phenol"]["slope"] == pytest.approx(0.5)
    assert cal["classes"]["pyridine"]["slope"] == pytest.approx(0.9)
    assert not cal["classes"]["thiol_aromatic"]["own"]
    pka, err = apply_calibration(cal, "pyridine", 2.0, 5.23)
    assert pka == pytest.approx(5.23 + 1.8) and err == pytest.approx(0.0, abs=1e-9)
    assert RT_LN10 == pytest.approx(1.364, abs=0.001)


# --------------------------------------------------------------------------- #
# Phase 2: flags, shrinkage, PM7 backend
# --------------------------------------------------------------------------- #
def test_new_difficulty_flags():
    two_cl = find_ionizable_sites(Chem.MolFromSmiles("Clc1ccccn1"))[0]
    assert "ortho_heteroatom" in two_cl["flags"]
    assert "ortho_heteroatom" not in find_ionizable_sites(Chem.MolFromSmiles("Cc1ccccn1"))[0]["flags"]
    acyl = find_ionizable_sites(Chem.MolFromSmiles("CC(=O)NC(N)=N"))[0]
    assert acyl["class"] == "guanidine" and "acyl_substituted" in acyl["flags"]
    assert "acyl_substituted" not in find_ionizable_sites(Chem.MolFromSmiles("NC(N)=N"))[0]["flags"]


def test_shrinkage_pulls_outlier_class_towards_global_slope():
    from pka_runtime import choose_shrinkage

    rows = [{"class": "phenol", "x": x, "y": 0.5 * x} for x in (-4, -3, -2, -1, 1, 2, 3)]
    rows += [{"class": "aniline", "x": x, "y": 0.5 * x} for x in (-2, -1, 1, 2)]
    rows += [{"class": "aniline", "x": -6.0, "y": -1.0}]          # one outlier drags the class slope
    free = fit_calibration(rows, min_class_n=5)["classes"]["aniline"]["slope"]
    shrunk = fit_calibration(rows, shrinkage=30.0)["classes"]["aniline"]["slope"]
    g = fit_calibration(rows)["global"]["slope"]
    assert abs(shrunk - g) < abs(free - g)
    assert choose_shrinkage(rows) is not None


def _fake_runner(hofs, fail_first=()):
    """MOPAC stand-in: heat of formation from the net charge in the input line."""
    calls = []

    def run(text, timeout=0):
        calls.append(text)
        first = text.splitlines()[0]
        charge = int(first.split("CHARGE=")[1].split()[0])
        if "GNORM=1.0" in first and charge in fail_first:
            return " ** EXCESS NUMBER OF OPTIMIZATION CYCLES **\n"
        return f"          FINAL HEAT OF FORMATION =   {hofs[charge]:.5f} KCAL/MOL\n"
    run.calls = calls
    return run


def test_mopac_heat_of_formation_parser():
    from pka_runtime import parse_mopac_heat_of_formation

    assert parse_mopac_heat_of_formation("  FINAL HEAT OF FORMATION =  -206.41478 KCAL/MOL = -863 KJ/MOL") == -206.41478
    assert parse_mopac_heat_of_formation("EXCESS NUMBER OF OPTIMIZATION CYCLES\nFINAL HEAT OF FORMATION = -1.0 KCAL/MOL") is None
    assert parse_mopac_heat_of_formation("") is None


def test_pm7_backend_retries_non_converged_states_and_is_relative_to_reference():
    from pka_runtime import MopacBackend

    # same energies for molecule and reference -> pKa equals the reference pKa (4.76)
    runner = _fake_runner({0: -111.0, -1: -206.0}, fail_first=(-1,))
    backend = MopacBackend(calibration={"global": {"slope": 1.0, "rmse": 0.5}, "classes": {}, "version": "t"},
                           n_conformers=2, workers=1, runner=runner)
    res = compute_pka(Chem.MolFromSmiles("CC(=O)O"), method="pm7", backend_options={"runner": runner,
                      "n_conformers": 2, "workers": 1, "calibration": backend.calibration})
    site = res["sites"][0]
    assert res["method"] == "pm7" and site["pka"] == pytest.approx(4.76) and site["pka_source"] == "pm7_cosmo"
    assert any("GNORM=5.0" in c for c in runner.calls)          # anion retried with looser settings


def test_pm7_failure_falls_back_to_empirical_with_warning():
    def broken(text, timeout=0):
        return ""
    res = compute_pka(Chem.MolFromSmiles("CCN"), method="pm7",
                      backend_options={"runner": broken, "n_conformers": 1, "workers": 1})
    assert res["method"] == "empirical" and any("pm7 failed" in w for w in res["warnings"])


def test_shipped_calibration_is_loadable():
    from pka_runtime import load_calibration

    cal = load_calibration("pm7")
    assert cal is not None and cal["classes"] and cal["version"].startswith("pm7-")
    assert all(0.2 < c["slope"] < 2.0 for c in cal["classes"].values())


def _has_mopac():
    import shutil

    return shutil.which("mopac") is not None


@pytest.mark.skipif(not _has_mopac(), reason="MOPAC not installed")
def test_pm7_real_calculations():
    acetic = compute_pka(Chem.MolFromSmiles("CC(=O)O"), method="pm7")
    assert acetic["sites"][0]["pka"] == pytest.approx(4.76, abs=0.01)          # the reference itself
    _mol, benz = protonate_molecule(Chem.MolFromSmiles("NC(=N)c1ccccc1"), 7.4, method="pm7")
    assert benz["method"] == "pm7" and benz["charge"] == 1
    assert 10.0 < benz["sites"][0]["pka"] < 13.5                                # exp 11.6
    _mol, gly = protonate_molecule(Chem.MolFromSmiles("NCC(=O)O"), 7.4, method="pm7")
    assert gly["smiles"] == "[NH3+]CC(=O)[O-]"                                  # zwitterion from coupled microstates


def test_pm7_not_recommended_class_uses_empirical_with_warning():
    runner = _fake_runner({0: -9.9, 1: 76.0})
    cal = {"global": {"slope": 1.0, "rmse": 0.5}, "version": "t",
           "classes": {"amine3": {"slope": 0.9, "rmse": 2.2, "n": 20, "pm7_recommended": False}}}
    res = compute_pka(Chem.MolFromSmiles("CCN(CC)CC"), method="pm7",
                      backend_options={"runner": runner, "n_conformers": 1, "workers": 1, "calibration": cal})
    site = res["sites"][0]
    assert site["pka"] == pytest.approx(site["empirical_pka"])
    assert "not recommended" in site["pka_source"]
    assert any("less accurate than the class-typical" in w for w in res["warnings"])


def test_shipped_calibration_records_data_provenance():
    import json
    import os

    import pka_runtime

    with open(os.path.join(os.path.dirname(pka_runtime.__file__), "pka_calibration.json")) as fh:
        pm7 = json.load(fh)["methods"]["pm7"]
    assert pm7["robust"] is True and len(pm7["data_sha256"]) == 64
    assert pm7["classes"]["amine3"]["pm7_recommended"] is False           # decided on the calibration set


# --------------------------------------------------------------------------- #
# Psi4 backend (single points on PM7 geometries)
# --------------------------------------------------------------------------- #
def _fake_mopac_with_geometry(hofs):
    """MOPAC stand-in that also prints the input geometry as the optimized one."""
    def run(text, timeout=0):
        lines = text.splitlines()
        charge = int(lines[0].split("CHARGE=")[1].split()[0])
        atoms = [ln.split() for ln in lines[3:] if ln.strip()]
        geom = "\n".join(f"  {i + 1:3d}    {a[0]:2s} {float(a[1]):14.9f} {float(a[3]):14.9f} {float(a[5]):14.9f}"
                         for i, a in enumerate(atoms))
        return (f"          FINAL HEAT OF FORMATION =   {hofs[charge]:.5f} KCAL/MOL\n\n"
                f"                             CARTESIAN COORDINATES\n\n{geom}\n\n\n Empirical Formula\n")
    return run


def _fake_psi4(energies):
    """Psi4 stand-in: energy (hartree) from the charge line of the molecule block."""
    calls = []

    def run(text, timeout=0, threads=1, exe=None):
        calls.append(text)
        lines = text.splitlines()
        charge = int(lines[lines.index("molecule {") + 1].split()[0])
        if energies.get(charge) is None:
            return "SCF failed"
        return f"  @DF-RHF Final Energy: {energies[charge]}\n\nLEONARDO_PKA_ENERGY {energies[charge]:.12f}\n"
    run.calls = calls
    return run


def test_psi4_parsers_and_input():
    from pka_runtime import build_psi4_pka_input, parse_mopac_final_geometry, parse_psi4_pka_energy

    out = ("                             CARTESIAN COORDINATES\n\n   1    C   0.0 0.0 0.0\n\n"
           "                             CARTESIAN COORDINATES\n\n"
           "   1    C       -0.659717925    -0.014220958    -0.017668622\n"
           "   2    O        1.420524631     1.098403480     0.237487378\n\n\n  Empirical Formula: C O\n")
    atoms = parse_mopac_final_geometry(out)
    assert atoms == [("C", -0.659717925, -0.014220958, -0.017668622), ("O", 1.420524631, 1.098403480, 0.237487378)]
    assert parse_mopac_final_geometry("no geometry") is None
    text = build_psi4_pka_input(atoms, -1, theory="hf", basis="3-21g")
    assert "-1 1" in text and "basis 3-21g" in text and 'energy("hf")' in text and "Area = 1" in text
    assert parse_psi4_pka_energy("LEONARDO_PKA_ENERGY -228.5\nLEONARDO_PKA_ENERGY -227.123456789012") == -227.123456789012
    assert parse_psi4_pka_energy("") is None


def test_psi4_backend_uses_pm7_geometry_and_is_relative_to_reference():
    psi4 = _fake_psi4({0: -227.0, -1: -226.5})
    opts = {"runner": _fake_mopac_with_geometry({0: -111.0, -1: -206.0}), "psi4_runner": psi4, "n_conformers": 2,
            "workers": 2, "threads": 1, "theory": "hf", "basis": "3-21g",
            "calibration": {"global": {"slope": 1.0, "rmse": 0.7}, "classes": {}, "version": "t",
                            "theory": "hf", "basis": "3-21g"}}
    res = compute_pka(Chem.MolFromSmiles("CC(=O)O"), method="psi4", backend_options=opts)
    site = res["sites"][0]
    assert res["method"] == "psi4" and site["pka"] == pytest.approx(4.76)       # same energies as the reference
    assert site["pka_source"] == "psi4_hf_3-21g_pcm"
    # 2 states of the molecule; acetic acid is also the class reference -> no duplicate single points
    assert len(psi4.calls) == 2
    assert res["provenance"]["calibration_version"] == "t|hf/3-21g"


def test_psi4_backend_never_uses_the_pm7_calibration():
    from pka_runtime import Psi4Backend

    backend = Psi4Backend(calibration={"global": {"slope": 0.5}, "classes": {}, "version": "x",
                                       "theory": "b3lyp", "basis": "6-31g*"}, theory="hf", basis="3-21g")
    assert backend.calibration is None and backend.calibration_version == "uncalibrated|hf/3-21g"
    res = compute_pka(Chem.MolFromSmiles("CCN"), method="psi4", backend_options={
        "runner": _fake_mopac_with_geometry({0: -10.0, 1: 140.0}), "psi4_runner": _fake_psi4({0: -134.0, 1: -134.4}),
        "n_conformers": 1, "workers": 1, "threads": 1, "theory": "hf", "basis": "sto-3g"})
    assert any("Psi4 pKa values are uncalibrated" in w for w in res["warnings"])


def test_psi4_failure_falls_back_to_empirical_with_warning():
    res = compute_pka(Chem.MolFromSmiles("CCN"), method="psi4", backend_options={
        "runner": _fake_mopac_with_geometry({0: -10.0, 1: 140.0}), "psi4_runner": _fake_psi4({0: -134.0, 1: None}),
        "n_conformers": 1, "workers": 1, "threads": 1})
    assert res["method"] == "empirical" and any("psi4 failed" in w for w in res["warnings"])


def _has_psi4():
    from pka_runtime import find_psi4_executable

    return find_psi4_executable() is not None and _has_mopac()


@pytest.mark.skipif(not _has_psi4(), reason="Psi4 or MOPAC not installed")
def test_psi4_real_calculation():
    acetic = compute_pka(Chem.MolFromSmiles("CC(=O)O"), method="psi4", backend_options={"n_conformers": 2})
    assert acetic["method"] == "psi4"
    assert acetic["sites"][0]["pka"] == pytest.approx(4.76, abs=0.01)          # the reference itself
