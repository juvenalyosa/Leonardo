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
    mol = Chem.MolFromSmiles("CCN")
    first = compute_pka(mol, cache_dir=str(tmp_path))
    assert first["provenance"]["cache"] == "miss"
    second = compute_pka(mol, cache_dir=str(tmp_path))
    assert second["provenance"]["cache"] == "hit" and second["sites"] == first["sites"]


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
