import numpy as np
from rdkit import Chem

from run_abfe_benchmark_probe_check import aggregate_probe_state_ensemble_payload
from run_abfe_benchmark_probe_check import enumerate_ligand_state_candidates
from run_abfe_benchmark_probe_check import select_probe_state_candidates_with_prefilter


def test_enumerate_ligand_state_candidates_returns_unique_states():
    mol = Chem.AddHs(Chem.MolFromSmiles("NC(=N)N"))
    states = enumerate_ligand_state_candidates(mol, target_ph=7.4, max_states=3, max_tautomers=4)
    assert len(list(states)) >= 1
    labels = [str(row.get("label", "")) for row in list(states)]
    smiles = [str(row.get("smiles", "")) for row in list(states)]
    assert len(labels) == len(set(labels))
    assert len(smiles) == len(set(smiles))
    assert all(Chem.Mol(row.get("mol")) is not None for row in list(states))


def test_aggregate_probe_state_ensemble_payload_prefers_accepted_states():
    payload = aggregate_probe_state_ensemble_payload(
        [
            {
                "state_label": "s1",
                "prior_weight": 1.0,
                "free_energy": {
                    "selected_estimator": "gbsa_obc2_hold_window_expavg",
                    "dg_selected_kj_mol": -20.0,
                    "reweight_confidence_quality": "high",
                },
            },
            {
                "state_label": "s2",
                "prior_weight": 0.5,
                "free_energy": {
                    "selected_estimator": "gbsa_obc2_hold_window_expavg",
                    "dg_selected_kj_mol": -10.0,
                    "reweight_confidence_quality": "medium",
                },
            },
            {
                "state_label": "s3",
                "prior_weight": 0.25,
                "free_energy": {
                    "selected_estimator": "gated_off",
                    "dg_selected_kj_mol": np.nan,
                    "reweight_confidence_quality": "low",
                },
            },
        ],
        temperature_k=300.0,
    )
    assert payload["ok"] is True
    assert str(payload["selected_estimator"]) == "weighted_state_ensemble"
    assert int(payload["accepted_state_count"]) == 2
    assert str(payload["best_state_label"]) == "s1"
    assert np.isfinite(float(payload["dg_selected_kj_mol"]))
    assert float(payload["dg_selected_kj_mol"]) <= -10.0


def test_select_probe_state_candidates_with_prefilter_keeps_input_and_same_charge_best(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "run_abfe_benchmark_probe_check.build_initial_pose_state_prefilter_payload",
        lambda **kwargs: {
            "ok": True,
            "score_kj": {
                "input": -10.0,
                "tautomer": -15.0,
                "protomer": -40.0,
            }[str(kwargs["ligand_mol"].GetProp("_ensemble_source"))],
            "rescored_total_kj": {
                "input": -10.0,
                "tautomer": -15.0,
                "protomer": -40.0,
            }[str(kwargs["ligand_mol"].GetProp("_ensemble_source"))],
            "raw_total_kj": -5.0,
            "formal_charge": {
                "input": -1,
                "tautomer": -1,
                "protomer": -2,
            }[str(kwargs["ligand_mol"].GetProp("_ensemble_source"))],
            "charge_diff": {
                "input": 0,
                "tautomer": 0,
                "protomer": 1,
            }[str(kwargs["ligand_mol"].GetProp("_ensemble_source"))],
            "contact_count_4a": 6,
            "min_dist_a": 2.2,
        },
    )

    def _mol(tag):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        mol.SetProp("_ensemble_source", tag)
        return mol

    payload = select_probe_state_candidates_with_prefilter(
        [
            {"label": "01_input", "source": "input_original", "formal_charge": -1, "prior_weight": 1.0, "mol": _mol("input")},
            {"label": "02_taut", "source": "tautomer", "formal_charge": -1, "prior_weight": 0.5, "mol": _mol("tautomer")},
            {"label": "03_prot", "source": "input+protomer", "formal_charge": -2, "prior_weight": 0.25, "mol": _mol("protomer")},
        ],
        protein_pdb=tmp_path / "protein.pdb",
        prefilter_root=tmp_path / "prefilter",
        antechamber_bin="antechamber",
        parmchk2_bin="parmchk2",
        tleap_bin="tleap",
        input_formal_charge=-1,
        keep_count=2,
        charge_tolerance=0,
    )
    selected = [str(row.get("label")) for row in list(payload.get("selected", []))]
    assert selected == ["01_input", "02_taut"]
