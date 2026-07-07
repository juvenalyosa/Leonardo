import numpy as np

from trajectory_interaction_runtime import trajectory_build_selected_trace_payload
from trajectory_interaction_runtime import trajectory_collect_interaction_occupancy
from trajectory_interaction_runtime import trajectory_collect_presence_counts
from trajectory_interaction_runtime import trajectory_frame_indices
from trajectory_interaction_runtime import trajectory_interaction_key
from trajectory_interaction_runtime import trajectory_interaction_label
from trajectory_interaction_runtime import trajectory_interaction_occ_key
from trajectory_interaction_runtime import trajectory_normalize_max_distance
from trajectory_interaction_runtime import trajectory_normalize_step
from trajectory_interaction_runtime import trajectory_pick_matching_interaction
from trajectory_interaction_runtime import trajectory_run_interaction_analysis


def test_trajectory_frame_indices_sanitizes_step_and_keeps_last_frame():
    assert trajectory_normalize_step(0) == 1
    assert trajectory_normalize_step("bad") == 1
    assert trajectory_frame_indices(10, step=4) == [1, 5, 9, 10]
    assert trajectory_frame_indices(10, step=4, include_last=False) == [1, 5, 9]
    assert trajectory_frame_indices("bad", step=2) == []
    assert trajectory_normalize_max_distance(0.0) is None
    assert trajectory_normalize_max_distance("x") is None
    assert trajectory_normalize_max_distance(4.5) == 4.5


def test_trajectory_interaction_keys_and_labels_are_stable_for_clusters_and_rings():
    hyd = {
        "type": "Hydrophobic",
        "prot_res": "VAL42",
        "prot_atom": "CG1,CG2",
        "lig_atom": "C1,C2",
        "details": "Cluster",
    }
    ring = {"type": "Pi-Pi Stacking", "prot_res": "PHE90", "prot_atom": "Ring", "lig_atom": "Ring"}
    hbond = {"type": "Hydrogen Bond", "prot_res": "ASP12", "prot_atom": "OD1", "lig_atom": "N1"}

    assert trajectory_interaction_key(hbond) == ("Hydrogen Bond", "ASP12", "OD1", "N1")
    assert trajectory_interaction_occ_key(hyd, coarse=True) == ("cluster", "Hydrophobic", "VAL42")
    assert trajectory_interaction_label(hyd) == "VAL42-Cluster"
    assert trajectory_interaction_occ_key(ring, coarse=True) == ("ring", "Pi-Pi Stacking", "PHE90")
    assert trajectory_interaction_label(ring) == "PHE90-Ring"


def test_trajectory_collect_interaction_occupancy_dedupes_per_frame_and_filters_distance():
    hbond_a = {
        "type": "Hydrogen Bond",
        "prot_res": "ASP12",
        "prot_atom": "OD1",
        "lig_atom": "N1",
        "distance": 2.7,
    }
    hbond_duplicate = dict(hbond_a, distance=2.8)
    hbond_far = dict(hbond_a, distance=8.0)
    hyd_a = {
        "type": "Hydrophobic",
        "prot_res": "VAL42",
        "prot_atom": "CG1",
        "lig_atom": "C1",
        "details": "Cluster",
        "distance": 3.8,
    }
    hyd_same_frame = dict(hyd_a, prot_atom="CG2", lig_atom="C2", distance=3.9)

    payload = trajectory_collect_interaction_occupancy(
        [
            [hbond_a, hbond_duplicate, hyd_a, hyd_same_frame],
            [hbond_far],
            [],
            [dict(hyd_a, prot_atom="CB", distance=3.7), {"type": "bad", "distance": "nan"}],
        ],
        max_distance=4.5,
        top_n=3,
        color_map={"Hydrogen Bond": "#00f", "Hydrophobic": "#0f0"},
    )

    assert payload["ok"] is True
    assert int(payload["frames_analyzed"]) == 4
    assert int(payload["skipped_distance"]) == 1
    assert int(payload["skipped_invalid"]) == 1
    hbond_key = ("atom", "Hydrogen Bond", "ASP12", "OD1", "N1")
    hyd_key = ("cluster", "Hydrophobic", "VAL42")
    assert int(payload["counts"][hbond_key]) == 1
    assert int(payload["counts"][hyd_key]) == 2
    assert np.allclose(sorted(np.asarray(payload["all_occ"], dtype=float)), [25.0, 50.0])
    assert payload["top_labels"][0] == "VAL42-Cluster"
    assert payload["top_colors"][0] == "#0f0"


def test_trajectory_run_interaction_analysis_samples_frames_and_restores_state():
    set_calls = []
    calc_calls = []

    def _set_frame(frame_idx):
        set_calls.append(int(frame_idx))

    def _calc(frame_idx):
        calc_calls.append(int(frame_idx))
        if int(frame_idx) == 4:
            return [
                {
                    "type": "Hydrophobic",
                    "prot_res": "VAL42",
                    "prot_atom": "CG1",
                    "lig_atom": "C1",
                    "details": "Cluster",
                    "distance": 3.8,
                }
            ]
        return [
            {
                "type": "Hydrogen Bond",
                "prot_res": "ASP12",
                "prot_atom": "OD1",
                "lig_atom": "N1",
                "distance": 2.7,
            }
        ]

    payload = trajectory_run_interaction_analysis(
        n_states=7,
        step=3,
        calculate_interactions_fn=_calc,
        set_frame_fn=_set_frame,
        get_current_frame_fn=lambda: 5,
        max_distance=4.5,
        top_n=5,
        color_map={"Hydrogen Bond": "#00f", "Hydrophobic": "#0f0"},
    )

    assert payload["ok"] is True
    assert payload["frame_plan"] == [1, 4, 7]
    assert payload["sampled_frames"] == [1, 4, 7]
    assert calc_calls == [1, 4, 7]
    assert set_calls == [1, 4, 7, 5]
    assert int(payload["frames_analyzed"]) == 3
    assert any(label == "ASP12-N1" for label in list(payload["top_labels"]))
    assert np.isclose(max(np.asarray(payload["top_freqs"], dtype=float)), 100.0 * 2.0 / 3.0)


def test_trajectory_run_interaction_analysis_handles_cancel_and_missing_callback():
    missing = trajectory_run_interaction_analysis(n_states=3, calculate_interactions_fn=None)
    assert missing["ok"] is False
    assert missing["reason"] == "calculate_interactions_fn_missing"

    set_calls = []
    calc_calls = []

    def _calc(frame_idx):
        calc_calls.append(int(frame_idx))
        return [{"type": "Hydrogen Bond", "prot_res": "ASP12", "prot_atom": "OD1", "lig_atom": "N1", "distance": 2.8}]

    payload = trajectory_run_interaction_analysis(
        n_states=5,
        step=2,
        calculate_interactions_fn=_calc,
        set_frame_fn=lambda frame_idx: set_calls.append(int(frame_idx)),
        get_current_frame_fn=lambda: 3,
        progress_canceled_fn=lambda: len(calc_calls) >= 1,
    )
    assert payload["ok"] is True
    assert payload["canceled"] is True
    assert payload["frame_plan"] == [1, 3, 5]
    assert payload["sampled_frames"] == [1]
    assert calc_calls == [1]
    assert set_calls == [1, 3]


def test_trajectory_collect_presence_counts_counts_once_per_frame():
    payload = trajectory_collect_presence_counts(
        [
            [
                {"prot_res": "ASP12", "type": "Hydrogen Bond"},
                {"prot_res": "ASP12", "type": "Hydrogen Bond"},
                {"prot_res": "TYR90", "type": "Pi-Pi Stacking"},
            ],
            [{"prot_res": "ASP12", "type": "Salt Bridge"}],
            [],
        ]
    )
    assert payload["ok"] is True
    assert int(payload["frames_analyzed"]) == 3
    assert payload["residue_counts"] == {"ASP12": 2, "TYR90": 1}
    assert payload["type_counts"] == {"Hydrogen Bond": 1, "Pi-Pi Stacking": 1, "Salt Bridge": 1}


def test_trajectory_pick_matching_interaction_handles_jitter_and_bad_distances():
    target = {
        "type": "Hydrophobic",
        "prot_res": "VAL42",
        "prot_atom": "CG1",
        "lig_atom": "C1",
        "details": "Cluster",
        "distance": 3.8,
    }
    strict = dict(target, distance=3.9)
    assert trajectory_pick_matching_interaction([strict], target)["distance"] == 3.9

    relaxed_a = dict(target, prot_atom="CG2", lig_atom="C2", distance="bad")
    relaxed_b = dict(target, prot_atom="CB", lig_atom="C3", distance=4.0)
    picked = trajectory_pick_matching_interaction([relaxed_a, relaxed_b], target)
    assert picked["prot_atom"] == "CB"
    assert float(picked["distance"]) == 4.0
    assert trajectory_pick_matching_interaction([], target) is None


def test_trajectory_build_selected_trace_payload_summarizes_distances():
    payload = trajectory_build_selected_trace_payload(
        frames=[1, 3, 5, 7],
        present=[1, 0, 1, np.nan],
        distances=[2.5, np.nan, 3.5, 4.0],
    )
    assert payload["ok"] is True
    assert int(payload["n_frames"]) == 4
    assert int(payload["n_present"]) == 2
    assert np.isclose(float(payload["occupancy_pct"]), 50.0)
    assert np.isclose(float(payload["distance_mean"]), (2.5 + 3.5 + 4.0) / 3.0)

    empty = trajectory_build_selected_trace_payload(frames=[], present=[], distances=[])
    assert empty["ok"] is False
    assert empty["reason"] == "no_frames_analyzed"
