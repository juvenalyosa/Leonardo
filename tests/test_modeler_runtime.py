import math
import os
import urllib.error

from modeler_runtime import build_modeler_energy_distribution_payload
from modeler_runtime import build_modeler_run_root
from modeler_runtime import build_modeler_mutation_highlight_indices
from modeler_runtime import filter_modeler_template_components
from modeler_runtime import fetch_hetatm_source_template_pdb
from modeler_runtime import extract_modeler_mutation_target_positions
from modeler_runtime import modeler_sequence_window_covers_positions
from modeler_runtime import map_modeler_highlight_indices_to_positions
from modeler_runtime import normalize_modeler_build_output
from modeler_runtime import normalize_modeler_template_input_token
from modeler_runtime import parse_modeler_template_code
from modeler_runtime import resolve_modeler_mutation_input_sequence
from modeler_runtime import select_modeler_search_result_variant
from modeler_runtime import resolve_modeler_template_selection
from modeler_runtime import resolve_modeler_template_transfer_flags
from modeler_runtime import should_modeler_autoresolve_mutation_start
from modeler_runtime import validate_modeler_model_count
from modeler_runtime import write_modeler_score_records_csv


def test_modeler_runtime_normalize_build_output_and_validate_count(tmp_path):
    run_root = tmp_path / "run_modeler"
    pdb_dir = run_root / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    best = pdb_dir / "modeljuv.B99990001.pdb"
    other = pdb_dir / "modeljuv.B99990002.pdb"
    cluster = pdb_dir / "median_cluster_1.pdb"
    for path in (best, other, cluster):
        path.write_text("MODEL\n", encoding="utf-8")

    payload = normalize_modeler_build_output(
        {
            "best_model_path": "pdbs/modeljuv.B99990001.pdb",
            "generated_model_paths": [
                "pdbs/modeljuv.B99990001.pdb",
                "pdbs/modeljuv.B99990002.pdb",
            ],
            "score_records": [
                {"path": "pdbs/modeljuv.B99990001.pdb", "dope_score": -120.0},
                {"path": "pdbs/modeljuv.B99990002.pdb", "dope_score": -118.5},
            ],
            "selected_cluster_records": [
                {
                    "source_path": "pdbs/modeljuv.B99990001.pdb",
                    "cluster_path": "pdbs/median_cluster_1.pdb",
                    "dope_score": -120.0,
                }
            ],
            "requested_model_count": 2,
            "generated_model_count": 2,
        },
        run_root=str(run_root),
    )

    assert str(payload["best_model_path"]) == str(best)
    assert payload["generated_model_paths"] == [str(best), str(other)]
    assert str(payload["selected_cluster_records"][0]["cluster_path"]) == str(cluster)

    count_ok = validate_modeler_model_count(payload["generated_model_paths"], 2)
    assert count_ok["ok"] is True
    assert int(count_ok["actual_count"]) == 2

    count_fail = validate_modeler_model_count(payload["generated_model_paths"], 3)
    assert count_fail["ok"] is False
    assert "generated 2 original model(s), but 3 were requested" in str(count_fail["error"])


def test_modeler_runtime_normalize_build_output_preserves_zero_elapsed_time(tmp_path):
    payload = normalize_modeler_build_output(
        {
            "best_model_path": "",
            "generated_model_paths": [],
            "elapsed_time_sec": 0.0,
        },
        run_root=str(tmp_path),
    )

    assert float(payload["elapsed_time_sec"]) == 0.0


def test_modeler_runtime_normalize_build_output_fallback_collects_only_original_models(tmp_path):
    run_root = tmp_path / "run_modeler"
    pdb_dir = run_root / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    best = pdb_dir / "modeljuv.B99990001.pdb"
    other = pdb_dir / "modeljuv.B99990002.pdb"
    extra = pdb_dir / "median_cluster_1.pdb"
    for path in (best, other, extra):
        path.write_text("MODEL\n", encoding="utf-8")

    payload = normalize_modeler_build_output(
        "pdbs/modeljuv.B99990001.pdb",
        run_root=str(run_root),
    )

    assert str(payload["best_model_path"]) == str(best)
    assert payload["generated_model_paths"] == [str(best), str(other)]
    assert int(payload["generated_model_count"]) == 2


def test_modeler_runtime_distribution_payload_and_score_csv(tmp_path):
    records = [
        {"path": str(tmp_path / "m1.pdb"), "dope_score": -121.0},
        {"path": str(tmp_path / "m2.pdb"), "dope_score": -119.0},
        {"path": str(tmp_path / "m3.pdb"), "dope_score": -118.0},
        {"path": str(tmp_path / "m4.pdb"), "dope_score": -117.5},
    ]

    payload = build_modeler_energy_distribution_payload(records)
    assert payload["ok"] is True
    assert int(payload["n_models"]) == 4
    assert math.isclose(float(payload["best_energy"]), -121.0, rel_tol=1e-12)
    assert len(list(payload["hist_density"])) >= 1
    assert "beta" not in payload
    assert "fit_x" not in payload

    out_csv = tmp_path / "modeler_scores.csv"
    wrote = write_modeler_score_records_csv(str(out_csv), records)
    assert str(wrote) == str(out_csv)
    assert os.path.isfile(out_csv)
    text = out_csv.read_text(encoding="utf-8")
    assert "rank,model_path,dope_score" in text
    assert "-121.000000" in text


def test_modeler_runtime_build_run_root_is_unique_when_timestamp_collides(tmp_path):
    class _FixedNow:
        def strftime(self, _fmt):
            return "run_20260310_150000_123456"

    existing = {
        str(tmp_path / "protein_modeling" / "run_20260310_150000_123456"),
        str(tmp_path / "protein_modeling" / "run_20260310_150000_123456_2"),
    }
    out = build_modeler_run_root(
        str(tmp_path),
        flow_label="protein_modeling",
        now_fn=lambda: _FixedNow(),
        exists_fn=lambda p: str(p) in existing,
    )
    assert str(out) == str(tmp_path / "protein_modeling" / "run_20260310_150000_123456_3")


def test_modeler_runtime_filters_template_components_to_selected_chain():
    comps = [
        {"chain": "A", "resi": "101", "resn": "SO4", "category": "ion"},
        {"chain": "B", "resi": "101", "resn": "SO4", "category": "ion"},
        {"chain": "A", "resi": "202", "resn": "HOH", "category": "water"},
    ]

    out = filter_modeler_template_components(comps, "1abcA")

    assert [(c["chain"], c["resi"], c["resn"]) for c in out] == [
        ("A", "101", "SO4"),
        ("A", "202", "HOH"),
    ]


def test_modeler_runtime_fetch_hetatm_source_template_pdb_falls_back_to_cif(tmp_path, monkeypatch):
    calls = []

    def _urlretrieve(url, out_path):
        calls.append((str(url), str(out_path)))
        if str(url).endswith(".pdb"):
            raise urllib.error.HTTPError(str(url), 404, "Not Found", hdrs=None, fp=None)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("data_9GBE\n")
        return str(out_path), None

    monkeypatch.setattr("modeler_runtime.urllib.request.urlretrieve", _urlretrieve)

    out = fetch_hetatm_source_template_pdb("9gbeA", str(tmp_path))

    assert str(out) == str(tmp_path / "modeling_job" / "9gbe.cif")
    assert os.path.isfile(out)
    assert calls[0][0].endswith("/9gbe.pdb")
    assert calls[1][0].endswith("/9gbe.cif")


def test_modeler_runtime_manual_template_list_overrides_checked_hits():
    payload = resolve_modeler_template_selection(
        table_template_codes=["1abcA", "2defB"],
        manual_text="3ghiC, 4jkl:D",
        available_template_codes=["1abcA", "2defB", "3ghiC", "4jklD"],
        fallback_template_codes=["1abcA"],
    )

    assert payload["source"] == "manual"
    assert payload["selected"] == ["3ghiC", "4jklD"]
    assert payload["manual_unresolved"] == []


def test_modeler_runtime_bare_pdb_manual_token_resolves_when_unique():
    assert normalize_modeler_template_input_token("3ptb", ["3ptbA"]) == "3ptbA"
    payload = resolve_modeler_template_selection(
        table_template_codes=[],
        manual_text="3ptb",
        available_template_codes=["3ptbA"],
    )
    assert payload["selected"] == ["3ptbA"]
    assert payload["manual_unresolved"] == []


def test_modeler_runtime_bare_pdb_manual_token_stays_unresolved_when_ambiguous():
    assert normalize_modeler_template_input_token("3ptb", ["3ptbA", "3ptbB"]) == ""
    payload = resolve_modeler_template_selection(
        table_template_codes=["1abcA"],
        manual_text="3ptb",
        available_template_codes=["3ptbA", "3ptbB", "1abcA"],
        fallback_template_codes=["1abcA"],
    )
    assert payload["selected"] == []
    assert payload["manual_unresolved"] == ["3ptb"]


def test_modeler_runtime_template_code_parser_accepts_display_formats():
    assert parse_modeler_template_code("3PTB:A") == {
        "template_code": "3ptbA",
        "pdb_id": "3ptb",
        "chain": "A",
    }
    assert normalize_modeler_template_input_token("3ptb:a", ["3ptbA"]) == "3ptbA"
    assert normalize_modeler_template_input_token("3ptb-a", ["3ptbA"]) == "3ptbA"


def test_modeler_runtime_manual_template_resolution_preserves_available_chain_case():
    payload = resolve_modeler_template_selection(
        table_template_codes=[],
        manual_text="3ptb:a, 4jkl-d",
        available_template_codes=["3ptbA", "4jklD"],
    )

    assert payload["source"] == "manual"
    assert payload["selected"] == ["3ptbA", "4jklD"]
    assert payload["manual_unresolved"] == []


def test_modeler_runtime_prefers_paginated_search_variant_over_minimal_default_page():
    out = select_modeler_search_result_variant(
        [
            {"tag": "strict|minimal", "result_set": list(range(10))},
            {"tag": "strict|paginate_sort", "result_set": list(range(50))},
        ],
        requested_rows=50,
    )
    assert out["tag"] == "strict|paginate_sort"
    assert int(out["count"]) == 50


def test_modeler_runtime_keeps_minimal_variant_when_it_is_the_only_nonempty_result():
    out = select_modeler_search_result_variant(
        [
            {"tag": "strict|minimal", "result_set": list(range(7))},
            {"tag": "strict|paginate_sort", "result_set": []},
        ],
        requested_rows=50,
    )
    assert out["tag"] == "strict|minimal"
    assert int(out["count"]) == 7


def test_modeler_runtime_resolve_transfer_flags_prefers_explicit_post_merge():
    receptor_plan = resolve_modeler_template_transfer_flags(
        include_nonprotein=True,
        receptor_obj="rec_obj",
        selected_component_counts={"cofactor": 1, "ion": 2, "water": 5},
        sequence_mode_template_fallback=False,
    )
    assert receptor_plan["strategy"] == "receptor_post_merge"
    assert receptor_plan["direct_template_hetatm"] is False
    assert receptor_plan["direct_template_water"] is False

    source_plan = resolve_modeler_template_transfer_flags(
        include_nonprotein=True,
        receptor_obj="",
        hetatm_source_code="3ptbA",
        selected_component_counts={"cofactor": 1, "ion": 0, "water": 3},
        sequence_mode_template_fallback=True,
    )
    assert source_plan["strategy"] == "template_source_post_merge"
    assert source_plan["direct_template_hetatm"] is False
    assert source_plan["direct_template_water"] is False

    source_disabled_plan = resolve_modeler_template_transfer_flags(
        include_nonprotein=False,
        receptor_obj="",
        hetatm_source_code="3ptbA",
        selected_component_counts={"cofactor": 1, "ion": 0, "water": 3},
        sequence_mode_template_fallback=True,
    )
    assert source_disabled_plan["strategy"] == "template_source_disabled"
    assert source_disabled_plan["direct_template_hetatm"] is False
    assert source_disabled_plan["direct_template_water"] is False

    fallback_plan = resolve_modeler_template_transfer_flags(
        include_nonprotein=True,
        receptor_obj="",
        hetatm_source_code="",
        selected_component_counts={"cofactor": 0, "ion": 0, "water": 0},
        sequence_mode_template_fallback=True,
    )
    assert fallback_plan["strategy"] == "modeller_direct_all"
    assert fallback_plan["direct_template_hetatm"] is True
    assert fallback_plan["direct_template_water"] is True


def test_modeler_runtime_mutation_start_policy_respects_sequence_mode():
    assert should_modeler_autoresolve_mutation_start("sequence") is False
    assert should_modeler_autoresolve_mutation_start("Sequence") is False
    assert should_modeler_autoresolve_mutation_start("gap") is False
    assert should_modeler_autoresolve_mutation_start("") is False


def test_modeler_runtime_extracts_explicit_mutation_positions():
    out = extract_modeler_mutation_target_positions(
        [
            "A1086V",
            "G1090del",
            "1092_1094del",
            "A1095_B1096insGG",
            "1098_1099dup",
            "A1100delinsGG",
        ]
    )
    assert out == [1086, 1090, 1092, 1093, 1094, 1095, 1096, 1098, 1099, 1100]


def test_modeler_runtime_sequence_window_check_uses_residue_numbering():
    assert modeler_sequence_window_covers_positions(1086, "A" * 308, [1086, 1171, 1393]) is True
    assert modeler_sequence_window_covers_positions(1, "A" * 308, [1171]) is False


def test_modeler_runtime_prefers_explicit_sequence_when_gap_window_misses_mutation():
    payload = resolve_modeler_mutation_input_sequence(
        prepared_sequence="A" * 308,
        prepared_start=1,
        explicit_sequence="A" * 308,
        explicit_start=1086,
        mutation_positions=[1171],
        mode="gap",
    )
    assert payload["used_explicit_sequence"] is True
    assert payload["strategy"] == "explicit_numbering_override"
    assert int(payload["start_pos"]) == 1086


def test_modeler_runtime_keeps_prepared_sequence_when_gap_window_matches_mutation():
    payload = resolve_modeler_mutation_input_sequence(
        prepared_sequence="A" * 308,
        prepared_start=1086,
        explicit_sequence="A" * 308,
        explicit_start=1086,
        mutation_positions=[1171],
        mode="gap",
    )
    assert payload["used_explicit_sequence"] is False
    assert payload["strategy"] == "prepared"


def test_modeler_runtime_sequence_mode_always_uses_explicit_sequence():
    payload = resolve_modeler_mutation_input_sequence(
        prepared_sequence="B" * 50,
        prepared_start=1,
        explicit_sequence="A" * 308,
        explicit_start=1086,
        mutation_positions=[1171],
        mode="sequence",
    )
    assert payload["used_explicit_sequence"] is True
    assert payload["strategy"] == "explicit_sequence_mode"
    assert payload["sequence"] == "A" * 308


def test_modeler_runtime_build_mutation_highlight_indices_handles_substitution_and_indels():
    sub_only = build_modeler_mutation_highlight_indices("ABCDE", "ABXDE")
    assert sub_only == [2]

    insertion = build_modeler_mutation_highlight_indices("ABCDE", "ABXCDE")
    assert insertion == [2]

    deletion = build_modeler_mutation_highlight_indices("ABCDE", "ABDE")
    assert deletion == [1, 2]


def test_modeler_runtime_maps_highlight_indices_to_observed_positions():
    positions = map_modeler_highlight_indices_to_positions(
        [2],
        mutated_sequence="ABXCDE",
        observed_sequence="ABXCDE",
        observed_positions=[11, 12, 13, 14, 15, 16],
    )
    assert positions == [13]

    shifted = map_modeler_highlight_indices_to_positions(
        [1, 2],
        mutated_sequence="ABDE",
        observed_sequence="ABDE",
        observed_positions=[101, 102, 103, 104],
    )
    assert shifted == [102, 103]


def _res(resseq, aa, bonded=True, icode=""):
    return {"resseq": resseq, "icode": icode, "resn": aa, "aa1": aa, "bonded_to_prev": bonded}


def test_chain_entry_separates_numbering_jumps_from_real_breaks():
    from modeler_runtime import build_modeler_chain_entry

    residues = [_res(16, "I", None), _res(17, "V"), _res(20, "G"),          # 17 -> 20 bonded: numbering jump
                _res(20, "S", True, "A"),                                    # insertion code kept
                _res(21, "K"), _res(25, "L", False),                         # real break: 22-24 missing
                _res(26, "W", None), _res(30, "Y", None)]                    # unknown bond: numbering decides
    e = build_modeler_chain_entry("A", residues)
    assert e["observed_seq"] == "IVGSKLWY"
    assert e["residue_labels"] == ["16", "17", "20", "20A", "21", "25", "26", "30"]
    assert e["numbering_jumps"] == [("17", "20")]
    assert e["internal_gaps"] == [(22, 24), (27, 29)]
    assert e["missing_total"] == 6
    assert len(e["positions"]) == len(e["observed_seq"])


def test_gap_fill_aligns_instead_of_trusting_residue_numbers():
    from modeler_runtime import build_modeler_chain_entry, modeler_gap_fill_sequence

    full = "MKTAYIAKQRQISFVKSHFSRQ"
    # observed 3..10 and 14..20 (real break), numbered from 103 to test that numbers are not used
    obs = [(i, aa) for i, aa in enumerate(full, start=1) if 3 <= i <= 10 or 14 <= i <= 20]
    residues = [_res(100 + i, aa, not (i == 14)) for i, aa in obs]
    residues[0]["bonded_to_prev"] = None
    e = build_modeler_chain_entry("A", residues)
    fill = modeler_gap_fill_sequence(e, full)
    assert fill["ok"] and fill["filled"] == 3
    assert fill["sequence"] == full[2:20]
    assert fill["residue_labels"][8:11] == ["111", "112", "113"]      # interpolated inside the break

    # the observed sequence itself as "reference" (auto-filled widget) must not invent residues
    same = modeler_gap_fill_sequence(e, e["observed_seq"])
    assert same["ok"] and same["filled"] == 0 and same["sequence"] == e["observed_seq"]

    wrong = modeler_gap_fill_sequence(e, "GGGGGGGGGGGGGGGGGGGGGG")
    assert not wrong["ok"] and "does not match" in wrong["error"]


def test_mutation_window_uses_structure_labels():
    from modeler_runtime import resolve_modeler_mutation_input_sequence

    payload = resolve_modeler_mutation_input_sequence(
        "ACDEF", 10, explicit_sequence="ACDEFGHIK", explicit_start=10,
        mutation_positions=[15], mode="gap", prepared_labels=["10", "11", "11A", "12", "15"])
    assert payload["prepared_fits"] is True
    assert payload["used_explicit_sequence"] is False
