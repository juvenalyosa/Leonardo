import importlib
import json
import os
import sys
import tempfile

import abfe_runtime
import numpy as np
import pytest
from abfe_runtime import (
    ABFE_POCKET_DESCRIPTOR_COLS,
    _nan_pocket_descriptor_row,
    abfe_distance_axis_label,
    build_abfe_adaptive_steering_state_payload,
    build_abfe_adaptive_steering_update_decision,
    build_abfe_bias_reweighted_population_payload,
    build_abfe_final_scene_restore_execution_default_payload,
    build_abfe_final_scene_restore_execution_payload,
    build_abfe_final_traj_build_plan,
    build_abfe_final_traj_post_build_ui_plan,
    build_abfe_free_energy_debug_line,
    build_abfe_free_energy_numeric_payload,
    build_abfe_free_energy_plot_payload,
    build_abfe_free_energy_report_payload,
    build_abfe_gbsa_mmgbsa_rescore_payload,
    build_abfe_live_plot_apply_plan,
    build_abfe_live_plot_setup_plan,
    build_abfe_live_plot_state_payload,
    build_abfe_live_plot_update_default_payload,
    build_abfe_live_plot_update_payload,
    build_abfe_live_update_decision,
    build_abfe_live_view_atom_count_payload,
    build_abfe_live_view_tracking_default_payload,
    build_abfe_live_view_update_plan,
    build_abfe_local_md_cleanup_plan,
    build_abfe_local_md_result_payload,
    build_abfe_local_md_teardown_plan,
    build_abfe_local_pocket_restraint_plan_payload,
    build_abfe_local_resource_plan,
    build_abfe_md_progress_dialog_message,
    build_abfe_md_progress_dialog_plan,
    build_abfe_md_progress_finalize_decision,
    build_abfe_md_progress_step_decision,
    build_abfe_md_sample_payload,
    build_abfe_object_inventory_payload,
    build_abfe_openmm_context_setup_plan,
    build_abfe_openmm_staged_prep_plan,
    build_abfe_post_interaction_followup_plan,
    build_abfe_post_interaction_report_payload,
    build_abfe_post_sampling_metrics,
    build_abfe_probe_direction_score_payload,
    build_abfe_replica_aggregate_payload,
    build_abfe_scene_apply_operations,
    build_abfe_scene_restore_default_payload,
    build_abfe_scene_restore_plan,
    build_abfe_state_coverage_diagnostic_payload,
    build_abfe_state_save_decision,
    build_abfe_steering_bandit_action_payload,
    build_abfe_steering_lambda_schedule,
    build_abfe_steering_policy_observation_payload,
    build_abfe_steering_policy_safe_action_payload,
    build_abfe_steering_policy_state_payload,
    build_abfe_steering_policy_summary_payload,
    build_abfe_steering_policy_trace_payload,
    build_abfe_surface_detach_direction_payload,
    build_abfe_traj_state_pick_plan,
    build_abfe_tunnel_restraint_payload,
    build_abfe_ui_process_events_decision,
    build_openmm_platform_selection_plan,
    collect_abfe_interaction_presence_sets,
    compute_runtime_pocket_descriptors,
    dg_to_kd_pkd,
    infer_abfe_target_hint,
    predict_abfe_compact_calibrated_dg,
    predict_abfe_runtime_calibrated_dg,
    prepare_abfe_live_view_preflight,
    prepare_abfe_post_interaction_followup_runtime_plan,
    resolve_abfe_post_interaction_dialog_payload,
)
from alphafold_runtime import (
    af_collect_existing_dirs,
    af_extract_plddt_pae_from_json,
    af_find_best_model_recursive,
    af_is_relaxed_name,
    af_model_patterns,
    af_pick_best_plot_image,
    af_plot_patterns,
    af_plot_title,
)
from dock_runtime import (
    build_legacy_python_tool_command,
    build_meeko_flexres_spec,
    build_meeko_receptor_input_args,
    build_smina_flexres_spec,
    resolve_tool_path,
    run_post_docking_pipeline,
    smina_supports_native_flexres,
)
from leadopt_runtime import (
    admet_score,
    build_leadopt_summary_lines,
    filter_fail_reason,
    find_best_leadopt_result,
    leadopt_append_charge_map_source,
    leadopt_apply_diversity_rescue_rows,
    leadopt_apply_refinement_result,
    leadopt_build_candidate_rows,
    leadopt_build_complex_selection_ids,
    leadopt_build_nonbonded_particle_data,
    leadopt_build_openmm_refinement_system,
    leadopt_compute_result_counts,
    leadopt_export_csv_outputs,
    leadopt_filter_and_rescue_candidates,
    leadopt_filter_rows_with_limits,
    leadopt_finalize_advanced_refinement_report,
    leadopt_finalize_refine_estimate,
    leadopt_heavy_atom_ids,
    leadopt_infer_atom_names_elements,
    leadopt_merge_candidate_and_docked_results,
    leadopt_needs_parameterization_build,
    leadopt_optional_refine_note,
    leadopt_pick_protein_eval_ids,
    leadopt_prepare_job_paths,
    leadopt_prepare_parameterization_for_refinement,
    leadopt_prepare_refine_param_dir,
    leadopt_rescue_top_admet_rows,
    leadopt_run_openmm_refinement_sampling,
    leadopt_sanitize_refine_parameterization_options,
    leadopt_select_advanced_refinement_rows,
    leadopt_status_text,
    make_leadopt_advanced_refinement_report,
    make_leadopt_refine_result,
    mpo_score,
    pass_filters,
    rank_leadopt_results,
    relax_filter_limits,
    run_leadopt_entries,
)
from psi4_runtime import (
    psi4_apply_mapping_to_dict,
    psi4_atomic_mass_amu,
    psi4_atomic_number,
    psi4_build_antechamber_cmd,
    psi4_build_driver_options,
    psi4_build_irc_profile_payload,
    psi4_build_leadopt_param_hit_updates,
    psi4_build_mulliken_esp_cube_from_density,
    psi4_build_net_charge_candidates,
    psi4_build_parmchk2_cmd,
    psi4_build_rebuild_failure_payload,
    psi4_build_rebuild_success_payload,
    psi4_choose_parameterization_source,
    psi4_collect_atoms_from_mol2,
    psi4_collect_atoms_from_xyz,
    psi4_collect_ligand_charges_from_file,
    psi4_detect_sqm_charge_mismatch_hint,
    psi4_element_token,
    psi4_estimate_net_charge_from_charges,
    psi4_finalize_leadopt_parameterization_success,
    psi4_format_antechamber_attempt_brief,
    psi4_format_antechamber_attempts_failure,
    psi4_format_tool_rc_failure,
    psi4_guess_multiplicity_from_atoms,
    psi4_infer_charge_multiplicity,
    psi4_load_history_data,
    psi4_load_mol2_atom_records,
    psi4_load_mol2_charges_and_bonds,
    psi4_load_vib_data,
    psi4_make_leadopt_cached_info,
    psi4_make_leadopt_generated_info,
    psi4_make_leadopt_parameterization_info,
    psi4_map_charge_transfer_between_mol2_records,
    psi4_map_ligand_charges_from_param_mol2,
    psi4_merge_notes,
    psi4_mol2_atom_type_to_element,
    psi4_normalize_memory_gb,
    psi4_normalize_method_basis,
    psi4_normalize_operation,
    psi4_normalize_rdkit_parameterization_payload,
    psi4_normalize_threads,
    psi4_orchestrate_leadopt_parameterization,
    psi4_parse_charge_multiplicity_from_block,
    psi4_pcm_solvent_candidates,
    psi4_pick_charged_mol2_for_rebuild,
    psi4_prepare_driver_job,
    psi4_prepare_ir_spectrum,
    psi4_prepare_leadopt_parameterization_paths,
    psi4_prepare_pose_source_for_parameterization,
    psi4_prepare_rdkit_source_for_parameterization,
    psi4_prepare_rebuild_output_paths,
    psi4_prepare_tleap_ligand_build,
    psi4_prepare_vibrational_animation,
    psi4_read_cube,
    psi4_reason_missing_parameterization_tools,
    psi4_reason_source_for_parameterization_unavailable,
    psi4_reason_tleap_parameterization_failed,
    psi4_reference_for_multiplicity,
    psi4_resolve_density_cube,
    psi4_resolve_esp_cube,
    psi4_resolve_homo_lumo_cubes,
    psi4_resolve_leadopt_cached_parameterization,
    psi4_resolve_leadopt_parameterization_source,
    psi4_resolve_leadopt_rdkit_state,
    psi4_run_leadopt_parameterization_pipeline,
    psi4_tail_file,
    psi4_transfer_mol2_charges_to_typed,
    psi4_validate_irc_ts_frequencies,
    psi4_validate_leadopt_parameterization_tools,
    psi4_write_charged_mol2,
    psi4_write_cube_data,
)
from redock_runtime import (
    _parse_redocking_flexres_spec,
    adjust_redocking_grid_size,
    aggregate_label_runs,
    aggregate_label_runs_runtime,
    attach_pose_object_to_rows,
    best_pose_for_member,
    build_redock_run_map,
    build_redocking_best_rmsd_config,
    build_redocking_bond_order_matches,
    build_redocking_candidate_atom_maps,
    build_redocking_candidate_map_callback_bundle,
    build_redocking_candidate_map_config,
    build_redocking_chainwise_ca_alignment,
    build_redocking_combo_object_name,
    build_redocking_combo_states,
    build_redocking_comparison_rows,
    build_redocking_directional_match_functions,
    build_redocking_mcs_payload,
    build_redocking_pose_object_name,
    build_redocking_post_apply_state_runtime,
    build_redocking_post_dialog_runtime,
    build_redocking_post_pipeline_context,
    build_redocking_post_summary_runtime,
    build_redocking_post_transform_runtime,
    build_redocking_rdkit_callback_bundle_for_config,
    build_redocking_rdkit_candidate_map_callback_bundle,
    build_redocking_receptor_structure_qc,
    build_redocking_reference_context_plan,
    build_redocking_reference_grid_info_message,
    build_redocking_reference_grid_size,
    build_redocking_rmsd_mol_config,
    build_redocking_row_object_actions,
    build_redocking_rows_for_repeat,
    build_redocking_run_debug_line,
    build_redocking_run_map_payload,
    build_redocking_run_map_runtime,
    build_redocking_runtime_input_actions,
    build_redocking_selected_interaction_post_apply_plan,
    build_redocking_selected_pose_execution_plan,
    build_redocking_selected_pose_view_plan,
    build_redocking_single_run_payload,
    build_redocking_site_qc_payload,
    build_redocking_smina_plan,
    build_redocking_state_row_map,
    build_redocking_summary_lines,
    build_redocking_summary_payload,
    build_redocking_table_rows,
    build_redocking_traj_focus_visibility_actions,
    build_redocking_traj_view_entry,
    build_redocking_traj_view_plan,
    build_redocking_traj_visibility_actions,
    build_redocking_unique_atom_maps,
    classify_delta,
    collect_redocking_rows_from_run_map,
    collect_redocking_traj_view_objects,
    compare_wt_mutant,
    compose_redocking_single_run_outcome,
    compute_redocking_rmsd_no_fit,
    compute_redocking_rmsd_values_for_repeat,
    dedupe_redocking_runtime_systems_by_signature,
    estimate_redocking_initial_autobox_add,
    finalize_redock_member_run,
    finalize_redocking_reference_ligand_export_result,
    finalize_redocking_reference_mol_result,
    finalize_redocking_rmsd_mol_result,
    invoke_prepare_flexible_receptor,
    is_better_repeat,
    load_redocking_pose_object_safe,
    load_redocking_traj_members,
    pick_best_pose_row,
    pick_comparison_metrics,
    prepare_redocking_base_payload,
    prepare_redocking_base_runtime,
    prepare_redocking_ligand_context,
    prepare_redocking_ligand_override_context,
    prepare_redocking_ligand_override_runtime,
    prepare_redocking_pose_outputs,
    prepare_redocking_receptor_flex_context,
    prepare_redocking_reference_flex_context,
    prepare_redocking_reference_flex_runtime,
    prepare_redocking_reference_ligand_export_plan,
    prepare_redocking_reference_mol_plan,
    prepare_redocking_repeat_outputs,
    prepare_redocking_rmsd_mol_plan,
    prepare_redocking_runtime_systems_context,
    prepare_redocking_runtime_systems_runtime,
    prepare_redocking_runtime_views_runtime,
    prepare_redocking_selected_interaction_context,
    prepare_redocking_selected_interaction_post_payload,
    prepare_redocking_selected_pose_context,
    prepare_redocking_single_run_context,
    prepare_redocking_traj_frame_payload,
    recover_redocking_reference_mol_from_pdb,
    redocking_ligand_qc_messages,
    redocking_pose_load_warning_text,
    redocking_traj_visible_for_label,
    resolve_redocking_best_rmsd,
    resolve_redocking_candidate_atom_maps,
    resolve_redocking_ligand_charge_policy,
    resolve_redocking_ligand_file,
    resolve_redocking_ligand_smiles_hint,
    resolve_redocking_pose_view_file,
    resolve_redocking_rdkit_candidate_atom_maps,
    resolve_redocking_rdkit_rmsd_mol,
    resolve_redocking_rdkit_rmsd_mol_with_config,
    resolve_redocking_receptor_for_docking,
    resolve_redocking_reference_grid_warning_message,
    resolve_redocking_reference_ligand_export_response,
    resolve_redocking_rmsd_mol,
    resolve_redocking_selected_interaction_warning_message,
    resolve_redocking_selected_pose_warning_message,
    resolve_redocking_site_qc_result,
    resolve_redocking_traj_state,
    resolve_state_for_member,
    rows_best_per_member,
    run_redocking_hetatm_preflight,
    run_redocking_hetatm_preflight_runtime,
    run_redocking_post_status_runtime,
    run_redocking_reference_context_runtime,
    run_redocking_repeat_pipeline,
    run_redocking_repeats,
    run_redocking_row_object_actions_runtime,
    run_redocking_runtime_systems,
    run_redocking_runtime_systems_runtime,
    run_redocking_selected_interaction_runtime,
    run_redocking_selected_pose_runtime,
    run_redocking_single_run_pipeline,
    run_redocking_single_system_orchestrator,
    run_redocking_single_system_runtime,
    run_redocking_traj_frame_runtime,
    run_redocking_traj_visibility_runtime,
    run_redocking_workflow_orchestrator,
    sampling_confidence_from_runs,
    sort_redocking_runtime_systems,
    stats_for_label,
    strip_redocking_residue_from_structure_file,
    summarize_redocking_repeat_rows,
    validate_redocking_flex_selection,
    validate_redocking_flexible_prepare_result,
    validate_redocking_ligand_prepare_result,
    validate_redocking_receptor_prepare_result,
    write_redocking_comparison_summary_csv,
    write_redocking_repeat_stats_csv,
    wt_quality_from_run,
)
from smina_runtime import parse_smina_pose_scores, run_smina_repeats
from vs_runtime import (
    build_virtual_screening_summary_lines,
    rank_virtual_screening_results,
    run_virtual_screening_entries,
    summarize_virtual_screening_results,
    vs_prepare_job_paths,
    vs_status_text,
)


def test_abfe_runtime_helpers_basic():
    row = _nan_pocket_descriptor_row()
    assert set(row.keys()) == set(ABFE_POCKET_DESCRIPTOR_COLS)
    assert len(row) >= 10

    hint = infer_abfe_target_hint(
        texts=["3PTB thrombin_wt", "other"],
        target_order=["bace", "thrombin", "cdk2"],
    )
    assert hint == "thrombin"

    dg_cal, missing = predict_abfe_compact_calibrated_dg(
        dg_expavg_kj=-120.0,
        dg_cumulant_kj=float("nan"),
        energy_mean_kj=-60.0,
        coul_mean_kj=-2.0,
        contact_frac_mean=0.7,
    )
    assert isinstance(dg_cal, float)
    assert "dg_cumulant_kj" in set(missing)

    kd_m, p_kd = dg_to_kd_pkd(-30.0, temperature_k=300.0)
    assert kd_m > 0.0
    assert p_kd > 0.0

    abfe_runtime._ABFE_RUNTIME_DEPLOY_CACHE = {}
    abfe_runtime._ABFE_RUNTIME_DEPLOY_SOURCE = ""
    pred = predict_abfe_runtime_calibrated_dg(
        {
            "dg_expavg_kj": -120.0,
            "dg_cumulant_kj": -250.0,
            "energy_mean_kj": -60.0,
            "coul_mean_kj": -2.0,
            "contact_frac_mean": 0.7,
        },
        target_hint="thrombin",
    )
    assert isinstance(pred, dict)
    assert "dg_cal_kj" in pred
    assert "mode_used" in pred


def test_abfe_runtime_predict_prefers_abfe_only_model_version():
    old_cache = abfe_runtime._ABFE_RUNTIME_DEPLOY_CACHE
    old_source = abfe_runtime._ABFE_RUNTIME_DEPLOY_SOURCE
    try:
        abfe_runtime._ABFE_RUNTIME_DEPLOY_CACHE = {
            "best_model_groupcv": "stack_cal_combo_ep_cal_expavg_dock_loto",
            "best_model_groupcv_abfe_only": "cal_combo_ep",
            "ridge_compact_weighted": {
                "feature_cols": ["dg_expavg_kj", "dg_cumulant_kj"],
                "med": [-10.0, -20.0],
                "mu": [-10.0, -20.0],
                "sd": [1.0, 1.0],
                "w": [0.5, -0.2],
                "y_mean": -5.0,
            },
        }
        abfe_runtime._ABFE_RUNTIME_DEPLOY_SOURCE = "unit-test"
        pred = predict_abfe_runtime_calibrated_dg(
            {
                "dg_expavg_kj": -12.0,
                "dg_cumulant_kj": -21.0,
            },
            target_hint="unknown",
        )
    finally:
        abfe_runtime._ABFE_RUNTIME_DEPLOY_CACHE = old_cache
        abfe_runtime._ABFE_RUNTIME_DEPLOY_SOURCE = old_source
    assert str(pred["mode_used"]) == "compact_unknown"
    assert str(pred["model_version"]).startswith("cal_combo_ep+ridge_compact")


def test_abfe_runtime_scene_restore_plan_helper():
    known = ["obj_a", "obj_b", "ctx", "final"]

    plan_final = build_abfe_scene_restore_plan(
        known_objects=known,
        final_view_object="final",
        context_object="ctx",
        source_object="src",
        restore_enabled=["obj_a"],
        keep_scene=False,
    )
    assert plan_final["ok"] is True
    assert str(plan_final["mode"]) == "final_view"
    assert list(plan_final["force_enable"]) == ["final", "ctx"]
    assert str(plan_final["zoom_selection"]) == "(ctx) or (final)"
    actions_final = [(str(a.get("action", "")), str(a.get("object", ""))) for a in list(plan_final.get("actions", []) or [])]
    assert actions_final == [
        ("disable", "obj_a"),
        ("disable", "obj_b"),
        ("enable", "ctx"),
        ("enable", "final"),
    ]

    plan_keep = build_abfe_scene_restore_plan(
        known_objects=known,
        final_view_object="final",
        context_object="ctx",
        source_object="src",
        restore_enabled=["obj_a"],
        keep_scene=True,
    )
    actions_keep = [(str(a.get("action", "")), str(a.get("object", ""))) for a in list(plan_keep.get("actions", []) or [])]
    assert actions_keep == [
        ("enable", "obj_a"),
        ("disable", "obj_b"),
        ("enable", "ctx"),
        ("enable", "final"),
    ]

    plan_restore = build_abfe_scene_restore_plan(
        known_objects=known,
        final_view_object="",
        context_object="ctx",
        source_object="src",
        restore_enabled=["obj_b"],
        keep_scene=False,
    )
    assert str(plan_restore["mode"]) == "restore_enabled"
    assert list(plan_restore["force_enable"]) == []
    actions_restore = [(str(a.get("action", "")), str(a.get("object", ""))) for a in list(plan_restore.get("actions", []) or [])]
    assert actions_restore == [
        ("disable", "obj_a"),
        ("enable", "obj_b"),
        ("enable", "ctx"),
        ("disable", "final"),
    ]

    plan_fallback = build_abfe_scene_restore_plan(
        known_objects=["src"],
        final_view_object="",
        context_object="ctx",
        source_object="src",
        restore_enabled=[],
        keep_scene=False,
    )
    assert str(plan_fallback["mode"]) == "fallback_source"
    assert list(plan_fallback["actions"]) == []
    assert list(plan_fallback["force_enable"]) == ["src"]
    assert str(plan_fallback["zoom_selection"]) == "(ctx) or (src)"


def test_abfe_runtime_scene_restore_default_payload_helpers():
    plan_default = build_abfe_scene_restore_default_payload()
    assert plan_default["ok"] is True
    assert str(plan_default["mode"]) == "none"
    assert list(plan_default["actions"]) == []
    assert list(plan_default["force_enable"]) == []
    assert str(plan_default["zoom_selection"]) == ""
    assert bool(plan_default["keep_scene"]) is False

    exec_default = build_abfe_final_scene_restore_execution_default_payload()
    assert exec_default["ok"] is True
    assert str(exec_default["mode"]) == "none"
    assert dict(exec_default["scene_plan"]) == dict(plan_default)
    assert list(exec_default["actions"]) == []
    assert list(exec_default["force_enable_actions"]) == []
    assert bool(exec_default["should_zoom"]) is False
    assert bool(exec_default["should_log_debug"]) is False
    assert str(exec_default["debug_line"]) == ""
    assert int(exec_default["restore_enabled_count"]) == 0


def test_abfe_runtime_scene_apply_operations():
    payload = build_abfe_scene_apply_operations(
        actions=[{"action": "disable", "object": "obj_a"}],
        force_enable_actions=["ctx", {"action": "enable", "object": "final"}],
        zoom_selection="(ctx) or (final)",
        default_action="disable",
        zoom_buffer_a=8.0,
    )
    assert payload["ok"] is True
    assert bool(payload["has_scene_actions"]) is True
    assert bool(payload["has_force_enable_actions"]) is True
    assert bool(payload["should_zoom"]) is True
    assert str(payload["default_action"]) == "disable"
    assert np.isclose(float(payload["zoom_buffer_a"]), 8.0)
    assert list(payload["scene_actions"]) == [{"action": "disable", "object": "obj_a"}]
    assert list(payload["force_enable_actions"]) == [
        {"action": "enable", "object": "ctx"},
        {"action": "enable", "object": "final"},
    ]
    kinds = [str(op.get("kind", "")) for op in list(payload.get("operations", []) or [])]
    assert kinds == ["scene_actions", "force_enable_actions", "zoom"]

    payload_sanitized = build_abfe_scene_apply_operations(
        actions=[{"action": "bad", "object": "obj_x"}, {"action": "enable", "object": ""}],
        force_enable_actions=[{"action": "bad", "object": "ctx"}, None],
        zoom_selection="",
        default_action="bad",
        zoom_buffer_a=-4.0,
    )
    assert str(payload_sanitized["default_action"]) == "disable"
    assert np.isclose(float(payload_sanitized["zoom_buffer_a"]), 8.0)
    assert list(payload_sanitized["scene_actions"]) == [{"action": "disable", "object": "obj_x"}]
    assert list(payload_sanitized["force_enable_actions"]) == [{"action": "enable", "object": "ctx"}]
    assert bool(payload_sanitized["should_zoom"]) is False

    payload_empty = build_abfe_scene_apply_operations()
    assert bool(payload_empty["has_scene_actions"]) is False
    assert bool(payload_empty["has_force_enable_actions"]) is False
    assert bool(payload_empty["should_zoom"]) is False
    assert list(payload_empty["operations"]) == []


def test_abfe_runtime_object_inventory_and_live_view_defaults():
    inv_default = build_abfe_object_inventory_payload()
    assert inv_default["ok"] is True
    assert list(inv_default["object_names"]) == []
    assert int(inv_default["count"]) == 0

    inv_names = build_abfe_object_inventory_payload(
        object_names=[" obj_a ", "", None, "obj_b"],
    )
    assert list(inv_names["object_names"]) == ["obj_a", "obj_b"]
    assert int(inv_names["count"]) == 2

    live_track = build_abfe_live_view_tracking_default_payload()
    assert live_track["ok"] is True
    assert list(live_track["local_model_indices"]) == []
    assert str(live_track["idx_selection"]) == ""

    atom_count_ok = build_abfe_live_view_atom_count_payload(atom_count=5)
    assert atom_count_ok["ok"] is True
    assert int(atom_count_ok["live_view_atom_count"]) == 5

    atom_count_bad = build_abfe_live_view_atom_count_payload(atom_count="bad")
    assert int(atom_count_bad["live_view_atom_count"]) == 0

    atom_count_negative = build_abfe_live_view_atom_count_payload(atom_count=-7)
    assert int(atom_count_negative["live_view_atom_count"]) == 0


def test_abfe_runtime_final_scene_restore_execution_payload():
    known = ["obj_a", "obj_b", "ctx", "final"]

    payload_final = build_abfe_final_scene_restore_execution_payload(
        known_objects=known,
        final_view_object="final",
        context_object="ctx",
        source_object="src",
        restore_enabled=["obj_a"],
        keep_scene_requested=True,
    )
    assert payload_final["ok"] is True
    assert str(payload_final["mode"]) == "final_view"
    assert bool(payload_final["keep_scene_applied"]) is True
    assert bool(payload_final["should_log_debug"]) is True
    assert str(payload_final["debug_line"]) == (
        "ABFE final view: object=final, keep_scene=True, context=ctx, restored_visible=1"
    )
    actions_final = [
        (str(a.get("action", "")), str(a.get("object", "")))
        for a in list(payload_final.get("actions", []) or [])
    ]
    assert actions_final == [
        ("enable", "obj_a"),
        ("disable", "obj_b"),
        ("enable", "ctx"),
        ("enable", "final"),
    ]
    force_enable_actions = [
        (str(a.get("action", "")), str(a.get("object", "")))
        for a in list(payload_final.get("force_enable_actions", []) or [])
    ]
    assert force_enable_actions == [("enable", "final"), ("enable", "ctx")]
    assert bool(payload_final["should_zoom"]) is True
    assert str(payload_final["zoom_selection"]) == "(ctx) or (final)"

    payload_restore = build_abfe_final_scene_restore_execution_payload(
        known_objects=known,
        final_view_object="",
        context_object="ctx",
        source_object="src",
        restore_enabled=["obj_b"],
        keep_scene_requested=True,
    )
    assert str(payload_restore["mode"]) == "restore_enabled"
    assert bool(payload_restore["keep_scene_applied"]) is False
    assert bool(payload_restore["should_log_debug"]) is False
    actions_restore = [
        (str(a.get("action", "")), str(a.get("object", "")))
        for a in list(payload_restore.get("actions", []) or [])
    ]
    assert actions_restore == [
        ("disable", "obj_a"),
        ("enable", "obj_b"),
        ("enable", "ctx"),
        ("disable", "final"),
    ]
    assert list(payload_restore.get("force_enable_actions", []) or []) == []
    assert bool(payload_restore["should_zoom"]) is False

    payload_fallback = build_abfe_final_scene_restore_execution_payload(
        known_objects=["src"],
        final_view_object="",
        context_object="ctx",
        source_object="src",
        restore_enabled=[],
        keep_scene_requested=False,
    )
    assert str(payload_fallback["mode"]) == "fallback_source"
    force_enable_fallback = [
        (str(a.get("action", "")), str(a.get("object", "")))
        for a in list(payload_fallback.get("force_enable_actions", []) or [])
    ]
    assert force_enable_fallback == [("enable", "src")]
    assert bool(payload_fallback["should_zoom"]) is True
    assert str(payload_fallback["zoom_selection"]) == "(ctx) or (src)"


def test_abfe_runtime_local_md_cleanup_plan():
    payload_full = build_abfe_local_md_cleanup_plan(
        live_view_object="abfe_live",
        known_objects=["obj_a", "abfe_live"],
        context_object="",
        source_object="prot_src",
        state_index=3,
    )
    assert payload_full["ok"] is True
    assert bool(payload_full["should_delete_live_view"]) is True
    assert str(payload_full["live_view_object"]) == "abfe_live"
    assert bool(payload_full["has_context_object"]) is False
    assert bool(payload_full["should_build_context_fallback"]) is True
    assert str(payload_full["context_source_object"]) == "prot_src"
    assert int(payload_full["context_state_index"]) == 3

    payload_no_delete = build_abfe_local_md_cleanup_plan(
        live_view_object="abfe_live",
        known_objects=["obj_a"],
        context_object="ctx_obj",
        source_object="prot_src",
        state_index=0,
    )
    assert bool(payload_no_delete["should_delete_live_view"]) is False
    assert bool(payload_no_delete["has_context_object"]) is True
    assert bool(payload_no_delete["should_build_context_fallback"]) is False
    assert int(payload_no_delete["context_state_index"]) == 1

    payload_no_source = build_abfe_local_md_cleanup_plan(
        live_view_object="",
        known_objects=[],
        context_object="",
        source_object="",
        state_index="bad",
    )
    assert bool(payload_no_source["should_delete_live_view"]) is False
    assert bool(payload_no_source["should_build_context_fallback"]) is False
    assert int(payload_no_source["context_state_index"]) == 1


def test_abfe_runtime_local_md_teardown_plan():
    payload_full = build_abfe_local_md_teardown_plan(
        has_context=True,
        has_integrator=True,
        clear_references=True,
    )
    assert payload_full["ok"] is True
    assert bool(payload_full["should_release_context"]) is True
    assert bool(payload_full["should_release_integrator"]) is True
    assert list(payload_full["release_order"]) == ["context", "integrator"]
    assert bool(payload_full["clear_references"]) is True

    payload_context_only = build_abfe_local_md_teardown_plan(
        has_context=True,
        has_integrator=False,
        clear_references=False,
    )
    assert bool(payload_context_only["should_release_context"]) is True
    assert bool(payload_context_only["should_release_integrator"]) is False
    assert list(payload_context_only["release_order"]) == ["context"]
    assert bool(payload_context_only["clear_references"]) is False

    payload_none = build_abfe_local_md_teardown_plan(
        has_context=False,
        has_integrator=False,
    )
    assert bool(payload_none["should_release_context"]) is False
    assert bool(payload_none["should_release_integrator"]) is False
    assert list(payload_none["release_order"]) == []


def test_abfe_runtime_openmm_staged_prep_plan():
    plan_hold = build_abfe_openmm_staged_prep_plan(
        minimize_max_iters=1200,
        has_lig_hold=True,
        sim_ns=3.0,
        timestep_ps=0.001,
    )
    assert plan_hold["ok"] is True
    assert int(plan_hold["max_it"]) == 1200
    min_ops = list(plan_hold["minimize_ops"])
    assert len(min_ops) == 4
    assert [float(op["set_k_hold"]) for op in min_ops] == [2600.0, 1100.0, 320.0, 0.0]
    assert [int(op["minimize_max_iters"]) for op in min_ops] == [1200, 900, 600, 480]
    assert np.isclose(float(plan_hold["eq_hold_ps"]), 40.0)
    assert np.isclose(float(plan_hold["eq_soft_ps"]), 60.0)
    assert np.isclose(float(plan_hold["eq_free_ps"]), 240.0)
    assert int(plan_hold["eq_hold_steps"]) == 40000
    assert int(plan_hold["eq_soft_steps"]) == 60000
    assert int(plan_hold["eq_free_steps"]) == 240000
    eq_ops = list(plan_hold["equilibration_ops"])
    assert len(eq_ops) == 3
    assert [float(op["set_k_hold"]) for op in eq_ops] == [320.0, 120.0, 0.0]
    assert [int(op["steps"]) for op in eq_ops] == [40000, 60000, 240000]
    assert np.isclose(float(plan_hold["equilibration_ps_total"]), 340.0)
    assert np.isclose(float(plan_hold["positive_energy_rescue_threshold_kj"]), 75.0)
    assert np.isclose(float(plan_hold["clash_min_dist_a_threshold"]), 1.55)
    assert int(plan_hold["rescue_max_rounds"]) == 3
    assert int(plan_hold["rescue_minimize_max_iters"]) == 480
    assert np.isclose(float(plan_hold["rescue_relax_ps"]), 25.0)
    assert int(plan_hold["rescue_relax_steps"]) == 25000

    plan_no_hold = build_abfe_openmm_staged_prep_plan(
        minimize_max_iters="bad",
        has_lig_hold=False,
        sim_ns=0.1,
        timestep_ps=0.0,
    )
    assert int(plan_no_hold["max_it"]) == 1500
    min_ops_no_hold = list(plan_no_hold["minimize_ops"])
    assert len(min_ops_no_hold) == 1
    assert min_ops_no_hold[0]["set_k_hold"] is None
    assert int(min_ops_no_hold[0]["minimize_max_iters"]) == 1500
    # timestep_ps invalid -> fallback default step counts
    assert int(plan_no_hold["eq_hold_steps"]) == 25000
    assert int(plan_no_hold["eq_soft_steps"]) == 25000
    assert int(plan_no_hold["eq_free_steps"]) == 120000
    # sim_ns below floor should clamp to 0.2 -> free phase 120 ps
    assert np.isclose(float(plan_no_hold["eq_free_ps"]), 120.0)
    eq_ops_no_hold = list(plan_no_hold["equilibration_ops"])
    assert len(eq_ops_no_hold) == 1
    assert eq_ops_no_hold[0]["set_k_hold"] is None
    assert int(eq_ops_no_hold[0]["steps"]) == 120000


def test_abfe_runtime_openmm_context_setup_plan():
    plan_platform = build_abfe_openmm_context_setup_plan(
        temperature_k=310.0,
        timestep_ps=0.002,
        friction_inv_ps=4.0,
        picked_platform_name="OpenCL",
        has_platform=True,
        platform_properties={"Precision": "mixed"},
    )
    assert plan_platform["ok"] is True
    assert np.isclose(float(plan_platform["temperature_k"]), 310.0)
    assert np.isclose(float(plan_platform["timestep_ps"]), 0.002)
    assert np.isclose(float(plan_platform["friction_inv_ps"]), 4.0)
    ics = list(plan_platform["integrator_candidates"])
    assert [str(x["kind"]) for x in ics] == ["LangevinMiddleIntegrator", "LangevinIntegrator"]
    attempts = list(plan_platform["context_attempts"])
    assert len(attempts) == 2
    assert str(attempts[0]["mode"]) == "platform"
    assert bool(attempts[0]["use_platform"]) is True
    assert bool(attempts[0]["use_properties"]) is True
    assert dict(attempts[0]["properties"]) == {"Precision": "mixed"}
    assert str(attempts[1]["mode"]) == "default"
    assert bool(attempts[1]["use_platform"]) is False

    plan_default = build_abfe_openmm_context_setup_plan(
        temperature_k="bad",
        timestep_ps="bad",
        friction_inv_ps="bad",
        picked_platform_name="",
        has_platform=False,
        platform_properties=None,
    )
    assert np.isclose(float(plan_default["temperature_k"]), 300.0)
    assert np.isclose(float(plan_default["timestep_ps"]), 0.001)
    assert np.isclose(float(plan_default["friction_inv_ps"]), 5.0)
    attempts_default = list(plan_default["context_attempts"])
    assert len(attempts_default) == 1
    assert str(attempts_default[0]["mode"]) == "default"
    assert bool(attempts_default[0]["use_platform"]) is False
    assert bool(attempts_default[0]["use_properties"]) is False


def test_openmm_platform_selection_plan_prefers_cpu_on_macos():
    plan = build_openmm_platform_selection_plan(
        ["Metal", "CPU", "Reference"],
        os_name="darwin",
        cpu_count=10,
    )

    assert plan["picked_platform_name"] == "CPU"
    # two cores stay free for PyMOL and the OS
    assert plan["platform_properties"] == {"Threads": "8"}
    assert "macos_auto_prefers_cpu" in plan["notes"]


def test_openmm_platform_selection_plan_honors_explicit_platform_override():
    plan = build_openmm_platform_selection_plan(
        ["Metal", "CPU", "Reference"],
        os_name="darwin",
        platform_override="Metal",
    )

    assert plan["picked_platform_name"] == "Metal"
    assert plan["platform_properties"] == {"Precision": "mixed"}
    assert "explicit_platform_override:Metal" in plan["notes"]


def test_openmm_platform_selection_plan_reads_the_leonardo_env_name(monkeypatch):
    # LEONARDO_OPENMM_PLATFORM made code_v2 skip the hardware pick, but this plan ignored it: macOS got the CPU
    monkeypatch.delenv("PROTLIGINTERACT_OPENMM_PLATFORM", raising=False)
    monkeypatch.setenv("LEONARDO_OPENMM_PLATFORM", "OpenCL")
    plan = build_openmm_platform_selection_plan(["OpenCL", "CPU", "Reference"], os_name="darwin")
    assert plan["picked_platform_name"] == "OpenCL"
    assert "explicit_platform_override:OpenCL" in plan["notes"]


def test_abfe_runtime_md_sample_payload():
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],   # protein heavy
            [0.2, 0.0, 0.0],   # site anchor
            [0.05, 0.0, 0.0],  # ligand heavy
            [0.05, 0.1, 0.0],  # ligand heavy
        ],
        dtype=float,
    )
    payload = build_abfe_md_sample_payload(
        xyz_nm=xyz,
        sample_index=4,
        sample_every_ps=2.0,
        e_lj=-10.0,
        e_c=-2.5,
        lam_value=0.75,
        prot_heavy_idx=[0],
        lig_heavy_idx=[2, 3],
        lig_masses=[12.0, 16.0],
        local_lig_idx=[2, 3],
        site_anchor_idx=[1],
        site_com0_nm=[0.1, 0.0, 0.0],
    )
    assert payload["ok"] is True
    assert int(payload["frame"]) == 5
    assert np.isclose(float(payload["time_ps"]), 10.0)
    assert np.isclose(float(payload["e_tot"]), -12.5)
    assert np.isclose(float(payload["dmin_a"]), 0.5)
    assert np.isclose(float(payload["protein_contact_count_4a"]), 1.0)
    assert np.isfinite(float(payload["site_com_dist_a"]))
    assert np.isfinite(float(payload["site_min_dist_a"]))
    assert np.isfinite(float(payload["reactive_dist_a"]))
    assert np.isclose(float(payload["lam_value"]), 0.75)
    coords_a = np.asarray(payload["coords_a"], dtype=float)
    assert coords_a.shape == (4, 3)
    assert np.isclose(float(coords_a[2, 0]), 0.5)

    payload_light = build_abfe_md_sample_payload(
        xyz_nm=xyz,
        sample_index=1,
        sample_every_ps=1.0,
        e_lj=-2.0,
        e_c=-1.0,
        prot_heavy_idx=[0],
        lig_heavy_idx=[2],
        lig_masses=[12.0],
        local_lig_idx=[2],
        site_anchor_idx=[1],
        site_com0_nm=[0.1, 0.0, 0.0],
        include_coords=False,
    )
    assert bool(payload_light["coords_included"]) is False
    assert np.asarray(payload_light["coords_a"], dtype=float).shape == (0,)
    assert np.isclose(float(payload_light["e_tot"]), -3.0)

    payload_no_site = build_abfe_md_sample_payload(
        xyz_nm=xyz,
        sample_index=0,
        sample_every_ps=1.0,
        e_lj=-1.0,
        e_c=-1.0,
        lam_value=0.0,
        prot_heavy_idx=[0],
        lig_heavy_idx=[2],
        lig_masses=[12.0],
        local_lig_idx=[2],
        site_anchor_idx=[],
        site_com0_nm=[0.0, 0.0, 0.0],
    )
    assert np.isnan(float(payload_no_site["site_min_dist_a"]))
    assert np.isnan(float(payload_no_site["site_com_dist_a"]))
    assert np.isnan(float(payload_no_site["reactive_dist_a"]))


def test_abfe_runtime_local_resource_plan_caps_coordinate_states():
    plan = build_abfe_local_resource_plan(
        n_atoms=50000,
        n_samples=5000,
        requested_max_traj_states=1500,
        coordinate_memory_budget_mb=64.0,
        live_plot_max_points=500,
        rescore_max_frames=128,
        rescore_memory_budget_mb=32.0,
    )
    assert plan["ok"] is True
    assert bool(plan["memory_guard_active"]) is True
    assert float(plan["estimated_saved_states_mb"]) <= 64.0 + float(plan["mb_per_state"])
    assert int(plan["estimated_saved_state_count"]) < 1500
    assert int(plan["plot_update_stride"]) >= 10
    saved = list(plan["saved_sample_indices"])
    assert saved[0] == 0
    assert saved[-1] == 4999

    save0 = build_abfe_state_save_decision(
        sample_index=0,
        n_samples=5000,
        saved_sample_indices=saved,
    )
    save_last = build_abfe_state_save_decision(
        sample_index=4999,
        n_samples=5000,
        saved_sample_indices=[],
        state_save_stride=999999,
    )
    assert bool(save0["should_store"]) is True
    assert bool(save_last["should_store"]) is True


def test_abfe_runtime_live_update_decision():
    d1 = build_abfe_live_update_decision(
        sample_index=0,
        n_samples=10,
        live_view_ready=True,
        view_update_stride=3,
        live_plot_enabled=True,
        plot_update_stride=4,
    )
    assert d1["ok"] is True
    assert bool(d1["is_last_sample"]) is False
    assert bool(d1["should_update_live_view"]) is True   # tick 0
    assert bool(d1["should_update_live_plot"]) is True   # tick 0

    d2 = build_abfe_live_update_decision(
        sample_index=5,
        n_samples=10,
        live_view_ready=True,
        view_update_stride=3,
        live_plot_enabled=True,
        plot_update_stride=4,
    )
    assert bool(d2["should_update_live_view"]) is False  # 5%3 != 0 and not last
    assert bool(d2["should_update_live_plot"]) is False  # 5%4 != 0 and not last

    d3 = build_abfe_live_update_decision(
        sample_index=9,
        n_samples=10,
        live_view_ready=False,
        view_update_stride=3,
        live_plot_enabled=False,
        plot_update_stride=4,
    )
    assert bool(d3["is_last_sample"]) is True
    assert bool(d3["should_update_live_view"]) is False  # disabled readiness
    assert bool(d3["should_update_live_plot"]) is False  # disabled flag

    d4 = build_abfe_live_update_decision(
        sample_index="bad",
        n_samples="bad",
        live_view_ready=True,
        view_update_stride=0,
        live_plot_enabled=True,
        plot_update_stride=0,
    )
    assert int(d4["sample_index"]) == 0
    assert int(d4["n_samples"]) == 0
    assert int(d4["view_update_stride"]) == 1
    assert int(d4["plot_update_stride"]) == 1


def test_abfe_runtime_live_view_update_plan():
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
        ],
        dtype=float,
    )

    skip_payload = build_abfe_live_view_update_plan(
        should_update_live_view=False,
        xyz_nm=xyz,
        live_view_object="abfe_obj",
    )
    assert skip_payload["ok"] is True
    assert bool(skip_payload["should_apply"]) is False
    assert str(skip_payload["reason"]) == "update_not_requested"

    payload = build_abfe_live_view_update_plan(
        should_update_live_view=True,
        xyz_nm=xyz,
        live_view_object="abfe_obj",
        state=2,
        scale_to_a=10.0,
        refresh=True,
    )
    assert bool(payload["should_apply"]) is True
    assert str(payload["reason"]) == ""
    assert str(payload["object_name"]) == "abfe_obj"
    assert int(payload["state"]) == 2
    assert bool(payload["refresh"]) is True
    coords_a = np.asarray(payload["coords_a"], dtype=float)
    assert coords_a.shape == (2, 3)
    assert np.allclose(coords_a[1], [1.0, 2.0, 3.0])

    missing_obj = build_abfe_live_view_update_plan(
        should_update_live_view=True,
        xyz_nm=xyz,
        live_view_object="",
    )
    assert bool(missing_obj["should_apply"]) is False
    assert str(missing_obj["reason"]) == "missing_live_view_object"

    bad_xyz = build_abfe_live_view_update_plan(
        should_update_live_view=True,
        xyz_nm=[1.0, 2.0, 3.0],
        live_view_object="abfe_obj",
    )
    assert bool(bad_xyz["should_apply"]) is False
    assert str(bad_xyz["reason"]) == "invalid_xyz_nm"

    bad_state_scale = build_abfe_live_view_update_plan(
        should_update_live_view=True,
        xyz_nm=xyz,
        live_view_object="abfe_obj",
        state=0,
        scale_to_a=float("nan"),
        refresh=False,
    )
    assert bool(bad_state_scale["should_apply"]) is True
    assert int(bad_state_scale["state"]) == 1
    assert bool(bad_state_scale["refresh"]) is False
    coords_bad = np.asarray(bad_state_scale["coords_a"], dtype=float)
    assert np.allclose(coords_bad[1], [1.0, 2.0, 3.0])


def test_abfe_runtime_ui_process_events_decision():
    d1 = build_abfe_ui_process_events_decision(
        sample_index=0,
        n_samples=10,
        pump_stride=3,
        enabled=True,
    )
    assert d1["ok"] is True
    assert bool(d1["pump_tick"]) is True
    assert bool(d1["is_last_sample"]) is False
    assert bool(d1["should_pump_events"]) is True

    d2 = build_abfe_ui_process_events_decision(
        sample_index=1,
        n_samples=10,
        pump_stride=3,
        enabled=True,
    )
    assert bool(d2["pump_tick"]) is False
    assert bool(d2["is_last_sample"]) is False
    assert bool(d2["should_pump_events"]) is False

    d3 = build_abfe_ui_process_events_decision(
        sample_index=9,
        n_samples=10,
        pump_stride=100,
        enabled=True,
    )
    assert bool(d3["pump_tick"]) is False
    assert bool(d3["is_last_sample"]) is True
    assert bool(d3["should_pump_events"]) is True

    d4 = build_abfe_ui_process_events_decision(
        sample_index=0,
        n_samples=10,
        pump_stride=1,
        enabled=False,
    )
    assert bool(d4["should_pump_events"]) is False

    d5 = build_abfe_ui_process_events_decision(
        sample_index="bad",
        n_samples="bad",
        pump_stride=0,
        enabled=True,
    )
    assert int(d5["sample_index"]) == 0
    assert int(d5["n_samples"]) == 0
    assert int(d5["pump_stride"]) == 1
    assert bool(d5["should_pump_events"]) is True


def test_abfe_runtime_md_progress_dialog_message():
    payload_default = build_abfe_md_progress_dialog_message(
        bias_enabled=False,
        steering_mode="pull_hold_return",
    )
    assert payload_default["ok"] is True
    assert bool(payload_default["bias_enabled"]) is False
    assert str(payload_default["steering_mode"]) == "pull_hold_return"
    assert str(payload_default["message"]) == "Running ABFE-lite whole-protein MD (gas phase)..."

    payload_return = build_abfe_md_progress_dialog_message(
        bias_enabled=True,
        steering_mode="pull_hold_return",
    )
    assert bool(payload_return["bias_enabled"]) is True
    assert str(payload_return["steering_mode"]) == "pull_hold_return"
    assert (
        str(payload_return["message"])
        == "Running ABFE-lite whole-protein MD (gas phase, pull/hold/return steering)..."
    )

    payload_unknown = build_abfe_md_progress_dialog_message(
        bias_enabled=True,
        steering_mode="weird_mode",
    )
    assert str(payload_unknown["steering_mode"]) == "pull_hold"
    assert str(payload_unknown["message"]) == "Running ABFE-lite whole-protein MD (gas phase, pull/hold steering)..."


def test_abfe_runtime_md_progress_dialog_plan():
    payload_default = build_abfe_md_progress_dialog_plan(
        n_samples=25,
        bias_enabled=False,
        steering_mode="pull_hold_return",
    )
    assert payload_default["ok"] is True
    assert int(payload_default["minimum"]) == 0
    assert int(payload_default["maximum"]) == 25
    assert str(payload_default["cancel_text"]) == "Cancel"
    assert str(payload_default["modality"]) == "window_modal"
    assert str(payload_default["steering_mode"]) == "pull_hold_return"
    assert str(payload_default["message"]) == "Running ABFE-lite whole-protein MD (gas phase)..."

    payload_bias_return = build_abfe_md_progress_dialog_plan(
        n_samples=10,
        bias_enabled=True,
        steering_mode="pull_hold_return",
        cancel_text="Stop",
        minimum=3,
        modality="non_modal",
    )
    assert int(payload_bias_return["minimum"]) == 3
    assert int(payload_bias_return["maximum"]) == 10
    assert str(payload_bias_return["cancel_text"]) == "Stop"
    assert str(payload_bias_return["modality"]) == "non_modal"
    assert str(payload_bias_return["steering_mode"]) == "pull_hold_return"
    assert (
        str(payload_bias_return["message"])
        == "Running ABFE-lite whole-protein MD (gas phase, pull/hold/return steering)..."
    )

    payload_sanitized = build_abfe_md_progress_dialog_plan(
        n_samples=-2,
        bias_enabled=True,
        steering_mode="unknown_mode",
        cancel_text="   ",
        minimum=-5,
        modality="bad_mode",
    )
    assert int(payload_sanitized["minimum"]) == 0
    assert int(payload_sanitized["maximum"]) == 0
    assert str(payload_sanitized["cancel_text"]) == "Cancel"
    assert str(payload_sanitized["modality"]) == "window_modal"
    assert str(payload_sanitized["steering_mode"]) == "pull_hold"
    assert str(payload_sanitized["message"]) == "Running ABFE-lite whole-protein MD (gas phase, pull/hold steering)..."


def test_abfe_runtime_md_progress_step_decision():
    step_first = build_abfe_md_progress_step_decision(
        sample_index=0,
        n_samples=10,
        was_canceled=False,
    )
    assert step_first["ok"] is True
    assert bool(step_first["should_abort"]) is False
    assert bool(step_first["is_last_sample"]) is False
    assert int(step_first["progress_current_value"]) == 0
    assert int(step_first["progress_next_value"]) == 1
    assert bool(step_first["should_set_progress"]) is True
    assert bool(step_first["completed"]) is False

    step_last = build_abfe_md_progress_step_decision(
        sample_index=9,
        n_samples=10,
        was_canceled=False,
    )
    assert bool(step_last["is_last_sample"]) is True
    assert int(step_last["progress_next_value"]) == 10
    assert bool(step_last["completed"]) is True

    step_canceled = build_abfe_md_progress_step_decision(
        sample_index=5,
        n_samples=10,
        was_canceled=True,
    )
    assert bool(step_canceled["was_canceled"]) is True
    assert bool(step_canceled["should_abort"]) is True
    assert bool(step_canceled["should_set_progress"]) is False
    assert bool(step_canceled["completed"]) is False

    step_invalid = build_abfe_md_progress_step_decision(
        sample_index="bad",
        n_samples="bad",
        was_canceled=False,
    )
    assert int(step_invalid["sample_index"]) == 0
    assert int(step_invalid["n_samples"]) == 0
    assert int(step_invalid["progress_current_value"]) == 0
    assert int(step_invalid["progress_next_value"]) == 0


def test_abfe_runtime_md_progress_finalize_decision():
    done = build_abfe_md_progress_finalize_decision(
        n_samples=10,
        n_sampled=10,
        was_canceled=False,
    )
    assert done["ok"] is True
    assert int(done["final_value"]) == 10
    assert bool(done["completed"]) is True

    canceled_partial = build_abfe_md_progress_finalize_decision(
        n_samples=10,
        n_sampled=4,
        was_canceled=True,
    )
    assert int(canceled_partial["final_value"]) == 4
    assert bool(canceled_partial["completed"]) is False

    canceled_force_max = build_abfe_md_progress_finalize_decision(
        n_samples=10,
        n_sampled=4,
        was_canceled=True,
        set_to_max_on_cancel=True,
    )
    assert int(canceled_force_max["final_value"]) == 10
    assert bool(canceled_force_max["completed"]) is False

    invalid = build_abfe_md_progress_finalize_decision(
        n_samples=-1,
        n_sampled=99,
        was_canceled=False,
    )
    assert int(invalid["n_samples"]) == 0
    assert int(invalid["n_sampled"]) == 0
    assert int(invalid["final_value"]) == 0
    assert bool(invalid["completed"]) is False


def test_abfe_runtime_live_view_preflight_helper():
    class _Atom:
        def __init__(self, index):
            self.index = index

    atoms = [_Atom(1), _Atom(2), _Atom(3)]

    payload_ok = prepare_abfe_live_view_preflight(
        local_globals=[0, 1, 2],
        atoms=atoms,
        n_local=3,
        loaded_object="prot_obj",
        live_view_object="abfe_local_md_live",
    )
    assert payload_ok["ok"] is True
    assert payload_ok["can_create"] is True
    assert int(payload_ok["n_local"]) == 3
    assert list(payload_ok["local_model_indices"]) == [1, 2, 3]
    assert str(payload_ok["idx_selection"]) == "index 1+2+3"
    assert str(payload_ok["create_selection"]) == "(prot_obj) and (index 1+2+3)"
    assert str(payload_ok["reason"]) == ""

    payload_missing_obj = prepare_abfe_live_view_preflight(
        local_globals=[0, 1, 2],
        atoms=atoms,
        n_local=3,
        loaded_object="",
    )
    assert payload_missing_obj["can_create"] is False
    assert str(payload_missing_obj["reason"]) == "missing_loaded_object"

    payload_mismatch = prepare_abfe_live_view_preflight(
        local_globals=[0, 1, 2],
        atoms=atoms,
        n_local=4,
        loaded_object="prot_obj",
    )
    assert payload_mismatch["can_create"] is False
    assert str(payload_mismatch["reason"]) == "local_index_count_mismatch"

    payload_bad_idx = prepare_abfe_live_view_preflight(
        local_globals=[99],
        atoms=atoms,
        n_local=1,
        loaded_object="prot_obj",
    )
    assert payload_bad_idx["can_create"] is False
    assert str(payload_bad_idx["reason"]) == "index_resolution_failed"
    assert list(payload_bad_idx["local_model_indices"]) == []

    payload_invalid_n = prepare_abfe_live_view_preflight(
        local_globals=[0],
        atoms=atoms,
        n_local=0,
        loaded_object="prot_obj",
    )
    assert payload_invalid_n["can_create"] is False
    assert str(payload_invalid_n["reason"]) == "invalid_n_local"


def test_abfe_runtime_build_steering_lambda_schedule():
    payload_disabled = build_abfe_steering_lambda_schedule(
        n_samples=25,
        steering_mode="pull_hold",
        steering_enabled=False,
    )
    lam_disabled = np.asarray(payload_disabled["lambda_schedule"], dtype=float)
    assert int(payload_disabled["n_samples"]) == 25
    assert int(payload_disabled["n_warm"]) == 0
    assert int(payload_disabled["n_active"]) == 0
    assert lam_disabled.shape == (25,)
    assert np.allclose(lam_disabled, 0.0)

    payload_single = build_abfe_steering_lambda_schedule(
        n_samples=1,
        steering_mode="pull_hold",
        steering_enabled=True,
    )
    lam_single = np.asarray(payload_single["lambda_schedule"], dtype=float)
    assert lam_single.shape == (1,)
    assert np.allclose(lam_single, [1.0])

    payload_pull_hold = build_abfe_steering_lambda_schedule(
        n_samples=100,
        steering_mode="pull_hold",
        steering_enabled=True,
    )
    lam_pull_hold = np.asarray(payload_pull_hold["lambda_schedule"], dtype=float)
    assert int(payload_pull_hold["n_warm"]) == 6
    assert int(payload_pull_hold["n_active"]) == 94
    assert int(payload_pull_hold["n_cycles"]) == 1
    assert np.allclose(lam_pull_hold[:6], 0.0)
    assert float(np.min(lam_pull_hold)) >= 0.0
    assert float(np.max(lam_pull_hold)) <= 1.0
    assert np.all(np.diff(lam_pull_hold[6:]) >= -1e-12)
    assert np.isclose(float(lam_pull_hold[-1]), 1.0)

    payload_pull_hold_return = build_abfe_steering_lambda_schedule(
        n_samples=900,
        steering_mode="pull_hold_return",
        steering_enabled=True,
    )
    lam_pull_hold_return = np.asarray(payload_pull_hold_return["lambda_schedule"], dtype=float)
    assert int(payload_pull_hold_return["n_warm"]) == 54
    assert int(payload_pull_hold_return["n_active"]) == 846
    assert int(payload_pull_hold_return["n_cycles"]) == 1
    assert np.allclose(lam_pull_hold_return[:54], 0.0)
    assert float(np.min(lam_pull_hold_return)) >= 0.0
    assert float(np.max(lam_pull_hold_return)) <= 1.0
    assert np.allclose(lam_pull_hold_return[-50:], 0.0)
    assert np.isclose(float(lam_pull_hold_return[-1]), 0.0)


def test_abfe_runtime_build_adaptive_steering_state_payload():
    payload_disabled = build_abfe_adaptive_steering_state_payload(
        enabled=False,
        steering_mode="pull_hold_return",
        initial_pull_k_kj_nm2=900.0,
        n_direction_candidates=4,
        ligand_heavy_atoms=18,
    )
    assert payload_disabled["ok"] is True
    assert bool(payload_disabled["enabled"]) is False
    assert int(payload_disabled["max_direction_switches"]) == 0

    payload_small = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=800.0,
        n_direction_candidates=2,
        ligand_heavy_atoms=8,
    )
    payload_large = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold_return",
        initial_pull_k_kj_nm2=800.0,
        n_direction_candidates=4,
        ligand_heavy_atoms=28,
    )
    assert bool(payload_small["enabled"]) is True
    assert int(payload_small["max_direction_switches"]) == 1
    assert int(payload_large["max_direction_switches"]) == 3
    assert float(payload_large["k_max_kj_nm2"]) > float(payload_small["k_max_kj_nm2"])
    assert float(payload_small["k_max_kj_nm2"]) > float(payload_small["initial_pull_k_kj_nm2"])
    assert int(payload_large["update_interval_samples"]) >= int(payload_small["update_interval_samples"])
    assert int(payload_large["stall_window_samples"]) > int(payload_large["update_interval_samples"])
    assert str(payload_large["phase"]) == "pull"
    assert int(payload_large["far_hold_target_samples"]) >= 2
    assert float(payload_large["far_unbound_site_com_a"]) >= 10.0
    assert float(payload_large["target_escape_site_com_a"]) >= float(payload_large["far_unbound_site_com_a"])
    assert bool(payload_large["keep_pull_lambda_high_until_far_hold"]) is True
    assert bool(payload_large["high_lambda_seen"]) is False
    assert bool(payload_large["surface_detach_enabled"]) is True

    payload_default = build_abfe_adaptive_steering_state_payload(enabled=True, steering_mode="invalid_mode")
    assert str(payload_default["steering_mode"]) == "pull_hold"


def test_abfe_runtime_adaptive_steering_state_probe_defaults_scale_with_descriptors():
    pocket_small = _nan_pocket_descriptor_row()
    pocket_small["pocket_atom_density_per_a3"] = 0.015
    pocket_small["pocket_frac_polar"] = 0.20
    pocket_small["pocket_frac_hydrophobic"] = 0.60
    pocket_small["pocket_charge_density"] = 0.00
    pocket_large = _nan_pocket_descriptor_row()
    pocket_large["pocket_atom_density_per_a3"] = 0.055
    pocket_large["pocket_frac_polar"] = 0.55
    pocket_large["pocket_frac_hydrophobic"] = 0.25
    pocket_large["pocket_charge_density"] = -0.045

    payload_small = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=800.0,
        n_direction_candidates=3,
        ligand_heavy_atoms=10,
        formal_charge_e=0.0,
        ring_count=1,
        rotatable_bonds=1,
        pocket_desc=pocket_small,
        direction_clearances_nm=[1.30, 1.10, 1.00],
        direction_collision_penalties=[0.0, 5.0, 10.0],
        enable_probe_phase=True,
    )
    payload_large = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=800.0,
        n_direction_candidates=4,
        ligand_heavy_atoms=26,
        formal_charge_e=2.0,
        ring_count=3,
        rotatable_bonds=8,
        pocket_desc=pocket_large,
        direction_clearances_nm=[0.55, 0.60, 0.75, 0.80],
        direction_collision_penalties=[120.0, 80.0, 35.0, 15.0],
        enable_probe_phase=True,
    )

    assert bool(payload_small["probe_enabled"]) is True
    assert bool(payload_large["probe_enabled"]) is True
    assert str(payload_small["phase"]) == "probe"
    assert str(payload_large["phase"]) == "bound_settle"
    assert bool(payload_large["bound_settle_enabled"]) is True
    assert int(payload_large["bound_settle_target_samples"]) >= 2
    assert float(payload_small["current_pull_k_kj_nm2"]) < float(payload_small["initial_pull_k_kj_nm2"])
    assert float(payload_large["current_pull_k_kj_nm2"]) < float(payload_large["initial_pull_k_kj_nm2"])
    assert np.isclose(float(payload_large["current_pull_distance_scale"]), 1.0)
    assert float(payload_large["initial_pull_k_kj_nm2"]) > float(payload_small["initial_pull_k_kj_nm2"])
    assert float(payload_large["k_max_kj_nm2"]) > float(payload_small["k_max_kj_nm2"])
    assert float(payload_large["far_unbound_site_com_a"]) > float(payload_small["far_unbound_site_com_a"])
    assert float(payload_large["far_unbound_min_dist_a"]) >= float(payload_small["far_unbound_min_dist_a"])
    assert int(payload_large["far_unbound_max_contact_count_4a"]) >= int(payload_small["far_unbound_max_contact_count_4a"])
    assert float(payload_large["best_corridor_clearance_nm"]) < float(payload_small["best_corridor_clearance_nm"])
    assert float(payload_large["target_pull_distance_scale"]) > float(payload_small["current_pull_distance_scale"])
    assert float(payload_large["pull_distance_scale_max"]) > float(payload_small["pull_distance_scale_max"])
    assert bool(payload_large["pull_surface_detach_enabled"]) is True


def test_abfe_runtime_large_ligand_stall_increases_pull_distance_and_activates_pull_detach():
    pocket_large = _nan_pocket_descriptor_row()
    pocket_large["pocket_atom_density_per_a3"] = 0.060
    pocket_large["pocket_frac_polar"] = 0.18
    pocket_large["pocket_frac_hydrophobic"] = 0.82
    pocket_large["pocket_charge_density"] = -0.010
    stalled_state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=1400.0,
        n_direction_candidates=4,
        ligand_heavy_atoms=28,
        formal_charge_e=-1.0,
        ring_count=4,
        rotatable_bonds=9,
        pocket_desc=pocket_large,
        direction_clearances_nm=[0.55, 0.60, 0.72, 0.80],
        direction_collision_penalties=[120.0, 80.0, 40.0, 15.0],
        update_interval_samples=1,
        stall_window_samples=4,
    )
    stalled_state["phase"] = "pull"
    stalled_state["current_pull_distance_scale"] = float(stalled_state["target_pull_distance_scale"])
    stalled_state["stall_count"] = 1
    stalled = build_abfe_adaptive_steering_update_decision(
        adaptive_state=stalled_state,
        sample_index=4,
        lam_value=0.8,
        current_pull_k_kj_nm2=float(stalled_state["current_pull_k_kj_nm2"]),
        current_direction_index=0,
        direction_count=4,
        switches_used=0,
        min_dist_a=[2.5, 2.4, 2.4, 2.3],
        reactive_dist_a=[1.02, 1.03, 1.03, 1.04],
        site_min_dist_a=[2.2, 2.2, 2.2, 2.2],
        site_com_dist_a=[2.20, 2.24, 2.27, 2.30],
        contact_count_4a=[10.0, 10.0, 9.0, 9.0],
        total_e_kj=[-150.0, -149.0, -148.0, -147.0],
    )
    assert bool(stalled["should_adapt"]) is True
    assert str(stalled["reason"]) == "stall_increase_force_and_distance"
    assert bool(stalled["update_pull_k"]) is True
    assert bool(stalled["update_pull_distance_scale"]) is True
    assert float(stalled["pull_distance_scale"]) > float(stalled_state["current_pull_distance_scale"])
    assert bool(stalled["adaptive_state"]["pull_surface_detach_active"]) is True
    assert "pull_surface_detach" in list(stalled["notes"])


def test_abfe_runtime_surface_grazing_activates_pull_detach_before_far_hold():
    pocket_large = _nan_pocket_descriptor_row()
    pocket_large["pocket_atom_density_per_a3"] = 0.060
    pocket_large["pocket_frac_hydrophobic"] = 0.80
    state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=1400.0,
        n_direction_candidates=4,
        ligand_heavy_atoms=28,
        formal_charge_e=1.0,
        ring_count=2,
        rotatable_bonds=4,
        pocket_desc=pocket_large,
        direction_clearances_nm=[0.55, 0.60, 0.72, 0.80],
        direction_collision_penalties=[120.0, 80.0, 40.0, 15.0],
        update_interval_samples=1,
        stall_window_samples=4,
    )
    state["phase"] = "pull"
    state["current_pull_distance_scale"] = float(state["target_pull_distance_scale"])
    decision = build_abfe_adaptive_steering_update_decision(
        adaptive_state=state,
        sample_index=4,
        lam_value=1.0,
        current_pull_k_kj_nm2=float(state["current_pull_k_kj_nm2"]),
        current_direction_index=0,
        direction_count=4,
        switches_used=0,
        min_dist_a=[2.8, 2.7, 2.7, 2.7],
        reactive_dist_a=[8.5, 10.5, 12.5, 14.0],
        site_min_dist_a=[6.0, 7.0, 8.0, 9.0],
        site_com_dist_a=[9.0, 11.0, 13.0, 16.0],
        contact_count_4a=[30.0, 31.0, 30.0, 28.0],
        total_e_kj=[-520.0, -515.0, -512.0, -510.0],
    )
    assert bool(decision["should_adapt"]) is True
    assert str(decision["reason"]) == "surface_graze_detach_boost"
    assert bool(decision["adaptive_state"]["pull_surface_detach_active"]) is True
    assert bool(decision["update_pull_distance_scale"]) is True
    assert "surface_grazing" in list(decision["notes"])


def test_abfe_runtime_bound_settle_phase_transitions_into_probe():
    pocket_large = _nan_pocket_descriptor_row()
    pocket_large["pocket_atom_density_per_a3"] = 0.055
    pocket_large["pocket_frac_hydrophobic"] = 0.78
    pocket_large["pocket_frac_polar"] = 0.20
    state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=1200.0,
        n_direction_candidates=4,
        ligand_heavy_atoms=26,
        formal_charge_e=-1.0,
        ring_count=3,
        rotatable_bonds=8,
        pocket_desc=pocket_large,
        enable_probe_phase=True,
        sample_every_ps=2.0,
    )
    assert str(state["phase"]) == "bound_settle"
    state["bound_settle_target_samples"] = 2
    first = build_abfe_adaptive_steering_update_decision(
        adaptive_state=state,
        sample_index=1,
        lam_value=0.1,
        current_pull_k_kj_nm2=float(state["current_pull_k_kj_nm2"]),
        current_direction_index=0,
        direction_count=4,
        switches_used=0,
        min_dist_a=[2.8, 2.8],
        reactive_dist_a=[1.2, 1.2],
        site_min_dist_a=[2.3, 2.3],
        site_com_dist_a=[3.5, 3.6],
        contact_count_4a=[8.0, 8.0],
        total_e_kj=[-140.0, -139.0],
    )
    assert str(first["reason"]) == "bound_settle"
    assert str(first["phase"]) == "bound_settle"
    second = build_abfe_adaptive_steering_update_decision(
        adaptive_state=dict(first["adaptive_state"]),
        sample_index=2,
        lam_value=0.1,
        current_pull_k_kj_nm2=float(first["adaptive_state"]["current_pull_k_kj_nm2"]),
        current_direction_index=0,
        direction_count=4,
        switches_used=0,
        min_dist_a=[2.8, 2.8, 2.8],
        reactive_dist_a=[1.2, 1.2, 1.2],
        site_min_dist_a=[2.3, 2.3, 2.3],
        site_com_dist_a=[3.5, 3.6, 3.7],
        contact_count_4a=[8.0, 8.0, 8.0],
        total_e_kj=[-140.0, -139.0, -138.5],
    )
    assert str(second["reason"]) == "bound_settle_complete_enter_probe"
    assert str(second["phase"]) == "probe"
    assert bool(second["update_pull_distance_scale"]) is True
    assert float(second["pull_distance_scale"]) > 1.0


def test_abfe_runtime_far_hold_relaxes_bias_for_detached_large_ligand():
    pocket_large = _nan_pocket_descriptor_row()
    pocket_large["pocket_atom_density_per_a3"] = 0.060
    pocket_large["pocket_frac_hydrophobic"] = 0.80
    pocket_large["pocket_frac_polar"] = 0.18
    state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=1600.0,
        n_direction_candidates=4,
        ligand_heavy_atoms=28,
        formal_charge_e=-1.0,
        ring_count=4,
        rotatable_bonds=9,
        pocket_desc=pocket_large,
        sample_every_ps=2.0,
        update_interval_samples=1,
        stall_window_samples=4,
    )
    state["phase"] = "far_hold"
    state["current_pull_k_kj_nm2"] = float(state["initial_pull_k_kj_nm2"])
    state["far_hold_count"] = int(state.get("far_hold_relax_after_samples", 3))
    state["far_hold_target_k_kj_nm2"] = float(state["current_pull_k_kj_nm2"])
    relaxed = build_abfe_adaptive_steering_update_decision(
        adaptive_state=state,
        sample_index=6,
        lam_value=1.0,
        current_pull_k_kj_nm2=float(state["current_pull_k_kj_nm2"]),
        current_direction_index=0,
        direction_count=4,
        switches_used=0,
        min_dist_a=[3.6, 3.8, 4.0, 4.2],
        reactive_dist_a=[6.0, 6.2, 6.3, 6.5],
        site_min_dist_a=[2.0, 2.0, 2.0, 2.0],
        site_com_dist_a=[13.2, 13.6, 14.0, 14.4],
        contact_count_4a=[1.0, 1.0, 0.0, 0.0],
        total_e_kj=[-90.0, -88.0, -87.0, -86.0],
    )
    assert str(relaxed["phase"]) == "far_hold"
    assert bool(relaxed["update_pull_k"]) is True
    assert float(relaxed["pull_k_kj_nm2"]) < float(state["current_pull_k_kj_nm2"])
    assert float(relaxed["pull_k_kj_nm2"]) >= float(state["far_hold_k_floor_kj_nm2"])
    assert "far_hold_bias_relax" in list(relaxed["notes"])


def test_abfe_runtime_far_hold_enters_detached_settle_for_large_ligand():
    pocket_large = _nan_pocket_descriptor_row()
    pocket_large["pocket_atom_density_per_a3"] = 0.060
    pocket_large["pocket_frac_hydrophobic"] = 0.80
    pocket_large["pocket_frac_polar"] = 0.18
    state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=1600.0,
        n_direction_candidates=4,
        ligand_heavy_atoms=28,
        formal_charge_e=-1.0,
        ring_count=4,
        rotatable_bonds=9,
        pocket_desc=pocket_large,
        sample_every_ps=2.0,
        update_interval_samples=1,
        stall_window_samples=4,
    )
    state["phase"] = "far_hold"
    state["current_pull_k_kj_nm2"] = float(state["initial_pull_k_kj_nm2"])
    state["far_hold_count"] = int(state.get("far_hold_target_samples", 4))
    state["far_hold_target_k_kj_nm2"] = float(state["current_pull_k_kj_nm2"])
    moved = build_abfe_adaptive_steering_update_decision(
        adaptive_state=state,
        sample_index=8,
        lam_value=1.0,
        current_pull_k_kj_nm2=float(state["current_pull_k_kj_nm2"]),
        current_direction_index=0,
        direction_count=4,
        switches_used=0,
        min_dist_a=[3.8, 4.0, 4.2, 4.4],
        reactive_dist_a=[6.2, 6.3, 6.5, 6.7],
        site_min_dist_a=[2.0, 2.0, 2.0, 2.0],
        site_com_dist_a=[13.5, 13.9, 14.3, 14.7],
        contact_count_4a=[1.0, 0.0, 0.0, 0.0],
        total_e_kj=[-88.0, -87.0, -86.0, -85.0],
    )
    assert str(moved["reason"]) == "enter_detached_settle"
    assert str(moved["phase"]) == "detached_settle"
    assert bool(moved["update_pull_k"]) is True
    assert float(state["detached_settle_lambda_value"]) <= 0.02
    assert float(state["detached_settle_k_floor_kj_nm2"]) <= 20.0
    assert float(moved["pull_k_kj_nm2"]) == float(state["detached_settle_k_floor_kj_nm2"])
    assert int(moved["adaptive_state"]["detached_settle_count"]) == 1


def test_abfe_runtime_probe_direction_score_prefers_clear_progress():
    score_good = build_abfe_probe_direction_score_payload(
        progress_site_com_a=0.90,
        progress_reactive_a=0.30,
        latest_min_dist_a=3.80,
        latest_contact_count_4a=1.0,
        latest_total_e_kj=-95.0,
        median_total_e_kj=-100.0,
        clearance_nm=1.20,
        collision_penalty=5.0,
        soft_clash=False,
        hard_clash=False,
        energy_spike=False,
    )
    score_bad = build_abfe_probe_direction_score_payload(
        progress_site_com_a=0.08,
        progress_reactive_a=0.01,
        latest_min_dist_a=1.70,
        latest_contact_count_4a=10.0,
        latest_total_e_kj=55.0,
        median_total_e_kj=-80.0,
        clearance_nm=0.35,
        collision_penalty=180.0,
        soft_clash=True,
        hard_clash=True,
        energy_spike=True,
    )

    assert score_good["ok"] is True
    assert score_bad["ok"] is True
    assert float(score_good["score"]) > float(score_bad["score"])
    assert float(score_good["score"]) > 1.0
    assert float(score_bad["score"]) < 0.0


def test_abfe_runtime_adaptive_steering_surface_detach_direction_payload():
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.35, 0.02, 0.0],
            [0.42, 0.08, 0.0],
            [1.2, 0.0, 0.0],
        ],
        dtype=float,
    )
    payload = build_abfe_surface_detach_direction_payload(
        coords_nm=xyz,
        prot_heavy_idx=[0, 1],
        lig_heavy_idx=[2, 3],
        site_anchor_idx=[0, 1],
        fallback_direction=[0.0, 0.0, 1.0],
        contact_cutoff_nm=0.45,
    )
    assert payload["ok"] is True
    assert str(payload["reason"]) == "surface_contacts"
    assert int(payload["contact_count"]) >= 1
    det = np.asarray(payload["direction"], dtype=float)
    assert np.isclose(float(np.linalg.norm(det)), 1.0)
    assert float(det[0]) > 0.5


def test_abfe_runtime_tunnel_restraint_payload_validates_axis_and_radius():
    payload = build_abfe_tunnel_restraint_payload(
        enabled=True,
        origin_nm=[1.0, 2.0, 3.0],
        direction=[0.0, 0.0, 5.0],
        radius_a=3.5,
        k_radial_kj_mol_nm2=700.0,
        pull_distance_a=12.0,
    )

    assert payload["ok"] is True
    assert payload["enabled"] is True
    assert np.allclose(np.asarray(payload["direction"], dtype=float), [0.0, 0.0, 1.0])
    assert np.isclose(float(payload["radius_nm"]), 0.35)
    assert float(payload["zmin_nm"]) < 0.0
    assert float(payload["zmax_nm"]) > 1.2
    assert np.isclose(float(payload["k_radial_kj_mol_nm2"]), 700.0)

    disabled = build_abfe_tunnel_restraint_payload(enabled=False)
    assert disabled["ok"] is False
    assert disabled["reason"] == "disabled"

    invalid = build_abfe_tunnel_restraint_payload(
        enabled=True,
        origin_nm=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 0.0],
    )
    assert invalid["ok"] is False
    assert invalid["reason"] == "zero_direction"


def test_abfe_runtime_probe_phase_selects_best_corridor_then_enters_pull():
    probe_state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=800.0,
        n_direction_candidates=3,
        ligand_heavy_atoms=12,
        direction_clearances_nm=[0.45, 1.10, 0.70],
        direction_collision_penalties=[90.0, 5.0, 25.0],
        enable_probe_phase=True,
        update_interval_samples=1,
        stall_window_samples=4,
    )

    probe_switch = build_abfe_adaptive_steering_update_decision(
        adaptive_state=probe_state,
        sample_index=4,
        lam_value=0.8,
        current_pull_k_kj_nm2=float(probe_state["current_pull_k_kj_nm2"]),
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
        min_dist_a=[2.3, 2.2, 2.2, 2.1, 2.1, 2.0],
        reactive_dist_a=[1.00, 1.01, 1.01, 1.02, 1.03, 1.03],
        site_min_dist_a=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        site_com_dist_a=[2.00, 2.02, 2.04, 2.05, 2.06, 2.08],
        contact_count_4a=[10.0, 10.0, 10.0, 10.0, 9.0, 9.0],
        total_e_kj=[-120.0, -119.0, -118.0, -117.0, -116.0, -115.0],
    )
    assert bool(probe_switch["should_adapt"]) is True
    assert bool(probe_switch["switch_direction"]) is True
    assert int(probe_switch["direction_index"]) == 1
    assert str(probe_switch["phase"]) == "probe"
    assert str(probe_switch["reason"]) == "probe_switch_next_corridor"

    final_probe_state = dict(probe_switch["adaptive_state"])
    final_probe_state["probe_scores"] = [0.25, 2.60, np.nan]
    final_probe_state["probe_counts"] = [1, 1, 0]
    final_probe_state["phase"] = "probe"
    probe_select = build_abfe_adaptive_steering_update_decision(
        adaptive_state=final_probe_state,
        sample_index=8,
        lam_value=0.8,
        current_pull_k_kj_nm2=float(final_probe_state["current_pull_k_kj_nm2"]),
        current_direction_index=2,
        direction_count=3,
        switches_used=0,
        min_dist_a=[2.7, 2.7, 2.7, 2.8, 2.8, 2.8],
        reactive_dist_a=[1.10, 1.12, 1.13, 1.15, 1.17, 1.20],
        site_min_dist_a=[2.2, 2.2, 2.3, 2.3, 2.3, 2.4],
        site_com_dist_a=[2.4, 2.5, 2.6, 2.7, 2.9, 3.2],
        contact_count_4a=[8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
        total_e_kj=[-130.0, -129.0, -129.0, -128.0, -128.0, -127.0],
    )
    assert bool(probe_select["should_adapt"]) is True
    assert str(probe_select["reason"]) == "probe_complete_select_corridor"
    assert str(probe_select["phase"]) == "pull"
    assert int(probe_select["direction_index"]) == 1
    assert bool(probe_select["switch_direction"]) is True
    assert float(probe_select["pull_k_kj_nm2"]) >= float(final_probe_state["k_min_kj_nm2"])


def test_abfe_runtime_build_adaptive_steering_update_decision():
    disabled = build_abfe_adaptive_steering_update_decision(
        adaptive_state={"enabled": False},
        sample_index=5,
        lam_value=0.8,
        current_pull_k_kj_nm2=900.0,
        current_direction_index=0,
        direction_count=2,
        switches_used=0,
        min_dist_a=[2.4, 2.3, 2.2, 2.1],
        reactive_dist_a=[1.0, 1.1, 1.2, 1.3],
        site_min_dist_a=[2.0, 2.1, 2.2, 2.3],
        site_com_dist_a=[3.0, 3.5, 4.0, 4.5],
        total_e_kj=[-50.0, -48.0, -47.0, -46.0],
    )
    assert disabled["ok"] is True
    assert bool(disabled["should_adapt"]) is False
    assert str(disabled["reason"]) == "disabled"
    assert str(disabled["phase"]) == "pull"

    stalled_state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        initial_pull_k_kj_nm2=800.0,
        n_direction_candidates=3,
        ligand_heavy_atoms=10,
        update_interval_samples=1,
        stall_window_samples=4,
    )
    stalled = build_abfe_adaptive_steering_update_decision(
        adaptive_state=stalled_state,
        sample_index=4,
        lam_value=0.8,
        current_pull_k_kj_nm2=800.0,
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
        min_dist_a=[2.4, 2.3, 2.4, 2.3],
        reactive_dist_a=[1.00, 1.01, 1.03, 1.04],
        site_min_dist_a=[2.0, 2.0, 2.0, 2.0],
        site_com_dist_a=[2.00, 2.05, 2.10, 2.14],
        contact_count_4a=[12.0, 12.0, 11.0, 11.0],
        total_e_kj=[-120.0, -118.0, -119.0, -117.0],
    )
    assert bool(stalled["should_adapt"]) is True
    assert bool(stalled["update_pull_k"]) is True
    assert bool(stalled["switch_direction"]) is False
    assert str(stalled["reason"]) == "stall_increase_force"
    assert float(stalled["pull_k_kj_nm2"]) > 800.0
    assert bool(stalled["adaptive_state"]["high_lambda_seen"]) is True
    assert "stalled_progress" in list(stalled["notes"])

    switch_state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        initial_pull_k_kj_nm2=1500.0,
        n_direction_candidates=3,
        ligand_heavy_atoms=12,
        update_interval_samples=1,
        stall_window_samples=4,
    )
    switch_state["k_max_kj_nm2"] = 1500.0
    switched = build_abfe_adaptive_steering_update_decision(
        adaptive_state=switch_state,
        sample_index=4,
        lam_value=0.7,
        current_pull_k_kj_nm2=1500.0,
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
        min_dist_a=[2.5, 2.4, 2.5, 2.4],
        reactive_dist_a=[1.00, 1.01, 1.02, 1.03],
        site_min_dist_a=[2.1, 2.1, 2.1, 2.1],
        site_com_dist_a=[2.0, 2.1, 2.2, 2.24],
        contact_count_4a=[10.0, 10.0, 10.0, 10.0],
        total_e_kj=[-130.0, -128.0, -129.0, -127.0],
    )
    assert bool(switched["should_adapt"]) is True
    assert bool(switched["update_pull_k"]) is False
    assert bool(switched["switch_direction"]) is True
    assert int(switched["direction_index"]) == 1
    assert str(switched["reason"]) == "stall_switch_corridor"

    clash_state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        initial_pull_k_kj_nm2=1000.0,
        n_direction_candidates=3,
        ligand_heavy_atoms=14,
        update_interval_samples=1,
        stall_window_samples=4,
    )
    hard_clash = build_abfe_adaptive_steering_update_decision(
        adaptive_state=clash_state,
        sample_index=4,
        lam_value=0.85,
        current_pull_k_kj_nm2=1000.0,
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
        min_dist_a=[2.0, 1.9, 1.7, 1.5],
        reactive_dist_a=[1.00, 1.02, 1.04, 1.05],
        site_min_dist_a=[2.0, 1.9, 1.8, 1.7],
        site_com_dist_a=[2.0, 2.1, 2.2, 2.25],
        contact_count_4a=[9.0, 8.0, 7.0, 6.0],
        total_e_kj=[-200.0, -192.0, -195.0, -20.0],
    )
    assert bool(hard_clash["should_adapt"]) is True
    assert bool(hard_clash["update_pull_k"]) is True
    assert bool(hard_clash["switch_direction"]) is True
    assert float(hard_clash["pull_k_kj_nm2"]) < 1000.0
    assert int(hard_clash["direction_index"]) == 1
    assert str(hard_clash["reason"]) == "hard_clash_switch_corridor"
    assert "hard_clash" in list(hard_clash["notes"])
    assert "energy_spike" in list(hard_clash["notes"])

    far_state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold_return",
        initial_pull_k_kj_nm2=1500.0,
        n_direction_candidates=3,
        ligand_heavy_atoms=14,
        sample_every_ps=2.0,
        update_interval_samples=1,
        stall_window_samples=4,
    )
    enter_far_hold = build_abfe_adaptive_steering_update_decision(
        adaptive_state=far_state,
        sample_index=4,
        lam_value=0.8,
        current_pull_k_kj_nm2=1500.0,
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
        min_dist_a=[2.8, 3.1, 3.5, 3.8],
        reactive_dist_a=[4.0, 5.0, 6.0, 7.0],
        site_min_dist_a=[2.0, 2.0, 2.0, 2.0],
        site_com_dist_a=[8.0, 9.5, 10.1, 10.6],
        contact_count_4a=[6.0, 5.0, 3.0, 2.0],
        total_e_kj=[-120.0, -110.0, -105.0, -100.0],
    )
    assert str(enter_far_hold["reason"]) == "enter_far_hold"
    assert str(enter_far_hold["phase"]) == "far_hold"
    assert np.isclose(float(enter_far_hold["hold_lambda_value"]), 1.0)
    assert bool(enter_far_hold["update_pull_k"]) is True
    assert float(enter_far_hold["pull_k_kj_nm2"]) < 1500.0

    hold_state = dict(enter_far_hold["adaptive_state"])
    hold_state["phase"] = "far_hold"
    hold_state["far_hold_count"] = int(hold_state.get("far_hold_target_samples", 6)) - 1
    hold_state["last_update_sample"] = -999999
    far_hold_done = build_abfe_adaptive_steering_update_decision(
        adaptive_state=hold_state,
        sample_index=5,
        lam_value=1.0,
        current_pull_k_kj_nm2=float(enter_far_hold["pull_k_kj_nm2"]),
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
        min_dist_a=[3.1, 3.4, 3.7, 3.9],
        reactive_dist_a=[5.0, 6.0, 7.0, 8.0],
        site_min_dist_a=[2.0, 2.0, 2.0, 2.0],
        site_com_dist_a=[9.8, 10.2, 10.5, 10.9],
        contact_count_4a=[4.0, 3.0, 2.0, 2.0],
        total_e_kj=[-110.0, -108.0, -107.0, -105.0],
    )
    assert str(far_hold_done["reason"]) == "far_hold_complete_enter_return"
    assert bool(far_hold_done["far_hold_stable"]) is True
    assert str(far_hold_done["phase"]) == "return"
    # the return leg starts with the full pull force, not the reduced far-hold force
    assert bool(far_hold_done["update_pull_k"]) is True
    assert np.isclose(float(far_hold_done["pull_k_kj_nm2"]), float(hold_state["initial_pull_k_kj_nm2"]))


def test_abfe_runtime_steering_policy_state_and_bandit_push():
    policy_state = build_abfe_steering_policy_state_payload(
        enabled=True,
        policy_mode="bandit",
        ligand_heavy_atoms=18,
        formal_charge_e=1.0,
    )
    assert policy_state["ok"] is True
    assert bool(policy_state["enabled"]) is True
    assert str(policy_state["policy_mode"]) == "bandit"
    assert float(policy_state["k_push_scale"]) > 1.0

    adaptive_state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=900.0,
        n_direction_candidates=3,
        ligand_heavy_atoms=18,
    )
    heuristic = {
        "reason": "no_change",
        "phase": "pull",
        "update_pull_k": False,
        "pull_k_kj_nm2": 900.0,
        "switch_direction": False,
        "direction_index": 0,
        "should_adapt": False,
        "far_hold_stable": False,
        "progress_site_com_a": 0.05,
        "progress_reactive_a": 0.01,
        "latest_min_dist_a": 2.40,
        "latest_site_com_dist_a": 2.30,
        "latest_reactive_dist_a": 1.15,
        "latest_contact_count_4a": 11.0,
        "latest_total_e_kj": -120.0,
        "notes": ["stalled_progress"],
        "adaptive_state": dict(adaptive_state),
    }
    obs = build_abfe_steering_policy_observation_payload(
        heuristic_decision=heuristic,
        adaptive_state=adaptive_state,
        policy_state=policy_state,
        sample_index=8,
        lam_value=0.85,
        current_pull_k_kj_nm2=900.0,
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
    )
    assert obs["ok"] is True
    assert bool(obs["actionable"]) is True
    assert bool(obs["stalled"]) is True

    action = build_abfe_steering_bandit_action_payload(
        observation=obs,
        policy_state=policy_state,
    )
    assert str(action["action_name"]) == "increase_pull"
    assert float(action["requested_pull_k_scale"]) > 1.0

    final = build_abfe_steering_policy_safe_action_payload(
        heuristic_decision=heuristic,
        policy_action=action,
        adaptive_state=adaptive_state,
        policy_state=policy_state,
        current_pull_k_kj_nm2=900.0,
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
    )
    assert bool(final["policy_applied"]) is True
    assert bool(final["update_pull_k"]) is True
    assert float(final["pull_k_kj_nm2"]) > 900.0
    assert str(final["policy_action_name"]) == "increase_pull"


def test_abfe_runtime_steering_policy_trace_and_summary():
    adaptive_state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=1000.0,
        n_direction_candidates=3,
        ligand_heavy_atoms=20,
    )
    policy_state = build_abfe_steering_policy_state_payload(
        enabled=True,
        policy_mode="bandit",
        ligand_heavy_atoms=20,
        formal_charge_e=1.0,
    )
    heuristic = {
        "reason": "no_change",
        "phase": "far_hold",
        "update_pull_k": False,
        "pull_k_kj_nm2": 1000.0,
        "switch_direction": False,
        "direction_index": 0,
        "should_adapt": False,
        "far_hold_stable": False,
        "progress_site_com_a": 0.60,
        "progress_reactive_a": 0.20,
        "latest_min_dist_a": 3.10,
        "latest_site_com_dist_a": 10.8,
        "latest_reactive_dist_a": 7.5,
        "latest_contact_count_4a": 3.0,
        "latest_total_e_kj": -105.0,
        "notes": [],
        "adaptive_state": dict(adaptive_state),
    }
    obs = build_abfe_steering_policy_observation_payload(
        heuristic_decision=heuristic,
        adaptive_state=adaptive_state,
        policy_state=policy_state,
        sample_index=12,
        lam_value=1.0,
        current_pull_k_kj_nm2=1000.0,
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
    )
    action = build_abfe_steering_bandit_action_payload(
        observation=obs,
        policy_state=policy_state,
    )
    final = build_abfe_steering_policy_safe_action_payload(
        heuristic_decision=heuristic,
        policy_action=action,
        adaptive_state=adaptive_state,
        policy_state=policy_state,
        current_pull_k_kj_nm2=1000.0,
        current_direction_index=0,
        direction_count=3,
        switches_used=0,
    )
    trace_row = build_abfe_steering_policy_trace_payload(
        observation=obs,
        heuristic_decision=heuristic,
        policy_action=action,
        final_decision=final,
    )
    assert trace_row["ok"] is True
    assert str(trace_row["policy_action_name"]) in {"extend_hold", "detach_boost", "keep"}
    assert np.isfinite(float(trace_row["reward_estimate"]))
    assert "bias" in trace_row
    assert "k_fraction" in trace_row

    summary = build_abfe_steering_policy_summary_payload(
        trace_rows=[trace_row, dict(trace_row, policy_applied=False, policy_action_name="keep")],
    )
    assert summary["ok"] is True
    assert int(summary["n_rows"]) == 2
    assert "keep" in dict(summary["action_counts"])
    assert np.isfinite(float(summary["mean_reward_estimate"]))
    assert isinstance(summary["model_versions"], list)


def test_abfe_runtime_steering_policy_model_file_can_bias_action():
    feature_names = list(abfe_runtime.ABFE_STEERING_POLICY_FEATURE_NAMES)
    zero_weights = [0.0 for _ in feature_names]
    reduce_weights = [0.0 for _ in feature_names]
    reduce_weights[0] = 4.0

    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "policy_model.json")
        with open(model_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": "unit_test_bandit_v1",
                    "feature_names": feature_names,
                    "score_blend_alpha": 1.0,
                    "min_action_score": 0.1,
                    "actions": {
                        "keep": {"weights": zero_weights},
                        "increase_pull": {"weights": zero_weights},
                        "reduce_pull": {"weights": reduce_weights},
                        "switch_corridor": {"weights": zero_weights},
                        "detach_boost": {"weights": zero_weights},
                        "extend_hold": {"weights": zero_weights},
                    },
                },
                fh,
            )

        policy_state = build_abfe_steering_policy_state_payload(
            enabled=True,
            policy_mode="bandit",
            ligand_heavy_atoms=18,
            formal_charge_e=1.0,
            policy_model_path=model_path,
        )
        assert bool(policy_state["policy_model_loaded"]) is True
        assert str(policy_state["model_version"]) == "unit_test_bandit_v1"

        adaptive_state = build_abfe_adaptive_steering_state_payload(
            enabled=True,
            steering_mode="pull_hold",
            initial_pull_k_kj_nm2=900.0,
            n_direction_candidates=3,
            ligand_heavy_atoms=18,
        )
        heuristic = {
            "reason": "no_change",
            "phase": "pull",
            "update_pull_k": False,
            "pull_k_kj_nm2": 900.0,
            "switch_direction": False,
            "direction_index": 0,
            "should_adapt": False,
            "far_hold_stable": False,
            "progress_site_com_a": 0.05,
            "progress_reactive_a": 0.01,
            "latest_min_dist_a": 2.40,
            "latest_site_com_dist_a": 2.30,
            "latest_reactive_dist_a": 1.15,
            "latest_contact_count_4a": 11.0,
            "latest_total_e_kj": -120.0,
            "notes": ["stalled_progress"],
            "adaptive_state": dict(adaptive_state),
        }
        obs = build_abfe_steering_policy_observation_payload(
            heuristic_decision=heuristic,
            adaptive_state=adaptive_state,
            policy_state=policy_state,
            sample_index=8,
            lam_value=0.85,
            current_pull_k_kj_nm2=900.0,
            current_direction_index=0,
            direction_count=3,
            switches_used=0,
        )
        action = build_abfe_steering_bandit_action_payload(
            observation=obs,
            policy_state=policy_state,
        )
        assert bool(action["policy_model_used"]) is True
        assert str(action["policy_model_version"]) == "unit_test_bandit_v1"
        assert str(action["action_name"]) == "reduce_pull"


def test_abfe_runtime_steering_policy_model_ood_falls_back_to_heuristic():
    feature_names = list(abfe_runtime.ABFE_STEERING_POLICY_FEATURE_NAMES)
    zero_weights = [0.0 for _ in feature_names]
    reduce_weights = [0.0 for _ in feature_names]
    reduce_weights[feature_names.index("bias")] = 4.0

    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "policy_model.json")
        with open(model_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": "unit_test_ood_bandit_v1",
                    "feature_names": feature_names,
                    "score_blend_alpha": 1.0,
                    "min_action_score": 0.1,
                    "ood_enabled": True,
                    "ood_feature_names": ["pocket_n_atoms_scaled"],
                    "ood_center": [0.25],
                    "ood_scale": [0.10],
                    "ood_distance_threshold": 1.0,
                    "ood_max_abs_z": 1.0,
                    "actions": {
                        "keep": {"weights": zero_weights},
                        "increase_pull": {"weights": zero_weights},
                        "reduce_pull": {"weights": reduce_weights},
                        "switch_corridor": {"weights": zero_weights},
                        "detach_boost": {"weights": zero_weights},
                        "extend_hold": {"weights": zero_weights},
                    },
                },
                fh,
            )

        policy_state = build_abfe_steering_policy_state_payload(
            enabled=True,
            policy_mode="bandit",
            ligand_heavy_atoms=18,
            formal_charge_e=0.0,
            pocket_desc={"pocket_n_atoms": 420.0},
            policy_model_path=model_path,
        )
        adaptive_state = build_abfe_adaptive_steering_state_payload(
            enabled=True,
            steering_mode="pull_hold",
            initial_pull_k_kj_nm2=900.0,
            n_direction_candidates=2,
            ligand_heavy_atoms=18,
        )
        heuristic = {
            "reason": "pull_phase_active",
            "phase": "pull",
            "update_pull_k": False,
            "pull_k_kj_nm2": 900.0,
            "switch_direction": False,
            "direction_index": 0,
            "should_adapt": False,
            "far_hold_stable": False,
            "progress_site_com_a": 0.4,
            "progress_reactive_a": 0.2,
            "latest_min_dist_a": 4.5,
            "latest_site_com_dist_a": 11.0,
            "latest_reactive_dist_a": 8.0,
            "latest_contact_count_4a": 0.0,
            "latest_total_e_kj": -5.0,
            "notes": [],
            "adaptive_state": dict(adaptive_state),
        }
        obs = build_abfe_steering_policy_observation_payload(
            heuristic_decision=heuristic,
            adaptive_state=adaptive_state,
            policy_state=policy_state,
            sample_index=6,
            lam_value=1.0,
            current_pull_k_kj_nm2=900.0,
            current_direction_index=0,
            direction_count=2,
            switches_used=0,
        )
        action = build_abfe_steering_bandit_action_payload(
            observation=obs,
            policy_state=policy_state,
        )
        assert bool(action["policy_model_ood_enabled"]) is True
        assert bool(action["policy_model_ood_in_domain"]) is False
        assert bool(action["policy_model_used"]) is False
        assert str(action["action_name"]) == "keep"
        assert str(action["reason"]) == "ood_fallback"


def test_abfe_runtime_steering_policy_extend_hold_can_relax_hold_gap():
    feature_names = list(abfe_runtime.ABFE_STEERING_POLICY_FEATURE_NAMES)
    zero_weights = [0.0 for _ in feature_names]
    extend_weights = [0.0 for _ in feature_names]
    extend_weights[feature_names.index("phase_far_hold")] = 2.0
    extend_weights[feature_names.index("far_hold_sustain")] = 1.5
    extend_weights[feature_names.index("hold_progress_fraction")] = 1.0

    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "policy_model.json")
        with open(model_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": "unit_test_hold_bandit_v1",
                    "feature_names": feature_names,
                    "score_blend_alpha": 1.0,
                    "min_action_score": 0.1,
                    "actions": {
                        "keep": {"weights": zero_weights},
                        "increase_pull": {"weights": zero_weights},
                        "reduce_pull": {"weights": zero_weights},
                        "switch_corridor": {"weights": zero_weights},
                        "detach_boost": {"weights": zero_weights},
                        "extend_hold": {"weights": extend_weights},
                    },
                },
                fh,
            )

        policy_state = build_abfe_steering_policy_state_payload(
            enabled=True,
            policy_mode="bandit",
            ligand_heavy_atoms=18,
            formal_charge_e=1.0,
            policy_model_path=model_path,
        )
        adaptive_state = build_abfe_adaptive_steering_state_payload(
            enabled=True,
            steering_mode="pull_hold",
            initial_pull_k_kj_nm2=900.0,
            n_direction_candidates=3,
            ligand_heavy_atoms=18,
        )
        adaptive_state["phase"] = "far_hold"
        adaptive_state["far_hold_count"] = 2
        adaptive_state["far_hold_target_samples"] = 8
        adaptive_state["far_unbound_site_com_a"] = 12.0
        adaptive_state["far_hold_release_site_com_a"] = 11.2
        adaptive_state["unbound_hold_max_gap_samples"] = 3
        adaptive_state["unbound_hold_max_gap_cap_samples"] = 6
        heuristic = {
            "reason": "far_hold_stable",
            "phase": "far_hold",
            "update_pull_k": False,
            "pull_k_kj_nm2": 900.0,
            "switch_direction": False,
            "direction_index": 0,
            "should_adapt": False,
            "far_hold_stable": False,
            "progress_site_com_a": 0.40,
            "progress_reactive_a": 0.10,
            "latest_min_dist_a": 3.8,
            "latest_site_com_dist_a": 11.4,
            "latest_reactive_dist_a": 7.5,
            "latest_contact_count_4a": 1.0,
            "latest_total_e_kj": -110.0,
            "notes": [],
            "adaptive_state": dict(adaptive_state),
        }
        obs = build_abfe_steering_policy_observation_payload(
            heuristic_decision=heuristic,
            adaptive_state=adaptive_state,
            policy_state=policy_state,
            sample_index=12,
            lam_value=1.0,
            current_pull_k_kj_nm2=900.0,
            current_direction_index=0,
            direction_count=3,
            switches_used=0,
        )
        action = build_abfe_steering_bandit_action_payload(
            observation=obs,
            policy_state=policy_state,
        )
        assert str(action["action_name"]) == "extend_hold"
        assert int(action["requested_hold_gap_samples"]) == 1
        final = build_abfe_steering_policy_safe_action_payload(
            heuristic_decision=heuristic,
            policy_action=action,
            adaptive_state=adaptive_state,
            policy_state=policy_state,
            current_pull_k_kj_nm2=900.0,
            current_direction_index=0,
            direction_count=3,
            switches_used=0,
        )
        assert bool(final["policy_applied"]) is True
        assert int(final["policy_requested_hold_gap_samples"]) == 1
        assert int(final["adaptive_state"]["unbound_hold_max_gap_samples"]) == 4


def test_abfe_runtime_policy_trace_payload_includes_target_and_pocket_metadata():
    obs = {
        "sample_index": 3,
        "phase": "pull",
        "policy_mode": "bandit",
        "actionable": True,
        "current_pull_k_kj_nm2": 900.0,
        "current_direction_index": 0,
        "progress_site_com_a": 0.3,
        "progress_reactive_a": 0.1,
        "latest_min_dist_a": 3.8,
        "latest_site_com_dist_a": 10.5,
        "latest_contact_count_4a": 2.0,
        "size_scale": 0.6,
        "ligand_heavy_atoms": 18,
        "formal_charge_e": 1.0,
        "charge_density": 0.05,
        "ring_count": 1,
        "rotatable_bonds": 2,
        "target_hint": "thrombin",
        "ligand_smiles": "NC(=N)c1ccccc1",
        "ligand_scaffold": "c1ccccc1",
        "pocket_n_atoms": 42.0,
        "pocket_frac_hydrophobic": 0.55,
        "pocket_frac_polar": 0.20,
        "pocket_charge_density": -0.01,
        "far_unbound_min_dist_a": 3.5,
        "far_unbound_site_com_a": 10.0,
        "far_unbound_max_contact_count_4a": 4,
    }
    heuristic = {"reason": "pull_phase_active", "phase": "pull"}
    action = {"action_name": "keep", "policy_mode": "bandit"}
    final = {"policy_action_name": "keep", "policy_applied": False}
    row = build_abfe_steering_policy_trace_payload(
        observation=obs,
        heuristic_decision=heuristic,
        policy_action=action,
        final_decision=final,
    )
    assert str(row["target_hint"]) == "thrombin"
    assert str(row["ligand_scaffold"]) == "c1ccccc1"
    assert np.isclose(float(row["pocket_n_atoms"]), 42.0)
    assert "pocket_n_atoms_scaled" in row


def test_abfe_runtime_build_live_plot_setup_plan():
    setup_plan = build_abfe_live_plot_setup_plan(enabled=True)
    assert setup_plan["ok"] is True
    assert bool(setup_plan["should_create"]) is True
    assert int(setup_plan["nrows"]) == 2
    assert int(setup_plan["ncols"]) == 1
    assert np.allclose(np.asarray(setup_plan["figsize"], dtype=float), [8.0, 5.0])
    assert bool(setup_plan["sharex"]) is True
    assert str(setup_plan["title"]) == "ABFE-lite Live Sampling"
    assert str(setup_plan["energy_label"]) == "Energy (kJ/mol)"
    assert str(setup_plan["distance_label"]) == "Distance (Å)"
    assert str(setup_plan["time_label"]) == "Time (ps)"
    assert str(setup_plan["energy_line_label"]) == "Total E"
    assert str(setup_plan["distance_line_label"]) == "Reactive dist"
    assert str(setup_plan["energy_line_color"]) == "#1F77B4"
    assert str(setup_plan["energy_point_color"]) == "#D62728"
    assert str(setup_plan["distance_line_color"]) == "#9467BD"
    assert str(setup_plan["distance_point_color"]) == "#FF7F0E"
    assert np.isclose(float(setup_plan["line_width"]), 1.2)
    assert str(setup_plan["point_marker"]) == "o"
    assert np.isclose(float(setup_plan["point_size"]), 5.0)
    assert np.isclose(float(setup_plan["grid_alpha"]), 0.25)
    assert str(setup_plan["legend_loc"]) == "best"
    assert np.isclose(float(setup_plan["legend_fontsize"]), 8.0)
    assert bool(setup_plan["tight_layout"]) is True
    assert bool(setup_plan["draw_idle"]) is True
    assert bool(setup_plan["flush_events"]) is True
    assert bool(setup_plan["interactive"]) is True

    disabled_plan = build_abfe_live_plot_setup_plan(enabled=False)
    assert bool(disabled_plan["should_create"]) is False
    assert str(disabled_plan["reason"]) == "disabled"

    sanitized_plan = build_abfe_live_plot_setup_plan(
        enabled=True,
        figsize=("bad", -1),
        line_width=0.0,
        point_size=-1.0,
        grid_alpha=2.5,
        legend_fontsize=0.0,
        point_marker="",
    )
    assert np.allclose(np.asarray(sanitized_plan["figsize"], dtype=float), [8.0, 5.0])
    assert np.isclose(float(sanitized_plan["line_width"]), 1.2)
    assert np.isclose(float(sanitized_plan["point_size"]), 5.0)
    assert np.isclose(float(sanitized_plan["grid_alpha"]), 1.0)
    assert np.isclose(float(sanitized_plan["legend_fontsize"]), 8.0)
    assert str(sanitized_plan["point_marker"]) == "o"


def test_abfe_runtime_build_live_plot_state_payload():
    disabled_payload = build_abfe_live_plot_state_payload(enabled=False)
    assert disabled_payload["ok"] is True
    assert bool(disabled_payload["enabled"]) is False
    assert str(disabled_payload["reason"]) == "disabled"
    assert disabled_payload["fig"] is None
    assert disabled_payload["line_e"] is None
    assert "fig" in list(disabled_payload["missing_handles"])
    assert "line_d" in list(disabled_payload["missing_handles"])

    obj_fig = object()
    obj_ax_e = object()
    obj_ax_d = object()
    obj_line_e = object()
    obj_point_e = object()
    obj_line_d = object()
    obj_point_d = object()
    enabled_payload = build_abfe_live_plot_state_payload(
        enabled=True,
        fig=obj_fig,
        ax_e=obj_ax_e,
        ax_d=obj_ax_d,
        line_e=obj_line_e,
        point_e=obj_point_e,
        line_d=obj_line_d,
        point_d=obj_point_d,
        strict_handles=True,
    )
    assert bool(enabled_payload["enabled"]) is True
    assert str(enabled_payload["reason"]) == ""
    assert list(enabled_payload["missing_handles"]) == []
    assert enabled_payload["fig"] is obj_fig
    assert enabled_payload["point_d"] is obj_point_d

    missing_payload = build_abfe_live_plot_state_payload(
        enabled=True,
        fig=obj_fig,
        ax_e=obj_ax_e,
        ax_d=obj_ax_d,
        line_e=None,
        point_e=obj_point_e,
        line_d=obj_line_d,
        point_d=obj_point_d,
        strict_handles=True,
    )
    assert bool(missing_payload["enabled"]) is False
    assert str(missing_payload["reason"]) == "missing_handles"
    assert "line_e" in list(missing_payload["missing_handles"])
    assert missing_payload["fig"] is None

    missing_non_strict = build_abfe_live_plot_state_payload(
        enabled=True,
        fig=obj_fig,
        strict_handles=False,
    )
    assert bool(missing_non_strict["enabled"]) is True
    assert str(missing_non_strict["reason"]) == ""
    assert missing_non_strict["fig"] is obj_fig


def test_abfe_runtime_build_live_plot_update_payload():
    payload_reactive = build_abfe_live_plot_update_payload(
        times_ps=[1.0, 2.0, 3.0],
        total_e=[-10.0, -9.0, -8.0],
        reactive_dist=[4.0, 4.5, 5.0],
        min_dist=[6.0, 6.5, 7.0],
    )
    assert str(payload_reactive["distance_source"]) == "reactive_dist"
    assert np.allclose(np.asarray(payload_reactive["t_arr"], dtype=float), [1.0, 2.0, 3.0])
    assert np.allclose(np.asarray(payload_reactive["e_arr"], dtype=float), [-10.0, -9.0, -8.0])
    assert np.allclose(np.asarray(payload_reactive["d_arr"], dtype=float), [4.0, 4.5, 5.0])
    assert list(payload_reactive["point_e_x"]) == [3.0]
    assert list(payload_reactive["point_e_y"]) == [-8.0]
    assert list(payload_reactive["point_d_x"]) == [3.0]
    assert list(payload_reactive["point_d_y"]) == [5.0]
    assert bool(payload_reactive["has_point_d"]) is True

    payload_fallback = build_abfe_live_plot_update_payload(
        times_ps=[1.0, 2.0, 3.0],
        total_e=[-10.0, -9.0, -8.0],
        reactive_dist=[np.nan, np.nan, np.nan],
        min_dist=[6.0, 6.5, 7.0],
    )
    assert str(payload_fallback["distance_source"]) == "min_dist"
    assert np.allclose(np.asarray(payload_fallback["d_arr"], dtype=float), [6.0, 6.5, 7.0])
    assert bool(payload_fallback["has_point_d"]) is True
    assert list(payload_fallback["point_d_y"]) == [7.0]

    payload_nan = build_abfe_live_plot_update_payload(
        times_ps=[1.0, 2.0],
        total_e=[-10.0, -9.0],
        reactive_dist=[np.nan, np.nan],
        min_dist=[np.nan, np.nan],
    )
    assert str(payload_nan["distance_source"]) == "min_dist"
    assert bool(payload_nan["has_point_d"]) is False
    assert list(payload_nan["point_d_x"]) == []
    assert list(payload_nan["point_d_y"]) == []

    payload_trim = build_abfe_live_plot_update_payload(
        times_ps=[1.0, 2.0, 3.0, 4.0],
        total_e=[-10.0, -9.0, -8.0],
        reactive_dist=[4.0, 5.0],
        min_dist=[6.0, 7.0, 8.0, 9.0],
    )
    assert np.allclose(np.asarray(payload_trim["t_arr"], dtype=float), [1.0, 2.0])
    assert np.allclose(np.asarray(payload_trim["e_arr"], dtype=float), [-10.0, -9.0])
    assert np.allclose(np.asarray(payload_trim["d_arr"], dtype=float), [4.0, 5.0])

    payload_decimated = build_abfe_live_plot_update_payload(
        times_ps=list(range(100)),
        total_e=[float(i) for i in range(100)],
        reactive_dist=[float(i) * 0.1 for i in range(100)],
        min_dist=[],
        max_points=25,
    )
    assert int(payload_decimated["n_raw"]) == 100
    assert np.asarray(payload_decimated["t_arr"], dtype=float).size <= 25
    assert list(payload_decimated["point_e_x"]) == [99.0]
    assert list(payload_decimated["point_e_y"]) == [99.0]


def test_abfe_runtime_build_live_plot_update_default_payload():
    payload = build_abfe_live_plot_update_default_payload()
    assert payload["ok"] is True
    assert str(payload["distance_source"]) == "min_dist"
    assert tuple(np.asarray(payload["t_arr"], dtype=float).shape) == (0,)
    assert tuple(np.asarray(payload["e_arr"], dtype=float).shape) == (0,)
    assert tuple(np.asarray(payload["d_arr"], dtype=float).shape) == (0,)
    assert list(payload["point_e_x"]) == []
    assert list(payload["point_e_y"]) == []
    assert list(payload["point_d_x"]) == []
    assert list(payload["point_d_y"]) == []
    assert bool(payload["has_point_d"]) is False


def test_abfe_runtime_build_live_plot_apply_plan():
    skip_plan = build_abfe_live_plot_apply_plan(
        should_update_live_plot=False,
        plot_payload={},
    )
    assert skip_plan["ok"] is True
    assert bool(skip_plan["should_apply"]) is False
    assert str(skip_plan["reason"]) == "update_not_requested"

    payload = build_abfe_live_plot_update_payload(
        times_ps=[1.0, 2.0, 3.0],
        total_e=[-10.0, -9.0, -8.0],
        reactive_dist=[4.0, 4.5, 5.0],
        min_dist=[6.0, 6.5, 7.0],
    )
    apply_plan = build_abfe_live_plot_apply_plan(
        should_update_live_plot=True,
        plot_payload=payload,
        relim=True,
        autoscale=True,
        draw_idle=True,
        flush_events=False,
    )
    assert bool(apply_plan["should_apply"]) is True
    assert str(apply_plan["reason"]) == ""
    assert np.allclose(np.asarray(apply_plan["line_e_x"], dtype=float), [1.0, 2.0, 3.0])
    assert np.allclose(np.asarray(apply_plan["line_e_y"], dtype=float), [-10.0, -9.0, -8.0])
    assert np.allclose(np.asarray(apply_plan["line_d_y"], dtype=float), [4.0, 4.5, 5.0])
    assert list(apply_plan["point_e_x"]) == [3.0]
    assert list(apply_plan["point_d_y"]) == [5.0]
    assert bool(apply_plan["relim"]) is True
    assert bool(apply_plan["autoscale"]) is True
    assert bool(apply_plan["draw_idle"]) is True
    assert bool(apply_plan["flush_events"]) is False

    payload_no_point = build_abfe_live_plot_update_payload(
        times_ps=[1.0, 2.0],
        total_e=[-1.0, -2.0],
        reactive_dist=[np.nan, np.nan],
        min_dist=[np.nan, np.nan],
    )
    no_point_plan = build_abfe_live_plot_apply_plan(
        should_update_live_plot=True,
        plot_payload=payload_no_point,
    )
    assert bool(no_point_plan["should_apply"]) is True
    assert list(no_point_plan["point_d_x"]) == []
    assert list(no_point_plan["point_d_y"]) == []

    invalid_payload_plan = build_abfe_live_plot_apply_plan(
        should_update_live_plot=True,
        plot_payload={"t_arr": [1, 2, 3], "e_arr": [4, 5], "d_arr": [6]},
    )
    assert np.allclose(np.asarray(invalid_payload_plan["line_e_x"], dtype=float), [1.0])
    assert np.allclose(np.asarray(invalid_payload_plan["line_e_y"], dtype=float), [4.0])
    assert np.allclose(np.asarray(invalid_payload_plan["line_d_y"], dtype=float), [6.0])


def test_abfe_runtime_build_traj_state_pick_plan():
    payload_empty = build_abfe_traj_state_pick_plan(
        n_sampled=0,
        max_states_view=1500,
    )
    assert int(payload_empty["n_sampled"]) == 0
    assert int(payload_empty["stride_view"]) == 1
    assert list(payload_empty["picked_indices"]) == []
    assert int(payload_empty["n_selected"]) == 0

    payload_small = build_abfe_traj_state_pick_plan(
        n_sampled=5,
        max_states_view=1500,
    )
    assert int(payload_small["stride_view"]) == 1
    assert list(payload_small["picked_indices"]) == [0, 1, 2, 3, 4]
    assert int(payload_small["n_selected"]) == 5

    payload_stride = build_abfe_traj_state_pick_plan(
        n_sampled=10,
        max_states_view=3,
    )
    assert int(payload_stride["stride_view"]) == 4
    assert list(payload_stride["picked_indices"]) == [0, 4, 8, 9]
    assert int(payload_stride["n_selected"]) == 4

    payload_bad_max = build_abfe_traj_state_pick_plan(
        n_sampled=10,
        max_states_view=0,
    )
    assert int(payload_bad_max["max_states_view"]) == 1
    assert list(payload_bad_max["picked_indices"]) == [0, 9]


def test_abfe_runtime_build_final_traj_build_plan():
    payload_live = build_abfe_final_traj_build_plan(
        sampled_state_count=10,
        n_local_model_indices=5,
        n_local=5,
        local_model_indices=[1, 2, 3, 4, 5],
        idx_selection="",
        loaded_object="prot_obj",
        state_index=3,
        live_view_object="abfe_live_obj",
        live_view_atom_count=5,
        timestamp_token="123456",
        existing_objects=["abfe_local_md_traj_123456"],
        object_prefix="abfe_local_md_traj",
        max_states_view=3,
        group_name="ABFE.LocalMD",
    )
    assert payload_live["ok"] is True
    assert bool(payload_live["should_build"]) is True
    assert str(payload_live["reason"]) == ""
    assert str(payload_live["final_view_object"]) == "abfe_local_md_traj_123456"
    assert bool(payload_live["should_delete_existing"]) is True
    assert bool(payload_live["use_live_source"]) is True
    assert str(payload_live["create_mode"]) == "from_object"
    assert str(payload_live["create_selection"]) == "abfe_live_obj"
    assert int(payload_live["create_source_state"]) == 1
    assert list(payload_live["picked_indices"]) == [0, 4, 8, 9]
    load_steps = list(payload_live["load_steps"])
    assert [int(s["out_state"]) for s in load_steps] == [1, 2, 3, 4]
    assert [bool(s["duplicate_state_before_load"]) for s in load_steps] == [False, True, True, True]
    assert str(payload_live["group_name"]) == "ABFE.LocalMD"
    assert int(payload_live["frame_after_build"]) == 1
    assert "ABFE trajectory object ready:" in str(payload_live["ready_log_line"])

    payload_fallback = build_abfe_final_traj_build_plan(
        sampled_state_count=2,
        n_local_model_indices=2,
        n_local=2,
        local_model_indices=[11, 12],
        idx_selection="",
        loaded_object="prot_obj",
        state_index=4,
        live_view_object="abfe_live_obj",
        live_view_atom_count=1,
        timestamp_token="12:34:56",
        existing_objects=[],
        object_prefix="abfe_local_md_traj",
        max_states_view=1500,
    )
    assert bool(payload_fallback["should_build"]) is True
    assert bool(payload_fallback["use_live_source"]) is False
    assert str(payload_fallback["create_mode"]) == "from_selection"
    assert str(payload_fallback["idx_selection"]) == "index 11+12"
    assert str(payload_fallback["create_selection"]) == "(prot_obj) and (index 11+12)"
    assert int(payload_fallback["create_source_state"]) == 4
    assert str(payload_fallback["final_view_object"]) == "abfe_local_md_traj_123456"

    payload_no_samples = build_abfe_final_traj_build_plan(
        sampled_state_count=0,
        n_local_model_indices=2,
        n_local=2,
    )
    assert bool(payload_no_samples["should_build"]) is False
    assert str(payload_no_samples["reason"]) == "no_sampled_states"

    payload_mismatch = build_abfe_final_traj_build_plan(
        sampled_state_count=5,
        n_local_model_indices=1,
        n_local=2,
    )
    assert bool(payload_mismatch["should_build"]) is False
    assert str(payload_mismatch["reason"]) == "local_index_count_mismatch"

    payload_missing_create = build_abfe_final_traj_build_plan(
        sampled_state_count=5,
        n_local_model_indices=2,
        n_local=2,
        local_model_indices=[],
        idx_selection="",
        loaded_object="",
        live_view_object="abfe_live_obj",
        live_view_atom_count=1,
    )
    assert bool(payload_missing_create["should_build"]) is False
    assert str(payload_missing_create["reason"]) == "missing_create_selection"


def test_abfe_runtime_build_final_traj_post_build_ui_plan():
    payload = build_abfe_final_traj_post_build_ui_plan(
        final_view_object="abfe_local_md_traj_123456",
        update_structure_combo=True,
        frame_after_build=1,
        n_selected=4,
        n_sampled=10,
        ready_log_line="",
    )
    assert payload["ok"] is True
    assert bool(payload["should_apply"]) is True
    assert str(payload["reason"]) == ""
    assert bool(payload["should_refresh_structure_combo"]) is True
    assert str(payload["structure_combo_text"]) == "abfe_local_md_traj_123456"
    assert bool(payload["should_set_frame"]) is True
    assert int(payload["frame_index"]) == 1
    assert bool(payload["should_log_ready"]) is True
    assert (
        str(payload["ready_log_line"])
        == "ABFE trajectory object ready: abfe_local_md_traj_123456 (4/10 sampled states)."
    )

    payload_custom_log = build_abfe_final_traj_post_build_ui_plan(
        final_view_object="abfe_obj",
        update_structure_combo=False,
        frame_after_build=3,
        n_selected=2,
        n_sampled=5,
        ready_log_line="custom-ready-line",
    )
    assert bool(payload_custom_log["should_refresh_structure_combo"]) is False
    assert int(payload_custom_log["frame_index"]) == 3
    assert str(payload_custom_log["ready_log_line"]) == "custom-ready-line"

    payload_missing = build_abfe_final_traj_post_build_ui_plan(
        final_view_object="",
        update_structure_combo=True,
    )
    assert bool(payload_missing["should_apply"]) is False
    assert str(payload_missing["reason"]) == "missing_final_view_object"
    assert bool(payload_missing["should_set_frame"]) is False
    assert bool(payload_missing["should_log_ready"]) is False

    payload_sanitized = build_abfe_final_traj_post_build_ui_plan(
        final_view_object="abfe_obj",
        frame_after_build=0,
        n_selected=-1,
        n_sampled=-5,
    )
    assert int(payload_sanitized["frame_index"]) == 1
    assert (
        str(payload_sanitized["ready_log_line"])
        == "ABFE trajectory object ready: abfe_obj (0/0 sampled states)."
    )


def test_abfe_runtime_build_post_sampling_metrics():
    payload = build_abfe_post_sampling_metrics(
        frames=[1, 2, 3, 4],
        times_ps=[1.0, 2.0, 3.0, 4.0],
        lam_hist=[0.0, 0.5, 1.0, 1.0],
        bias_e=[0.0, 0.4, 2.0, 2.2],
        lj_e=[-2.0, -2.5, -3.0, -3.5],
        coul_e=[-0.5, -0.5, -0.5, -0.5],
        total_e=[-2.5, -3.0, -3.5, -4.0],
        min_dist=[2.8, 3.2, 4.1, 4.4],
        reactive_dist=[np.nan, np.nan, np.nan, np.nan],
        site_min_dist=[2.0, 2.5, 5.0, 6.0],
        site_com_dist=[3.0, 3.5, 10.5, 11.0],
        protein_contact_count_4a=[10.0, 9.0, 2.0, 1.0],
        lig_com=[[0.0, 0.0, 0.0], [0.4, 0.2, 0.2], [0.3, 0.2, 0.1], [0.2, 0.1, 0.2]],
        far_hold_stable_mask=[False, False, True, True],
        temperature_k=300.0,
        local_cutoff_nm=1.2,
    )
    assert payload["ok"] is True
    assert str(payload["distance_metric"]) == "site_anchor_min"
    assert np.allclose(np.asarray(payload["dist_arr"], dtype=float), [2.0, 2.5, 5.0, 6.0])
    assert float(payload["bound_cutoff_a"]) == 4.5
    assert int(payload["n_dist_valid"]) == 4
    assert int(payload["n_bound"]) == 2
    assert int(payload["n_unbound"]) == 2
    assert int(np.sum(np.asarray(payload["intermediate_mask"], dtype=bool))) == 0
    assert np.isclose(float(payload["p_bound"]), 0.5)
    assert np.isclose(float(payload["p_unbound"]), 0.5)
    assert np.isclose(float(payload["dg_pop_kj"]), 0.0)
    assert np.isfinite(float(payload["v_eff_nm3"]))
    assert np.isfinite(float(payload["kd_app_m"]))
    assert np.isfinite(float(payload["p_kd_app"]))
    assert np.allclose(np.asarray(payload["frames_valid"], dtype=int), [1, 2, 3, 4])
    assert np.isfinite(float(payload["dg_reweighted_kj"]))
    assert np.isfinite(float(payload["reweight_ess_total"]))
    assert int(np.sum(np.asarray(payload["bound_valid_mask"], dtype=bool))) == 2
    assert int(np.sum(np.asarray(payload["unbound_valid_mask"], dtype=bool))) == 2
    assert str(payload["estimator_window_mode"]) == "state_fallback"
    assert str(payload["binding_estimate_status"]) == "unbound_geometry_without_stable_hold"
    assert bool(payload["binding_estimate_allowed"]) is False

    payload_ragged = build_abfe_post_sampling_metrics(
        frames=[1, 2, 3, 4],
        times_ps=[1.0, 2.0],
        lam_hist=[0.0, 1.0, 1.0, 1.0],
        bias_e=[0.0, np.nan, 2.0],
        lj_e=[-2.0],
        coul_e=[-0.5],
        total_e=[-2.5, -3.0, -3.5, -4.0],
        min_dist=[2.8, 3.2],
        site_min_dist=[2.0, 2.5, 5.0, 6.0],
        site_com_dist=[3.0, 3.5, 10.5, 11.0],
        protein_contact_count_4a=[10.0, 9.0, 2.0, 1.0],
        lig_com=[[np.nan, np.nan, np.nan], [0.4, 0.2, 0.2], [0.8, 0.2, 0.1], [1.2, 0.1, 0.2]],
        far_hold_stable_mask=[False, False, True, True],
        temperature_k=300.0,
        local_cutoff_nm=1.2,
    )
    assert payload_ragged["ok"] is True
    assert np.asarray(payload_ragged["times_arr"], dtype=float).shape == (4,)
    assert np.asarray(payload_ragged["bias_arr"], dtype=float).shape == (4,)
    assert np.isfinite(float(payload_ragged["v_eff_nm3"]))

    payload_site_com = build_abfe_post_sampling_metrics(
        frames=[1, 2, 3],
        times_ps=[1.0, 2.0, 3.0],
        lam_hist=[0.0, 0.5, 1.0],
        lj_e=[-2.0, -2.5, -3.0],
        coul_e=[-0.5, -0.5, -0.5],
        total_e=[-2.5, -3.0, -3.5],
        min_dist=[2.5, 3.1, 4.0],
        reactive_dist=[np.nan, np.nan, np.nan],
        site_min_dist=[np.nan, np.nan, np.nan],
        site_com_dist=[3.5, 7.0, 10.5],
        protein_contact_count_4a=[9.0, 5.0, 2.0],
        lig_com=[[0.0, 0.0, 0.0]],
        far_hold_stable_mask=[False, False, True],
        temperature_k=300.0,
        local_cutoff_nm=1.0,
    )
    assert str(payload_site_com["distance_metric"]) == "reaction_site_com"
    assert float(payload_site_com["bound_cutoff_a"]) == 8.0
    assert np.allclose(np.asarray(payload_site_com["dist_arr"], dtype=float), [3.5, 7.0, 10.5], equal_nan=True)
    assert int(payload_site_com["n_bound"]) == 1
    assert int(payload_site_com["n_unbound"]) == 1
    assert int(np.sum(np.asarray(payload_site_com["intermediate_mask"], dtype=bool))) == 1

    payload_requires_hold = build_abfe_post_sampling_metrics(
        frames=[1, 2, 3],
        times_ps=[1.0, 2.0, 3.0],
        lam_hist=[0.0, 0.5, 1.0],
        lj_e=[-2.0, -2.5, -3.0],
        coul_e=[-0.5, -0.5, -0.5],
        total_e=[-2.5, -3.0, -3.5],
        min_dist=[2.5, 3.1, 4.0],
        reactive_dist=[np.nan, np.nan, np.nan],
        site_min_dist=[np.nan, np.nan, np.nan],
        site_com_dist=[3.5, 7.0, 10.5],
        protein_contact_count_4a=[9.0, 5.0, 2.0],
        lig_com=[[0.0, 0.0, 0.0]],
        far_hold_stable_mask=[False, False, False],
        temperature_k=300.0,
        local_cutoff_nm=1.0,
    )
    assert int(payload_requires_hold["n_bound"]) == 1
    assert int(payload_requires_hold["n_unbound"]) == 0
    assert int(np.sum(np.asarray(payload_requires_hold["intermediate_mask"], dtype=bool))) == 2
    assert str(payload_requires_hold["binding_estimate_status"]) == "no_stable_far_hold"

    payload_min_fallback = build_abfe_post_sampling_metrics(
        frames=[1, 2],
        times_ps=[1.0, 2.0],
        lam_hist=[0.0, 0.5],
        lj_e=[-2.0, -2.5],
        coul_e=[-0.5, -0.5],
        total_e=[-2.5, -3.0],
        min_dist=[4.0, 5.0],
        reactive_dist=[np.nan, np.nan],
        site_min_dist=[np.nan, np.nan],
        site_com_dist=[np.nan, np.nan],
        lig_com=[],
        temperature_k=300.0,
        local_cutoff_nm=1.0,
    )
    assert str(payload_min_fallback["distance_metric"]) == "min_heavy_any_protein"
    assert np.allclose(np.asarray(payload_min_fallback["dist_arr"], dtype=float), [4.0, 5.0], equal_nan=True)


def test_abfe_runtime_state_coverage_diagnostic_rejects_surface_association():
    diagnostic = build_abfe_state_coverage_diagnostic_payload(
        site_com_dist=[3.0, 8.0, 12.0, 16.2, 16.4],
        min_dist=[2.8, 2.7, 2.6, 2.7, 2.8],
        contact_count_4a=[12, 20, 28, 31, 30],
        bound_mask=[True, False, False, False, False],
        unbound_mask=[False, False, False, False, False],
        bound_hold_mask=[True, True, True, True, False],
        unbound_hold_mask=[False, False, False, False, False],
        far_hold_stable_mask=[False, False, False, False, False],
        unbound_site_com_threshold_a=10.0,
        unbound_min_dist_a=3.5,
        unbound_max_contact_count_4a=4,
        min_hold_state_frames=4,
    )
    assert diagnostic["ok"] is True
    assert str(diagnostic["status"]) == "surface_associated_not_bulk_unbound"
    assert bool(diagnostic["exit_coordinate_reached"]) is True
    assert bool(diagnostic["contact_clearance_reached"]) is False
    assert bool(diagnostic["binding_estimate_allowed"]) is False
    assert np.isclose(float(diagnostic["contact_count_at_max_exit_4a"]), 30.0)


def test_abfe_runtime_build_local_md_result_payload():
    result = build_abfe_local_md_result_payload(
        frames_arr=np.asarray([1, 2], dtype=int),
        times_arr=np.asarray([1.0, 2.0], dtype=float),
        lam_arr=np.asarray([0.0, 1.0], dtype=float),
        bias_arr=np.asarray([0.0, 1.5], dtype=float),
        total_arr=np.asarray([-5.0, -4.0], dtype=float),
        lj_arr=np.asarray([-4.0, -3.0], dtype=float),
        coul_arr=np.asarray([-1.0, -1.0], dtype=float),
        min_arr=np.asarray([4.0, 5.0], dtype=float),
        reactive_arr=np.asarray([np.nan, np.nan], dtype=float),
        site_min_arr=np.asarray([4.0, 5.0], dtype=float),
        site_com_arr=np.asarray([8.0, 9.0], dtype=float),
        dist_arr=np.asarray([4.0, 5.0], dtype=float),
        dist_metric="site_anchor_min",
        valid=np.asarray([True, True], dtype=bool),
        analysis_mask=np.asarray([True, True], dtype=bool),
        valid_fe=np.asarray([True, True], dtype=bool),
        frames_valid=np.asarray([1, 2], dtype=int),
        bias_valid=np.asarray([0.0, 1.5], dtype=float),
        total_valid=np.asarray([-5.0, -4.0], dtype=float),
        lj_valid=np.asarray([-4.0, -3.0], dtype=float),
        coul_valid=np.asarray([-1.0, -1.0], dtype=float),
        min_valid=np.asarray([4.0, 5.0], dtype=float),
        distance_valid=np.asarray([4.0, 5.0], dtype=float),
        bound_valid_mask=np.asarray([True, False], dtype=bool),
        unbound_valid_mask=np.asarray([False, True], dtype=bool),
        platform_name="CPU",
        sampling_mode="biased_pull_hold",
        steering_mode="pull_hold",
        use_whole_requested=True,
        local_cutoff_a=8.0,
        protein_atoms_total=100,
        protein_atoms_used=20,
        ligand_atoms_used=15,
        bias_enabled=True,
        guide_tunnel_enabled=True,
        guide_tunnel_active=True,
        guide_tunnel_radius_a=3.5,
        guide_tunnel_k_kj_nm2=650.0,
        guide_tunnel_axis=[1.0, 0.0, 0.0],
        guide_tunnel_origin_nm=[0.1, 0.2, 0.3],
        pull_distance_a=3.0,
        pull_k_kj_nm2=150.0,
        pull_k_effective_kj_nm2=np.nan,
        pull_clearance_nm=0.8,
        pull_collision_penalty=2000.0,
        bound_cutoff_a=4.5,
        n_dist_valid=2,
        n_bound=1,
        n_unbound=1,
        p_bound=0.5,
        p_unbound=0.5,
        dg_pop=0.0,
        p_bound_jeff=0.5,
        p_unbound_jeff=0.5,
        dg_pop_jeff=0.0,
        kd_app_m=1e-6,
        p_kd_app=6.0,
        kd_app_jeff_m=2e-6,
        p_kd_jeff=5.69897,
        p_bound_reweighted=0.6,
        p_unbound_reweighted=0.4,
        dg_reweighted=-1.0,
        kd_app_reweighted_m=1.5e-6,
        p_kd_reweighted=5.82,
        reweight_ess_total=1.8,
        reweight_ess_bound=1.0,
        reweight_ess_unbound=1.0,
        v_eff_nm3=1.2,
        sim_ns=0.5,
        sample_every_ps=1.0,
        ligand_charge_source="am1bcc",
        ligand_ff_mode="gaff2",
        ligand_xml_source="lig.xml",
        ligand_target_ph=6.8,
        protein_gap_count=0,
        abfe_traj_object="traj_obj",
        abfe_context_object="ctx_obj",
        target_hint="thrombin",
        pocket_desc={"pocket_n_atoms": 42.0},
        random_seed=12345,
        replica_index=2,
        replica_count=3,
    )
    assert np.allclose(np.asarray(result["frames"], dtype=int), [1, 2])
    assert str(result["energy_mode"]) == "openmm (CPU)"
    assert str(result["distance_metric"]) == "site_anchor_min"
    assert bool(result["local_mode_requested"]) is False
    assert bool(result["whole_protein_mode"]) is True
    assert bool(result["guide_tunnel_active"]) is True
    assert np.isclose(float(result["guide_tunnel_radius_a"]), 3.5)
    assert np.allclose(np.asarray(result["guide_tunnel_axis"], dtype=float), [1.0, 0.0, 0.0])
    assert np.isnan(float(result["pull_k_effective_kj_nm2"]))
    assert np.isclose(float(result["pull_clearance_a"]), 8.0)
    assert np.allclose(np.asarray(result["bias_energy_kj"], dtype=float), [0.0, 1.5])
    assert np.isclose(float(result["dg_reweighted_kj"]), -1.0)
    assert str(result["selected_estimator"]) == "gated_off"
    assert str(result["abfe_traj_object"]) == "traj_obj"
    assert str(result["target_hint"]) == "thrombin"
    assert np.isclose(float(result["ligand_target_ph"]), 6.8)
    assert np.isclose(float(result["pocket_n_atoms"]), 42.0)
    assert int(result["random_seed"]) == 12345
    assert int(result["replica_index"]) == 2
    assert int(result["replica_count"]) == 3
    assert "pocket_n_residues" in result


def test_abfe_runtime_free_energy_report_and_debug_helpers():
    assert str(abfe_distance_axis_label("reaction_site_com")) == "Reactive Distance to Initial Pocket (Å)"
    assert str(abfe_distance_axis_label("site_anchor_min")) == "Min Distance to Initial Pocket Anchor (Å)"
    assert str(abfe_distance_axis_label("unknown")) == "Distance (Å)"

    payload = build_abfe_free_energy_report_payload(
        protocol="abfe_lite_local_md",
        whole_protein_mode=True,
        local_active=False,
        prot_used=120,
        prot_total=300,
        local_cutoff_a=8.0,
        lig_chain="A",
        lig_resi="301",
        lig_resn="LIG",
        energy_mode="openmm (CPU)",
        dg_exp=-12.34,
        ci_low=-13.0,
        ci_high=-11.5,
        dg_cum=-11.8,
        n_samples=500,
        n_dist_valid=400,
        n_bound=250,
        n_unbound=150,
        p_bound=0.625,
        p_unbound=0.375,
        dg_pop=-1.5,
        dg_pop_jeff=-1.3,
        p_bound_reweighted=0.70,
        p_unbound_reweighted=0.30,
        dg_reweighted=-2.1,
        kd_app_m=2.3e-6,
        p_kd_app=5.64,
        kd_app_jeff_m=2.8e-6,
        p_kd_jeff=5.55,
        kd_app_reweighted_m=1.7e-6,
        p_kd_reweighted=5.77,
        reweight_ci_low=-2.6,
        reweight_ci_high=-1.8,
        reweight_ess_total=42.0,
        sim_ns=3.0,
        sample_every_ps=2.0,
        bias_enabled=True,
        sampling_mode="biased_pull_hold_return",
        pull_distance_a=12.0,
        pull_k_kj_nm2=550.0,
        pull_k_effective_kj_nm2=35.0,
        pull_clearance_a=4.5,
        bound_cutoff_a=4.5,
        ligand_ff_mode="gaff2",
        ligand_charge_source="am1bcc",
        ligand_target_ph=7.4,
        target_hint="thrombin",
        distance_metric="site_anchor_min",
        replica_summary={
            "replica_count": 3,
            "requested_replica_count": 3,
            "dg_exp_mean": -12.1,
            "dg_exp_sem": 0.4,
            "dg_exp_span": 1.1,
            "sim_ns_per_replica": 3.0,
            "sim_ns_total": 9.0,
            "unbound_coverage_fraction": 1.0,
            "exp_cum_gap_mean": 0.8,
            "convergence_quality": "high",
            "convergence_note": "replica spread is small",
        },
    )
    assert payload["ok"] is True
    assert "ABFE-lite whole-protein MD" in str(payload["suptitle"])
    assert str(payload["dist_axis_label"]) == "Min Distance to Initial Pocket Anchor (Å)"
    assert "Bias-reweighted" in str(payload["reweight_line"])
    assert "Selected estimator" in str(payload["selected_line"])
    assert "ΔG(pop)" in str(payload["kd_line"])
    assert "Steering:" in str(payload["message_text"])
    assert "Sampling:" in str(payload["message_text"])
    assert "Replicas:" in str(payload["message_text"])
    assert "Replica convergence" in str(payload["message_text"])
    assert "target pH=7.40" in str(payload["message_text"])
    assert "target hint=thrombin" in str(payload["message_text"])
    assert "Selected ΔG" in str(payload["message_html"])
    assert "Confidence:" in str(payload["message_html"])
    assert "ABFE-lite whole-protein MD Summary" == str(payload["dialog_title"])
    # ABFE-lite is experimental (not validated against experiment): every report says so
    assert "EXPERIMENTAL" in str(payload["message_text"])
    assert "EXPERIMENTAL" in str(payload["message_html"])

    payload_jeff = build_abfe_free_energy_report_payload(
        protocol="trajectory_energy_rescoring",
        whole_protein_mode=False,
        local_active=True,
        prot_used=20,
        prot_total=300,
        local_cutoff_a=8.0,
        lig_chain="A",
        lig_resi="301",
        lig_resn="LIG",
        energy_mode="numpy",
        dg_exp=-5.0,
        ci_low=-6.0,
        ci_high=-4.0,
        dg_cum=-4.8,
        n_samples=100,
        n_dist_valid=10,
        n_bound=10,
        n_unbound=0,
        p_bound=1.0,
        p_unbound=0.0,
        dg_pop=np.nan,
        dg_pop_jeff=-3.2,
        kd_app_m=np.nan,
        p_kd_app=np.nan,
        kd_app_jeff_m=1.0e-4,
        p_kd_jeff=4.0,
        sim_ns=np.nan,
        sample_every_ps=np.nan,
        bias_enabled=False,
        sampling_mode="",
        pull_distance_a=np.nan,
        pull_k_kj_nm2=np.nan,
        pull_k_effective_kj_nm2=np.nan,
        pull_clearance_a=np.nan,
        bound_cutoff_a=8.0,
        binding_estimate_status="surface_associated_not_bulk_unbound",
        binding_estimate_note="The ligand retained close protein contacts.",
        ligand_ff_mode="approximate",
        ligand_charge_source="heuristic",
        distance_metric="reaction_site_com",
    )
    assert "Single-state sampled: missing unbound state" in str(payload_jeff["kd_line"])
    assert "not reported as BFE/Kd" in str(payload_jeff["kd_line"])
    assert "Kd(app)" not in str(payload_jeff["kd_line"])
    assert "state_status=surface_associated_not_bulk_unbound" in str(payload_jeff["selected_line"])
    assert "fallback mode" in str(payload_jeff["intro"])
    assert str(payload_jeff["scope_txt"]) == "ABFE-lite local"

    dbg = build_abfe_free_energy_debug_line(
        protocol="abfe_lite_local_md",
        energy_mode="openmm (CPU)",
        whole_protein_mode=True,
        local_active=False,
        prot_used=120,
        prot_total=300,
        n_samples=500,
        sampling_mode="biased_pull_hold_return",
        bias_enabled=True,
        n_bound=250,
        n_dist_valid=400,
        n_unbound=150,
        dg_exp=-12.34,
        dg_cum=-11.8,
        dg_pop=-1.5,
        dg_reweighted=-2.1,
        kd_app_m=2.3e-6,
        kd_app_reweighted_m=1.7e-6,
        contact_frac_mean=0.625,
        target_hint="thrombin",
        ci_low=-13.0,
        ci_high=-11.5,
        replica_summary={"replica_count": 3, "convergence_quality": "high", "dg_exp_sem": 0.4, "dg_exp_span": 1.1},
    )
    assert "scope=whole" in str(dbg)
    assert "DGreweight=" in str(dbg)
    assert "target_hint=thrombin" in str(dbg)
    assert "CI95=" in str(dbg)
    assert "replicas=3" in str(dbg)


def test_abfe_runtime_free_energy_numeric_payload():
    n = 30
    frames_valid = np.arange(1, n + 1, dtype=int)
    total_valid = np.linspace(-40.0, -10.0, n, dtype=float)
    distance_valid = np.linspace(3.0, 9.0, n, dtype=float)
    coul_valid = np.linspace(-2.0, -0.5, n, dtype=float)

    payload = build_abfe_free_energy_numeric_payload(
        frames_valid=frames_valid,
        bias_valid=np.linspace(0.1, 2.5, n, dtype=float),
        total_valid=total_valid,
        distance_valid=distance_valid,
        coul_valid=coul_valid,
        bound_valid_mask=np.asarray(distance_valid <= 4.5, dtype=bool),
        unbound_valid_mask=np.asarray(distance_valid >= 7.5, dtype=bool),
        bound_cutoff_a=4.5,
        n_bound=12,
        n_dist_valid=30,
        use_count_fraction=True,
        temperature_k=300.0,
        v_eff_nm3=1.2,
        bootstrap_seed=1234,
        n_boot=64,
    )
    assert payload["ok"] is True
    assert int(payload["n_samples"]) == n
    assert np.isfinite(float(payload["dg_exp"]))
    assert np.isfinite(float(payload["dg_cum"]))
    assert np.isfinite(float(payload["coul_mean"]))
    assert np.isclose(float(payload["contact_frac_mean"]), 12.0 / 30.0)
    assert np.isfinite(float(payload["dg_reweighted_kj"]))
    assert np.isfinite(float(payload["kd_app_reweighted_m"]))
    assert np.isfinite(float(payload["reweight_ess_total"]))
    assert str(payload["selected_estimator"]) in {"bias_reweighted_hold_windows", "gated_off"}
    assert float(payload["ci_low"]) <= float(payload["ci_high"])
    assert len(np.asarray(payload["run_frames"], dtype=int)) == (n - 2)
    assert len(np.asarray(payload["run_exp"], dtype=float)) == (n - 2)
    assert len(np.asarray(payload["run_cum"], dtype=float)) == (n - 2)
    assert len(np.asarray(payload["e_cent"], dtype=float)) == len(np.asarray(payload["e_pmf"], dtype=float))
    assert len(np.asarray(payload["finite_min"], dtype=bool)) == n
    assert bool(payload["has_surface"]) is True
    assert payload["Xc"] is not None and payload["Yc"] is not None and payload["Fsurf"] is not None
    assert np.isfinite(float(payload["surface_clip_max_kj"]))
    assert np.isfinite(float(dict(payload["bound_center"]).get("x", np.nan)))
    assert np.isfinite(float(dict(payload["unbound_center"]).get("x", np.nan)))

    payload_contact_fallback = build_abfe_free_energy_numeric_payload(
        frames_valid=frames_valid,
        total_valid=total_valid,
        distance_valid=distance_valid,
        coul_valid=[],
        bound_cutoff_a=4.5,
        n_bound=0,
        n_dist_valid=0,
        use_count_fraction=False,
        temperature_k=300.0,
        bootstrap_seed=1234,
        n_boot=32,
    )
    assert payload_contact_fallback["ok"] is True
    expected_contact = float(np.mean(distance_valid <= 4.5))
    assert np.isclose(float(payload_contact_fallback["contact_frac_mean"]), expected_contact)

    payload_nonfinite = build_abfe_free_energy_numeric_payload(
        frames_valid=np.arange(1, 7, dtype=int),
        total_valid=np.asarray([-10.0, -9.0, np.nan, -4.0, -3.0, -2.0], dtype=float),
        bias_valid=np.asarray([0.2, 0.3, 0.4, np.nan, 1.8, 2.0], dtype=float),
        distance_valid=np.asarray([3.0, 3.2, 4.0, 8.0, 8.2, 8.4], dtype=float),
        coul_valid=np.asarray([-1.0, -0.9, -0.8, -0.2, -0.1, -0.1], dtype=float),
        bound_valid_mask=np.asarray([True, True, True, False, False, False], dtype=bool),
        unbound_valid_mask=np.asarray([False, False, False, True, True, True], dtype=bool),
        bound_cutoff_a=4.5,
        n_bound=2,
        n_dist_valid=5,
        temperature_k=300.0,
        v_eff_nm3=1.2,
        bootstrap_seed=1234,
        n_boot=16,
    )
    assert payload_nonfinite["ok"] is True
    assert int(payload_nonfinite["n_samples"]) == 5
    assert np.all(np.isfinite(np.asarray(payload_nonfinite["run_exp"], dtype=float)))


def test_abfe_runtime_free_energy_numeric_payload_prefers_rescored_hold_estimate():
    n = 24
    frames_valid = np.arange(1, n + 1, dtype=int)
    total_valid = np.linspace(-12.0, -4.0, n, dtype=float)
    rescored = np.asarray(
        [-16.0, -8.0, -7.5, -7.0, -6.8, -6.5, -6.4, -6.2, -6.1, -6.0, -5.9, -5.8]
        + [0.1, 0.2, 0.3, 0.2, 0.4, 0.3, 0.5, 0.4, 0.6, 0.5, 0.7, 0.6],
        dtype=float,
    )
    ligand_internal = np.asarray(
        [2.6, 2.2, 2.1, 2.0, 2.0, 1.9, 1.8, 1.8, 1.7, 1.7, 1.6, 1.6]
        + [0.5, 0.4, 0.4, 0.4, 0.5, 0.4, 0.5, 0.5, 0.6, 0.5, 0.6, 0.6],
        dtype=float,
    )
    distance_valid = np.concatenate([np.linspace(3.2, 4.4, 12), np.linspace(8.0, 10.0, 12)]).astype(float)
    bound_mask = np.asarray([True] * 12 + [False] * 12, dtype=bool)
    unbound_mask = np.asarray([False] * 12 + [True] * 12, dtype=bool)
    payload = build_abfe_free_energy_numeric_payload(
        frames_valid=frames_valid,
        bias_valid=np.linspace(0.1, 1.6, n, dtype=float),
        total_valid=total_valid,
        rescored_total_valid=rescored,
        ligand_internal_valid=ligand_internal,
        rescored_energy_label="gbsa_obc2",
        distance_valid=distance_valid,
        coul_valid=np.linspace(-1.5, -0.2, n, dtype=float),
        bound_valid_mask=bound_mask,
        unbound_valid_mask=unbound_mask,
        bound_cutoff_a=4.5,
        n_bound=12,
        n_dist_valid=24,
        n_bound_raw=12,
        n_unbound_raw=12,
        n_bound_hold=12,
        n_unbound_hold=12,
        estimator_window_mode="hold_only",
        use_count_fraction=True,
        temperature_k=300.0,
        v_eff_nm3=1.5,
        bootstrap_seed=321,
        n_boot=32,
    )
    assert payload["ok"] is True
    assert bool(payload["rescored_energy_available"]) is True
    assert np.isfinite(float(payload["dg_rescored_state_expavg_kj"]))
    assert bool(payload["ligand_internal_available"]) is True
    assert np.isfinite(float(payload["dg_rescored_state_blockexpavg_strain_kj"]))
    assert float(payload["dg_rescored_state_blockexpavg_strain_kj"]) > float(payload["dg_rescored_state_expavg_strain_kj"])
    assert str(payload["selected_estimator"]) in {
        "gbsa_obc2_hold_window_expavg",
        "gbsa_obc2_hold_window_expavg_strain_consensus",
        "gbsa_obc2_hold_window_blockexpavg",
        "gbsa_obc2_hold_window_expavg_strain",
    }
    assert np.isfinite(float(payload["dg_selected_kj"]))


def test_abfe_runtime_gbsa_mmgbsa_rescore_payload_smoke():
    coords_nm = np.asarray(
        [
            [
                [0.00, 0.00, 0.00],
                [0.22, 0.00, 0.00],
            ],
            [
                [0.00, 0.00, 0.00],
                [0.65, 0.00, 0.00],
            ],
        ],
        dtype=float,
    )
    payload = build_abfe_gbsa_mmgbsa_rescore_payload(
        coords_nm_series=coords_nm,
        atom_rows=[
            {"element": "C"},
            {"element": "N"},
        ],
        charges_e=np.asarray([0.0, 1.0], dtype=float),
        protein_atom_idx=np.asarray([0], dtype=int),
        ligand_atom_idx=np.asarray([1], dtype=int),
        raw_interaction_energy_kj=np.asarray([-4.0, -0.5], dtype=float),
        frame_mask=np.asarray([True, True], dtype=bool),
    )
    assert payload["ok"] is True
    rescored = np.asarray(payload["rescored_total_kj"], dtype=float)
    assert rescored.shape == (2,)
    assert np.all(np.isfinite(rescored))
    assert np.isfinite(float(np.asarray(payload["gbsa_delta_solv_kj"], dtype=float)[0]))

    coords_a_list = [np.asarray(frame, dtype=np.float32) * 10.0 for frame in list(coords_nm) * 3]
    capped = build_abfe_gbsa_mmgbsa_rescore_payload(
        coords_nm_series=coords_a_list,
        coords_scale=0.1,
        atom_rows=[
            {"element": "C"},
            {"element": "N"},
        ],
        charges_e=np.asarray([0.0, 1.0], dtype=float),
        protein_atom_idx=np.asarray([0], dtype=int),
        ligand_atom_idx=np.asarray([1], dtype=int),
        raw_interaction_energy_kj=np.asarray([-4.0, -0.5, -4.0, -0.5, -4.0, -0.5], dtype=float),
        frame_mask=np.asarray([True, True, True, True, True, True], dtype=bool),
        original_frame_indices=np.asarray([0, 1, 2, 3, 4, 5], dtype=int),
        max_rescore_frames=2,
        memory_budget_mb=8.0,
    )
    assert capped["ok"] is True
    assert int(capped["n_input_frames"]) == 6
    assert int(capped["n_frames"]) == 2
    assert list(np.asarray(capped["selected_original_frame_indices"], dtype=int)) == [0, 5]


def test_abfe_runtime_local_pocket_restraint_plan_softens_binding_site_sidechains():
    coords_nm = np.asarray(
        [
            [0.00, 0.00, 0.00],  # ligand
            [0.32, 0.00, 0.00],  # near sidechain
            [0.34, 0.05, 0.00],  # near backbone
            [1.45, 0.00, 0.00],  # far sidechain
        ],
        dtype=float,
    )
    atom_rows = [
        {"name": "C1", "element": "C", "chain": "L", "resi": "1", "resn": "LIG"},
        {"name": "CB", "element": "C", "chain": "A", "resi": "10", "resn": "PHE"},
        {"name": "CA", "element": "C", "chain": "A", "resi": "10", "resn": "PHE"},
        {"name": "CG", "element": "C", "chain": "A", "resi": "40", "resn": "LEU"},
    ]
    payload = build_abfe_local_pocket_restraint_plan_payload(
        atom_rows=atom_rows,
        coords_nm=coords_nm,
        protein_atom_idx=np.asarray([1, 2, 3], dtype=int),
        ligand_atom_idx=np.asarray([0], dtype=int),
        ligand_heavy_idx=np.asarray([0], dtype=int),
        pocket_desc={"pocket_atom_density_per_a3": 0.055, "pocket_frac_hydrophobic": 0.7},
    )
    assert payload["ok"] is True
    rows = {int(row["index"]): dict(row) for row in list(payload["rows"])}
    assert float(rows[1]["k_rest_kj_nm2"]) < float(rows[3]["k_rest_kj_nm2"])
    assert float(rows[2]["k_rest_kj_nm2"]) > float(rows[1]["k_rest_kj_nm2"])
    assert str(rows[1]["shell"]) == "inner"
    assert int(payload["n_sidechain_softened"]) >= 1


def test_abfe_runtime_replica_aggregate_payload():
    rep_a = {
        "replica_index": 1,
        "random_seed": 111,
        "total_valid": np.linspace(-42.0, -12.0, 30, dtype=float),
        "distance_valid": np.linspace(3.0, 10.0, 30, dtype=float),
        "coul_valid": np.linspace(-2.5, -0.6, 30, dtype=float),
        "bias_valid": np.linspace(0.2, 2.1, 30, dtype=float),
        "bound_valid_mask": np.asarray(np.linspace(3.0, 10.0, 30, dtype=float) <= 4.5, dtype=bool),
        "unbound_valid_mask": np.asarray(np.linspace(3.0, 10.0, 30, dtype=float) >= 7.5, dtype=bool),
        "bound_cutoff_a": 4.5,
        "n_bound": 12,
        "n_unbound": 10,
        "n_dist_valid": 30,
        "dg_pop_kj": -1.5,
        "p_bound": 0.40,
        "p_unbound": 0.33,
        "v_eff_nm3": 1.3,
        "sim_ns": 0.5,
        "sample_every_ps": 2.0,
    }
    rep_b = {
        "replica_index": 2,
        "random_seed": 112,
        "total_valid": np.linspace(-39.0, -11.0, 30, dtype=float),
        "distance_valid": np.linspace(3.5, 10.5, 30, dtype=float),
        "coul_valid": np.linspace(-2.3, -0.4, 30, dtype=float),
        "bias_valid": np.linspace(0.3, 2.2, 30, dtype=float),
        "bound_valid_mask": np.asarray(np.linspace(3.5, 10.5, 30, dtype=float) <= 4.5, dtype=bool),
        "unbound_valid_mask": np.asarray(np.linspace(3.5, 10.5, 30, dtype=float) >= 7.5, dtype=bool),
        "bound_cutoff_a": 4.5,
        "n_bound": 10,
        "n_unbound": 12,
        "n_dist_valid": 30,
        "dg_pop_kj": -1.1,
        "p_bound": 0.33,
        "p_unbound": 0.40,
        "v_eff_nm3": 1.3,
        "sim_ns": 0.5,
        "sample_every_ps": 2.0,
    }
    payload = build_abfe_replica_aggregate_payload(
        replica_results=[rep_a, rep_b],
        requested_replica_count=3,
        temperature_k=300.0,
        bootstrap_seed=123,
        n_boot=64,
    )
    assert payload["ok"] is True
    assert int(payload["replica_count"]) == 2
    assert int(payload["requested_replica_count"]) == 3
    summary = dict(payload["summary"])
    assert int(summary["failed_replica_count"]) == 1
    assert np.isfinite(float(summary["dg_exp_mean"]))
    assert np.isfinite(float(summary["dg_exp_ci_low"]))
    assert np.isfinite(float(summary["dg_exp_ci_high"]))
    assert int(summary["n_samples_total"]) == 60
    assert int(summary["n_dist_valid_total"]) == 60
    assert int(summary["n_bound_total"]) == 22
    assert int(summary["n_unbound_total"]) == 22
    assert float(summary["sim_ns_total"]) == 1.0
    assert np.isfinite(float(summary["dg_reweighted_kj"]))
    assert np.isfinite(float(summary["reweight_ess_total"]))
    assert "selected_estimator" in summary
    assert len(np.asarray(payload["run_frames_common"], dtype=int)) == 28
    assert len(np.asarray(payload["run_exp_sem"], dtype=float)) == 28


def test_abfe_runtime_replica_aggregate_prefers_replica_selected_estimates():
    rep_a = {
        "replica_index": 1,
        "random_seed": 301,
        "total_valid": np.asarray([-12.0] * 8 + [-3.0] * 8, dtype=float),
        "distance_valid": np.asarray([3.0] * 8 + [8.0] * 8, dtype=float),
        "coul_valid": np.asarray([-1.5] * 8 + [-0.2] * 8, dtype=float),
        "bias_valid": np.asarray([0.6] * 16, dtype=float),
        "bound_valid_mask": np.asarray([True] * 8 + [False] * 8, dtype=bool),
        "unbound_valid_mask": np.asarray([False] * 8 + [True] * 8, dtype=bool),
        "bound_cutoff_a": 4.5,
        "n_bound": 8,
        "n_unbound": 8,
        "n_dist_valid": 16,
        "n_bound_raw": 10,
        "n_unbound_raw": 10,
        "n_bound_hold": 8,
        "n_unbound_hold": 8,
        "estimator_window_mode": "hold_only",
        "dg_pop_kj": 0.0,
        "p_bound": 0.5,
        "p_unbound": 0.5,
        "v_eff_nm3": 1.2,
        "sim_ns": 0.25,
        "sample_every_ps": 2.0,
    }
    rep_b = {
        "replica_index": 2,
        "random_seed": 302,
        "total_valid": np.asarray([-11.0] * 8 + [-2.5] * 8, dtype=float),
        "distance_valid": np.asarray([3.2] * 8 + [8.4] * 8, dtype=float),
        "coul_valid": np.asarray([-1.4] * 8 + [-0.1] * 8, dtype=float),
        "bias_valid": np.asarray([0.5] * 16, dtype=float),
        "bound_valid_mask": np.asarray([True] * 8 + [False] * 8, dtype=bool),
        "unbound_valid_mask": np.asarray([False] * 8 + [True] * 8, dtype=bool),
        "bound_cutoff_a": 4.5,
        "n_bound": 8,
        "n_unbound": 8,
        "n_dist_valid": 16,
        "n_bound_raw": 10,
        "n_unbound_raw": 10,
        "n_bound_hold": 8,
        "n_unbound_hold": 8,
        "estimator_window_mode": "hold_only",
        "dg_pop_kj": 0.0,
        "p_bound": 0.5,
        "p_unbound": 0.5,
        "v_eff_nm3": 1.2,
        "sim_ns": 0.25,
        "sample_every_ps": 2.0,
    }
    payload = build_abfe_replica_aggregate_payload(
        replica_results=[rep_a, rep_b],
        requested_replica_count=2,
        temperature_k=300.0,
        bootstrap_seed=456,
        n_boot=64,
    )
    assert payload["ok"] is True
    summary = dict(payload["summary"])
    assert int(summary["selected_replica_count"]) == 2
    assert bool(summary["reweight_confidence_ok"]) is True
    assert str(summary["selected_estimator"]) == "replica_aggregate_bias_reweighted_hold_windows"
    assert np.isfinite(float(summary["dg_selected_kj"]))
    assert np.isfinite(float(summary["kd_selected_m"]))


def test_abfe_runtime_hold_window_selection_prefers_stable_windows():
    payload = build_abfe_post_sampling_metrics(
        frames=list(range(1, 13)),
        times_ps=[float(i) for i in range(1, 13)],
        lam_hist=[0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        bias_e=[0.1, 0.2, 0.2, 0.2, 0.8, 1.2, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8],
        lj_e=[-5.0] * 12,
        coul_e=[-1.0] * 12,
        total_e=[-6.0] * 12,
        min_dist=[3.0, 3.0, 3.1, 3.1, 3.4, 3.8, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2],
        reactive_dist=[np.nan] * 12,
        site_min_dist=[2.5, 2.6, 2.6, 2.7, 3.4, 4.0, 5.5, 5.8, 6.0, 6.2, 6.4, 6.6],
        site_com_dist=[3.0, 3.1, 3.2, 3.3, 5.0, 8.0, 10.5, 10.8, 11.0, 11.2, 11.4, 11.6],
        protein_contact_count_4a=[10, 10, 9, 9, 6, 4, 2, 2, 2, 2, 1, 1],
        lig_com=[[0.0, 0.0, 0.0]] * 12,
        far_hold_stable_mask=[False, False, False, False, False, False, True, True, True, True, True, True],
        temperature_k=300.0,
        local_cutoff_nm=1.2,
    )
    assert str(payload["estimator_window_mode"]) == "hold_only"
    assert int(payload["n_bound_hold"]) >= 4
    assert int(payload["n_unbound_hold"]) >= 4


def test_abfe_runtime_unbound_hold_tolerates_short_sample_gaps():
    far_hold = [False] * 8 + [True, False, False, False] * 4
    payload = build_abfe_post_sampling_metrics(
        frames=list(range(1, 25)),
        times_ps=[float(i) for i in range(1, 25)],
        lam_hist=[0.0] * 4 + [0.4] * 4 + [1.0] * 16,
        bias_e=[0.2] * 24,
        lj_e=[-5.0] * 24,
        coul_e=[-1.0] * 24,
        total_e=[-6.0] * 24,
        min_dist=[3.0] * 4 + [4.0] * 4 + [12.0] * 16,
        reactive_dist=[np.nan] * 24,
        site_min_dist=[2.5] * 4 + [4.0] * 4 + [6.0] * 16,
        site_com_dist=[3.0] * 4 + [8.0] * 4 + [22.0] * 16,
        protein_contact_count_4a=[10] * 4 + [5] * 4 + [0] * 16,
        lig_com=[[0.0, 0.0, 0.0]] * 24,
        far_hold_stable_mask=far_hold,
        temperature_k=300.0,
        local_cutoff_nm=1.2,
    )
    assert payload["ok"] is True
    assert int(payload["n_bound_hold"]) >= 4
    assert int(payload["n_unbound_hold"]) == 4
    assert int(np.sum(np.asarray(payload["unbound_hold_mask"], dtype=bool))) == 4
    assert str(payload["estimator_window_mode"]) == "hold_only"


def test_abfe_runtime_unbound_hold_accepts_sparse_detached_far_hold_frames():
    far_hold = [False] * 8 + [True, False, False, False, False] * 4
    payload = build_abfe_post_sampling_metrics(
        frames=list(range(1, 29)),
        times_ps=[float(i) for i in range(1, 29)],
        lam_hist=[0.0] * 8 + [1.0] * 20,
        bias_e=[0.2] * 28,
        lj_e=[-5.0] * 28,
        coul_e=[-1.0] * 28,
        total_e=[-6.0] * 28,
        min_dist=[3.0] * 8 + [12.0] * 20,
        reactive_dist=[np.nan] * 28,
        site_min_dist=[2.5] * 8 + [6.0] * 20,
        site_com_dist=[3.0] * 8 + [22.0] * 20,
        protein_contact_count_4a=[10] * 8 + [0] * 20,
        lig_com=[[0.0, 0.0, 0.0]] * 28,
        far_hold_stable_mask=far_hold,
        temperature_k=300.0,
        local_cutoff_nm=1.2,
    )
    assert payload["ok"] is True
    assert bool(payload["unbound_hold_sparse_fallback_used"]) is True
    assert int(payload["n_bound_hold"]) >= 4
    assert int(payload["n_unbound_hold"]) == 4
    assert int(np.sum(np.asarray(payload["unbound_hold_mask"], dtype=bool))) == 4
    assert str(payload["estimator_window_mode"]) == "hold_only"


def test_abfe_runtime_unbound_hold_prefers_detached_settle_low_bias_frames():
    payload = build_abfe_post_sampling_metrics(
        frames=list(range(1, 13)),
        times_ps=[float(i) for i in range(1, 13)],
        lam_hist=[0.0] * 4 + [0.6] * 4 + [0.05] * 4,
        bias_e=[0.2] * 12,
        lj_e=[-5.0] * 12,
        coul_e=[-1.0] * 12,
        total_e=[-6.0] * 12,
        min_dist=[3.0] * 4 + [4.0] * 4 + [12.0] * 4,
        reactive_dist=[np.nan] * 12,
        site_min_dist=[2.5] * 4 + [4.0] * 4 + [6.0] * 4,
        site_com_dist=[3.0] * 4 + [8.0] * 4 + [22.0] * 4,
        protein_contact_count_4a=[10] * 4 + [5] * 4 + [0] * 4,
        lig_com=[[0.0, 0.0, 0.0]] * 12,
        far_hold_stable_mask=[False] * 12,
        detached_settle_mask=[False] * 8 + [True] * 4,
        detached_settle_max_lambda=0.10,
        temperature_k=300.0,
        local_cutoff_nm=1.2,
    )
    assert payload["ok"] is True
    assert bool(payload["detached_settle_used_for_unbound_hold"]) is True
    assert int(payload["n_bound_hold"]) >= 4
    assert int(payload["n_unbound_hold"]) == 4
    assert int(np.sum(np.asarray(payload["unbound_hold_mask"], dtype=bool))) == 4
    assert str(payload["estimator_window_mode"]) == "hold_only"


def test_abfe_runtime_unbound_hold_accepts_sparse_detached_settle_frames():
    payload = build_abfe_post_sampling_metrics(
        frames=list(range(1, 29)),
        times_ps=[float(i) for i in range(1, 29)],
        lam_hist=[0.0] * 4 + [0.6] * 4 + ([0.05, 0.30, 0.30, 0.30, 0.30] * 4),
        bias_e=[0.2] * 28,
        lj_e=[-5.0] * 28,
        coul_e=[-1.0] * 28,
        total_e=[-6.0] * 28,
        min_dist=[3.0] * 4 + [4.0] * 4 + [12.0] * 20,
        reactive_dist=[np.nan] * 28,
        site_min_dist=[2.5] * 4 + [4.0] * 4 + [6.0] * 20,
        site_com_dist=[3.0] * 4 + [8.0] * 4 + [22.0] * 20,
        protein_contact_count_4a=[10] * 4 + [5] * 4 + [0] * 20,
        lig_com=[[0.0, 0.0, 0.0]] * 28,
        far_hold_stable_mask=[False] * 28,
        detached_settle_mask=[False] * 8 + ([True, False, False, False, False] * 4),
        detached_settle_max_lambda=0.10,
        temperature_k=300.0,
        local_cutoff_nm=1.2,
    )
    assert payload["ok"] is True
    assert bool(payload["detached_settle_used_for_unbound_hold"]) is True
    assert bool(payload["detached_settle_sparse_fallback_used"]) is True
    assert int(payload["n_unbound_hold"]) == 4
    assert int(np.sum(np.asarray(payload["unbound_hold_mask"], dtype=bool))) == 4
    assert str(payload["estimator_window_mode"]) == "hold_only"


def test_abfe_runtime_bias_reweighted_population_payload():
    payload = build_abfe_bias_reweighted_population_payload(
        bound_valid_mask=np.asarray([True, True, False, False], dtype=bool),
        unbound_valid_mask=np.asarray([False, False, True, True], dtype=bool),
        bias_valid=np.asarray([0.2, 0.4, 1.8, 2.0], dtype=float),
        temperature_k=300.0,
        v_eff_nm3=1.1,
        bootstrap_seed=123,
        n_boot=32,
    )
    assert payload["ok"] is True
    assert np.isfinite(float(payload["dg_reweighted_kj"]))
    assert float(payload["p_bound_reweighted"]) > 0.0
    assert float(payload["p_unbound_reweighted"]) > 0.0
    assert np.isfinite(float(payload["ess_total"]))
    assert np.asarray(payload["weights"], dtype=float).shape == (4,)

    payload_nan_bias = build_abfe_bias_reweighted_population_payload(
        bound_valid_mask=np.asarray([True, True, False, False], dtype=bool),
        unbound_valid_mask=np.asarray([False, False, True, True], dtype=bool),
        bias_valid=np.asarray([0.2, np.nan, 1.8, 2.0], dtype=float),
        temperature_k=300.0,
        v_eff_nm3=1.1,
        bootstrap_seed=123,
        n_boot=0,
    )
    assert payload_nan_bias["ok"] is True
    weights_nan_bias = np.asarray(payload_nan_bias["weights"], dtype=float)
    assert weights_nan_bias.shape == (4,)
    assert float(weights_nan_bias[1]) == 0.0
    assert np.asarray(payload_nan_bias["weights_finite"], dtype=float).shape == (3,)


def test_abfe_runtime_free_energy_plot_payload():
    payload = build_abfe_free_energy_plot_payload(
        run_frames=np.array([3, 4, 5], dtype=int),
        run_exp=np.array([-10.0, -9.5, -9.0], dtype=float),
        run_cum=np.array([-9.8, -9.4, -8.9], dtype=float),
        dg_exp=-9.0,
        ci_low=-10.5,
        ci_high=-8.5,
        e_cent=np.array([-20.0, -10.0], dtype=float),
        e_pmf=np.array([0.0, 1.2], dtype=float),
        r_cent=np.array([3.0, 4.0], dtype=float),
        r_pmf=np.array([0.0, 0.8], dtype=float),
        dist_axis_label="Min Distance to Initial Pocket Anchor (Å)",
        has_surface=True,
        Xc=np.array([3.0, 4.0], dtype=float),
        Yc=np.array([-20.0, -10.0], dtype=float),
        Fsurf=np.array([[0.0, 0.5], [1.0, 1.5]], dtype=float),
        surface_clip_max_kj=1.4,
        bound_cutoff_a=4.5,
        bound_center={"x": 3.2, "y": -18.0, "n": 8},
        unbound_center={"x": 4.1, "y": -11.0, "n": 7},
        replica_count=3,
        replica_run_exp_sem=np.array([0.3, 0.2, 0.1], dtype=float),
        replica_run_cum_sem=np.array([0.4, 0.3, 0.2], dtype=float),
    )
    assert payload["ok"] is True
    conv = dict(payload["convergence"])
    assert str(conv["title"]) == "Free-Energy Convergence"
    assert np.allclose(np.asarray(conv["x_exp"], dtype=float), [3.0, 4.0, 5.0])
    assert np.allclose(np.asarray(conv["y_exp"], dtype=float), [-10.0, -9.5, -9.0])
    assert str(dict(conv["line_exp"])["label"]) == "ΔG exp-avg"
    rep_overlay = dict(conv["replica_overlay"])
    assert bool(rep_overlay["enabled"]) is True
    assert int(rep_overlay["replica_count"]) == 3
    theme = dict(payload["theme"])
    assert str(theme["figure_facecolor"]) == "#F7F8FA"
    assert str(theme["axes_facecolor"]) == "#FFFFFF"
    epmf = dict(payload["energy_pmf"])
    assert str(epmf["xlabel"]) == "Interaction Energy (kJ/mol)"
    assert np.isclose(float(epmf["fill_alpha"]), 0.12)
    rpmf = dict(payload["distance_pmf"])
    assert bool(rpmf["has_data"]) is True
    assert str(rpmf["xlabel"]) == "Min Distance to Initial Pocket Anchor (Å)"
    assert str(dict(rpmf["bound_cutoff_line"])["label"]) == "Bound cutoff"
    surf = dict(payload["surface"])
    assert bool(surf["has_data"]) is True
    assert str(surf["cmap"]) == "viridis"     # perceptual; unsampled bins get their own grey
    assert np.asarray(surf["Xc"], dtype=float).shape == (2,)
    assert np.asarray(surf["Fsurf"], dtype=float).shape == (2, 2)
    assert str(dict(surf["bound_marker"])["label"]) == "bound basin"
    assert str(dict(surf["unbound_marker"])["label"]) == "unbound basin"

    dense_payload = build_abfe_free_energy_plot_payload(
        run_frames=np.arange(2000, dtype=int),
        run_exp=np.linspace(-20.0, -10.0, 2000),
        run_cum=np.linspace(-19.0, -9.0, 2000),
        max_convergence_points=200,
    )
    dense_conv = dict(dense_payload["convergence"])
    assert int(dense_conv["n_raw"]) == 2000
    assert np.asarray(dense_conv["x_exp"], dtype=int).size <= 200
    assert int(np.asarray(dense_conv["x_exp"], dtype=int)[-1]) == 1999

    payload_no_surface = build_abfe_free_energy_plot_payload(
        run_frames=np.array([], dtype=int),
        run_exp=np.array([], dtype=float),
        run_cum=np.array([], dtype=float),
        dg_exp=np.nan,
        ci_low=np.nan,
        ci_high=np.nan,
        e_cent=np.array([], dtype=float),
        e_pmf=np.array([], dtype=float),
        r_cent=np.array([], dtype=float),
        r_pmf=np.array([], dtype=float),
        dist_axis_label="Distance (Å)",
        has_surface=False,
        Xc=None,
        Yc=None,
        Fsurf=None,
    )
    rpmf2 = dict(payload_no_surface["distance_pmf"])
    surf2 = dict(payload_no_surface["surface"])
    assert bool(rpmf2["has_data"]) is False
    assert bool(surf2["has_data"]) is False
    assert str(rpmf2["no_data_text"]) == "No valid distance samples"
    assert str(surf2["no_data_text"]) == "Insufficient samples for 2D surface"


def test_abfe_runtime_post_interaction_followup_plan():
    ready = build_abfe_post_interaction_followup_plan(
        protocol="abfe_lite_local_md",
        abfe_traj_object="abfe_local_md_traj_123",
        object_exists=True,
        n_states=601,
        max_target_states=300,
        top_n=12,
    )
    assert ready["ok"] is True
    assert bool(ready["should_run"]) is True
    assert str(ready["reason"]) == "ready"
    assert int(ready["n_states"]) == 601
    assert int(ready["step"]) == 3
    assert int(ready["top_n"]) == 12

    bad_protocol = build_abfe_post_interaction_followup_plan(
        protocol="trajectory_energy_rescoring",
        abfe_traj_object="abfe_local_md_traj_123",
        object_exists=True,
        n_states=100,
    )
    assert bool(bad_protocol["should_run"]) is False
    assert str(bad_protocol["reason"]) == "protocol_not_local_md"

    missing_object = build_abfe_post_interaction_followup_plan(
        protocol="abfe_lite_local_md",
        abfe_traj_object="",
        object_exists=False,
        n_states=0,
    )
    assert bool(missing_object["should_run"]) is False
    assert str(missing_object["reason"]) == "missing_abfe_object"

    not_loaded = build_abfe_post_interaction_followup_plan(
        protocol="abfe_lite_local_md",
        abfe_traj_object="abfe_local_md_traj_123",
        object_exists=False,
        n_states=100,
    )
    assert bool(not_loaded["should_run"]) is False
    assert str(not_loaded["reason"]) == "abfe_object_not_loaded"
    assert int(not_loaded["step"]) == 1


def test_abfe_runtime_post_interaction_followup_runtime_plan():
    ready = prepare_abfe_post_interaction_followup_runtime_plan(
        protocol="abfe_lite_local_md",
        result_payload={"abfe_traj_object": "abfe_local_md_traj_123"},
        get_names_fn=lambda mode: ["obj_a", "abfe_local_md_traj_123"] if mode == "objects" else [],
        count_states_fn=lambda obj: 601 if obj == "abfe_local_md_traj_123" else 0,
        max_target_states=300,
        top_n=12,
    )
    assert ready["ok"] is True
    assert bool(ready["should_run"]) is True
    assert str(ready["reason"]) == "ready"
    assert int(ready["n_states"]) == 601
    assert int(ready["step"]) == 3
    assert int(ready["top_n"]) == 12

    missing = prepare_abfe_post_interaction_followup_runtime_plan(
        protocol="abfe_lite_local_md",
        result_payload={"abfe_traj_object": "abfe_local_md_traj_123"},
        get_names_fn=lambda _mode: ["obj_a", "obj_b"],
        count_states_fn=lambda _obj: 500,
    )
    assert bool(missing["should_run"]) is False
    assert str(missing["reason"]) == "abfe_object_not_loaded"
    assert int(missing["n_states"]) == 0

    def _count_fail(_obj):
        raise RuntimeError("count failed")

    count_fail = prepare_abfe_post_interaction_followup_runtime_plan(
        protocol="abfe_lite_local_md",
        result_payload={"abfe_traj_object": "abfe_local_md_traj_123"},
        get_names_fn=lambda _mode: ["abfe_local_md_traj_123"],
        count_states_fn=_count_fail,
    )
    assert bool(count_fail["should_run"]) is True
    assert str(count_fail["reason"]) == "ready"
    assert int(count_fail["n_states"]) == 0
    assert int(count_fail["step"]) == 1

    wrong_protocol = prepare_abfe_post_interaction_followup_runtime_plan(
        protocol="endpoint_rescore",
        result_payload={"abfe_traj_object": "abfe_local_md_traj_123"},
        get_names_fn=lambda _mode: ["abfe_local_md_traj_123"],
        count_states_fn=lambda _obj: 100,
    )
    assert bool(wrong_protocol["should_run"]) is False
    assert str(wrong_protocol["reason"]) == "protocol_not_local_md"


def test_abfe_runtime_post_interaction_dialog_payload():
    ready = resolve_abfe_post_interaction_dialog_payload(
        post_result={"summary": "ASP45 (80.0%), TYR99 (50.0%)"},
    )
    assert ready["ok"] is True
    assert bool(ready["show_dialog"]) is True
    assert str(ready["reason"]) == "ready"
    assert str(ready["title"]) == "ABFE Key Interactions"
    assert str(ready["message"]) == "Top binding residues (occupancy): ASP45 (80.0%), TYR99 (50.0%)"

    missing_summary = resolve_abfe_post_interaction_dialog_payload(
        post_result={"summary": "   "},
    )
    assert bool(missing_summary["show_dialog"]) is False
    assert str(missing_summary["reason"]) == "missing_summary"
    assert str(missing_summary["message"]) == ""

    missing_post = resolve_abfe_post_interaction_dialog_payload(post_result=None)
    assert bool(missing_post["show_dialog"]) is False
    assert str(missing_post["reason"]) == "missing_post_result"
    assert str(missing_post["message"]) == ""


def test_abfe_runtime_collect_interaction_presence_sets():
    frame = collect_abfe_interaction_presence_sets(
        interactions=[
            {"prot_res": "ASP45", "type": "H-bond"},
            {"prot_res": "ASP45", "type": "H-bond"},
            {"prot_res": "TYR99", "type": "Pi-Pi"},
            {"prot_res": " ", "type": ""},
            {},
            None,
        ]
    )
    assert frame["ok"] is True
    assert list(frame["residues"]) == ["ASP45", "TYR99"]
    assert list(frame["types"]) == ["H-bond", "Pi-Pi"]


def test_abfe_runtime_post_interaction_report_payload():
    payload = build_abfe_post_interaction_report_payload(
        residue_counts={"ASP45": 8, "TYR99": 5, "LYS12": 1},
        type_counts={"H-bond": 7, "Pi-Pi": 3},
        frames_analyzed=10,
        top_n=2,
        top_types_n=2,
        preview_n=2,
        type_color_map={"H-bond": "#111111"},
    )
    assert payload["ok"] is True
    assert str(payload["reason"]) == "ready"
    assert int(payload["frames_analyzed"]) == 10
    assert list(payload["res_labels"]) == ["ASP45", "TYR99"]
    assert np.allclose(np.asarray(payload["res_occ"], dtype=float), [80.0, 50.0])
    assert list(payload["type_labels"]) == ["H-bond", "Pi-Pi"]
    assert np.allclose(np.asarray(payload["type_occ"], dtype=float), [70.0, 30.0])
    assert list(payload["type_colors"]) == ["#111111", "#7f7f7f"]
    assert str(payload["summary"]) == "ASP45 (80.0%), TYR99 (50.0%)"
    assert len(list(payload["top_residues"])) == 2
    assert str(payload["plot_suptitle"]) == "ABFE Post-Processing (10 sampled frames)"

    no_frames = build_abfe_post_interaction_report_payload(
        residue_counts={"ASP45": 3},
        type_counts={},
        frames_analyzed=0,
    )
    assert no_frames["ok"] is False
    assert str(no_frames["reason"]) == "no_frames_analyzed"

    no_res = build_abfe_post_interaction_report_payload(
        residue_counts={},
        type_counts={"H-bond": 3},
        frames_analyzed=5,
    )
    assert no_res["ok"] is False
    assert str(no_res["reason"]) == "no_residue_counts"


def test_abfe_runtime_pocket_descriptor_compute():
    atom_infos = [
        {"name": "CA", "element": "C", "resn": "PHE", "resi": "10", "chain": "A"},
        {"name": "NZ", "element": "N", "resn": "LYS", "resi": "11", "chain": "A"},
        {"name": "C1", "element": "C", "resn": "LIG", "resi": "1", "chain": "L"},
    ]
    coords_nm = [
        [0.00, 0.00, 0.00],
        [0.20, 0.00, 0.00],
        [0.05, 0.00, 0.00],
    ]
    desc = compute_runtime_pocket_descriptors(
        atom_infos=atom_infos,
        coords_nm=coords_nm,
        protein_ids=[0, 1],
        ligand_ids=[2],
        pocket_cutoff_a=6.0,
        infer_element_fn=lambda name, _sym: ("N" if str(name).upper().startswith("N") else "C"),
    )
    assert float(desc["pocket_n_atoms"]) >= 1.0
    assert float(desc["pocket_n_residues"]) >= 1.0
    assert float(desc["pocket_volume_bbox_a3"]) >= 0.0
    assert float(desc["pocket_frac_positive"]) >= 0.0


def test_alphafold_runtime_helpers(tmp_path):
    assert af_is_relaxed_name("model_relaxed_rank_001.pdb") is True
    assert af_is_relaxed_name("model_unrelaxed_rank_001.pdb") is False
    assert "ranked_0*.pdb" in af_model_patterns(prefer_relaxed=True)
    assert af_plot_title("pae") == "PAE/Error"
    assert any("pae" in p.lower() for p in af_plot_patterns("pae"))

    root = tmp_path / "af"
    root.mkdir(parents=True, exist_ok=True)
    r1 = root / "jobA_rank_001_unrelaxed.pdb"
    r2 = root / "jobA_rank_001_relaxed.pdb"
    r1.write_text("ATOM 1 C\n", encoding="utf-8")
    r2.write_text("ATOM 1 C\n", encoding="utf-8")
    best_relaxed = af_find_best_model_recursive(str(root), token="joba", prefer_relaxed=True)
    assert str(best_relaxed).lower().endswith("joba_rank_001_relaxed.pdb")

    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    p1 = plot_dir / "jobA_pae.png"
    p2 = plot_dir / "other_pae.png"
    p1.write_text("x", encoding="utf-8")
    p2.write_text("x", encoding="utf-8")
    roots = af_collect_existing_dirs([str(plot_dir), str(plot_dir), ""])
    best_plot = af_pick_best_plot_image(
        roots=roots,
        patterns=["*pae*.png"],
        token="joba",
    )
    assert str(best_plot).lower().endswith("joba_pae.png")

    payload = {
        "scores": {
            "predicted_lddt": [0.7] * 30,
            "predicted_aligned_error": [[1.0 for _ in range(12)] for _ in range(12)],
        }
    }
    plddt, pae = af_extract_plddt_pae_from_json(payload)
    assert plddt is not None and int(plddt.size) == 30
    assert pae is not None and pae.shape == (12, 12)


def test_psi4_runtime_driver_plan_helpers():
    assert psi4_normalize_operation("single point") == "energy"
    assert psi4_normalize_operation("bad-op") == "optimize"
    assert psi4_parse_charge_multiplicity_from_block("0 2\nH 0 0 0") == (0, 2)
    assert psi4_reference_for_multiplicity("hf", multiplicity=2) == "uhf"
    assert psi4_reference_for_multiplicity("b3lyp", multiplicity=3) == "uks"
    assert psi4_reference_for_multiplicity("hf", multiplicity=1) == ""

    mb = psi4_normalize_method_basis("hf / sto-3g", "def2-SVP")
    assert mb["method"] == "hf"
    assert mb["basis"] == "sto-3g"
    assert mb["method_call"] == "hf"
    assert mb["embedded_basis"] == "sto-3g"

    assert int(psi4_normalize_memory_gb("2048 MB")) == 2
    assert int(psi4_normalize_memory_gb("0 GB")) == 1
    assert int(psi4_normalize_threads(99, cpu_count=4)) == 4

    ts_opts = psi4_build_driver_options("ts", "sto-3g", scf_type="pk", geom_maxiter=30)
    assert ts_opts["options"]["opt_type"] == "TS"
    assert ts_opts["options"]["full_hess_every"] == 0         # exact Hessian first: a model Hessian has no climb
    assert ts_opts["options"]["geom_maxiter"] == 30
    assert "full_hess_every" not in psi4_build_driver_options("optimize", "sto-3g")["options"]
    assert "irc_direction" not in ts_opts["options"]
    assert psi4_build_driver_options("irc", "sto-3g")["options"]["covalent_connect"] == 1.6

    opts = psi4_build_driver_options(
        "irc",
        "sto-3g",
        scf_type="bad",
        geom_maxiter=5,
        irc_direction="forward",
        irc_points=2,
        irc_step=0.0,
        freeze_atom_indices=[3, "x", 1, 3],
    )
    assert opts["options"]["basis"] == "sto-3g"
    assert opts["options"]["scf_type"] == "df"
    assert opts["options"]["geom_maxiter"] == 20
    assert opts["options"]["opt_type"] == "IRC"
    assert opts["options"]["irc_direction"] == "FORWARD"
    assert opts["options"]["irc_points"] == 4
    assert opts["options"]["irc_step_size"] == 0.01
    assert opts["freeze_atom_indices"] == [1, 3]
    # OptKing's FROZEN_CARTESIAN is a string of "atom xyz" pairs (the old [1, 2, 3, 7, 8, 9] list was rejected
    # by psi4.set_options); the Cartesian indices stay available as freeze_cartesian.
    assert opts["options"]["frozen_cartesian"] == "1 xyz 3 xyz"
    assert opts["freeze_cartesian"] == [1, 2, 3, 7, 8, 9]


def test_psi4_frozen_cartesian_option_is_accepted_by_psi4():
    psi4 = pytest.importorskip("psi4")
    opts = psi4_build_driver_options("optimize", "sto-3g", freeze_atom_indices=[1, 3])
    psi4.core.clean_options()
    try:
        psi4.set_options({"frozen_cartesian": opts["options"]["frozen_cartesian"]})
        assert psi4.core.get_option("OPTKING", "FROZEN_CARTESIAN").split() == ["1", "XYZ", "3", "XYZ"]
    finally:
        psi4.core.clean_options()

    both_opts = psi4_build_driver_options("irc", "sto-3g", irc_direction="BOTH", irc_points=8, irc_step=0.1)
    assert both_opts["irc_direction"] == "BOTH"
    assert both_opts["options"]["irc_direction"] == "FORWARD"

    plan = psi4_prepare_driver_job(
        {
            "operation": "sp",
            "method": "b3lyp/6-31g",
            "basis": "def2-SVP",
            "memory_gb": "3 GB",
            "threads": 8,
            "implicit": True,
            "solvent": "THF",
            "pcm_solver": "bad",
        },
        solvent_candidates_fn=lambda _: ["THF", "Tetrahydrofurane"],
        cpu_count=2,
    )
    assert plan["operation"] == "energy"
    assert plan["method"] == "b3lyp"
    assert plan["basis"] == "6-31g"
    assert plan["threads"] == 2
    assert plan["memory_gb"] == 3
    assert plan["implicit"] is True
    assert plan["solvent"] == "THF"
    assert plan["pcm_solver"] == "IEFPCM"
    assert "Solvent = THF" in plan["pcm_input_block"]

    open_shell = psi4_prepare_driver_job(
        {
            "operation": "energy",
            "method": "hf",
            "basis": "sto-3g",
            "molecule_block": "0 2\nH 0 0 0\nH 0 0 1.0\nH 0 0 2.0",
        },
        cpu_count=1,
    )
    assert open_shell["multiplicity"] == 2
    assert open_shell["reference"] == "uhf"
    assert open_shell["options"]["reference"] == "uhf"


def test_psi4_runtime_cube_helpers(tmp_path):
    cube_in = tmp_path / "density.cube"
    cube_out = tmp_path / "esp.cube"
    cube_copy = tmp_path / "copy.cube"
    cube_in.write_text(
        "\n".join(
            [
                "CUBE TEST",
                "generated",
                "1 0.0 0.0 0.0",
                "2 1.0 0.0 0.0",
                "2 0.0 1.0 0.0",
                "2 0.0 0.0 1.0",
                "6 6.0 0.0 0.0 0.0",
                "0.1 0.2 0.3 0.4 0.5 0.6",
                "0.7 0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    meta = psi4_read_cube(str(cube_in))
    assert int(meta["nat"]) == 1
    assert tuple(meta["data"].shape) == (2, 2, 2)

    psi4_write_cube_data(meta, meta["data"], str(cube_copy))
    meta_copy = psi4_read_cube(str(cube_copy))
    assert tuple(meta_copy["data"].shape) == (2, 2, 2)

    ok, msg = psi4_build_mulliken_esp_cube_from_density(str(cube_in), [0.5], str(cube_out))
    assert ok is True, msg
    assert cube_out.exists()
    meta_out = psi4_read_cube(str(cube_out))
    assert tuple(meta_out["data"].shape) == (2, 2, 2)


def test_psi4_runtime_path_resolvers(tmp_path):
    jdir = tmp_path / "job"
    jdir.mkdir(parents=True, exist_ok=True)

    homo = jdir / "orb_homo.cube"
    lumo = jdir / "orb_lumo.cube"
    dens = jdir / "run_Dt_001.cube"
    esp = jdir / "qm_esp.cube"
    for fp in (homo, lumo, dens, esp):
        fp.write_text("x\n", encoding="utf-8")

    dens_pick = psi4_resolve_density_cube(
        {"density_cube_file": "", "cube_files": [homo.name, lumo.name]},
        str(jdir),
    )
    assert dens_pick == str(dens)

    dens_explicit = psi4_resolve_density_cube({"density_cube_file": dens.name}, str(jdir))
    assert dens_explicit == str(dens)

    esp_existing = psi4_resolve_esp_cube(
        {"esp_cube_file": esp.name},
        str(jdir),
        str(dens),
        has_charges=False,
    )
    assert esp_existing["esp_source"] == "qm_esp_cube"
    assert esp_existing["needs_build"] is False
    assert esp_existing["error"] == ""
    assert esp_existing["esp_cube"] == str(esp)

    esp_missing = psi4_resolve_esp_cube({}, str(jdir), str(dens), has_charges=False)
    assert esp_missing["esp_source"] == "mulliken_grid_fit"
    assert esp_missing["needs_build"] is False
    assert esp_missing["error"] == "missing_charges"
    assert os.path.basename(str(esp_missing["esp_cube"])) == "mulliken_esp_from_charges.cube"

    esp_build = psi4_resolve_esp_cube({}, str(jdir), str(dens), has_charges=True)
    assert esp_build["esp_source"] == "mulliken_grid_fit"
    assert esp_build["needs_build"] is True
    assert esp_build["error"] == ""

    orb_payload = psi4_resolve_homo_lumo_cubes({"cube_files": [homo.name, lumo.name]}, str(jdir))
    assert orb_payload["homo_cube"] == str(homo)
    assert orb_payload["lumo_cube"] == str(lumo)

    orb_explicit = psi4_resolve_homo_lumo_cubes(
        {"homo_cube_file": str(homo), "lumo_cube_file": str(lumo)},
        str(jdir),
    )
    assert orb_explicit["homo_cube"] == str(homo)
    assert orb_explicit["lumo_cube"] == str(lumo)


def test_psi4_runtime_pcm_and_data_loaders(tmp_path):
    assert psi4_pcm_solvent_candidates("DMSO")[:2] == ["DMSO", "DimethylSulfoxide"]
    assert "Water" in psi4_pcm_solvent_candidates("")
    assert "Water" in psi4_pcm_solvent_candidates("MeCN")

    tail_path = tmp_path / "psi4.out"
    tail_path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    assert psi4_tail_file(str(tail_path), max_lines=2) == "c\nd"

    jdir = tmp_path / "job"
    jdir.mkdir(parents=True, exist_ok=True)

    hist_npz = jdir / "history_data.npz"
    np.savez_compressed(
        str(hist_npz),
        coords_ang=np.asarray([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=np.float32),
        energy_hartree=np.asarray([-100.0, -101.0, -102.0], dtype=float),
        atom_symbols=np.asarray(["C"], dtype="U4"),
    )
    hist = psi4_load_history_data({"history_npz": "history_data.npz"}, str(jdir))
    assert tuple(hist["coords_ang"].shape) == (2, 1, 3)
    assert tuple(hist["energy_hartree"].shape) == (2,)
    assert hist["symbols"] == ["C"]

    hist_fb = psi4_load_history_data(
        {"history_energy_hartree": [-10.0, -11.0], "atom_symbols": ["N", "H"]},
        str(jdir),
    )
    assert tuple(hist_fb["energy_hartree"].shape) == (2,)
    assert hist_fb["symbols"] == ["N", "H"]

    vib_npz = jdir / "vib_modes.npz"
    np.savez_compressed(
        str(vib_npz),
        mode_cart_ang=np.asarray([[[0.1, 0.0, 0.0]]], dtype=np.float32),
        eq_geom_ang=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        atom_symbols=np.asarray(["C"], dtype="U4"),
        freq_cm_real=np.asarray([123.4], dtype=float),
        freq_cm_imag=np.asarray([0.0], dtype=float),
        ir_intensity_km_mol=np.asarray([9.9], dtype=float),
        trv=np.asarray(["V"], dtype="U12"),
    )
    vib = psi4_load_vib_data({"vib_modes_npz": "vib_modes.npz"}, str(jdir))
    assert tuple(vib["mode_cart_ang"].shape) == (1, 1, 3)
    assert tuple(vib["eq_geom_ang"].shape) == (1, 3)
    assert tuple(vib["freq_real"].shape) == (1,)
    assert tuple(vib["ir"].shape) == (1,)
    assert vib["symbols"] == ["C"]

    vib_fb = psi4_load_vib_data(
        {
            "freq_cm_real": [50.0, 80.0],
            "freq_cm_imag": [0.0, 0.0],
            "ir_intensity_km_mol": [1.0, 2.0],
            "freq_trv": ["V", "V"],
            "atom_symbols": ["O", "H"],
        },
        str(jdir),
    )
    assert tuple(vib_fb["freq_real"].shape) == (2,)
    assert tuple(vib_fb["ir"].shape) == (2,)
    assert vib_fb["symbols"] == ["O", "H"]


def test_psi4_irc_ts_frequency_validation_requires_one_imaginary_mode():
    one = psi4_validate_irc_ts_frequencies([0.0, 500.0, 1200.0], [850.0, 0.0, 0.0])
    assert one["ok"] is True
    assert one["imaginary_mode_count"] == 1
    assert one["primary_mode_index"] == 0
    assert one["primary_imag_cm1"] == 850.0

    none = psi4_validate_irc_ts_frequencies([100.0, 500.0], [0.0, 0.0])
    assert none["ok"] is False
    assert none["imaginary_mode_count"] == 0

    two = psi4_validate_irc_ts_frequencies([-120.0, 500.0], [0.0, 50.0])
    assert two["ok"] is False
    assert two["imaginary_mode_count"] == 2


def test_psi4_build_irc_profile_uses_mass_weighted_coordinate():
    coords = np.asarray(
        [
            [[-0.2, 0.0, 0.0], [0.8, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=float,
    )
    energies = np.asarray([-1.10, -1.00, -1.12], dtype=float)

    profile = psi4_build_irc_profile_payload(coords, energies, ["H", "H"])

    assert profile["ok"] is True
    assert profile["ts_frame_index"] == 1
    assert profile["branch"] == ["backward", "ts", "forward"]
    assert profile["coordinate_amu05_bohr"][0] < 0.0
    assert profile["coordinate_amu05_bohr"][1] == 0.0
    assert profile["coordinate_amu05_bohr"][2] > 0.0
    step_bohr = 0.2 / 0.529177210903
    expected = float(np.sqrt(2.0 * psi4_atomic_mass_amu("H") * step_bohr * step_bohr))
    assert np.isclose(abs(profile["coordinate_amu05_bohr"][0]), expected)
    assert np.isclose(profile["relative_energy_kcal_mol"][1], 0.12 * 627.5094740631)


def test_psi4_runtime_prepare_vibrational_animation():
    vib = {
        "mode_cart_ang": np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[0.0, 0.3, 0.0], [0.0, 0.0, 0.4]],
            ],
            dtype=float,
        ),
        "eq_geom_ang": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        "symbols": ["C", "H"],
        "freq_real": [120.0, 220.0],
        "freq_imag": [0.0, 0.0],
        "ir": [5.0, 12.0],
        "trv": ["V", "V"],
    }

    payload = psi4_prepare_vibrational_animation(vib, mode_index=None, amplitude_a=0.30, n_frames=12)
    assert payload["ok"] is True
    assert int(payload["mode_index"]) == 1
    assert tuple(np.asarray(payload["coords_ang"]).shape) == (12, 2, 3)
    assert str(payload["obj_prefix"]) == "psi4_vib_m2"
    assert int(payload["n_frames"]) == 12

    payload_clamp = psi4_prepare_vibrational_animation(vib, mode_index=0, amplitude_a=2.5, n_frames=2)
    assert payload_clamp["ok"] is True
    assert int(payload_clamp["n_frames"]) == 8

    vib_zero = dict(vib)
    vib_zero["mode_cart_ang"] = np.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=float)
    bad = psi4_prepare_vibrational_animation(vib_zero, mode_index=0)
    assert bad["ok"] is False
    assert str(bad["error"]) == "zero_mode_vector"


def test_psi4_runtime_prepare_ir_spectrum():
    vib = {
        "freq_real": [100.0, 300.0, 50.0, float("nan")],
        "freq_imag": [0.0, 0.0, 0.01, 0.0],
        "ir": [5.0, 20.0, 15.0, 10.0],
        "trv": ["V", "T", "V", "V"],
    }

    prep = psi4_prepare_ir_spectrum(vib, include_broadened=True, fwhm_cm=18.0)
    assert prep["ok"] is True
    assert tuple(np.asarray(prep["mode_indices"]).shape) == (1,)
    assert int(np.asarray(prep["mode_indices"])[0]) == 0
    assert int(prep["first_mode_index"]) == 0
    assert tuple(np.asarray(prep["x_grid"]).shape) == (2400,)
    assert tuple(np.asarray(prep["y_norm"]).shape) == (2400,)
    assert float(np.max(np.asarray(prep["y_norm"]))) <= 100.0 + 1e-6

    prep_stick = psi4_prepare_ir_spectrum(vib, include_broadened=False, fwhm_cm=18.0)
    assert prep_stick["ok"] is True
    assert tuple(np.asarray(prep_stick["x_grid"]).shape) == (0,)
    assert tuple(np.asarray(prep_stick["y_norm"]).shape) == (0,)

    bad = psi4_prepare_ir_spectrum(
        {"freq_real": [2.0, 4.0], "freq_imag": [0.0, 0.0], "ir": [1.0, 2.0], "trv": ["V", "V"]},
        include_broadened=True,
    )
    assert bad["ok"] is False
    assert str(bad["error"]) == "no_valid_modes"


def test_psi4_runtime_atom_and_spin_helpers(tmp_path):
    assert psi4_element_token("cl") == "Cl"
    assert psi4_element_token("") == "C"
    assert int(psi4_atomic_number("Cl")) == 17
    assert int(psi4_atomic_number("xx")) == 6

    atoms = [{"element": "C"}, {"element": "H"}, {"element": "H"}, {"element": "H"}]
    assert int(psi4_guess_multiplicity_from_atoms(atoms, charge=0)) == 2
    assert int(psi4_guess_multiplicity_from_atoms(atoms, charge=1)) == 1

    xyz = tmp_path / "mol.xyz"
    xyz.write_text(
        "\n".join(
            [
                "2",
                "methane fragment",
                "C 0.0 0.1 0.2",
                "H 1.0 1.1 1.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = psi4_collect_atoms_from_xyz(str(xyz))
    assert len(parsed) == 2
    assert parsed[0]["name"] == "C1"
    assert parsed[1]["symbol"] == "H"

    infer_fc = psi4_infer_charge_multiplicity(
        meta={"formal_charge": -1, "radical_electrons": 0},
        atoms=atoms,
        current_charge=5,
        multiplicity_fn=lambda aa, charge=0: psi4_guess_multiplicity_from_atoms(aa, charge=charge),
    )
    assert int(infer_fc["charge"]) == -1
    assert int(infer_fc["multiplicity"]) in (1, 2)
    assert str(infer_fc["reason"]).startswith("formal_charge,")

    infer_rad = psi4_infer_charge_multiplicity(
        meta={"formal_charge": None, "radical_electrons": 2},
        atoms=atoms,
        current_charge=0,
    )
    assert int(infer_rad["charge"]) == 0
    assert int(infer_rad["multiplicity"]) == 3
    assert str(infer_rad["reason"]) == "current_charge,radical_electrons"


def test_psi4_runtime_mol2_helpers(tmp_path):
    assert psi4_mol2_atom_type_to_element("C.ar", atom_name="C1") == "C"
    assert psi4_mol2_atom_type_to_element("CL", atom_name="CL1") == "CL"
    assert (
        psi4_mol2_atom_type_to_element(
            "1",
            atom_name="XX3",
            infer_element_fn=lambda nm: ("N" if str(nm).upper().startswith("XX") else "C"),
        )
        == "N"
    )

    mol2 = tmp_path / "lig.mol2"
    mol2.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "LIG",
                "3 0 0 0 0",
                "SMALL",
                "USER_CHARGES",
                "@<TRIPOS>ATOM",
                "1 C1 0.0 0.1 0.2 C.ar 1 LIG 0.000",
                "2 CL2 1.0 1.1 1.2 CL 1 LIG -0.100",
                "3 XX3 2.0 2.1 2.2 1 1 LIG 0.200",
                "@<TRIPOS>BOND",
                "1 1 2 1",
                "2 2 3 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    atoms = psi4_collect_atoms_from_mol2(
        str(mol2),
        atom_type_to_element_fn=lambda atom_type, atom_name: psi4_mol2_atom_type_to_element(
            atom_type,
            atom_name=atom_name,
            infer_element_fn=lambda nm: ("N" if str(nm).upper().startswith("XX") else "C"),
        ),
        element_token_fn=psi4_element_token,
    )
    assert len(atoms) == 3
    assert atoms[0]["name"] == "C1"
    assert atoms[0]["element"] == "C"
    assert atoms[1]["element"] == "CL"
    assert atoms[1]["symbol"] == "Cl"
    assert atoms[2]["element"] == "N"

    records = psi4_load_mol2_atom_records(
        str(mol2),
        atom_type_to_element_fn=lambda atom_type, atom_name: psi4_mol2_atom_type_to_element(
            atom_type,
            atom_name=atom_name,
            infer_element_fn=lambda nm: ("N" if str(nm).upper().startswith("XX") else "C"),
        ),
    )
    assert len(records) == 3
    assert records[0]["id"] == 1
    assert records[0]["name"] == "C1"
    assert records[1]["element"] == "CL"
    assert abs(float(records[2]["charge"]) - 0.2) < 1e-9

    charges_by_name, bond_pairs = psi4_load_mol2_charges_and_bonds(str(mol2))
    assert abs(float(charges_by_name["CL2"]) - (-0.1)) < 1e-9
    assert ("C1", "CL2") in set(bond_pairs)
    assert ("CL2", "XX3") in set(bond_pairs)
    charges_mol2 = psi4_collect_ligand_charges_from_file(str(mol2))
    assert len(charges_mol2) == 3
    assert abs(float(charges_mol2[1]) - (-0.1)) < 1e-9
    assert int(psi4_estimate_net_charge_from_charges(charges_mol2)) == 0

    ligand_ids = [0, 1, 2]
    atom_names = {0: "C1", 1: "CL2", 2: "XX3"}
    elems = {0: "C", 1: "CL", 2: "N"}

    mapped_name, info_name = psi4_map_ligand_charges_from_param_mol2(
        ligand_ids=ligand_ids,
        atom_names=atom_names,
        elems=elems,
        records=records,
        charges_by_name=None,
    )
    assert len(mapped_name) == 3
    assert info_name["mode"] == "name_bucket"
    assert abs(float(mapped_name[1]) - (-0.1)) < 1e-9

    recs_idx = [
        {"name": "A", "element": "C", "charge": -0.2},
        {"name": "B", "element": "CL", "charge": 0.1},
        {"name": "C", "element": "N", "charge": 0.3},
    ]
    mapped_idx, info_idx = psi4_map_ligand_charges_from_param_mol2(
        ligand_ids=ligand_ids,
        atom_names={0: "X1", 1: "X2", 2: "X3"},
        elems=elems,
        records=recs_idx,
        charges_by_name=None,
    )
    assert len(mapped_idx) == 3
    assert info_idx["mode"] == "index_element_exact"

    mapped_dict, info_dict = psi4_map_ligand_charges_from_param_mol2(
        ligand_ids=ligand_ids,
        atom_names=atom_names,
        elems=elems,
        records=[],
        charges_by_name={"C1": -0.5, "XX3": 0.2},
    )
    assert len(mapped_dict) == 2
    assert info_dict["mode"] == "name_dict_partial"
    assert int(info_dict["missing"]) == 1

    src_recs = [
        {"name": "A1", "element": "C", "charge": -0.2},
        {"name": "B1", "element": "N", "charge": 0.1},
        {"name": "C1", "element": "O", "charge": 0.1},
    ]
    dst_recs_name = [
        {"name": "A1", "element": "X", "charge": 0.0},
        {"name": "B1", "element": "X", "charge": 0.0},
        {"name": "C1", "element": "X", "charge": 0.0},
    ]
    tx_name = psi4_map_charge_transfer_between_mol2_records(src_recs, dst_recs_name)
    assert tx_name["ok"] is True
    assert tx_name["mode"] in {"name_bucket", "name_order"}
    assert len(tx_name["mapped"]) == 3

    dst_recs_elem = [
        {"name": "X1", "element": "C", "charge": 0.0},
        {"name": "X2", "element": "N", "charge": 0.0},
        {"name": "X3", "element": "O", "charge": 0.0},
    ]
    tx_elem = psi4_map_charge_transfer_between_mol2_records(src_recs, dst_recs_elem)
    assert tx_elem["ok"] is True
    assert tx_elem["mode"] == "element_order"

    dst_recs_fail = [
        {"name": "Z1", "element": "S", "charge": 0.0},
        {"name": "Z2", "element": "P", "charge": 0.0},
    ]
    tx_fail = psi4_map_charge_transfer_between_mol2_records(src_recs, dst_recs_fail)
    assert tx_fail["ok"] is False
    assert tx_fail["error"] in {"charge_mapping_failed", "mol2_atom_parse_failed"}

    pdbqt = tmp_path / "lig.pdbqt"
    pdbqt.write_text(
        "\n".join(
            [
                "ATOM      1  C1  LIG A   1      0.000   0.000   0.000  0.00  0.00      0.123 C",
                "ATOM      2  N1  LIG A   1      1.000   0.000   0.000  0.00  0.00     -0.456 N",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    charges_pdbqt = psi4_collect_ligand_charges_from_file(str(pdbqt))
    assert len(charges_pdbqt) == 2
    assert abs(float(charges_pdbqt[0]) - 0.123) < 1e-9
    assert int(psi4_estimate_net_charge_from_charges([0.51, 0.49])) == 1

    typed = tmp_path / "typed.mol2"
    typed.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "TYP",
                "3 0 0 0 0",
                "SMALL",
                "USER_CHARGES",
                "@<TRIPOS>ATOM",
                "1 C1 0.0 0.1 0.2 C.ar 1 LIG 0.000",
                "2 CL2 1.0 1.1 1.2 CL 1 LIG 0.000",
                "3 XX3 2.0 2.1 2.2 N.3 1 LIG 0.000",
                "@<TRIPOS>BOND",
                "1 1 2 1",
                "2 2 3 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_charged = tmp_path / "typed_charged.mol2"
    ok_w, msg_w = psi4_write_charged_mol2(str(typed), [-0.2, 0.1, 0.1], str(out_charged))
    assert ok_w is True, msg_w
    out_charges = psi4_collect_ligand_charges_from_file(str(out_charged))
    assert len(out_charges) == 3
    assert abs(float(np.sum(np.asarray(out_charges, dtype=float))) - 0.0) < 1e-9

    out_transfer = tmp_path / "transfer_out.mol2"
    ok_tx_file, info_tx_file = psi4_transfer_mol2_charges_to_typed(
        src_charged_mol2=str(mol2),
        typed_mol2=str(typed),
        out_mol2=str(out_transfer),
    )
    assert ok_tx_file is True
    assert info_tx_file["mode"] in {"name_bucket", "name_order", "element_order", "element_bucket"}
    assert int(info_tx_file["src_atoms"]) == 3
    assert int(info_tx_file["dst_atoms"]) == 3

    frc = tmp_path / "lig.frcmod"
    frc.write_text("MASS\n", encoding="utf-8")
    prep = psi4_prepare_tleap_ligand_build(
        mol2_file=str(typed),
        frcmod_file=str(frc),
        atom_type="gaff2",
        out_dir=str(tmp_path / "tleap"),
        tleap_bin="tleap_bin",
    )
    assert prep["ok"] is True
    assert str(prep["leaprc"]) == "leaprc.gaff2"
    assert str(prep["cmdline"][0]) == "tleap_bin"
    assert str(prep["cmdline"][1]) == "-f"
    assert str(prep["leapin"]).endswith("tleap_ligand.in")
    assert str(prep["prmtop"]).endswith("ligand.prmtop")

    prep_bad = psi4_prepare_tleap_ligand_build(
        mol2_file=str(tmp_path / "missing.mol2"),
        frcmod_file=str(frc),
        atom_type="gaff",
        out_dir=str(tmp_path),
        tleap_bin="tleap",
    )
    assert prep_bad["ok"] is False
    assert str(prep_bad["error"]).startswith("Missing MOL2")


def test_psi4_runtime_rebuild_helper_payloads(tmp_path):
    src_param = tmp_path / "param.mol2"
    src_dock = tmp_path / "dock.mol2"
    src_param.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    src_dock.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")

    picked = psi4_pick_charged_mol2_for_rebuild(str(src_param), str(src_dock))
    assert picked == str(src_param)
    picked_fallback = psi4_pick_charged_mol2_for_rebuild(str(tmp_path / "missing.mol2"), str(src_dock))
    assert picked_fallback == str(src_dock)

    prep = psi4_prepare_rebuild_output_paths(str(src_param), work_dir="", fallback_work_dir=str(tmp_path / "wk"))
    assert str(prep["work"]) == str(src_param.parent)
    assert str(prep["out_mol2"]).endswith("ligand_rc_preserved.mol2")
    assert str(prep["out_frcmod"]).endswith("ligand_rc_preserved.frcmod")
    assert str(prep["typed_tmp"]).endswith("ligand_typed_tmp.mol2")

    cmd1 = psi4_build_antechamber_cmd(
        antechamber_bin="antechamber_bin",
        source_file=str(src_param),
        source_format="mol2",
        out_mol2=str(tmp_path / "out.mol2"),
        charge_method="rc",
        net_charge=-1,
        atom_type="gaff2",
        dr="n",
    )
    assert cmd1[0] == "antechamber_bin"
    assert "-dr" in set(cmd1)
    assert cmd1[cmd1.index("-c") + 1] == "rc"
    assert cmd1[cmd1.index("-nc") + 1] == "-1"

    cmd2 = psi4_build_parmchk2_cmd("parmchk2_bin", str(tmp_path / "out.mol2"), str(tmp_path / "out.frcmod"))
    assert cmd2 == ["parmchk2_bin", "-i", str(tmp_path / "out.mol2"), "-f", "mol2", "-o", str(tmp_path / "out.frcmod")]

    fail_rc = psi4_build_rebuild_failure_payload(
        stage="antechamber_rc",
        rc=7,
        stderr="E" * 1500,
        stdout="O" * 1600,
        fallback_errors=["a", "b", "c", "d", "e"],
    )
    assert fail_rc["ok"] is False
    assert str(fail_rc["error"]) == "antechamber(rc) failed rc=7"
    assert str(fail_rc["fallback"]) == "b; c; d; e"
    assert len(str(fail_rc["stderr"])) == 1200
    assert len(str(fail_rc["stdout"])) == 1200

    fail_pc = psi4_build_rebuild_failure_payload(stage="parmchk2", rc=3, stderr="x", stdout="y")
    assert fail_pc["ok"] is False
    assert str(fail_pc["error"]) == "parmchk2 failed rc=3"
    assert "fallback" not in fail_pc

    success = psi4_build_rebuild_success_payload(
        out_mol2=str(tmp_path / "lig.mol2"),
        out_frcmod=str(tmp_path / "lig.frcmod"),
        out_prmtop=str(tmp_path / "lig.prmtop"),
        out_inpcrd=str(tmp_path / "lig.inpcrd"),
        xml_path=str(tmp_path / "lig.xml"),
        net_charge=-2,
        atom_type="gaff2",
        rebuild_mode="typed_gas_charge_transfer",
        rebuild_notes=["fallback=gas", "", "map=name_bucket"],
    )
    assert success["ok"] is True
    assert int(success["charge"]) == -2
    assert str(success["atom_type"]) == "gaff2"
    assert str(success["rebuild_mode"]) == "typed_gas_charge_transfer"
    assert success["rebuild_notes"] == ["fallback=gas", "map=name_bucket"]


def test_psi4_runtime_charge_candidates_and_failure_formatters():
    cands = psi4_build_net_charge_candidates(
        requested_charge=0,
        formal_charge=-1,
        atomic_z_sum=41,
        clip_min=-8,
        clip_max=8,
    )
    assert isinstance(cands, list) and len(cands) >= 2
    assert int(cands[0]) == 0
    assert -1 in set(int(v) for v in cands)

    cands_bad = psi4_build_net_charge_candidates(
        requested_charge="bad",
        formal_charge=None,
        atomic_z_sum=None,
    )
    assert cands_bad == [0]

    hint = psi4_detect_sqm_charge_mismatch_hint("SQM ERROR: odd number of electrons in qmcharge setup")
    assert str(hint) == "sqm_hint=odd_electron_or_charge_mismatch"
    no_hint = psi4_detect_sqm_charge_mismatch_hint("all good")
    assert str(no_hint) == ""

    brief_a = psi4_format_antechamber_attempt_brief(net_charge=-1, rc=3, sqm_hint=hint)
    assert brief_a == "nc=-1 rc=3 sqm_hint=odd_electron_or_charge_mismatch"
    brief_b = psi4_format_antechamber_attempt_brief(net_charge=0, rc=0, sqm_hint="")
    assert brief_b == "nc=0 rc=0"

    fail_ante = psi4_format_antechamber_attempts_failure(
        attempt_brief=[brief_a, "", brief_b],
        stderr="E" * 800,
        stdout="O" * 700,
        stderr_tail=120,
        stdout_tail=90,
    )
    assert fail_ante.startswith("antechamber failed after nc attempts [")
    assert "nc=-1 rc=3" in fail_ante
    assert "stderr=" in fail_ante and "stdout=" in fail_ante

    fail_tool = psi4_format_tool_rc_failure(
        tool_name="parmchk2",
        rc=9,
        stderr="err",
        stdout="out",
        stderr_tail=50,
        stdout_tail=40,
    )
    assert fail_tool == "parmchk2 failed (rc=9) | stderr=err | stdout=out"


def test_psi4_runtime_parameterization_source_helpers(tmp_path):
    paths = psi4_prepare_leadopt_parameterization_paths(str(tmp_path / "wk"))
    assert str(paths["rdkit_sdf"]).endswith("ligand_for_param.sdf")
    assert str(paths["pose_pdb"]).endswith("ligand_for_param.pdb")
    assert str(paths["out_mol2"]).endswith("ligand_gaff2.mol2")
    assert str(paths["out_frcmod"]).endswith("ligand_gaff2.frcmod")

    pose_pdb = tmp_path / "pose.pdb"
    pose_pdb.write_text("ATOM 1 C\n", encoding="utf-8")
    pose_payload = psi4_prepare_pose_source_for_parameterization(
        hit={"pose_file": str(pose_pdb)},
        work_dir=str(tmp_path),
        dock_input_format_fn=lambda p: ("pdb" if str(p).lower().endswith(".pdb") else ""),
    )
    assert pose_payload["ok"] is True
    assert str(pose_payload["source_file"]) == str(pose_pdb)
    assert str(pose_payload["source_fi"]) == "pdb"
    assert "source=pose_file:.pdb" in set(pose_payload["notes"])

    pose_pdbqt = tmp_path / "pose.pdbqt"
    pose_pdbqt.write_text("ATOM 1 C\n", encoding="utf-8")
    converted = tmp_path / "converted_pose.pdb"

    def _convert_ok(src, dst):
        _ = src
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return True

    pose_payload_qt = psi4_prepare_pose_source_for_parameterization(
        hit={"pose_view_file": str(pose_pdbqt)},
        work_dir=str(tmp_path),
        convert_pdbqt_to_pdb_fn=_convert_ok,
        pose_pdb_path=str(converted),
    )
    assert pose_payload_qt["ok"] is True
    assert str(pose_payload_qt["source_file"]) == str(converted)
    assert str(pose_payload_qt["source_fi"]) == "pdb"
    assert "source=pose_pdbqt_to_pdb" in set(pose_payload_qt["notes"])

    pose_payload_qt_fail = psi4_prepare_pose_source_for_parameterization(
        hit={"pose_view_file": str(pose_pdbqt)},
        work_dir=str(tmp_path),
        convert_pdbqt_to_pdb_fn=lambda _src, _dst: False,
    )
    assert pose_payload_qt_fail["ok"] is False
    assert str(pose_payload_qt_fail["error"]) == "pdbqt_conversion_failed"

    empty_pose = tmp_path / "empty_pose.pdb"
    empty_pose.write_text("", encoding="utf-8")
    empty_pose_payload = psi4_prepare_pose_source_for_parameterization(
        hit={"pose_file": str(empty_pose)},
        work_dir=str(tmp_path),
    )
    assert empty_pose_payload["ok"] is False
    assert str(empty_pose_payload["error"]) == "pose_source_empty"

    rdkit_sdf = tmp_path / "ligand_for_param.sdf"
    rdkit_sdf.write_text("$$$$\n", encoding="utf-8")
    chosen_rd = psi4_choose_parameterization_source(
        rdkit_source_file=str(rdkit_sdf),
        pose_payload=pose_payload,
    )
    assert chosen_rd["ok"] is True
    assert str(chosen_rd["source_file"]) == str(rdkit_sdf)
    assert str(chosen_rd["source_fi"]) == "sdf"
    assert "source=rdkit_mol" in set(chosen_rd["notes"])

    chosen_pose = psi4_choose_parameterization_source(
        rdkit_source_file=str(tmp_path / "missing.sdf"),
        pose_payload=pose_payload,
    )
    assert chosen_pose["ok"] is True
    assert str(chosen_pose["source_file"]) == str(pose_pdb)
    assert str(chosen_pose["source_fi"]) == "pdb"

    empty_rdkit_sdf = tmp_path / "empty_rdkit.sdf"
    empty_rdkit_sdf.write_text("", encoding="utf-8")
    chosen_empty_rdkit = psi4_choose_parameterization_source(
        rdkit_source_file=str(empty_rdkit_sdf),
        pose_payload=pose_payload,
    )
    assert chosen_empty_rdkit["ok"] is True
    assert str(chosen_empty_rdkit["source_file"]) == str(pose_pdb)

    chosen_fail = psi4_choose_parameterization_source(
        rdkit_source_file=str(tmp_path / "missing2.sdf"),
        pose_payload={"ok": False, "error": "pose_source_missing"},
    )
    assert chosen_fail["ok"] is False
    assert str(chosen_fail["error"]) == "could not build ligand source for per-hit parameterization"
    assert str(chosen_fail["error_detail"]) == "pose_source_missing"


def test_psi4_runtime_prepare_rdkit_source_for_parameterization(tmp_path):
    out_sdf = tmp_path / "rdkit_param.sdf"

    class DummyMol(dict):
        pass

    def _clone(src):
        return DummyMol({"src": src, "has_conf": False})

    def _formal_charge(m):
        _ = m
        return -1

    def _atomic_z_sum(m):
        _ = m
        return 41

    def _has_conf(m):
        return bool(m.get("has_conf", False))

    def _add_hs(m):
        mm = DummyMol(m)
        mm["had_h"] = True
        return mm

    def _embed(m):
        m["has_conf"] = True
        return 0

    def _opt(m):
        m["opt"] = True
        return 0

    def _write(m, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"{m.get('src','x')}\n")
        return str(path)

    payload = psi4_prepare_rdkit_source_for_parameterization(
        mol_src="seed_mol",
        out_sdf=str(out_sdf),
        clone_mol_fn=_clone,
        formal_charge_fn=_formal_charge,
        atomic_z_sum_fn=_atomic_z_sum,
        has_conformer_fn=_has_conf,
        add_hs_fn=_add_hs,
        embed_fn=_embed,
        optimize_fn=_opt,
        write_sdf_fn=_write,
    )
    assert payload["ok"] is True
    assert str(payload["source_file"]) == str(out_sdf)
    assert int(payload["formal_charge"]) == -1
    assert int(payload["atomic_z_sum"]) == 41

    nested_sdf = tmp_path / "nested" / "rdkit_param.sdf"
    nested_payload = psi4_prepare_rdkit_source_for_parameterization(
        mol_src="seed_mol",
        out_sdf=str(nested_sdf),
        clone_mol_fn=_clone,
        write_sdf_fn=_write,
    )
    assert nested_payload["ok"] is True
    assert str(nested_payload["source_file"]) == str(nested_sdf)

    missing_cb = psi4_prepare_rdkit_source_for_parameterization(
        mol_src="seed_mol",
        out_sdf=str(tmp_path / "x.sdf"),
        clone_mol_fn=None,
        write_sdf_fn=None,
    )
    assert missing_cb["ok"] is False
    assert str(missing_cb["error"]) == "rdkit_callbacks_missing"

    clone_fail = psi4_prepare_rdkit_source_for_parameterization(
        mol_src="seed_mol",
        out_sdf=str(tmp_path / "y.sdf"),
        clone_mol_fn=lambda _src: None,
        write_sdf_fn=_write,
    )
    assert clone_fail["ok"] is False
    assert str(clone_fail["error"]) == "mol_clone_failed"

    write_fail = psi4_prepare_rdkit_source_for_parameterization(
        mol_src="seed_mol",
        out_sdf=str(tmp_path / "z.sdf"),
        clone_mol_fn=_clone,
        write_sdf_fn=lambda _m, _p: "",
    )
    assert write_fail["ok"] is False
    assert str(write_fail["error"]) == "rdkit_sdf_not_written"

    empty_write = psi4_prepare_rdkit_source_for_parameterization(
        mol_src="seed_mol",
        out_sdf=str(tmp_path / "empty_written.sdf"),
        clone_mol_fn=_clone,
        write_sdf_fn=lambda _m, p: (open(p, "w", encoding="utf-8").close() or str(p)),
    )
    assert empty_write["ok"] is False
    assert str(empty_write["error"]) == "rdkit_sdf_not_written"


def test_psi4_runtime_normalize_rdkit_parameterization_payload(tmp_path):
    sdf = tmp_path / "rdkit_ok.sdf"
    sdf.write_text("$$$$\n", encoding="utf-8")

    norm_ok = psi4_normalize_rdkit_parameterization_payload(
        {
            "ok": True,
            "source_file": str(sdf),
            "formal_charge": "-1",
            "atomic_z_sum": "41",
            "error": "",
            "notes": ["", "embedded", " optimized "],
        }
    )
    assert str(norm_ok["source_file"]) == str(sdf)
    assert int(norm_ok["formal_charge"]) == -1
    assert int(norm_ok["atomic_z_sum"]) == 41
    assert "rdkit_formal_charge=-1" in set(norm_ok["notes"])
    assert "embedded" in set(norm_ok["notes"])
    assert "optimized" in set(norm_ok["notes"])

    norm_bad = psi4_normalize_rdkit_parameterization_payload(
        {
            "ok": False,
            "source_file": str(sdf),
            "formal_charge": "x",
            "atomic_z_sum": "y",
            "error": "mol_clone_failed",
            "notes": ["debug_note"],
        }
    )
    assert str(norm_bad["source_file"]) == ""
    assert norm_bad["formal_charge"] is None
    assert norm_bad["atomic_z_sum"] is None
    assert "rdkit_source_failed:mol_clone_failed" in set(norm_bad["notes"])
    assert "debug_note" in set(norm_bad["notes"])

    empty_sdf = tmp_path / "empty_rdkit_payload.sdf"
    empty_sdf.write_text("", encoding="utf-8")
    norm_empty = psi4_normalize_rdkit_parameterization_payload(
        {"ok": True, "source_file": str(empty_sdf), "formal_charge": 0}
    )
    assert str(norm_empty["source_file"]) == ""
    assert int(norm_empty["formal_charge"]) == 0


def test_psi4_runtime_resolve_leadopt_rdkit_state(tmp_path):
    paths = {"rdkit_sdf": str(tmp_path / "leadopt_rdkit.sdf")}

    no_rdkit = psi4_resolve_leadopt_rdkit_state(
        hit={"mol": "x"},
        paths=paths,
        has_rdkit=False,
        prepare_rdkit_source_fn=lambda _mol, _sdf: {"ok": True},
    )
    assert no_rdkit["source_file"] == ""
    assert no_rdkit["formal_charge"] is None
    assert no_rdkit["atomic_z_sum"] is None
    assert no_rdkit["notes"] == []

    no_mol = psi4_resolve_leadopt_rdkit_state(
        hit={},
        paths=paths,
        has_rdkit=True,
        prepare_rdkit_source_fn=lambda _mol, _sdf: {"ok": True},
    )
    assert no_mol["source_file"] == ""
    assert no_mol["notes"] == []

    def _prep_ok(mol_src, out_sdf):
        _ = mol_src
        with open(out_sdf, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return {
            "ok": True,
            "source_file": out_sdf,
            "formal_charge": -2,
            "atomic_z_sum": 30,
            "notes": ["prepared"],
        }

    ok_state = psi4_resolve_leadopt_rdkit_state(
        hit={"mol": "seed"},
        paths=paths,
        has_rdkit=True,
        prepare_rdkit_source_fn=_prep_ok,
    )
    assert str(ok_state["source_file"]).endswith("leadopt_rdkit.sdf")
    assert int(ok_state["formal_charge"]) == -2
    assert int(ok_state["atomic_z_sum"]) == 30
    assert "rdkit_formal_charge=-2" in set(ok_state["notes"])
    assert "prepared" in set(ok_state["notes"])

    fail_state = psi4_resolve_leadopt_rdkit_state(
        hit={"mol": "seed"},
        paths=paths,
        has_rdkit=True,
        prepare_rdkit_source_fn=lambda _mol, _sdf: (_ for _ in ()).throw(RuntimeError("prep_failed")),
    )
    assert fail_state["source_file"] == ""
    assert "rdkit_source_failed:prep_failed" in set(fail_state["notes"])


def test_psi4_runtime_resolve_leadopt_parameterization_source(tmp_path):
    work = tmp_path / "wk"
    work.mkdir(parents=True, exist_ok=True)
    pose_pdb = work / "pose.pdb"
    pose_pdb.write_text("ATOM 1 C\n", encoding="utf-8")
    rd_sdf = work / "rd.sdf"
    rd_sdf.write_text("$$$$\n", encoding="utf-8")
    paths = {"pose_pdb": str(work / "ligand_for_param.pdb")}

    src_rd = psi4_resolve_leadopt_parameterization_source(
        hit={"pose_file": str(pose_pdb)},
        work_dir=str(work),
        paths=paths,
        rdkit_source_file=str(rd_sdf),
        prepare_pose_source_fn=lambda **_kw: {"ok": False, "error": "pose_source_missing"},
    )
    assert src_rd["ok"] is True
    assert src_rd["source_file"] == str(rd_sdf)
    assert src_rd["source_fi"] == "sdf"

    src_pose = psi4_resolve_leadopt_parameterization_source(
        hit={"pose_file": str(pose_pdb)},
        work_dir=str(work),
        paths=paths,
        rdkit_source_file=str(work / "missing.sdf"),
        prepare_pose_source_fn=lambda **_kw: {
            "ok": True,
            "source_file": str(pose_pdb),
            "source_fi": "pdb",
            "notes": ["pose_ok"],
        },
    )
    assert src_pose["ok"] is True
    assert src_pose["source_file"] == str(pose_pdb)
    assert src_pose["source_fi"] == "pdb"
    assert "pose_ok" in set(src_pose["notes"])

    src_fail = psi4_resolve_leadopt_parameterization_source(
        hit={"pose_file": str(pose_pdb)},
        work_dir=str(work),
        paths=paths,
        rdkit_source_file=str(work / "missing2.sdf"),
        prepare_pose_source_fn=lambda **_kw: {"ok": False, "error": "pose_source_missing"},
    )
    assert src_fail["ok"] is False
    assert src_fail["source_file"] == ""
    assert src_fail["source_fi"] == ""
    assert src_fail["reason"] == "could not build ligand source for per-hit parameterization"


def test_psi4_runtime_leadopt_info_helpers():
    info0 = psi4_make_leadopt_parameterization_info()
    assert info0["ok"] is False
    assert info0["reason"] == ""
    assert isinstance(info0["notes"], list) and len(info0["notes"]) == 0

    cached = psi4_make_leadopt_cached_info("a.mol2", "a.frcmod", "a.prmtop", "a.inpcrd")
    assert cached["ok"] is True
    assert cached["source"] == "cached"
    assert cached["mol2"] == "a.mol2"

    generated = psi4_make_leadopt_generated_info("b.mol2", "b.frcmod", "b.prmtop", "b.inpcrd")
    assert generated["ok"] is True
    assert generated["source"] == "generated"
    assert generated["prmtop"] == "b.prmtop"

    miss = psi4_reason_missing_parameterization_tools(["antechamber", "", "tleap"])
    assert miss == "ligand parameterization required for advanced refinement, but missing tools: antechamber, tleap"
    assert psi4_reason_missing_parameterization_tools([]) == ""

    src_reason = psi4_reason_source_for_parameterization_unavailable()
    assert src_reason == "could not build ligand source for per-hit parameterization"

    tl_reason = psi4_reason_tleap_parameterization_failed("boom")
    assert tl_reason == "tleap failed for ligand parameterization: boom"

    merged = psi4_merge_notes(["a", "  ", "b"], ["", "c", " d "])
    assert merged == ["a", "b", "c", "d"]

    updates = psi4_build_leadopt_param_hit_updates("m.mol2", "m.frcmod", "m.prmtop", "m.inpcrd")
    assert updates == {
        "param_mol2": "m.mol2",
        "param_frcmod": "m.frcmod",
        "param_prmtop": "m.prmtop",
        "param_inpcrd": "m.inpcrd",
    }

    target = {"x": 1}
    assert psi4_apply_mapping_to_dict(target, updates) is True
    assert target["param_mol2"] == "m.mol2"
    assert psi4_apply_mapping_to_dict([], updates) is False

    final_payload = psi4_finalize_leadopt_parameterization_success(
        mol2="g.mol2",
        frcmod="g.frcmod",
        prmtop="g.prmtop",
        inpcrd="g.inpcrd",
    )
    assert final_payload["info_update"]["source"] == "generated"
    assert final_payload["info_update"]["prmtop"] == "g.prmtop"
    assert final_payload["hit_updates"]["param_inpcrd"] == "g.inpcrd"


def test_psi4_runtime_cached_and_orchestrator_helpers(tmp_path):
    cached_mol2 = tmp_path / "cached.mol2"
    cached_frcmod = tmp_path / "cached.frcmod"
    cached_prmtop = tmp_path / "cached.prmtop"
    cached_inpcrd = tmp_path / "cached.inpcrd"
    cached_mol2.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    cached_frcmod.write_text("MASS\n", encoding="utf-8")
    cached_prmtop.write_text("PRM\n", encoding="utf-8")
    hit_cached = {
        "param_mol2": str(cached_mol2),
        "param_frcmod": str(cached_frcmod),
        "param_prmtop": str(cached_prmtop),
        "param_inpcrd": str(cached_inpcrd),
    }
    cached = psi4_resolve_leadopt_cached_parameterization(hit=hit_cached)
    assert cached["ok"] is True
    assert cached["mol2"] == str(cached_mol2)
    assert cached["frcmod"] == str(cached_frcmod)
    assert cached["prmtop"] == str(cached_prmtop)
    assert cached["inpcrd"] == ""
    assert cached["info_update"]["source"] == "cached"

    tools_missing = psi4_validate_leadopt_parameterization_tools(
        antechamber_bin="/usr/bin/antechamber",
        parmchk2_bin="",
        tleap_bin="",
    )
    assert tools_missing["ok"] is False
    assert tools_missing["missing"] == ["parmchk2", "tleap"]
    assert "missing tools" in tools_missing["reason"]

    ran = {"run_pipeline": False}

    def _run_should_not_execute(**_kwargs):
        ran["run_pipeline"] = True
        return {"ok": True}

    missing_payload = psi4_orchestrate_leadopt_parameterization(
        hit={},
        work_dir=str(tmp_path / "wk_missing"),
        antechamber_bin="",
        parmchk2_bin="",
        tleap_bin="",
        run_pipeline_fn=_run_should_not_execute,
    )
    assert missing_payload["ok"] is False
    assert "missing tools" in str(missing_payload["info"].get("reason", ""))
    assert ran["run_pipeline"] is False

    hit_generated = {}
    captured = {"charge_candidates": []}
    src_file = tmp_path / "ligand_src.sdf"
    src_file.write_text("$$$$\n", encoding="utf-8")

    def _run_ok(**kwargs):
        captured["charge_candidates"] = list(kwargs.get("charge_candidates", []) or [])
        return {
            "ok": True,
            "out_mol2": str(tmp_path / "ligand_gaff2.mol2"),
            "out_frcmod": str(tmp_path / "ligand_gaff2.frcmod"),
            "prmtop": str(tmp_path / "ligand.prmtop"),
            "inpcrd": str(tmp_path / "ligand.inpcrd"),
            "notes": ["run_ok"],
        }

    generated_payload = psi4_orchestrate_leadopt_parameterization(
        hit=hit_generated,
        work_dir=str(tmp_path / "wk_generated"),
        net_charge=2,
        charge_method="bcc",
        atom_type="gaff2",
        antechamber_bin="/usr/bin/antechamber",
        parmchk2_bin="/usr/bin/parmchk2",
        tleap_bin="/usr/bin/tleap",
        resolve_cached_fn=lambda **_kw: {"ok": False},
        resolve_rdkit_state_fn=lambda **_kw: {
            "source_file": str(src_file),
            "formal_charge": 1,
            "atomic_z_sum": 30,
            "notes": ["rdkit_ok"],
        },
        resolve_source_fn=lambda **_kw: {
            "ok": True,
            "source_file": str(src_file),
            "source_fi": "sdf",
            "notes": ["source_ok"],
        },
        build_charge_candidates_fn=lambda **kw: [kw.get("requested_charge"), kw.get("formal_charge")],
        run_pipeline_fn=_run_ok,
    )
    assert generated_payload["ok"] is True
    info = dict(generated_payload["info"] or {})
    assert info["source"] == "generated"
    assert info["prmtop"] == str(tmp_path / "ligand.prmtop")
    assert info["notes"] == ["rdkit_ok", "source_ok", "run_ok"]
    assert captured["charge_candidates"] == [2, 1]
    assert hit_generated["param_mol2"] == str(tmp_path / "ligand_gaff2.mol2")
    assert hit_generated["param_prmtop"] == str(tmp_path / "ligand.prmtop")


def test_psi4_runtime_run_leadopt_parameterization_pipeline(tmp_path):
    work = tmp_path / "wk"
    work.mkdir(parents=True, exist_ok=True)
    src = work / "src.sdf"
    src.write_text("$$$$\n", encoding="utf-8")
    out_mol2 = work / "ligand_gaff2.mol2"
    out_frcmod = work / "ligand_gaff2.frcmod"
    out_prmtop = work / "ligand.prmtop"
    out_inpcrd = work / "ligand.inpcrd"

    def _run_ok(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        exe = os.path.basename(str(argv[0]))
        if "antechamber" in exe:
            out_path = str(argv[argv.index("-o") + 1])
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("@<TRIPOS>MOLECULE\n")
            return True, "ok", "", 0
        if "parmchk2" in exe:
            out_path = str(argv[argv.index("-o") + 1])
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("MASS\n")
            return True, "ok", "", 0
        return False, "", "bad", 9

    def _tleap_ok(mol2, frcmod, atom_type="gaff2", out_dir=None):
        _ = (mol2, frcmod, atom_type, out_dir)
        with open(out_prmtop, "w", encoding="utf-8") as fh:
            fh.write("PRM\n")
        with open(out_inpcrd, "w", encoding="utf-8") as fh:
            fh.write("CRD\n")
        return True, "ok", str(out_prmtop), str(out_inpcrd)

    ok_payload = psi4_run_leadopt_parameterization_pipeline(
        antechamber_bin="antechamber_bin",
        parmchk2_bin="parmchk2_bin",
        source_file=str(src),
        source_format="sdf",
        out_mol2=str(out_mol2),
        out_frcmod=str(out_frcmod),
        work_dir=str(work),
        charge_candidates=[-1, 0],
        charge_method="bcc",
        atom_type="gaff2",
        run_command_fn=_run_ok,
        build_ligand_amber_fn=_tleap_ok,
    )
    assert ok_payload["ok"] is True
    assert str(ok_payload["stage"]) == ""
    assert str(ok_payload["out_mol2"]) == str(out_mol2)
    assert str(ok_payload["out_frcmod"]) == str(out_frcmod)
    assert str(ok_payload["prmtop"]) == str(out_prmtop)
    assert str(ok_payload["inpcrd"]) == str(out_inpcrd)
    assert "antechamber_nc=-1" in set(ok_payload["notes"])

    sqm = work / "sqm.out"
    sqm.write_text("odd number of electrons\n", encoding="utf-8")

    def _run_fail_ante(argv, cwd=None, timeout=0):
        _ = (argv, timeout)
        with open(os.path.join(str(cwd or work), "sqm.out"), "w", encoding="utf-8") as fh:
            fh.write("odd number of electrons\n")
        return False, "stdout text", "stderr text", 2

    fail_ante = psi4_run_leadopt_parameterization_pipeline(
        antechamber_bin="antechamber_bin",
        parmchk2_bin="parmchk2_bin",
        source_file=str(src),
        source_format="sdf",
        out_mol2=str(work / "fail_ante.mol2"),
        out_frcmod=str(work / "fail_ante.frcmod"),
        work_dir=str(work),
        charge_candidates=[0],
        run_command_fn=_run_fail_ante,
        build_ligand_amber_fn=_tleap_ok,
    )
    assert fail_ante["ok"] is False
    assert str(fail_ante["stage"]) == "antechamber"
    assert "antechamber failed after nc attempts" in str(fail_ante["reason"])
    assert "sqm_hint=odd_electron_or_charge_mismatch" in str(fail_ante["reason"])

    empty_src = work / "empty_src.sdf"
    empty_src.write_text("", encoding="utf-8")
    setup_fail = psi4_run_leadopt_parameterization_pipeline(
        antechamber_bin="antechamber_bin",
        parmchk2_bin="parmchk2_bin",
        source_file=str(empty_src),
        source_format="sdf",
        out_mol2=str(work / "setup_fail.mol2"),
        out_frcmod=str(work / "setup_fail.frcmod"),
        work_dir=str(work),
        charge_candidates=[0],
        run_command_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("should_not_run")),
        build_ligand_amber_fn=_tleap_ok,
    )
    assert setup_fail["ok"] is False
    assert str(setup_fail["stage"]) == "setup"
    assert str(setup_fail["reason"]) == "source_file_missing_or_empty"

    stale_mol2 = work / "stale_output.mol2"
    stale_mol2.write_text("@<TRIPOS>MOLECULE\nSTALE\n", encoding="utf-8")
    stale_calls = []

    def _run_fail_stale(argv, cwd=None, timeout=0):
        _ = timeout
        if "antechamber" in os.path.basename(str(argv[0])):
            stale_calls.append(int(argv[argv.index("-nc") + 1]))
            with open(os.path.join(str(cwd or work), "sqm.out"), "w", encoding="utf-8") as fh:
                fh.write("odd number of electrons\n")
        return False, "stdout text", "stderr text", 2

    stale_fail = psi4_run_leadopt_parameterization_pipeline(
        antechamber_bin="antechamber_bin",
        parmchk2_bin="parmchk2_bin",
        source_file=str(src),
        source_format="sdf",
        out_mol2=str(stale_mol2),
        out_frcmod=str(work / "stale_output.frcmod"),
        work_dir=str(work),
        charge_candidates=[0, "0", 1],
        run_command_fn=_run_fail_stale,
        build_ligand_amber_fn=_tleap_ok,
    )
    assert stale_fail["ok"] is False
    assert str(stale_fail["stage"]) == "antechamber"
    assert stale_calls == [0, 1]
    assert not stale_mol2.exists()

    def _run_fail_parm(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        exe = os.path.basename(str(argv[0]))
        if "antechamber" in exe:
            out_path = str(argv[argv.index("-o") + 1])
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("@<TRIPOS>MOLECULE\n")
            return True, "", "", 0
        return False, "parm out", "parm err", 5

    fail_parm = psi4_run_leadopt_parameterization_pipeline(
        antechamber_bin="antechamber_bin",
        parmchk2_bin="parmchk2_bin",
        source_file=str(src),
        source_format="sdf",
        out_mol2=str(work / "ok_ante.mol2"),
        out_frcmod=str(work / "fail_parm.frcmod"),
        work_dir=str(work),
        charge_candidates=[0],
        run_command_fn=_run_fail_parm,
        build_ligand_amber_fn=_tleap_ok,
    )
    assert fail_parm["ok"] is False
    assert str(fail_parm["stage"]) == "parmchk2"
    assert "parmchk2 failed (rc=5)" in str(fail_parm["reason"])

    def _run_ok_2(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        os.path.basename(str(argv[0]))
        out_path = str(argv[argv.index("-o") + 1])
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        return True, "", "", 0

    def _tleap_fail(mol2, frcmod, atom_type="gaff2", out_dir=None):
        _ = (mol2, frcmod, atom_type, out_dir)
        return False, "boom", "", ""

    fail_tleap = psi4_run_leadopt_parameterization_pipeline(
        antechamber_bin="antechamber_bin",
        parmchk2_bin="parmchk2_bin",
        source_file=str(src),
        source_format="sdf",
        out_mol2=str(work / "ok2_ante.mol2"),
        out_frcmod=str(work / "ok2_parm.frcmod"),
        work_dir=str(work),
        charge_candidates=[1],
        run_command_fn=_run_ok_2,
        build_ligand_amber_fn=_tleap_fail,
    )
    assert fail_tleap["ok"] is False
    assert str(fail_tleap["stage"]) == "tleap"
    assert str(fail_tleap["reason"]) == "tleap failed for ligand parameterization: boom"

    partial_no_tleap = psi4_run_leadopt_parameterization_pipeline(
        antechamber_bin="antechamber_bin",
        parmchk2_bin="parmchk2_bin",
        source_file=str(src),
        source_format="sdf",
        out_mol2=str(work / "partial_ante.mol2"),
        out_frcmod=str(work / "partial_parm.frcmod"),
        work_dir=str(work),
        charge_candidates=[0],
        run_command_fn=_run_ok_2,
        build_ligand_amber_fn=_tleap_fail,
        require_amber_files=False,
    )
    assert partial_no_tleap["ok"] is True
    assert str(partial_no_tleap["prmtop"]) == ""
    assert any(str(x).startswith("tleap_warning:") for x in list(partial_no_tleap["notes"] or []))


def test_leadopt_runtime_scoring_and_filters():
    good_props = {
        "mw": 350.0,
        "logp": 2.2,
        "hba": 6,
        "hbd": 1,
        "tpsa": 75.0,
        "rotb": 4,
        "pains": False,
    }
    bad_props = {
        "mw": 900.0,
        "logp": 9.0,
        "hba": 18,
        "hbd": 12,
        "tpsa": 280.0,
        "rotb": 24,
        "pains": True,
    }
    good_score = admet_score(good_props)
    bad_score = admet_score(bad_props)
    assert 0.0 <= bad_score < good_score <= 1.0

    limits = {
        "mw_max": 550.0,
        "logp_max": 5.5,
        "hba_max": 12,
        "hbd_max": 7,
        "tpsa_max": 160.0,
        "rot_max": 12,
        "block_pains": True,
    }
    assert filter_fail_reason(good_props, limits) == ""
    assert pass_filters(good_props, limits) is True
    assert pass_filters(bad_props, limits) is False

    relaxed = relax_filter_limits(limits, level=2)
    assert relaxed["mw_max"] > limits["mw_max"]

    mpo_good = mpo_score(-9.0, 0.9, 0.8, 0.7)
    mpo_bad = mpo_score(-5.0, 0.2, 0.1, 0.1)
    assert mpo_good > mpo_bad


def test_leadopt_runtime_refine_helpers(tmp_path):
    base = make_leadopt_refine_result()
    assert base["refine_status"] == "failed"
    assert base["refine_param_mol2"] == ""

    opts = leadopt_sanitize_refine_parameterization_options(
        net_charge="2",
        charge_method="",
        atom_type=None,
    )
    assert opts == {"net_charge": 2, "charge_method": "bcc", "atom_type": "gaff2"}

    prep_dir = leadopt_prepare_refine_param_dir(
        pose_file="/tmp/job/poses/hit_001.pdbqt",
        hit_idx=4,
        tag="12345678",
        workspace_dir=str(tmp_path),
    )
    assert prep_dir["job_guess"] == "/tmp/job"
    assert prep_dir["param_dir"].endswith("refine_params/hit_005_12345678")

    missing_fn = leadopt_prepare_parameterization_for_refinement(
        hit={},
        hit_idx=0,
        pose_file="/tmp/a.pdbqt",
        workspace_dir=str(tmp_path),
        tag="00000000",
    )
    assert missing_fn["ok"] is False
    assert missing_fn["reason"] == "prepare_parameterized_fn_missing"

    p_mol2 = str(tmp_path / "ligand.mol2")
    p_prm = str(tmp_path / "ligand.prmtop")
    existing = {p_mol2, p_prm}

    def _prepare_ok(hit, param_dir, net_charge=0, charge_method="bcc", atom_type="gaff2"):
        _ = (hit, param_dir, net_charge, charge_method, atom_type)
        return True, {"mol2": p_mol2, "prmtop": p_prm, "source": "generated"}

    payload_ok = leadopt_prepare_parameterization_for_refinement(
        hit={"name": "x"},
        hit_idx=1,
        pose_file="/tmp/b.pdbqt",
        workspace_dir=str(tmp_path),
        tag="11111111",
        net_charge=1,
        charge_method="bcc",
        atom_type="gaff2",
        prepare_parameterized_fn=_prepare_ok,
        load_charges_and_bonds_fn=lambda _mol2: ({"C1": -0.1}, []),
        load_atom_records_fn=lambda _mol2: [],
        file_exists_fn=lambda path: str(path) in existing,
    )
    assert payload_ok["ok"] is True
    assert payload_ok["param_mol2"] == p_mol2
    assert payload_ok["param_prmtop"] == p_prm
    assert payload_ok["param_source"] == "generated"
    assert payload_ok["param_charges_by_name"]["C1"] == -0.1

    payload_empty = leadopt_prepare_parameterization_for_refinement(
        hit={},
        hit_idx=0,
        pose_file="/tmp/c.pdbqt",
        workspace_dir=str(tmp_path),
        tag="22222222",
        prepare_parameterized_fn=_prepare_ok,
        load_charges_and_bonds_fn=lambda _mol2: ({}, []),
        load_atom_records_fn=lambda _mol2: [],
        file_exists_fn=lambda path: str(path) in existing,
    )
    assert payload_empty["ok"] is False
    assert payload_empty["reason"] == "parameterized_mol2_has_no_atom_charges"

    src = leadopt_append_charge_map_source("cached", "name_bucket", 23, 26)
    assert src == "cached|charge_map=name_bucket:23/26"

    too_few = leadopt_finalize_refine_estimate(
        est={"ok": True, "dg_kj": -1.0},
        energies=[-1.0, -2.0],
        dmins=[2.0, 3.0],
        min_samples=4,
    )
    assert too_few["ok"] is False
    assert too_few["reason"] == "too_few_valid_energy_samples"

    est_fail = leadopt_finalize_refine_estimate(
        est={"ok": False, "reason": "pymbar_fail"},
        energies=[-1.0, -2.0, -3.0, -4.0],
        dmins=[2.0, 3.0, 4.0, 5.0],
        min_samples=4,
    )
    assert est_fail["ok"] is False
    assert est_fail["reason"] == "pymbar_fail"

    est_ok = leadopt_finalize_refine_estimate(
        est={
            "ok": True,
            "method": "pymbar.exp",
            "dg_kj": -10.5,
            "dg_err_kj": 1.1,
            "ci_low_kj": -12.0,
            "ci_high_kj": -9.0,
            "n_eff": 7,
        },
        energies=[-1.0, -2.0, -3.0, -4.0],
        dmins=[2.0, 3.0, 4.0, 5.0],
        min_samples=4,
    )
    assert est_ok["ok"] is True
    upd = dict(est_ok["update"] or {})
    assert upd["refine_status"] == "ok"
    assert upd["refine_method"] == "pymbar.exp"
    assert upd["refine_samples"] == 4
    assert upd["refine_n_eff"] == 7
    assert np.isfinite(float(upd["refine_mean_e_kj"]))
    assert np.isfinite(float(upd["refine_min_dist_a"]))

    class _Atom:
        def __init__(self, index, name, symbol, resn="LIG", chain="A", resi="1"):
            self.index = index
            self.name = name
            self.symbol = symbol
            self.resn = resn
            self.chain = chain
            self.resi = resi

    atoms = [
        _Atom(10, "CA", "C", resn="ALA", chain="A", resi="1"),
        _Atom(11, "N", "N", resn="ALA", chain="A", resi="1"),
        _Atom(12, "C1", "C", resn="LIG", chain="L", resi="2"),
        _Atom(13, "H1", "H", resn="LIG", chain="L", resi="2"),
    ]
    sel = leadopt_build_complex_selection_ids(
        atoms=atoms,
        polymer_index_rows=[("obj", 10), ("obj", 11)],
        ligand_index_rows=[("obj", 12), ("obj", 13)],
    )
    assert sel["protein_ids"] == [0, 1]
    assert sel["ligand_ids"] == [2, 3]

    atom_meta = leadopt_infer_atom_names_elements(atoms, infer_element_fn=lambda _nm, sy: sy)
    assert atom_meta["atom_names"] == ["CA", "N", "C1", "H1"]
    assert atom_meta["elements"] == ["C", "N", "C", "H"]

    nb = leadopt_build_nonbonded_particle_data(
        atoms=atoms,
        elements=atom_meta["elements"],
        atom_names=atom_meta["atom_names"],
        ligand_ids=[2, 3],
        ligand_charge_map={2: -0.4},
        nb_params_for_element_fn=lambda el: (0.31 if el == "C" else 0.29, 0.12 if el != "H" else 0.03),
        mass_for_element_fn=lambda el: 1.0 if el == "H" else 12.0,
        estimate_charge_fn=lambda info: 0.2 if info.get("origin") == "protein" else -0.1,
    )
    assert len(nb["sigmas"]) == 4
    assert float(nb["charges"][2]) == -0.4
    assert float(nb["charges"][0]) == 0.2

    assert leadopt_heavy_atom_ids([0, 1, 2, 3], atom_meta["elements"]) == [0, 1, 2]
    prot_eval = leadopt_pick_protein_eval_ids(
        coords_nm=np.zeros((4, 3), dtype=float),
        protein_ids=[0, 1],
        ligand_ids_for_selection=[2],
        select_local_interacting_fn=lambda *_args, **_kwargs: [0],
        cutoff_nm=0.9,
        min_keep=2,
    )
    assert prot_eval == [0, 1]

    class _FakeUnit:
        dalton = 1.0
        nanometer = 1.0
        kelvin = 1.0
        picosecond = 1.0
        picoseconds = 1.0
        kilojoule_per_mole = 1.0

    class _FakeSystem:
        def __init__(self):
            self.particles = []
            self.forces = []

        def addParticle(self, v):
            self.particles.append(float(v))

        def addForce(self, f):
            self.forces.append(f)

    class _FakeCNF:
        CutoffNonPeriodic = 1

        def __init__(self, expr):
            self.expr = expr
            self.params = []
            self.particles = []
            self.groups = []
            self.method = None
            self.cutoff = None
            self.lrc = False
            self.force_group = 0

        def addPerParticleParameter(self, name):
            self.params.append(str(name))

        def addParticle(self, vals):
            self.particles.append(list(vals))

        def setNonbondedMethod(self, m):
            self.method = m

        def setCutoffDistance(self, c):
            self.cutoff = float(c)

        def setUseLongRangeCorrection(self, v):
            self.lrc = bool(v)

        def addInteractionGroup(self, a, b):
            self.groups.append((set(a), set(b)))

        def setForceGroup(self, g):
            self.force_group = int(g)

    class _FakeCEF:
        def __init__(self, expr):
            self.expr = expr
            self.params = []
            self.particles = []
            self.force_group = 0

        def addPerParticleParameter(self, name):
            self.params.append(str(name))

        def addParticle(self, idx, vals):
            self.particles.append((int(idx), list(vals)))

        def setForceGroup(self, g):
            self.force_group = int(g)

    class _FakeIntegrator:
        def __init__(self, *_args):
            self.steps = 0

        def step(self, n):
            self.steps += int(n)

    class _FakePos:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=float)

        def value_in_unit(self, _unit):
            return np.asarray(self._arr, dtype=float)

    class _FakeState:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=float)

        def getPositions(self, asNumpy=False):
            _ = asNumpy
            return _FakePos(self._arr)

    class _FakeContext:
        def __init__(self, system, integrator, platform=None, properties=None):
            _ = (system, integrator, platform, properties)
            self._xyz = np.zeros((4, 3), dtype=float)

        def setPositions(self, xyz):
            self._xyz = np.asarray(xyz, dtype=float)

        def setVelocitiesToTemperature(self, _temp):
            return None

        def getState(self, getPositions=False):
            _ = getPositions
            return _FakeState(self._xyz)

    class _FakeMin:
        @staticmethod
        def minimize(_context, _tol, _iters):
            return None

    class _FakeOMM:
        System = _FakeSystem
        CustomNonbondedForce = _FakeCNF
        CustomExternalForce = _FakeCEF
        LangevinMiddleIntegrator = _FakeIntegrator
        Context = _FakeContext
        LocalEnergyMinimizer = _FakeMin

    sys_payload = leadopt_build_openmm_refinement_system(
        omm_mod=_FakeOMM,
        omm_unit=_FakeUnit(),
        masses=nb["masses"],
        sigmas=nb["sigmas"],
        epsilons=nb["epsilons"],
        charges=nb["charges"],
        protein_ids=[0, 1],
        ligand_ids=[2, 3],
        protein_eval=[0, 1],
        coords_nm=np.zeros((4, 3), dtype=float),
        elements=atom_meta["elements"],
        local_infos=nb["local_infos"],
        param_prmtop="ligand.prmtop",
        inject_ligand_ff_terms_fn=lambda *_args, **_kwargs: (True, {"terms": 5}),
        cutoff_nm=1.2,
    )
    assert sys_payload["ok"] is True
    assert isinstance(sys_payload["system"], _FakeSystem)
    assert sys_payload["ff_info"]["terms"] == 5

    sample_fail = leadopt_run_openmm_refinement_sampling(
        omm_mod=_FakeOMM,
        omm_unit=_FakeUnit(),
        system=sys_payload["system"],
        coords_nm=np.zeros((4, 3), dtype=float),
        protein_eval=[0, 1],
        ligand_ids=[2, 3],
        sigmas=nb["sigmas"],
        epsilons=nb["epsilons"],
        charges=nb["charges"],
        protein_heavy=[0, 1],
        ligand_heavy=[2],
        compute_pairwise_energy_fn=None,
    )
    assert sample_fail["ok"] is False
    assert sample_fail["reason"] == "compute_pairwise_energy_fn_missing"

    sample_ok = leadopt_run_openmm_refinement_sampling(
        omm_mod=_FakeOMM,
        omm_unit=_FakeUnit(),
        system=sys_payload["system"],
        coords_nm=np.zeros((4, 3), dtype=float),
        protein_eval=[0, 1],
        ligand_ids=[2, 3],
        sigmas=nb["sigmas"],
        epsilons=nb["epsilons"],
        charges=nb["charges"],
        protein_heavy=[0, 1],
        ligand_heavy=[2],
        temperature_k=300.0,
        equil_steps=60,
        prod_steps=120,
        sample_interval=12,
        minimize_max_iters=10,
        cutoff_nm=1.2,
        auto_pick_platform_fn=lambda: (None, {}, "cpu"),
        compute_pairwise_energy_fn=lambda *_args, **_kwargs: (1.5, -0.2),
    )
    assert sample_ok["ok"] is True
    assert sample_ok["platform"] == "default"
    assert len(sample_ok["energies"]) > 0
    assert len(sample_ok["dmins"]) == len(sample_ok["energies"])

    report0 = make_leadopt_advanced_refinement_report()
    assert report0 == {
        "requested": 0,
        "attempted": 0,
        "refined": 0,
        "cancelled": False,
        "method": "",
        "message": "",
    }

    results_rows = [
        {"status": "ok", "mpo": 0.6, "name": "a"},
        {"status": "failed", "mpo": 0.5, "name": "b"},
        {"status": "ok", "mpo": 0.3, "name": "c"},
        {"status": "ok", "mpo": np.nan, "name": "d"},
    ]
    rows_payload = leadopt_select_advanced_refinement_rows(results_rows, top_n=2)
    assert rows_payload["requested"] == 2
    assert len(rows_payload["rows"]) == 2
    assert rows_payload["rows"][0][0] == 0
    assert rows_payload["rows"][1][0] == 2

    no_rows = leadopt_select_advanced_refinement_rows([{"status": "failed", "mpo": 1.0}], top_n=3)
    assert no_rows["rows"] == []
    assert "no successful docked candidates" in no_rows["message"].lower()

    rows_for_param = [
        (0, {"param_mol2": "/tmp/a.mol2", "param_prmtop": "/tmp/a.prmtop"}),
        (1, {"param_mol2": "/tmp/b.mol2", "param_prmtop": "/tmp/b.prmtop"}),
    ]
    exists = {"/tmp/a.mol2", "/tmp/a.prmtop", "/tmp/b.mol2", "/tmp/b.prmtop"}
    assert leadopt_needs_parameterization_build(rows_for_param, file_exists_fn=lambda p: str(p) in exists) is False
    exists.remove("/tmp/b.prmtop")
    assert leadopt_needs_parameterization_build(rows_for_param, file_exists_fn=lambda p: str(p) in exists) is True

    results_apply = [{"status": "ok", "mpo": 0.5, "notes": "dock"}]
    apply_ok = leadopt_apply_refinement_result(
        results=results_apply,
        idx=0,
        ref={"refine_status": "ok", "refine_dg_kj": -30.0, "refine_method": "pymbar.exp", "refine_samples": 12},
    )
    assert apply_ok["refined"] is True
    assert apply_ok["method"] == "pymbar.exp"
    assert np.isfinite(float(results_apply[0].get("mpo", np.nan)))
    assert "refine_dG=-30.00 kJ/mol" in str(results_apply[0].get("notes", ""))

    apply_fail = leadopt_apply_refinement_result(
        results=results_apply,
        idx=0,
        ref={"refine_status": "failed", "refine_reason": "too_few_valid_energy_samples"},
    )
    assert apply_fail["refined"] is False
    assert apply_fail["failure_reason"] == "too_few_valid_energy_samples"

    rep_no_success = leadopt_finalize_advanced_refinement_report(
        report={"attempted": 3, "refined": 0, "message": "", "method": ""},
        methods=[],
        fail_reasons=["a", "a", "b", "c"],
    )
    assert "no candidate produced a valid pymbar estimate" in rep_no_success["message"].lower()
    assert "a (n=2)" in rep_no_success["message"]

    rep_methods = leadopt_finalize_advanced_refinement_report(
        report={"attempted": 2, "refined": 1, "message": "", "method": ""},
        methods=["mbar", "exp", "mbar"],
        fail_reasons=[],
    )
    assert rep_methods["method"] == "exp,mbar"

    note_off = leadopt_optional_refine_note(False, {"attempted": 2, "refined": 1, "method": "exp"})
    assert note_off == ""
    note_msg = leadopt_optional_refine_note(True, {"message": "custom warning"})
    assert note_msg == "custom warning"
    note_none = leadopt_optional_refine_note(True, {"attempted": 0, "refined": 0, "method": ""})
    assert "no eligible leads" in note_none.lower()
    note_ok = leadopt_optional_refine_note(
        True,
        {"attempted": 3, "refined": 2, "method": "mbar", "cancelled": True},
    )
    assert "2/3" in note_ok and "[mbar]" in note_ok and "cancelled early" in note_ok

    cand_rows = [
        {
            "name": "L1",
            "source": "gen",
            "smiles": "CC",
            "admet": 0.6,
            "qed": 0.5,
            "sim_seed": 0.7,
            "mw": 250.0,
            "logp": 2.1,
            "hba": 2,
            "hbd": 1,
            "tpsa": 35.0,
            "rotb": 3,
            "pains": False,
            "filter_keep": True,
            "filter_reason": "pass",
            "mol": None,
        },
        {
            "name": "L2",
            "source": "gen",
            "smiles": "CCC",
            "admet": 0.4,
            "qed": 0.3,
            "sim_seed": 0.5,
            "mw": 320.0,
            "logp": 3.2,
            "hba": 3,
            "hbd": 1,
            "tpsa": 48.0,
            "rotb": 4,
            "pains": False,
            "filter_keep": False,
            "filter_reason": "mw",
            "mol": None,
        },
    ]
    docked_rows = [
        {"name": "L1", "status": "ok", "mpo": 0.8, "affinity": -8.5, "notes": ""},
        {"name": "L3", "status": "ok", "mpo": 0.6, "affinity": -7.1, "notes": ""},
    ]
    merged = leadopt_merge_candidate_and_docked_results(cand_rows, docked_rows)
    assert len(merged) == 3
    by_name = {str(x.get("name")): x for x in merged}
    assert by_name["L1"]["filter_keep"] is True
    assert by_name["L2"]["status"] == "filtered_out"
    assert by_name["L3"]["status"] == "ok"

    counts = leadopt_compute_result_counts(
        docked_results=docked_rows,
        refine_report={"attempted": 2, "refined": 1, "method": "exp"},
    )
    assert counts["n_total"] == 2
    assert counts["n_ok"] == 2
    assert counts["n_fail"] == 0
    assert counts["n_ref_attempted"] == 2
    assert counts["n_ref_ok"] == 1
    assert counts["ref_method"] == "exp"

    txt_best = leadopt_status_text({"name": "L1", "mpo": 0.8123, "affinity": -8.45}, n_ok=3, n_total=5)
    assert "Best: L1" in txt_best and "3/5" in txt_best
    txt_none = leadopt_status_text(None, n_ok=1, n_total=2)
    assert txt_none == "Status: done (1/2 scored)."

    built_rows = leadopt_build_candidate_rows(
        analogs=[
            {"name": "A", "mol": "mA", "smiles": "CC", "source": "seed"},
            {"name": "B", "mol": "mB", "smiles": "CCC", "source": "gen"},
        ],
        compute_props_fn=lambda mol: {"qed": 0.8 if str(mol) == "mA" else 0.4, "mw": 300.0},
        admet_score_fn=lambda props: float(props.get("qed", 0.0)),
    )
    assert len(built_rows) == 2
    assert float(built_rows[0]["admet"]) == 0.8
    assert float(built_rows[1]["admet"]) == 0.4
    assert float(built_rows[0]["mw"]) == 300.0

    rows_for_filter = [
        {"name": "A", "admet": 0.9, "qed": 0.8, "sim_seed": 0.6, "mol": "mA"},
        {"name": "B", "admet": 0.4, "qed": 0.5, "sim_seed": 0.2, "mol": "mB"},
        {"name": "C", "admet": 0.7, "qed": 0.6, "sim_seed": 0.4, "mol": "mC"},
    ]

    kept, rej = leadopt_filter_rows_with_limits(
        rows_for_filter,
        limits={"tag": "x"},
        filter_fail_reason_fn=lambda row, limits=None: "" if str((row or {}).get("name", "")) != "B" else "mw",
    )
    assert len(kept) == 2
    assert rej["mw"] == 1
    assert all(bool(k.get("filter_keep", False)) for k in kept)

    rescue, rescue_rej = leadopt_rescue_top_admet_rows(rows_for_filter)
    assert len(rescue) >= 1
    assert all(str(r.get("filter_reason", "")) == "rescue_top_admet" for r in rescue)
    assert "rescue" in rescue_rej

    diverse = leadopt_apply_diversity_rescue_rows(
        candidate_rows=rows_for_filter,
        filtered_rows=[{"name": "A", "filter_keep": True, "filter_reason": "pass", "mol": "mA", "admet": 0.9, "qed": 0.8, "sim_seed": 0.6}],
        has_rdkit=True,
        fingerprint_fn=lambda mol: str(mol),
        similarity_fn=lambda fp1, fp2: 0.9 if fp1 == fp2 else 0.2,
        min_dock_n=3,
    )
    assert len(diverse) == 3
    by_name_div = {str(x.get("name", "")): x for x in diverse}
    assert by_name_div["A"]["filter_reason"] == "pass"
    assert by_name_div["B"]["filter_reason"] == "diversity_rescue_dock"
    assert by_name_div["C"]["filter_reason"] == "diversity_rescue_dock"

    full_filter = leadopt_filter_and_rescue_candidates(
        candidate_rows=rows_for_filter,
        base_limits={"tag": "x"},
        filter_fail_reason_fn=lambda row, limits=None: "mw",
        relax_limits_fn=lambda base, level=1: {"lvl": int(level)},
        relax_levels=(1, 2, 3),
        has_rdkit=False,
        fingerprint_fn=None,
        similarity_fn=None,
    )
    assert len(full_filter["filtered"]) >= 1
    assert "rescue_top_admet" in str(full_filter["filter_mode"])

    paths = leadopt_prepare_job_paths(
        workspace_dir=str(tmp_path),
        receptor_obj="Rec 1",
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        tag="20260303_010203",
    )
    assert paths["tag"] == "20260303_010203"
    assert os.path.isdir(paths["job_dir"])
    assert os.path.isdir(paths["lig_dir"])
    assert os.path.isdir(paths["pose_dir"])
    assert "leadopt_Rec_1_20260303_010203" in paths["job_dir"]

    csv_payload = leadopt_export_csv_outputs(
        job_dir=paths["job_dir"],
        candidate_rows=cand_rows,
        docked_results=docked_rows,
        full_results=merged,
    )
    assert os.path.isfile(csv_payload["csv_candidates"])
    assert os.path.isfile(csv_payload["csv_docked"])
    assert os.path.isfile(csv_payload["csv_all"])
    with open(csv_payload["csv_candidates"], "r", encoding="utf-8") as fh:
        hdr = fh.readline().strip()
    assert "name,source,smiles,filter_keep,filter_reason" in hdr
    with open(csv_payload["csv_all"], "r", encoding="utf-8") as fh:
        body = fh.read()
    assert "filtered_out" in body


def test_smina_runtime_selects_best_repeat(tmp_path):
    cmd_calls = {"i": 0}
    pose_to_aff = {}

    def _run_command(argv, cwd=None, timeout=0):
        cmd_calls["i"] += 1
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        rep = cmd_calls["i"]
        aff = {-1: -7.0, 1: -7.0, 2: -9.1, 3: -8.2}.get(rep, -7.0)
        pose_to_aff[out_pose] = aff
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        return True, "", "", 0

    def _parse_pose_scores(out_pose, _out_log):
        aff = pose_to_aff[out_pose]
        return [{"mode": 1, "affinity": aff, "rmsd_lb": 0.0, "rmsd_ub": 0.0}]

    payload = run_smina_repeats(
        base_cmd=["smina", "-r", "r.pdbqt", "-l", "l.pdbqt"],
        job_dir=str(tmp_path),
        n_repeats=3,
        seed0=100,
        run_command=_run_command,
        parse_pose_scores=_parse_pose_scores,
        timeout=10,
    )

    assert len(payload["repeat_rows"]) == 3
    assert int(payload["best_repeat"]["repeat"]) == 2
    assert float(payload["best_repeat"]["best_affinity"]) < -9.0


def test_smina_runtime_parse_smina_minimized_affinity_pose_file(tmp_path):
    pose_file = tmp_path / "poses.pdbqt"
    pose_file.write_text(
        "\n".join(
            [
                "MODEL 1",
                "REMARK minimizedAffinity -5.80438757",
                "REMARK minimizedRMSD -1",
                "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C",
                "ENDMDL",
                "MODEL 2",
                "REMARK minimizedAffinity -5.12411165",
                "REMARK minimizedRMSD -1",
                "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00    +0.000 C",
                "ENDMDL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    poses = parse_smina_pose_scores(str(pose_file))
    assert len(poses) == 2
    assert int(poses[0]["mode"]) == 1
    assert float(poses[0]["affinity"]) == -5.80438757
    assert np.isnan(float(poses[0]["rmsd_lb"]))
    assert int(poses[1]["mode"]) == 2
    assert float(poses[1]["affinity"]) == -5.12411165


def test_smina_runtime_prefers_log_table_rmsd_over_minimized_rmsd_sentinel(tmp_path):
    pose_file = tmp_path / "poses.pdbqt"
    pose_file.write_text(
        "\n".join(
            [
                "MODEL 1",
                "REMARK minimizedAffinity -5.80438757",
                "REMARK minimizedRMSD -1",
                "ENDMDL",
                "MODEL 2",
                "REMARK minimizedAffinity -5.12411165",
                "REMARK minimizedRMSD -1",
                "ENDMDL",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log_file = tmp_path / "docking.log"
    log_file.write_text(
        "\n".join(
            [
                "-----+------------+----------+----------",
                "    1       -5.8      0.000      0.000",
                "    2       -5.1      1.250      2.500",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    poses = parse_smina_pose_scores(str(pose_file), str(log_file))

    assert len(poses) == 2
    assert float(poses[0]["rmsd_lb"]) == 0.0
    assert float(poses[0]["rmsd_ub"]) == 0.0
    assert float(poses[1]["rmsd_lb"]) == 1.25
    assert float(poses[1]["rmsd_ub"]) == 2.5


def test_smina_runtime_retries_flex_autobox_when_search_space_is_too_small(tmp_path):
    calls = []

    def _run_command(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        calls.append(list(argv))
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        autobox_add = float(argv[argv.index("--autobox_add") + 1])
        if len(calls) == 1:
            with open(out_pose, "w", encoding="utf-8") as fh:
                fh.write("MODEL 1\nENDMDL\n")
            with open(out_log, "w", encoding="utf-8") as fh:
                fh.write(
                    "WARNING: Could not find any conformations completely within the search space.\n"
                    "WARNING: Check that it is large enough for all movable atoms, including those in the flexible side chains.\n"
                )
            return True, "", "", 0
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("MODEL 1\nREMARK minimizedAffinity -6.5\nREMARK minimizedRMSD -1\nENDMDL\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write(f"Using autobox_add {autobox_add:.3f}\n")
        return True, "", "", 0

    payload = run_smina_repeats(
        base_cmd=[
            "smina",
            "-r",
            "rec.pdbqt",
            "-l",
            "lig.pdbqt",
            "--autobox_ligand",
            "ref.pdb",
            "--autobox_add",
            "4.000",
            "--flexres",
            "A:16",
        ],
        job_dir=str(tmp_path),
        n_repeats=1,
        seed0=0,
        run_command=_run_command,
        parse_pose_scores=parse_smina_pose_scores,
        timeout=10,
    )

    assert payload["best_repeat"] is not None
    assert float(payload["best_repeat"]["best_affinity"]) == -6.5
    assert len(calls) == 2
    assert float(calls[0][calls[0].index("--autobox_add") + 1]) == 4.0
    assert float(calls[1][calls[1].index("--autobox_add") + 1]) == 10.0


def test_redock_runtime_finalize_member_and_build_run_map():
    raw = {
        "ok": True,
        "rows": [{"mode": 1, "affinity": -7.5}],
    }
    finalized = finalize_redock_member_run(
        raw,
        system_label="WT",
        member_tag="WT_model_001",
        ensemble_mode=True,
    )
    assert finalized["ensemble_member"] == "WT_model_001"
    assert finalized["ensemble_mode"] is True
    assert finalized["rows"][0]["system"] == "WT"
    assert finalized["rows"][0]["system_member"] == "WT_model_001"
    assert "system" not in raw["rows"][0]

    calls = []

    def _aggregate(label, runs):
        calls.append((str(label), int(len(runs))))
        return {"ok": True, "rows": [{"system": str(label), "n": int(len(runs))}]}

    run_map = build_redock_run_map(
        ordered_labels=["WT", "Mutant"],
        run_groups={
            "WT": [{"ok": True, "rows": [{"system": "WT"}]}],
            "Mutant": [{"ok": True, "rows": []}, {"ok": True, "rows": []}],
        },
        aggregate_label_runs=_aggregate,
    )
    assert "WT" in run_map and "Mutant" in run_map
    assert calls == [("Mutant", 2)]
    assert int(run_map["Mutant"]["rows"][0]["n"]) == 2

    bad = ""
    try:
        _ = build_redock_run_map(
            ordered_labels=["Mutant"],
            run_groups={"Mutant": [{"ok": True}, {"ok": True}]},
            aggregate_label_runs=lambda _label, _runs: {"ok": False, "error": "bad"},
        )
    except RuntimeError as e:
        bad = str(e)
    assert "aggregation failed: bad" in bad


def test_redock_runtime_run_map_and_post_pipeline_helpers():
    payload_ok = build_redocking_run_map_payload(
        systems=[("WT", "wt_rec"), ("Mutant", "mut_rec")],
        run_groups={
            "WT": [{"ok": True, "rows": [{"system": "WT", "affinity": -8.1}]}],
            "Mutant": [{"ok": True, "rows": [{"system": "Mutant", "affinity": -7.1}]}],
        },
        aggregate_label_runs_fn=lambda _label, runs: {"ok": True, "rows": list(runs[0].get("rows", []))},
    )
    assert payload_ok["ok"] is True
    assert set(payload_ok["run_map"].keys()) == {"WT", "Mutant"}

    payload_fail = build_redocking_run_map_payload(
        systems=[("Mutant", "mut_rec")],
        run_groups={"Mutant": [{"ok": True}, {"ok": True}]},
        aggregate_label_runs_fn=lambda _label, _runs: {"ok": False, "error": "agg bad"},
    )
    assert payload_fail["ok"] is False
    assert "agg bad" in str(payload_fail["error"])

    calls = []

    def _builder(**kwargs):
        calls.append(dict(kwargs or {}))
        return {"ok": True, "run_map": {"WT": {"rows": [{"system": "WT"}]}}}

    def _aggregate(_label, runs):
        return {"ok": True, "rows": list(runs[0].get("rows", []))}

    runtime_ok = build_redocking_run_map_runtime(
        systems=[("WT", "wt_rec")],
        run_groups={"WT": [{"ok": True, "rows": [{"system": "WT"}]}]},
        aggregate_label_runs_fn=_aggregate,
        run_map_payload_builder_fn=_builder,
    )
    assert runtime_ok["ok"] is True
    assert "WT" in runtime_ok["run_map"]
    assert int(len(calls)) == 1
    sent = dict(calls[0] or {})
    assert sent["systems"] == [("WT", "wt_rec")]
    assert sent["run_groups"] == {"WT": [{"ok": True, "rows": [{"system": "WT"}]}]}
    assert sent["aggregate_label_runs_fn"] is _aggregate

    runtime_fail = build_redocking_run_map_runtime(
        systems=[("WT", "wt_rec")],
        run_groups={"WT": []},
        aggregate_label_runs_fn=_aggregate,
        run_map_payload_builder_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert runtime_fail["ok"] is False
    assert str(runtime_fail["error"]) == "boom"
    assert runtime_fail["run_map"] == {}

    rows = collect_redocking_rows_from_run_map(
        run_map={
            "WT": {"rows": [{"system": "WT", "mode": 1}]},
            "Mutant": {"rows": [{"system": "Mutant", "mode": 2}]},
            "Other": {"rows": [{"system": "Other", "mode": 3}]},
        },
        preferred_labels=["WT", "Mutant"],
    )
    assert int(len(rows)) == 3
    assert str(rows[0]["system"]) == "WT"
    assert str(rows[1]["system"]) == "Mutant"
    assert str(rows[2]["system"]) == "Other"

    summary = build_redocking_summary_payload(
        run_map={"WT": {"rows": []}, "Mutant": {"rows": []}},
        nref=12,
        center=(1.0, 2.0, 3.0),
        size=(18.0, 18.0, 18.0),
        use_reference_ligand=True,
        use_modeler_ensemble=False,
        ensemble_counts={"WT": 1, "Mutant": 1},
        repeats=2,
        summary_lines_fn=lambda *_a, **_k: ["line_a", "line_b"],
    )
    assert summary["summary_lines"] == ["line_a", "line_b"]
    assert summary["summary_text"] == "line_a\nline_b"
    assert int(len(summary["comparison_rows"])) == 1
    assert str(summary["summary_csv"]) == ""

    transform_payload = build_redocking_post_transform_runtime(
        context={
            "run_map": {
                "WT": {"rows": [{"system": "WT", "mode": 1}]},
                "Mutant": {"rows": [{"system": "Mutant", "mode": 2}]},
                "Other": {"rows": [{"system": "Other", "mode": 3}]},
            }
        },
        preferred_labels=["WT", "Mutant"],
    )
    assert int(len(transform_payload["results"])) == 3
    assert str(transform_payload["results"][0]["system"]) == "WT"
    assert str(transform_payload["results"][1]["system"]) == "Mutant"
    assert str(transform_payload["results"][2]["system"]) == "Other"

    seen = []

    def _summary_builder(**kwargs):
        seen.append(dict(kwargs or {}))
        return {"summary_lines": ["s1", "s2"], "summary_text": "s1\ns2"}

    post_summary = build_redocking_post_summary_runtime(
        context={
            "run_map": {"WT": {"rows": []}},
            "nref": 10,
            "center": (1.0, 2.0, 3.0),
            "size": (18.0, 18.0, 18.0),
            "use_reference_ligand": True,
            "use_modeler_ensemble": True,
            "ensemble_counts": {"WT": 2},
            "repeats": 4,
            "summary_csv_path": "",
        },
        summary_builder_fn=_summary_builder,
    )
    assert post_summary == {
        "summary": {
            "summary_lines": ["s1", "s2"],
            "summary_text": "s1\ns2",
            "comparison_rows": [],
            "summary_csv": "",
        }
    }
    assert int(len(seen)) == 1
    sent = dict(seen[0] or {})
    assert sent["run_map"] == {"WT": {"rows": []}}
    assert int(sent["nref"]) == 10
    assert sent["center"] == (1.0, 2.0, 3.0)
    assert sent["size"] == (18.0, 18.0, 18.0)
    assert bool(sent["use_reference_ligand"]) is True
    assert bool(sent["use_modeler_ensemble"]) is True
    assert sent["ensemble_counts"] == {"WT": 2}
    assert int(sent["repeats"]) == 4
    assert str(sent["summary_csv_path"]) == ""

    apply_payload = build_redocking_post_apply_state_runtime(
        context={"run_map": {"WT": {"rows": [{"system": "WT"}]}}},
        results=[{"system": "WT", "mode": 1}],
    )
    assert apply_payload["run_map"] == {"WT": {"rows": [{"system": "WT"}]}}
    assert apply_payload["results"] == [{"system": "WT", "mode": 1}]

    dialog_payload = build_redocking_post_dialog_runtime(
        summary={"summary_text": "done", "summary_lines": ["done"]},
    )
    assert dialog_payload == {"summary_text": "done", "summary_csv": ""}
    dialog_payload_empty = build_redocking_post_dialog_runtime(summary=None)
    assert dialog_payload_empty == {"summary_text": "", "summary_csv": ""}

    ctx = build_redocking_post_pipeline_context(
        run_map={"WT": {"rows": []}},
        nref=8,
        center=(0.1, 0.2, 0.3),
        size=(18.0, 19.0, 20.0),
        use_reference_ligand=False,
        use_modeler_ensemble=True,
        ensemble_counts={"WT": 5},
        repeats=1,
    )
    assert int(ctx["nref"]) == 8
    assert tuple(ctx["center"]) == (0.1, 0.2, 0.3)
    assert tuple(ctx["size"]) == (18.0, 19.0, 20.0)
    assert ctx["use_modeler_ensemble"] is True
    assert int(ctx["ensemble_counts"]["WT"]) == 5
    assert int(ctx["repeats"]) == 1
    assert ctx["system_labels"] == []
    assert str(ctx["summary_csv_path"]) == ""


def test_redock_runtime_run_post_status_runtime_helper():
    calls = []
    debug_lines = []

    def _build_combo():
        calls.append("build_combo")
        raise RuntimeError("combo boom")

    def _hide_pose():
        calls.append("hide_pose")
        raise RuntimeError("hide boom")

    def _hide_receptor():
        calls.append("hide_receptor")

    def _clear_preview():
        calls.append("clear_preview")

    def _update_label():
        calls.append("update_label")

    def _traj_vis():
        calls.append("traj_vis")

    def _load_pose(silent=False):
        calls.append(("load_pose", bool(silent)))
        raise RuntimeError("load boom")

    payload = run_redocking_post_status_runtime(
        build_bestpose_combo_trajectories_fn=_build_combo,
        hide_all_pose_objects_fn=_hide_pose,
        hide_all_runtime_receptors_fn=_hide_receptor,
        clear_selected_preview_objects_fn=_clear_preview,
        update_ligand_status_label_fn=_update_label,
        on_traj_visibility_changed_fn=_traj_vis,
        load_selected_pose_fn=_load_pose,
        append_debug_log_fn=lambda msg: debug_lines.append(str(msg)),
    )
    assert payload == {}
    assert calls == [
        "build_combo",
        "hide_pose",
        "hide_receptor",
        "clear_preview",
        "update_label",
        "traj_vis",
    ]
    assert any("Redocking combo-trajectory warning: combo boom" in str(m) for m in debug_lines)

    calls_autoload = []

    def _load_pose_noarg():
        calls_autoload.append("load_noarg")

    payload_noarg = run_redocking_post_status_runtime(
        load_selected_pose_fn=_load_pose_noarg,
        auto_load_selected_pose=True,
    )
    assert payload_noarg == {}
    assert calls_autoload == ["load_noarg"]

    calls_default = []

    payload_default = run_redocking_post_status_runtime(
        load_selected_pose_fn=lambda silent=False: calls_default.append(("load_default", bool(silent))),
    )
    assert payload_default == {}
    assert calls_default == []

    err = ""
    try:
        run_redocking_post_status_runtime(
            update_ligand_status_label_fn=lambda: (_ for _ in ()).throw(RuntimeError("status boom")),
        )
    except RuntimeError as e:
        err = str(e)
    assert "status boom" in err


def test_redock_runtime_rows_and_state_helpers():
    rows = [
        {"system": "WT", "system_member": "WT_model_002", "rmsd_ref": 2.4, "affinity": -8.0},
        {"system": "WT", "system_member": "WT_model_002", "rmsd_ref": 1.2, "affinity": -7.0},
        {"system": "Mutant", "system_member": "", "rmsd_ref": float("nan"), "affinity": -8.1},
        {"system": "Mutant", "system_member": "", "rmsd_ref": float("nan"), "affinity": -8.9},
    ]
    best_rows = rows_best_per_member(rows)
    assert int(len(best_rows)) == 2
    assert str(best_rows[0]["system"]) == "WT"
    assert str(best_rows[0]["system_member"]) == "WT_model_002"
    # the top-scored pose, not the one closest to the crystal
    assert float(best_rows[0]["rmsd_ref"]) == 2.4 and float(best_rows[0]["affinity"]) == -8.0
    assert str(best_rows[1]["system"]) == "Mutant"
    assert str(best_rows[1]["system_member"]) == "Mutant_model_001"
    assert float(best_rows[1]["affinity"]) == -8.9

    assert int(resolve_state_for_member({"WT_model_002": 5}, "WT_model_002", default_state=1)) == 5
    assert int(resolve_state_for_member({"wt_model_002": 6}, "WT_MODEL_002", default_state=1)) == 6
    assert int(resolve_state_for_member({}, "Mutant_model_014", default_state=1)) == 14
    assert int(resolve_state_for_member({}, "", default_state=3)) == 3


def test_redock_runtime_best_pose_for_member_helper():
    rows = [
        {"system": "WT", "system_member": "WT_model_001", "rmsd_ref": 2.0, "affinity": -8.0},
        {"system": "WT", "system_member": "WT_model_001", "rmsd_ref": 1.2, "affinity": -7.6},
        {"system": "WT", "system_member": "WT_model_002", "rmsd_ref": float("nan"), "affinity": -9.1},
        {"system": "Mutant", "system_member": "Mut_model_001", "rmsd_ref": 1.5, "affinity": -7.5},
    ]

    best_exact = best_pose_for_member(rows, "WT", "WT_model_001")
    assert isinstance(best_exact, dict)
    assert float(best_exact["rmsd_ref"]) == 2.0            # top-scored pose of the member
    assert float(best_exact["affinity"]) == -8.0

    best_fallback = best_pose_for_member(rows, "WT", "WT_model_999")
    assert isinstance(best_fallback, dict)
    assert float(best_fallback["affinity"]) == -9.1         # top-scored WT pose

    best_aff_only = best_pose_for_member(rows, "WT", "WT_model_002")
    assert isinstance(best_aff_only, dict)
    assert float(best_aff_only["affinity"]) == -9.1

    assert best_pose_for_member(rows, "Other", "") is None


def test_redock_runtime_build_state_row_map_helper():
    rows = [
        {"system": "WT", "system_member": "WT_model_001", "traj_state": 2, "rmsd_ref": 2.0, "affinity": -8.0},
        {"system": "WT", "system_member": "WT_model_001", "traj_state": 2, "rmsd_ref": 1.1, "affinity": -7.4},
        {"system": "WT", "system_member": "WT_model_002", "traj_state": 0, "rmsd_ref": 1.5, "affinity": -8.7},
        {"system": "Mutant", "system_member": "Mut_model_001", "traj_state": 1, "rmsd_ref": 1.3, "affinity": -7.5},
    ]

    payload = build_redocking_state_row_map(
        rows=rows,
        label="WT",
        state_map={"WT_model_002": 1, "WT_model_003": 3},
        n_states=4,
    )
    assert int(len(payload["rows_label"])) == 3
    fallback = dict(payload["fallback"] or {})
    assert float(fallback["affinity"]) == -8.7              # top-scored WT pose

    row_by_state = dict(payload["row_by_state"] or {})
    assert set(row_by_state.keys()) == {1, 2, 3, 4}
    assert float(row_by_state[2]["rmsd_ref"]) == 2.0        # top-scored pose of state 2
    assert str(row_by_state[1]["system_member"]) == "WT_model_002"
    assert float(row_by_state[3]["affinity"]) == -8.7
    assert float(row_by_state[4]["affinity"]) == -8.7

    payload_empty = build_redocking_state_row_map(
        rows=rows,
        label="Other",
        state_map={},
        n_states=3,
    )
    assert payload_empty["rows_label"] == []
    assert payload_empty["fallback"] is None
    assert payload_empty["row_by_state"] == {}

    callback_calls = []

    def _best_for_member(all_rows, system_label, member_tag):
        callback_calls.append((int(len(list(all_rows or []))), str(system_label), str(member_tag)))
        if str(system_label) == "WT" and str(member_tag) == "WT_model_003":
            return {"system": "WT", "system_member": "WT_model_003", "traj_state": 0, "rmsd_ref": 0.9, "affinity": -9.2}
        return None

    payload_cb = build_redocking_state_row_map(
        rows=rows,
        label="WT",
        state_map={"WT_model_003": 3},
        n_states=3,
        best_pose_for_member_fn=_best_for_member,
    )
    row_by_state_cb = dict(payload_cb.get("row_by_state", {}) or {})
    assert set(row_by_state_cb.keys()) == {1, 2, 3}
    assert str(row_by_state_cb[3]["system_member"]) == "WT_model_003"
    assert float(row_by_state_cb[3]["rmsd_ref"]) == 0.9
    assert callback_calls == [(4, "WT", "WT_model_003")]


def test_redock_runtime_build_state_row_map_can_avoid_fallback_pose_duplication():
    rows = [
        {
            "system": "Mutant",
            "system_member": "Mutant_model_001",
            "traj_state": 1,
            "rmsd_ref": 0.8,
            "affinity": -7.1,
            "pose_object": "pose_mut_001",
            "state": 6,
        },
        {
            "system": "Mutant",
            "system_member": "Mutant_model_001",
            "traj_state": 1,
            "rmsd_ref": 1.4,
            "affinity": -7.9,
            "pose_object": "pose_mut_001",
            "state": 2,
        },
    ]

    payload = build_redocking_state_row_map(
        rows=rows,
        label="Mutant",
        state_map={"Mutant_model_001": 1, "Mutant_model_002": 2, "Mutant_model_003": 3},
        n_states=3,
        fill_missing_states_with_fallback=False,
        synthesize_missing_state_rows=True,
    )

    fallback = dict(payload.get("fallback") or {})
    assert str(fallback.get("pose_object", "")) == "pose_mut_001"

    row_by_state = dict(payload.get("row_by_state", {}) or {})
    assert set(row_by_state.keys()) == {1, 2, 3}
    assert str(row_by_state[1]["pose_object"]) == "pose_mut_001"
    assert int(row_by_state[1]["state"]) == 2            # top-scored pose (-7.9) of the member
    assert str(row_by_state[2]["pose_object"]) == ""
    assert str(row_by_state[2]["system_member"]) == "Mutant_model_002"
    assert str(row_by_state[3]["pose_object"]) == ""
    assert str(row_by_state[3]["system_member"]) == "Mutant_model_003"


def test_redock_runtime_build_table_rows_helper():
    rows = [
        {"system": "WT", "system_member": "WT_model_002", "rmsd_ref": 2.4, "affinity": -8.0, "mode": 2},
        {"system": "WT", "system_member": "WT_model_002", "rmsd_ref": 1.2, "affinity": -7.0, "mode": 1},
        {"system": "Mutant", "system_member": "Mutant_model_001", "rmsd_ref": float("nan"), "affinity": -8.9, "mode": 1},
    ]
    views = {
        "WT": {"state_map": {"WT_model_002": 5}, "n_states": 4},
        "Mutant": {"state_map": {"Mutant_model_001": 1}, "n_states": 2},
    }
    out = build_redocking_table_rows(rows, views=views)
    assert int(len(out)) == 2

    wt = dict(out[0])
    mut = dict(out[1])
    assert str(wt["system"]) == "WT"
    assert int(wt["rank"]) == 1
    assert str(wt["system_display"]) == "WT [WT_model_002]"
    assert int(wt["traj_state"]) == 4  # clamped from 5 to n_states=4

    assert str(mut["system"]) == "Mutant"
    assert int(mut["rank"]) == 1
    assert str(mut["system_display"]) == "Mutant"
    assert int(mut["traj_state"]) == 1

    # Default tags should not expand display text.
    out2 = build_redocking_table_rows(
        rows=[{"system": "WT", "system_member": "WT_model_001", "rmsd_ref": 1.1, "affinity": -8.1}],
        views={"WT": {"state_map": {}, "n_states": 0}},
    )
    assert int(len(out2)) == 1
    assert str(out2[0]["system_display"]) == "WT"
    assert int(out2[0]["traj_state"]) == 1


def test_redock_runtime_traj_visibility_and_state_helpers():
    assert redocking_traj_visible_for_label("WT", show_wt=True, show_mut=False) is True
    assert redocking_traj_visible_for_label("WT_model", show_wt=False, show_mut=True) is False
    assert redocking_traj_visible_for_label("Mutant", show_wt=True, show_mut=False) is False
    assert redocking_traj_visible_for_label("Mut", show_wt=False, show_mut=True) is True
    assert redocking_traj_visible_for_label("Other", show_wt=False, show_mut=False, default_visible=True) is True

    st1 = resolve_redocking_traj_state(
        explicit_state=0,
        state_map={"WT_model_002": 5},
        member_tag="WT_model_002",
        n_states=4,
    )
    assert int(st1) == 4

    st2 = resolve_redocking_traj_state(
        explicit_state=3,
        state_map={"WT_model_002": 1},
        member_tag="WT_model_002",
        n_states=2,
    )
    assert int(st2) == 2

    st3 = resolve_redocking_traj_state(
        explicit_state=0,
        state_map={},
        member_tag="",
        n_states=0,
        default_state=7,
    )
    assert int(st3) == 7

    views = {
        "WT": {
            "traj_object": "wt_traj",
            "frame_object": "wt_frame",
            "pose_frame_object": "wt_pose",
            "combo_object": "wt_combo",
        },
        "Mutant": {
            "traj_object": "mut_traj",
            "frame_object": "mut_frame",
            "pose_frame_object": "mut_pose",
            "combo_object": "",
            "n_states": 3,
        },
    }
    known = {"wt_traj", "wt_frame", "wt_pose", "wt_combo", "mut_traj", "mut_frame", "mut_pose"}
    actions = build_redocking_traj_visibility_actions(
        views=views,
        known_objects=known,
        show_wt=True,
        show_mut=False,
    )
    action_tuples = [(str(a.get("action")), str(a.get("object"))) for a in list(actions or [])]
    assert ("disable", "wt_traj") in action_tuples
    assert ("enable", "wt_combo") in action_tuples
    assert ("disable", "wt_frame") in action_tuples
    assert ("disable", "wt_pose") in action_tuples
    assert ("disable", "mut_traj") in action_tuples
    assert ("disable", "mut_frame") in action_tuples
    assert ("disable", "mut_pose") in action_tuples

    actions2 = build_redocking_traj_visibility_actions(
        views=views,
        known_objects=known,
        visible_for_label_fn=lambda lbl: str(lbl).lower().startswith("mut"),
    )
    action_tuples2 = [(str(a.get("action")), str(a.get("object"))) for a in list(actions2 or [])]
    assert ("disable", "wt_combo") in action_tuples2
    assert ("enable", "mut_traj") in action_tuples2
    assert ("disable", "mut_frame") in action_tuples2
    assert ("enable", "mut_pose") in action_tuples2


def test_redock_runtime_traj_visibility_runtime_helper():
    build_calls = []
    apply_calls = []

    payload = run_redocking_traj_visibility_runtime(
        views={
            "WT": {
                "traj_object": "wt_traj",
                "frame_object": "wt_frame",
                "pose_frame_object": "wt_pose",
                "combo_object": "wt_combo",
            }
        },
        get_known_objects_fn=lambda: {"wt_traj", "wt_frame", "wt_pose", "wt_combo"},
        visible_for_label_fn=lambda lbl: str(lbl) == "WT",
        build_traj_visibility_actions_fn=lambda **kwargs: build_calls.append(dict(kwargs or {}))
        or [{"action": "enable", "object": "wt_combo"}],
        apply_object_actions_fn=lambda actions, known_objects=None: apply_calls.append(
            {
                "actions": list(actions or []),
                "known_objects": set(known_objects or set()),
            }
        ),
    )
    assert payload["ok"] is True
    assert payload["error"] == ""
    assert payload["actions"] == [{"action": "enable", "object": "wt_combo"}]
    assert payload["actions_applied"] is True
    assert payload["known_objects"] == {"wt_traj", "wt_frame", "wt_pose", "wt_combo"}
    assert int(len(build_calls)) == 1
    assert build_calls[0]["visible_for_label_fn"] is not None
    assert int(len(apply_calls)) == 1
    assert apply_calls[0]["actions"] == payload["actions"]
    assert apply_calls[0]["known_objects"] == payload["known_objects"]

    payload_known_fallback = run_redocking_traj_visibility_runtime(
        views={},
        known_objects={"fallback_obj"},
        get_known_objects_fn=lambda: (_ for _ in ()).throw(RuntimeError("known boom")),
    )
    assert payload_known_fallback["ok"] is True
    assert payload_known_fallback["known_objects"] == {"fallback_obj"}
    assert payload_known_fallback["actions"] == []
    assert payload_known_fallback["actions_applied"] is False

    payload_build_fail = run_redocking_traj_visibility_runtime(
        views={},
        known_objects={"obj_a"},
        build_traj_visibility_actions_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("build boom")),
    )
    assert payload_build_fail["ok"] is False
    assert payload_build_fail["actions"] == []
    assert payload_build_fail["actions_applied"] is False
    assert "build boom" in str(payload_build_fail["error"])

    payload_apply_fail = run_redocking_traj_visibility_runtime(
        views={},
        known_objects={"obj_a"},
        build_traj_visibility_actions_fn=lambda **_kwargs: [{"action": "disable", "object": "obj_a"}],
        apply_object_actions_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("apply boom")),
    )
    assert payload_apply_fail["ok"] is False
    assert payload_apply_fail["actions"] == [{"action": "disable", "object": "obj_a"}]
    assert payload_apply_fail["actions_applied"] is False
    assert "apply boom" in str(payload_apply_fail["error"])


def test_redock_runtime_row_object_actions_helper():
    rows = [
        {"pose_object": "pose_b", "receptor": "rec_2"},
        {"pose_object": "pose_a", "receptor": "rec_1"},
        {"pose_object": "pose_b", "receptor": "rec_1"},
        {"pose_object": "", "receptor": ""},
        {"receptor": "rec_3"},
        "bad_row_type",
    ]

    actions_pose = build_redocking_row_object_actions(
        rows,
        object_key="pose_object",
        action="DISABLE",
        known_objects={"pose_a", "pose_b", "pose_c"},
    )
    assert actions_pose == [
        {"action": "disable", "object": "pose_a", "source_key": "pose_object"},
        {"action": "disable", "object": "pose_b", "source_key": "pose_object"},
    ]

    actions_rec = build_redocking_row_object_actions(
        rows,
        object_key="receptor",
        action="enable",
        known_objects=None,
    )
    assert actions_rec == [
        {"action": "enable", "object": "rec_1", "source_key": "receptor"},
        {"action": "enable", "object": "rec_2", "source_key": "receptor"},
        {"action": "enable", "object": "rec_3", "source_key": "receptor"},
    ]

    actions_none = build_redocking_row_object_actions(
        rows,
        object_key="pose_object",
        action="disable",
        known_objects=set(),
    )
    assert actions_none == []

    actions_missing_key = build_redocking_row_object_actions(
        rows,
        object_key="",
        action="disable",
        known_objects=None,
    )
    assert actions_missing_key == []


def test_redock_runtime_row_object_actions_runtime_helper():
    rows = [
        {"pose_object": "pose_b", "receptor": "rec_2"},
        {"pose_object": "pose_a", "receptor": "rec_1"},
        {"pose_object": "pose_b", "receptor": "rec_1"},
    ]
    apply_calls = []

    payload = run_redocking_row_object_actions_runtime(
        rows=rows,
        object_key="pose_object",
        action="disable",
        get_known_objects_fn=lambda: {"pose_a", "pose_b", "rec_1"},
        apply_object_actions_fn=lambda actions, known_objects=None: apply_calls.append(
            {
                "actions": list(actions or []),
                "known_objects": set(known_objects or set()),
            }
        ),
    )
    assert payload["ok"] is True
    assert payload["reason"] == ""
    assert payload["error"] == ""
    assert payload["actions_applied"] is True
    assert payload["known_objects"] == {"pose_a", "pose_b", "rec_1"}
    assert payload["actions"] == [
        {"action": "disable", "object": "pose_a", "source_key": "pose_object"},
        {"action": "disable", "object": "pose_b", "source_key": "pose_object"},
    ]
    assert int(len(apply_calls)) == 1
    assert apply_calls[0]["actions"] == payload["actions"]
    assert apply_calls[0]["known_objects"] == payload["known_objects"]

    payload_known_fallback = run_redocking_row_object_actions_runtime(
        rows=rows,
        object_key="receptor",
        action="disable",
        known_objects={"rec_1"},
        get_known_objects_fn=lambda: (_ for _ in ()).throw(RuntimeError("known boom")),
    )
    assert payload_known_fallback["ok"] is True
    assert payload_known_fallback["known_objects"] == {"rec_1"}
    assert payload_known_fallback["actions"] == [
        {"action": "disable", "object": "rec_1", "source_key": "receptor"},
    ]

    payload_build_error = run_redocking_row_object_actions_runtime(
        rows=rows,
        object_key="pose_object",
        action="disable",
        build_row_object_actions_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("build boom")),
    )
    assert payload_build_error["ok"] is False
    assert payload_build_error["reason"] == "build_error"
    assert "build boom" in str(payload_build_error["error"])
    assert payload_build_error["actions_applied"] is False

    payload_apply_error = run_redocking_row_object_actions_runtime(
        rows=rows,
        object_key="pose_object",
        action="disable",
        known_objects={"pose_a", "pose_b"},
        apply_object_actions_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("apply boom")),
    )
    assert payload_apply_error["ok"] is False
    assert payload_apply_error["reason"] == "apply_error"
    assert "apply boom" in str(payload_apply_error["error"])
    assert payload_apply_error["actions"] == [
        {"action": "disable", "object": "pose_a", "source_key": "pose_object"},
        {"action": "disable", "object": "pose_b", "source_key": "pose_object"},
    ]


def test_redock_runtime_input_actions_helper():
    systems = [
        ("WT", "wt_obj", "wt_src", "WT_model_001"),
        ("Mutant", "mut_obj", "mut_src", "Mutant_model_001"),
        {"label": "WT", "object": "wt_obj_alt", "member_tag": "WT_model_002"},
        {"system": "Mutant", "receptor": "mut_obj_alt"},
        ("WT", "", "", ""),
        "bad_entry",
    ]

    actions = build_redocking_runtime_input_actions(
        systems,
        action="DISABLE",
        known_objects={"wt_obj", "mut_obj", "wt_obj_alt", "mut_obj_alt"},
    )
    assert actions == [
        {"action": "disable", "object": "wt_obj", "label": "WT", "member_tag": "WT_model_001"},
        {"action": "disable", "object": "mut_obj", "label": "Mutant", "member_tag": "Mutant_model_001"},
        {"action": "disable", "object": "wt_obj_alt", "label": "WT", "member_tag": "WT_model_002"},
        {"action": "disable", "object": "mut_obj_alt", "label": "Mutant", "member_tag": "Mutant_model_001"},
    ]

    actions_filtered = build_redocking_runtime_input_actions(
        systems,
        action="invalid_action",
        known_objects={"mut_obj"},
    )
    assert actions_filtered == [
        {"action": "disable", "object": "mut_obj", "label": "Mutant", "member_tag": "Mutant_model_001"},
    ]


def test_redock_runtime_prepare_runtime_views_runtime_helper():
    build_calls = []
    apply_calls = []
    debug_lines = []

    payload = prepare_redocking_runtime_views_runtime(
        runtime_systems=[("WT", "wt_traj_obj", "wt_src", "WT_model_001")],
        input_systems=[("WT", "wt_input_obj", "wt_src", "WT_model_001")],
        build_traj_views_fn=lambda systems: build_calls.append(list(systems or [])),
        get_known_objects_fn=lambda: {"wt_input_obj", "other_obj"},
        apply_scene_actions_fn=lambda actions, known_objects=None, default_action="disable": apply_calls.append(
            {
                "actions": list(actions or []),
                "known_objects": set(known_objects or set()),
                "default_action": str(default_action),
            }
        ),
        append_debug_log_fn=lambda msg: debug_lines.append(str(msg)),
        input_action="disable",
        default_action="disable",
    )
    assert payload["ok"] is True
    assert payload["traj_views_ok"] is True
    assert payload["traj_views_error"] == ""
    assert int(len(build_calls)) == 1
    assert build_calls[0] == [("WT", "wt_traj_obj", "wt_src", "WT_model_001")]
    assert int(len(payload["actions"])) == 1
    assert payload["actions"][0]["object"] == "wt_input_obj"
    assert payload["actions_applied"] is True
    assert int(len(apply_calls)) == 1
    assert apply_calls[0]["actions"] == payload["actions"]
    assert "wt_input_obj" in apply_calls[0]["known_objects"]
    assert apply_calls[0]["default_action"] == "disable"
    assert debug_lines == []

    payload_warn = prepare_redocking_runtime_views_runtime(
        runtime_systems=[("WT", "wt_traj_obj", "wt_src", "WT_model_001")],
        input_systems=[("WT", "wt_input_obj", "wt_src", "WT_model_001")],
        build_traj_views_fn=lambda _systems: (_ for _ in ()).throw(RuntimeError("traj boom")),
        get_known_objects_fn=lambda: {"wt_input_obj"},
        apply_scene_actions_fn=lambda actions, known_objects=None, default_action="disable": apply_calls.append(
            {
                "actions": list(actions or []),
                "known_objects": set(known_objects or set()),
                "default_action": str(default_action),
            }
        ),
        append_debug_log_fn=lambda msg: debug_lines.append(str(msg)),
        input_action="disable",
        default_action="disable",
    )
    assert payload_warn["ok"] is True
    assert payload_warn["traj_views_ok"] is False
    assert "traj boom" in str(payload_warn["traj_views_error"])
    assert any("Redocking trajectory-view warning: traj boom" in str(m) for m in debug_lines)

    payload_apply_fail = prepare_redocking_runtime_views_runtime(
        runtime_systems=[],
        input_systems=[("WT", "wt_input_obj", "wt_src", "WT_model_001")],
        build_traj_views_fn=None,
        get_known_objects_fn=lambda: (_ for _ in ()).throw(RuntimeError("known boom")),
        apply_scene_actions_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("apply boom")),
        append_debug_log_fn=lambda msg: debug_lines.append(str(msg)),
    )
    assert payload_apply_fail["ok"] is True
    assert payload_apply_fail["actions"] == []
    assert payload_apply_fail["actions_applied"] is False


def test_redock_runtime_traj_frame_payload_and_focus_visibility_helpers():
    actions = build_redocking_traj_focus_visibility_actions(
        frame_object="wt_frame",
        pose_frame_object="wt_pose",
        combo_object="wt_combo",
        visible=True,
        known_objects={"wt_frame", "wt_pose", "wt_combo"},
    )
    assert actions == [
        {"action": "enable", "object": "wt_combo"},
        {"action": "disable", "object": "wt_frame"},
        {"action": "disable", "object": "wt_pose"},
    ]

    actions2 = build_redocking_traj_focus_visibility_actions(
        traj_object="mut_traj",
        frame_object="mut_frame",
        pose_frame_object="mut_pose",
        combo_object="",
        visible=False,
        known_objects={"mut_traj", "mut_frame", "mut_pose"},
        prefer_traj_object=True,
    )
    assert actions2 == [
        {"action": "disable", "object": "mut_traj"},
        {"action": "disable", "object": "mut_frame"},
        {"action": "disable", "object": "mut_pose"},
    ]

    payload = prepare_redocking_traj_frame_payload(
        label="WT",
        info={
            "traj_object": "wt_traj",
            "frame_object": "wt_frame",
            "pose_frame_object": "wt_pose",
            "combo_object": "wt_combo",
            "state_map": {"WT_model_001": 3},
            "n_states": 2,
        },
        member_tag="WT_model_001",
        explicit_state=0,
        known_objects={"wt_traj", "wt_frame", "wt_pose", "wt_combo", "pose_best"},
        resolve_state_for_member_fn=lambda smap, tag, default_state=0: int(smap.get(tag, default_state) or 0),
        best_pose_for_member_fn=lambda _label, _tag: {"pose_object": "pose_best", "state": "5"},
        visible_for_label_fn=lambda _label: True,
        default_visible=False,
    )
    assert payload["ok"] is True
    assert int(payload["state"]) == 2  # clamped by n_states=2
    assert payload["sync_combo_state"] is True
    assert payload["clear_frame_object"] is True
    assert payload["clear_pose_frame_object"] is True
    assert payload["overlay_available"] is True
    assert str(payload["overlay_pose_object"]) == "pose_best"
    assert int(payload["overlay_pose_state"]) == 5
    assert payload["visible"] is True
    vt = [(str(a.get("action", "")), str(a.get("object", ""))) for a in list(payload["visibility_actions"] or [])]
    assert ("enable", "wt_combo") in vt
    assert ("disable", "wt_frame") in vt
    assert ("disable", "wt_pose") in vt

    payload_hidden = prepare_redocking_traj_frame_payload(
        label="Mutant",
        info={
            "traj_object": "mut_traj",
            "frame_object": "mut_frame",
            "pose_frame_object": "mut_pose",
            "combo_object": "",
            "state_map": {},
            "n_states": 3,
        },
        member_tag="Mut_model_001",
        explicit_state=0,
        known_objects={"mut_traj", "mut_frame", "mut_pose"},
        best_pose_for_member_fn=lambda _label, _tag: {"pose_object": "missing_pose", "state": 1},
        visible_for_label_fn=lambda _label: False,
        default_visible=True,
    )
    assert payload_hidden["ok"] is True
    assert payload_hidden["prefer_traj_object"] is True
    assert payload_hidden["sync_traj_state"] is True
    assert payload_hidden["overlay_available"] is False
    vt2 = [(str(a.get("action", "")), str(a.get("object", ""))) for a in list(payload_hidden["visibility_actions"] or [])]
    assert ("disable", "mut_traj") in vt2
    assert ("disable", "mut_frame") in vt2
    assert ("disable", "mut_pose") in vt2

    payload_missing_traj = prepare_redocking_traj_frame_payload(
        label="WT",
        info={"traj_object": "wt_traj", "frame_object": "wt_frame"},
        known_objects={"wt_frame"},
    )
    assert payload_missing_traj["ok"] is False
    assert str(payload_missing_traj["reason"]) == "traj_not_loaded"

    payload_missing_frame = prepare_redocking_traj_frame_payload(
        label="WT",
        info={"traj_object": "wt_traj", "frame_object": ""},
        known_objects={"wt_traj"},
    )
    assert payload_missing_frame["ok"] is False
    assert str(payload_missing_frame["reason"]) == "missing_traj_or_frame"


def test_redock_runtime_run_traj_frame_runtime_helper():
    sync_calls = []
    set_calls = []
    delete_calls = []
    create_calls = []
    hide_calls = []
    show_calls = []
    color_calls = []
    group_calls = []
    style_calls = []
    element_calls = []
    visible_calls = []
    focus_calls = []
    apply_calls = []

    def _visible(label):
        visible_calls.append(str(label))
        return True

    def _build_focus_actions(**kwargs):
        focus_calls.append(dict(kwargs or {}))
        return [{"action": "enable", "object": "wt_combo"}]

    payload = run_redocking_traj_frame_runtime(
        label="WT",
        info={
            "traj_object": "wt_traj",
            "frame_object": "wt_frame",
            "pose_frame_object": "wt_pose",
            "combo_object": "wt_combo",
            "state_map": {"WT_model_001": 3},
            "n_states": 2,
        },
        member_tag="WT_model_001",
        explicit_state=0,
        known_objects={"wt_traj", "wt_frame", "wt_pose", "wt_combo", "pose_best"},
        resolve_state_for_member_fn=lambda smap, tag, default_state=0: int(smap.get(tag, default_state) or 0),
        best_pose_for_member_fn=lambda _label, _tag: {"pose_object": "pose_best", "state": "4"},
        visible_for_label_fn=_visible,
        default_visible=False,
        model_selection_fn=lambda obj: f"model::{obj}",
        sync_frame_fn=lambda st: sync_calls.append(int(st)),
        set_state_fn=lambda st, obj=None: set_calls.append((int(st), str(obj or ""))),
        delete_object_fn=lambda obj: delete_calls.append(str(obj)),
        create_object_fn=lambda obj, sel, src_state, dst_state: create_calls.append(
            (str(obj), str(sel), int(src_state), int(dst_state))
        ),
        hide_fn=lambda rep, sel: hide_calls.append((str(rep), str(sel))),
        show_fn=lambda rep, sel: show_calls.append((str(rep), str(sel))),
        color_fn=lambda color_name, sel: color_calls.append((str(color_name), str(sel))),
        group_fn=lambda group_name, members: group_calls.append((str(group_name), str(members))),
        apply_ball_and_stick_style_fn=lambda obj, sphere_scale=0.0, stick_radius=0.0: style_calls.append(
            (str(obj), float(sphere_scale), float(stick_radius))
        ),
        apply_element_colors_fn=lambda obj: element_calls.append(str(obj)),
        get_known_objects_fn=lambda: {"wt_frame", "wt_pose", "wt_combo"},
        build_focus_actions_fn=_build_focus_actions,
        apply_object_actions_fn=lambda actions, known_objects=None: apply_calls.append(
            {
                "actions": list(actions or []),
                "known_objects": set(known_objects or set()),
            }
        ),
    )
    assert payload["ok"] is True
    assert str(payload["frame_object"]) == "wt_frame"
    assert int(payload["state"]) == 2
    assert sync_calls == [2]
    assert set_calls == []
    assert delete_calls == ["wt_frame", "wt_pose"]
    assert create_calls == [
        ("wt_frame", "model::wt_traj", 2, 1),
        ("wt_pose", "model::pose_best", 4, 1),
    ]
    assert hide_calls == [("everything", "(wt_frame) and not polymer")]
    assert show_calls == [("cartoon", "(wt_frame) and polymer")]
    assert color_calls == [("white", "(wt_frame) and polymer")]
    assert ("Redocking.Trajectory", "wt_frame") in group_calls
    assert ("Redocking.Trajectory", "wt_pose") in group_calls
    assert style_calls == [("wt_pose", 0.22, 0.16)]
    assert element_calls == ["wt_pose"]
    assert visible_calls == ["WT"]
    assert int(len(focus_calls)) == 1
    assert focus_calls[0]["frame_object"] == "wt_frame"
    assert focus_calls[0]["pose_frame_object"] == "wt_pose"
    assert focus_calls[0]["combo_object"] == "wt_combo"
    assert focus_calls[0]["visible"] is True
    assert focus_calls[0]["known_objects"] == {"wt_frame", "wt_pose", "wt_combo"}
    assert payload["actions"] == [{"action": "enable", "object": "wt_combo"}]
    assert int(len(apply_calls)) == 1
    assert apply_calls[0]["actions"] == payload["actions"]
    assert apply_calls[0]["known_objects"] == {"wt_frame", "wt_pose", "wt_combo"}

    payload_invalid = run_redocking_traj_frame_runtime(
        label="WT",
        info={"traj_object": "wt_traj"},
        prepare_traj_frame_payload_fn=lambda **_kwargs: {
            "ok": True,
            "traj_object": "wt_traj",
            "frame_object": "",
            "state": 1,
        },
    )
    assert payload_invalid["ok"] is False
    assert str(payload_invalid["error"]) == "invalid_frame_payload"


def test_redock_runtime_run_traj_frame_runtime_prefers_traj_object_without_combo():
    sync_calls = []
    set_calls = []
    focus_calls = []
    apply_calls = []

    payload = run_redocking_traj_frame_runtime(
        label="Mutant",
        info={
            "traj_object": "mut_traj",
            "frame_object": "mut_frame",
            "pose_frame_object": "mut_pose",
            "combo_object": "",
            "state_map": {"Mut_model_001": 2},
            "n_states": 4,
        },
        member_tag="Mut_model_001",
        explicit_state=0,
        known_objects={"mut_traj", "mut_frame", "mut_pose", "pose_best"},
        resolve_state_for_member_fn=lambda smap, tag, default_state=0: int(smap.get(tag, default_state) or 0),
        best_pose_for_member_fn=lambda _label, _tag: {"pose_object": "pose_best", "state": 3},
        visible_for_label_fn=lambda _label: True,
        model_selection_fn=lambda obj: f"model::{obj}",
        sync_frame_fn=lambda st: sync_calls.append(int(st)),
        set_state_fn=lambda st, obj=None: set_calls.append((int(st), str(obj or ""))),
        delete_object_fn=lambda _obj: None,
        create_object_fn=lambda _obj, _sel, _src_state, _dst_state: None,
        hide_fn=lambda _rep, _sel: None,
        show_fn=lambda _rep, _sel: None,
        color_fn=lambda _color_name, _sel: None,
        group_fn=lambda _group_name, _members: None,
        apply_ball_and_stick_style_fn=lambda _obj, sphere_scale=0.0, stick_radius=0.0: None,
        apply_element_colors_fn=lambda _obj: None,
        get_known_objects_fn=lambda: {"mut_traj", "mut_frame", "mut_pose"},
        build_focus_actions_fn=lambda **kwargs: focus_calls.append(dict(kwargs or {}))
        or [
            {"action": "enable", "object": "mut_traj"},
            {"action": "disable", "object": "mut_frame"},
            {"action": "enable", "object": "mut_pose"},
        ],
        apply_object_actions_fn=lambda actions, known_objects=None: apply_calls.append(
            {
                "actions": list(actions or []),
                "known_objects": set(known_objects or set()),
            }
        ),
    )
    assert payload["ok"] is True
    assert str(payload["frame_object"]) == "mut_frame"
    assert int(payload["state"]) == 2
    assert sync_calls == [2]
    assert set_calls == []
    assert len(focus_calls) == 1
    assert focus_calls[0]["traj_object"] == "mut_traj"
    assert focus_calls[0]["prefer_traj_object"] is True
    assert payload["actions"] == [
        {"action": "enable", "object": "mut_traj"},
        {"action": "disable", "object": "mut_frame"},
        {"action": "enable", "object": "mut_pose"},
    ]
    assert len(apply_calls) == 1
    assert apply_calls[0]["actions"] == payload["actions"]
    assert apply_calls[0]["known_objects"] == {"mut_traj", "mut_frame", "mut_pose"}

    payload_prepare_fail = run_redocking_traj_frame_runtime(
        label="WT",
        info={},
        prepare_traj_frame_payload_fn=lambda **_kwargs: {"ok": False, "reason": "prepare_failed"},
    )
    assert payload_prepare_fail["ok"] is False
    assert str(payload_prepare_fail["error"]) == "prepare_failed"


def test_redock_runtime_run_traj_frame_runtime_falls_back_to_object_state_when_frame_sync_missing():
    set_calls = []

    payload = run_redocking_traj_frame_runtime(
        label="Mutant",
        info={
            "traj_object": "mut_traj",
            "frame_object": "mut_frame",
            "pose_frame_object": "mut_pose",
            "combo_object": "",
            "state_map": {"Mut_model_001": 2},
            "n_states": 4,
        },
        member_tag="Mut_model_001",
        explicit_state=0,
        known_objects={"mut_traj", "mut_frame", "mut_pose"},
        resolve_state_for_member_fn=lambda smap, tag, default_state=0: int(smap.get(tag, default_state) or 0),
        best_pose_for_member_fn=lambda _label, _tag: None,
        visible_for_label_fn=lambda _label: True,
        sync_frame_fn=None,
        set_state_fn=lambda st, obj=None: set_calls.append((int(st), str(obj or ""))),
        delete_object_fn=lambda _obj: None,
        create_object_fn=lambda _obj, _sel, _src_state, _dst_state: None,
        hide_fn=lambda _rep, _sel: None,
        show_fn=lambda _rep, _sel: None,
        color_fn=lambda _color_name, _sel: None,
        group_fn=lambda _group_name, _members: None,
        get_known_objects_fn=lambda: {"mut_traj", "mut_frame", "mut_pose"},
    )
    assert payload["ok"] is True
    assert int(payload["state"]) == 2
    assert set_calls == [(2, "mut_traj")]


def test_redock_runtime_selected_pose_context_and_view_plan_helpers():
    no_rows = prepare_redocking_selected_pose_context(
        [],
        row_index=0,
        known_objects={"pose_a"},
    )
    assert no_rows["ok"] is False
    assert str(no_rows["reason"]) == "no_results"
    assert str(no_rows["warning"]) == ""

    out_range = prepare_redocking_selected_pose_context(
        [{"pose_object": "pose_a"}],
        row_index=9,
        known_objects={"pose_a"},
    )
    assert out_range["ok"] is False
    assert str(out_range["reason"]) == "row_out_of_range"
    assert "Select a row" in str(out_range["warning"])

    pose_missing = prepare_redocking_selected_pose_context(
        [{"pose_object": "pose_a", "state": "3"}],
        row_index=0,
        known_objects=set(),
    )
    assert pose_missing["ok"] is False
    assert str(pose_missing["reason"]) == "pose_missing"
    assert "Pose object" in str(pose_missing["warning"])

    ctx = prepare_redocking_selected_pose_context(
        [
            {
                "pose_object": "pose_a",
                "state": "not_an_int",
                "receptor": "rec_a",
                "system": "WT",
                "system_member": "WT_model_002",
                "traj_state": "7",
            }
        ],
        row_index=0,
        known_objects=set(),
        require_pose_in_scene=False,
        preview_object="preview_a",
    )
    assert ctx["ok"] is True
    assert str(ctx["pose_object"]) == "pose_a"
    assert int(ctx["pose_state"]) == 1
    assert str(ctx["receptor_object"]) == "rec_a"
    assert str(ctx["system_label"]) == "WT"
    assert str(ctx["member_tag"]) == "WT_model_002"
    assert int(ctx["traj_state"]) == 7
    assert str(ctx["preview_object"]) == "preview_a"

    vp = build_redocking_selected_pose_view_plan(
        preview_object="preview_a",
        traj_frame_object="traj_frame_a",
        receptor_object="rec_a",
        known_objects={"preview_a", "traj_frame_a", "rec_a"},
    )
    assert str(vp["receptor_candidate_object"]) == "traj_frame_a"
    assert str(vp["receptor_view_object"]) == "traj_frame_a"
    assert str(vp["focus_selection"]) == "(preview_a) or (traj_frame_a) and polymer"

    # Preserve current UI behavior: no receptor fallback when a non-empty trajectory frame name is unavailable.
    vp2 = build_redocking_selected_pose_view_plan(
        preview_object="preview_b",
        traj_frame_object="traj_frame_b",
        receptor_object="rec_b",
        known_objects={"preview_b", "rec_b"},
    )
    assert str(vp2["receptor_candidate_object"]) == "traj_frame_b"
    assert str(vp2["receptor_view_object"]) == ""

    exec_plan = build_redocking_selected_pose_execution_plan(
        receptor_view_object=str(vp["receptor_view_object"]),
        focus_selection=str(vp["focus_selection"]),
        default_action="disable",
        focus_zoom_buffer_a=6.0,
    )
    assert exec_plan["ok"] is True
    assert str(exec_plan["receptor_polymer_selection"]) == "(traj_frame_a) and polymer"
    assert bool(exec_plan["has_scene_actions"]) is True
    op_kinds = [str((op or {}).get("kind", "")) for op in list(exec_plan["operations"] or [])]
    assert op_kinds == ["scene_actions", "orient", "zoom"]
    first_actions = list((list(exec_plan["operations"] or [])[0] or {}).get("actions", []) or [])
    assert int(len(first_actions)) == 4
    assert str((list(exec_plan["operations"] or [])[2] or {}).get("selection", "")) == "(preview_a) or (traj_frame_a) and polymer"
    assert float((list(exec_plan["operations"] or [])[2] or {}).get("buffer_a", 0.0)) == 6.0

    exec_plan_no_receptor = build_redocking_selected_pose_execution_plan(
        receptor_view_object="",
        focus_selection="(preview_b)",
        default_action="unknown",
        focus_zoom_buffer_a=float("nan"),
    )
    assert str(exec_plan_no_receptor["default_action"]) == "disable"
    assert bool(exec_plan_no_receptor["has_scene_actions"]) is False
    assert [str((op or {}).get("kind", "")) for op in list(exec_plan_no_receptor["operations"] or [])] == ["orient", "zoom"]
    assert float((list(exec_plan_no_receptor["operations"] or [])[1] or {}).get("buffer_a", 0.0)) == 6.0


def test_redock_runtime_run_selected_pose_runtime_helper():
    calls = {
        "hide_pose": 0,
        "hide_rec": 0,
        "delete": [],
        "create": [],
        "style": [],
        "color": [],
        "group": [],
        "set_traj": [],
        "scene": [],
        "render_ref": 0,
        "orient": [],
        "zoom": [],
    }

    payload = run_redocking_selected_pose_runtime(
        rows=[
            {
                "pose_object": "pose_a",
                "state": "3",
                "receptor": "rec_a",
                "system": "WT",
                "system_member": "WT_model_002",
                "traj_state": "7",
            }
        ],
        row_index=0,
        known_objects={"pose_a", "preview_a"},
        preview_object="preview_a",
        hide_all_pose_objects_fn=lambda: calls.__setitem__("hide_pose", int(calls["hide_pose"]) + 1),
        hide_all_runtime_receptors_fn=lambda: calls.__setitem__("hide_rec", int(calls["hide_rec"]) + 1),
        delete_object_fn=lambda obj: calls["delete"].append(str(obj)),
        create_object_fn=lambda obj, sel, src_state, dst_state: calls["create"].append(
            (str(obj), str(sel), int(src_state), int(dst_state))
        ),
        model_selection_fn=lambda obj: f"model::{obj}",
        apply_ball_and_stick_style_fn=lambda obj, sphere_scale=0.0, stick_radius=0.0: calls["style"].append(
            (str(obj), float(sphere_scale), float(stick_radius))
        ),
        apply_element_colors_fn=lambda obj: calls["color"].append(str(obj)),
        group_fn=lambda group_name, members: calls["group"].append((str(group_name), str(members))),
        set_traj_frame_fn=lambda label, member_tag, explicit_state=0: calls["set_traj"].append(
            (str(label), str(member_tag), int(explicit_state))
        )
        or ("traj_frame_a", 7),
        get_known_objects_fn=lambda: {"preview_a", "traj_frame_a", "rec_a"},
        apply_scene_actions_fn=lambda actions, known_objects=None, default_action="disable": calls["scene"].append(
            {
                "actions": list(actions or []),
                "known_objects": set(known_objects or set()),
                "default_action": str(default_action),
            }
        ),
        render_reference_context_fn=lambda: calls.__setitem__("render_ref", int(calls["render_ref"]) + 1),
        orient_fn=lambda sel: calls["orient"].append(str(sel)),
        zoom_fn=lambda sel, buf: calls["zoom"].append((str(sel), float(buf))),
    )
    assert payload["ok"] is True
    assert str(payload["reason"]) == ""
    assert str(payload["error"]) == ""
    assert str(payload["preview_object"]) == "preview_a"
    assert str(payload["traj_frame_object"]) == "traj_frame_a"
    assert int(payload["traj_state"]) == 7
    assert int(calls["hide_pose"]) == 1
    assert int(calls["hide_rec"]) == 1
    assert calls["delete"] == ["preview_a"]
    assert calls["create"] == [("preview_a", "model::pose_a", 3, 1)]
    assert calls["style"] == [("preview_a", 0.22, 0.16)]
    assert calls["color"] == ["preview_a"]
    assert ("Redocking.Results", "preview_a") in list(calls["group"] or [])
    assert calls["set_traj"] == [("WT", "WT_model_002", 7)]
    assert int(len(calls["scene"])) == 1
    assert str(calls["scene"][0]["default_action"]) == "disable"
    assert calls["scene"][0]["known_objects"] == {"preview_a", "traj_frame_a", "rec_a"}
    assert int(len(calls["scene"][0]["actions"])) == 4
    assert int(calls["render_ref"]) == 1
    assert calls["orient"] == ["(preview_a) or (traj_frame_a) and polymer"]
    assert calls["zoom"] == [("(preview_a) or (traj_frame_a) and polymer", 6.0)]

    payload_no_rows = run_redocking_selected_pose_runtime(
        rows=[],
        row_index=0,
        known_objects={"pose_a"},
    )
    assert payload_no_rows["ok"] is False
    assert str(payload_no_rows["reason"]) == "no_results"
    assert str(payload_no_rows["warning"]) == ""

    payload_runtime_error = run_redocking_selected_pose_runtime(
        rows=[{"pose_object": "pose_a", "state": 1}],
        row_index=0,
        known_objects={"pose_a"},
        create_object_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("create boom")),
    )
    assert payload_runtime_error["ok"] is False
    assert str(payload_runtime_error["reason"]) == "runtime_error"
    assert "create boom" in str(payload_runtime_error["error"])
    assert "Failed to load selected pose" in str(payload_runtime_error["warning"])


def test_redock_runtime_selected_pose_warning_message_helper():
    warn_no_rows = resolve_redocking_selected_pose_warning_message(
        {"reason": "no_results", "warning": "ignored"},
        silent=False,
    )
    assert warn_no_rows["show_warning"] is False
    assert str(warn_no_rows["warning"]) == ""
    assert str(warn_no_rows["reason"]) == "no_results"

    warn_silent = resolve_redocking_selected_pose_warning_message(
        {"reason": "runtime_error", "warning": "boom"},
        silent=True,
    )
    assert warn_silent["show_warning"] is False
    assert str(warn_silent["warning"]) == ""

    warn_show = resolve_redocking_selected_pose_warning_message(
        {"reason": "runtime_error", "warning": "show me"},
        silent=False,
    )
    assert warn_show["show_warning"] is True
    assert str(warn_show["warning"]) == "show me"
    assert str(warn_show["reason"]) == "runtime_error"

    warn_empty = resolve_redocking_selected_pose_warning_message(
        {"reason": "pose_missing", "warning": ""},
        silent=False,
    )
    assert warn_empty["show_warning"] is False
    assert str(warn_empty["warning"]) == ""


def test_redock_runtime_reference_grid_warning_message_helper():
    warn_missing_obj = resolve_redocking_reference_grid_warning_message(
        ref_object="",
        ref_fields=("A", "10", "LIG"),
        center=(1.0, 2.0, 3.0),
    )
    assert warn_missing_obj["show_warning"] is True
    assert str(warn_missing_obj["reason"]) == "missing_reference_or_fields"
    assert "Select reference object and crystal ligand" in str(warn_missing_obj["warning"])

    warn_missing_fields = resolve_redocking_reference_grid_warning_message(
        ref_object="ref_obj",
        ref_fields=(),
        center=(1.0, 2.0, 3.0),
    )
    assert warn_missing_fields["show_warning"] is True
    assert str(warn_missing_fields["reason"]) == "missing_reference_or_fields"

    warn_not_found = resolve_redocking_reference_grid_warning_message(
        ref_object="ref_obj",
        ref_fields=("A", "10", "LIG"),
        center=None,
    )
    assert warn_not_found["show_warning"] is True
    assert str(warn_not_found["reason"]) == "selection_not_found"
    assert "Reference ligand selection was not found" in str(warn_not_found["warning"])

    warn_ok = resolve_redocking_reference_grid_warning_message(
        ref_object="ref_obj",
        ref_fields=("A", "10", "LIG"),
        center=(1.0, 2.0, 3.0),
    )
    assert warn_ok["show_warning"] is False
    assert str(warn_ok["reason"]) == ""
    assert str(warn_ok["warning"]) == ""

    warn_custom = resolve_redocking_reference_grid_warning_message(
        ref_object="",
        ref_fields=(),
        center=None,
        missing_setup_warning="custom missing",
        selection_missing_warning="custom not found",
    )
    assert warn_custom["show_warning"] is True
    assert str(warn_custom["warning"]) == "custom missing"


def test_redock_runtime_reference_grid_info_message_helper():
    info = build_redocking_reference_grid_info_message(
        center=(1.23456, 2.34567, 3.45678),
        size=(18.0, 19.5, 20.25),
        nref=25,
    )
    assert info["ok"] is True
    msg = str(info["message"])
    assert "Grid centered on reference ligand (25 atoms)." in msg
    assert "Center=(1.235, 2.346, 3.457)" in msg
    assert "Size=(18.00, 19.50, 20.25) A" in msg

    info_bad = build_redocking_reference_grid_info_message(
        center=None,
        size=(18.0, 19.5, 20.25),
        nref=25,
    )
    assert info_bad["ok"] is False
    assert str(info_bad["message"]) == ""


def test_redock_runtime_reference_grid_size_helper():
    payload = build_redocking_reference_grid_size(
        span=(5.0, 12.0, 30.0),
        padding=3.0,
        min_edge=10.0,
        max_edge=24.0,
    )
    assert payload["ok"] is True
    # padding on each side; the 24 A cap never cuts the 30 A ligand (span + 1 A per side)
    assert payload["base_size"] == [11.0, 18.0, 32.0]
    assert payload["size"] == [11.0, 18.0, 32.0]
    assert payload["size_delta"] == [0.0, 0.0, 0.0]

    payload_adj = build_redocking_reference_grid_size(
        span=(5.0, 12.0, 30.0),
        padding=3.0,
        min_edge=10.0,
        max_edge=24.0,
        size_delta=(2.0, -4.0, 10.0),
        final_min_edge=4.0,
        final_max_edge=40.0,
    )
    assert payload_adj["ok"] is True
    assert payload_adj["base_size"] == [11.0, 18.0, 32.0]
    # a manual shrink stops at the ligand plus 1 A per side (y: 12 + 2)
    assert payload_adj["size"] == [13.0, 14.0, 40.0]

    payload_bad = build_redocking_reference_grid_size(span=("bad",))
    assert payload_bad["ok"] is False
    assert str(payload_bad["reason"]) == "invalid_span"


def test_redock_runtime_adjust_grid_size_helper():
    grow_y = adjust_redocking_grid_size(
        size=(10.0, 12.0, 14.0),
        axis="y",
        delta=2.5,
        min_edge=2.0,
        max_edge=40.0,
    )
    assert grow_y["ok"] is True
    assert grow_y["size"] == [10.0, 14.5, 14.0]

    shrink_z = adjust_redocking_grid_size(
        size=(10.0, 12.0, 14.0),
        axis="z",
        delta=-20.0,
        min_edge=6.0,
        max_edge=40.0,
    )
    assert shrink_z["ok"] is True
    assert shrink_z["size"] == [10.0, 12.0, 6.0]

    bad_axis = adjust_redocking_grid_size(size=(10.0, 12.0, 14.0), axis="q", delta=1.0)
    assert bad_axis["ok"] is False
    assert str(bad_axis["reason"]) == "invalid_axis"

    bad_size = adjust_redocking_grid_size(size=("bad",), axis="x", delta=1.0)
    assert bad_size["ok"] is False
    assert str(bad_size["reason"]) == "invalid_size"


def test_redock_runtime_reference_context_plan_helper():
    missing = build_redocking_reference_context_plan(
        ref_object="",
        ref_fields=("A", "10", "LIG"),
        known_objects={"ref_obj"},
    )
    assert missing["ok"] is False
    assert str(missing["reason"]) == "missing_reference"

    not_loaded = build_redocking_reference_context_plan(
        ref_object="ref_obj",
        ref_fields=("A", "10", "LIG"),
        known_objects=set(),
    )
    assert not_loaded["ok"] is False
    assert str(not_loaded["reason"]) == "reference_not_loaded"

    calls = []

    def _ref_sel(ref_obj, fields):
        calls.append((str(ref_obj), tuple(fields)))
        return f"({ref_obj}) and chain {fields[0]} and resi {fields[1]} and resn {fields[2]}"

    payload = build_redocking_reference_context_plan(
        ref_object="ref_obj",
        ref_fields=("A", "10", "LIG"),
        known_objects={"ref_obj"},
        reference_selection_fn=_ref_sel,
    )
    assert payload["ok"] is True
    assert str(payload["polymer_selection"]) == "(ref_obj) and polymer"
    assert "chain A" in str(payload["highlight_selection"])
    assert calls == [("ref_obj", ("A", "10", "LIG"))]
    actions = list(payload.get("actions", []) or [])
    action_tuples = [
        (
            str(a.get("action", "")),
            str(a.get("representation", "")),
            str(a.get("selection", "")),
            str(a.get("object", "")),
            str(a.get("color", "")),
            str(a.get("setting", "")),
        )
        for a in actions
    ]
    assert ("enable", "", "", "ref_obj", "", "") in action_tuples
    assert ("show", "cartoon", "(ref_obj) and polymer", "", "", "") in action_tuples
    assert ("color", "", "(ref_obj) and polymer", "", "gray70", "") in action_tuples
    assert ("set", "", "(ref_obj) and polymer", "", "", "cartoon_transparency") in action_tuples
    assert ("show", "sticks", str(payload["highlight_selection"]), "", "", "") in action_tuples
    assert ("color", "", str(payload["highlight_selection"]), "", "yellow", "") in action_tuples

    payload_no_highlight = build_redocking_reference_context_plan(
        ref_object="ref_obj",
        ref_fields=("A",),
        known_objects={"ref_obj"},
        reference_selection_fn=_ref_sel,
    )
    assert payload_no_highlight["ok"] is True
    assert str(payload_no_highlight["highlight_selection"]) == ""
    reps = [str(a.get("representation", "")) for a in list(payload_no_highlight.get("actions", []) or [])]
    assert "sticks" not in reps

    payload_cb_error = build_redocking_reference_context_plan(
        ref_object="ref_obj",
        ref_fields=("A", "10", "LIG"),
        known_objects={"ref_obj"},
        reference_selection_fn=lambda _r, _f: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert payload_cb_error["ok"] is True
    assert str(payload_cb_error["highlight_selection"]) == ""
    reps2 = [str(a.get("representation", "")) for a in list(payload_cb_error.get("actions", []) or [])]
    assert "sticks" not in reps2


def test_redock_runtime_reference_context_runtime_helper():
    apply_calls = []
    sel_calls = []

    payload = run_redocking_reference_context_runtime(
        ref_object="ref_obj",
        ref_fields=("A", "10", "LIG"),
        get_known_objects_fn=lambda: {"ref_obj"},
        reference_selection_fn=lambda ref_obj, fields: sel_calls.append((str(ref_obj), tuple(fields)))
        or f"({ref_obj}) and chain {fields[0]} and resi {fields[1]} and resn {fields[2]}",
        apply_scene_actions_fn=lambda actions, known_objects=None, default_action="disable": apply_calls.append(
            {
                "actions": list(actions or []),
                "known_objects": set(known_objects or set()),
                "default_action": str(default_action),
            }
        ),
        default_action="disable",
    )
    assert payload["ok"] is True
    assert payload["reason"] == ""
    assert payload["error"] == ""
    assert payload["actions_applied"] is True
    assert payload["known_objects"] == {"ref_obj"}
    assert int(len(payload["actions"])) >= 4
    assert sel_calls == [("ref_obj", ("A", "10", "LIG"))]
    assert int(len(apply_calls)) == 1
    assert apply_calls[0]["actions"] == payload["actions"]
    assert apply_calls[0]["known_objects"] == {"ref_obj"}
    assert apply_calls[0]["default_action"] == "disable"

    payload_known_fallback = run_redocking_reference_context_runtime(
        ref_object="ref_obj",
        ref_fields=(),
        known_objects={"ref_obj"},
        get_known_objects_fn=lambda: (_ for _ in ()).throw(RuntimeError("known boom")),
    )
    assert payload_known_fallback["ok"] is True
    assert payload_known_fallback["known_objects"] == {"ref_obj"}

    payload_invalid = run_redocking_reference_context_runtime(
        ref_object="",
        ref_fields=(),
        known_objects={"ref_obj"},
    )
    assert payload_invalid["ok"] is False
    assert str(payload_invalid["reason"]) == "missing_reference"
    assert payload_invalid["actions_applied"] is False

    payload_plan_error = run_redocking_reference_context_runtime(
        ref_object="ref_obj",
        ref_fields=(),
        known_objects={"ref_obj"},
        build_reference_context_plan_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("plan boom")),
    )
    assert payload_plan_error["ok"] is False
    assert str(payload_plan_error["reason"]) == "plan_error"
    assert "plan boom" in str(payload_plan_error["error"])

    payload_apply_error = run_redocking_reference_context_runtime(
        ref_object="ref_obj",
        ref_fields=(),
        known_objects={"ref_obj"},
        apply_scene_actions_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("apply boom")),
    )
    assert payload_apply_error["ok"] is False
    assert str(payload_apply_error["reason"]) == "apply_error"
    assert "apply boom" in str(payload_apply_error["error"])
    assert payload_apply_error["actions_applied"] is False


def test_redock_runtime_selected_interaction_context_helper():
    sel_ctx = {
        "receptor_object": "rec_a",
        "system_label": "WT",
        "member_tag": "WT_model_001",
        "traj_state": "4",
    }

    payload = prepare_redocking_selected_interaction_context(
        sel_ctx,
        traj_frame_object="traj_frame_a",
        known_objects={"redock_selected_pose_preview", "traj_frame_a", "rec_a"},
        preview_object="redock_selected_pose_preview",
        complex_object="redock_selected_complex",
        require_preview_in_scene=True,
        require_receptor_in_scene=False,
    )
    assert payload["ok"] is True
    assert str(payload["receptor_context_object"]) == "traj_frame_a"
    assert str(payload["complex_object"]) == "redock_selected_complex"
    assert str(payload["complex_selection"]) == "((traj_frame_a) and polymer) or (redock_selected_pose_preview)"
    assert str(payload["system_label"]) == "WT"
    assert str(payload["member_tag"]) == "WT_model_001"
    assert int(payload["traj_state"]) == 4

    missing_preview = prepare_redocking_selected_interaction_context(
        sel_ctx,
        traj_frame_object="traj_frame_a",
        known_objects={"traj_frame_a", "rec_a"},
        require_preview_in_scene=True,
    )
    assert missing_preview["ok"] is False
    assert str(missing_preview["reason"]) == "preview_missing"

    missing_receptor = prepare_redocking_selected_interaction_context(
        {"receptor_object": "", "system_label": "WT"},
        traj_frame_object="",
        known_objects={"redock_selected_pose_preview"},
        require_preview_in_scene=True,
    )
    assert missing_receptor["ok"] is False
    assert str(missing_receptor["reason"]) == "missing_receptor_context"

    receptor_not_loaded = prepare_redocking_selected_interaction_context(
        sel_ctx,
        traj_frame_object="traj_frame_a",
        known_objects={"redock_selected_pose_preview"},
        require_receptor_in_scene=True,
    )
    assert receptor_not_loaded["ok"] is False
    assert str(receptor_not_loaded["reason"]) == "receptor_missing"


def test_redock_runtime_selected_interaction_post_payload_helper():
    payload_empty = prepare_redocking_selected_interaction_post_payload(
        complex_object="",
        ligand_info=("A", "101", "LIG"),
    )
    assert payload_empty["ok"] is False
    assert "complex_object is empty" in str(payload_empty["error"])

    payload_tuple = prepare_redocking_selected_interaction_post_payload(
        complex_object="redock_selected_complex",
        ligand_info=("A", 101, "LIG"),
    )
    assert payload_tuple["ok"] is True
    assert str(payload_tuple["loaded_object"]) == "redock_selected_complex"
    assert str(payload_tuple["traj_topology_object"]) == "redock_selected_complex"
    assert payload_tuple["set_ligand_info"] is True
    assert tuple(payload_tuple["ligand_info"]) == ("A", "101", "LIG")
    assert dict(payload_tuple["ligand_fields"]) == {"chain": "A", "resi": "101", "resn": "LIG"}

    payload_dict = prepare_redocking_selected_interaction_post_payload(
        complex_object="cx",
        ligand_info={"chain": "B", "resi": "7", "resn": "BEN"},
    )
    assert payload_dict["set_ligand_info"] is True
    assert tuple(payload_dict["ligand_info"]) == ("B", "7", "BEN")

    payload_none = prepare_redocking_selected_interaction_post_payload(
        complex_object="cx",
        ligand_info=None,
    )
    assert payload_none["ok"] is True
    assert payload_none["set_ligand_info"] is False
    assert payload_none["ligand_info"] is None
    assert dict(payload_none["ligand_fields"]) == {"chain": "", "resi": "", "resn": ""}

    payload_short = prepare_redocking_selected_interaction_post_payload(
        complex_object="cx",
        ligand_info=("A",),
    )
    assert payload_short["set_ligand_info"] is False


def test_redock_runtime_selected_interaction_post_apply_plan_helper():
    plan_tuple = build_redocking_selected_interaction_post_apply_plan(
        post_payload={
            "loaded_object": "redock_selected_complex",
            "traj_topology_object": "traj_top",
            "set_ligand_info": True,
            "ligand_info": ("A", "101", "LIG"),
            "ligand_fields": {"chain": "A", "resi": "101", "resn": "LIG"},
        },
        default_complex_object="fallback_complex",
    )
    assert plan_tuple["ok"] is True
    assert str(plan_tuple["loaded_object"]) == "redock_selected_complex"
    assert str(plan_tuple["traj_topology_object"]) == "traj_top"
    assert bool(plan_tuple["set_ligand_info"]) is True
    assert tuple(plan_tuple["ligand_info"]) == ("A", "101", "LIG")
    assert dict(plan_tuple["ligand_fields"]) == {"chain": "A", "resi": "101", "resn": "LIG"}
    ops = list(plan_tuple.get("operations", []) or [])
    assert ops[0] == {"kind": "set_attr", "name": "loaded_object", "value": "redock_selected_complex"}
    assert ops[1] == {"kind": "set_attr", "name": "_traj_topology_object", "value": "traj_top"}
    assert {"kind": "set_attr", "name": "ligand_info", "value": ("A", "101", "LIG")} in ops
    assert {"kind": "set_text", "widget": "ligand_chain", "value": "A"} in ops
    assert {"kind": "set_text", "widget": "ligand_resi", "value": "101"} in ops
    assert {"kind": "set_text", "widget": "ligand_resn", "value": "LIG"} in ops

    plan_min = build_redocking_selected_interaction_post_apply_plan(
        post_payload={},
        default_complex_object="fallback_complex",
    )
    assert plan_min["ok"] is True
    assert str(plan_min["loaded_object"]) == "fallback_complex"
    assert str(plan_min["traj_topology_object"]) == "fallback_complex"
    assert bool(plan_min["set_ligand_info"]) is False
    assert tuple(plan_min["ligand_info"]) == ()
    assert dict(plan_min["ligand_fields"]) == {"chain": "", "resi": "", "resn": ""}
    assert list(plan_min.get("operations", []) or []) == [
        {"kind": "set_attr", "name": "loaded_object", "value": "fallback_complex"},
        {"kind": "set_attr", "name": "_traj_topology_object", "value": "fallback_complex"},
    ]

    plan_nondict = build_redocking_selected_interaction_post_apply_plan(
        post_payload="bad_type",
        default_complex_object="fallback_complex",
    )
    assert str(plan_nondict["loaded_object"]) == "fallback_complex"
    assert bool(plan_nondict["set_ligand_info"]) is False


def test_redock_runtime_run_selected_interaction_runtime_helper():
    calls = {
        "load_selected": [],
        "set_traj": [],
        "delete": [],
        "create": [],
        "apply_post": [],
        "populate": 0,
        "calc": 0,
    }

    payload = run_redocking_selected_interaction_runtime(
        rows=[
            {
                "pose_object": "pose_a",
                "state": 1,
                "receptor": "rec_a",
                "system": "WT",
                "system_member": "WT_model_001",
                "traj_state": 4,
            }
        ],
        row_index=0,
        preview_object="redock_selected_pose_preview",
        complex_object="redock_selected_complex",
        load_selected_pose_fn=lambda silent=True: calls["load_selected"].append(bool(silent)),
        set_traj_frame_fn=lambda label, member_tag, explicit_state=0: calls["set_traj"].append(
            (str(label), str(member_tag), int(explicit_state))
        )
        or ("traj_frame_a", int(explicit_state)),
        get_known_objects_fn=lambda: {"redock_selected_pose_preview", "traj_frame_a", "redock_selected_complex"},
        delete_object_fn=lambda obj: calls["delete"].append(str(obj)),
        create_object_fn=lambda obj, sel, src_state, dst_state: calls["create"].append(
            (str(obj), str(sel), int(src_state), int(dst_state))
        ),
        auto_detect_ligand_fn=lambda: ("A", "101", "LIG"),
        apply_post_payload_fn=lambda post: calls["apply_post"].append(dict(post or {})),
        populate_limit_combo_fn=lambda: calls.__setitem__("populate", int(calls["populate"]) + 1),
        calc_interactions_fn=lambda: calls.__setitem__("calc", int(calls["calc"]) + 1),
    )
    assert payload["ok"] is True
    assert str(payload["reason"]) == ""
    assert str(payload["warning"]) == ""
    assert str(payload["error"]) == ""
    assert str(payload["complex_object"]) == "redock_selected_complex"
    assert str(payload["traj_frame_object"]) == "traj_frame_a"
    assert calls["load_selected"] == [True]
    assert calls["set_traj"] == [("WT", "WT_model_001", 4)]
    assert calls["delete"] == ["redock_selected_complex"]
    assert calls["create"] == [
        (
            "redock_selected_complex",
            "((traj_frame_a) and polymer) or (redock_selected_pose_preview)",
            1,
            1,
        )
    ]
    assert int(len(calls["apply_post"])) == 2
    assert str(calls["apply_post"][0].get("loaded_object", "")) == "redock_selected_complex"
    assert bool(calls["apply_post"][0].get("set_ligand_info", False)) is False
    assert bool(calls["apply_post"][1].get("set_ligand_info", False)) is True
    assert int(calls["populate"]) == 1
    assert int(calls["calc"]) == 1
    assert str(payload["complex_selection"]) == "((traj_frame_a) and polymer) or (redock_selected_pose_preview)"
    assert bool(payload["post_payload"].get("set_ligand_info", False)) is True

    state = {"loaded_object": ""}
    payload_context_first = run_redocking_selected_interaction_runtime(
        rows=[
            {
                "pose_object": "pose_a",
                "state": 1,
                "receptor": "rec_a",
                "system": "WT",
                "system_member": "WT_model_001",
                "traj_state": 4,
            }
        ],
        row_index=0,
        preview_object="redock_selected_pose_preview",
        complex_object="redock_selected_complex",
        load_selected_pose_fn=lambda silent=True: None,
        set_traj_frame_fn=lambda label, member_tag, explicit_state=0: ("traj_frame_a", int(explicit_state)),
        get_known_objects_fn=lambda: {"redock_selected_pose_preview", "traj_frame_a"},
        create_object_fn=lambda *_a, **_k: None,
        apply_post_payload_fn=lambda post: state.__setitem__("loaded_object", str((post or {}).get("loaded_object", "") or "")),
        auto_detect_ligand_fn=lambda: ("A", "101", "LIG")
        if str(state.get("loaded_object", "")) == "redock_selected_complex"
        else (_ for _ in ()).throw(RuntimeError("context not applied")),
        populate_limit_combo_fn=lambda: None,
        calc_interactions_fn=lambda: None,
    )
    assert payload_context_first["ok"] is True

    payload_object_autodetect = run_redocking_selected_interaction_runtime(
        rows=[
            {
                "pose_object": "pose_a",
                "state": 1,
                "receptor": "rec_a",
                "system": "WT",
                "system_member": "WT_model_001",
                "traj_state": 4,
            }
        ],
        row_index=0,
        preview_object="redock_selected_pose_preview",
        complex_object="redock_selected_complex",
        load_selected_pose_fn=lambda silent=True: None,
        set_traj_frame_fn=lambda label, member_tag, explicit_state=0: ("traj_frame_a", int(explicit_state)),
        get_known_objects_fn=lambda: {"redock_selected_pose_preview", "traj_frame_a"},
        create_object_fn=lambda *_a, **_k: None,
        apply_post_payload_fn=lambda post: None,
        auto_detect_ligand_fn=lambda obj_name: ("A", "101", "LIG")
        if str(obj_name) == "redock_selected_complex"
        else (_ for _ in ()).throw(RuntimeError("wrong object")),
        populate_limit_combo_fn=lambda: None,
        calc_interactions_fn=lambda: None,
    )
    assert payload_object_autodetect["ok"] is True
    assert bool(payload_object_autodetect["post_payload"].get("set_ligand_info", False)) is True

    payload_autodetect_fail = run_redocking_selected_interaction_runtime(
        rows=[
            {
                "pose_object": "pose_a",
                "state": 1,
                "receptor": "rec_a",
                "system": "WT",
                "system_member": "WT_model_001",
                "traj_state": 4,
            }
        ],
        row_index=0,
        preview_object="redock_selected_pose_preview",
        complex_object="redock_selected_complex",
        load_selected_pose_fn=lambda silent=True: None,
        set_traj_frame_fn=lambda label, member_tag, explicit_state=0: ("traj_frame_a", int(explicit_state)),
        get_known_objects_fn=lambda: {"redock_selected_pose_preview", "traj_frame_a"},
        create_object_fn=lambda *_a, **_k: None,
        apply_post_payload_fn=lambda post: None,
        auto_detect_ligand_fn=lambda: (_ for _ in ()).throw(RuntimeError("auto detect boom")),
        populate_limit_combo_fn=lambda: None,
        calc_interactions_fn=lambda: None,
    )
    assert payload_autodetect_fail["ok"] is True
    assert bool(payload_autodetect_fail["post_payload"].get("set_ligand_info", False)) is False

    payload_select_fail = run_redocking_selected_interaction_runtime(
        rows=[],
        row_index=0,
    )
    assert payload_select_fail["ok"] is False
    assert str(payload_select_fail["reason"]) == "no_results"
    assert str(payload_select_fail["warning"]) == "Select a redocking row first."

    payload_interaction_fail = run_redocking_selected_interaction_runtime(
        rows=[
            {
                "pose_object": "pose_a",
                "receptor": "rec_a",
                "system": "WT",
                "system_member": "WT_model_001",
                "traj_state": 2,
            }
        ],
        row_index=0,
        get_known_objects_fn=lambda: {"traj_frame_a"},
        set_traj_frame_fn=lambda *_a, **_k: ("traj_frame_a", 2),
    )
    assert payload_interaction_fail["ok"] is False
    assert str(payload_interaction_fail["reason"]) == "preview_missing"
    assert "Could not build selected pose preview for interactions" in str(payload_interaction_fail["warning"])

    payload_runtime_fail = run_redocking_selected_interaction_runtime(
        rows=[
            {
                "pose_object": "pose_a",
                "receptor": "rec_a",
                "system": "WT",
                "system_member": "WT_model_001",
                "traj_state": 1,
            }
        ],
        row_index=0,
        get_known_objects_fn=lambda: {"redock_selected_pose_preview", "traj_frame_a"},
        set_traj_frame_fn=lambda *_a, **_k: ("traj_frame_a", 1),
        create_object_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("create boom")),
    )
    assert payload_runtime_fail["ok"] is False
    assert str(payload_runtime_fail["reason"]) == "runtime_error"
    assert "create boom" in str(payload_runtime_fail["error"])
    assert "Failed to compute interactions for selected redock pose" in str(payload_runtime_fail["warning"])


def test_redock_runtime_selected_interaction_warning_message_helper():
    warn_default = resolve_redocking_selected_interaction_warning_message(
        {},
        default_warning="fallback warning",
    )
    assert warn_default["show_warning"] is True
    assert str(warn_default["warning"]) == "fallback warning"
    assert str(warn_default["reason"]) == ""

    warn_payload = resolve_redocking_selected_interaction_warning_message(
        {"reason": "preview_missing", "warning": "preview not found"},
        default_warning="fallback warning",
    )
    assert warn_payload["show_warning"] is True
    assert str(warn_payload["warning"]) == "preview not found"
    assert str(warn_payload["reason"]) == "preview_missing"

    warn_empty = resolve_redocking_selected_interaction_warning_message(
        {"reason": "runtime_error", "warning": ""},
        default_warning="fallback warning",
    )
    assert warn_empty["show_warning"] is True
    assert str(warn_empty["warning"]) == "fallback warning"
    assert str(warn_empty["reason"]) == "runtime_error"


def test_redock_runtime_build_traj_view_plan_helper():
    runtime_systems = [
        ("WT", "wt_obj_b", "wt_src_b", "WT_model_002"),
        ("WT", "wt_obj_c", "wt_src_c", "WT_model_010"),
        ("WT", "wt_obj_a", "wt_src_a", "WT_model_001"),
        ("WT", "", "wt_src_c", "WT_model_003"),
        ("Mutant", "mut_obj_b", "mut_src_b", "Mut_model_010"),
        ("Mutant", "mut_obj_a", "mut_src_a", "Mut_model_001"),
        ("Other", "oth_obj", "oth_src", "Other_model_001"),
    ]
    payload = build_redocking_traj_view_plan(
        runtime_systems,
        labels=["WT", "Mutant"],
        timestamp_tag="20260304_120000_111111",
        safe_name_fn=lambda s: str(s).replace(" ", "_"),
        max_name_len=40,
    )
    assert str(payload["timestamp_tag"]) == "20260304_120000_111111"
    plan = dict(payload.get("views_plan", {}) or {})
    assert set(plan.keys()) == {"WT", "Mutant"}
    assert int(len(plan["WT"]["members"])) == 3
    assert int(len(plan["Mutant"]["members"])) == 2
    assert plan["WT"]["members"] == [
        ("wt_obj_a", "WT_model_001"),
        ("wt_obj_b", "WT_model_002"),
        ("wt_obj_c", "WT_model_010"),
    ]
    assert plan["Mutant"]["members"] == [
        ("mut_obj_a", "Mut_model_001"),
        ("mut_obj_b", "Mut_model_010"),
    ]
    assert str(plan["WT"]["traj_object"]).startswith("redock_wt_ensemble_traj_")
    assert len(str(plan["WT"]["traj_object"])) <= 40
    assert str(plan["WT"]["frame_object"]).startswith("redock_wt_traj_frame")
    assert str(plan["WT"]["pose_frame_object"]).startswith("redock_wt_traj_best_pose")
    assert str(plan["Mutant"]["traj_object"]).startswith("redock_mutant_ensemble_traj_")


def test_redock_runtime_sort_runtime_systems_helper():
    runtime_systems = [
        ("Mutant", "mut_obj_b", "mut_src", "Mut_model_010"),
        ("WT", "wt_obj_c", "wt_src", "WT_model_010"),
        ("WT", "wt_obj_b", "wt_src", "WT_model_002"),
        ("Mutant", "mut_obj_a", "mut_src", "Mut_model_001"),
        ("WT", "wt_obj_a", "wt_src", "WT_model_001"),
        ("Other", "oth_obj", "oth_src", "Other_model_003"),
    ]
    assert sort_redocking_runtime_systems(runtime_systems) == [
        ("WT", "wt_obj_a", "wt_src", "WT_model_001"),
        ("WT", "wt_obj_b", "wt_src", "WT_model_002"),
        ("WT", "wt_obj_c", "wt_src", "WT_model_010"),
        ("Mutant", "mut_obj_a", "mut_src", "Mut_model_001"),
        ("Mutant", "mut_obj_b", "mut_src", "Mut_model_010"),
        ("Other", "oth_obj", "oth_src", "Other_model_003"),
    ]


def test_redock_runtime_dedupe_runtime_systems_by_signature_helper():
    runtime_systems = [
        ("Mutant", "mut_obj_a", "mut_src", "Mutant_model_001"),
        ("Mutant", "mut_obj_b", "mut_src", "Mutant_model_002"),
        ("Mutant", "mut_obj_c", "mut_src", "Mutant_model_003"),
    ]

    def _sig(_label, rec_obj, _src, _member):
        if rec_obj in ("mut_obj_a", "mut_obj_b"):
            return "same"
        return f"sig:{rec_obj}"

    payload = dedupe_redocking_runtime_systems_by_signature(
        runtime_systems,
        signature_for_entry_fn=_sig,
    )
    assert payload["systems"] == [
        ("Mutant", "mut_obj_a", "mut_src", "Mutant_model_001"),
        ("Mutant", "mut_obj_c", "mut_src", "Mutant_model_003"),
    ]
    dropped = list(payload.get("dropped", []) or [])
    assert int(payload.get("duplicate_count", 0) or 0) == 1
    assert str(dropped[0]["member_tag"]) == "Mutant_model_002"
    assert str(dropped[0]["duplicate_of"]) == "Mutant_model_001"


def test_redock_runtime_traj_view_object_helpers():
    entry = build_redocking_traj_view_entry(
        traj_object="wt_traj",
        frame_object="wt_frame",
        pose_frame_object="wt_pose",
        combo_object="wt_combo",
        state_map={"WT_model_001": 1},
        n_states=3,
    )
    assert str(entry["traj_object"]) == "wt_traj"
    assert str(entry["frame_object"]) == "wt_frame"
    assert str(entry["pose_frame_object"]) == "wt_pose"
    assert str(entry["combo_object"]) == "wt_combo"
    assert int(entry["n_states"]) == 3
    assert int(entry["state_map"]["WT_model_001"]) == 1

    views = {
        "WT": entry,
        "Mutant": build_redocking_traj_view_entry(
            traj_object="mut_traj",
            frame_object="mut_frame",
            pose_frame_object="mut_pose",
            combo_object="",
            state_map={"Mut_model_001": 1},
            n_states=1,
        ),
    }
    objs = collect_redocking_traj_view_objects(views)
    assert objs == ["wt_frame", "wt_pose", "wt_combo", "wt_traj", "mut_frame", "mut_pose", "mut_traj"]

    objs2 = collect_redocking_traj_view_objects(
        views,
        object_keys=["traj_object", "frame_object"],
    )
    assert objs2 == ["wt_traj", "wt_frame", "mut_traj", "mut_frame"]


def test_redock_runtime_load_traj_members_helper():
    members = [
        ("wt_obj_a", "WT_model_001"),
        ("wt_obj_b", "WT_model_002"),
        ("", "WT_model_003"),
        ("wt_obj_c", "WT_model_004"),
    ]
    created = []

    def _count_atoms(rec_obj):
        if str(rec_obj) == "wt_obj_a":
            return 12
        if str(rec_obj) == "wt_obj_b":
            return 9
        if str(rec_obj) == "wt_obj_c":
            return 3
        return 0

    def _create(rec_obj, state_i, member_tag):
        created.append((str(rec_obj), int(state_i), str(member_tag)))
        return str(rec_obj) != "wt_obj_b"

    payload = load_redocking_traj_members(
        members,
        count_atoms_fn=_count_atoms,
        create_state_fn=_create,
    )
    assert payload["ok"] is True
    assert int(payload["n_states"]) == 2
    assert dict(payload["state_map"]) == {"WT_model_001": 1, "WT_model_004": 2}
    assert int(len(payload["loaded_members"])) == 2
    assert int(len(payload["skipped_members"])) == 2
    reasons = [str(x.get("reason", "")) for x in list(payload["skipped_members"] or [])]
    assert "create_failed" in reasons
    assert "empty_receptor" in reasons
    assert created == [
        ("wt_obj_a", 1, "WT_model_001"),
        ("wt_obj_b", 2, "WT_model_002"),
        ("wt_obj_c", 2, "WT_model_004"),
    ]

    bad_count = load_redocking_traj_members(
        members,
        count_atoms_fn=None,
        create_state_fn=_create,
    )
    assert bad_count["ok"] is False
    assert "count_atoms_fn is not callable" in str(bad_count["error"])

    bad_create = load_redocking_traj_members(
        members,
        count_atoms_fn=_count_atoms,
        create_state_fn=None,
    )
    assert bad_create["ok"] is False
    assert "create_state_fn is not callable" in str(bad_create["error"])


def test_redock_runtime_combo_object_and_state_builder_helpers():
    obj_name = build_redocking_combo_object_name(
        "WT System",
        timestamp_tag="20260304_130101_123456",
        safe_name_fn=lambda s: str(s).replace(" ", "_"),
        max_name_len=120,
    )
    assert " " not in str(obj_name)
    assert str(obj_name).startswith("redock_wt_system_traj_combo_")
    assert "20260304_130101_123456" in str(obj_name)

    calls = []

    def _create_state(dest_state, row_rec, pose_obj, pose_state, source):
        calls.append((int(dest_state), str(row_rec.get("pose_object", "")), str(pose_obj), int(pose_state), str(source)))
        return int(dest_state) != 2

    payload = build_redocking_combo_states(
        4,
        row_by_state={
            1: {"pose_object": "pose_1", "state": 3},
            "2": {"pose_object": "pose_2", "state": "4"},
        },
        fallback_row={"pose_object": "pose_fb", "state": "not_an_int"},
        create_state_fn=_create_state,
    )
    assert payload["ok"] is True
    assert payload["built_any"] is True
    assert [int(x.get("state", 0)) for x in list(payload["built_states"] or [])] == [1, 3, 4]
    skipped = list(payload["skipped_states"] or [])
    assert int(len(skipped)) == 1
    assert int(skipped[0]["state"]) == 2
    assert str(skipped[0]["reason"]) == "create_failed"
    assert calls == [
        (1, "pose_1", "pose_1", 3, "state_map"),
        (2, "pose_2", "pose_2", 4, "state_map"),
        (3, "pose_fb", "pose_fb", 1, "fallback"),
        (4, "pose_fb", "pose_fb", 1, "fallback"),
    ]

    payload_invalid = build_redocking_combo_states(
        2,
        row_by_state={1: "not_a_dict"},
        fallback_row=None,
        create_state_fn=_create_state,
    )
    reasons = [str(x.get("reason", "")) for x in list(payload_invalid.get("skipped_states", []) or [])]
    assert "invalid_row" in reasons
    assert "no_row" in reasons

    bad_payload = build_redocking_combo_states(
        2,
        row_by_state={},
        fallback_row=None,
        create_state_fn=None,
    )
    assert bad_payload["ok"] is False
    assert "create_state_fn is not callable" in str(bad_payload["error"])


def test_redock_runtime_aggregate_label_runs_and_csv(tmp_path):
    run1 = {
        "ok": True,
        "rows": [
            {
                "mode": 1,
                "state": 1,
                "affinity": -8.1,
                "rmsd_ref": 1.6,
                "receptor": "wt_rec",
                "receptor_source": "wt_src",
            }
        ],
        "best": {"mode": 1, "affinity": -8.1, "rmsd_ref": 1.6, "receptor": "wt_rec", "receptor_source": "wt_src"},
        "repeat_rows": [{"seed": 111, "pose_file": "a.pdbqt", "log_file": "a.log"}],
        "job_dir": "job_a",
        "receptor_object": "wt_rec",
        "flex_enabled": True,
    }
    run2 = {
        "ok": True,
        "rows": [
            {
                "mode": 1,
                "state": 1,
                "affinity": -7.3,
                "rmsd_ref": 1.1,
                "receptor": "wt_rec",
                "receptor_source": "wt_src",
            }
        ],
        "best": {"mode": 1, "affinity": -7.3, "rmsd_ref": 1.1, "receptor": "wt_rec", "receptor_source": "wt_src"},
        "repeat_rows": [{"seed": 222, "pose_file": "b.pdbqt", "log_file": "b.log"}],
        "job_dir": "job_b",
        "receptor_object": "wt_rec",
        "flex_enabled": False,
    }

    def _repeat_stats(vals, bootstrap_n=0, seed=0):
        _ = (bootstrap_n, seed)
        vv = [float(v) for v in list(vals or [])]
        n = int(len(vv))
        mean_v = float(sum(vv) / float(n)) if n > 0 else float("nan")
        return {"n": int(n), "mean": float(mean_v), "ci_low": -1.0, "ci_high": 1.0, "confidence": ("high" if n >= 2 else "low")}

    payload = aggregate_label_runs(
        system_label="WT",
        run_list=[run1, run2],
        repeat_statistics_fn=_repeat_stats,
        workspace_dir=str(tmp_path),
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        timestamp_tag="20260303_010101",
    )
    assert payload["ok"] is True
    assert payload["ensemble_mode"] is True
    assert int(payload["ensemble_members"]) == 2
    assert int(payload["repeat_count"]) == 2
    assert int(payload["repeat_best"]) == 1                  # better docking score (-8.1 vs -7.3)
    assert int(payload["repeat_stats_affinity"]["n"]) == 2
    assert int(payload["top1_success"]["n_success"]) == 2 and int(payload["top1_success"]["n"]) == 2
    assert int(payload["repeat_stats_rmsd"]["n"]) == 2
    assert str(payload["receptor_object"]) == "wt_rec"
    assert os.path.isfile(str(payload["repeat_csv"]))

    basic_csv = write_redocking_repeat_stats_csv(
        str(tmp_path / "basic_repeat_stats.csv"),
        payload["repeat_rows"],
        include_member_fields=False,
    )
    assert os.path.isfile(str(basic_csv))
    with open(str(basic_csv), "r", encoding="utf-8") as fh:
        hdr = fh.readline().strip()
    assert "seed" in hdr
    assert "model_member" not in hdr
    assert "top1_rmsd_a" in hdr and "min_rmsd_all_poses_a" in hdr


def test_redock_runtime_aggregate_label_runs_runtime_helper(tmp_path):
    calls = []

    def _aggregate(
        system_label,
        run_list,
        repeat_statistics_fn=None,
        workspace_dir="",
        safe_name_fn=None,
        timestamp_tag="",
    ):
        calls.append(
            {
                "system_label": str(system_label),
                "run_list": list(run_list or []),
                "repeat_statistics_fn": repeat_statistics_fn,
                "workspace_dir": str(workspace_dir),
                "safe_name_fn": safe_name_fn,
                "timestamp_tag": str(timestamp_tag),
            }
        )
        return {"ok": True, "rows": [{"system": str(system_label)}], "repeat_count": int(len(list(run_list or [])))}

    repeat_stats = lambda vals, bootstrap_n=0, seed=0: {"n": len(list(vals or []))}
    payload = aggregate_label_runs_runtime(
        system_label="WT",
        run_list=[{"ok": True}],
        repeat_statistics_fn=repeat_stats,
        workspace_dir_fn=lambda: str(tmp_path),
        safe_name_fn=lambda x: str(x),
        timestamp_tag_fn=lambda: "20260305_101010",
        aggregate_fn=_aggregate,
    )
    assert payload["ok"] is True
    assert int(payload["repeat_count"]) == 1
    assert int(len(calls)) == 1
    sent = dict(calls[0] or {})
    assert str(sent["system_label"]) == "WT"
    assert sent["run_list"] == [{"ok": True}]
    assert sent["repeat_statistics_fn"] is repeat_stats
    assert str(sent["workspace_dir"]) == str(tmp_path)
    assert callable(sent["safe_name_fn"])
    assert str(sent["timestamp_tag"]) == "20260305_101010"

    payload_fail = aggregate_label_runs_runtime(
        system_label="WT",
        run_list=[{"ok": True}],
        repeat_statistics_fn=repeat_stats,
        workspace_dir_fn=lambda: str(tmp_path),
        safe_name_fn=lambda x: str(x),
        timestamp_tag_fn=lambda: "x",
        aggregate_fn=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert payload_fail["ok"] is False
    assert str(payload_fail["error"]) == "boom"


def test_redock_runtime_summarize_repeat_rows_helper():
    rows = [
        {"ok": True, "best_affinity": -8.1, "best_rmsd": 1.2},
        {"ok": True, "best_affinity": -7.9, "best_rmsd": 1.5},
        {"ok": False, "best_affinity": -9.0, "best_rmsd": 0.9},
        {"ok": True, "best_affinity": 0.0, "best_rmsd": float("nan")},
    ]

    def _repeat_stats(vals, bootstrap_n=0, seed=0):
        _ = (bootstrap_n, seed)
        vv = [float(v) for v in list(vals or [])]
        n = int(len(vv))
        mean_v = float(sum(vv) / float(n)) if n > 0 else float("nan")
        return {"n": int(n), "mean": float(mean_v), "ci_low": -1.0, "ci_high": 1.0, "confidence": "high"}

    payload = summarize_redocking_repeat_rows(
        repeat_rows_meta=rows,
        repeat_statistics_fn=_repeat_stats,
        seed=999,
        bootstrap_n=111,
    )
    assert int(len(payload["aff_samples"])) == 3
    assert int(len(payload["rms_samples"])) == 2
    assert int(payload["aff_stats"]["n"]) == 3
    assert int(payload["rms_stats"]["n"]) == 2
    assert payload["affinity_zero_plateau"] is False

    zero_payload = summarize_redocking_repeat_rows(
        repeat_rows_meta=[
            {"ok": True, "best_affinity": 0.0, "best_rmsd": 1.0},
            {"ok": True, "best_affinity": 0.0, "best_rmsd": 1.1},
        ],
        repeat_statistics_fn=None,
    )
    assert zero_payload["affinity_zero_plateau"] is True
    assert int(zero_payload["aff_stats"]["n"]) == 2


def test_redock_runtime_prepare_repeat_outputs_helper(tmp_path):
    rows = [
        {"ok": True, "best_affinity": -8.1, "best_rmsd": 1.2, "seed": 111},
        {"ok": True, "best_affinity": -7.9, "best_rmsd": 1.5, "seed": 222},
    ]

    def _repeat_stats(vals, bootstrap_n=0, seed=0):
        _ = (bootstrap_n, seed)
        vv = [float(v) for v in list(vals or [])]
        n = int(len(vv))
        mean_v = float(sum(vv) / float(n)) if n > 0 else float("nan")
        return {"n": int(n), "mean": float(mean_v), "ci_low": -1.0, "ci_high": 1.0, "confidence": "high"}

    out_csv = str(tmp_path / "repeat_stats.csv")
    payload = prepare_redocking_repeat_outputs(
        repeat_rows_meta=rows,
        repeat_statistics_fn=_repeat_stats,
        seed=123,
        bootstrap_n=456,
        csv_path=out_csv,
        include_member_fields=False,
    )
    assert int(payload["repeat_stats_affinity"]["n"]) == 2
    assert int(payload["repeat_stats_rmsd"]["n"]) == 2
    assert bool(payload["affinity_zero_plateau"]) is False
    assert os.path.isfile(str(payload["repeat_csv"]))
    with open(str(payload["repeat_csv"]), "r", encoding="utf-8") as fh:
        hdr = fh.readline().strip()
    assert "seed" in hdr
    assert "model_member" not in hdr

    payload_no_csv = prepare_redocking_repeat_outputs(
        repeat_rows_meta=rows,
        repeat_statistics_fn=None,
        csv_path="",
    )
    assert str(payload_no_csv["repeat_csv"]) == ""
    assert int(payload_no_csv["repeat_stats_affinity"]["n"]) == 2


def test_redock_runtime_pose_view_and_row_attach_helpers(tmp_path):
    pose_name = build_redocking_pose_object_name(
        receptor_obj="Rec WT",
        tag="20260303_101010",
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        prefix="redock",
    )
    assert pose_name == "redock_Rec_WT_20260303_101010"

    src_pdbqt = tmp_path / "docked_poses.pdbqt"
    src_pdbqt.write_text("ATOM 1 C\n", encoding="utf-8")

    calls = {"sdf": 0, "pdb": 0}

    def _to_sdf(src, dst):
        _ = src
        calls["sdf"] += 1
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True

    def _to_pdb(_src, _dst):
        calls["pdb"] += 1
        return False

    view1 = resolve_redocking_pose_view_file(
        str(src_pdbqt),
        convert_pdbqt_to_sdf_fn=_to_sdf,
        convert_pdbqt_to_pdb_fn=_to_pdb,
    )
    assert view1.endswith("docked_poses_view.sdf")
    assert int(calls["sdf"]) == 1
    assert int(calls["pdb"]) == 0

    def _to_sdf_fail(_src, _dst):
        calls["sdf"] += 1
        return False

    def _to_pdb_ok(src, dst):
        _ = src
        calls["pdb"] += 1
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return True

    view2 = resolve_redocking_pose_view_file(
        str(src_pdbqt),
        convert_pdbqt_to_sdf_fn=_to_sdf_fail,
        convert_pdbqt_to_pdb_fn=_to_pdb_ok,
    )
    assert view2.endswith("docked_poses_view.pdb")
    assert int(calls["pdb"]) == 1

    src_sdf = tmp_path / "docked_poses.sdf"
    src_sdf.write_text("$$$$\n", encoding="utf-8")
    view3 = resolve_redocking_pose_view_file(str(src_sdf))
    assert str(view3) == str(src_sdf)

    rows = [{"mode": 1}, {"mode": 2}]
    attached = attach_pose_object_to_rows(rows, "pose_obj")
    assert int(len(attached)) == 2
    assert str(attached[0]["pose_object"]) == "pose_obj"
    assert str(attached[1]["pose_object"]) == "pose_obj"
    assert "pose_object" not in rows[0]


def test_redock_runtime_pose_load_wrapper_helpers():
    calls = []

    def _load_ok(pose_view, pose_obj):
        calls.append((str(pose_view), str(pose_obj)))
        return {"ok": True}

    ok_payload = load_redocking_pose_object_safe(
        pose_object="pose_obj",
        pose_view_file="pose_view.sdf",
        load_pose_object_fn=_load_ok,
    )
    assert ok_payload["ok"] is True
    assert calls == [("pose_view.sdf", "pose_obj")]

    def _load_fail(_pose_view, _pose_obj):
        raise RuntimeError("scene load failed")

    fail_payload = load_redocking_pose_object_safe(
        pose_object="pose_obj",
        pose_view_file="pose_view.sdf",
        load_pose_object_fn=_load_fail,
    )
    assert fail_payload["ok"] is False
    assert "scene load failed" in str(fail_payload["error"])

    bad_payload = load_redocking_pose_object_safe(
        pose_object="pose_obj",
        pose_view_file="pose_view.sdf",
        load_pose_object_fn=None,
    )
    assert bad_payload["ok"] is False
    assert "not callable" in str(bad_payload["error"])

    warn_txt = redocking_pose_load_warning_text("WT", "scene load failed")
    assert warn_txt == "Redocking pose load warning (WT): scene load failed"


def test_redock_runtime_prepare_pose_outputs_helper(tmp_path):
    src_pdbqt = tmp_path / "docked_poses.pdbqt"
    src_pdbqt.write_text("ATOM 1 C\n", encoding="utf-8")

    def _to_sdf(_src, dst):
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True

    calls = []

    def _load_ok(pose_view, pose_obj):
        calls.append((str(pose_view), str(pose_obj)))

    payload = prepare_redocking_pose_outputs(
        best_rows=[{"mode": 1, "affinity": -8.2}],
        best_pose_file=str(src_pdbqt),
        receptor_obj="Rec WT",
        system_label="WT",
        tag="20260303_123000",
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        convert_pdbqt_to_sdf_fn=_to_sdf,
        convert_pdbqt_to_pdb_fn=lambda _s, _d: False,
        load_pose_object_fn=_load_ok,
        prefix="redock",
    )
    assert str(payload["pose_object"]) == "redock_Rec_WT_20260303_123000"
    assert payload["pose_load"]["ok"] is True
    assert str(payload["load_warning"]) == ""
    assert int(len(payload["rows"])) == 1
    assert str(payload["rows"][0]["pose_object"]) == "redock_Rec_WT_20260303_123000"
    assert int(len(calls)) == 1
    assert str(calls[0][1]) == "redock_Rec_WT_20260303_123000"

    def _load_fail(_pose_view, _pose_obj):
        raise RuntimeError("cannot load pose")

    payload_fail = prepare_redocking_pose_outputs(
        best_rows=[{"mode": 1}],
        best_pose_file=str(src_pdbqt),
        receptor_obj="Rec WT",
        system_label="Mutant",
        tag="x",
        safe_name_fn=None,
        convert_pdbqt_to_sdf_fn=lambda _s, _d: False,
        convert_pdbqt_to_pdb_fn=lambda _s, _d: False,
        load_pose_object_fn=_load_fail,
    )
    assert payload_fail["pose_load"]["ok"] is False
    assert "Redocking pose load warning (Mutant): cannot load pose" in str(payload_fail["load_warning"])
    assert int(len(payload_fail["rows"])) == 1
    assert str(payload_fail["rows"][0]["pose_object"]) == "redock_Rec WT_x"


def test_redock_runtime_prep_validation_helpers():
    lig_ok = resolve_redocking_ligand_file(
        ligand_override_file="",
        picked_ligand_file="lig.pdbqt",
        file_exists_fn=lambda p: str(p) == "lig.pdbqt",
    )
    assert lig_ok["ok"] is True
    assert str(lig_ok["ligand_file"]) == "lig.pdbqt"

    lig_fail = resolve_redocking_ligand_file(
        ligand_override_file="",
        picked_ligand_file="missing.pdbqt",
        file_exists_fn=lambda _p: False,
    )
    assert lig_fail["ok"] is False
    assert "No ligand input is prepared" in str(lig_fail["error"])

    rec_cancel = validate_redocking_receptor_prepare_result(
        receptor_input="",
        rec_info={"error": "cancelled"},
        job_dir="job_a",
        file_exists_fn=lambda _p: False,
    )
    assert rec_cancel["ok"] is False
    assert rec_cancel["cancelled"] is True

    rec_fail = validate_redocking_receptor_prepare_result(
        receptor_input="missing_rec.pdbqt",
        rec_info={"notes": ["x"]},
        job_dir="job_b",
        file_exists_fn=lambda _p: False,
    )
    assert rec_fail["ok"] is False
    assert rec_fail["cancelled"] is False
    assert "Failed receptor preparation" in str(rec_fail["error"])

    rec_ok = validate_redocking_receptor_prepare_result(
        receptor_input="rec_ok.pdbqt",
        rec_info={"notes": ["ok"]},
        job_dir="job_c",
        file_exists_fn=lambda p: str(p) == "rec_ok.pdbqt",
    )
    assert rec_ok["ok"] is True
    assert str(rec_ok["receptor_input"]) == "rec_ok.pdbqt"

    rec_sel = resolve_redocking_receptor_for_docking(
        receptor_input="rec_input.pdbqt",
        stripped_file="rec_stripped.pdbqt",
        removed_reference_atoms=5,
        file_exists_fn=lambda p: str(p) == "rec_stripped.pdbqt",
    )
    assert str(rec_sel["receptor_for_dock"]) == "rec_stripped.pdbqt"
    assert int(rec_sel["removed_reference_atoms"]) == 5

    rec_sel_keep = resolve_redocking_receptor_for_docking(
        receptor_input="rec_input.pdbqt",
        stripped_file="missing.pdbqt",
        removed_reference_atoms=2,
        file_exists_fn=lambda _p: False,
    )
    assert str(rec_sel_keep["receptor_for_dock"]) == "rec_input.pdbqt"

    flex_sel_ok = validate_redocking_flex_selection(
        flex_enabled=True,
        flex_rows=[{"resi": "10"}],
        job_dir="job_d",
    )
    assert flex_sel_ok["ok"] is True
    assert int(len(flex_sel_ok["flex_rows"])) == 1

    flex_sel_fail = validate_redocking_flex_selection(
        flex_enabled=True,
        flex_rows=[],
        job_dir="job_e",
    )
    assert flex_sel_fail["ok"] is False
    assert "Flexible redocking is enabled" in str(flex_sel_fail["error"])

    flex_prep_ok = validate_redocking_flexible_prepare_result(
        rigid_in="rigid.pdbqt",
        flex_in="flex.pdbqt",
        flex_info={"ok": True},
        job_dir="job_f",
    )
    assert flex_prep_ok["ok"] is True
    assert str(flex_prep_ok["receptor_for_dock"]) == "rigid.pdbqt"
    assert str(flex_prep_ok["flex_file"]) == "flex.pdbqt"

    flex_prep_fail = validate_redocking_flexible_prepare_result(
        rigid_in="",
        flex_in="",
        flex_info={"message": "bad flex"},
        job_dir="job_g",
    )
    assert flex_prep_fail["ok"] is False
    assert "bad flex" in str(flex_prep_fail["error"])

    lig_prep_ok = validate_redocking_ligand_prepare_result(
        ligand_input="lig_ok.pdbqt",
        lig_info={"notes": ["ok"]},
        job_dir="job_h",
        file_exists_fn=lambda p: str(p) == "lig_ok.pdbqt",
    )
    assert lig_prep_ok["ok"] is True
    assert str(lig_prep_ok["ligand_input"]) == "lig_ok.pdbqt"

    lig_prep_fail = validate_redocking_ligand_prepare_result(
        ligand_input="missing_lig.pdbqt",
        lig_info={"error": "x"},
        job_dir="job_i",
        file_exists_fn=lambda _p: False,
    )
    assert lig_prep_fail["ok"] is False
    assert "Failed ligand preparation" in str(lig_prep_fail["error"])

    lig_prep_qc_fail = validate_redocking_ligand_prepare_result(
        ligand_input="lig_bad.pdbqt",
        lig_info={"qc": {"status": "red", "errors": ["heavy atom count changed"]}},
        job_dir="job_qc",
        file_exists_fn=lambda p: str(p) == "lig_bad.pdbqt",
    )
    assert lig_prep_qc_fail["ok"] is False
    assert "Ligand preparation QC failed" in str(lig_prep_qc_fail["error"])
    assert "heavy atom count changed" in str(lig_prep_qc_fail["error"])

    qc_msgs = redocking_ligand_qc_messages(
        {
            "qc": {
                "status": "yellow",
                "warnings": ["input ligand had no explicit hydrogens before preparation"],
                "errors": [],
            },
            "qc_report": "job/ligand_prep_qc.json",
        }
    )
    assert len(qc_msgs) == 1
    assert "Redocking ligand QC: YELLOW" in str(qc_msgs[0])
    assert "ligand_prep_qc.json" in str(qc_msgs[0])


def test_redock_runtime_charge_policy_and_smiles_hint_helpers():
    charge_switched = resolve_redocking_ligand_charge_policy(
        ligand_file="lig_in.sdf",
        preserve_input_charges=False,
        charge_mode="preserve",
        parameterized_ligand_mol2="lig_param.mol2",
        file_exists_fn=lambda p: str(p) == "lig_param.mol2",
        samefile_or_equal_fn=lambda _a, _b: False,
        chemically_equivalent_fn=lambda _a, _b: True,
    )
    assert str(charge_switched["ligand_file"]) == "lig_param.mol2"
    assert bool(charge_switched["preserve_input_charges"]) is True
    assert bool(charge_switched["used_parameterized_template"]) is True
    assert "Redocking ligand charge policy" in str(charge_switched["debug_message"])

    charge_pdbqt_switched = resolve_redocking_ligand_charge_policy(
        ligand_file="lig_ready.pdbqt",
        preserve_input_charges=True,
        charge_mode="preserve",
        parameterized_ligand_mol2="lig_param.mol2",
        file_exists_fn=lambda p: str(p) == "lig_param.mol2",
    )
    assert str(charge_pdbqt_switched["ligand_file"]) == "lig_ready.pdbqt"
    assert bool(charge_pdbqt_switched["preserve_input_charges"]) is True
    assert bool(charge_pdbqt_switched["used_parameterized_template"]) is False
    assert "Ligand Source Prep" in str(charge_pdbqt_switched["debug_message"])

    charge_kept = resolve_redocking_ligand_charge_policy(
        ligand_file="lig_in.sdf",
        preserve_input_charges=False,
        charge_mode="add",
        parameterized_ligand_mol2="lig_param.mol2",
        file_exists_fn=lambda _p: True,
        chemically_equivalent_fn=lambda _a, _b: True,
    )
    assert str(charge_kept["ligand_file"]) == "lig_in.sdf"
    assert bool(charge_kept["preserve_input_charges"]) is False
    assert bool(charge_kept["used_parameterized_template"]) is False
    assert str(charge_kept["debug_message"]) == ""

    smiles_pubchem = resolve_redocking_ligand_smiles_hint(
        ligand_override_file="override.sdf",
        ref_fields=("A", "101", "ATP"),
        pubchem_smiles_hint_fn=lambda: "C1=CC=CC=C1",
        rcsb_smiles_by_het_fn=lambda _rn: "N/A",
    )
    assert str(smiles_pubchem["smiles_hint"]) == "C1=CC=CC=C1"
    assert str(smiles_pubchem["hint_source"]) == "pubchem"
    assert str(smiles_pubchem["residue_name"]) == ""

    smiles_rcsb = resolve_redocking_ligand_smiles_hint(
        ligand_override_file="override.sdf",
        ref_fields=("B", "202", "hem"),
        pubchem_smiles_hint_fn=lambda: "",
        rcsb_smiles_by_het_fn=lambda rn: ("C1CCCCC1" if str(rn) == "HEM" else ""),
    )
    assert str(smiles_rcsb["smiles_hint"]) == "C1CCCCC1"
    assert str(smiles_rcsb["hint_source"]) == "rcsb"
    assert str(smiles_rcsb["residue_name"]) == "HEM"

    smiles_none = resolve_redocking_ligand_smiles_hint(
        ligand_override_file="",
        ref_fields=("A", "10", "LIG"),
        pubchem_smiles_hint_fn=lambda: "",
        rcsb_smiles_by_het_fn=lambda _rn: "CC",
    )
    assert str(smiles_none["smiles_hint"]) == ""
    assert str(smiles_none["hint_source"]) == "none"


def test_redock_runtime_prepare_ligand_context_helper():
    calls = []

    def _prepare_ligand_ok(src, job_dir, target_ph=7.4, smiles_hint="", preserve_input_charges=False):
        calls.append(
            {
                "src": str(src),
                "job_dir": str(job_dir),
                "target_ph": float(target_ph),
                "smiles_hint": str(smiles_hint),
                "preserve_input_charges": bool(preserve_input_charges),
            }
        )
        return "lig_prepared.pdbqt", {"notes": ["ok"]}

    payload = prepare_redocking_ligand_context(
        ligand_file="lig_in.sdf",
        job_dir="job_lig",
        target_ph=6.5,
        ligand_override_file="override.sdf",
        ref_fields=("A", "101", "atp"),
        preserve_input_charges=False,
        charge_mode="preserve",
        parameterized_ligand_mol2="lig_param.mol2",
        file_exists_fn=lambda p: str(p) in {"lig_param.mol2", "lig_prepared.pdbqt"},
        samefile_or_equal_fn=lambda _a, _b: False,
        chemically_equivalent_fn=lambda _a, _b: True,
        pubchem_smiles_hint_fn=lambda: "",
        rcsb_smiles_by_het_fn=lambda rn: ("C1CCCCC1" if str(rn) == "ATP" else ""),
        prepare_ligand_fn=_prepare_ligand_ok,
    )
    assert payload["ok"] is True
    assert str(payload["ligand_file"]) == "lig_param.mol2"
    assert bool(payload["preserve_input_charges"]) is True
    assert str(payload["smiles_hint"]) == "C1CCCCC1"
    assert str(payload["hint_source"]) == "rcsb"
    assert str(payload["residue_name"]) == "ATP"
    assert str(payload["ligand_input"]) == "lig_prepared.pdbqt"
    assert int(len(calls)) == 1
    assert str(calls[0]["src"]) == "lig_param.mol2"
    assert float(calls[0]["target_ph"]) == 6.5
    assert bool(calls[0]["preserve_input_charges"]) is True
    assert str(calls[0]["smiles_hint"]) == "C1CCCCC1"
    dbg = list(payload.get("debug_messages", []) or [])
    assert any("Redocking ligand charge policy" in str(m) for m in dbg)
    assert any("Redocking ligand bond-order hint from RCSB CCD: ATP" in str(m) for m in dbg)

    payload_fail = prepare_redocking_ligand_context(
        ligand_file="lig_in.sdf",
        job_dir="job_lig2",
        target_ph=7.4,
        ligand_override_file="",
        ref_fields=("A", "1", "LIG"),
        preserve_input_charges=False,
        charge_mode="add",
        parameterized_ligand_mol2="",
        file_exists_fn=lambda _p: False,
        samefile_or_equal_fn=None,
        chemically_equivalent_fn=None,
        pubchem_smiles_hint_fn=lambda: "",
        rcsb_smiles_by_het_fn=lambda _rn: "",
        prepare_ligand_fn=lambda *_a, **_k: ("missing_lig.pdbqt", {"error": "x"}),
    )
    assert payload_fail["ok"] is False
    assert "Failed ligand preparation" in str(payload_fail["error"])
    assert isinstance(payload_fail.get("debug_messages", []), list)


def test_redock_runtime_prepare_single_run_context_helper(tmp_path):
    mk_calls = []

    def _makedirs(path, exist_ok=False):
        mk_calls.append((str(path), bool(exist_ok)))
        os.makedirs(path, exist_ok=bool(exist_ok))

    payload = prepare_redocking_single_run_context(
        smina_bin="smina_bin",
        ligand_override_file="",
        picked_ligand_file="lig.sdf",
        workspace_dir=str(tmp_path),
        system_label="WT System",
        receptor_obj="Rec 01",
        params={"ph": 6.8},
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        file_exists_fn=lambda p: str(p) == "lig.sdf",
        makedirs_fn=_makedirs,
        timestamp_tag="20260303_123000",
    )
    assert payload["ok"] is True
    assert str(payload["smina_bin"]) == "smina_bin"
    assert str(payload["ligand_file"]) == "lig.sdf"
    assert str(payload["tag"]) == "20260303_123000"
    assert str(payload["job_dir"]).endswith("redock_WT_System_Rec_01_20260303_123000")
    assert os.path.isdir(str(payload["job_dir"]))
    assert float(payload["target_ph"]) == 6.8
    assert int(len(mk_calls)) == 1
    assert mk_calls[0][1] is True

    missing_smina = prepare_redocking_single_run_context(
        smina_bin="",
        ligand_override_file="",
        picked_ligand_file="lig.sdf",
        workspace_dir=str(tmp_path),
        system_label="WT",
        receptor_obj="Rec",
        params={"ph": 7.4},
        file_exists_fn=lambda _p: True,
    )
    assert missing_smina["ok"] is False
    assert "smina was not found in PATH." in str(missing_smina["error"])

    missing_lig = prepare_redocking_single_run_context(
        smina_bin="smina_bin",
        ligand_override_file="",
        picked_ligand_file="missing.sdf",
        workspace_dir=str(tmp_path),
        system_label="WT",
        receptor_obj="Rec",
        params={"ph": 7.4},
        file_exists_fn=lambda _p: False,
    )
    assert missing_lig["ok"] is False
    assert "No ligand input is prepared" in str(missing_lig["error"])


def test_redock_runtime_reference_ligand_export_plan_and_result_helpers(tmp_path):
    empty_sel = prepare_redocking_reference_ligand_export_plan(
        selection="",
        job_dir=str(tmp_path),
    )
    assert empty_sel["ok"] is False
    assert str(empty_sel["reason"]) == "empty_selection"
    assert "selection is empty" in str(empty_sel["error"]).lower()

    plan = prepare_redocking_reference_ligand_export_plan(
        selection="(ref_obj) and chain A and resi 101 and resn ATP",
        job_dir=str(tmp_path),
    )
    assert plan["ok"] is True
    assert str(plan["output_name"]) == "reference_ligand_redock.pdb"
    assert str(plan["output_path"]).endswith("reference_ligand_redock.pdb")

    save_fail = finalize_redocking_reference_ligand_export_result(
        output_path=str(plan["output_path"]),
        save_error="permission denied",
        file_exists=False,
        atom_count=None,
    )
    assert save_fail["ok"] is False
    assert str(save_fail["reason"]) == "save_failed"
    assert "Failed to export reference ligand: permission denied" in str(save_fail["message"])

    missing_out = finalize_redocking_reference_ligand_export_result(
        output_path=str(plan["output_path"]),
        save_error="",
        file_exists=False,
        atom_count=12,
    )
    assert missing_out["ok"] is False
    assert str(missing_out["reason"]) == "output_missing"
    assert str(missing_out["message"]) == "Reference ligand export failed"

    zero_atoms = finalize_redocking_reference_ligand_export_result(
        output_path=str(plan["output_path"]),
        save_error="",
        file_exists=True,
        atom_count=0,
    )
    assert zero_atoms["ok"] is False
    assert str(zero_atoms["reason"]) == "zero_atoms"
    assert str(zero_atoms["message"]) == "Reference ligand has zero atoms"

    ok_payload = finalize_redocking_reference_ligand_export_result(
        output_path=str(plan["output_path"]),
        save_error="",
        file_exists=True,
        atom_count=5,
    )
    assert ok_payload["ok"] is True
    assert str(ok_payload["reason"]) == "ok"
    assert str(ok_payload["message"]) == "ok"


def test_redock_runtime_reference_ligand_export_response_helper():
    from_plan_error = resolve_redocking_reference_ligand_export_response(
        export_plan={"ok": False, "reason": "empty_selection", "error": ""},
        export_result=None,
        fallback_output_path="",
    )
    assert from_plan_error["ok"] is False
    assert str(from_plan_error["reason"]) == "empty_selection"
    assert str(from_plan_error["message"]) == "Reference ligand selection is empty"
    assert str(from_plan_error["output_path"]) == ""

    from_result_error = resolve_redocking_reference_ligand_export_response(
        export_plan={"ok": True},
        export_result={"ok": False, "reason": "output_missing", "message": ""},
        fallback_output_path="/tmp/ref.pdb",
    )
    assert from_result_error["ok"] is False
    assert str(from_result_error["reason"]) == "output_missing"
    assert str(from_result_error["message"]) == "Reference ligand export failed"

    from_result_ok = resolve_redocking_reference_ligand_export_response(
        export_plan={"ok": True},
        export_result={"ok": True, "reason": "ok", "message": "ok", "output_path": ""},
        fallback_output_path="/tmp/ref.pdb",
    )
    assert from_result_ok["ok"] is True
    assert str(from_result_ok["reason"]) == "ok"
    assert str(from_result_ok["message"]) == "ok"
    assert str(from_result_ok["output_path"]) == "/tmp/ref.pdb"

    missing_result = resolve_redocking_reference_ligand_export_response(
        export_plan={"ok": True},
        export_result=None,
        fallback_output_path="",
    )
    assert missing_result["ok"] is False
    assert str(missing_result["reason"]) == "missing_export_result"
    assert str(missing_result["message"]) == "Reference ligand export failed"


def test_redock_runtime_reference_mol_plan_and_result_helpers(tmp_path):
    no_rdkit = prepare_redocking_reference_mol_plan(
        has_rdkit=False,
        selection="(ref_obj) and chain A and resi 101 and resn ATP",
        job_dir=str(tmp_path),
    )
    assert no_rdkit["ok"] is False
    assert str(no_rdkit["reason"]) == "rdkit_unavailable"
    assert str(no_rdkit["error"]) == "RDKit not available"

    empty_sel = prepare_redocking_reference_mol_plan(
        has_rdkit=True,
        selection="",
        job_dir=str(tmp_path),
    )
    assert empty_sel["ok"] is False
    assert str(empty_sel["reason"]) == "empty_selection"
    assert "selection is empty" in str(empty_sel["error"]).lower()

    plan = prepare_redocking_reference_mol_plan(
        has_rdkit=True,
        selection="(ref_obj) and chain A and resi 101 and resn ATP",
        job_dir=str(tmp_path),
    )
    assert plan["ok"] is True
    assert str(plan["output_name"]) == "reference_ligand.pdb"
    assert str(plan["output_path"]).endswith("reference_ligand.pdb")

    save_fail = finalize_redocking_reference_mol_result(
        save_error="permission denied",
        has_mol=False,
        atom_count=None,
        conformer_count=None,
    )
    assert save_fail["ok"] is False
    assert str(save_fail["reason"]) == "save_failed"
    assert "Failed to export reference ligand: permission denied" in str(save_fail["message"])

    read_fail = finalize_redocking_reference_mol_result(
        save_error="",
        has_mol=False,
        atom_count=None,
        conformer_count=None,
    )
    assert read_fail["ok"] is False
    assert str(read_fail["reason"]) == "rdkit_read_failed"
    assert str(read_fail["message"]) == "Failed to read reference ligand into RDKit"

    no_atoms = finalize_redocking_reference_mol_result(
        save_error="",
        has_mol=True,
        atom_count=0,
        conformer_count=1,
    )
    assert no_atoms["ok"] is False
    assert str(no_atoms["reason"]) == "no_atoms"
    assert str(no_atoms["message"]) == "Reference ligand contains no atoms"

    no_conformer = finalize_redocking_reference_mol_result(
        save_error="",
        has_mol=True,
        atom_count=6,
        conformer_count=0,
    )
    assert no_conformer["ok"] is False
    assert str(no_conformer["reason"]) == "no_conformer"
    assert str(no_conformer["message"]) == "Reference ligand has no 3D conformer"

    ok_payload = finalize_redocking_reference_mol_result(
        save_error="",
        has_mol=True,
        atom_count=6,
        conformer_count=1,
    )
    assert ok_payload["ok"] is True
    assert str(ok_payload["reason"]) == "ok"
    assert str(ok_payload["message"]) == "ok"


def test_redock_runtime_recover_reference_mol_from_pdb_prefers_template_bond_orders(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    assert int(AllChem.EmbedMolecule(mol, randomSeed=20260310)) == 0
    pdb_path = tmp_path / "benzene_ref.pdb"
    with open(str(pdb_path), "w", encoding="utf-8") as fh:
        fh.write(Chem.MolToPDBBlock(mol))

    payload = recover_redocking_reference_mol_from_pdb(
        str(pdb_path),
        smiles_hint="c1ccccc1",
        chem_module=Chem,
        all_chem_module=AllChem,
        select_best_fragment_fn=lambda m: m,
    )
    assert payload["ok"] is True
    assert str(payload["source"]) == "smiles_template"
    work = payload["mol"]
    assert int(work.GetNumConformers()) == 1
    assert int(sum(1 for bond in work.GetBonds() if bool(bond.GetIsAromatic()))) == 6

    missing = recover_redocking_reference_mol_from_pdb(
        str(tmp_path / "missing.pdb"),
        smiles_hint="c1ccccc1",
        chem_module=Chem,
        all_chem_module=AllChem,
        select_best_fragment_fn=lambda m: m,
    )
    assert missing["ok"] is False
    assert str(missing["message"]) == "reference_ligand_pdb_missing"


def test_redock_runtime_rmsd_mol_plan_and_result_helpers():
    no_rdkit = prepare_redocking_rmsd_mol_plan(
        has_rdkit=False,
        has_input_mol=True,
        min_atoms=3,
    )
    assert no_rdkit["ok"] is False
    assert str(no_rdkit["reason"]) == "rdkit_unavailable"

    no_mol = prepare_redocking_rmsd_mol_plan(
        has_rdkit=True,
        has_input_mol=False,
        min_atoms=3,
    )
    assert no_mol["ok"] is False
    assert str(no_mol["reason"]) == "missing_input_mol"

    prep = prepare_redocking_rmsd_mol_plan(
        has_rdkit=True,
        has_input_mol=True,
        min_atoms="bad",
    )
    assert prep["ok"] is True
    assert str(prep["reason"]) == "ok"
    assert int(prep["min_atoms"]) == 3

    missing_work = finalize_redocking_rmsd_mol_result(
        has_work_mol=False,
        atom_count=None,
        conformer_count=None,
        min_atoms=3,
    )
    assert missing_work["ok"] is False
    assert str(missing_work["reason"]) == "missing_work_mol"

    insufficient_atoms = finalize_redocking_rmsd_mol_result(
        has_work_mol=True,
        atom_count=2,
        conformer_count=1,
        min_atoms=3,
    )
    assert insufficient_atoms["ok"] is False
    assert str(insufficient_atoms["reason"]) == "insufficient_atoms"

    missing_conformer = finalize_redocking_rmsd_mol_result(
        has_work_mol=True,
        atom_count=6,
        conformer_count=0,
        min_atoms=3,
    )
    assert missing_conformer["ok"] is False
    assert str(missing_conformer["reason"]) == "missing_conformer"

    ok_payload = finalize_redocking_rmsd_mol_result(
        has_work_mol=True,
        atom_count=6,
        conformer_count=1,
        min_atoms=3,
    )
    assert ok_payload["ok"] is True
    assert str(ok_payload["reason"]) == "ok"


def test_redock_runtime_rmsd_mol_config_helper():
    defaults = build_redocking_rmsd_mol_config()
    assert defaults == {
        "min_atoms": 3,
    }

    tuned = build_redocking_rmsd_mol_config(
        overrides={"min_atoms": "6"},
    )
    assert tuned == {
        "min_atoms": 6,
    }

    sanitized = build_redocking_rmsd_mol_config(
        overrides={"min_atoms": 0},
        min_atoms=5,
    )
    assert sanitized == {
        "min_atoms": 1,
    }

    mapping_overrides = [("min_atoms", "8")]
    from_items = build_redocking_rmsd_mol_config(overrides=mapping_overrides)
    assert from_items == {
        "min_atoms": 8,
    }


def test_redock_runtime_resolve_rmsd_mol_helper():
    no_rdkit = resolve_redocking_rmsd_mol(
        has_rdkit=False,
        input_mol="mol",
        min_atoms=3,
    )
    assert no_rdkit["ok"] is False
    assert str(no_rdkit["reason"]) == "rdkit_unavailable"
    assert no_rdkit["work_mol"] is None

    class _Mol:
        def __init__(self, name, atoms, conformers):
            self.name = str(name)
            self._atoms = int(atoms)
            self._conformers = int(conformers)

        def GetNumAtoms(self):
            return int(self._atoms)

        def GetNumConformers(self):
            return int(self._conformers)

    calls = []

    def _clone(in_obj):
        calls.append(("clone", str(in_obj)))
        return _Mol("cloned", 6, 1)

    def _select(work_obj):
        calls.append(("select", work_obj.name))
        return _Mol("best", 5, 1)

    def _remove(work_obj):
        calls.append(("remove_h", work_obj.name))
        return _Mol("no_h", 4, 1)

    def _sanitize(work_obj):
        calls.append(("sanitize", work_obj.name))

    ok_payload = resolve_redocking_rmsd_mol(
        has_rdkit=True,
        input_mol="input_mol",
        min_atoms=3,
        clone_mol_fn=_clone,
        select_best_fragment_fn=_select,
        remove_hs_fn=_remove,
        sanitize_mol_fn=_sanitize,
        get_atom_count_fn=lambda m: int(m.GetNumAtoms()),
        get_conformer_count_fn=lambda m: int(m.GetNumConformers()),
    )
    assert ok_payload["ok"] is True
    assert str(ok_payload["reason"]) == "ok"
    assert str(ok_payload["work_mol"].name) == "no_h"
    assert calls == [
        ("clone", "input_mol"),
        ("select", "cloned"),
        ("remove_h", "best"),
        ("sanitize", "no_h"),
    ]

    small_payload = resolve_redocking_rmsd_mol(
        has_rdkit=True,
        input_mol="input_mol",
        min_atoms=3,
        clone_mol_fn=lambda _m: _Mol("small", 2, 1),
    )
    assert small_payload["ok"] is False
    assert str(small_payload["reason"]) == "insufficient_atoms"
    assert small_payload["work_mol"] is None

    direct_mol = _Mol("direct", 6, 1)
    direct_payload = resolve_redocking_rmsd_mol(
        has_rdkit=True,
        input_mol=direct_mol,
        min_atoms=3,
        clone_mol_fn=None,
        get_atom_count_fn=lambda m: int(m.GetNumAtoms()),
        get_conformer_count_fn=lambda m: int(m.GetNumConformers()),
    )
    assert direct_payload["ok"] is True
    assert direct_payload["work_mol"] is direct_mol


def test_redock_runtime_resolve_rdkit_rmsd_mol_helper():
    class _Mol:
        def __init__(self, name, atoms, conformers):
            self.name = str(name)
            self._atoms = int(atoms)
            self._conformers = int(conformers)

        def GetNumAtoms(self):
            return int(self._atoms)

        def GetNumConformers(self):
            return int(self._conformers)

    class _Chem:
        def __init__(self):
            self.calls = []

        def Mol(self, in_obj):
            self.calls.append(("Mol", str(getattr(in_obj, "name", in_obj))))
            return _Mol("cloned", 6, 1)

        def RemoveHs(self, work_obj):
            self.calls.append(("RemoveHs", work_obj.name))
            return _Mol("no_h", 4, 1)

        def SanitizeMol(self, work_obj):
            self.calls.append(("SanitizeMol", work_obj.name))

    chem = _Chem()
    select_calls = []

    def _select(work_obj):
        select_calls.append(work_obj.name)
        return _Mol("best", 5, 1)

    payload_ok = resolve_redocking_rdkit_rmsd_mol(
        has_rdkit=True,
        input_mol="input_mol",
        min_atoms=3,
        chem_module=chem,
        select_best_fragment_fn=_select,
    )
    assert payload_ok["ok"] is True
    assert str(payload_ok["reason"]) == "ok"
    assert str(payload_ok["work_mol"].name) == "no_h"
    assert select_calls == ["cloned"]
    assert chem.calls == [
        ("Mol", "input_mol"),
        ("RemoveHs", "best"),
        ("SanitizeMol", "no_h"),
    ]

    payload_no_rdkit = resolve_redocking_rdkit_rmsd_mol(
        has_rdkit=False,
        input_mol="input_mol",
        chem_module=chem,
    )
    assert payload_no_rdkit["ok"] is False
    assert str(payload_no_rdkit["reason"]) == "rdkit_unavailable"

    direct_mol = _Mol("direct", 6, 1)
    payload_direct = resolve_redocking_rdkit_rmsd_mol(
        has_rdkit=True,
        input_mol=direct_mol,
        min_atoms=3,
        chem_module=None,
    )
    assert payload_direct["ok"] is True
    assert payload_direct["work_mol"] is direct_mol


def test_redock_runtime_resolve_rdkit_rmsd_mol_with_config_helper():
    class _Mol:
        def __init__(self, name, atoms, conformers):
            self.name = str(name)
            self._atoms = int(atoms)
            self._conformers = int(conformers)

        def GetNumAtoms(self):
            return int(self._atoms)

        def GetNumConformers(self):
            return int(self._conformers)

    class _Chem:
        def Mol(self, _in_obj):
            return _Mol("cloned", 5, 1)

        def RemoveHs(self, _work_obj):
            return _Mol("no_h", 5, 1)

        def SanitizeMol(self, _work_obj):
            return None

    chem = _Chem()

    payload_fail = resolve_redocking_rdkit_rmsd_mol_with_config(
        has_rdkit=True,
        input_mol="input_mol",
        rmsd_mol_config={"min_atoms": "6"},
        chem_module=chem,
        default_min_atoms=3,
    )
    assert payload_fail["ok"] is False
    assert str(payload_fail["reason"]) == "insufficient_atoms"

    payload_ok = resolve_redocking_rdkit_rmsd_mol_with_config(
        has_rdkit=True,
        input_mol="input_mol",
        rmsd_mol_config={"min_atoms": "bad"},
        chem_module=chem,
        default_min_atoms=4,
    )
    assert payload_ok["ok"] is True
    assert str(payload_ok["reason"]) == "ok"

    direct_mol = _Mol("direct", 4, 1)
    payload_map = resolve_redocking_rdkit_rmsd_mol_with_config(
        has_rdkit=True,
        input_mol=direct_mol,
        rmsd_mol_config=[("min_atoms", "4")],
        chem_module=None,
        default_min_atoms=3,
    )
    assert payload_map["ok"] is True
    assert payload_map["work_mol"] is direct_mol

    payload_no_rdkit = resolve_redocking_rdkit_rmsd_mol_with_config(
        has_rdkit=False,
        input_mol="input_mol",
        rmsd_mol_config={"min_atoms": 3},
        chem_module=chem,
    )
    assert payload_no_rdkit["ok"] is False
    assert str(payload_no_rdkit["reason"]) == "rdkit_unavailable"


def test_redock_runtime_unique_atom_maps_helper():
    maps = [
        [(0, 1), (2, 3)],
        [(0, 1), (2, 3)],
        [(4, 5), (6, 7)],
        [],
        None,
        [("8", "9"), (10, 11)],
        [("bad", 1)],
    ]
    uniq = build_redocking_unique_atom_maps(maps, max_maps=3)
    assert uniq == [
        [(0, 1), (2, 3)],
        [(4, 5), (6, 7)],
        [(8, 9), (10, 11)],
    ]

    limited = build_redocking_unique_atom_maps(maps, max_maps=1)
    assert limited == [[(0, 1), (2, 3)]]

    min_clamped = build_redocking_unique_atom_maps(
        [
            [(1, 2), (3, 4)],
            [(5, 6), (7, 8)],
        ],
        max_maps=0,
    )
    assert min_clamped == [[(1, 2), (3, 4)]]


def test_redock_runtime_rmsd_no_fit_helper():
    class _Point:
        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class _Conformer:
        def __init__(self, pts):
            self._pts = list(pts)

        def GetAtomPosition(self, idx):
            return self._pts[int(idx)]

    class _Mol:
        def __init__(self, pts):
            self._conf = _Conformer(pts)

        def GetConformer(self):
            return self._conf

    ref = _Mol([_Point(0.0, 0.0, 0.0), _Point(1.0, 0.0, 0.0), _Point(0.0, 1.0, 0.0)])
    pose = _Mol([_Point(0.0, 0.0, 0.0), _Point(2.0, 0.0, 0.0), _Point(0.0, 1.0, 0.0)])
    amap = [(0, 0), (1, 1), (2, 2)]

    rms = compute_redocking_rmsd_no_fit(ref, pose, amap, min_pairs=3)
    assert abs(float(rms) - (1.0 / 3.0) ** 0.5) < 1.0e-12

    assert str(compute_redocking_rmsd_no_fit(ref, pose, [(0, 0), (1, 1)], min_pairs=3)) == "nan"
    assert str(compute_redocking_rmsd_no_fit(None, pose, amap, min_pairs=3)) == "nan"
    assert str(compute_redocking_rmsd_no_fit(ref, None, amap, min_pairs=3)) == "nan"
    assert str(compute_redocking_rmsd_no_fit(ref, pose, [(0, 0), ("bad", 1), (9, 9)], min_pairs=2)) == "nan"


def test_redock_runtime_best_rmsd_config_helper():
    defaults = build_redocking_best_rmsd_config()
    assert defaults == {
        "max_maps": 256,
        "min_pairs": 3,
    }

    tuned = build_redocking_best_rmsd_config(
        overrides={
            "max_maps": "100",
            "min_pairs": "4",
        }
    )
    assert tuned == {
        "max_maps": 100,
        "min_pairs": 4,
    }

    sanitized = build_redocking_best_rmsd_config(
        overrides={
            "max_maps": 0,
            "min_pairs": "bad",
        },
        max_maps=7,
        min_pairs=5,
    )
    assert sanitized == {
        "max_maps": 1,
        "min_pairs": 5,
    }

    mapping_overrides = [("max_maps", "12")]
    from_items = build_redocking_best_rmsd_config(overrides=mapping_overrides)
    assert from_items == {
        "max_maps": 12,
        "min_pairs": 3,
    }


def test_redock_runtime_resolve_best_rmsd_helper():
    assert str(
        resolve_redocking_best_rmsd(
            has_rdkit=False,
            ref_mol="ref",
            pose_mol="pose",
            prepare_rmsd_mol_fn=lambda m: m,
            candidate_atom_maps_fn=lambda _r, _p, _lim: [[(0, 0), (1, 1), (2, 2)]],
            compute_rmsd_no_fit_fn=lambda _r, _p, _am, _mp: 1.0,
        )
    ) == "nan"

    prep_calls = []
    map_calls = []
    rms_calls = []

    def _prep(mol_obj):
        prep_calls.append(str(mol_obj))
        if str(mol_obj) == "bad":
            return None
        return f"prep:{mol_obj}"

    def _maps_positional(ref_obj, pose_obj, limit):
        map_calls.append((str(ref_obj), str(pose_obj), int(limit)))
        return [[(0, 0), (1, 1), (2, 2)], [(0, 1), (1, 2)]]

    def _rmsd_positional(ref_obj, pose_obj, amap, min_pairs_i):
        rms_calls.append((str(ref_obj), str(pose_obj), list(amap), int(min_pairs_i)))
        if len(list(amap)) < 3:
            return float("nan")
        return 1.234

    best = resolve_redocking_best_rmsd(
        has_rdkit=True,
        ref_mol="ref",
        pose_mol="pose",
        prepare_rmsd_mol_fn=_prep,
        candidate_atom_maps_fn=_maps_positional,
        compute_rmsd_no_fit_fn=_rmsd_positional,
        max_maps=0,
        min_pairs=3,
    )
    assert abs(float(best) - 1.234) < 1.0e-12
    assert prep_calls == ["ref", "pose"]
    assert map_calls == [("prep:ref", "prep:pose", 1)]
    assert int(rms_calls[0][3]) == 3

    bad_prep = resolve_redocking_best_rmsd(
        has_rdkit=True,
        ref_mol="bad",
        pose_mol="pose",
        prepare_rmsd_mol_fn=_prep,
        candidate_atom_maps_fn=_maps_positional,
        compute_rmsd_no_fit_fn=_rmsd_positional,
    )
    assert str(bad_prep) == "nan"


def test_redock_runtime_candidate_atom_maps_helper():
    payload_small = build_redocking_candidate_atom_maps(
        nref=2,
        npose=2,
        max_maps=10,
    )
    assert payload_small["ok"] is True
    assert payload_small["atom_maps"] == []
    assert str(payload_small["strategy"]) == "none"

    payload_forward = build_redocking_candidate_atom_maps(
        nref=3,
        npose=3,
        max_maps=5,
        forward_matches_fn=lambda _lim: [(2, 0, 1), (2, 0, 1)],
        reverse_matches_fn=lambda _lim: [(1, 2, 0)],
    )
    assert str(payload_forward["strategy"]) == "forward"
    assert payload_forward["atom_maps"] == [[(0, 2), (1, 0), (2, 1)]]

    payload_reverse = build_redocking_candidate_atom_maps(
        nref=3,
        npose=3,
        max_maps=5,
        forward_matches_fn=lambda _lim: [],
        reverse_matches_fn=lambda _lim: [(1, 2, 0)],
    )
    assert str(payload_reverse["strategy"]) == "reverse"
    assert payload_reverse["atom_maps"] == [[(1, 0), (2, 1), (0, 2)]]

    payload_bond = build_redocking_candidate_atom_maps(
        nref=3,
        npose=3,
        max_maps=5,
        forward_matches_fn=lambda _lim: [],
        reverse_matches_fn=lambda _lim: [],
        bond_order_matches_fn=lambda _lim: [(0, 1, 2)],
    )
    assert str(payload_bond["strategy"]) == "bond_order"
    assert payload_bond["atom_maps"] == [[(0, 0), (1, 1), (2, 2)]]

    payload_mcs = build_redocking_candidate_atom_maps(
        nref=4,
        npose=4,
        max_maps=5,
        forward_matches_fn=lambda _lim: [],
        reverse_matches_fn=lambda _lim: [],
        bond_order_matches_fn=lambda _lim: [],
        mcs_payload_fn=lambda: {
            "num_atoms": 4,
            "smarts": "[#6]-[#6]",
            "ref_matches": [(10, 11, 12, 13)],
            "pose_matches": [(0, 1, 2, 3)],
        },
    )
    assert str(payload_mcs["strategy"]) == "mcs"
    assert bool(payload_mcs["used_mcs"]) is True
    assert payload_mcs["atom_maps"] == [[(0, 10), (1, 11), (2, 12), (3, 13)]]

    payload_mcs_filtered = build_redocking_candidate_atom_maps(
        nref=10,
        npose=10,
        max_maps=5,
        mcs_payload_fn=lambda: {
            "num_atoms": 4,
            "smarts": "[#6]-[#6]",
            "ref_matches": [(1, 2, 3, 4)],
            "pose_matches": [(0, 1, 2, 3)],
        },
    )
    assert payload_mcs_filtered["atom_maps"] == []
    assert str(payload_mcs_filtered["strategy"]) == "none"


def test_redock_runtime_candidate_map_config_helper():
    defaults = build_redocking_candidate_map_config()
    assert defaults == {
        "min_atoms": 4,
        "mcs_match_limit": 32,
        "mcs_timeout": 8,
        "mcs_coverage_ratio": 0.85,
    }

    tuned = build_redocking_candidate_map_config(
        overrides={
            "min_atoms": "6",
            "mcs_match_limit": "12",
            "mcs_timeout": "15",
            "mcs_coverage_ratio": "0.92",
        }
    )
    assert tuned == {
        "min_atoms": 6,
        "mcs_match_limit": 12,
        "mcs_timeout": 15,
        "mcs_coverage_ratio": 0.92,
    }

    sanitized = build_redocking_candidate_map_config(
        overrides={
            "min_atoms": "bad",
            "mcs_match_limit": 0,
            "mcs_timeout": -3,
            "mcs_coverage_ratio": "nan",
        },
        min_atoms=5,
        mcs_match_limit=7,
        mcs_timeout=9,
        mcs_coverage_ratio=0.7,
    )
    assert sanitized == {
        "min_atoms": 5,
        "mcs_match_limit": 1,
        "mcs_timeout": 1,
        "mcs_coverage_ratio": 0.7,
    }

    mapping_overrides = [("min_atoms", "8"), ("mcs_match_limit", "11")]
    from_items = build_redocking_candidate_map_config(overrides=mapping_overrides)
    assert from_items["min_atoms"] == 8
    assert from_items["mcs_match_limit"] == 11
    assert from_items["mcs_timeout"] == 8
    assert abs(float(from_items["mcs_coverage_ratio"]) - 0.85) < 1.0e-12


def test_redock_runtime_resolve_candidate_atom_maps_helper():
    class _BareMol:
        def __init__(self, name, atoms):
            self.name = str(name)
            self.atoms = int(atoms)

    assert resolve_redocking_candidate_atom_maps(ref_mol=None, pose_mol="pose") == []

    bundle_calls = []
    uniq_calls = []

    def _build_bundle(ref_obj, pose_obj, nref, npose, cfg):
        bundle_calls.append((ref_obj.name, pose_obj.name, int(nref), int(npose), dict(cfg)))
        return {
            "forward_matches_fn": lambda _lim: [(2, 0, 1), (1, 2, 0)],
            "reverse_matches_fn": lambda _lim: [(1, 2, 0)],
        }

    def _uniq(maps, max_maps):
        uniq_calls.append((list(maps), int(max_maps)))
        return list(maps)

    resolved = resolve_redocking_candidate_atom_maps(
        ref_mol=_BareMol("ref", 3),
        pose_mol=_BareMol("pose", 3),
        max_maps=0,
        candidate_map_config={
            "min_atoms": "6",
            "mcs_match_limit": "9",
            "mcs_timeout": "11",
            "mcs_coverage_ratio": "0.91",
        },
        build_callback_bundle_fn=_build_bundle,
        unique_atom_maps_fn=_uniq,
        get_atom_count_fn=lambda mol: int(mol.atoms),
    )
    assert resolved == [[(0, 2), (1, 0), (2, 1)]]
    assert bundle_calls[0] == (
        "ref",
        "pose",
        3,
        3,
        {
            "min_atoms": 6,
            "mcs_match_limit": 9,
            "mcs_timeout": 11,
            "mcs_coverage_ratio": 0.91,
        },
    )
    assert uniq_calls[0][1] == 1

    failed_bundle = resolve_redocking_candidate_atom_maps(
        ref_mol=_BareMol("ref", 3),
        pose_mol=_BareMol("pose", 3),
        build_callback_bundle_fn=lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
        get_atom_count_fn=lambda mol: int(mol.atoms),
    )
    assert failed_bundle == []


def test_redock_runtime_resolve_rdkit_candidate_atom_maps_helper():
    class _Mol:
        def __init__(self, name, atoms, match_map):
            self.name = str(name)
            self._atoms = int(atoms)
            self._match_map = dict(match_map or {})

        def GetNumAtoms(self):
            return int(self._atoms)

        def GetSubstructMatches(self, pattern_obj, uniquify=False, useChirality=False):
            _ = bool(uniquify), bool(useChirality)
            p_name = str(getattr(pattern_obj, "name", pattern_obj))
            return list(self._match_map.get(p_name, []))

    ref = _Mol("ref", 3, {"pose": [(2, 0, 1), (2, 0, 1)]})
    pose = _Mol("pose", 3, {"ref": [(1, 2, 0)]})

    uniq_calls = []

    def _uniq(maps, max_maps):
        uniq_calls.append(int(max_maps))
        return build_redocking_unique_atom_maps(maps, max_maps)

    resolved = resolve_redocking_rdkit_candidate_atom_maps(
        ref_mol=ref,
        pose_mol=pose,
        max_maps=5,
        candidate_map_config={"min_atoms": "bad", "mcs_match_limit": "11"},
        chem_module=None,
        all_chem_module=None,
        mcs_module=None,
        unique_atom_maps_fn=_uniq,
    )
    assert resolved == [[(0, 2), (1, 0), (2, 1)]]
    assert uniq_calls == [5]

    assert resolve_redocking_rdkit_candidate_atom_maps(ref_mol=None, pose_mol=pose) == []


def test_redock_runtime_directional_match_functions_helper():
    empty_payload = build_redocking_directional_match_functions(
        ref_mol=None,
        pose_mol="pose",
        get_substruct_matches_fn=lambda *_args: [(0, 1, 2)],
    )
    assert empty_payload["forward_matches_fn"](5) == []
    assert empty_payload["reverse_matches_fn"](5) == []

    calls = []

    def _get_matches(query_obj, pattern_obj, uniq, chir):
        calls.append((query_obj, pattern_obj, bool(uniq), bool(chir)))
        if str(query_obj) == "ref":
            return [(2, 1, 0), (0, 1, 2)]
        return [(1, 0, 2), (2, 0, 1)]

    payload = build_redocking_directional_match_functions(
        ref_mol="ref",
        pose_mol="pose",
        get_substruct_matches_fn=_get_matches,
        uniquify=False,
        use_chirality=True,
    )

    forward_matches = payload["forward_matches_fn"]("bad_limit")
    reverse_matches = payload["reverse_matches_fn"](1)
    assert forward_matches == [(2, 1, 0)]
    assert reverse_matches == [(1, 0, 2)]
    assert calls[0] == ("ref", "pose", False, True)
    assert calls[1] == ("pose", "ref", False, True)

    broken = build_redocking_directional_match_functions(
        ref_mol="ref",
        pose_mol="pose",
        get_substruct_matches_fn=lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert broken["forward_matches_fn"](3) == []
    assert broken["reverse_matches_fn"](3) == []


def test_redock_runtime_candidate_map_callback_bundle_helper():
    calls = []

    def _directional(query_obj, pattern_obj, uniq, chir):
        calls.append(("directional", query_obj, pattern_obj, bool(uniq), bool(chir)))
        if str(query_obj) == "ref" and str(pattern_obj) == "pose":
            return [(2, 1, 0), (0, 1, 2)]
        if str(query_obj) == "pose" and str(pattern_obj) == "ref":
            return [(1, 0, 2), (2, 0, 1)]
        if str(query_obj) == "ref" and str(pattern_obj) == "pose_bo":
            return [(0, 1, 2), (2, 1, 0)]
        if str(query_obj) == "ref" and str(pattern_obj) == "mcs_patt":
            return [(10, 11, 12, 13), (20, 21, 22, 23)]
        if str(query_obj) == "pose" and str(pattern_obj) == "mcs_patt":
            return [(0, 1, 2, 3), (4, 5, 6, 7)]
        return []

    def _assign(ref_obj, pose_obj):
        calls.append(("assign", ref_obj, pose_obj))
        return "pose_bo"

    def _sanitize(pose_bo_obj):
        calls.append(("sanitize", pose_bo_obj))

    class _Mcs:
        def __init__(self):
            self.numAtoms = 4
            self.smartsString = "[#6]-[#6]"

    bundle = build_redocking_candidate_map_callback_bundle(
        ref_mol="ref",
        pose_mol="pose",
        nref=3,
        npose=3,
        min_atoms=4,
        mcs_match_limit=1,
        directional_get_substruct_matches_fn=_directional,
        directional_uniquify=False,
        directional_use_chirality=True,
        assign_bond_orders_fn=_assign,
        sanitize_mol_fn=_sanitize,
        find_mcs_fn=lambda _r, _p: _Mcs(),
        mcs_num_atoms_fn=lambda m: int(m.numAtoms),
        mcs_smarts_fn=lambda m: str(m.smartsString),
        mcs_mol_from_smarts_fn=lambda _smarts: "mcs_patt",
    )

    assert bundle["forward_matches_fn"](1) == [(2, 1, 0)]
    assert bundle["reverse_matches_fn"](2) == [(1, 0, 2), (2, 0, 1)]
    assert bundle["bond_order_matches_fn"](2) == [(0, 1, 2), (2, 1, 0)]
    mcs_payload = bundle["mcs_payload_fn"]()
    assert int(mcs_payload["num_atoms"]) == 4
    assert mcs_payload["ref_matches"] == [(10, 11, 12, 13)]
    assert mcs_payload["pose_matches"] == [(0, 1, 2, 3)]
    assert calls[0] == ("directional", "ref", "pose", False, True)
    assert calls[1] == ("directional", "pose", "ref", False, True)
    assert calls[2] == ("assign", "ref", "pose")
    assert calls[3] == ("sanitize", "pose_bo")
    assert calls[4] == ("directional", "ref", "pose_bo", False, False)

    minimal_bundle = build_redocking_candidate_map_callback_bundle(
        ref_mol="ref",
        pose_mol="pose",
        directional_get_substruct_matches_fn=None,
    )
    assert minimal_bundle["forward_matches_fn"](5) == []
    assert minimal_bundle["reverse_matches_fn"](5) == []
    assert minimal_bundle["bond_order_matches_fn"](5) == []
    assert minimal_bundle["mcs_payload_fn"]() == {}


def test_redock_runtime_rdkit_candidate_map_callback_bundle_helper():
    class _McsResult:
        def __init__(self, atoms=4, smarts="[#6]-[#6]"):
            self.numAtoms = int(atoms)
            self.smartsString = str(smarts)

    class _Pattern:
        def __init__(self, name):
            self.name = str(name)

    class _Mol:
        def __init__(self, name, match_map):
            self.name = str(name)
            self.match_map = dict(match_map or {})
            self.calls = []

        def GetSubstructMatches(self, pattern_obj, uniquify=False, useChirality=False):
            p_name = str(getattr(pattern_obj, "name", pattern_obj))
            self.calls.append((p_name, bool(uniquify), bool(useChirality)))
            return list(self.match_map.get(p_name, []))

    class _Chem:
        def __init__(self):
            self.mol_calls = []
            self.sanitize_calls = []
            self.smarts_calls = []

        def Mol(self, pose_obj):
            self.mol_calls.append(str(getattr(pose_obj, "name", pose_obj)))
            return _Mol("pose_copy", {})

        def SanitizeMol(self, pose_obj):
            self.sanitize_calls.append(str(getattr(pose_obj, "name", pose_obj)))

        def MolFromSmarts(self, smarts):
            self.smarts_calls.append(str(smarts))
            return _Pattern("mcs_pattern")

    class _AllChem:
        def __init__(self):
            self.assign_calls = []

        def AssignBondOrdersFromTemplate(self, ref_obj, pose_obj):
            self.assign_calls.append((str(getattr(ref_obj, "name", ref_obj)), str(getattr(pose_obj, "name", pose_obj))))
            return _Mol("pose_bo", {})

    class _FMCS:
        class AtomCompare:
            CompareElements = "cmp_atoms"

        class BondCompare:
            CompareOrder = "cmp_bonds"

        def __init__(self):
            self.calls = []

        def FindMCS(self, mols, **kwargs):
            self.calls.append(([str(getattr(m, "name", m)) for m in list(mols or [])], dict(kwargs or {})))
            return _McsResult(4, "[#6]-[#6]")

    ref = _Mol(
        "ref",
        {
            "pose": [(2, 1, 0), (0, 1, 2)],
            "pose_bo": [(0, 1, 2), (2, 1, 0)],
            "mcs_pattern": [(10, 11, 12, 13), (20, 21, 22, 23)],
        },
    )
    pose = _Mol(
        "pose",
        {
            "ref": [(1, 0, 2), (2, 0, 1)],
            "mcs_pattern": [(0, 1, 2, 3), (4, 5, 6, 7)],
        },
    )
    chem = _Chem()
    all_chem = _AllChem()
    fmcs = _FMCS()

    bundle = build_redocking_rdkit_candidate_map_callback_bundle(
        ref_mol=ref,
        pose_mol=pose,
        nref=3,
        npose=3,
        chem_module=chem,
        all_chem_module=all_chem,
        mcs_module=fmcs,
        min_atoms=4,
        mcs_match_limit=1,
        mcs_timeout=9,
    )

    assert bundle["forward_matches_fn"](1) == [(2, 1, 0)]
    assert bundle["reverse_matches_fn"](2) == [(1, 0, 2), (2, 0, 1)]
    assert bundle["bond_order_matches_fn"](2) == [(0, 1, 2), (2, 1, 0)]

    mcs_payload = dict(bundle["mcs_payload_fn"]() or {})
    assert int(mcs_payload["num_atoms"]) == 4
    assert str(mcs_payload["smarts"]) == "[#6]-[#6]"
    assert mcs_payload["ref_matches"] == [(10, 11, 12, 13)]
    assert mcs_payload["pose_matches"] == [(0, 1, 2, 3)]

    assert all_chem.assign_calls == [("ref", "pose_copy")]
    assert chem.mol_calls == ["pose"]
    assert chem.sanitize_calls == ["pose_bo"]
    assert chem.smarts_calls == ["[#6]-[#6]"]
    assert fmcs.calls[0][0] == ["ref", "pose"]
    assert int(fmcs.calls[0][1]["timeout"]) == 9
    assert bool(fmcs.calls[0][1]["ringMatchesRingOnly"]) is True
    assert bool(fmcs.calls[0][1]["completeRingsOnly"]) is True

    minimal_bundle = build_redocking_rdkit_candidate_map_callback_bundle(
        ref_mol=ref,
        pose_mol=pose,
        nref=3,
        npose=3,
        chem_module=None,
        all_chem_module=None,
        mcs_module=None,
    )
    assert minimal_bundle["forward_matches_fn"](5) == [(2, 1, 0), (0, 1, 2)]
    assert minimal_bundle["reverse_matches_fn"](5) == [(1, 0, 2), (2, 0, 1)]
    assert minimal_bundle["bond_order_matches_fn"](5) == []
    assert minimal_bundle["mcs_payload_fn"]() == {}


def test_redock_runtime_rdkit_callback_bundle_for_config_helper():
    class _Mol:
        def __init__(self, name, match_map):
            self.name = str(name)
            self.match_map = dict(match_map or {})

        def GetSubstructMatches(self, pattern_obj, uniquify=False, useChirality=False):
            _ = bool(uniquify), bool(useChirality)
            p_name = str(getattr(pattern_obj, "name", pattern_obj))
            return list(self.match_map.get(p_name, []))

    ref = _Mol("ref", {"pose": [(2, 1, 0), (0, 1, 2)]})
    pose = _Mol("pose", {"ref": [(1, 0, 2), (2, 0, 1)]})

    bundle = build_redocking_rdkit_callback_bundle_for_config(
        ref,
        pose,
        3,
        3,
        {"min_atoms": "6", "mcs_match_limit": "7", "mcs_timeout": "9"},
        chem_module=None,
        all_chem_module=None,
        mcs_module=None,
    )
    assert bundle["forward_matches_fn"](1) == [(2, 1, 0)]
    assert bundle["reverse_matches_fn"](2) == [(1, 0, 2), (2, 0, 1)]
    assert bundle["bond_order_matches_fn"](5) == []
    assert bundle["mcs_payload_fn"]() == {}

    bundle_bad_cfg = build_redocking_rdkit_callback_bundle_for_config(
        ref,
        pose,
        3,
        3,
        [("min_atoms", "8")],
        chem_module=None,
        all_chem_module=None,
        mcs_module=None,
    )
    assert bundle_bad_cfg["forward_matches_fn"](5) == [(2, 1, 0), (0, 1, 2)]


def test_redock_runtime_bond_order_matches_and_mcs_payload_helpers():
    payload_skip = build_redocking_bond_order_matches(
        ref_mol="ref",
        pose_mol="pose",
        nref=3,
        npose=4,
        max_maps=5,
        assign_bond_orders_fn=lambda _r, _p: {"probe": True},
        forward_matches_for_mol_fn=lambda _r, _p, _lim: [(0, 1, 2)],
    )
    assert payload_skip == []

    calls = []

    def _assign(ref_obj, pose_obj):
        calls.append(("assign", ref_obj, pose_obj))
        return {"probe": True}

    def _sanitize(probe_obj):
        calls.append(("sanitize", probe_obj))

    def _forward(ref_obj, probe_obj, lim):
        calls.append(("forward", ref_obj, probe_obj, int(lim)))
        return [(0, 1, 2), (2, 1, 0)]

    payload_ok = build_redocking_bond_order_matches(
        ref_mol="ref",
        pose_mol="pose",
        nref=3,
        npose=3,
        max_maps=1,
        assign_bond_orders_fn=_assign,
        sanitize_mol_fn=_sanitize,
        forward_matches_for_mol_fn=_forward,
    )
    assert payload_ok == [(0, 1, 2)]
    assert calls[0][0] == "assign"
    assert calls[1][0] == "sanitize"
    assert calls[2][0] == "forward"

    payload_no_assign = build_redocking_bond_order_matches(
        ref_mol="ref",
        pose_mol="pose",
        nref=3,
        npose=3,
        max_maps=3,
        assign_bond_orders_fn=None,
        forward_matches_for_mol_fn=_forward,
    )
    assert payload_no_assign == []

    class _Mcs:
        def __init__(self, num_atoms, smarts):
            self.numAtoms = int(num_atoms)
            self.smartsString = str(smarts)

    missing_find = build_redocking_mcs_payload(
        ref_mol="ref",
        pose_mol="pose",
        find_mcs_fn=None,
    )
    assert missing_find == {}

    small_mcs = build_redocking_mcs_payload(
        ref_mol="ref",
        pose_mol="pose",
        min_atoms=4,
        find_mcs_fn=lambda _r, _p: _Mcs(3, "[#6]-[#6]"),
        mcs_num_atoms_fn=lambda m: int(m.numAtoms),
        mcs_smarts_fn=lambda m: str(m.smartsString),
    )
    assert int(small_mcs["num_atoms"]) == 3
    assert small_mcs["ref_matches"] == []
    assert small_mcs["pose_matches"] == []

    full_mcs = build_redocking_mcs_payload(
        ref_mol="ref",
        pose_mol="pose",
        min_atoms=4,
        mcs_match_limit=2,
        find_mcs_fn=lambda _r, _p: _Mcs(4, "[#6]-[#6]"),
        mcs_num_atoms_fn=lambda m: int(m.numAtoms),
        mcs_smarts_fn=lambda m: str(m.smartsString),
        mol_from_smarts_fn=lambda _s: "patt_obj",
        get_substruct_matches_fn=lambda mol, _patt, lim, _uniq: (
            [(10, 11, 12), (20, 21, 22), (30, 31, 32)]
            if str(mol) == "ref"
            else [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        )[: int(lim)],
    )
    assert int(full_mcs["num_atoms"]) == 4
    assert str(full_mcs["smarts"]) == "[#6]-[#6]"
    assert full_mcs["ref_matches"] == [(10, 11, 12), (20, 21, 22)]
    assert full_mcs["pose_matches"] == [(0, 1, 2), (3, 4, 5)]


def test_redock_runtime_prepare_base_and_ligand_override_and_ref_flex_helpers(tmp_path):
    grid_calls = []

    def _grid_from_ref(ref_obj, ref_fields):
        grid_calls.append((str(ref_obj), tuple(ref_fields or ())))
        return (1.1, 2.2, 3.3), (18.0, 18.0, 18.0), 42

    payload_base = prepare_redocking_base_payload(
        ref_obj="ref_obj",
        wt_obj="wt_obj",
        ref_fields=("A", "101", "ATP"),
        run_mutant=True,
        mut_obj="mut_obj",
        params={"ph": 7.4, "repeats": 3},
        use_modeler_wt=True,
        modeler_wt_obj="wt_modeler_obj",
        grid_from_reference_fn=_grid_from_ref,
    )
    assert payload_base["ok"] is True
    assert payload_base["used_modeler_wt"] is True
    assert int(len(grid_calls)) == 1
    base = dict(payload_base.get("payload", {}) or {})
    assert str(base["ref_obj"]) == "ref_obj"
    assert str(base["wt_obj"]) == "wt_modeler_obj"
    assert tuple(base["center"]) == (1.1, 2.2, 3.3)
    assert tuple(base["size"]) == (18.0, 18.0, 18.0)
    assert int(base["nref"]) == 42
    assert base["systems"] == [("WT", "wt_modeler_obj"), ("Mutant", "mut_obj")]

    payload_multi_mut = prepare_redocking_base_payload(
        ref_obj="ref_obj",
        wt_obj="wt_obj",
        ref_fields=("A", "101", "ATP"),
        run_mutant=True,
        mut_obj="",
        mutant_systems=[("Mutant_001", "mut_obj_1"), ("Mutant_002", "mut_obj_2")],
        params={"ph": 7.4},
        grid_from_reference_fn=_grid_from_ref,
    )
    assert payload_multi_mut["ok"] is True
    systems_multi = list(dict(payload_multi_mut).get("payload", {}).get("systems", []) or [])
    assert systems_multi == [("WT", "wt_obj"), ("Mutant_001", "mut_obj_1"), ("Mutant_002", "mut_obj_2")]
    assert list(dict(payload_multi_mut).get("payload", {}).get("dropped_mutant_identical_to_wt", []) or []) == []

    payload_multi_with_same = prepare_redocking_base_payload(
        ref_obj="ref_obj",
        wt_obj="wt_obj",
        ref_fields=("A", "101", "ATP"),
        run_mutant=True,
        mut_obj="",
        mutant_systems=[("Mutant_001", "wt_obj"), ("Mutant_002", "mut_obj_2")],
        params={"ph": 7.4},
        grid_from_reference_fn=_grid_from_ref,
    )
    assert payload_multi_with_same["ok"] is True
    systems_filtered = list(dict(payload_multi_with_same).get("payload", {}).get("systems", []) or [])
    assert systems_filtered == [("WT", "wt_obj"), ("Mutant_002", "mut_obj_2")]
    dropped = list(dict(payload_multi_with_same).get("payload", {}).get("dropped_mutant_identical_to_wt", []) or [])
    assert dropped == [("Mutant_001", "wt_obj")]

    payload_base_fail = prepare_redocking_base_payload(
        ref_obj="",
        wt_obj="wt_obj",
        ref_fields=("A", "1", "LIG"),
        grid_from_reference_fn=_grid_from_ref,
    )
    assert payload_base_fail["ok"] is False
    assert "Select a reference complex object." in str(payload_base_fail["error"])

    payload_base_center_fail = prepare_redocking_base_payload(
        ref_obj="ref_obj",
        wt_obj="wt_obj",
        ref_fields=("A", "1", "LIG"),
        grid_from_reference_fn=lambda _r, _f: (None, (18.0, 18.0, 18.0), 0),
    )
    assert payload_base_center_fail["ok"] is False
    assert "Reference ligand was not found" in str(payload_base_center_fail["error"])

    ref_lig = tmp_path / "ref_lig.sdf"
    ref_lig.write_text("$$$$\n", encoding="utf-8")
    pick_lig = tmp_path / "pick_lig.pdbqt"
    pick_lig.write_text("ATOM\n", encoding="utf-8")

    payload_lig_ref = prepare_redocking_ligand_override_context(
        use_reference_ligand=True,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        workspace_dir=str(tmp_path),
        export_reference_ligand_fn=lambda _r, _f, _d: (str(ref_lig), "ok"),
        pick_ligand_input_fn=lambda: str(pick_lig),
        file_exists_fn=os.path.isfile,
        makedirs_fn=os.makedirs,
    )
    assert payload_lig_ref["ok"] is True
    assert payload_lig_ref["use_reference_ligand"] is True
    assert str(payload_lig_ref["ligand_override_file"]) == str(ref_lig)

    payload_lig_pick = prepare_redocking_ligand_override_context(
        use_reference_ligand=False,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        workspace_dir=str(tmp_path),
        export_reference_ligand_fn=lambda _r, _f, _d: ("", "bad"),
        pick_ligand_input_fn=lambda: str(pick_lig),
        file_exists_fn=os.path.isfile,
    )
    assert payload_lig_pick["ok"] is True
    assert payload_lig_pick["use_reference_ligand"] is False
    assert str(payload_lig_pick["ligand_override_file"]) == str(pick_lig)

    payload_lig_fail = prepare_redocking_ligand_override_context(
        use_reference_ligand=True,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        workspace_dir=str(tmp_path),
        export_reference_ligand_fn=lambda _r, _f, _d: ("", "export failed"),
        pick_ligand_input_fn=lambda: "",
        file_exists_fn=os.path.isfile,
    )
    assert payload_lig_fail["ok"] is False
    assert "Failed to export crystal reference ligand for redocking: export failed" in str(payload_lig_fail["error"])

    payload_ref_flex = prepare_redocking_reference_flex_context(
        has_rdkit=True,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        workspace_dir=str(tmp_path),
        reference_mol_from_object_fn=lambda _r, _f, _d: ({"mol": True}, "ok"),
        flex_enabled=True,
        flex_rows=[{"resi": "10"}],
        makedirs_fn=os.makedirs,
    )
    assert payload_ref_flex["ok"] is True
    assert payload_ref_flex["ref_mol"] == {"mol": True}
    assert payload_ref_flex["flex_enabled"] is True
    assert int(len(payload_ref_flex["flex_rows"])) == 1

    payload_ref_flex_fail = prepare_redocking_reference_flex_context(
        has_rdkit=True,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        workspace_dir=str(tmp_path),
        reference_mol_from_object_fn=lambda _r, _f, _d: (None, "rdkit failure"),
        flex_enabled=True,
        flex_rows=[],
        makedirs_fn=os.makedirs,
    )
    assert payload_ref_flex_fail["ok"] is False
    assert "Flexible redocking is enabled, but no flexible residues are selected." in str(payload_ref_flex_fail["error"])
    dbg = list(payload_ref_flex_fail.get("debug_messages", []) or [])
    assert any("Redocking RMSD note: reference ligand RDKit conversion failed" in str(m) for m in dbg)


def test_redock_runtime_prepare_base_runtime_helper():
    calls = []

    def _builder(**kwargs):
        calls.append(dict(kwargs or {}))
        return {
            "ok": True,
            "error": "",
            "used_modeler_wt": bool(kwargs.get("use_modeler_wt", False)),
            "payload": {
                "ref_obj": str(kwargs.get("ref_obj", "")),
                "wt_obj": str(kwargs.get("wt_obj", "")),
                "ref_fields": tuple(kwargs.get("ref_fields", ()) or ()),
                "center": (1.0, 2.0, 3.0),
                "size": (18.0, 18.0, 18.0),
                "nref": 25,
                "systems": [("WT", str(kwargs.get("wt_obj", "")))],
                "params": dict(kwargs.get("params", {}) or {}),
            },
        }

    payload = prepare_redocking_base_runtime(
        reference_obj_fn=lambda: "ref_obj",
        wt_obj_fn=lambda: "wt_obj",
        use_modeler_wt_fn=lambda: True,
        modeler_wt_obj_fn=lambda: "wt_model_obj",
        ref_fields_fn=lambda: ("A", "101", "ATP"),
        run_mutant_fn=lambda: False,
        mutant_obj_fn=lambda: "",
        mutant_systems_fn=lambda: [],
        params_fn=lambda: {"ph": 7.4, "repeats": 3},
        grid_from_reference_fn=lambda _ref, _fields: ((1.0, 2.0, 3.0), (18.0, 18.0, 18.0), 25),
        base_payload_builder_fn=_builder,
    )
    assert payload["ok"] is True
    assert payload["used_modeler_wt"] is True
    assert str(payload["modeler_wt_obj"]) == "wt_model_obj"
    assert str(payload["payload"]["ref_obj"]) == "ref_obj"
    assert str(payload["payload"]["wt_obj"]) == "wt_obj"
    assert int(len(calls)) == 1
    sent = dict(calls[0] or {})
    assert str(sent["ref_obj"]) == "ref_obj"
    assert str(sent["wt_obj"]) == "wt_obj"
    assert sent["ref_fields"] == ("A", "101", "ATP")
    assert bool(sent["run_mutant"]) is False
    assert str(sent["mut_obj"]) == ""
    assert sent["mutant_systems"] == []
    assert sent["params"] == {"ph": 7.4, "repeats": 3}
    assert bool(sent["use_modeler_wt"]) is True
    assert str(sent["modeler_wt_obj"]) == "wt_model_obj"
    assert callable(sent["grid_from_reference_fn"])

    payload_fail = prepare_redocking_base_runtime(
        reference_obj_fn=lambda: "ref_obj",
        wt_obj_fn=lambda: "wt_obj",
        use_modeler_wt_fn=lambda: False,
        modeler_wt_obj_fn=lambda: "",
        ref_fields_fn=lambda: ("A", "101", "ATP"),
        run_mutant_fn=lambda: False,
        mutant_obj_fn=lambda: "",
        mutant_systems_fn=lambda: [],
        params_fn=lambda: {},
        grid_from_reference_fn=lambda _ref, _fields: ((0.0, 0.0, 0.0), (18.0, 18.0, 18.0), 0),
        base_payload_builder_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert payload_fail["ok"] is False
    assert str(payload_fail["error"]) == "boom"
    assert payload_fail["payload"] == {}


def test_redock_runtime_prepare_ligand_override_runtime_helper(tmp_path):
    calls = []

    def _builder(**kwargs):
        calls.append(dict(kwargs or {}))
        return {
            "ok": True,
            "error": "",
            "ligand_override_file": "lig_override.sdf",
            "use_reference_ligand": bool(kwargs.get("use_reference_ligand", False)),
            "prep_dir": str(kwargs.get("workspace_dir", "")),
        }

    payload = prepare_redocking_ligand_override_runtime(
        use_reference_ligand=True,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        workspace_dir_fn=lambda: str(tmp_path),
        export_reference_ligand_fn=lambda _r, _f, _d: ("", ""),
        pick_ligand_input_fn=lambda: "",
        file_exists_fn=os.path.isfile,
        makedirs_fn=os.makedirs,
        reference_prep_dir="",
        context_builder_fn=_builder,
    )
    assert payload["ok"] is True
    assert payload["use_reference_ligand"] is True
    assert str(payload["ligand_override_file"]) == "lig_override.sdf"
    assert int(len(calls)) == 1
    sent = dict(calls[0] or {})
    assert bool(sent["use_reference_ligand"]) is True
    assert str(sent["ref_obj"]) == "ref_obj"
    assert sent["ref_fields"] == ("A", "101", "ATP")
    assert str(sent["workspace_dir"]) == str(tmp_path)
    assert callable(sent["export_reference_ligand_fn"])
    assert callable(sent["pick_ligand_input_fn"])
    assert callable(sent["file_exists_fn"])
    assert callable(sent["makedirs_fn"])

    payload_fail = prepare_redocking_ligand_override_runtime(
        use_reference_ligand=False,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        workspace_dir_fn=lambda: str(tmp_path),
        context_builder_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert payload_fail["ok"] is False
    assert str(payload_fail["error"]) == "boom"
    assert str(payload_fail["ligand_override_file"]) == ""
    assert payload_fail["use_reference_ligand"] is False


def test_redock_runtime_prepare_reference_flex_runtime_helper(tmp_path):
    calls = []
    debug_lines = []

    def _builder(**kwargs):
        calls.append(dict(kwargs or {}))
        return {
            "ok": True,
            "error": "",
            "ref_mol": {"mol": True},
            "ref_msg": "ok",
            "flex_enabled": True,
            "flex_rows": [{"resi": "10"}],
            "debug_messages": ["runtime_debug", "", "   "],
        }

    payload = prepare_redocking_reference_flex_runtime(
        has_rdkit=True,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        workspace_dir_fn=lambda: str(tmp_path),
        reference_mol_from_object_fn=lambda _r, _f, _d: ({"mol": True}, "ok"),
        flex_enabled=True,
        flex_rows=[{"resi": "10"}],
        makedirs_fn=os.makedirs,
        append_debug_log_fn=lambda msg: debug_lines.append(str(msg)),
        context_builder_fn=_builder,
    )
    assert payload["ok"] is True
    assert payload["ref_mol"] == {"mol": True}
    assert payload["ref_msg"] == "ok"
    assert payload["flex_enabled"] is True
    assert payload["flex_rows"] == [{"resi": "10"}]
    assert debug_lines == ["runtime_debug"]
    assert int(len(calls)) == 1
    sent = dict(calls[0] or {})
    assert bool(sent["has_rdkit"]) is True
    assert str(sent["ref_obj"]) == "ref_obj"
    assert sent["ref_fields"] == ("A", "101", "ATP")
    assert str(sent["workspace_dir"]) == str(tmp_path)
    assert callable(sent["reference_mol_from_object_fn"])
    assert bool(sent["flex_enabled"]) is True
    assert sent["flex_rows"] == [{"resi": "10"}]
    assert callable(sent["makedirs_fn"])

    payload_fail = prepare_redocking_reference_flex_runtime(
        has_rdkit=False,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        workspace_dir_fn=lambda: str(tmp_path),
        reference_mol_from_object_fn=None,
        flex_enabled=False,
        flex_rows=[],
        context_builder_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert payload_fail["ok"] is False
    assert str(payload_fail["error"]) == "boom"
    assert payload_fail["ref_mol"] is None
    assert payload_fail["ref_msg"] == ""


def test_redock_runtime_strip_residue_from_structure_file_helper(tmp_path):
    in_pdb = tmp_path / "rec_in.pdb"
    out_chain_a = tmp_path / "rec_out_chain_a.pdb"
    out_any_chain = tmp_path / "rec_out_any_chain.pdb"
    in_pdb.write_text(
        "\n".join(
            [
                "HEADER TEST",
                "HETATM    1  C1  BEN A 123      10.000  10.000  10.000  1.00 20.00           C",
                "HETATM    2  C2  BEN B 123      11.000  11.000  11.000  1.00 20.00           C",
                "ATOM      3  CA  ALA A 124      12.000  12.000  12.000  1.00 20.00           C",
                "HETATM    4  C3  BEN A 123      13.000  13.000  13.000  1.00 20.00           C",
                "",
            ]
        ),
        encoding="utf-8",
    )

    removed_chain_a = strip_redocking_residue_from_structure_file(
        in_file=str(in_pdb),
        out_file=str(out_chain_a),
        chain="A",
        resi="123",
        resn="BEN",
    )
    assert int(removed_chain_a) == 2
    txt_chain_a = out_chain_a.read_text(encoding="utf-8")
    assert "BEN A 123" not in txt_chain_a
    assert "BEN B 123" in txt_chain_a
    assert "ALA A 124" in txt_chain_a

    removed_any_chain = strip_redocking_residue_from_structure_file(
        in_file=str(in_pdb),
        out_file=str(out_any_chain),
        chain="",
        resi="123",
        resn="BEN",
    )
    assert int(removed_any_chain) == 3
    txt_any_chain = out_any_chain.read_text(encoding="utf-8")
    assert "BEN A 123" not in txt_any_chain
    assert "BEN B 123" not in txt_any_chain
    assert "ALA A 124" in txt_any_chain

    # receptor preparation renumbered the ligand: removed by residue name instead of kept in the site
    out_renum = tmp_path / "rec_out_renumbered.pdb"
    removed_renum = strip_redocking_residue_from_structure_file(
        in_file=str(in_pdb),
        out_file=str(out_renum),
        chain="A",
        resi="1",
        resn="BEN",
    )
    assert int(removed_renum) == 3
    assert "BEN" not in out_renum.read_text(encoding="utf-8")
    assert int(strip_redocking_residue_from_structure_file(
        in_file=str(in_pdb), out_file=str(out_renum), chain="A", resi="1", resn="BEN", fallback_by_resn=False,
    )) == 0

    missing = strip_redocking_residue_from_structure_file(
        in_file=str(tmp_path / "missing.pdb"),
        out_file=str(tmp_path / "no_out.pdb"),
        chain="A",
        resi="123",
        resn="BEN",
    )
    assert int(missing) == 0


def test_redock_runtime_prepare_receptor_flex_context_helper(tmp_path):
    rec_in = tmp_path / "receptor_input.pdbqt"
    rec_in.write_text("RECEPTOR\n", encoding="utf-8")

    def _prepare_receptor(_obj, _job_dir, target_ph=7.4, flow_label="", auto_use_cached_hetatm=False):
        _ = (target_ph, flow_label, auto_use_cached_hetatm)
        return str(rec_in), {"notes": ["prepared"]}

    def _strip_ref(_receptor_input, stripped_file, _ch, _ri, _rn):
        with open(stripped_file, "w", encoding="utf-8") as fh:
            fh.write("STRIPPED\n")
        return 4

    payload = prepare_redocking_receptor_flex_context(
        receptor_obj="wt_obj",
        job_dir=str(tmp_path),
        target_ph=7.4,
        ref_fields=("A", "101", "ATP"),
        flex_enabled=False,
        flex_residues=[],
        prepare_receptor_fn=_prepare_receptor,
        strip_reference_residue_fn=_strip_ref,
        prepare_flexible_receptor_fn=None,
        file_exists_fn=os.path.isfile,
    )
    assert payload["ok"] is True
    assert str(payload["receptor_input"]) == str(rec_in)
    assert str(payload["receptor_for_dock"]).endswith("receptor_stripped.pdbqt")
    assert int(payload["removed_reference_atoms"]) == 4
    assert str(payload["flex_file"]) == ""
    assert int(len(payload["flex_rows"])) == 0

    payload_flex_missing = prepare_redocking_receptor_flex_context(
        receptor_obj="wt_obj",
        job_dir=str(tmp_path),
        target_ph=7.4,
        ref_fields=("A", "101", "ATP"),
        flex_enabled=True,
        flex_residues=[],
        prepare_receptor_fn=_prepare_receptor,
        strip_reference_residue_fn=_strip_ref,
        prepare_flexible_receptor_fn=None,
        file_exists_fn=os.path.isfile,
    )
    assert payload_flex_missing["ok"] is False
    assert "Flexible redocking is enabled" in str(payload_flex_missing["error"])

    def _prepare_flex(_receptor_for_dock, _job_dir, _flex_rows):
        return "rigid_out.pdbqt", "flex_out.pdbqt", {"ok": True, "used": True}

    payload_flex = prepare_redocking_receptor_flex_context(
        receptor_obj="wt_obj",
        job_dir=str(tmp_path),
        target_ph=7.4,
        ref_fields=("A", "101", "ATP"),
        flex_enabled=True,
        flex_residues=[{"resi": "10"}],
        prepare_receptor_fn=_prepare_receptor,
        strip_reference_residue_fn=_strip_ref,
        prepare_flexible_receptor_fn=_prepare_flex,
        file_exists_fn=os.path.isfile,
    )
    assert payload_flex["ok"] is True
    assert str(payload_flex["receptor_for_dock"]) == "rigid_out.pdbqt"
    assert str(payload_flex["flex_file"]) == "flex_out.pdbqt"
    assert payload_flex["flex_info"]["used"] is True
    assert int(len(payload_flex["flex_rows"])) == 1


def test_redock_runtime_prepare_receptor_flex_context_strips_prepared_pdb_for_flex(tmp_path):
    rec_in = tmp_path / "receptor_input.pdbqt"
    rec_in.write_text("RECEPTOR\n", encoding="utf-8")
    prep_pdb = tmp_path / "receptor_prepared.pdb"
    prep_pdb.write_text("ATOM\nHETATM ATP A 101\n", encoding="utf-8")
    captured = {}

    def _prepare_receptor(_obj, _job_dir, **_kwargs):
        return str(rec_in), {"notes": ["prepared"], "prepared_pdb": str(prep_pdb)}

    def _strip_ref(src_file, stripped_file, _ch, _ri, _rn):
        captured.setdefault("strip_calls", []).append((str(src_file), str(stripped_file)))
        with open(stripped_file, "w", encoding="utf-8") as fh:
            fh.write(f"STRIPPED {os.path.basename(str(src_file))}\n")
        return 2

    def _prepare_flex(_receptor_for_dock, _job_dir, _flex_rows, prepared_pdb="", rec_info=None):
        captured["prepared_pdb"] = str(prepared_pdb or "")
        captured["rec_info"] = dict(rec_info or {})
        rigid = os.path.join(str(_job_dir), "rigid_out.pdbqt")
        flex = os.path.join(str(_job_dir), "flex_out.pdbqt")
        with open(rigid, "w", encoding="utf-8") as fh:
            fh.write("ATOM      1  C   GLY A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C \n")
        with open(flex, "w", encoding="utf-8") as fh:
            fh.write("ATOM      1  CB  TYR A  10       0.000   0.000   0.000  0.00  0.00    +0.000 C \n")
        return rigid, flex, {"ok": True, "used": True, "tool": "mk_prepare_receptor.py"}

    payload = prepare_redocking_receptor_flex_context(
        receptor_obj="wt_obj",
        job_dir=str(tmp_path),
        target_ph=7.4,
        ref_fields=("A", "101", "ATP"),
        flex_enabled=True,
        flex_residues=[{"chain": "A", "resi": "10", "resn": "TYR"}],
        prepare_receptor_fn=_prepare_receptor,
        strip_reference_residue_fn=_strip_ref,
        prepare_flexible_receptor_fn=_prepare_flex,
        file_exists_fn=os.path.isfile,
    )
    assert payload["ok"] is True
    assert str(captured["prepared_pdb"]).endswith("receptor_prepared_stripped.pdb")
    assert os.path.isfile(str(captured["prepared_pdb"]))
    assert str(captured["rec_info"]["prepared_pdb"]) == str(prep_pdb)
    assert len(list(captured.get("strip_calls", []))) >= 2


def test_redock_runtime_prepare_receptor_flex_context_uses_native_smina_flexres(tmp_path):
    rec_in = tmp_path / "receptor_input.pdbqt"
    rec_in.write_text("RECEPTOR\n", encoding="utf-8")

    def _prepare_receptor(_obj, _job_dir, **_kwargs):
        return str(rec_in), {"notes": ["prepared"]}

    def _strip_ref(_receptor_input, stripped_file, _ch, _ri, _rn):
        with open(stripped_file, "w", encoding="utf-8") as fh:
            fh.write("STRIPPED\n")
        return 1

    payload = prepare_redocking_receptor_flex_context(
        receptor_obj="wt_obj",
        job_dir=str(tmp_path),
        target_ph=7.4,
        ref_fields=("A", "101", "ATP"),
        flex_enabled=True,
        flex_residues=[{"chain": "A", "resi": "10", "resn": "TYR"}],
        prepare_receptor_fn=_prepare_receptor,
        strip_reference_residue_fn=_strip_ref,
        prepare_flexible_receptor_fn=None,
        native_flexres_supported=True,
        build_native_flexres_spec_fn=lambda rows: build_smina_flexres_spec(rows),
        file_exists_fn=os.path.isfile,
    )
    assert payload["ok"] is True
    assert str(payload["receptor_for_dock"]).endswith("receptor_stripped.pdbqt")
    assert str(payload["flex_file"]) == ""
    assert payload["flex_info"]["native"] is True
    assert payload["flex_info"]["spec"] == "A:10"


def test_redock_runtime_prepare_receptor_flex_context_prefers_prepared_flex_over_native(tmp_path):
    rec_in = tmp_path / "receptor_input.pdbqt"
    rec_in.write_text("RECEPTOR\n", encoding="utf-8")
    prep_pdb = tmp_path / "receptor_prepared.pdb"
    prep_pdb.write_text("ATOM\n", encoding="utf-8")

    def _prepare_receptor(_obj, _job_dir, **_kwargs):
        return str(rec_in), {"notes": ["prepared"], "prepared_pdb": str(prep_pdb)}

    def _prepare_flex(_receptor_for_dock, _job_dir, _flex_rows, prepared_pdb="", rec_info=None):
        _ = (prepared_pdb, rec_info)
        rigid = os.path.join(str(_job_dir), "rigid_out.pdbqt")
        flex = os.path.join(str(_job_dir), "flex_out.pdbqt")
        with open(rigid, "w", encoding="utf-8") as fh:
            fh.write("ATOM      1  C   GLY A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C \n")
        with open(flex, "w", encoding="utf-8") as fh:
            fh.write("ATOM      1  CB  TYR A  10       0.000   0.000   0.000  0.00  0.00    +0.000 C \n")
        return rigid, flex, {"ok": True, "used": True, "tool": "mk_prepare_receptor.py"}

    payload = prepare_redocking_receptor_flex_context(
        receptor_obj="wt_obj",
        job_dir=str(tmp_path),
        target_ph=7.4,
        ref_fields=(),
        flex_enabled=True,
        flex_residues=[{"chain": "A", "resi": "10", "resn": "TYR"}],
        prepare_receptor_fn=_prepare_receptor,
        strip_reference_residue_fn=None,
        prepare_flexible_receptor_fn=_prepare_flex,
        native_flexres_supported=True,
        build_native_flexres_spec_fn=lambda rows: build_smina_flexres_spec(rows),
        file_exists_fn=os.path.isfile,
    )
    assert payload["ok"] is True
    assert str(payload["receptor_for_dock"]).endswith("rigid_out.pdbqt")
    assert str(payload["flex_file"]).endswith("flex_out.pdbqt")
    assert payload["flex_info"]["tool"] == "mk_prepare_receptor.py"
    assert payload["flex_info"].get("native", False) is False


def test_redock_runtime_build_smina_plan():
    plan = build_redocking_smina_plan(
        smina_bin="smina_bin",
        receptor_for_dock="rec.pdbqt",
        ligand_input="lig.pdbqt",
        center=(1.23456, -2.34567, 3.45678),
        size=(18.1234, 19.2345, 20.3456),
        params={
            "repeats": 250,
            "seed0": 321,
            "exhaustiveness": 12,
            "num_modes": 40,
            "energy_range": 5.6789,
            "min_rmsd": 0.8765,
        },
        flex_file="flex.pdbqt",
    )
    assert int(plan["n_repeats"]) == 100
    assert int(plan["seed0"]) == 321
    cmd = list(plan["base_cmd"])
    assert cmd[0] == "smina_bin"
    assert cmd[cmd.index("-r") + 1] == "rec.pdbqt"
    assert cmd[cmd.index("-l") + 1] == "lig.pdbqt"
    assert cmd[cmd.index("--center_x") + 1] == "1.2346"
    assert cmd[cmd.index("--center_y") + 1] == "-2.3457"
    assert cmd[cmd.index("--center_z") + 1] == "3.4568"
    assert cmd[cmd.index("--size_x") + 1] == "18.1234"
    assert cmd[cmd.index("--size_y") + 1] == "19.2345"
    assert cmd[cmd.index("--size_z") + 1] == "20.3456"
    assert cmd[cmd.index("--exhaustiveness") + 1] == "12"
    assert cmd[cmd.index("--num_modes") + 1] == "40"
    assert cmd[cmd.index("--energy_range") + 1] == "5.679"
    assert cmd[cmd.index("--min_rmsd_filter") + 1] == "0.876"
    assert cmd[cmd.index("--flex") + 1] == "flex.pdbqt"
    assert "--autobox_ligand" not in set(cmd)

    plan2 = build_redocking_smina_plan(
        smina_bin="smina",
        receptor_for_dock="rec",
        ligand_input="lig",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        params={"repeats": 0, "seed0": "bad"},
        flex_file="",
    )
    assert int(plan2["n_repeats"]) == 1
    assert int(plan2["seed0"]) == 0
    assert "--flex" not in set(plan2["base_cmd"])

    plan3 = build_redocking_smina_plan(
        smina_bin="smina",
        receptor_for_dock="rec",
        ligand_input="lig",
        autobox_ligand="ref_ligand.pdb",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        params={"autobox_add": 3.25},
        flex_file="",
    )
    cmd3 = list(plan3["base_cmd"])
    assert cmd3[cmd3.index("--autobox_ligand") + 1] == "ref_ligand.pdb"
    assert cmd3[cmd3.index("--autobox_add") + 1] == "3.250"
    assert "--center_x" not in set(cmd3)
    assert "--size_x" not in set(cmd3)

    plan4 = build_redocking_smina_plan(
        smina_bin="smina",
        receptor_for_dock="rec",
        ligand_input="lig",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        native_flexres="A:10,B:23",
        flex_file="flex_should_not_be_used.pdbqt",
    )
    cmd4 = list(plan4["base_cmd"])
    assert cmd4[cmd4.index("--flexres") + 1] == "A:10,B:23"
    assert "--flex" not in set(cmd4)

    plan5 = build_redocking_smina_plan(
        smina_bin="smina",
        receptor_for_dock="rec",
        ligand_input="lig",
        autobox_ligand="ref_ligand.pdb",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        native_flexres="A:10,B:23,C:57",
        params={"autobox_add": 4.0},
    )
    cmd5 = list(plan5["base_cmd"])
    assert float(cmd5[cmd5.index("--autobox_add") + 1]) >= 10.0


def test_redock_runtime_estimate_initial_autobox_add(tmp_path):
    assert float(estimate_redocking_initial_autobox_add(4.0, native_flexres="")) == 4.0
    assert float(estimate_redocking_initial_autobox_add(4.0, flex_file="flex.pdbqt")) == 10.0
    assert float(estimate_redocking_initial_autobox_add(4.0, native_flexres="A:10")) == 10.0
    assert float(estimate_redocking_initial_autobox_add(4.0, native_flexres="A:10,B:23,C:57,D:190,E:191")) >= 13.5

    ref_pdb = tmp_path / "ref_ligand.pdb"
    ref_pdb.write_text(
        "\n".join(
            [
                "HETATM    1  C1  BEN A   1       0.000   0.000   0.000  1.00 20.00           C",
                "HETATM    2  C2  BEN A   1       1.000   1.000   1.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rec_pdbqt = tmp_path / "rec.pdbqt"
    rec_pdbqt.write_text(
        "\n".join(
            [
                "ATOM      1  CA  SER A  16      22.000   0.000   0.000  1.00 20.00      A    C",
                "ATOM      2  CB  SER A  16      23.500   0.000   0.000  1.00 20.00      A    C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    geom_add = float(
        estimate_redocking_initial_autobox_add(
            4.0,
            native_flexres="A:16",
            autobox_ligand=str(ref_pdb),
            receptor_structure=str(rec_pdbqt),
        )
    )
    assert geom_add > 20.0


def test_redock_runtime_rmsd_and_row_builder_helpers(tmp_path):
    out_pose_pdbqt = tmp_path / "docked_poses.pdbqt"
    out_pose_pdbqt.write_text("ATOM 1 C\n", encoding="utf-8")
    out_pose_sdf = tmp_path / "docked_poses_rmsd.sdf"

    def _convert_ok(src, dst):
        _ = src
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True

    def _load_mols(_pose_sdf):
        return [object(), None, object()]

    ref_marker = object()
    rmsd_ref_args = []

    def _best_rmsd(ref_obj, mol):
        rmsd_ref_args.append((ref_obj, mol))
        return 1.1 if mol is not None else float("nan")

    vals = compute_redocking_rmsd_values_for_repeat(
        out_pose=str(out_pose_pdbqt),
        poses_count=3,
        has_rdkit=True,
        ref_mol=ref_marker,
        convert_pdbqt_to_sdf_fn=_convert_ok,
        load_pose_mols_fn=_load_mols,
        best_rmsd_fn=_best_rmsd,
    )
    assert int(len(vals)) == 3
    assert float(vals[0]) == 1.1
    assert str(vals[1]) == "nan"
    assert float(vals[2]) == 1.1
    assert os.path.isfile(str(out_pose_sdf))
    assert int(len(rmsd_ref_args)) == 2
    assert all(ref_obj is ref_marker for ref_obj, _mol in list(rmsd_ref_args or []))

    warns = []

    def _load_fail(_pose_sdf):
        raise RuntimeError("bad sdf parse")

    vals_fail = compute_redocking_rmsd_values_for_repeat(
        out_pose=str(out_pose_pdbqt),
        poses_count=2,
        has_rdkit=True,
        ref_mol=object(),
        convert_pdbqt_to_sdf_fn=_convert_ok,
        load_pose_mols_fn=_load_fail,
        best_rmsd_fn=lambda _ref, _mol: 0.0,
        log_warning_fn=lambda msg: warns.append(str(msg)),
    )
    assert int(len(vals_fail)) == 2
    assert "bad sdf parse" in str(warns[0])

    # the display converter writes the SDF but rejects it (RDKit cannot sanitize the Open Babel
    # re-perception of a charged ligand): the RMSD is still computed from that file
    def _convert_written_but_rejected(src, dst):
        _ = src
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return False

    vals_rejected = compute_redocking_rmsd_values_for_repeat(
        out_pose=str(out_pose_pdbqt),
        poses_count=1,
        has_rdkit=True,
        ref_mol=object(),
        convert_pdbqt_to_sdf_fn=_convert_written_but_rejected,
        load_pose_mols_fn=lambda _f: [object()],
        best_rmsd_fn=lambda _ref, _mol: 0.5,
    )
    assert float(vals_rejected[0]) == 0.5

    direct_sdf = tmp_path / "direct.sdf"
    direct_sdf.write_text("$$$$\n", encoding="utf-8")
    vals_sdf = compute_redocking_rmsd_values_for_repeat(
        out_pose=str(direct_sdf),
        poses_count=1,
        has_rdkit=True,
        ref_mol=object(),
        load_pose_mols_fn=lambda _f: [object()],
        best_rmsd_fn=lambda _ref, _mol: 2.2,
    )
    assert float(vals_sdf[0]) == 2.2

    vals_none = compute_redocking_rmsd_values_for_repeat(
        out_pose=str(out_pose_pdbqt),
        poses_count=2,
        has_rdkit=False,
        ref_mol=None,
    )
    assert int(len(vals_none)) == 2
    assert all(str(v) == "nan" for v in vals_none)

    rows = build_redocking_rows_for_repeat(
        rep=4,
        poses=[{"mode": 3, "affinity": -8.3}, {"affinity": -7.1}],
        rmsd_vals=[1.25],
        system_label="WT",
        receptor_obj="wt_rec",
        source_receptor_obj="wt_src",
        job_dir="job_x",
    )
    assert int(len(rows)) == 2
    assert str(rows[0]["system"]) == "WT"
    assert str(rows[0]["receptor"]) == "wt_rec"
    assert str(rows[0]["receptor_source"]) == "wt_src"
    assert int(rows[0]["mode"]) == 3
    assert float(rows[0]["rmsd_ref"]) == 1.25
    assert int(rows[0]["repeat"]) == 4
    assert int(rows[1]["mode"]) == 2
    assert str(rows[1]["rmsd_ref"]) == "nan"


def test_redock_runtime_build_run_debug_line():
    txt = build_redocking_run_debug_line(
        system_label="WT",
        receptor_obj="rec_wt",
        source_receptor_obj="rec_src",
        removed_ref_atoms=12,
        repeats=4,
        aff_stats={"n": 4, "ci_low": -8.4, "ci_high": -7.2, "confidence": "high"},
        best_repeat_id=2,
        pose_count=30,
        best_row={"mode": 3, "affinity": -8.1, "rmsd_ref": 1.25},
        rms_stats={"ci_low": 1.0, "ci_high": 1.8, "confidence": "moderate"},
        flex_enabled=True,
        flex_count=5,
        flex_file="flex.pdbqt",
        rec_info={"notes": ["kept cofactor"]},
        lig_info={"notes": ["pH=7.4"]},
        flex_info={"ok": True},
    )
    assert txt.startswith("Redocking WT: receptor=rec_wt, source=rec_src")
    assert "removed_ref_atoms=12" in txt
    assert "repeat_success=4" in txt
    assert "best_repeat=2" in txt
    assert "poses=30" in txt
    assert "top1_mode=3" in txt
    assert "top1_aff=-8.100" in txt
    assert "top1_rmsd=1.25" in txt
    assert "aff_conf=high" in txt
    assert "rms_conf=moderate" in txt
    assert "flex_enabled=True" in txt
    assert "flex_count=5" in txt
    assert "flex_file=flex.pdbqt" in txt
    assert "receptor_prep_notes=['kept cofactor']" in txt
    assert "ligand_prep_notes=['pH=7.4']" in txt
    assert "flex_info={'ok': True}" in txt

    txt2 = build_redocking_run_debug_line(
        system_label="Mutant",
        receptor_obj="mut_rec",
        source_receptor_obj="",
        removed_ref_atoms=0,
        repeats=1,
        aff_stats={},
        best_repeat_id=None,
        pose_count=0,
        best_row={},
        rms_stats={},
        flex_enabled=False,
        flex_count=0,
        flex_file="",
        rec_info="n/a",
        lig_info="n/a",
        flex_info={},
    )
    assert "best_repeat=n/a" in txt2
    assert "top1_mode=n/a" in txt2
    assert "top1_rmsd=n/a" in txt2
    assert "source=mut_rec" in txt2


def test_redock_runtime_build_single_run_payload():
    rows = [{"mode": 1, "affinity": -8.2}]
    payload = build_redocking_single_run_payload(
        rows=rows,
        best_row={"mode": 1, "affinity": -8.2},
        job_dir="job_dir_a",
        receptor_object="rec_a",
        source_receptor="rec_src_a",
        receptor_input="rec.pdbqt",
        ligand_input="lig.pdbqt",
        removed_reference_atoms=7,
        flex_enabled=True,
        flex_file="flex.pdbqt",
        flex_info={"ok": True},
        repeat_rows=[{"repeat": 1, "ok": True}],
        repeat_stats_affinity={"n": 1, "mean": -8.2},
        repeat_stats_rmsd={"n": 1, "mean": 1.4},
        repeat_csv="rep.csv",
        repeat_best=1,
        repeat_count=3,
        affinity_zero_plateau=False,
    )
    assert payload["ok"] is True
    assert payload["rows"] == rows
    assert int(payload["best"]["mode"]) == 1
    assert str(payload["job_dir"]) == "job_dir_a"
    assert str(payload["receptor_object"]) == "rec_a"
    assert str(payload["source_receptor"]) == "rec_src_a"
    assert str(payload["receptor_input"]) == "rec.pdbqt"
    assert str(payload["ligand_input"]) == "lig.pdbqt"
    assert int(payload["removed_reference_atoms"]) == 7
    assert payload["flex_enabled"] is True
    assert str(payload["flex_file"]) == "flex.pdbqt"
    assert payload["flex_info"]["ok"] is True
    assert int(payload["repeat_stats_affinity"]["n"]) == 1
    assert int(payload["repeat_stats_rmsd"]["n"]) == 1
    assert str(payload["repeat_csv"]) == "rep.csv"
    assert int(payload["repeat_best"]) == 1
    assert int(payload["repeat_count"]) == 3
    assert payload["affinity_zero_plateau"] is False

    payload2 = build_redocking_single_run_payload(
        rows=None,
        best_row=None,
        job_dir="",
        receptor_object="",
        source_receptor="",
        receptor_input="",
        ligand_input="",
        repeat_best=None,
    )
    assert payload2["best"] is None
    assert payload2["repeat_best"] is None
    assert int(payload2["repeat_count"]) == 1


def test_redock_runtime_run_repeat_pipeline_helper():
    def _runner_ok(
        base_cmd,
        job_dir,
        n_repeats,
        seed0,
        run_command,
        parse_pose_scores,
        build_rows_for_repeat,
        timeout=7200,
    ):
        _ = (base_cmd, n_repeats, seed0, run_command, parse_pose_scores, timeout)
        rows = list(
            build_rows_for_repeat(
                2,
                [{"mode": 1, "affinity": -8.1}],
                os.path.join(str(job_dir), "docked_poses.pdbqt"),
                os.path.join(str(job_dir), "docking.log"),
            )
            or []
        )
        best = dict(rows[0]) if rows else None
        return {
            "ok": True,
            "repeat_rows": [{"repeat": 2, "ok": True}],
            "best_rows": list(rows),
            "best_repeat": best,
            "best_repeat_id": 2,
            "best_pose_file": os.path.join(str(job_dir), "docked_poses.pdbqt"),
        }

    seen = []

    def _compute(rep, poses, out_pose, out_log):
        seen.append((int(rep), int(len(list(poses or []))), str(out_pose), str(out_log)))
        return [1.5 for _ in list(poses or [])]

    payload = run_redocking_repeat_pipeline(
        base_cmd=["smina", "-r", "rec.pdbqt", "-l", "lig.pdbqt"],
        job_dir="job_repeat",
        n_repeats=3,
        seed0=200,
        system_label="WT",
        receptor_obj="wt_rec",
        source_receptor_obj="wt_src",
        run_command=lambda *_a, **_k: (True, "", "", 0),
        parse_pose_scores_fn=lambda _p, _l: [{"mode": 1, "affinity": -8.0}],
        compute_rmsd_values_fn=_compute,
        run_repeats_fn=_runner_ok,
        timeout=15,
    )
    assert payload["ok"] is True
    assert str(payload["job_dir"]) == "job_repeat"
    assert int(payload["best_repeat_id"]) == 2
    assert int(len(payload["best_rows"])) == 1
    assert str(payload["best_rows"][0]["system"]) == "WT"
    assert str(payload["best_rows"][0]["receptor"]) == "wt_rec"
    assert str(payload["best_rows"][0]["receptor_source"]) == "wt_src"
    assert float(payload["best_rows"][0]["rmsd_ref"]) == 1.5
    assert int(len(seen)) == 1
    assert int(seen[0][0]) == 2

    def _runner_fail(
        base_cmd,
        job_dir,
        n_repeats,
        seed0,
        run_command,
        parse_pose_scores,
        build_rows_for_repeat,
        timeout=7200,
    ):
        _ = (base_cmd, job_dir, n_repeats, seed0, run_command, parse_pose_scores, build_rows_for_repeat, timeout)
        return {"ok": False, "error": "all repeats failed", "repeat_rows": [{"repeat": 1, "ok": False}]}

    payload_fail = run_redocking_repeat_pipeline(
        base_cmd=["smina"],
        job_dir="job_repeat_fail",
        n_repeats=1,
        seed0=0,
        system_label="WT",
        receptor_obj="wt_rec",
        source_receptor_obj="",
        run_command=lambda *_a, **_k: (False, "", "x", 1),
        parse_pose_scores_fn=lambda _p, _l: [],
        compute_rmsd_values_fn=lambda *_a, **_k: [],
        run_repeats_fn=_runner_fail,
    )
    assert payload_fail["ok"] is False
    assert "all repeats failed" in str(payload_fail["error"])
    assert str(payload_fail["job_dir"]) == "job_repeat_fail"
    assert int(len(payload_fail["repeat_rows"])) == 1


def test_redock_runtime_run_runtime_systems_helper():
    runtime_systems = [
        ("WT", "wt_rec_a", "wt_src_a", "WT_model_001"),
        ("Mutant", "mut_rec_a", "mut_src_a", "Mut_model_001"),
    ]

    calls = []

    def _run_single(
        label,
        rec_obj,
        ref_obj,
        ref_fields,
        ref_mol,
        center,
        size,
        params,
        source_receptor_obj=None,
        flex_enabled=False,
        flex_residues=None,
        auto_use_cached_hetatm=False,
        ligand_override_file="",
    ):
        calls.append(
            {
                "label": str(label),
                "rec_obj": str(rec_obj),
                "ref_obj": str(ref_obj),
                "ref_fields": tuple(ref_fields or ()),
                "ref_mol": ref_mol,
                "center": tuple(center or ()),
                "size": tuple(size or ()),
                "params": dict(params or {}),
                "source_receptor_obj": str(source_receptor_obj or ""),
                "flex_enabled": bool(flex_enabled),
                "flex_residues": list(flex_residues or []),
                "auto_use_cached_hetatm": bool(auto_use_cached_hetatm),
                "ligand_override_file": str(ligand_override_file or ""),
            }
        )
        return {
            "ok": True,
            "rows": [
                {
                    "system": str(label),
                    "receptor": str(rec_obj),
                    "receptor_source": str(source_receptor_obj or rec_obj),
                    "mode": 1,
                    "affinity": -8.0,
                    "rmsd_ref": 1.2,
                }
            ],
        }

    payload = run_redocking_runtime_systems(
        runtime_systems=runtime_systems,
        run_single_system_fn=_run_single,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        ref_mol=object(),
        center=(1.0, 2.0, 3.0),
        size=(18.0, 18.0, 18.0),
        params={"ph": 7.4},
        use_modeler_ensemble=True,
        flex_enabled=True,
        flex_residues=[{"resi": "10"}],
        ligand_override_file="lig_override.sdf",
        auto_use_cached_hetatm=True,
    )
    assert payload["ok"] is True
    assert payload["cancelled"] is False
    assert int(len(payload["rows"])) == 2
    assert set(payload["run_groups"].keys()) == {"WT", "Mutant"}
    assert int(len(payload["run_groups"]["WT"])) == 1
    rr_wt = dict(payload["run_groups"]["WT"][0])
    assert str(rr_wt["ensemble_member"]) == "WT_model_001"
    assert rr_wt["ensemble_mode"] is True
    assert str(rr_wt["rows"][0]["system_member"]) == "WT_model_001"
    assert int(len(calls)) == 2
    assert calls[0]["auto_use_cached_hetatm"] is True
    assert calls[0]["flex_enabled"] is True
    assert str(calls[0]["ligand_override_file"]) == "lig_override.sdf"

    def _run_single_cancel(*_a, **_k):
        return {"cancelled": True}

    payload_cancel = run_redocking_runtime_systems(
        runtime_systems=[("WT", "wt_rec_a", "wt_src_a", "WT_model_001")],
        run_single_system_fn=_run_single_cancel,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        ref_mol=None,
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
    )
    assert payload_cancel["ok"] is False
    assert payload_cancel["cancelled"] is True

    def _run_single_fail(*_a, **_k):
        return {"ok": False, "error": "bad run"}

    payload_fail = run_redocking_runtime_systems(
        runtime_systems=[("WT", "wt_rec_a", "wt_src_a", "WT_model_001")],
        run_single_system_fn=_run_single_fail,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        ref_mol=None,
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
    )
    assert payload_fail["ok"] is False
    assert payload_fail["cancelled"] is False
    assert "WT run failed (WT_model_001): bad run" in str(payload_fail["error"])


def test_redock_runtime_run_runtime_systems_runtime_helper():
    calls = []

    def _runner(**kwargs):
        calls.append(dict(kwargs or {}))
        return {
            "ok": True,
            "cancelled": False,
            "rows": [{"system": "WT", "mode": 1}],
            "run_groups": {"WT": [{"ok": True}]},
        }

    run_single = lambda *_a, **_k: {"ok": True, "rows": []}

    payload = run_redocking_runtime_systems_runtime(
        runtime_systems=[("WT", "wt_rec_a", "wt_src_a", "WT_model_001")],
        run_single_system_fn=run_single,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        ref_mol={"ref": True},
        center=(1.0, 2.0, 3.0),
        size=(18.0, 18.0, 18.0),
        params={"repeats": 3},
        use_modeler_ensemble=True,
        flex_enabled=True,
        flex_residues=[{"resi": "10"}],
        ligand_override_file="lig_override.sdf",
        auto_use_cached_hetatm=False,
        runtime_runner_fn=_runner,
    )
    assert payload["ok"] is True
    assert payload["cancelled"] is False
    assert int(len(payload["rows"])) == 1
    assert "WT" in payload["run_groups"]

    sent = dict(calls[0] or {})
    assert sent["runtime_systems"] == [("WT", "wt_rec_a", "wt_src_a", "WT_model_001")]
    assert sent["run_single_system_fn"] is run_single
    assert str(sent["ref_obj"]) == "ref_obj"
    assert sent["ref_fields"] == ("A", "101", "ATP")
    assert sent["ref_mol"] == {"ref": True}
    assert sent["center"] == (1.0, 2.0, 3.0)
    assert sent["size"] == (18.0, 18.0, 18.0)
    assert sent["params"] == {"repeats": 3}
    assert bool(sent["use_modeler_ensemble"]) is True
    assert bool(sent["flex_enabled"]) is True
    assert sent["flex_residues"] == [{"resi": "10"}]
    assert str(sent["ligand_override_file"]) == "lig_override.sdf"
    assert bool(sent["auto_use_cached_hetatm"]) is False

    payload_fail = run_redocking_runtime_systems_runtime(
        runtime_systems=[("WT", "wt_rec_a", "wt_src_a", "WT_model_001")],
        run_single_system_fn=run_single,
        ref_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        ref_mol=None,
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        runtime_runner_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert payload_fail["ok"] is False
    assert payload_fail["cancelled"] is False
    assert str(payload_fail["error"]) == "boom"
    assert payload_fail["rows"] == []
    assert payload_fail["run_groups"] == {}


def test_redock_runtime_receptor_structure_qc_flags_missing_altloc_and_hetatm(tmp_path):
    receptor = tmp_path / "receptor_problem.pdb"
    ligand = tmp_path / "ref_lig.pdb"
    receptor.write_text(
        "\n".join(
            [
                "REMARK 465   GLY A   11",
                "REMARK 470   ALA A  10    CG",
                "ATOM      1  N   ALA A  10      -1.000   0.000   0.000  1.00 20.00           N  ",
                "ATOM      2  CA AALA A  10       1.000   0.000   0.000  0.60 20.00           C  ",
                "ATOM      3  CA BALA A  10       1.300   0.000   0.000  0.40 20.00           C  ",
                "HETATM    4 ZN    ZN A 301       2.000   0.000   0.000  1.00 20.00          ZN  ",
                "HETATM    5  C1  ATP A 302       3.000   0.000   0.000  1.00 20.00           C  ",
                "HETATM    6  O   HOH A 401       4.000   0.000   0.000  1.00 20.00           O  ",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ligand.write_text(
        "\n".join(
            [
                "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 20.00           C  ",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qc = build_redocking_receptor_structure_qc(
        str(receptor),
        reference_ligand_file=str(ligand),
        site_cutoff_a=2.0,
    )

    assert qc["ok"] is True
    assert qc["status"] == "yellow"
    assert qc["missing_residue_remark_count"] == 1
    assert qc["missing_atom_remark_count"] == 1
    assert qc["altloc_count"] == 2
    assert qc["altloc_near_site"] == 2
    assert qc["metal_count"] == 1
    assert qc["cofactor_count"] == 1
    assert qc["water_count"] == 1
    assert any("missing residues" in str(w) for w in qc["warnings"])
    assert any("alternate locations near reference ligand" in str(w) for w in qc["warnings"])


def test_redock_runtime_receptor_structure_qc_blocks_empty_receptor(tmp_path):
    receptor = tmp_path / "empty.pdb"
    receptor.write_text("HEADER EMPTY\nEND\n", encoding="utf-8")

    qc = build_redocking_receptor_structure_qc(str(receptor))

    assert qc["ok"] is False
    assert qc["status"] == "red"
    assert any("no ATOM/HETATM" in str(e) for e in qc["errors"])


def test_redock_runtime_run_hetatm_preflight_helper():
    runtime_systems = [
        ("WT", "wt_runtime_obj", "wt_source_obj", "WT_model_001"),
        ("WT", "wt_runtime_obj_b", "wt_source_obj", "WT_model_002"),
        ("Mutant", "mut_runtime_obj", "mut_source_obj", "Mut_model_001"),
    ]
    cache = {}
    collect_calls = []
    prompt_calls = []

    def _collect(prompt_obj):
        collect_calls.append(str(prompt_obj))
        if str(prompt_obj) == "wt_source_obj":
            return [{"id": "a"}, {"id": "b"}]
        if str(prompt_obj) == "mut_source_obj":
            return [{"id": "c"}]
        return []

    def _prompt(prompt_obj, comps, flow_label=""):
        prompt_calls.append((str(prompt_obj), int(len(list(comps or []))), str(flow_label)))
        if str(prompt_obj) == "wt_source_obj":
            return {
                "selected": [{"id": "a"}],
                "selected_counts": {"cofactor": 1, "ion": 0, "water": 0},
            }
        return {
            "selected": [{"id": "c"}],
            "selected_counts": {"cofactor": 0, "ion": 1, "water": 0},
        }

    payload = run_redocking_hetatm_preflight(
        runtime_systems=runtime_systems,
        selection_cache=cache,
        collect_components_fn=_collect,
        prompt_selection_fn=_prompt,
        component_cache_key_fn=lambda c: str((c or {}).get("id", "")),
    )
    assert payload["ok"] is True
    assert payload["cancelled"] is False
    assert int(len(payload["lines"])) == 2
    assert "WT HETATM keep (wt_source_obj): cof=1, ion=0, wat=0" in str(payload["lines"][0])
    assert "Mutant HETATM keep (mut_source_obj): cof=0, ion=1, wat=0" in str(payload["lines"][1])
    assert collect_calls == ["wt_source_obj", "mut_source_obj"]
    assert int(len(prompt_calls)) == 2
    assert str(prompt_calls[0][2]) == "redocking_wt"
    assert str(prompt_calls[1][2]) == "redocking_mutant"
    assert cache["wt_source_obj"] == ["a"]
    assert cache["wt_runtime_obj"] == ["a"]
    assert cache["wt_runtime_obj_b"] == ["a"]
    assert cache["mut_source_obj"] == ["c"]
    assert cache["mut_runtime_obj"] == ["c"]

    payload_cancel = run_redocking_hetatm_preflight(
        runtime_systems=[("WT", "wt_runtime_obj", "wt_source_obj", "WT_model_001")],
        selection_cache={},
        collect_components_fn=lambda _obj: [{"id": "a"}],
        prompt_selection_fn=lambda _obj, _comps, flow_label="": {"cancelled": True},
        component_cache_key_fn=lambda c: str((c or {}).get("id", "")),
    )
    assert payload_cancel["ok"] is False
    assert payload_cancel["cancelled"] is True
    assert payload_cancel["lines"] == []

    payload_error = run_redocking_hetatm_preflight(
        runtime_systems=[("WT", "wt_runtime_obj", "wt_source_obj", "WT_model_001")],
        selection_cache={},
        collect_components_fn=lambda _obj: (_ for _ in ()).throw(RuntimeError("collect failed")),
        prompt_selection_fn=lambda _obj, _comps, flow_label="": {"selected": []},
        component_cache_key_fn=lambda c: str((c or {}).get("id", "")),
    )
    assert payload_error["ok"] is False
    assert payload_error["cancelled"] is False
    assert "collect failed" in str(payload_error["error"])


def test_redock_runtime_run_hetatm_preflight_runtime_helper():
    cache = {}
    prompt_calls = []
    runtime_calls = []

    def _prompt(prompt_obj, comps, flow_label="", cache_dict=None, allow_cached_auto=True):
        prompt_calls.append(
            {
                "prompt_obj": str(prompt_obj),
                "n_comps": int(len(list(comps or []))),
                "flow_label": str(flow_label),
                "cache_same": bool(cache_dict is cache),
                "allow_cached_auto": bool(allow_cached_auto),
            }
        )
        return {
            "selected": [{"id": "a"}],
            "selected_counts": {"cofactor": 1, "ion": 0, "water": 0},
        }

    def _runner(**kwargs):
        runtime_calls.append(dict(kwargs or {}))
        pick = dict(kwargs["prompt_selection_fn"]("wt_src_obj", [{"id": "a"}], flow_label="redocking_wt") or {})
        assert int(len(list(pick.get("selected", []) or []))) == 1
        return {
            "ok": True,
            "cancelled": False,
            "error": "",
            "lines": ["ok_line"],
            "selection_cache": dict(kwargs.get("selection_cache", {}) or {}),
        }

    payload = run_redocking_hetatm_preflight_runtime(
        runtime_systems=[("WT", "wt_runtime_obj", "wt_src_obj", "WT_model_001")],
        selection_cache=cache,
        collect_components_fn=lambda _obj: [{"id": "a"}],
        prompt_selection_adapter_fn=_prompt,
        component_cache_key_fn=lambda c: str((c or {}).get("id", "")),
        allow_cached_auto=False,
        runtime_runner_fn=_runner,
    )
    assert payload["ok"] is True
    assert payload["cancelled"] is False
    assert payload["lines"] == ["ok_line"]
    assert int(len(prompt_calls)) == 1
    assert prompt_calls[0]["cache_same"] is True
    assert prompt_calls[0]["allow_cached_auto"] is False
    assert str(prompt_calls[0]["flow_label"]) == "redocking_wt"
    assert int(len(runtime_calls)) == 1
    assert runtime_calls[0]["runtime_systems"] == [("WT", "wt_runtime_obj", "wt_src_obj", "WT_model_001")]
    assert runtime_calls[0]["selection_cache"] is cache
    assert callable(runtime_calls[0]["prompt_selection_fn"])

    payload_fail = run_redocking_hetatm_preflight_runtime(
        runtime_systems=[("WT", "wt_runtime_obj", "wt_src_obj", "WT_model_001")],
        selection_cache={},
        collect_components_fn=lambda _obj: [{"id": "a"}],
        prompt_selection_adapter_fn=_prompt,
        component_cache_key_fn=lambda c: str((c or {}).get("id", "")),
        runtime_runner_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert payload_fail["ok"] is False
    assert payload_fail["cancelled"] is False
    assert str(payload_fail["error"]) == "boom"
    assert payload_fail["lines"] == []
    assert payload_fail["selection_cache"] == {}

    payload_bad_prompt = run_redocking_hetatm_preflight_runtime(
        runtime_systems=[],
        selection_cache={},
        collect_components_fn=lambda _obj: [],
        prompt_selection_adapter_fn=None,
        component_cache_key_fn=lambda c: str((c or {}).get("id", "")),
    )
    assert payload_bad_prompt["ok"] is False
    assert payload_bad_prompt["cancelled"] is False
    assert "prompt_selection_adapter_fn is not callable" in str(payload_bad_prompt["error"])


def test_redock_runtime_run_workflow_orchestrator_helper():
    calls = []

    def _prepare_base():
        calls.append("base")
        return {
            "ref_obj": "ref_obj",
            "ref_fields": ("A", "101", "ATP"),
            "center": (1.0, 2.0, 3.0),
            "size": (18.0, 18.0, 18.0),
            "nref": 25,
            "systems": [("WT", "wt_obj")],
            "params": {"repeats": 3, "ph": 7.4},
        }

    def _resolve_lig(ref_obj="", ref_fields=None):
        calls.append(("lig", str(ref_obj), tuple(ref_fields or ())))
        return {"ligand_override_file": "lig.sdf", "use_reference_ligand": True}

    def _prepare_ref_flex(ref_obj="", ref_fields=None):
        calls.append(("ref_flex", str(ref_obj), tuple(ref_fields or ())))
        return {"ref_mol": {"mol": True}, "flex_enabled": True, "flex_rows": [{"resi": "10"}]}

    def _prepare_runtime(ref_obj="", systems=None, params=None):
        calls.append(("runtime", str(ref_obj), int(len(list(systems or [])))))
        return {
            "params": {"repeats": 1, "ph": 7.4},
            "use_modeler_ensemble": True,
            "ensemble_counts": {"WT": 2},
            "input_systems": [("WT", "wt_obj", "wt_obj", "WT_model_001")],
            "runtime_systems": [("WT", "wt_obj_aligned", "wt_obj", "WT_model_001")],
        }

    def _prepare_views(runtime_systems=None, input_systems=None):
        calls.append(("views", int(len(list(runtime_systems or []))), int(len(list(input_systems or [])))))

    def _run_hetatm(runtime_systems=None):
        calls.append(("hetatm", int(len(list(runtime_systems or [])))))
        return {"ok": True, "cancelled": False, "lines": ["ok"]}

    def _run_runtime(**kwargs):
        calls.append(
            (
                "run",
                str(kwargs.get("ref_obj", "")),
                str(kwargs.get("ligand_override_file", "")),
                dict(kwargs.get("ref_mol", {}) or {}),
            )
        )
        return {"ok": True, "cancelled": False, "run_groups": {"WT": [{"ok": True, "rows": []}]}, "rows": []}

    def _build_run_map(systems=None, run_groups=None):
        calls.append(("agg", int(len(list(systems or []))), int(len(dict(run_groups or {})))))
        return {"ok": True, "run_map": {"WT": {"rows": [{"system": "WT"}]}}}

    def _build_context(**kwargs):
        calls.append(("ctx", int(kwargs.get("nref", 0)), int(kwargs.get("repeats", 0))))
        return {"run_map": dict(kwargs.get("run_map", {}) or {}), "nref": int(kwargs.get("nref", 0)), "repeats": int(kwargs.get("repeats", 0))}

    payload = run_redocking_workflow_orchestrator(
        prepare_base_payload_fn=_prepare_base,
        resolve_ligand_override_fn=_resolve_lig,
        prepare_refmol_flex_fn=_prepare_ref_flex,
        prepare_runtime_systems_fn=_prepare_runtime,
        prepare_runtime_views_fn=_prepare_views,
        run_hetatm_preflight_fn=_run_hetatm,
        run_runtime_systems_fn=_run_runtime,
        build_run_map_fn=_build_run_map,
        build_post_context_fn=_build_context,
    )
    assert payload["ok"] is True
    assert payload["stage"] == "done"
    assert int(payload["pipeline_context"]["nref"]) == 25
    assert int(payload["pipeline_context"]["repeats"]) == 1
    assert "WT" in dict(payload["pipeline_context"]["run_map"])
    assert calls[0] == "base"
    assert calls[1][0] == "lig"
    assert calls[2][0] == "ref_flex"
    assert calls[3][0] == "runtime"
    assert calls[4][0] == "views"
    assert calls[5][0] == "hetatm"
    assert calls[6][0] == "run"
    assert dict(calls[6][3] or {}) == {"mol": True}
    assert calls[7][0] == "agg"
    assert calls[8][0] == "ctx"

    payload_cancel = run_redocking_workflow_orchestrator(
        prepare_base_payload_fn=_prepare_base,
        resolve_ligand_override_fn=_resolve_lig,
        prepare_refmol_flex_fn=_prepare_ref_flex,
        prepare_runtime_systems_fn=_prepare_runtime,
        prepare_runtime_views_fn=None,
        run_hetatm_preflight_fn=lambda runtime_systems=None: {"ok": False, "cancelled": True, "lines": []},
        run_runtime_systems_fn=lambda **kwargs: {"ok": True},
        build_run_map_fn=lambda **kwargs: {"ok": True, "run_map": {}},
    )
    assert payload_cancel["ok"] is False
    assert payload_cancel["aborted"] is True
    assert payload_cancel["cancelled"] is True
    assert payload_cancel["stage"] == "hetatm_preflight"


def test_redock_runtime_chainwise_ca_alignment_helper_keeps_mutations_and_chain_map():
    ref_trace = []
    rec_trace = []
    for idx, aa1 in enumerate(["A", "L", "A", "G", "L", "Y"], start=1):
        ref_trace.append({"chain": "A", "resi": idx, "aa1": aa1, "coord": np.array([float(idx), 0.0, 0.0])})
    for idx, aa1 in enumerate(["G", "L", "Y", "S", "E", "R"], start=1):
        ref_trace.append({"chain": "B", "resi": idx, "aa1": aa1, "coord": np.array([float(idx), 1.0, 0.0])})

    # Receptor chain A contains a mutation (F instead of Y) but should still align.
    for idx, aa1 in enumerate(["A", "L", "A", "G", "L", "F"], start=1):
        rec_trace.append({"chain": "A", "resi": idx, "aa1": aa1, "coord": np.array([float(idx), 0.1, 0.0])})
    for idx, aa1 in enumerate(["G", "L", "Y", "S", "E", "R"], start=1):
        rec_trace.append({"chain": "B", "resi": idx, "aa1": aa1, "coord": np.array([float(idx), 1.1, 0.0])})

    def _pairs(seq_a, seq_b):
        assert str(seq_a)
        assert str(seq_b)
        n = min(len(str(seq_a)), len(str(seq_b)))
        return [(i, i) for i in range(n)], n

    payload = build_redocking_chainwise_ca_alignment(
        ref_trace,
        rec_trace,
        sequence_pairs_fn=_pairs,
        min_chain_matches=4,
        min_total_matches=8,
        min_chain_coverage=0.50,
    )
    assert payload["ok"] is True
    assert int(payload["matched_ca"]) == 12
    assert int(payload["identity_count"]) == 11
    assert int(payload["mismatch_count"]) == 1
    chain_pairs = list(payload.get("chain_pairs", []) or [])
    assert chain_pairs[0]["ref_chain"] == "A"
    assert chain_pairs[0]["rec_chain"] == "A"
    assert int(chain_pairs[0]["mismatch_count"]) == 1


def test_redock_runtime_chainwise_ca_alignment_helper_handles_insertions_and_rejects_scattered_matches():
    ref_trace = [
        {"chain": "A", "resi": idx + 1, "aa1": aa1, "coord": np.array([float(idx), 0.0, 0.0])}
        for idx, aa1 in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"])
    ]
    rec_trace = [
        {"chain": "A", "resi": idx + 1, "aa1": aa1, "coord": np.array([float(idx), 0.2, 0.0])}
        for idx, aa1 in enumerate(["A", "B", "X", "Y", "C", "D", "E", "F", "G", "H"])
    ]

    def _pairs_with_insertion(_seq_a, _seq_b):
        return [(0, 0), (1, 1), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9)], 12

    payload = build_redocking_chainwise_ca_alignment(
        ref_trace,
        rec_trace,
        sequence_pairs_fn=_pairs_with_insertion,
        min_chain_matches=6,
        min_total_matches=6,
        min_chain_coverage=0.70,
        min_contiguous_block=4,
    )
    assert payload["ok"] is True
    chain_pair = list(payload.get("chain_pairs", []) or [])[0]
    assert int(chain_pair["max_block_len"]) == 6
    assert int(chain_pair["block_count"]) == 2

    def _pairs_scattered(_seq_a, _seq_b):
        return [(0, 0), (2, 2), (4, 4), (6, 6), (7, 7), (1, 9)], 6

    payload_fail = build_redocking_chainwise_ca_alignment(
        ref_trace,
        rec_trace,
        sequence_pairs_fn=_pairs_scattered,
        min_chain_matches=6,
        min_total_matches=6,
        min_chain_coverage=0.70,
        min_contiguous_block=4,
    )
    assert payload_fail["ok"] is False
    assert "only 0 chain-aware CA matches" in str(payload_fail["error"])


def test_redock_runtime_site_qc_helpers():
    pair_records = [
        {
            "ref_chain": "A",
            "ref_resi": "10",
            "ref_coord": [0.0, 0.0, 0.0],
            "aligned_coord": [0.2, 0.0, 0.0],
        },
        {
            "ref_chain": "A",
            "ref_resi": "11",
            "ref_coord": [1.0, 0.0, 0.0],
            "aligned_coord": [1.2, 0.1, 0.0],
        },
        {
            "ref_chain": "A",
            "ref_resi": "12",
            "ref_coord": [2.0, 0.0, 0.0],
            "aligned_coord": [2.1, -0.1, 0.0],
        },
        {
            "ref_chain": "A",
            "ref_resi": "13",
            "ref_coord": [3.0, 0.0, 0.0],
            "aligned_coord": [3.2, 0.0, 0.1],
        },
    ]
    payload = build_redocking_site_qc_payload(
        pair_records,
        [("A", "10"), ("A", "11"), ("A", "12"), ("A", "13")],
        min_anchor_matches=4,
    )
    assert payload["ok"] is True
    assert int(payload["site_anchor_matches"]) == 4
    assert int(payload["site_anchor_total"]) == 4
    assert float(payload["site_rmsd_a"]) < 0.25

    verdict = resolve_redocking_site_qc_result(
        payload,
        rmsd_max_a=0.50,
        min_anchor_matches=4,
        system_label="WT",
        member_tag="WT_model_001",
    )
    assert verdict["ok"] is True
    assert "WT WT_model_001" in str(verdict["note"])

    verdict_fail = resolve_redocking_site_qc_result(
        payload,
        rmsd_max_a=0.10,
        min_anchor_matches=4,
    )
    assert verdict_fail["ok"] is False
    assert "local site RMSD too high" in str(verdict_fail["error"])

    payload_few = build_redocking_site_qc_payload(
        pair_records[:2],
        [("A", "10"), ("A", "11"), ("A", "12"), ("A", "13")],
        min_anchor_matches=3,
    )
    assert payload_few["ok"] is False
    assert "matched reference-site anchor residues" in str(payload_few["error"])


def test_redock_runtime_prepare_runtime_systems_context_helper():
    payload_plain = prepare_redocking_runtime_systems_context(
        systems=[("WT", "wt_rec")],
        params={"repeats": 3, "seed0": 77},
        use_modeler_ensemble=False,
        align_to_ref=False,
        ref_obj="wt_rec",                    # docking into the reference object itself
    )
    assert payload_plain["ok"] is True
    assert int(payload_plain["params"]["repeats"]) == 3
    assert int(payload_plain["params"]["seed0"]) == 77
    assert payload_plain["runtime_systems"] == [("WT", "wt_rec", "wt_rec", "WT_model_001")]
    assert "Alignment step: disabled." in list(payload_plain.get("align_notes", []))

    # another object with alignment off: box and crystal reference are in the reference frame,
    # so the alignment is forced on
    forced = prepare_redocking_runtime_systems_context(
        systems=[("WT", "wt_rec")],
        params={"repeats": 3, "seed0": 77},
        use_modeler_ensemble=False,
        align_to_ref=False,
        ref_obj="ref_obj",
        align_receptor_to_reference_fn=lambda ref, rec, system_label="": {
            "ok": True, "object": f"{rec}_ali", "rmsd": 0.8, "created": True, "method": "kabsch"},
    )
    assert forced["ok"] is True
    assert forced["runtime_systems"][0][1] == "wt_rec_ali"
    assert any("forced ON" in n for n in forced.get("align_notes", []))

    def _collect(label, rec_obj):
        return {
            "ok": True,
            "systems": [
                (str(label), f"{rec_obj}_m10", str(rec_obj), f"{label}_model_010"),
                (str(label), f"{rec_obj}_m2", str(rec_obj), f"{label}_model_002"),
                (str(label), f"{rec_obj}_m1", str(rec_obj), f"{label}_model_001"),
            ],
            "model_count": 3,
            "note": f"{label} ensemble ready",
        }

    align_calls = []

    def _align(ref_obj, rec_obj, system_label=""):
        align_calls.append((str(ref_obj), str(rec_obj), str(system_label)))
        idx = 1
        if str(rec_obj).endswith("_m2"):
            idx = 2
        elif str(rec_obj).endswith("_m10"):
            idx = 10
        return {
            "ok": True,
            "object": f"{rec_obj}_ali",
            "rmsd": float(1.0 + (0.2 * idx)),
            "created": True,
            "method": "kabsch",
            "site_rmsd_a": float(0.60 + (0.05 * idx)),
            "site_anchor_matches": 8,
            "site_anchor_total": 10,
            "site_anchor_cutoff_a": 6.0,
        }

    payload_ensemble = prepare_redocking_runtime_systems_context(
        systems=[("WT", "wt_rec")],
        params={"repeats": 5, "seed0": 11},
        use_modeler_ensemble=True,
        align_to_ref=False,
        ref_obj="ref_obj",
        collect_modeler_ensemble_fn=_collect,
        align_receptor_to_reference_fn=_align,
        align_rmsd_max_a=6.0,
        site_qc_enabled=True,
        site_rmsd_max_a=1.5,
        site_qc_min_anchor_matches=4,
        site_anchor_cutoff_a=6.0,
    )
    assert payload_ensemble["ok"] is True
    assert int(payload_ensemble["params"]["repeats"]) == 1
    # one run per member, but a requested seed is kept (reproducible ensemble redocking)
    assert int(payload_ensemble["params"]["seed0"]) == 11
    assert payload_ensemble["align_to_ref"] is True
    assert int(payload_ensemble["ensemble_counts"]["WT"]) == 3
    assert payload_ensemble["input_systems"] == [
        ("WT", "wt_rec_m1", "wt_rec", "WT_model_001"),
        ("WT", "wt_rec_m2", "wt_rec", "WT_model_002"),
        ("WT", "wt_rec_m10", "wt_rec", "WT_model_010"),
    ]
    assert payload_ensemble["runtime_systems"] == [
        ("WT", "wt_rec_m1_ali", "wt_rec", "WT_model_001"),
        ("WT", "wt_rec_m2_ali", "wt_rec", "WT_model_002"),
        ("WT", "wt_rec_m10_ali", "wt_rec", "WT_model_010"),
    ]
    assert int(len(align_calls)) == 3
    assert align_calls == [
        ("ref_obj", "wt_rec_m1", "WT"),
        ("ref_obj", "wt_rec_m2", "WT"),
        ("ref_obj", "wt_rec_m10", "WT"),
    ]
    assert any("forced ON" in str(line) for line in payload_ensemble.get("align_notes", []))
    assert any("median RMSD=" in str(line) for line in payload_ensemble.get("align_notes", []))
    assert payload_ensemble["site_qc_enabled"] is True
    assert any("local-site QC" in str(line) for line in payload_ensemble.get("site_qc_notes", []))

    def _align_high(_ref_obj, _rec_obj, system_label=""):
        return {
            "ok": True,
            "object": "x",
            "rmsd": 9.2,
            "details": f"{system_label} too far",
        }

    payload_fail = prepare_redocking_runtime_systems_context(
        systems=[("WT", "wt_rec")],
        params={},
        use_modeler_ensemble=False,
        align_to_ref=True,
        ref_obj="ref_obj",
        align_receptor_to_reference_fn=_align_high,
        align_rmsd_max_a=6.0,
    )
    assert payload_fail["ok"] is False
    assert "alignment RMSD too high" in str(payload_fail["error"])

    def _align_site_fail(_ref_obj, rec_obj, system_label=""):
        _ = (rec_obj, system_label)
        return {
            "ok": True,
            "object": "wt_rec_ali",
            "rmsd": 1.1,
            "site_rmsd_a": 2.6,
            "site_anchor_matches": 7,
            "site_anchor_total": 9,
            "site_anchor_cutoff_a": 6.0,
        }

    payload_site_fail = prepare_redocking_runtime_systems_context(
        systems=[("WT", "wt_rec")],
        params={},
        use_modeler_ensemble=False,
        align_to_ref=True,
        ref_obj="ref_obj",
        align_receptor_to_reference_fn=_align_site_fail,
        align_rmsd_max_a=6.0,
        site_qc_enabled=True,
        site_rmsd_max_a=2.0,
        site_qc_min_anchor_matches=4,
        site_anchor_cutoff_a=6.0,
    )
    assert payload_site_fail["ok"] is False
    assert "local site RMSD too high" in str(payload_site_fail["error"])


def test_redock_runtime_prepare_runtime_systems_runtime_helper():
    calls = []

    def _builder(**kwargs):
        calls.append(dict(kwargs or {}))
        return {
            "ok": True,
            "error": "",
            "params": dict(kwargs.get("params", {}) or {}),
            "use_modeler_ensemble": bool(kwargs.get("use_modeler_ensemble", False)),
            "ensemble_notes": ["ok"],
            "ensemble_counts": {"WT": 2},
            "input_systems": [("WT", "wt_rec", "wt_rec", "WT_model_001")],
            "runtime_systems": [("WT", "wt_rec_ali", "wt_rec", "WT_model_001")],
            "align_notes": ["aligned"],
            "align_to_ref": bool(kwargs.get("align_to_ref", False)),
        }

    collect_fn = lambda label, rec_obj: {"ok": True, "systems": [(label, rec_obj)]}
    align_fn = lambda ref_obj, rec_obj, system_label="": {"ok": True, "object": rec_obj, "rmsd": 1.0}
    payload = prepare_redocking_runtime_systems_runtime(
        systems=[("WT", "wt_rec")],
        params={"repeats": 5},
        use_modeler_ensemble=True,
        align_to_ref=True,
        ref_obj="ref_obj",
        collect_modeler_ensemble_fn=collect_fn,
        align_receptor_to_reference_fn=align_fn,
        force_align_when_ensemble=False,
        align_rmsd_max_a=4.5,
        site_qc_enabled=True,
        site_rmsd_max_a=1.8,
        site_qc_min_anchor_matches=5,
        site_anchor_cutoff_a=6.0,
        context_builder_fn=_builder,
    )
    assert payload["ok"] is True
    assert payload["align_to_ref"] is True
    assert int(payload["ensemble_counts"]["WT"]) == 2
    assert int(len(calls)) == 1
    sent = dict(calls[0] or {})
    assert sent["systems"] == [("WT", "wt_rec")]
    assert sent["params"] == {"repeats": 5}
    assert bool(sent["use_modeler_ensemble"]) is True
    assert bool(sent["align_to_ref"]) is True
    assert str(sent["ref_obj"]) == "ref_obj"
    assert sent["collect_modeler_ensemble_fn"] is collect_fn
    assert sent["align_receptor_to_reference_fn"] is align_fn
    assert bool(sent["force_align_when_ensemble"]) is False
    assert float(sent["align_rmsd_max_a"]) == 4.5
    assert bool(sent["site_qc_enabled"]) is True
    assert float(sent["site_rmsd_max_a"]) == 1.8
    assert int(sent["site_qc_min_anchor_matches"]) == 5
    assert float(sent["site_anchor_cutoff_a"]) == 6.0

    payload_fail = prepare_redocking_runtime_systems_runtime(
        systems=[("WT", "wt_rec")],
        params={},
        use_modeler_ensemble=False,
        align_to_ref=False,
        ref_obj="",
        context_builder_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert payload_fail["ok"] is False
    assert str(payload_fail["error"]) == "boom"


def test_redock_runtime_run_single_run_pipeline_helper(tmp_path):
    def _runner_ok(
        base_cmd,
        job_dir,
        n_repeats,
        seed0,
        run_command,
        parse_pose_scores,
        build_rows_for_repeat,
        timeout=7200,
    ):
        _ = (run_command, parse_pose_scores, timeout)
        assert str(base_cmd[0]) == "smina_bin"
        assert int(n_repeats) == 3
        assert int(seed0) == 77
        out_pose = os.path.join(str(job_dir), "docked_poses.pdbqt")
        out_log = os.path.join(str(job_dir), "docking.log")
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        rows = list(build_rows_for_repeat(1, [{"mode": 1, "affinity": -8.2}], out_pose, out_log) or [])
        best = dict(rows[0]) if rows else None
        return {
            "ok": True,
            "repeat_rows": [{"repeat": 1, "ok": True, "best_affinity": -8.2, "best_rmsd": 1.1}],
            "best_rows": list(rows),
            "best_repeat": best,
            "best_repeat_id": 1,
            "best_pose_file": str(out_pose),
        }

    def _to_sdf(_src, dst):
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True

    payload = run_redocking_single_run_pipeline(
        smina_bin="smina_bin",
        receptor_for_dock="rec.pdbqt",
        ligand_input="lig.pdbqt",
        center=(1.0, 2.0, 3.0),
        size=(18.0, 18.0, 18.0),
        params={"repeats": 3, "seed0": 77, "num_modes": 10},
        flex_file="",
        job_dir=str(tmp_path),
        system_label="WT",
        receptor_obj="wt_rec",
        source_receptor_obj="wt_src",
        tag="20260304_010203",
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        convert_pdbqt_to_sdf_fn=_to_sdf,
        convert_pdbqt_to_pdb_fn=lambda _s, _d: False,
        load_pose_object_fn=lambda _pv, _po: {"ok": True},
        run_command=lambda *_a, **_k: (True, "", "", 0),
        parse_pose_scores_fn=lambda _p, _l: [{"mode": 1, "affinity": -8.2}],
        compute_rmsd_values_fn=lambda *_a, **_k: [1.1],
        run_repeats_fn=_runner_ok,
        repeat_statistics_fn=lambda vals, bootstrap_n=0, seed=0: {
            "n": int(len(list(vals or []))),
            "mean": float(sum(list(vals or [])) / float(max(1, len(list(vals or []))))),
            "ci_low": -1.0,
            "ci_high": 1.0,
            "confidence": "high",
        },
        repeat_seed=12345,
        repeat_bootstrap_n=2000,
        repeat_csv_path=str(tmp_path / "repeat_stats.csv"),
        receptor_input="rec.pdbqt",
        removed_reference_atoms=3,
        flex_enabled=False,
        flex_rows=[],
        rec_info={"notes": ["ok"]},
        lig_info={"notes": ["ok"]},
        flex_info={"ok": False},
        timeout=30,
        pose_prefix="redock",
    )
    assert payload["ok"] is True
    assert int(payload["plan_payload"]["n_repeats"]) == 3
    assert int(payload["plan_payload"]["seed0"]) == 77
    assert int(payload["repeat_payload"]["best_repeat_id"]) == 1
    assert int(len(payload.get("debug_messages", []))) >= 1
    assert any("Redocking WT: receptor=wt_rec, source=wt_src" in str(m) for m in payload.get("debug_messages", []))
    run_payload = dict(payload.get("run_payload", {}) or {})
    assert run_payload.get("ok") is True
    assert int(run_payload.get("repeat_count", 0)) == 3
    assert int(run_payload.get("repeat_best", 0)) == 1
    assert os.path.isfile(str(run_payload.get("repeat_csv", "")))

    def _runner_fail(
        base_cmd,
        job_dir,
        n_repeats,
        seed0,
        run_command,
        parse_pose_scores,
        build_rows_for_repeat,
        timeout=7200,
    ):
        _ = (base_cmd, job_dir, n_repeats, seed0, run_command, parse_pose_scores, build_rows_for_repeat, timeout)
        return {"ok": False, "error": "pipeline failed", "repeat_rows": [{"repeat": 1, "ok": False}]}

    payload_fail = run_redocking_single_run_pipeline(
        smina_bin="smina_bin",
        receptor_for_dock="rec.pdbqt",
        ligand_input="lig.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        params={},
        job_dir=str(tmp_path),
        system_label="WT",
        receptor_obj="wt_rec",
        run_command=lambda *_a, **_k: (False, "", "err", 1),
        parse_pose_scores_fn=lambda _p, _l: [],
        compute_rmsd_values_fn=lambda *_a, **_k: [],
        run_repeats_fn=_runner_fail,
    )
    assert payload_fail["ok"] is False
    assert "pipeline failed" in str(payload_fail["error"])
    assert payload_fail.get("debug_messages") == []


def test_redock_runtime_run_single_system_orchestrator_helper(tmp_path):
    rec_in = tmp_path / "rec_input.pdbqt"
    rec_in.write_text("RECEPTOR\n", encoding="utf-8")

    def _prepare_receptor(_obj, _job_dir, target_ph=7.4, flow_label="", auto_use_cached_hetatm=False):
        _ = (target_ph, flow_label, auto_use_cached_hetatm)
        return str(rec_in), {"notes": ["rec_ok"]}

    def _strip_ref(_receptor_input, stripped_file, _ch, _ri, _rn):
        with open(stripped_file, "w", encoding="utf-8") as fh:
            fh.write("STRIPPED\n")
        return 2

    def _prepare_ligand(_lig_file, _job_dir, target_ph=7.4, smiles_hint="", preserve_input_charges=False):
        _ = (target_ph, smiles_hint, preserve_input_charges)
        lig_out = str(tmp_path / "lig_prepared.pdbqt")
        with open(lig_out, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return lig_out, {"notes": ["lig_ok"]}

    def _single_runner(**kwargs):
        _ = kwargs.get("compute_rmsd_values_fn")(
            2,
            [{"mode": 1, "affinity": -8.0}],
            "pose.pdbqt",
            out_log="dock.log",
        )
        return {
            "ok": True,
            "run_payload": {"ok": True, "rows": [{"mode": 1}], "repeat_count": 1},
            "debug_messages": ["pipeline_debug"],
            "plan_payload": {"n_repeats": 1, "seed0": 0},
            "repeat_payload": {"ok": True},
        }

    def _rmsd_builder(
        *,
        out_pose,
        poses_count,
        has_rdkit=False,
        ref_mol=None,
        convert_pdbqt_to_sdf_fn=None,
        load_pose_mols_fn=None,
        best_rmsd_fn=None,
        log_warning_fn=None,
    ):
        _ = (out_pose, poses_count, has_rdkit, ref_mol, convert_pdbqt_to_sdf_fn, load_pose_mols_fn, best_rmsd_fn)
        if callable(log_warning_fn):
            log_warning_fn("synthetic warning")
        return [1.25]

    payload = run_redocking_single_system_orchestrator(
        system_label="WT",
        receptor_obj="wt_rec",
        ref_fields=("A", "101", "ATP"),
        ref_mol=object(),
        center=(1.0, 2.0, 3.0),
        size=(18.0, 18.0, 18.0),
        params={"ph": 7.0},
        source_receptor_obj="wt_src",
        flex_enabled=False,
        flex_residues=[],
        auto_use_cached_hetatm=False,
        ligand_override_file="",
        smina_bin="smina_bin",
        picked_ligand_file="lig.sdf",
        workspace_dir=str(tmp_path),
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        file_exists_fn=lambda p: str(p) in {"lig.sdf", str(rec_in), str(tmp_path / "lig_prepared.pdbqt")} or os.path.isfile(str(p)),
        makedirs_fn=os.makedirs,
        prepare_receptor_fn=_prepare_receptor,
        strip_reference_residue_fn=_strip_ref,
        prepare_flexible_receptor_fn=None,
        should_preserve_charges_fn=lambda _f: False,
        current_charge_mode_fn=lambda: "add",
        parameterized_ligand_mol2="",
        samefile_or_equal_fn=None,
        chemically_equivalent_fn=None,
        pubchem_smiles_hint_fn=lambda: "",
        rcsb_smiles_by_het_fn=lambda _rn: "",
        prepare_ligand_fn=_prepare_ligand,
        has_rdkit=True,
        convert_pdbqt_to_sdf_fn=lambda _s, _d: False,
        convert_pdbqt_to_pdb_fn=lambda _s, _d: False,
        load_pose_molecules_fn=lambda _f: [],
        best_rmsd_fn=lambda _r, _p: 0.0,
        load_pose_object_fn=lambda _pv, _po: {"ok": True},
        run_command=lambda *_a, **_k: (True, "", "", 0),
        parse_pose_scores_fn=lambda _p, _l: [{"mode": 1, "affinity": -8.0}],
        repeat_statistics_fn=lambda vals, bootstrap_n=0, seed=0: {
            "n": int(len(list(vals or []))),
            "mean": 0.0,
            "ci_low": -1.0,
            "ci_high": 1.0,
            "confidence": "high",
        },
        timeout=30,
        compute_rmsd_values_for_repeat_fn=_rmsd_builder,
        single_run_pipeline_fn=_single_runner,
    )
    assert payload["ok"] is True
    assert payload["run_payload"]["ok"] is True
    dbg = list(payload.get("debug_messages", []) or [])
    assert any("synthetic warning" in str(m) for m in dbg)
    assert any("pipeline_debug" in str(m) for m in dbg)

    payload_fail = run_redocking_single_system_orchestrator(
        system_label="WT",
        receptor_obj="wt_rec",
        ref_fields=("A", "101", "ATP"),
        ref_mol=None,
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        params={},
        smina_bin="",
        picked_ligand_file="lig.sdf",
        workspace_dir=str(tmp_path),
    )
    assert payload_fail["ok"] is False
    assert "smina was not found in PATH." in str(payload_fail["error"])
    assert isinstance(payload_fail.get("debug_messages", []), list)


def test_redock_runtime_orchestrator_prefers_explicit_grid_over_reference_autobox(tmp_path):
    rec_in = tmp_path / "rec_input.pdbqt"
    rec_in.write_text("RECEPTOR\n", encoding="utf-8")
    lig_override = tmp_path / "ref_ligand.sdf"
    lig_override.write_text("ligand\n", encoding="utf-8")

    captured = {}

    def _prepare_receptor(_obj, _job_dir, target_ph=7.4, flow_label="", auto_use_cached_hetatm=False):
        _ = (target_ph, flow_label, auto_use_cached_hetatm)
        return str(rec_in), {"notes": ["rec_ok"]}

    def _strip_ref(_receptor_input, stripped_file, _ch, _ri, _rn):
        with open(stripped_file, "w", encoding="utf-8") as fh:
            fh.write("STRIPPED\n")
        return 2

    def _prepare_ligand(_lig_file, _job_dir, target_ph=7.4, smiles_hint="", preserve_input_charges=False):
        _ = (target_ph, smiles_hint, preserve_input_charges)
        lig_out = str(tmp_path / "lig_prepared.pdbqt")
        with open(lig_out, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return lig_out, {"notes": ["lig_ok"]}

    def _single_runner(**kwargs):
        captured.update(dict(kwargs or {}))
        return {
            "ok": True,
            "run_payload": {"ok": True, "rows": [{"mode": 1}], "repeat_count": 1},
            "debug_messages": [],
            "plan_payload": {"n_repeats": 1, "seed0": 0},
            "repeat_payload": {"ok": True},
        }

    payload = run_redocking_single_system_orchestrator(
        system_label="WT",
        receptor_obj="wt_aligned",
        reference_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        ref_mol=object(),
        center=(10.0, 20.0, 30.0),
        size=(18.0, 19.0, 20.0),
        params={"ph": 7.0},
        source_receptor_obj="wt_source",
        flex_enabled=False,
        flex_residues=[],
        auto_use_cached_hetatm=False,
        ligand_override_file=str(lig_override),
        smina_bin="smina_bin",
        picked_ligand_file="lig.sdf",
        workspace_dir=str(tmp_path),
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        file_exists_fn=lambda p: str(p) in {"lig.sdf", str(rec_in), str(lig_override), str(tmp_path / "lig_prepared.pdbqt")} or os.path.isfile(str(p)),
        makedirs_fn=os.makedirs,
        prepare_receptor_fn=_prepare_receptor,
        strip_reference_residue_fn=_strip_ref,
        prepare_flexible_receptor_fn=None,
        should_preserve_charges_fn=lambda _f: False,
        current_charge_mode_fn=lambda: "add",
        parameterized_ligand_mol2="",
        samefile_or_equal_fn=None,
        chemically_equivalent_fn=None,
        pubchem_smiles_hint_fn=lambda: "",
        rcsb_smiles_by_het_fn=lambda _rn: "",
        prepare_ligand_fn=_prepare_ligand,
        has_rdkit=True,
        convert_pdbqt_to_sdf_fn=lambda _s, _d: False,
        convert_pdbqt_to_pdb_fn=lambda _s, _d: False,
        load_pose_molecules_fn=lambda _f: [],
        best_rmsd_fn=lambda _r, _p: 0.0,
        load_pose_object_fn=lambda _pv, _po: {"ok": True},
        run_command=lambda *_a, **_k: (True, "", "", 0),
        parse_pose_scores_fn=lambda _p, _l: [{"mode": 1, "affinity": -8.0}],
        repeat_statistics_fn=lambda vals, bootstrap_n=0, seed=0: {
            "n": int(len(list(vals or []))),
            "mean": 0.0,
            "ci_low": -1.0,
            "ci_high": 1.0,
            "confidence": "high",
        },
        timeout=30,
        single_run_pipeline_fn=_single_runner,
    )

    assert payload["ok"] is True
    assert payload["run_payload"]["ok"] is True
    assert str(captured.get("autobox_ligand", "") or "") == ""
    assert captured.get("center") == (10.0, 20.0, 30.0)
    assert captured.get("size") == (18.0, 19.0, 20.0)
    dbg = list(payload.get("debug_messages", []) or [])
    assert any("using explicit aligned grid center/size" in str(m) for m in dbg)


def test_redock_runtime_run_single_system_runtime_helper(tmp_path):
    calls = []

    def _orchestrator(**kwargs):
        calls.append(dict(kwargs))
        return {
            "ok": True,
            "run_payload": {"ok": True, "rows": [{"mode": 1}], "repeat_count": 1},
            "debug_messages": ["runtime_debug", "", "   "],
        }

    debug_lines = []

    payload = run_redocking_single_system_runtime(
        system_label="WT",
        receptor_obj="wt_obj",
        reference_obj="ref_obj",
        ref_fields=("A", "101", "ATP"),
        ref_mol=object(),
        center=(1.0, 2.0, 3.0),
        size=(18.0, 18.0, 18.0),
        params={"ph": 7.4},
        source_receptor_obj="",
        flex_enabled=True,
        flex_residues=[("A", "10", "GLU")],
        auto_use_cached_hetatm=True,
        ligand_override_file="override.sdf",
        tool_path_fn=lambda name: "smina_bin" if str(name) == "smina" else "",
        pick_ligand_input_fn=lambda: "lig.sdf",
        workspace_dir_fn=lambda: str(tmp_path),
        safe_name_fn=lambda x: str(x),
        file_exists_fn=lambda _p: True,
        makedirs_fn=os.makedirs,
        prepare_receptor_fn=lambda *_a, **_k: ("rec.pdbqt", {}),
        strip_reference_residue_fn=lambda *_a, **_k: 0,
        prepare_flexible_receptor_fn=lambda *_a, **_k: ("rec_flex.pdbqt", "flex.pdbqt", {}),
        should_preserve_charges_fn=lambda _f: False,
        current_charge_mode_fn=lambda: "add",
        parameterized_ligand_mol2="param.mol2",
        samefile_or_equal_fn=lambda _a, _b: False,
        chemically_equivalent_fn=lambda _a, _b: False,
        pubchem_smiles_hint_fn=lambda: "",
        rcsb_smiles_by_het_fn=lambda _rn: "",
        prepare_ligand_fn=lambda *_a, **_k: ("lig.pdbqt", {}),
        has_rdkit=True,
        convert_pdbqt_to_sdf_fn=lambda _s, _d: False,
        convert_pdbqt_to_pdb_fn=lambda _s, _d: False,
        load_pose_molecules_fn=lambda _f: [],
        best_rmsd_fn=lambda _r, _p: 0.0,
        load_pose_object_fn=lambda _pv, _po: {"ok": True},
        run_command_fn=lambda *_a, **_k: (True, "", "", 0),
        parse_pose_scores_fn=lambda _p, _l: [{"mode": 1, "affinity": -8.0}],
        repeat_statistics_fn=lambda vals, bootstrap_n=0, seed=0: {"n": len(list(vals or [])), "mean": 0.0},
        timeout=45,
        flow_label="redocking",
        pose_prefix="redock",
        repeat_seed=999,
        repeat_bootstrap_n=111,
        repeat_csv_filename="stats.csv",
        append_debug_log_fn=lambda msg: debug_lines.append(str(msg)),
        orchestrator_fn=_orchestrator,
    )
    assert payload["ok"] is True
    assert payload["repeat_count"] == 1
    assert debug_lines == ["runtime_debug"]

    sent = dict(calls[0] or {})
    assert str(sent["smina_bin"]) == "smina_bin"
    assert str(sent["picked_ligand_file"]) == "lig.sdf"
    assert str(sent["workspace_dir"]) == str(tmp_path)
    assert str(sent["source_receptor_obj"]) == "wt_obj"
    assert str(sent["parameterized_ligand_mol2"]) == "param.mol2"
    assert int(sent["timeout"]) == 45
    assert str(sent["repeat_csv_filename"]) == "stats.csv"
    assert bool(sent["auto_use_cached_hetatm"]) is True
    assert bool(sent["flex_enabled"]) is True

    payload_fail = run_redocking_single_system_runtime(
        system_label="WT",
        receptor_obj="wt_obj",
        ref_fields=("A", "101", "ATP"),
        ref_mol=None,
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        params={},
        tool_path_fn=lambda _name: "",
        pick_ligand_input_fn=lambda: "lig.sdf",
        workspace_dir_fn=lambda: str(tmp_path),
        append_debug_log_fn=lambda msg: debug_lines.append(str(msg)),
        orchestrator_fn=lambda **_kwargs: {"ok": False, "error": "boom", "debug_messages": ["fail_debug"]},
    )
    assert payload_fail["ok"] is False
    assert str(payload_fail["error"]) == "boom"
    assert "fail_debug" in debug_lines


def test_redock_runtime_compose_single_run_outcome(tmp_path):
    rows = [
        {"mode": 2, "affinity": -8.5, "rmsd_ref": 1.4, "receptor": "rec_a", "receptor_source": "rec_src_a"},
        {"mode": 1, "affinity": -7.9, "rmsd_ref": 1.8, "receptor": "rec_a", "receptor_source": "rec_src_a"},
    ]
    repeat_rows = [
        {"repeat": 1, "ok": True, "best_affinity": -8.5, "best_rmsd": 1.4, "seed": 111},
        {"repeat": 2, "ok": True, "best_affinity": -8.1, "best_rmsd": 1.6, "seed": 222},
    ]

    def _repeat_stats(vals, bootstrap_n=0, seed=0):
        _ = (bootstrap_n, seed)
        vv = [float(v) for v in list(vals or [])]
        n = int(len(vv))
        mean_v = float(sum(vv) / float(n)) if n > 0 else float("nan")
        return {"n": int(n), "mean": float(mean_v), "ci_low": -1.0, "ci_high": 1.0, "confidence": "high"}

    payload = compose_redocking_single_run_outcome(
        best_rows=rows,
        repeat_rows_meta=repeat_rows,
        job_dir="job_dir_a",
        repeat_statistics_fn=_repeat_stats,
        repeat_seed=999,
        repeat_bootstrap_n=321,
        repeat_csv_path=str(tmp_path / "repeat_stats.csv"),
        system_label="WT",
        receptor_obj="rec_a",
        source_receptor_obj="rec_src_a",
        receptor_input="rec.pdbqt",
        ligand_input="lig.pdbqt",
        removed_reference_atoms=5,
        n_repeats=2,
        best_repeat_id=1,
        flex_enabled=True,
        flex_count=3,
        flex_file="flex.pdbqt",
        rec_info={"notes": ["ok"]},
        lig_info={"notes": ["ok"]},
        flex_info={"ok": True},
    )
    assert "Redocking WT: receptor=rec_a, source=rec_src_a" in str(payload["debug_line"])
    assert "best_repeat=1" in str(payload["debug_line"])
    assert "flex_count=3" in str(payload["debug_line"])
    assert int(payload["repeat_stats_affinity"]["n"]) == 2
    assert int(payload["repeat_stats_rmsd"]["n"]) == 2
    assert os.path.isfile(str(payload["repeat_csv"]))

    run_payload = dict(payload["run_payload"])
    assert run_payload["ok"] is True
    assert str(run_payload["job_dir"]) == "job_dir_a"
    assert str(run_payload["receptor_object"]) == "rec_a"
    assert str(run_payload["source_receptor"]) == "rec_src_a"
    assert int(run_payload["repeat_count"]) == 2
    assert int(run_payload["repeat_best"]) == 1


def test_redock_runtime_repeat_selection(tmp_path):
    call_idx = {"i": 0}

    def _run_command(argv, cwd=None, timeout=0):
        call_idx["i"] += 1
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        return True, "", "", 0

    def _parse_pose_scores(_pose, _log):
        return [{"mode": 1, "affinity": -8.0}]

    def _build_rows_for_repeat(rep, poses, _out_pose, _out_log):
        _ = poses
        rmsd_map = {1: 2.4, 2: 1.2, 3: 1.2}
        aff_map = {1: -8.0, 2: -7.0, 3: -9.0}
        return [
            {
                "mode": 1,
                "state": 1,
                "repeat": int(rep),
                "rmsd_ref": float(rmsd_map[int(rep)]),
                "affinity": float(aff_map[int(rep)]),
            }
        ]

    payload = run_redocking_repeats(
        base_cmd=["smina", "-r", "r.pdbqt", "-l", "l.pdbqt"],
        job_dir=str(tmp_path),
        n_repeats=3,
        seed0=200,
        run_command=_run_command,
        parse_pose_scores=_parse_pose_scores,
        build_rows_for_repeat=_build_rows_for_repeat,
        timeout=10,
    )

    assert payload["ok"] is True
    assert int(payload["best_repeat_id"]) == 3
    assert float(payload["best_repeat"]["rmsd_ref"]) == 1.2
    assert float(payload["best_repeat"]["affinity"]) == -9.0
    # the repeat with the better top-scored pose wins, whatever its RMSD
    assert is_better_repeat({"rmsd_ref": 1.0, "affinity": -6.0}, {"rmsd_ref": 2.0, "affinity": -10.0}) is False
    assert is_better_repeat({"rmsd_ref": 3.0, "affinity": -10.0}, {"rmsd_ref": 1.0, "affinity": -6.0}) is True
    assert is_better_repeat({"rmsd_ref": 1.0, "affinity": -8.0}, {"rmsd_ref": 2.0, "affinity": -8.0}) is True
    assert pick_best_pose_row(payload["best_rows"]) is not None
    meta = {int(r["repeat"]): r for r in payload["repeat_rows"]}
    assert float(meta[1]["best_rmsd"]) == 2.4 and float(meta[1]["min_rmsd_all_poses"]) == 2.4


def test_redock_runtime_repeat_selection_falls_back_to_stdout_scores_and_sets_out_flex(tmp_path):
    seen = {"argv": None}

    def _run_command(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        seen["argv"] = list(argv)
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("MODEL 1\nENDMDL\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("Weights\n")
        return True, "    1   -7.6    0.000    0.000\n", "", 0

    def _build_rows_for_repeat(rep, poses, _out_pose, _out_log):
        pose = dict(list(poses or [])[0] or {})
        return [
            {
                "mode": int(pose.get("mode", 0) or 0),
                "state": 1,
                "repeat": int(rep),
                "rmsd_ref": 0.55,
                "affinity": float(pose.get("affinity", 0.0) or 0.0),
            }
        ]

    payload = run_redocking_repeats(
        base_cmd=["smina", "-r", "r.pdbqt", "-l", "l.pdbqt", "--flexres", "A:45"],
        job_dir=str(tmp_path),
        n_repeats=1,
        seed0=0,
        run_command=_run_command,
        parse_pose_scores=lambda _pose, _log: [],
        build_rows_for_repeat=_build_rows_for_repeat,
        timeout=10,
    )

    assert payload["ok"] is True
    assert float(payload["best_repeat"]["affinity"]) == -7.6
    assert "--out_flex" in list(seen["argv"] or [])
    out_flex = seen["argv"][seen["argv"].index("--out_flex") + 1]
    assert str(out_flex).endswith("flex_residues_out.pdbqt")


def test_redock_runtime_repeat_selection_retries_flex_autobox_on_search_space_warning(tmp_path):
    calls = []

    def _run_command(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        calls.append(list(argv))
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        autobox_add = float(argv[argv.index("--autobox_add") + 1])
        if len(calls) == 1:
            with open(out_pose, "w", encoding="utf-8") as fh:
                fh.write("MODEL 1\nENDMDL\n")
            with open(out_log, "w", encoding="utf-8") as fh:
                fh.write(
                    "WARNING: Could not find any conformations completely within the search space.\n"
                    "WARNING: Check that it is large enough for all movable atoms, including those in the flexible side chains.\n"
                )
            return True, "", "", 0
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("MODEL 1\nREMARK minimizedAffinity -7.2\nREMARK minimizedRMSD -1\nENDMDL\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write(f"Using autobox_add {autobox_add:.3f}\n")
        return True, "", "", 0

    def _build_rows_for_repeat(rep, poses, _out_pose, _out_log):
        pose = dict(list(poses or [])[0] or {})
        return [
            {
                "mode": int(pose.get("mode", 0) or 0),
                "state": 1,
                "repeat": int(rep),
                "rmsd_ref": 0.42,
                "affinity": float(pose.get("affinity", 0.0) or 0.0),
            }
        ]

    payload = run_redocking_repeats(
        base_cmd=[
            "smina",
            "-r",
            "rec.pdbqt",
            "-l",
            "lig.pdbqt",
            "--autobox_ligand",
            "ref.pdb",
            "--autobox_add",
            "4.000",
            "--flexres",
            "A:16",
        ],
        job_dir=str(tmp_path),
        n_repeats=1,
        seed0=0,
        run_command=_run_command,
        parse_pose_scores=parse_smina_pose_scores,
        build_rows_for_repeat=_build_rows_for_repeat,
        timeout=10,
    )

    assert payload["ok"] is True
    assert float(payload["best_repeat"]["affinity"]) == -7.2
    assert len(calls) == 2
    assert float(calls[0][calls[0].index("--autobox_add") + 1]) == 4.0
    assert float(calls[1][calls[1].index("--autobox_add") + 1]) == 10.0


def test_redock_runtime_comparison_and_summary_helpers():
    wt = {
        "best": {"mode": 1, "affinity": -8.0, "rmsd_ref": 1.2},
        "repeat_stats_affinity": {"n": 4, "mean": -7.8, "median": -7.9, "confidence": "high"},
        "repeat_stats_rmsd": {"n": 4, "mean": 1.35, "median": 1.2, "confidence": "high", "ci_half": 0.4},
    }
    mut = {
        "best": {"mode": 1, "affinity": -6.2, "rmsd_ref": 2.8},
        "repeat_stats_affinity": {"n": 4, "mean": -6.1, "median": -6.0, "confidence": "medium"},
        "repeat_stats_rmsd": {"n": 4, "mean": 2.7, "median": 2.9, "confidence": "medium", "ci_half": 0.7},
    }

    aff, rms, aff_src, rms_src, best = pick_comparison_metrics(wt)
    assert np.isclose(float(aff), -7.9)
    assert np.isclose(float(rms), 1.2)
    assert "ensemble median" in aff_src
    assert "ensemble median" in rms_src
    assert int(best.get("mode", 0)) == 1

    mut_aff, mut_rms, _mut_aff_src, _mut_rms_src, _mut_best = pick_comparison_metrics(mut)
    d_aff = float(mut_aff) - float(aff)
    d_rms = float(mut_rms) - float(rms)
    cls, rationale, conf, evidence = classify_delta(d_aff, d_rms)
    assert cls in {"Resistant-like", "Sensitive-like", "Neutral-like", "Impacted (mixed effect)"}
    assert isinstance(rationale, str) and rationale
    assert conf in {"low", "medium", "high"}
    assert float(evidence) >= 0.0

    sample_conf, sample_note = sampling_confidence_from_runs(wt, mut)
    assert sample_conf in {"low", "medium", "high", "undefined"}
    assert isinstance(sample_note, str)

    wt_quality, wt_note = wt_quality_from_run(wt, wt_mean_rms=1.35, wt_best_rms=1.2)
    assert wt_quality in {"excellent", "good", "borderline", "poor", "unknown"}
    assert isinstance(wt_note, str)

    wt_stats = stats_for_label(wt)
    assert str(wt_stats["aff_metric_label"]) == "median"
    assert str(wt_stats["rms_metric_label"]) == "median"
    assert int(wt_stats["n_aff"]) == 4
    assert int(wt_stats["n_rms"]) == 4

    cmp_txt = compare_wt_mutant(wt, mut)
    assert "Classification:" in cmp_txt
    assert "WT:" in cmp_txt and "Mutant:" in cmp_txt
    assert "ensemble median" in cmp_txt
    assert "ligand RMSD to crystal=" in cmp_txt

    lines = build_redocking_summary_lines(
        {"WT": wt, "Mutant": mut},
        nref=28,
        center=(1.0, 2.0, 3.0),
        size=(18.0, 18.0, 18.0),
        use_reference_ligand=True,
        use_modeler_ensemble=False,
        repeats=4,
        site_qc_enabled=True,
        site_qc_rmsd_max_a=2.0,
        site_qc_min_anchor_matches=4,
        site_anchor_cutoff_a=6.0,
        site_qc_notes=["WT ensemble local-site QC: ok=3, median anchor CA RMSD=0.820 A"],
    )
    assert any("Reference ligand atoms: 28" in ln for ln in lines)
    assert any("Local site QC:" in ln for ln in lines)
    assert any("Site QC:" in ln for ln in lines)
    assert any("RMSD of the top-scored pose to the crystal reference ligand" in ln for ln in lines)
    assert any("WT: median affinity=" in ln and "top-1 ligand RMSD=" in ln for ln in lines)
    assert any("Verdict" in ln for ln in lines)

    run_map_multi = {"WT": wt, "Mutant_001": mut, "Mutant_002": mut}
    cmp_rows = build_redocking_comparison_rows(run_map_multi)
    assert int(len(cmp_rows)) == 2
    assert {str(r.get("mutant_label", "")) for r in cmp_rows} == {"Mutant_001", "Mutant_002"}
    assert all(str(r.get("wt_label", "")) == "WT" for r in cmp_rows)
    assert all(np.isclose(float(r.get("wt_affinity_kcal", np.nan)), -7.9) for r in cmp_rows)
    assert all(np.isclose(float(r.get("wt_rmsd_a", np.nan)), 1.2) for r in cmp_rows)
    assert all(str(r.get("wt_affinity_label", "")) == "median" for r in cmp_rows)
    assert all(str(r.get("mut_rmsd_label", "")) == "median" for r in cmp_rows)

    with tempfile.TemporaryDirectory() as tdir:
        out_csv = os.path.join(tdir, "redock_multi_summary.csv")
        wrote = write_redocking_comparison_summary_csv(out_csv, cmp_rows)
        assert str(wrote) == str(out_csv)
        assert os.path.isfile(out_csv)
        with open(out_csv, "r", encoding="utf-8") as fh:
            txt = fh.read()
        assert "mutant_label" in txt
        assert "Mutant_001" in txt

        summary_payload = build_redocking_summary_payload(
            run_map=run_map_multi,
            nref=28,
            center=(1.0, 2.0, 3.0),
            size=(18.0, 18.0, 18.0),
            use_reference_ligand=True,
            use_modeler_ensemble=False,
            repeats=4,
            summary_csv_path=out_csv,
        )
        assert int(len(summary_payload.get("comparison_rows", []) or [])) == 2
        assert str(summary_payload.get("summary_csv", "")) == str(out_csv)
        assert "Summary CSV:" in str(summary_payload.get("summary_text", ""))


def test_vs_runtime_run_rank_and_summary(tmp_path):
    paths = vs_prepare_job_paths(
        workspace_dir=str(tmp_path),
        receptor_obj="Rec VS",
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        tag="20260303_010204",
    )
    assert paths["tag"] == "20260303_010204"
    assert os.path.isdir(paths["job_dir"])
    assert os.path.isdir(paths["lig_dir"])
    assert os.path.isdir(paths["pose_dir"])
    assert "virtual_screen_Rec_VS_20260303_010204" in paths["job_dir"]

    def _safe_name(name):
        return str(name).replace(" ", "_")

    def _write_sdf(_mol, lig_sdf, _name):
        with open(lig_sdf, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True, ""

    def _prepare_with_meeko(_mol, out_pdbqt):
        with open(out_pdbqt, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return True, "ok"

    def _fallback_prepare_ligand(lig_sdf, fallback_job, _target_ph):
        out_pdbqt = os.path.join(str(fallback_job), "ligand_prepared.pdbqt")
        os.makedirs(str(fallback_job), exist_ok=True)
        with open(out_pdbqt, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return out_pdbqt, {"notes": ["fallback"]}

    pose_aff = {}

    def _run_command(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        if "00001" in out_pose:
            pose_aff[out_pose] = -8.2
        else:
            pose_aff[out_pose] = -7.1
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        return True, "", "", 0

    def _parse_pose_scores(out_pose, _log):
        return [{"mode": 1, "affinity": float(pose_aff[out_pose])}]

    def _convert_pose_for_view(out_pose, _pose_dir, _stem):
        return str(out_pose)

    entries = [
        {"index": 1, "name": "Lig A", "identifier": "A", "mol": object()},
        {"index": 2, "name": "Lig B", "identifier": "B", "mol": object()},
    ]

    lig_dir = tmp_path / "ligands"
    pose_dir = tmp_path / "poses"
    lig_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    results, cancelled = run_virtual_screening_entries(
        entries=entries,
        smina_bin="smina",
        receptor_input="rec.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        exhaustiveness=8,
        timeout_lig=60,
        lig_dir=str(lig_dir),
        pose_dir=str(pose_dir),
        target_ph=7.4,
        safe_name_fn=_safe_name,
        write_sdf=_write_sdf,
        prepare_with_meeko=_prepare_with_meeko,
        fallback_prepare_ligand=_fallback_prepare_ligand,
        run_command=_run_command,
        parse_pose_scores=_parse_pose_scores,
        convert_pose_for_view=_convert_pose_for_view,
    )

    assert cancelled is False
    ranked = rank_virtual_screening_results(results)
    summary = summarize_virtual_screening_results(ranked)
    assert len(ranked) == 2
    assert int(ranked[0]["rank"]) == 1
    assert float(ranked[0]["affinity"]) < float(ranked[1]["affinity"])
    assert int(summary["n_ok"]) == 2
    txt_best = vs_status_text(summary["best"], summary["n_ok"], summary["n_total"])
    txt_none = vs_status_text(None, summary["n_ok"], summary["n_total"])
    assert "Best:" in txt_best and "2/2" in txt_best
    assert txt_none == "Status: done (2/2 scored)."

    lines = build_virtual_screening_summary_lines(
        receptor_obj="rec_obj",
        library_path="lib.sdf",
        loaded_count=2,
        skipped_count=0,
        n_ok=int(summary["n_ok"]),
        n_fail=int(summary["n_fail"]),
        output_dir=str(tmp_path),
        best=summary["best"],
        cancelled=False,
    )
    assert any("Virtual screening completed." in ln for ln in lines)
    assert any("Best hit:" in ln for ln in lines)


def test_vs_runtime_run_accepts_generator_input(tmp_path):
    def _safe_name(name):
        return str(name).replace(" ", "_")

    def _write_sdf(_mol, lig_sdf, _name):
        with open(lig_sdf, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True, ""

    def _prepare_with_meeko(_mol, out_pdbqt):
        with open(out_pdbqt, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return True, "ok"

    seen = []
    entries = ({"index": idx + 1, "name": f"Lig {idx}", "identifier": str(idx), "mol": object()} for idx in range(3))

    lig_dir = tmp_path / "ligands"
    pose_dir = tmp_path / "poses"
    lig_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    results, cancelled = run_virtual_screening_entries(
        entries=entries,
        smina_bin="smina",
        receptor_input="rec.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        exhaustiveness=8,
        timeout_lig=60,
        lig_dir=str(lig_dir),
        pose_dir=str(pose_dir),
        target_ph=7.4,
        safe_name_fn=_safe_name,
        write_sdf=_write_sdf,
        prepare_with_meeko=_prepare_with_meeko,
        fallback_prepare_ligand=lambda *_a, **_k: ("", {}),
        run_command=lambda *_a, **_k: (False, "", "fail", 1),
        parse_pose_scores=lambda *_a, **_k: [],
        convert_pose_for_view=lambda *_a, **_k: "",
        on_progress=lambda i, total, item: seen.append((i, total, item["name"])) or True,
    )

    assert cancelled is False
    assert len(results) == 3
    assert seen == [(0, 3, "Lig 0"), (1, 3, "Lig 1"), (2, 3, "Lig 2")]


def test_dock_runtime_run_post_docking_pipeline():
    calls = []

    def _rank(results, context, summary):
        _ = (context, summary)
        calls.append("rank")
        return {"results": [int(x) + 1 for x in list(results or [])], "context": {"ranked": True}}

    def _transform(results, context, summary):
        _ = (context, summary)
        calls.append("transform")
        return {"results": [int(x) * 2 for x in list(results or [])], "context": {"transformed": True}}

    def _summary(results, context, summary):
        _ = (context, summary)
        calls.append("summary")
        rr = list(results or [])
        return {"summary": {"count": int(len(rr)), "sum": int(sum(rr))}}

    def _apply_state(results, context, summary):
        _ = (results, context, summary)
        calls.append("apply_state")
        return {"context": {"applied": True}}

    def _log(results, context, summary):
        _ = (results, context, summary)
        calls.append("log")
        return {"summary": {"logged": True}}

    def _status(results, context, summary):
        _ = (results, context, summary)
        calls.append("status")
        return {}

    def _dialog(results, context, summary):
        _ = (results, context, summary)
        calls.append("dialog")
        return {"context": {"dialog_shown": True}}

    payload = run_post_docking_pipeline(
        results=[1, 2, 3],
        context={"seed": "x"},
        rank_fn=_rank,
        transform_fn=_transform,
        summary_fn=_summary,
        apply_state_fn=_apply_state,
        log_fn=_log,
        status_fn=_status,
        dialog_fn=_dialog,
    )

    assert calls == ["rank", "transform", "summary", "apply_state", "log", "status", "dialog"]
    assert payload["results"] == [4, 6, 8]
    assert payload["context"]["seed"] == "x"
    assert payload["context"]["ranked"] is True
    assert payload["context"]["transformed"] is True
    assert payload["context"]["applied"] is True
    assert payload["context"]["dialog_shown"] is True
    assert int(payload["summary"]["count"]) == 3
    assert int(payload["summary"]["sum"]) == 18
    assert payload["summary"]["logged"] is True


def test_dock_runtime_build_legacy_python_tool_command_prefers_shebang_executable(tmp_path):
    script = tmp_path / "prepare_flexreceptor4.py"
    script.write_text("#!/usr/bin/env pythonsh\nprint 'legacy'\n", encoding="utf-8")
    script.chmod(0o755)

    cmd = build_legacy_python_tool_command(
        str(script),
        ["-r", "rec.pdbqt"],
        which_fn=lambda _name: "",
        sys_executable="/usr/bin/python3",
    )
    assert cmd == [str(script), "-r", "rec.pdbqt"]


def test_dock_runtime_build_legacy_python_tool_command_uses_pythonsh_for_nonexec_legacy_script(tmp_path):
    script = tmp_path / "prepare_flexreceptor4.py"
    script.write_text("#!/usr/bin/env pythonsh\nprint 'legacy'\n", encoding="utf-8")
    script.chmod(0o644)

    cmd = build_legacy_python_tool_command(
        str(script),
        ["-r", "rec.pdbqt"],
        which_fn=lambda name: "/usr/bin/pythonsh" if str(name) == "pythonsh" else "",
        sys_executable="/usr/bin/python3",
    )
    assert cmd == ["/usr/bin/pythonsh", str(script), "-r", "rec.pdbqt"]


def test_dock_runtime_build_legacy_python_tool_command_falls_back_to_current_python_for_modern_script(tmp_path):
    script = tmp_path / "helper.py"
    script.write_text("#!/usr/bin/env python3\nprint('ok')\n", encoding="utf-8")
    script.chmod(0o644)

    cmd = build_legacy_python_tool_command(
        str(script),
        ["--flag"],
        which_fn=lambda _name: "",
        sys_executable="/usr/bin/python3",
    )
    assert cmd == ["/usr/bin/python3", str(script), "--flag"]


def test_dock_runtime_resolve_tool_path_checks_active_protliginteract_conda_prefix_bin(tmp_path, monkeypatch):
    env_bin = tmp_path / "protliginteract-full" / "bin"
    env_bin.mkdir(parents=True)
    smina = env_bin / "smina"
    smina.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    smina.chmod(0o755)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "protliginteract-full"))
    monkeypatch.setenv("PATH", "")

    found = resolve_tool_path(
        "smina",
        sys_prefix=str(tmp_path / "python_prefix"),
        sys_executable=str(tmp_path / "python_prefix" / "bin" / "python"),
    )

    assert found == str(smina)


def test_dock_runtime_resolve_tool_path_prefers_protliginteract_env_before_path(monkeypatch):
    captured = {}

    def _first_existing(candidates):
        captured["candidates"] = list(candidates)
        return str(candidates[0])

    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("PATH", "/usr/local/bin")
    monkeypatch.setattr("dock_runtime._first_existing", _first_existing)

    found = resolve_tool_path(
        "smina",
        sys_prefix="/tmp/non_protlig_python",
        sys_executable="/tmp/non_protlig_python/bin/python",
    )

    candidates = captured.get("candidates", [])
    assert found == "/opt/anaconda3/envs/protliginteract-full/bin/smina"
    assert candidates[0] == "/opt/anaconda3/envs/protliginteract-full/bin/smina"
    assert candidates[-1] == "smina"


def test_dock_runtime_build_meeko_flexres_spec_groups_residues():
    spec = build_meeko_flexres_spec(
        [
            {"chain": "A", "resi": "315", "resn": "THR"},
            {"chain": "A", "resi": "217", "resn": "TYR"},
            {"chain": "", "resi": "42", "resn": "SER"},
            {"chain": "B", "resi": "23", "resn": "LYS"},
            {"chain": "A", "resi": "221A", "resn": "GLY"},
            {"chain": "A", "resi": "315", "resn": "THR"},
        ]
    )
    assert spec == "A:315,A:217,:42,B:23,A:221A"


def test_dock_runtime_build_meeko_receptor_input_args():
    assert build_meeko_receptor_input_args("rec.pdb") == ["--read_pdb", "rec.pdb"]
    assert build_meeko_receptor_input_args("rec.ent") == ["--read_pdb", "rec.ent"]
    assert build_meeko_receptor_input_args("rec.pqr") == ["--read_pqr", "rec.pqr"]
    assert build_meeko_receptor_input_args("rec.cif") == ["-i", "rec.cif"]


def test_dock_runtime_build_smina_flexres_spec():
    assert build_smina_flexres_spec([{"chain": "A", "resi": "10"}, {"chain": "B", "resi": "23"}]) == "A:10,B:23"
    assert build_smina_flexres_spec([{"chain": "A", "resi": "221A"}]) == "A:221:A"


def test_redock_runtime_parse_flexres_spec_normalizes_insertion_code_formats():
    assert _parse_redocking_flexres_spec("A:221A") == [{"chain": "A", "resi": "221A"}]
    assert _parse_redocking_flexres_spec("A:221:A") == [{"chain": "A", "resi": "221A"}]


def test_dock_runtime_smina_supports_native_flexres():
    def _run_ok(argv, timeout=0, cwd=None):
        _ = (argv, timeout, cwd)
        return True, "Usage...\n  --flexres ARG\n  --flexdist ARG\n", "", 0

    def _run_no(argv, timeout=0, cwd=None):
        _ = (argv, timeout, cwd)
        return True, "Usage only", "", 0

    assert smina_supports_native_flexres("smina_bin", run_command=_run_ok) is True
    assert smina_supports_native_flexres("smina_bin", run_command=_run_no) is False


def test_redock_runtime_invoke_prepare_flexible_receptor_passes_optional_kwargs():
    captured = {}

    def _prepare(_rec, _job, _rows, prepared_pdb="", rec_info=None):
        captured["prepared_pdb"] = str(prepared_pdb or "")
        captured["rec_info"] = dict(rec_info or {})
        return "rigid.pdbqt", "flex.pdbqt", {"ok": True}

    rigid, flex, info = invoke_prepare_flexible_receptor(
        _prepare,
        "rec.pdbqt",
        "/tmp/job",
        [{"resi": "10"}],
        prepared_pdb="/tmp/job/receptor_prepared_stripped.pdb",
        rec_info={"prepared_pdb": "/tmp/job/receptor_prepared.pdb"},
    )
    assert rigid == "rigid.pdbqt"
    assert flex == "flex.pdbqt"
    assert info["ok"] is True
    assert captured["prepared_pdb"] == "/tmp/job/receptor_prepared_stripped.pdb"
    assert captured["rec_info"]["prepared_pdb"] == "/tmp/job/receptor_prepared.pdb"


def test_leadopt_runtime_run_rank_and_summary(tmp_path):
    def _safe_name(name):
        return str(name).replace(" ", "_")

    def _write_sdf(_mol, lig_sdf, _name):
        with open(lig_sdf, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True, ""

    def _prepare_with_meeko(_mol, out_pdbqt):
        with open(out_pdbqt, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return True, "ok"

    def _fallback_prepare_ligand(lig_sdf, fallback_job, _item, _target_ph):
        _ = lig_sdf
        out_pdbqt = os.path.join(str(fallback_job), "ligand_prepared.pdbqt")
        os.makedirs(str(fallback_job), exist_ok=True)
        with open(out_pdbqt, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return out_pdbqt, {"notes": ["fallback"]}

    pose_aff = {}

    def _run_command(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        pose_aff[out_pose] = -8.4 if "00001" in out_pose else -7.2
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        return True, "", "", 0

    def _parse_pose_scores(out_pose, _log):
        return [{"mode": 1, "affinity": float(pose_aff[out_pose])}]

    def _convert_pose_for_view(out_pose, _pose_dir, _stem):
        return str(out_pose)

    def _mpo_fn(affinity, admet, qed, sim):
        return mpo_score(affinity, admet, qed, sim)

    entries = [
        {
            "name": "Lead A",
            "source": "pool",
            "smiles": "CCO",
            "admet": 0.8,
            "qed": 0.6,
            "sim_seed": 0.7,
            "mw": 250.0,
            "logp": 2.1,
            "hba": 3,
            "hbd": 1,
            "tpsa": 42.0,
            "rotb": 2,
            "pains": False,
            "filter_keep": True,
            "filter_reason": "pass",
            "mol": object(),
        },
        {
            "name": "Lead B",
            "source": "brics",
            "smiles": "CCN",
            "admet": 0.6,
            "qed": 0.5,
            "sim_seed": 0.5,
            "mw": 280.0,
            "logp": 2.8,
            "hba": 4,
            "hbd": 1,
            "tpsa": 50.0,
            "rotb": 3,
            "pains": False,
            "filter_keep": True,
            "filter_reason": "pass",
            "mol": object(),
        },
    ]

    lig_dir = tmp_path / "ligands"
    pose_dir = tmp_path / "poses"
    lig_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    results, cancelled = run_leadopt_entries(
        entries=entries,
        smina_bin="smina",
        receptor_input="rec.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        exhaustiveness=8,
        timeout_lig=60,
        work_dir=str(tmp_path),
        lig_dir=str(lig_dir),
        pose_dir=str(pose_dir),
        target_ph=7.4,
        safe_name_fn=_safe_name,
        write_sdf=_write_sdf,
        prepare_with_meeko=_prepare_with_meeko,
        fallback_prepare_ligand=_fallback_prepare_ligand,
        run_command=_run_command,
        parse_pose_scores=_parse_pose_scores,
        convert_pose_for_view=_convert_pose_for_view,
        mpo_fn=_mpo_fn,
    )
    assert cancelled is False
    ranked = rank_leadopt_results(results)
    best = find_best_leadopt_result(ranked)
    assert len(ranked) == 2
    assert int(ranked[0]["rank"]) == 1
    assert best is not None
    assert float(best["affinity"]) < -8.0

    lines = build_leadopt_summary_lines(
        seed_name="SeedX",
        n_analogs=10,
        n_total=2,
        n_ok=2,
        n_fail=0,
        n_ref_ok=0,
        n_ref_attempted=0,
        ref_method="",
        filter_mode="strict",
        best=best,
        refine_note="",
        cancelled=False,
    )
    assert any("Lead optimization completed." in ln for ln in lines)
    assert any("Top lead:" in ln for ln in lines)


def test_leadopt_runtime_run_accepts_generator_input(tmp_path):
    def _safe_name(name):
        return str(name).replace(" ", "_")

    def _write_sdf(_mol, lig_sdf, _name):
        with open(lig_sdf, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True, ""

    def _prepare_with_meeko(_mol, out_pdbqt):
        with open(out_pdbqt, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return True, "ok"

    def _mpo_fn(affinity, admet, qed, sim):
        return mpo_score(affinity, admet, qed, sim)

    seen = []
    entries = (
        {
            "name": f"Lead {idx}",
            "source": "pool",
            "smiles": "CCO",
            "admet": 0.8,
            "qed": 0.6,
            "sim_seed": 0.7,
            "mw": 250.0,
            "logp": 2.1,
            "hba": 3,
            "hbd": 1,
            "tpsa": 42.0,
            "rotb": 2,
            "pains": False,
            "filter_keep": True,
            "filter_reason": "pass",
            "mol": object(),
        }
        for idx in range(2)
    )

    lig_dir = tmp_path / "ligands"
    pose_dir = tmp_path / "poses"
    lig_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    results, cancelled = run_leadopt_entries(
        entries=entries,
        smina_bin="smina",
        receptor_input="rec.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        exhaustiveness=8,
        timeout_lig=60,
        work_dir=str(tmp_path),
        lig_dir=str(lig_dir),
        pose_dir=str(pose_dir),
        target_ph=7.4,
        safe_name_fn=_safe_name,
        write_sdf=_write_sdf,
        prepare_with_meeko=_prepare_with_meeko,
        fallback_prepare_ligand=lambda *_a, **_k: ("", {}),
        run_command=lambda *_a, **_k: (False, "", "fail", 1),
        parse_pose_scores=lambda *_a, **_k: [],
        convert_pose_for_view=lambda *_a, **_k: "",
        mpo_fn=_mpo_fn,
        on_progress=lambda i, total, item: seen.append((i, total, item["name"])) or True,
    )

    assert cancelled is False
    assert len(results) == 2
    assert seen == [(0, 2, "Lead 0"), (1, 2, "Lead 1")]


def test_package_import_is_lazy_and_does_not_require_pymol(monkeypatch):
    monkeypatch.syspath_prepend(str(os.path.dirname(os.path.dirname(__file__))))
    sys.modules.pop("ProtLigInteract", None)
    pkg = importlib.import_module("ProtLigInteract")
    assert getattr(pkg, "__version__", None) == "0.2.0a15"
    assert callable(getattr(pkg, "__init_plugin__", None))
