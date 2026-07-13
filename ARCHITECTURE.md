# Leonardo Architecture

Leonardo is the user-facing suite name. Legacy internal module/file names may still reference `ProtLigInteract` for compatibility with the existing PyMOL plugin packaging.

Runtime implementations are canonical under `src/leonardo/`. Root-level
modules with matching names are import aliases kept for legacy PyMOL packages
and external scripts.

This plugin is transitioning from a monolithic UI/controller file to a layered structure:

- `code_v2.py`
  - Qt/PyMOL UI wiring.
  - User interactions, dialogs, object loading, scene updates.
  - Orchestrates workflows by calling runtime modules.

- `background_tasks.py`
  - Generic QRunnable wrapper for background execution.
  - Centralized success/error/finished signaling for UI-safe callbacks.

- `dock_runtime.py`
  - External tool resolution and subprocess execution wrappers.
  - Shared post-docking callback pipeline executor (`run_post_docking_pipeline`).
  - No UI dependencies.

- `smina_runtime.py`
  - Pure repeat-engine for smina docking jobs.
  - Deterministic loop, output collection, failure-tail formatting.
  - No Qt/PyMOL dependencies.

- `leadopt_runtime.py`
  - Lead-optimization scoring/filter math.
  - ADMET score, limit relaxation, filter checks, MPO scoring.
  - Lead-optimization docking entry runtime loop, ranking, and summary formatting helpers.
  - No Qt/PyMOL dependencies.

- `abfe_runtime.py`
  - ABFE-lite calibration models and deploy-model selection logic.
  - Pocket-descriptor constants and target-hint helpers used by ABFE flows.
  - ABFE scene-restore planning helper for keep-scene/restore-enabled visibility decisions and zoom target selection.
  - ABFE steering lambda-schedule builder, openmm context-setup plan helper, openmm staged-prep plan helper, per-sample MD payload helper, live-update decision helper, MD progress-dialog message/plan helpers, progress step/finalize decision helpers, UI process-events pacing helper, live-view preflight/update-plan helpers, live-plot setup/state/payload/apply-plan helpers, default-payload helpers for scene-restore/live-plot updates, scene-action apply-operations helper, object-inventory/live-view defaults helpers, trajectory-state picking helper, final-trajectory build plan helper, final-trajectory post-build UI plan helper, final-scene restore execution payload helper, local-MD cleanup/teardown plan helpers, post-sampling metrics helper, result-payload helper, free-energy numeric payload helper, free-energy plot payload helper, free-energy report/debug text formatters, post-interaction follow-up plan/runtime-context helpers, post-interaction dialog payload helper, and post-interaction occupancy/report payload helpers for local-MD setup/update extraction.
  - No Qt/PyMOL dependencies.

- `alphafold_runtime.py`
  - AlphaFold/ColabFold model-discovery and plot-discovery helpers.
  - JSON score extraction (pLDDT/PAE) used by AlphaFold plotting.
  - No Qt/PyMOL dependencies.

- `psi4_runtime.py`
  - Psi4 cube parsing/writing and Mulliken-charge ESP cube generation helpers.
  - No Qt/PyMOL dependencies.

- `scientific_readiness_runtime.py`
  - Suite-wide scientific workflow catalog for the implemented tools.
  - Maps workflows to the relevant external scientific-skill guidance and local dependency checks.
  - Produces readiness reports for setup validation and CI.
  - No Qt/PyMOL dependencies.

- `redock_runtime.py`
  - Redocking repeat execution helpers.
  - Ligand/receptor/flex preparation validation helpers for redocking orchestration.
  - Ligand charge-policy and SMILES-hint resolution helpers for redocking ligand prep orchestration.
  - Ligand-prep orchestration helper combining charge policy, SMILES-hint resolution, and ligand prep validation.
  - UI-agnostic base/redock-input payload helper for reference/WT/mutant validation and reference-grid payload assembly, plus base runtime binding helper for UI callback/default wiring.
  - UI-agnostic ligand-override resolver for reference-ligand export vs prepared-ligand selection, plus reference-ligand export plan/result/response payload helpers and ligand-override runtime binding helper for UI callback/default wiring.
  - UI-agnostic reference-molecule/flexible-selection helper for RDKit reference setup and flexible-residue validation, plus reference-mol export/load plan/result payload helpers and reference/flexible runtime binding helper for UI callback/default wiring.
  - Single-run preflight/context helper for tool-check, ligand-source resolution, and job-dir/tag setup.
  - Receptor/flex orchestration helper for receptor prep validation, reference-residue stripping, and flexible-prep resolution, plus standalone structure-file residue-strip helper.
  - Smina command/repeat-plan builder for redocking runs.
  - Repeat-run pipeline helper for callback wiring, repeat execution invocation, and failure payload mapping.
  - High-level single-run pipeline orchestrator composing plan build, repeat run, pose outputs, and run-outcome assembly.
  - Top-level single-system orchestrator entrypoint composing preflight, receptor/flex, ligand-prep, RMSD callback wiring, and single-run pipeline, plus single-system runtime binding helper for UI callback/default wiring.
  - RMSD-value computation and per-repeat row builder helpers for injected repeat callbacks, plus RMSD-molecule prep plan/result helpers, RMSD-molecule config-normalization helper, RMSD-molecule runtime resolver helper, RDKit RMSD-molecule runtime resolver helper, RDKit RMSD-molecule config-aware runtime resolver helper, candidate atom-map strategy helper, candidate-map config-normalization helper, directional-match adapter helper, candidate-map callback-bundle helper, RDKit callback-preset helper, RDKit config-to-callback bridge helper, candidate-map runtime resolver helper, RDKit candidate-map runtime resolver helper, best-RMSD config-normalization helper, best-RMSD runtime resolver helper, bond-order/MCS payload helpers, atom-map dedup helper, and no-fit RMSD helper.
  - Pose-object naming, pose-view resolution, and row pose-object attachment helpers.
  - Callback-based safe wrapper for pose loading and warning-text formatting.
  - Shared pose-output assembly helper for single-system redocking (view/load/row attachment + warning text).
  - Member-run normalization and ordered run-map aggregation helpers.
  - Runtime systems-loop helper for multi-system redocking execution and per-label member grouping, plus runtime-systems binding helper for UI callback/default and exception-shaping wiring.
  - Runtime systems-context helper for MODELLER ensemble expansion and alignment/QC decision flow before execution, plus runtime binding helper for UI callback/default and exception-shaping wiring.
  - HETATM preflight runtime helper for prompt-loop orchestration, cache propagation, and per-system selection summary lines, plus runtime binding helper for UI prompt-callback/default wiring.
  - Top-level redocking workflow orchestrator helper that composes prep/hetatm/run/aggregate steps via injected callbacks and returns post-pipeline context payloads.
  - Run-map payload and post-pipeline helper set for aggregation payloads, row extraction, summary text, and shared context shaping, plus run-map/label-aggregation and post-transform/post-summary/post-apply-state/post-status/post-dialog runtime binding helpers for UI callback/default and exception-shaping wiring.
  - Member-specific best-pose selector helper shared by trajectory-frame overlays and runtime-driven preview logic.
  - Trajectory state-to-row mapping helper for redocking combo-trajectory assembly (state assignment, member fallback, full-frame safety fill).
  - Table-row shaping helper for redocking results (per-system rank, display label, and trajectory-state assignment).
  - Trajectory-visibility/state helpers for label visibility policy, frame-state resolution, visibility action planning, trajectory-visibility runtime binding orchestration, runtime-input hide-action planning, row-derived object visibility actions, and row-object hide-action runtime binding orchestration.
  - Trajectory frame-payload helper plus trajectory-frame runtime binding helper for callback-driven state/overlay resolution, object lifecycle orchestration, and focus-visibility actions during frame selection.
  - Selected-pose/reference/interactions helper set for table-row validation, pose availability checks, selected-pose/selected-interaction/reference-context runtime binding orchestration, selected-pose/selected-interaction/reference-grid warning-message selection, reference-grid info-message shaping, receptor-view resolution, focus/reference styling selection, selected-pose scene/camera execution-operation planning, complex-selection preflight, post-complex ligand-field payload shaping, and selected-interaction post-payload UI-apply plan shaping.
  - Trajectory-view planning helper for runtime-system grouping and deterministic object-name payload generation, plus runtime-view prep binding helper for UI callback/default wiring.
  - Trajectory-view object helper pair for normalized view-entry payloads and object-name collection during cleanup.
  - Trajectory member-load helper for callback-driven state-map/n-state bookkeeping during ensemble trajectory assembly.
  - Combo-trajectory object-name helper and callback-driven combo-state assembly helper for best-pose overlay trajectories.
  - Per-member best-row ranking, trajectory-state resolution, and repeat-stats CSV export helpers.
  - Shared repeat-statistics summary helper for single-system and ensemble redocking outputs.
  - Shared repeat-output assembly helper (stats + optional CSV) used by single-system and ensemble redocking flows.
  - Shared run-summary debug-line formatter for redocking execution logs.
  - Shared single-run result payload builder used by redocking orchestration.
  - Shared single-run outcome composer that unifies debug-line and payload assembly for redocking orchestration.
  - Best-pose/repeat selection.
  - WT vs mutant comparison/classification and redocking summary text builders.
  - No Qt/PyMOL dependencies.

- `vs_runtime.py`
  - Virtual-screening entry runtime loop.
  - Result ranking and summary text builders.
  - No Qt/PyMOL dependencies.

## Design Rules

1. Keep UI-only code in `code_v2.py`.
2. Move pure computations and subprocess loops to runtime modules.
3. Runtime modules must not import Qt/PyMOL.
4. Use explicit data contracts (dict fields) between UI and runtime modules.
5. Keep side effects isolated:
   - Runtime modules: files/processes only.
   - UI layer: dialogs, widgets, PyMOL scene updates.

## Shared Workflow Pattern

Main docking-driven workflows (virtual screening, lead optimization, redocking) now use the same orchestration shape:

1. Preflight/UI validation in `code_v2.py`.
2. Runtime entry execution loop (VS/Leadopt/Redock runtime helpers).
3. Shared post-processing callback pipeline:
   - rank/transform
   - summary build
   - state apply
   - logging/status updates
   - dialog presentation

The callback pipeline core lives in `dock_runtime.run_post_docking_pipeline`, while workflow-specific callbacks remain in `code_v2.py`.

## Next Modularization Targets

- Redocking comparison/report export helpers extracted from UI handlers.
- Modeler/AlphaFold/Psi4 long workflows split into domain runtime/service modules.
- ABFE-lite post-processing/report generation split from UI handlers.
