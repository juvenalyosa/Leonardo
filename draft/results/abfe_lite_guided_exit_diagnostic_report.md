# ABFE-lite Guided-Exit Diagnostic Report

This reviewer-facing report describes trajectory and state-gating diagnostics only.
Affinity-like estimator fields present in older raw run reports are intentionally excluded because this short guided-exit run is not a production absolute binding free-energy calculation.

## System and Sampling

- Source run: `draft/results/abfe_lite_3ptb_ben_5ns_rerun_20260702`
- Simulation length: `5.0` ns
- Samples used for diagnostics: `2500`
- Sequential analysis segments: `12`
- Bound/unbound hold-window counts: `340` / `69`
- Total/unbound effective sample diagnostic: `185.514` / `25.127`

## Exit-State Diagnostics

- Maximum exit coordinate: `37.844` Å
- Minimum protein-ligand distance at maximum exit: `29.005` Å
- Contact count at maximum exit: `0.0` contacts
- Far-hold frames: `128`

## Reporting Scope

- `diagnostic_only`: true
- `binding_estimate_allowed`: false
- No binding affinity, Kd, Ki or IC50 is reported for this manuscript diagnostic.
- The corresponding manuscript panel should be interpreted as a guided-exit trajectory and plotting check.
