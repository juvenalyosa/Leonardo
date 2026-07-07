# 3PTB/BEN guided-exit diagnostic check

Diagnostic state-gating status: hold-window diagnostic incomplete

## System

- Raw PDB: `3PTB.pdb`
- Complex PDB used for ABFE: `draft/results/abfe_lite_3ptb_ben_smoke/3PTB_BEN_CA_complex.pdb`
- Metal ions retained: `[{'resn': 'CA', 'chain': 'A', 'resi': '480', 'atoms': 1}]`
- Min metal-ligand distance: `22.945 A`

## Energetics

- Bound mean total interaction: `-1611.342 kJ/mol`
- Unbound mean total interaction: `nan kJ/mol`
- Bound minus unbound: `nan kJ/mol`
- Bound mean Coulomb term: `-1641.126 kJ/mol`
- Bound mean metal contribution: `0.000 kJ/mol`
- Estimator window mode: `state_fallback`
- Raw state counts: `bound=2`, `unbound=0`
- Hold-window counts: `bound=0`, `unbound=0`
- Diagnostic status: `hold windows populated for state-gating diagnostics`
- Binding estimate note: `Diagnostic-only guided-exit run; hold windows are not converted to affinity.`
- Max exit coordinate: `4.676 A`
- Min distance at max exit: `2.494 A`
- Contact count at max exit: `25.0`
- Far-hold frames: `0`
- Affinity-like estimates: `suppressed; this reviewer-facing artifact reports state-gating diagnostics only.`
- Bias-reweighted ESS(total): `nan`

## Biology

- Asp189 amidinium occupancy: `1.000`
- Top bound contacts:
  - ASP A 189: occupancy=1.000
  - SER A 190: occupancy=1.000
  - CYS A 191: occupancy=1.000
  - ASP A 194: occupancy=1.000
  - SER A 195: occupancy=1.000
  - SER A 214: occupancy=1.000
  - TRP A 215: occupancy=1.000
  - GLY A 219: occupancy=1.000

## Checks

- Physics:
  - parameterized_charge_close_to_plus_one: pass
  - ligand_ring_geometry_preserved: pass
  - steering_reaches_unbound_region: fail
  - bound_state_more_favorable_than_unbound: fail
  - bound_state_coulomb_is_attractive: pass
  - metal_does_not_dominate_binding_energy: pass
- Biology:
  - asp189_amidinium_contact_persistent: pass
  - asp189_ranked_among_top_contacts: pass
  - known_s1_pocket_residues_enriched: pass
