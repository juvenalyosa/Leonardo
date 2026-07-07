# 3PTB/BEN Metal-Preserving ABFE Check

Overall pass: yes

## System

- Raw PDB: `3PTB.pdb`
- Complex PDB used for ABFE: `draft/results/abfe_lite_3ptb_ben_5ns_rerun_20260702/3PTB_BEN_CA_complex.pdb`
- Metal ions retained: `[{'resn': 'CA', 'chain': 'A', 'resi': '480', 'atoms': 1}]`
- Min metal-ligand distance: `19.080 A`

## Energetics

- Bound mean total interaction: `-1469.078 kJ/mol`
- Unbound mean total interaction: `-7.202 kJ/mol`
- Bound minus unbound: `-1461.876 kJ/mol`
- Bound mean Coulomb term: `-1501.057 kJ/mol`
- Bound mean metal contribution: `0.000 kJ/mol`
- Estimator window mode: `hold_only`
- Raw state counts: `bound=400`, `unbound=755`
- Hold-window counts: `bound=340`, `unbound=69`
- Binding estimate status: `estimable_hold_windows`
- Binding estimate note: `Bound and unbound hold windows are populated; selected estimator may be used if ESS checks pass.`
- Max exit coordinate: `37.844 A`
- Min distance at max exit: `29.005 A`
- Contact count at max exit: `0.0`
- Far-hold frames: `128`
- Selected estimator: `bias_reweighted_hold_windows` | confidence=`high`
- Selected note: `stable hold windows are present for both states and reweighting ESS is strong`
- Estimate kind: `standard_state_apparent` | reliability=`apparent`
- Reliability note: `Bound/unbound hold windows are populated; report as an apparent short-run estimate, not production ABFE.`
- State-ratio ΔG: `-5.147 kJ/mol`
- Standard-state correction: `-3.980 kJ/mol` (Ceff=`2.028e-01 M`)
- Selected ΔG(app, standard): `-9.127 kJ/mol`
- Selected Kd(app): `2.576e-02 M`
- Bias-reweighted ΔG(state): `-5.147 kJ/mol`
- Bias-reweighted ΔG(standard): `-9.127 kJ/mol`
- Bias-reweighted Kd(app): `2.576e-02 M`
- Bias-reweighted ESS(total): `185.51`
- Population diagnostic ΔG: `-3.978 kJ/mol`
- Jeffreys-smoothed diagnostic: `-3.964 kJ/mol` (pseudo-count only; not reported as BFE/Kd when one state is missing)

## Biology

- Asp189 amidinium occupancy: `1.000`
- Top bound contacts:
  - ASP A 189: occupancy=1.000
  - SER A 190: occupancy=1.000
  - ASP A 194: occupancy=1.000
  - GLY A 219: occupancy=1.000
  - CYS A 220: occupancy=0.970
  - TRP A 215: occupancy=0.950
  - GLY A 216: occupancy=0.943
  - CYS A 191: occupancy=0.935

## Checks

- Physics:
  - parameterized_charge_close_to_plus_one: pass
  - ligand_ring_geometry_preserved: pass
  - steering_reaches_unbound_region: pass
  - bound_state_more_favorable_than_unbound: pass
  - bound_state_coulomb_is_attractive: pass
  - metal_does_not_dominate_binding_energy: pass
- Biology:
  - asp189_amidinium_contact_persistent: pass
  - asp189_ranked_among_top_contacts: pass
  - known_s1_pocket_residues_enriched: pass
