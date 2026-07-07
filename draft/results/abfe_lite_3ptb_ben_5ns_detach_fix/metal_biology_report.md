# 3PTB/BEN Metal-Preserving ABFE Check

Overall pass: yes

## System

- Raw PDB: `3PTB.pdb`
- Complex PDB used for ABFE: `draft/results/abfe_lite_3ptb_ben_5ns_detach_fix/3PTB_BEN_CA_complex.pdb`
- Metal ions retained: `[{'resn': 'CA', 'chain': 'A', 'resi': '480', 'atoms': 1}]`
- Min metal-ligand distance: `18.839 A`

## Energetics

- Bound mean total interaction: `-1456.855 kJ/mol`
- Unbound mean total interaction: `-6.889 kJ/mol`
- Bound minus unbound: `-1449.966 kJ/mol`
- Bound mean Coulomb term: `-1487.659 kJ/mol`
- Bound mean metal contribution: `0.000 kJ/mol`
- Estimator window mode: `hold_only`
- Raw state counts: `bound=406`, `unbound=769`
- Hold-window counts: `bound=339`, `unbound=67`
- Binding estimate status: `estimable_hold_windows`
- Binding estimate note: `Bound and unbound hold windows are populated; selected estimator may be used if ESS checks pass.`
- Max exit coordinate: `36.857 A`
- Min distance at max exit: `26.733 A`
- Contact count at max exit: `0.0`
- Far-hold frames: `133`
- Selected estimator: `bias_reweighted_hold_windows` | confidence=`high`
- Selected note: `stable hold windows are present for both states and reweighting ESS is strong`
- Selected ΔG: `-5.282 kJ/mol`
- Selected Kd(app): `1.826e-02 M`
- Bias-reweighted ΔG: `-5.282 kJ/mol`
- Bias-reweighted Kd(app): `1.826e-02 M`
- Bias-reweighted ESS(total): `181.67`
- Population diagnostic ΔG: `-4.044 kJ/mol`
- Jeffreys-smoothed diagnostic: `-4.029 kJ/mol` (pseudo-count only; not reported as BFE/Kd when one state is missing)

## Biology

- Asp189 amidinium occupancy: `0.970`
- Top bound contacts:
  - ASP A 189: occupancy=0.998
  - GLY A 219: occupancy=0.995
  - SER A 190: occupancy=0.988
  - ASP A 194: occupancy=0.980
  - CYS A 220: occupancy=0.958
  - GLY A 216: occupancy=0.943
  - TRP A 215: occupancy=0.938
  - GLN A 192: occupancy=0.929

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
