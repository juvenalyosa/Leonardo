# Ramachandran plot

Validates the backbone of a structure, a model or a trajectory. The backbone φ/ψ of each residue is compared
with the favoured and allowed regions of high-resolution PDB chains, in the six MolProbity classes: general,
glycine, trans-proline, cis-proline, pre-proline and Ile/Val.

- **Window:** **Ramachandran Plot...** in the main window. Choose an object, optional chains and *All states*
  for a trajectory or NMR ensemble. Click an outlier to show it in PyMOL. Save the figure as a PNG.
- **Assistant:** `ramachandran_plot`. For example, *"Make the Ramachandran plot of my model and tell me which
  residues are outliers."*
- **Automatic checks:** `build_homology_model`, `model_mutant`, `repair_structure_gaps` and `prepare_protein`
  report the Ramachandran statistics of their result, and warn when the backbone is poor.

## Regions

| Region | Contour | Goal (MolProbity) |
|---|---|---|
| favoured | 98% of reference residues | > 98% of residues |
| allowed | 99.95% (general) / 99.8% (other classes) | |
| outlier | outside allowed | < 0.05% of residues |

Verdicts:

- **excellent:** both goals are met.
- **good:** ≥ 95% favoured and ≤ 0.5% outliers.
- **acceptable:** ≥ 90% favoured and ≤ 2% outliers.
- **poor:** anything below that.

## Reference data

`src/leonardo/rama_reference.npz` holds φ/ψ histograms in 2° bins. `scripts/build_rama_reference.py` builds
them, following the Top8000 recipe on a smaller scale:

- **Chains:** 2,914 PDB chains, one per 30%-identity cluster.
- **Entries:** X-ray, ≤ 1.5 Å, R-free ≤ 0.21.
- **Residues:** full occupancy, no alternate locations, backbone B < 30 Å², no chain break next to them.

That is about 475,000 residues in total. The data come from the PDB (CC0).

Leonardo smooths each histogram with a Gaussian of two widths: a narrow one for the favoured core and a wide
one for the sparse allowed tails. The wide width was calibrated with `scripts/calibrate_rama_reference.py`
against the Ramachandran outlier percentages of the wwPDB validation reports, which use MolProbity, on X-ray
entries at 1.8–3.0 Å that are not in the reference set:

| Set | Entries | Mean error | Mean absolute error | Correlation |
|---|---|---|---|---|
| calibration | 80 | +0.03 points | 0.19 points | r = 0.97 |
| held out | 80 | −0.04 points | 0.21 points | r = 0.98 |

Leonardo's outlier percentages therefore follow MolProbity's closely. They are not identical: MolProbity's own
contours come from about 8,000 chains with adaptive smoothing.

The wwPDB reports give no favoured percentage, so that one is not calibrated. On 203 entries at 1.8–3.0 Å,
the favoured share averages 93.6% (median 95.2%). That is probably slightly stricter than MolProbity, so read
favoured values just under 98% with that in mind.
