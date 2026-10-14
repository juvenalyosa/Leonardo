# Interaction views

`leonardo.interaction_view` draws the interactions found by Leonardo. It does not change what is detected.

## Protein–ligand, 3D (`build_scene`)

- White background and an orthoscopic camera that looks from the solvent into the pocket. The view is
  chosen to show the ligand face-on with few protein atoms in front, and the frame is fitted to the ligand
  and the residues that make specific interactions.
- Context is a thin translucent backbone tube. The ligand is shown once, as balls and sticks with amber
  carbons; non-polar hydrogens are hidden.
- **Residues:**
  - only the residues that interact are drawn, as sticks;
  - residues with only van der Waals or hydrophobic contacts are thinner and lighter;
  - backbone atoms are shown only when a backbone atom interacts.
- **Labels:** one label per residue (`Asp189`), placed in the camera plane away from the ligand and from
  other labels. A leader line is added when a label had to move away from its residue.
- **Dashes:** one CGO object per interaction type (`iv_H-bond`, `iv_Salt_bridge`, ...), coloured by
  type, with distance labels for polar and π interactions.
- Waters are shown only when they bridge.
- Used by `analyze_interactions` and by **Calculate** in the Leonardo window. In the window it can be
  turned off with `settings["publication_view"] = False`.

## Protein–ligand, 2D (`draw_2d_map`)

- The ligand is drawn by RDKit with bond orders from the PDB Chemical Component Dictionary when the residue
  name matches. Otherwise bond orders are perceived from the 3D geometry, and formal charges come from
  Leonardo's pKa model at pH 7.4.
- Residues are drawn as pills coloured by class (acidic, basic, polar, hydrophobic, glycine, water, metal).
  They are placed around the atoms they touch without overlapping each other or the molecule.
- Dashes are coloured by type and carry the distance in Å (donor···acceptor; ring centroids for π
  interactions).
- In the window, clicking a residue focuses its interaction in 3D.
- Used by `show_2d_interaction_map`.

## Interfaces (`build_interface_scene`)

- The two partners are shown as cartoons in two soft colours. Interface residues are drawn as sticks with
  carbons in the darker shade of their side.
- **Camera:** side-on, with partner A on the left. A DNA or RNA duplex is viewed from the side, never
  along its axis.
- Labels are placed on the hotspot residues.
- **Not drawn by default:** water bridges and hydrophobic pairs, which would clutter the figure. They are
  still reported in the results.
- Used by `analyze_interface`; `figure_path` also saves a ray-traced figure with a legend.

## Figures

`export_interaction_figure` saves the 3D view (ray-traced, with a title and a legend) and the 2D diagram.
