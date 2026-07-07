load 3PTB.pdb, complex
remove solvent
remove not (polymer.protein or resn BEN)
hide everything
bg_color white
set orthoscopic, on
set depth_cue, 0
set ray_shadows, 0
set ambient, 0.62
set spec_reflect, 0
set antialias, 2
set cartoon_transparency, 0.30
set dash_gap, 0.20
set dash_radius, 0.035
set_color leonardo_ligand, [0.94, 0.43, 0.12]
set_color contact_hydrophobic, [0.00, 0.45, 0.70]
set_color contact_polar, [0.00, 0.62, 0.45]
set_color contact_salt, [0.84, 0.34, 0.00]
set_color contact_vdw, [0.45, 0.45, 0.45]
show cartoon, polymer.protein
color gray85, polymer.protein
select contact_residues, polymer.protein and chain A and resi 189+190+192+195+213+215+216+219+220+226
show sticks, contact_residues
set stick_radius, 0.13, contact_residues
color gray45, contact_residues and elem C
color red, contact_residues and elem O
color blue, contact_residues and elem N
show sticks, resn BEN
set stick_radius, 0.19, resn BEN
color leonardo_ligand, resn BEN and elem C
color blue, resn BEN and elem N
distance contact_01, (polymer.protein and chain A and resi 219 and resn GLY and name O), (resn BEN and resi 1 and name N1)
hide labels, contact_01
color contact_polar, contact_01
distance contact_02, (polymer.protein and chain A and resi 189 and resn ASP and name OD2), (resn BEN and resi 1 and name N2)
hide labels, contact_02
color contact_polar, contact_02
distance contact_03, (polymer.protein and chain A and resi 189 and resn ASP and name OD1), (resn BEN and resi 1 and name N1)
hide labels, contact_03
color contact_salt, contact_03
distance contact_04, (polymer.protein and chain A and resi 190 and resn SER and name OG), (resn BEN and resi 1 and name N2)
hide labels, contact_04
color contact_polar, contact_04
distance contact_05, (polymer.protein and chain A and resi 190 and resn SER and name O), (resn BEN and resi 1 and name C)
hide labels, contact_05
color contact_vdw, contact_05
distance contact_06, (polymer.protein and chain A and resi 226 and resn GLY and name CA), (resn BEN and resi 1 and name N2)
hide labels, contact_06
color contact_vdw, contact_06
distance contact_07, (polymer.protein and chain A and resi 189 and resn ASP and name CG), (resn BEN and resi 1 and name N2)
hide labels, contact_07
color contact_vdw, contact_07
distance contact_08, (polymer.protein and chain A and resi 195 and resn SER and name OG), (resn BEN and resi 1 and name C5)
hide labels, contact_08
color contact_vdw, contact_08
distance contact_09, (polymer.protein and chain A and resi 219 and resn GLY and name O), (resn BEN and resi 1 and name C2)
hide labels, contact_09
color contact_vdw, contact_09
distance contact_10, (polymer.protein and chain A and resi 192 and resn GLN and name OE1), (resn BEN and resi 1 and name C3)
hide labels, contact_10
color contact_vdw, contact_10
distance contact_11, (polymer.protein and chain A and resi 216 and resn GLY and name N), (resn BEN and resi 1 and name C2)
hide labels, contact_11
color contact_vdw, contact_11
distance contact_12, (polymer.protein and chain A and resi 213 and resn VAL and name CG1), (resn BEN and resi 1 and name C5)
hide labels, contact_12
color contact_hydrophobic, contact_12
distance contact_13, (polymer.protein and chain A and resi 220 and resn CYS and name CA), (resn BEN and resi 1 and name N1)
hide labels, contact_13
color contact_vdw, contact_13
distance contact_14, (polymer.protein and chain A and resi 215 and resn TRP and name O), (resn BEN and resi 1 and name N2)
hide labels, contact_14
color contact_polar, contact_14
distance contact_15, (polymer.protein and chain A and resi 215 and resn TRP and name C), (resn BEN and resi 1 and name C1)
hide labels, contact_15
color contact_vdw, contact_15
orient resn BEN or contact_residues
turn x, 18
turn y, -22
turn z, -8
zoom resn BEN or contact_residues, 2.8
png draft/figures/figure_4_3ptb_ben_pymol_binding_site.png, width=1500, height=1150, dpi=300, ray=1
