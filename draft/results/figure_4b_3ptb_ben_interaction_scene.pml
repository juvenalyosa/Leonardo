load /Users/juvenalyosa/ProtLigInteract/Leonardo_clean/3PTB.pdb, complex
remove solvent
remove not (polymer.protein or resn BEN)
hide everything
bg_color white
set orthoscopic, on
set depth_cue, 0
set antialias, 2
set ray_shadows, 0
set ambient, 0.56
set direct, 0.65
set spec_reflect, 0.18
set shininess, 35
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set cartoon_transparency, 0.42
set dash_gap, 0.15
set dash_length, 0.16
set dash_radius, 0.045
set_color leonardo_ligand, [0.94, 0.43, 0.12]
set_color contact_hydrophobic, [0.00, 0.45, 0.70]
set_color contact_polar, [0.00, 0.62, 0.45]
set_color contact_salt, [0.84, 0.34, 0.00]
set_color contact_vdw, [0.52, 0.52, 0.52]
select ligand, resn BEN
select contact_residues, polymer.protein and chain A and resi 189+190+191+192+195+213+215+216+219+220+226
select pocket_shell, polymer.protein within 7.0 of ligand
show cartoon, polymer.protein
color gray88, polymer.protein
show surface, pocket_shell
set transparency, 0.80, pocket_shell
color gray92, pocket_shell
show sticks, contact_residues
set stick_radius, 0.145, contact_residues
color gray38, contact_residues and elem C
color red, contact_residues and elem O
color blue, contact_residues and elem N
color yelloworange, contact_residues and elem S
show sticks, ligand
set stick_radius, 0.235, ligand
color leonardo_ligand, ligand and elem C
color blue, ligand and elem N
distance scene_contact_01, (polymer.protein and chain A and resi 219 and resn GLY and name O), (resn BEN and resi 1 and name N1)
hide labels, scene_contact_01
color contact_polar, scene_contact_01
distance scene_contact_02, (polymer.protein and chain A and resi 189 and resn ASP and name OD2), (resn BEN and resi 1 and name N2)
hide labels, scene_contact_02
color contact_salt, scene_contact_02
distance scene_contact_03, (polymer.protein and chain A and resi 190 and resn SER and name OG), (resn BEN and resi 1 and name N2)
hide labels, scene_contact_03
color contact_polar, scene_contact_03
distance scene_contact_04, (polymer.protein and chain A and resi 190 and resn SER and name O), (resn BEN and resi 1 and name C)
hide labels, scene_contact_04
color contact_vdw, scene_contact_04
distance scene_contact_05, (polymer.protein and chain A and resi 226 and resn GLY and name CA), (resn BEN and resi 1 and name N2)
hide labels, scene_contact_05
color contact_vdw, scene_contact_05
distance scene_contact_06, (polymer.protein and chain A and resi 189 and resn ASP and name CG), (resn BEN and resi 1 and name N2)
hide labels, scene_contact_06
color contact_vdw, scene_contact_06
distance scene_contact_07, (polymer.protein and chain A and resi 195 and resn SER and name OG), (resn BEN and resi 1 and name C5)
hide labels, scene_contact_07
color contact_vdw, scene_contact_07
distance scene_contact_08, (polymer.protein and chain A and resi 219 and resn GLY and name O), (resn BEN and resi 1 and name C2)
hide labels, scene_contact_08
color contact_vdw, scene_contact_08
distance scene_contact_09, (polymer.protein and chain A and resi 213 and resn VAL and name CG1), (resn BEN and resi 1 and name C5)
hide labels, scene_contact_09
color contact_hydrophobic, scene_contact_09
orient ligand or contact_residues
turn x, 20
turn y, -28
turn z, -5
zoom ligand or contact_residues, 3.5
clip slab, 24
png /Users/juvenalyosa/ProtLigInteract/Leonardo_clean/draft/figures/figure_4b_3ptb_ben_interaction_scene_render.png, width=1800, height=1270, dpi=300, ray=1
