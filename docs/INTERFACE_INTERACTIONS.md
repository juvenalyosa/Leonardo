# Macromolecular interface interactions

`src/leonardo/macro_interaction_runtime.py` analyses the interface between two parts of a
complex: protein–protein (two chains or any two selections), protein–DNA and protein–RNA.
It is pure Python + numpy (no PLIP, no PyMOL needed) and reuses the geometric predicates of
`interaction_runtime.py`, the module behind Leonardo's protein–ligand interactions.

The assistant tool is `analyze_interface(object, selection_a, selection_b, include_waters,
sasa_method, show, max_rows)` (permission `compute_light`). With no selections it compares
protein with nucleic acid when both are present. Otherwise it compares the first chain with
the chain it touches most. A selection can be chain ids (`A`, `A+B`), `protein`, `nucleic`,
`dna`, `rna`, chain:residues (`A:10-50,60`) or any PyMOL selection inside the object. In
PyMOL the tool creates the selections `iface_<obj>_a/_b/_res` and one dash object for each
family (`iface_<obj>_hb`, `_sb`, `_pi`, `_cpi`, `_hyd`, `_wat`, `_metal`, `_ss`). Styling is
done in `style_interface_minimal()` in `agent/tools/macro_interactions.py`.

## Record schema

Every interaction has `type`, `subtype` (part of A – part of B, e.g. `side chain–phosphate`),
`a` and `b` (chain, resn, resi, atom, residue label, part), `distance` plus `distance_def`, and
`angle` plus `angle_def`. When a nucleic acid is involved, it also has `na_part`, `na_groove`
and `readout`. For bridges, `via` holds the water or ion. Ring contacts name the ring
(`ring(CG/CD1/…)`) and keep a `representative_atom`, which is used for drawing.

## What is detected and the criteria

| Type | Criterion |
|---|---|
| Hydrogen bond (backbone and side chain) | With H present: D···A ≤ 3.9 Å, H···A ≤ 2.7 Å, D–H···A ≥ 120°. Heavy atoms only: D···A ≤ 3.5 Å, every D···A–X ≥ 90° and every X–D···A ≥ 90° (X = bonded heavy atoms). Uses donor/acceptor tables for amino acids and nucleotides. |
| Salt bridge | Arg/Lys/His⁺ vs Asp/Glu/C-terminus/phosphate (OP1/OP2): charge centres ≤ 5.5 Å (PLIP) **and** the closest charged N···O ≤ 4.0 Å (Barlow & Thornton). His counts as charged when HIP/HSP or when both HD1 and HE2 are present. With no hydrogens it is *assumed* protonated and the record has `conditional`. |
| Hydrophobic | Apolar side-chain carbons (Leonardo's protein table) plus the thymine methyl and the deoxyribose C2′; C···C ≤ 4.0 Å; one record per residue pair (closest pair, with the pair count). |
| π-stacking / T-shaped | Phe/Tyr/Trp/His rings and nucleobase rings (both purine rings): centroids ≤ 5.5 Å, offset ≤ 2.0 Å, normals within 30° of parallel or of perpendicular (`classify_pi_ring_pair`). |
| Cation–π | Lys NZ or Arg CZ over a ring of the other side, including bases: ≤ 6.0 Å, offset ≤ 2.0 Å, ≥ 1.2 Å above the plane (`validate_ring_charge_interaction`). |
| Water bridge (optional) | A crystal water O ≤ 3.5 Å from a polar N/O of each side. Reported once per water and flagged `optional`. |
| Disulfide | SG–SG ≤ 2.5 Å across the interface. |
| Metal bridge | One ion coordinated by both sides, with the element-specific cutoffs of `validate_metal_coordination`. |

**Nucleic-acid labels.** `phosphate` is P, OP1/2/3, O5′ and O3′. `sugar` is every other
primed atom. Everything else is `base`. For DNA base atoms the groove edge is taken from
B-form: major is A N6/N7/C8, G O6/N7/C8, C N4/C5/C6 and T O4/C7/C5/C6; minor is A N3/C2,
G N2/N3/C2, and C/T O2. `readout` is then `phosphate backbone`, `sugar`, `base readout
(major|minor groove)`, `base`, or `base (stacking)` for ring contacts, which have no groove.
`nucleic_acid` in the result counts atom contacts (≤ 4.0 Å) by part and groove. It also
counts specific interactions by readout and lists the residues behind each readout.

**Interface summary.** Interface residues are those with a heavy atom within 4.5 Å of the
other side. `residue_contacts` lists residue pairs (minimum distance, atom contacts ≤ 4.0 Å,
interaction types). `hotspots` gives per-residue counts by type, the number of specific
interactions, buried area and number of partners. It is sorted by specific interactions,
then by buried area. These are counts, not ΔΔG predictions.

**Buried SASA** = SASA(A) + SASA(B) − SASA(A∪B), using heavy atoms of the two sides (no
water or ions). The default is a numpy Shrake–Rupley calculation: probe 1.4 Å, 192
golden-spiral points, Bondi radii. It also gives per-residue buried area. With
`sasa_method="pymol"` the tool uses PyMOL `get_area` (dot_solvent, density 3) and reports
only totals. `sasa.method` states which one was used. For 1BRS A/D the two methods give
1558 and 1554 Å².

## Reference complexes (tests)

`tests/data/macro` holds trimmed RCSB entries, used by `tests/test_macro_interaction_runtime.py`
and `tests/test_agent_macro_interactions.py`.

- **1BRS** barnase–barstar: Asp39(D)–Arg83/Arg87/His102(A), Glu76–Arg59, Arg59 NH–Asp35, His102 ND1–Gly31 N; buried area about 1560 Å².
- **1AAY** Zif268–DNA: Arg118/124/146/174/180 to guanine O6/N7 and His149 to G N7, all in the major groove; His125/His153/Arg103/Arg170 to phosphates. The Zn sites are not reported as bridges.
- **1URN** U1A–RNA: Tyr13 stacks on C10 and Phe56 on A11; Lys22 and Lys50 bind phosphates.

## Limitations

- No protonation or hydrogen placement. Without hydrogens, H-bonds are judged on heavy atoms. His charge is an assumption (flagged), and Asp/Glu/Lys/Arg are taken as charged.
- A pair that is both a salt bridge and an H-bond is reported as both. The H-bond has `also_salt_bridge`.
- Only the first model and alternate location blank/A are read. No biological assembly is generated (use the deposited chains). The mmCIF reader handles the `_atom_site` loop only.
- Grooves use B-form edge definitions and are given for DNA only. RNA base contacts are labelled `base`, without Hoogsteen/Watson–Crick/sugar-edge classification.
- Modified residues are typed only when listed (MSE, SEP, TPO, PTR, Amber/CHARMM protonation names). Ligands and other hetero groups in a selection count for contacts and area but have no donor/acceptor typing.
- Hydrophobic contacts are one record per residue pair. Barnase–barstar is a polar interface: its only tabled C···C pairs within 4 Å are the two between Phe82(A) and Trp44(D), so there is one record. For nucleic acids, only the thymine methyl and the deoxyribose C2′ count as apolar.
- Water bridges use one distance cutoff and no angles, and every crystal water in the file is considered.
- Heavy-atom Bondi radii without implicit hydrogens give somewhat smaller absolute SASA than NACCESS-style united-atom radii. Buried areas agree with literature to within about 5%.
