# Protein preparation

**Prepare Protein…** in the main window, or `prepare_protein` from the AI assistant, turns a structure from the
PDB, a cryo-EM model or a prediction into a complete, protonated model ready for docking, MD or QM/MM. The
original object is not changed: the result is a new object (`<name>_prep`) with the changed residues as named
selections, and a JSON report next to the PDB file.

```
Prepare 1M17 at pH 7 and show me which residues changed protonation.
```

## What it does, in order

| Step | How | Notes |
|---|---|---|
| Alternate locations | The conformation with the highest mean occupancy of each residue (ties: the first letter) | Listed in the report |
| Existing hydrogens | Removed and rebuilt | Names from other programs would not match the force field |
| Modified residues | MSE → MET; other modified residues → their parent, **with a warning** | Phosphorylated SEP / TPO / PTR are **kept** (Amber ff14SB + phosaa14SB templates, phosphate as the dianion) |
| Waters, ligands, ions | Waters removed unless you keep them; ligands, ions and cofactors kept **unchanged** | Prepare ligands with *pKa & Protonation* / `prepare_docking_ligand` |
| Missing loops | Full sequence from SEQRES, RCSB (PDB entries) or the sequence you give; MODELLER builds the loop moving **only the missing residues and one neighbour on each side** (every other atom keeps its crystal position); PDBFixer when MODELLER or its license is missing (its loops need the minimization: without it their backbone is rough, which is reported) | Termini are not built unless you ask (they are usually disordered). A sequence that does not match the structure (a mutant, another construct) is ignored with a warning |
| Gaps left open | The chain is split there and both ends get proper termini | A residue left alone between two gaps is removed (reported) |
| Missing side-chain atoms | PDBFixer | |
| Protonation | PROPKA 3 pKa of every Asp, Glu, His, Lys, Cys, Tyr in its environment; protonated when pKa > pH | Without PROPKA: standard pKa values (reported) |
| Flips and His tautomers | `reduce -FLIP`: Asn/Gln/His side chains turned to the orientation with the best hydrogen bonds; the neutral His tautomer (HID/HIE) from reduce's hydrogen placement | |
| Metals and covalent ligands | Cys bound to a metal → thiolate (CYM); His bound through NE2 → HID, through ND1 → HIE; Cys bound to a ligand → thioether (CYX, no HG) | Ser/Lys adducts are reported: build them with the component library |
| Hydrogens | OpenMM `Modeller.addHydrogens` with the states above (Amber names) | |
| Minimization | Hydrogens and built atoms relax; crystal heavy atoms restrained at 10 kcal/mol/Å²; Amber14 + GBn2 implicit solvent (vacuum when waters are kept), CPU | Typical heavy-atom shift 0.1–0.2 Å |

The states survive the rest of Leonardo: the MD window and the component library (`build_validated_system`)
read the protonation from the hydrogens of a prepared structure instead of re-guessing it from the pH.

## In PyMOL

| Selection | Shown as |
|---|---|
| `<name>_built` | yellow cartoon, side chains as sticks |
| `<name>_protonated` | magenta sticks with the polar hydrogens (ASH, GLH, HIP, LYN, CYM, CYX) |
| `<name>_flipped` | cyan sticks |
| `<name>_completed` | orange sticks |

## Checked on

| Entry | What it tests | Result |
|---|---|---|
| Lysozyme 2LZT | Glu35 (experimental pKa ≈ 6.2) | PROPKA 6.5: GLH at pH 5, GLU at pH 7.4; Asp52 (exp. 3.7) 3.9, charged |
| RNase A 7RSA | 13 residues with alternate locations, hydrogens in the file | One conformation each, all hydrogens rebuilt |
| EGFR kinase 1M17 | 12-residue loop missing (965–976), erlotinib | Loop built with MODELLER, crystal atoms unchanged (< 0.01 Å outside the loop) |
| PKA 1ATP | Phospho-Thr197, phospho-Ser338, Mn²⁺-ATP | Phosphates kept and protonated, Mn-ATP kept |
| Zif268 1AAY | Three zinc fingers on DNA | 6 Cys → CYM, 6 His → HID (bound through NE2), DNA kept |
| BTK 5P9J | Ibrutinib bound to Cys481 | Cys481 a thioether (no HG) |
| Hemoglobin 4HHB, trypsin 3PTB, barnase–barstar 1BRS | Heme, multi-chain, chain break without sequence | Prepared; the break is split with termini |

Tests: `tests/test_protein_prep_runtime.py`, `tests/test_agent_protein_prep.py`.

## Limits

- PROPKA is an empirical method: it is good at the direction of a pKa shift, less at its size (typical error
  about 1 pH unit). Residues within about 1 unit of the pH can be in either state; the report lists every pKa.
- Loops longer than about 15 residues are hard for any method; the report gives MODELLER's DOPE score.
- Ligands are not protonated here.
