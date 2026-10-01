"""Pose RMSD with real RDKit molecules (the other redocking tests use stub objects)."""
import math

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem  # noqa: E402
from redock_runtime import compute_redocking_graph_rmsd, redocking_heavy_atom_skeleton  # noqa: E402


def _embedded(smiles, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _pdbqt_like_pose(ref, shift=(0.0, 0.0, 0.0), order=None):
    """What Open Babel gives back from a PDBQT pose: same atoms, every bond single, explicit
    hydrogens, no charges, atoms in another order (not sanitizable for charged groups)."""
    order = list(order if order is not None else range(ref.GetNumAtoms()))
    rw = Chem.RWMol()
    for old in order:
        a = Chem.Atom(ref.GetAtomWithIdx(old).GetAtomicNum())
        a.SetNoImplicit(True)
        rw.AddAtom(a)
    pos = {old: new for new, old in enumerate(order)}
    for b in ref.GetBonds():
        rw.AddBond(pos[b.GetBeginAtomIdx()], pos[b.GetEndAtomIdx()], Chem.BondType.SINGLE)
    conf = Chem.Conformer(len(order))
    ref_conf = ref.GetConformer()
    for old, new in pos.items():
        p = ref_conf.GetAtomPosition(old)
        conf.SetAtomPosition(new, (p.x + shift[0], p.y + shift[1], p.z + shift[2]))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    return mol


def test_unsanitizable_pdbqt_pose_gives_the_true_rmsd():
    ref = _embedded("NC(=[NH2+])c1ccccc1")           # benzamidinium, as in 3PTB
    rng = np.random.default_rng(1)
    order = [int(i) for i in rng.permutation(ref.GetNumAtoms())]
    pose = _pdbqt_like_pose(ref, shift=(0.3, 0.0, 0.0), order=order)
    # Open Babel's re-perception of the 3PTB poses: an extra H on an amidine N (N with four
    # bonds and no charge), "Explicit valence for atom N, 4", so sanitization fails
    rw = Chem.RWMol(pose)
    n_idx = next(a.GetIdx() for a in rw.GetAtoms()
                 if a.GetSymbol() == "N" and sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "H") == 2)
    h = Chem.Atom(1)
    h.SetNoImplicit(True)
    h_idx = rw.AddAtom(h)
    rw.AddBond(n_idx, h_idx, Chem.BondType.SINGLE)
    p = rw.GetConformer().GetAtomPosition(n_idx)
    rw.GetConformer().SetAtomPosition(h_idx, (p.x + 1.0, p.y, p.z))
    pose = rw.GetMol()
    with pytest.raises(Exception):
        Chem.SanitizeMol(Chem.Mol(pose))                 # the old path could not use it
    assert compute_redocking_graph_rmsd(ref, pose) == pytest.approx(0.3, abs=1e-6)


def test_symmetry_is_corrected():
    ref = _embedded("OC(=O)c1ccc(C)cc1")                 # carboxylate O's and ring flip
    heavy = [a.GetIdx() for a in ref.GetAtoms() if a.GetAtomicNum() > 1]
    ox = [a.GetIdx() for a in ref.GetAtoms() if a.GetSymbol() == "O"]
    order = list(range(ref.GetNumAtoms()))
    order[ox[0]], order[ox[1]] = order[ox[1]], order[ox[0]]   # swap the two oxygens
    pose = _pdbqt_like_pose(ref, order=order)
    # the same geometry: with the swap it is still an exact match
    assert compute_redocking_graph_rmsd(ref, pose) == pytest.approx(0.0, abs=1e-9)
    # a naive atom-order RMSD would see the swapped oxygens
    naive = math.sqrt(sum(
        (ref.GetConformer().GetAtomPosition(i) - pose.GetConformer().GetAtomPosition(i)).LengthSq()
        for i in heavy) / len(heavy))
    assert naive > 0.3


def test_different_connectivity_is_not_matched():
    ref = _embedded("NC(=[NH2+])c1ccccc1")
    other = _embedded("NC(=O)c1ccccc1C")                  # same heavy-atom count, other graph
    assert math.isnan(compute_redocking_graph_rmsd(ref, other))


def test_largest_fragment_is_used():
    ref = _embedded("OC(=O)c1ccccc1")
    pose = Chem.CombineMols(_pdbqt_like_pose(ref, shift=(0.0, 0.2, 0.0)), _embedded("[Na+]"))
    skel, xyz = redocking_heavy_atom_skeleton(pose)
    assert skel.GetNumAtoms() == 9 and len(xyz) == 9
    assert compute_redocking_graph_rmsd(ref, pose) == pytest.approx(0.2, abs=1e-6)
