import numpy as np

from interaction_runtime import assign_ligand_hydrophobic_flags_from_rdkit
from interaction_runtime import assign_ligand_ionic_groups_from_topology
from interaction_runtime import build_ligand_charge_groups
from interaction_runtime import build_protein_charge_groups
from interaction_runtime import classify_hbond_geometry
from interaction_runtime import classify_pi_ring_pair
from interaction_runtime import is_supported_ligand_cation_center
from interaction_runtime import reduce_hydrophobic_pairs
from interaction_runtime import validate_halogen_bond
from interaction_runtime import validate_metal_coordination
from interaction_runtime import validate_ring_charge_interaction
from interaction_runtime import vdw_contact_cutoff


def _atom(name, element, xyz, **extra):
    payload = {
        "name": name,
        "element": element,
        "coord": np.array(xyz, dtype=float),
        "resn": extra.pop("resn", "LIG"),
        "resi": extra.pop("resi", 1),
        "chain": extra.pop("chain", "Z"),
    }
    payload.update(extra)
    return payload


def test_build_protein_charge_groups_uses_group_centers():
    atoms = [
        {"resn": "ARG", "resi": 15, "chain": "A", "name": "NE", "coord": np.array([0.0, 0.0, 0.0])},
        {"resn": "ARG", "resi": 15, "chain": "A", "name": "CZ", "coord": np.array([1.0, 0.0, 0.0])},
        {"resn": "ARG", "resi": 15, "chain": "A", "name": "NH1", "coord": np.array([2.0, 1.0, 0.0])},
        {"resn": "ARG", "resi": 15, "chain": "A", "name": "NH2", "coord": np.array([2.0, -1.0, 0.0])},
        {"resn": "ASP", "resi": 27, "chain": "A", "name": "CG", "coord": np.array([5.0, 0.0, 0.0])},
        {"resn": "ASP", "resi": 27, "chain": "A", "name": "OD1", "coord": np.array([6.0, 1.0, 0.0])},
        {"resn": "ASP", "resi": 27, "chain": "A", "name": "OD2", "coord": np.array([6.0, -1.0, 0.0])},
    ]

    payload = build_protein_charge_groups(atoms)

    assert len(payload["positive"]) == 1
    assert len(payload["negative"]) == 1
    assert np.allclose(payload["positive"][0]["center"], np.array([2.0, 0.0, 0.0]))
    assert np.allclose(payload["negative"][0]["center"], np.array([6.0, 0.0, 0.0]))
    assert payload["positive"][0]["representative_atom"] == "NH1"
    assert payload["negative"][0]["representative_atom"] == "OD1"


def test_build_ligand_charge_groups_respects_shared_group_id_and_center_members():
    lig_atoms = [
        {
            "resn": "LIG",
            "resi": 1,
            "chain": "Z",
            "name": "O1",
            "coord": np.array([0.0, 1.0, 0.0]),
            "group_type": "Carboxylate",
            "is_anion": True,
            "ionic_group_id": "Carboxylate:1",
            "ionic_group_center_members": ("O1", "O2"),
        },
        {
            "resn": "LIG",
            "resi": 1,
            "chain": "Z",
            "name": "O2",
            "coord": np.array([0.0, -1.0, 0.0]),
            "group_type": "Carboxylate",
            "is_anion": True,
            "ionic_group_id": "Carboxylate:1",
            "ionic_group_center_members": ("O1", "O2"),
        },
        {
            "resn": "LIG",
            "resi": 1,
            "chain": "Z",
            "name": "N1",
            "coord": np.array([4.0, 0.0, 0.0]),
            "group_type": "Ammonium",
            "is_cation": True,
            "charge_formal": 1,
            "ionic_group_id": "Ammonium:2",
            "ionic_group_center_members": ("N1",),
        },
    ]

    groups = build_ligand_charge_groups(lig_atoms)
    groups = sorted(groups, key=lambda g: g["charge_sign"])

    assert len(groups) == 2
    assert groups[0]["charge_sign"] == -1
    assert np.allclose(groups[0]["center"], np.array([0.0, 0.0, 0.0]))
    assert groups[1]["charge_sign"] == +1
    assert np.allclose(groups[1]["center"], np.array([4.0, 0.0, 0.0]))


def test_ligand_cation_typing_rejects_neutral_nitrogens_and_bogus_flags():
    neutral_tertiary = _atom(
        "N1",
        "N",
        [0.0, 0.0, 0.0],
        charge_formal=0,
        is_cation=True,
        neighbor_elements=("C", "C", "C"),
    )
    neutral_pyridine = _atom(
        "N2",
        "N",
        [0.0, 0.0, 0.0],
        charge_formal=0,
        neighbor_elements=("C", "C"),
    )
    nitro_like = _atom(
        "N3",
        "N",
        [0.0, 0.0, 0.0],
        charge_formal=1,
        neighbor_elements=("O", "O", "C"),
    )
    ammonium = _atom(
        "N4",
        "N",
        [0.0, 0.0, 0.0],
        charge_formal=1,
        group_type="Ammonium",
        neighbor_elements=("C", "H", "H", "H"),
    )
    guanidinium_member = _atom(
        "N5",
        "N",
        [0.0, 0.0, 0.0],
        charge_formal=0,
        is_cation=True,
        group_type="Guanidinium",
        ionic_group_id="Guanidinium:1",
        neighbor_elements=("C", "H", "H"),
    )
    bogus_ammonium_group = _atom(
        "N6",
        "N",
        [0.0, 0.0, 0.0],
        charge_formal=0,
        is_cation=True,
        group_type="Ammonium",
        ionic_group_id="Ammonium:bad",
        neighbor_elements=("C", "H", "H"),
    )
    ammonium_without_formal_charge = _atom(
        "N7",
        "N",
        [0.0, 0.0, 0.0],
        charge_formal=0,
        neighbor_elements=("C", "H", "H", "H"),
    )

    assert is_supported_ligand_cation_center(neutral_tertiary) is False
    assert is_supported_ligand_cation_center(neutral_pyridine) is False
    assert is_supported_ligand_cation_center(nitro_like) is False
    assert is_supported_ligand_cation_center(ammonium) is True
    assert is_supported_ligand_cation_center(guanidinium_member) is True
    assert is_supported_ligand_cation_center(ammonium_without_formal_charge) is True
    assert build_ligand_charge_groups([neutral_tertiary]) == []
    assert build_ligand_charge_groups([bogus_ammonium_group]) == []


def test_topology_ionic_assignment_recovers_uncharged_carboxylate_for_salt_bridges():
    atoms = [
        _atom("C1", "C", [0.0, 0.0, 0.0], charge_formal=0),
        _atom("O1", "O", [1.25, 0.0, 0.0], charge_formal=0),
        _atom("O2", "O", [-1.25, 0.0, 0.0], charge_formal=0),
        _atom("C2", "C", [0.0, 1.55, 0.0], charge_formal=0),
    ]

    typed = assign_ligand_ionic_groups_from_topology(atoms)
    groups = build_ligand_charge_groups(typed)

    assert len(groups) == 1
    assert groups[0]["charge_sign"] == -1
    assert groups[0]["group_type"] == "Carboxylate"
    assert groups[0]["representative_atom"] == "O1"
    assert np.allclose(groups[0]["center"], np.array([0.0, 0.0, 0.0]))


def test_topology_ionic_assignment_does_not_mark_protonated_carboxylic_acid_oxygen():
    atoms = [
        _atom("C1", "C", [0.0, 0.0, 0.0], charge_formal=0),
        _atom("O1", "O", [1.25, 0.0, 0.0], charge_formal=0),
        _atom("O2", "O", [-1.25, 0.0, 0.0], charge_formal=0),
        _atom("H2", "H", [-1.95, 0.0, 0.0], charge_formal=0),
    ]

    typed = assign_ligand_ionic_groups_from_topology(atoms)
    groups = build_ligand_charge_groups(typed)

    assert groups == []


def test_topology_ionic_assignment_recovers_explicit_ammonium_without_formal_charge():
    atoms = [
        _atom("N1", "N", [0.0, 0.0, 0.0], charge_formal=0),
        _atom("C1", "C", [1.45, 0.0, 0.0], charge_formal=0),
        _atom("H1", "H", [-0.75, 0.75, 0.0], charge_formal=0),
        _atom("H2", "H", [-0.75, -0.75, 0.0], charge_formal=0),
        _atom("H3", "H", [0.0, 0.0, 1.0], charge_formal=0),
    ]

    typed = assign_ligand_ionic_groups_from_topology(atoms)
    groups = build_ligand_charge_groups(typed)

    assert len(groups) == 1
    assert groups[0]["charge_sign"] == +1
    assert groups[0]["group_type"] == "Ammonium"
    assert groups[0]["representative_atom"] == "N1"


def test_topology_ionic_assignment_recovers_terminal_amidinium_without_hydrogens():
    atoms = [
        _atom("C1", "C", [0.0, 0.0, 0.0], charge_formal=0),
        _atom("N1", "N", [1.30, 0.0, 0.0], charge_formal=0),
        _atom("N2", "N", [-1.30, 0.0, 0.0], charge_formal=0),
        _atom("C2", "C", [0.0, 1.45, 0.0], charge_formal=0),
    ]

    typed = assign_ligand_ionic_groups_from_topology(atoms)
    groups = build_ligand_charge_groups(typed)

    assert len(groups) == 1
    assert groups[0]["charge_sign"] == +1
    assert groups[0]["group_type"] == "Amidinium"
    assert groups[0]["representative_atom"] == "N1"
    assert np.allclose(groups[0]["center"], np.array([0.0, 0.0, 0.0]))


def test_reduce_hydrophobic_pairs_applies_two_stage_closest_filter():
    prot_a = {"chain": "A", "resi": 10, "name": "CG", "coord": np.array([0.0, 0.0, 0.0]), "pymol_index": 10}
    prot_b = {"chain": "A", "resi": 10, "name": "CD1", "coord": np.array([0.5, 0.0, 0.0]), "pymol_index": 11}
    lig_1 = {"chain": "Z", "resi": 1, "name": "C1", "coord": np.array([3.2, 0.0, 0.0]), "_lig_idx": 1}
    lig_2 = {"chain": "Z", "resi": 1, "name": "C2", "coord": np.array([3.1, 0.0, 0.0]), "_lig_idx": 2}

    pairs = [
        {"prot_atom": prot_a, "lig_atom": lig_1, "distance": 3.7},
        {"prot_atom": prot_b, "lig_atom": lig_1, "distance": 3.4},
        {"prot_atom": prot_b, "lig_atom": lig_2, "distance": 3.2},
    ]

    reduced = reduce_hydrophobic_pairs(pairs)

    assert len(reduced) == 1
    assert reduced[0]["prot_atom"]["name"] == "CD1"
    assert reduced[0]["lig_atom"]["name"] == "C2"
    assert float(reduced[0]["distance"]) == 3.2


def test_vdw_contact_cutoff_uses_element_radii_with_padding():
    atom_a = {"element": "C"}
    atom_b = {"element": "O"}
    cutoff = vdw_contact_cutoff(atom_a, atom_b, padding=0.5, max_cutoff=10.0)
    assert abs(float(cutoff) - 3.72) < 1.0e-6


def test_hbond_geometry_accepts_linear_explicit_h_and_rejects_bent_contact():
    donor = _atom("N", "N", [0.0, 0.0, 0.0])
    hydrogen = _atom("H", "H", [1.0, 0.0, 0.0])
    acceptor = _atom("O", "O", [2.7, 0.0, 0.0])
    bent_acceptor = _atom("O", "O", [1.2, 2.6, 0.0])

    good = classify_hbond_geometry(donor, acceptor, hydrogen=hydrogen)
    bad = classify_hbond_geometry(donor, bent_acceptor, hydrogen=hydrogen)

    assert good["ok"] is True
    assert good["mode"] == "explicit_hydrogen"
    assert abs(float(good["angle"]) - 180.0) < 1.0e-6
    assert bad["ok"] is False
    assert bad["reason"] == "donor_h_acceptor_angle"


def test_hbond_fallback_uses_acceptor_heavy_neighbor_angle():
    donor = _atom("N", "N", [0.0, 0.0, 0.0])
    acceptor = _atom("O", "O", [3.0, 0.0, 0.0])
    neighbor_good = _atom("C", "C", [4.2, 0.0, 0.0])
    neighbor_bad = _atom("C", "C", [2.7, 0.15, 0.0])

    good = classify_hbond_geometry(donor, acceptor, acceptor_neighbor=neighbor_good)
    bad = classify_hbond_geometry(donor, acceptor, acceptor_neighbor=neighbor_bad)

    assert good["ok"] is True
    assert good["mode"] == "heavy_atom_fallback"
    assert bad["ok"] is False
    assert bad["reason"] == "donor_acceptor_neighbor_angle"


def test_halogen_bond_requires_sigma_hole_geometry_and_excludes_fluorine():
    carbon = _atom("C1", "C", [0.0, 0.0, 0.0])
    chlorine = _atom("CL1", "CL", [1.8, 0.0, 0.0])
    fluorine = _atom("F1", "F", [1.35, 0.0, 0.0])
    acceptor = _atom("OD1", "O", [3.6, 0.0, 0.0], resn="ASP", chain="A")
    acc_neighbor = _atom("CG", "C", [4.2, 1.039, 0.0], resn="ASP", chain="A")
    bent_acceptor = _atom("OD1", "O", [1.8, 2.5, 0.0], resn="ASP", chain="A")

    good = validate_halogen_bond(chlorine, carbon, acceptor, acceptor_neighbor=acc_neighbor)
    bad_angle = validate_halogen_bond(chlorine, carbon, bent_acceptor, acceptor_neighbor=acc_neighbor)
    bad_f = validate_halogen_bond(fluorine, carbon, acceptor, acceptor_neighbor=acc_neighbor)

    assert good["ok"] is True
    assert abs(float(good["donor_angle"]) - 180.0) < 1.0e-6
    assert bad_angle["ok"] is False
    assert bad_angle["reason"] == "donor_angle"
    assert bad_f["ok"] is False
    assert bad_f["reason"] == "unsupported_halogen_donor"


def test_metal_coordination_uses_element_specific_cutoffs_and_donor_elements():
    potassium = _atom("K", "K", [0.0, 0.0, 0.0], resn="K", chain="")
    oxygen = _atom("O1", "O", [3.25, 0.0, 0.0])
    carbon = _atom("C1", "C", [2.2, 0.0, 0.0])

    good = validate_metal_coordination(potassium, oxygen)
    bad = validate_metal_coordination(potassium, carbon)

    assert good["ok"] is True
    assert good["cutoff"] >= 3.25
    assert bad["ok"] is False
    assert bad["reason"] == "unsupported_donor_element"


def test_pi_ring_pair_classification_and_ring_charge_height_filter():
    protein_ring = {
        "resn": "PHE",
        "resi": 10,
        "chain": "A",
        "centroid": np.array([0.0, 0.0, 0.0]),
        "normal": np.array([0.0, 0.0, 1.0]),
    }
    ligand_parallel = {
        "resn": "LIG",
        "resi": 1,
        "chain": "Z",
        "centroid": np.array([0.4, 0.2, 3.6]),
        "normal": np.array([0.0, 0.0, -1.0]),
    }
    ligand_t = {
        "resn": "LIG",
        "resi": 1,
        "chain": "Z",
        "centroid": np.array([0.4, 0.2, 4.8]),
        "normal": np.array([1.0, 0.0, 0.0]),
    }

    stacked = classify_pi_ring_pair(protein_ring, ligand_parallel)
    tshape = classify_pi_ring_pair(protein_ring, ligand_t)
    cation_good = validate_ring_charge_interaction(protein_ring, np.array([0.2, 0.1, 3.0]))
    cation_coplanar = validate_ring_charge_interaction(protein_ring, np.array([0.2, 0.1, 0.2]))

    assert stacked["ok"] is True
    assert stacked["type"] == "Pi-Pi Stacking"
    assert tshape["ok"] is True
    assert tshape["type"] == "T-Shaped Pi-Pi"
    assert cation_good["ok"] is True
    assert cation_coplanar["ok"] is False
    assert cation_coplanar["reason"] == "coplanar_or_too_low"


def test_rdkit_hydrophobic_assignment_excludes_heteroatom_bound_carbons():
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    ligand_atoms = [
        _atom("C1", "C", [0.0, 0.0, 0.0], is_hydrophobic=True),
        _atom("C2", "C", [1.5, 0.0, 0.0], is_hydrophobic=True),
        _atom("O1", "O", [2.9, 0.0, 0.0], is_hydrophobic=True),
    ]
    rd_to_feat_idx = {0: 0, 1: 1, 2: 2}

    typed = assign_ligand_hydrophobic_flags_from_rdkit(ligand_atoms, mol, rd_to_feat_idx)

    assert typed[0]["is_hydrophobic"] is True
    assert typed[1]["is_hydrophobic"] is False
    assert typed[2]["is_hydrophobic"] is False


def _embedded_ligand_atoms(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    conf = mol.GetConformer()
    return [
        _atom(f"{a.GetSymbol()}{a.GetIdx()}", a.GetSymbol(), list(conf.GetAtomPosition(a.GetIdx())), charge_formal=0)
        for a in mol.GetAtoms()
    ]


def test_topology_ionic_assignment_keeps_neutral_aromatic_heterocycles_uncharged():
    # A ring carbon between ring nitrogens is a neutral heteroaromatic (adenine, guanine,
    # aminopyrimidine, imidazole), not an amidinium/guanidinium: these were typed as cations
    # and gave salt bridges and cation-pi contacts to ATP, GTP and kinase inhibitors.
    for smiles in ("Nc1ncnc2[nH]cnc12", "Nc1nc2[nH]cnc2c(=O)[nH]1", "Nc1ncccn1", "c1c[nH]cn1"):
        groups = build_ligand_charge_groups(assign_ligand_ionic_groups_from_topology(_embedded_ligand_atoms(smiles)))
        assert [g for g in groups if g["charge_sign"] > 0] == [], smiles
    # acyclic amidine / guanidine stay cationic
    for smiles, group in (("NC(=N)c1ccccc1", "Amidinium"), ("NC(=N)NCC", "Guanidinium")):
        groups = build_ligand_charge_groups(assign_ligand_ionic_groups_from_topology(_embedded_ligand_atoms(smiles)))
        assert [(g["group_type"], g["charge_sign"]) for g in groups] == [(group, +1)], smiles


def test_hbond_distance_is_always_donor_acceptor():
    donor = _atom("N", "N", [0.0, 0.0, 0.0])
    hydrogen = _atom("H", "H", [1.0, 0.0, 0.0])
    acceptor = _atom("O", "O", [2.9, 0.0, 0.0])
    explicit = classify_hbond_geometry(donor, acceptor, hydrogen=hydrogen)
    heavy = classify_hbond_geometry(donor, acceptor)
    # it used to be H...A (1.9) with explicit H and D...A (2.9) without
    assert abs(explicit["distance"] - 2.9) < 1e-6 and abs(explicit["ha_distance"] - 1.9) < 1e-6
    assert abs(heavy["distance"] - 2.9) < 1e-6 and heavy["ha_distance"] is None
    assert explicit["distance_kind"] == heavy["distance_kind"] == "D...A"


def test_hbond_heavy_atom_mode_uses_polar_contact_cutoff():
    donor = _atom("N", "N", [0.0, 0.0, 0.0])
    near = classify_hbond_geometry(donor, _atom("O", "O", [3.5, 0.0, 0.0]))
    far = classify_hbond_geometry(donor, _atom("O", "O", [3.9, 0.0, 0.0]))
    hydrogen = _atom("H", "H", [1.0, 0.0, 0.0])
    far_explicit = classify_hbond_geometry(donor, _atom("O", "O", [3.7, 0.0, 0.0]), hydrogen=hydrogen)
    assert near["ok"] is True
    assert far["ok"] is False and far["reason"] == "donor_acceptor_distance"
    assert far_explicit["ok"] is True            # with the H known PLIP's criteria apply


def test_hbond_ring_nitrogen_judged_by_lone_pair_direction():
    # pyridine-like N with ring neighbours at +-120 deg; donor straight along the lone pair
    acceptor = _atom("N1", "N", [0.0, 0.0, 0.0])
    c2 = _atom("C2", "C", [-0.67, 1.16, 0.0])
    c6 = _atom("C6", "C", [-0.67, -1.16, 0.0])
    donor = _atom("N", "N", [2.9, 0.0, 0.0], resn="MET", chain="A")
    res = classify_hbond_geometry(donor, acceptor, acceptor_neighbors=[c2, c6])
    assert res["ok"] is True and res["angle"] > 170.0
    # the closest single neighbour gave ~120 deg here and 97 deg for the real EGFR hinge
    # a donor in the ring plane behind C2 is not an H-bond
    behind = _atom("N", "N", [-2.2, 2.3, 0.0], resn="MET", chain="A")
    assert classify_hbond_geometry(behind, acceptor, acceptor_neighbors=[c2, c6])["ok"] is False


def test_hbond_backbone_donor_must_point_at_acceptor():
    # backbone N with C(prev) and CA: its H lies opposite them
    donor = _atom("N", "N", [0.0, 0.0, 0.0], resn="GLY", chain="A")
    c_prev = _atom("C", "C", [-0.7, 1.2, 0.0], resn="GLY", chain="A")
    ca = _atom("CA", "C", [-0.7, -1.2, 0.0], resn="GLY", chain="A")
    front = _atom("O1", "O", [2.9, 0.0, 0.0])
    back = _atom("O1", "O", [-2.0, 2.0, 0.5])
    assert classify_hbond_geometry(donor, front, donor_neighbors=[c_prev, ca])["ok"] is True
    bad = classify_hbond_geometry(donor, back, donor_neighbors=[c_prev, ca])
    assert bad["ok"] is False and bad["reason"] == "donor_neighbor_angle"


def test_pi_ring_angle_is_reported_between_0_and_90():
    protein_ring = {"centroid": np.zeros(3), "normal": np.array([0.0, 0.0, 1.0])}
    # anti-parallel normals (157 deg) are the same plane orientation as 23 deg
    tilt = np.radians(157.0)
    ligand_ring = {"centroid": np.array([0.3, 0.0, 3.8]), "normal": np.array([np.sin(tilt), 0.0, np.cos(tilt)])}
    res = classify_pi_ring_pair(protein_ring, ligand_ring)
    assert res["ok"] is True and res["type"] == "Pi-Pi Stacking"
    assert abs(res["angle"] - 23.0) < 1e-6 and "angle 23°" in res["details"]


def test_every_record_says_what_its_distance_measures():
    from interaction_runtime import INTERACTION_DISTANCE_KIND, build_interaction_record

    prot = _atom("OD1", "O", [0.0, 0.0, 0.0], resn="ASP", chain="A")
    lig = _atom("N1", "N", [2.9, 0.0, 0.0])
    for itype, kind in INTERACTION_DISTANCE_KIND.items():
        assert build_interaction_record(itype, prot, lig, 2.9)["distance_kind"] == kind


def test_protein_hydrophobic_atoms_are_carbons_bonded_only_to_carbon():
    from interaction_runtime import protein_hydrophobic_atom_names

    assert "NE1" not in protein_hydrophobic_atom_names("TRP")
    assert "CD1" not in protein_hydrophobic_atom_names("TRP")      # bonded to NE1
    assert "CZ" not in protein_hydrophobic_atom_names("TYR")       # bonded to OH
    assert "CD" not in protein_hydrophobic_atom_names("PRO")       # bonded to N
    assert "CD1" in protein_hydrophobic_atom_names("LEU")


def test_protein_donor_hydrogen_counts():
    from interaction_typing_runtime import protein_donor_hydrogen_count

    assert protein_donor_hydrogen_count("THR", "OG1") == 1
    assert protein_donor_hydrogen_count("LYS", "NZ") == 3
    assert protein_donor_hydrogen_count("ARG", "NH1") == 2
    assert protein_donor_hydrogen_count("GLN", "NE2") == 2
    assert protein_donor_hydrogen_count("HIS", "NE2") == 1
    assert protein_donor_hydrogen_count("PRO", "N") == 0
    assert protein_donor_hydrogen_count("ALA", "N") == 1


_BEN_PDB = """\
HETATM 1632  C1  BEN A   1      -1.853  14.311  16.658  1.00 19.86           C
HETATM 1633  C2  BEN A   1      -2.107  15.653  16.758  1.00 19.86           C
HETATM 1634  C3  BEN A   1      -1.774  16.341  17.932  1.00 19.86           C
HETATM 1635  C4  BEN A   1      -1.175  15.662  19.005  1.00 19.86           C
HETATM 1636  C5  BEN A   1      -0.914  14.295  18.885  1.00 19.86           C
HETATM 1637  C6  BEN A   1      -1.257  13.634  17.708  1.00 19.86           C
HETATM 1638  C   BEN A   1      -2.193  13.627  15.496  1.00 19.86           C
HETATM 1639  N1  BEN A   1      -2.797  14.235  14.491  1.00 19.86           N
HETATM 1640  N2  BEN A   1      -1.762  12.391  15.309  1.00 19.86           N
"""


def test_ligand_bond_orders_from_pdb_keeps_the_aromatic_ring(tmp_path, monkeypatch):
    """A PDB has no bond orders: read as all-single the benzamidine ring became a cyclohexane and
    protonation gave 17 H instead of 9."""
    import pytest

    pytest.importorskip("rdkit")
    from rdkit import Chem
    from leonardo.component_library_runtime import ligand_bond_orders_from_pdb

    monkeypatch.setenv("LEONARDO_PARAMETERS_DIR", str(tmp_path))      # empty: no CCD template
    for kwargs in ({"smiles_hint": "[NH2+]=C(N)c1ccccc1"}, {}):       # template, then Open Babel
        if not kwargs:
            pytest.importorskip("openbabel")
        mol, source = ligand_bond_orders_from_pdb(_BEN_PDB, resn="", fetch_ccd=False, **kwargs)
        assert mol is not None, source
        assert sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) == 6, source
        assert [a.GetPDBResidueInfo().GetName().strip() for a in mol.GetAtoms()][-2:] == ["N1", "N2"]
        # benzamidine (8 H) or, with the charged template, benzamidinium (9 H); never 16-17
        assert Chem.AddHs(mol).GetNumAtoms() - mol.GetNumAtoms() in (8, 9)
