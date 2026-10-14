from interaction_typing_runtime import (
    is_protein_hydrophobic_residue,
    normalize_interaction_residue_name,
    protein_aromatic_ring_atom_names,
    protein_hbond_acceptor_atom_names,
    protein_hbond_donor_atom_names,
    protein_hydrophobic_atom_names,
    protein_metal_donor_atom_names,
    protein_positive_atom_names,
)


def test_histidine_variants_keep_distinct_charge_and_hbond_roles():
    assert protein_positive_atom_names("HIS") == ()
    assert protein_positive_atom_names("HIP") == ("ND1", "NE2")
    assert protein_positive_atom_names("HSP") == ("ND1", "NE2")

    assert protein_hbond_donor_atom_names("HID") == ("ND1",)
    assert protein_hbond_acceptor_atom_names("HID") == ("NE2",)

    assert protein_hbond_donor_atom_names("HIE") == ("NE2",)
    assert protein_hbond_acceptor_atom_names("HIE") == ("ND1",)

    assert protein_hbond_donor_atom_names("HIP") == ("ND1", "NE2")
    assert protein_hbond_acceptor_atom_names("HIP") == ()


def test_interaction_residue_aliases_cover_common_md_variants():
    assert normalize_interaction_residue_name("hse") == "HIE"
    assert normalize_interaction_residue_name(" hsp ") == "HIP"
    assert normalize_interaction_residue_name("mse") == "MET"

    assert protein_aromatic_ring_atom_names("HSD") == ("CG", "ND1", "CD2", "CE1", "NE2")
    assert is_protein_hydrophobic_residue("MSE") is True
    assert is_protein_hydrophobic_residue("LYN") is False


def test_protein_atom_role_tables_cover_hydrophobic_and_metal_specificity():
    assert protein_hydrophobic_atom_names("LEU") == ("CB", "CG", "CD1", "CD2")
    assert protein_hydrophobic_atom_names("SER") == ()
    assert protein_metal_donor_atom_names("HSP") == ("ND1", "NE2")
    assert protein_metal_donor_atom_names("GLU") == ("OE1", "OE2")
