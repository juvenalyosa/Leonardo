import code_v2


def _dialog(chain_data, meta=None):
    class DummyDialog:
        pass

    for name in (
        "_normalize_chain_id",
        "_bioseq_align_sequence_index_map",
        "_bioseq_build_seq_index_to_resi_map",
        "_bioseq_residue_ranges_for_seq_range",
        "_bioseq_resi_expression",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    dlg = DummyDialog()
    dlg._bioseq_chain_data = chain_data
    dlg._bioseq_sequence_meta = dict(meta or {})
    return dlg


def test_structure_map_keeps_insertion_codes_and_aligns_full_sequence():
    labels = ["183", "184", "184A", "185", "188", "188A"]
    observed = "ACDEFG"
    dlg = _dialog({"A": {"observed_seq": observed, "residue_labels": labels, "positions": [183, 184, 184, 185, 188, 188]}})
    same = dlg._bioseq_build_seq_index_to_resi_map(observed, "obj", "A")
    assert same == {1: "183", 2: "184", 3: "184A", 4: "185", 5: "188", 6: "188A"}
    # RCSB/full sequence with extra N-terminal residues and a mutation still maps by alignment
    full = "MKTA" + "ACDWFG"
    mapped = dlg._bioseq_build_seq_index_to_resi_map(full, "obj", "A")
    assert mapped[5] == "183" and mapped[7] == "184A" and mapped[10] == "188A"
    assert dlg._bioseq_residue_ranges_for_seq_range(5, 7, mapped) == ["183", "184", "184A"]


def test_resi_expression_ranges_numbers_and_escapes_negatives():
    dlg = _dialog({})
    assert dlg._bioseq_resi_expression(["183", "184", "184A", "185", "186", "190"]) == "183-184+184A+185-186+190"
    assert dlg._bioseq_resi_expression(["-3", "-2", "-1", "0", "1"]) == "\\-3+\\-2+\\-1+0-1"
    assert dlg._bioseq_resi_expression([]) == ""
