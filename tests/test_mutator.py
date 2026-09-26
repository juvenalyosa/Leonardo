import pytest

from mutator import MutationError
from mutator import mutate_sequence
from mutator import needleman_wunsch_affine


def test_mutator_accepts_string_input_prefix_and_case_normalizes():
    assert mutate_sequence("ACDE", "p.c2w", start_pos=1) == "AWDE"
    assert mutate_sequence("ACDE", "C2W; D3del", start_pos=1) == "AWE"


def test_mutator_rejects_unknown_mutation_instead_of_silent_noop():
    with pytest.raises(MutationError, match="Unsupported mutation format"):
        mutate_sequence("ACDE", ["not_a_mutation"], start_pos=1)


def test_mutator_deletion_validates_residue_and_range_presence():
    assert mutate_sequence("ACDEFG", ["C2_E4del"], start_pos=1) == "AFG"

    with pytest.raises(MutationError, match="mismatch"):
        mutate_sequence("ACDEFG", ["D2del"], start_pos=1)

    with pytest.raises(MutationError, match="missing/out-of-bounds"):
        mutate_sequence("ACDEFG", ["C2_Z99del"], start_pos=1)


def test_mutator_delins_validates_end_residue():
    assert mutate_sequence("ACDEFG", ["C2_E4delinsGG"], start_pos=1) == "AGGFG"

    with pytest.raises(MutationError, match="range end"):
        mutate_sequence("ACDEFG", ["C2_F4delinsGG"], start_pos=1)


def test_mutator_insertion_validates_flanks_and_adjacency():
    assert mutate_sequence("ACDE", ["C2_D3insGG"], start_pos=1) == "ACGGDE"

    with pytest.raises(MutationError, match="adjacent"):
        mutate_sequence("ACDE", ["C2_E4insGG"], start_pos=1)

    with pytest.raises(MutationError, match="right flank"):
        mutate_sequence("ACDE", ["C2_E3insGG"], start_pos=1)


def test_mutator_duplication_requires_complete_matching_range():
    assert mutate_sequence("ACDEFG", ["C2_E4dup"], start_pos=1) == "ACDECDEFG"

    with pytest.raises(MutationError, match="range end"):
        mutate_sequence("ACDEFG", ["C2_F4dup"], start_pos=1)


def test_needleman_wunsch_affine_handles_empty_sequences():
    assert needleman_wunsch_affine("", "") == ("", "")
    assert needleman_wunsch_affine("", "ABC") == ("---", "ABC")
    assert needleman_wunsch_affine("ABC", "") == ("ABC", "---")


def test_needleman_wunsch_affine_terminates_with_leading_gaps():
    # a leading gap of two or more residues used to loop forever in the backtrack
    for a, b in [("MKTAYIAK", "TAYIAK"), ("MKTAYIAK", "AYIAK"), ("TAYIAK", "MKTAYIAK"), ("A", "MKTA"), ("MKTA", "A")]:
        x, y = needleman_wunsch_affine(a, b)
        assert len(x) == len(y)
        assert x.replace("-", "") == a and y.replace("-", "") == b
    assert needleman_wunsch_affine("MKTAYIAK", "TAYIAK") == ("MKTAYIAK", "--TAYIAK")


def test_mutations_refer_to_the_reference_sequence_not_to_the_shifted_one():
    seq = "MKTAYIAKQR"
    # A7 is still called A7 after the deletion of residues 2-3 (HGVS numbering)
    assert mutate_sequence(seq, ["K2_T3del", "A7G"]) == "MAYIGKQR"
    # the residue that *moves* into position 5 must not be mutated silently
    with pytest.raises(MutationError, match="mismatch at position 5"):
        mutate_sequence(seq, ["K2_T3del", "A5G"])
    # the result does not depend on the order of the list
    assert mutate_sequence(seq, ["A7G", "K2_T3del"]) == mutate_sequence(seq, ["K2_T3del", "A7G"])
    assert mutate_sequence(seq, ["Y5dup", "M1W"]) == "WKTAYYIAKQR"
    assert mutate_sequence(seq, ["K2_T3del", "T3_A4insW"]) == "MWAYIAKQR"


def test_overlapping_mutations_are_rejected():
    for muts in (["K2_T3del", "T3A"], ["2_3insG", "K2_T3del"], ["A4G", "A4C"], ["3_4insG", "3_4insW"]):
        with pytest.raises(MutationError, match="overlapping"):
            mutate_sequence("MKTAYIAKQR", muts)


def test_insertion_requires_existing_flanks():
    with pytest.raises(MutationError, match="outside the current sequence"):
        mutate_sequence("MKTAYIAKQR", ["50_51insG"])
    with pytest.raises(MutationError, match="outside the current sequence"):
        mutate_sequence("MKTAYIAKQR", ["10_11insG"])


def test_structure_numbering_with_jumps_and_insertion_codes():
    labels = [10, 11, "11A", 12, 15]          # 11A: insertion code; 12 -> 15: numbering jump
    assert mutate_sequence("ACDEF", ["C11W"], residue_numbers=labels) == "AWDEF"
    assert mutate_sequence("ACDEF", ["F15Y"], residue_numbers=labels) == "ACDEY"
    assert mutate_sequence("ACDEF", ["11_12del"], residue_numbers=labels) == "AF"   # removes 11A too
    with pytest.raises(MutationError, match="not adjacent"):
        mutate_sequence("ACDEF", ["C11_E12insG"], residue_numbers=labels)
    with pytest.raises(MutationError, match="outside"):
        mutate_sequence("ACDEF", ["D13G"], residue_numbers=labels)


def test_negative_positions():
    assert mutate_sequence("ACDEF", ["A-3G"], start_pos=-3) == "GCDEF"
    assert mutate_sequence("ACDEF", ["C-2_E0del"], start_pos=-3) == "AF"
    assert mutate_sequence("ACDEF", ["-1_0insW"], start_pos=-3) == "ACDWEF"


def test_insertion_code_positions():
    labels = [183, 184, "184A", "184B", 185]
    assert mutate_sequence("ACDEF", ["D184AW"], residue_numbers=labels) == "ACWEF"
    assert mutate_sequence("ACDEF", ["C184W"], residue_numbers=labels) == "AWDEF"
    assert mutate_sequence("ACDEF", ["C184_D184AinsGG"], residue_numbers=labels) == "ACGGDEF"
    assert mutate_sequence("ACDEF", ["184A_184Bdel"], residue_numbers=labels) == "ACF"
    assert mutate_sequence("ACDEF", ["183_185del"], residue_numbers=labels) == ""
    with pytest.raises(MutationError, match="184A"):
        mutate_sequence("ACDEF", ["D184AW"], start_pos=183)       # no insertion codes in plain numbering


def test_numbering_jump_insertion_between_bonded_residues():
    # 12 -> 15 is a numbering jump, but the residues are neighbours
    assert mutate_sequence("ACDEF", ["E12_F15insG"], residue_numbers=[10, 11, "11A", 12, 15]) == "ACDEGF"


def test_three_letter_labels():
    seq = "MKTAYIAKQR"
    assert mutate_sequence(seq, ["p.Lys2Gly"]) == "MGTAYIAKQR"
    assert mutate_sequence(seq, ["p.(Lys2Gly)"]) == "MGTAYIAKQR"
    assert mutate_sequence(seq, ["Lys2_Thr3delinsTrpSer"]) == "MWSAYIAKQR"
    assert mutate_sequence(seq, ["Tyr5dup"]) == "MKTAYYIAKQR"
    assert mutate_sequence(seq, ["A4_Y5insSER"]) == "MKTASERYIAKQR"     # one-letter S-E-R stays one-letter
    # an insertion code must not merge with the next three-letter code
    labels = [183, 184, "184G", "184I", 185]
    assert mutate_sequence("ACDEF", ["Asp184GLys"], residue_numbers=labels) == "ACKEF"
    assert mutate_sequence("ACDEF", ["Glu184ILeu"], residue_numbers=labels) == "ACDLF"
    assert mutate_sequence("ACDEF", ["Asp184G_Glu184IdelinsGlySer"], residue_numbers=labels) == "ACGSF"
    assert mutate_sequence(seq, ["A4_Y5insDEL"]) == "MKTADELYIAKQR"          # D-E-L, not a deletion


def test_nonsense_mutation_truncates():
    seq = "MKTAYIAKQR"
    assert mutate_sequence(seq, ["Y5*"]) == "MKTA"
    assert mutate_sequence(seq, ["p.Tyr5Ter"]) == "MKTA"
    assert mutate_sequence(seq, ["K2G", "Y5*"]) == "MGTA"
    with pytest.raises(MutationError, match="after the stop codon"):
        mutate_sequence(seq, ["Y5*", "K8R"])
    with pytest.raises(MutationError, match="Y5\\* lies after the stop codon K2\\*"):
        mutate_sequence(seq, ["Y5*", "K2*"])
