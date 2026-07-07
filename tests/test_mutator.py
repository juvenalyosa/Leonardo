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
