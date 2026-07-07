"""
Protein Sequence Mutator 

The program applies specified mutations to a protein sequence and verifies the
changes by aligning the original sequence with the mutated one. This repaired
version uses a more advanced alignment algorithm (Needleman-Wunsch with
affine gap penalty) to produce clearer, more intuitive alignments for complex
mutations.

author = "El_Dryosa"
email = "juvenal.yosa@gmail.com"
version = "1.0"
"""
import re


class MutationError(Exception):
    """
    Raised when a requested mutation cannot be parsed or does not
    match the reference sequence (non-existent position, AA mismatch, etc.).

    This is intentionally a normal Python exception rather than an
    immediate program exit. Library / package code should *raise*
    MutationError and let the top-level application decide whether to:

        - catch it and handle/report the problem, or
        - not catch it, in which case the process will stop with a traceback.

    Example (recommended in a CLI script):

        from mutator import mutate_sequence, MutationError

        try:
            mutated = mutate_sequence(seq, mutations, start_pos)
        except MutationError as exc:
            print(f"ERROR: {exc}")
            raise SystemExit(1)
    """
    pass


def _clean_protein_sequence(sequence):
    """
    Normalize a user/API protein sequence without hiding invalid characters.

    UI callers already pass cleaned sequences, but this module is also imported
    directly by genmodels.py and tests. Keeping validation here prevents a bad
    command-line/API input from silently shifting mutation coordinates.
    """
    seq = re.sub(r"\s+", "", str(sequence or "")).upper()
    bad = sorted({ch for ch in seq if not ("A" <= ch <= "Z")})
    if bad:
        raise MutationError("Protein sequence contains invalid character(s): " + "".join(bad))
    return seq


def _iter_mutation_labels(mutations):
    """Return normalized mutation labels from a string or an iterable."""
    if mutations is None:
        return []
    if isinstance(mutations, str):
        raw_items = re.split(r"[\s,;]+", mutations)
    else:
        raw_items = list(mutations or [])
    labels = []
    for item in raw_items:
        label = str(item or "").strip()
        if not label:
            continue
        label = re.sub(r"\s+", "", label).upper()
        if label.startswith("P."):
            label = label[2:]
        if label:
            labels.append(label)
    return labels


def _coerce_start_pos(start_pos):
    try:
        return int(start_pos)
    except Exception as exc:
        raise MutationError(f"Invalid sequence start position: {start_pos!r}") from exc


def _validate_present_position(aa_dict, pos, mutation):
    if int(pos) not in aa_dict:
        raise MutationError(
            f"Mutation {mutation} references position {int(pos)}, "
            "but that position is outside the current sequence."
        )


def _validate_expected_aa(aa_dict, pos, expected_aa, mutation, label="position"):
    expected = str(expected_aa or "").strip().upper()
    if not expected:
        return
    _validate_present_position(aa_dict, pos, mutation)
    found = str(aa_dict.get(int(pos), "") or "")
    if found != expected:
        raise MutationError(
            f"Mutation {mutation} mismatch at {label} {int(pos)}: "
            f"found '{found}' but expected '{expected}'."
        )


def _validate_contiguous_range(aa_dict, start_pos, end_pos, mutation):
    start_i = int(start_pos)
    end_i = int(end_pos)
    if end_i < start_i:
        raise MutationError(f"Mutation {mutation} has an inverted range ({start_i}>{end_i}).")
    missing = [pos for pos in range(start_i, end_i + 1) if pos not in aa_dict]
    if missing:
        preview = ", ".join(str(x) for x in missing[:8])
        if len(missing) > 8:
            preview += ", ..."
        raise MutationError(
            f"Mutation {mutation} references missing/out-of-bounds position(s): {preview}."
        )
    return [aa_dict[pos] for pos in range(start_i, end_i + 1)]


def _shift_after_deletion(aa_dict, start_del, end_del):
    shift = (int(end_del) - int(start_del)) + 1
    new_dict = {}
    for pos_key, aa in sorted(aa_dict.items()):
        if pos_key < int(start_del):
            new_dict[pos_key] = aa
        elif pos_key > int(end_del):
            new_dict[pos_key - shift] = aa
    return new_dict


def _replace_range_with_sequence(aa_dict, start_pos, end_pos, insertion):
    deleted_len = int(end_pos) - int(start_pos) + 1
    net_shift = int(len(insertion)) - int(deleted_len)
    new_dict = {}
    for pos_key, aa in sorted(aa_dict.items()):
        if pos_key < int(start_pos):
            new_dict[pos_key] = aa
        elif pos_key > int(end_pos):
            new_dict[pos_key + net_shift] = aa
    for i, aa in enumerate(str(insertion)):
        new_dict[int(start_pos) + i] = aa
    return new_dict


def _insert_after_position(aa_dict, pos_after, insertion):
    insertion_len = len(insertion)
    new_dict = {}
    for pos_key, aa in sorted(aa_dict.items()):
        if pos_key <= int(pos_after):
            new_dict[pos_key] = aa
        else:
            new_dict[pos_key + insertion_len] = aa
    for i, aa in enumerate(str(insertion)):
        new_dict[int(pos_after) + 1 + i] = aa
    return new_dict

# =============================================================================
#  1. HELPER FUNCTIONS FOR SEQUENCE MANIPULATION
# =============================================================================

def convert_to_dict(sequence, start_pos):
    """
    Converts a sequence string to a dictionary mapping position numbers to amino acids.
    This data structure is ideal for mutations because it handles insertions and
    deletions without needing to manually re-calculate list indices.
    
    Example: "AGK", start_pos=1 -> {1: 'A', 2: 'G', 3: 'K'}
    """
    start_i = _coerce_start_pos(start_pos)
    seq = _clean_protein_sequence(sequence)
    return {i + start_i: aa for i, aa in enumerate(seq)}

def convert_to_sequence(aa_dict):
    """
    Converts an amino acid dictionary back to a continuous sequence string.
    It correctly handles any gaps in position numbers by creating a contiguous
    string from the minimum to the maximum position.
    """
    if not aa_dict:
        return ""
    # Determine the full range of the sequence after mutations.
    min_pos = min(aa_dict.keys())
    max_pos = max(aa_dict.keys())
    # Rebuild the string, getting the amino acid for each position or an empty string if a position was deleted.
    return ''.join(aa_dict.get(i, '') for i in range(min_pos, max_pos + 1))

# =============================================================================
#  2. CORE MUTATION LOGIC
# =============================================================================

def mutate_sequence(sequence, mutations, start_pos=1):
    """
    Applies a list of mutations to a protein sequence using a dictionary-based approach
    to robustly handle changes in sequence length.

    Args:
        sequence (str): The original amino acid sequence.
        mutations (list): A list of mutations in a simplified HGVS-like format.
                          Supported formats:
                          - Substitution: "A123G"
                          - Deletion: "A10del", "10_12del"
                          - Insertion: "A10_B11insGGG"
                          - Deletion-Insertion: "A10delinsGGG", "10_12delinsGGG"
                          - Duplication: "A10dup", "10_12dup"
        start_pos (int): The starting position number of the sequence (e.g., 1 for 1-based indexing).

    Returns:
        str: The mutated sequence.
    """
    # Convert the sequence to a dictionary for position-addressed edits. The
    # dictionary is rebuilt after indels so subsequent mutations are applied to
    # the current sequence state, matching the historical behavior of this tool.
    aa_dict = convert_to_dict(sequence, start_pos)

    for mutation in _iter_mutation_labels(mutations):
        # --- Case 1: Substitution (e.g., A123B) ---
        # Regex: ^([A-Z])(\d+)([A-Z])$ -> Start, one letter, numbers, one letter, End.
        snp_match = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation)
        if snp_match:
            old_aa, pos, new_aa = snp_match.groups()
            pos = int(pos)
            _validate_expected_aa(aa_dict, pos, old_aa, mutation)
            aa_dict[pos] = new_aa
            continue  # Move to the next mutation.

        # --- Case 2: Deletion (e.g., A10del or 10_12del) ---
        # Check for 'del' but exclude 'delins' which is handled separately.
        if 'DEL' in mutation and 'DELINS' not in mutation:
            # First, try to match a range deletion (e.g., 10_12del).
            range_del_match = re.match(r'^([A-Z]?)(\d+)_([A-Z]?)(\d+)DEL$', mutation)
            if range_del_match:
                start_aa, start_del, end_aa, end_del = range_del_match.groups()
                start_del, end_del = int(start_del), int(end_del)
            else:
                # If not a range, try to match a single-residue deletion (e.g., A10del).
                single_del_match = re.match(r'^([A-Z]?)(\d+)DEL$', mutation)
                if single_del_match:
                    start_aa, pos_del = single_del_match.groups()
                    end_aa = start_aa
                    start_del, end_del = int(pos_del), int(pos_del)
                else:
                    raise MutationError(f"Could not parse deletion mutation: '{mutation}'.")

            _validate_contiguous_range(aa_dict, start_del, end_del, mutation)
            _validate_expected_aa(aa_dict, start_del, start_aa, mutation, label="range start")
            _validate_expected_aa(aa_dict, end_del, end_aa, mutation, label="range end")
            aa_dict = _shift_after_deletion(aa_dict, start_del, end_del)
            continue
        
        # --- Case 3: Deletion-Insertion (e.g., A10delinsGGG or 10_12delinsGGG) ---
        if 'DELINS' in mutation:
            # This regex is robust: it handles optional start/end amino acids and optional end positions.
            # Example: A10delinsGGG -> start_pos=10, end_pos=10, insertion="GGG"
            # Example: 10_12delinsGGG -> start_pos=10, end_pos=12, insertion="GGG"
            delins_match = re.match(r'^([A-Z]?)(\d+)(?:_([A-Z]?)(\d+))?DELINS([A-Z]+)$', mutation)
            if delins_match:
                start_aa, start_pos_del, end_aa, end_pos_del, insertion = delins_match.groups()
                start_pos_del = int(start_pos_del)
                end_pos_del = int(end_pos_del) if end_pos_del else start_pos_del
                _validate_contiguous_range(aa_dict, start_pos_del, end_pos_del, mutation)
                _validate_expected_aa(aa_dict, start_pos_del, start_aa, mutation, label="range start")
                _validate_expected_aa(aa_dict, end_pos_del, end_aa, mutation, label="range end")
                aa_dict = _replace_range_with_sequence(aa_dict, start_pos_del, end_pos_del, insertion)
            else:
                raise MutationError(f"Could not parse delins mutation: '{mutation}'.")
            continue

        # --- Case 4: Insertion (e.g., A10_B11insGGG) ---
        # Insertions happen *between* two residues.
        if 'INS' in mutation:
            ins_match = re.match(r'^([A-Z]?)(\d+)_([A-Z]?)(\d+)INS([A-Z]+)$', mutation)
            if ins_match:
                left_aa, pos_after, right_aa, pos_before, insertion = ins_match.groups()
                pos_after = int(pos_after)
                pos_before = int(pos_before)
                if pos_before != pos_after + 1:
                    raise MutationError(
                        f"Insertion {mutation} must reference adjacent flanking residues "
                        f"({pos_after} and {pos_after + 1})."
                    )
                _validate_expected_aa(aa_dict, pos_after, left_aa, mutation, label="left flank")
                _validate_expected_aa(aa_dict, pos_before, right_aa, mutation, label="right flank")
                aa_dict = _insert_after_position(aa_dict, pos_after, insertion)
            else:
                raise MutationError(f"Could not parse insertion mutation: '{mutation}'.")
            continue

        # --- Case 5: Duplication (e.g., A10dup or 10_12dup) ---
        if 'DUP' in mutation:
            # Check for range duplication first.
            range_dup_match = re.match(r'^([A-Z]?)(\d+)_([A-Z]?)(\d+)DUP$', mutation)
            if range_dup_match:
                start_aa, start_dup, end_aa, end_dup = range_dup_match.groups()
                start_dup, end_dup = int(start_dup), int(end_dup)
            else:
                # Check for single residue duplication.
                single_dup_match = re.match(r'^([A-Z]?)(\d+)DUP$', mutation)
                if single_dup_match:
                    start_aa, pos_dup = single_dup_match.groups()
                    end_aa = start_aa
                    start_dup, end_dup = int(pos_dup), int(pos_dup)
                else:
                    raise MutationError(f"Could not parse duplication mutation: '{mutation}'.")
            
            duplicated_segment = _validate_contiguous_range(aa_dict, start_dup, end_dup, mutation)
            _validate_expected_aa(aa_dict, start_dup, start_aa, mutation, label="range start")
            _validate_expected_aa(aa_dict, end_dup, end_aa, mutation, label="range end")
            dup_len = len(duplicated_segment)
            
            # Rebuild the dictionary.
            new_dict = {}
            # 1. Copy residues up to the end of the duplicated segment.
            for pos_key, aa in sorted(aa_dict.items()):
                if pos_key <= end_dup:
                    new_dict[pos_key] = aa
                # 2. Shift all subsequent residues.
                else: 
                    new_dict[pos_key + dup_len] = aa

            # 3. Insert the duplicated segment immediately after the original segment.
            for i, aa in enumerate(duplicated_segment):
                new_dict[end_dup + 1 + i] = aa
            
            aa_dict = new_dict # Update the main dictionary.
            continue

        raise MutationError(f"Unsupported mutation format: '{mutation}'.")

    # Convert the final dictionary back to a sequence string.
    return convert_to_sequence(aa_dict)

# =============================================================================
#  3. ALIGNMENT AND VERIFICATION
# =============================================================================

def needleman_wunsch_affine(seq1, seq2, match_score=1, mismatch_score=-1, gap_open=-10, gap_extend=-1):
    """
    Performs global alignment using Needleman-Wunsch with an affine gap penalty.
    This is more biologically realistic than a simple linear gap penalty because it
    penalizes opening a new gap (e.g., an insertion/deletion event) more heavily
    than extending an existing gap.
    """
    seq1 = str(seq1 or "")
    seq2 = str(seq2 or "")
    n, m = len(seq1), len(seq2)
    if n == 0 and m == 0:
        return "", ""
    if n == 0:
        return "-" * m, seq2
    if m == 0:
        return seq1, "-" * n
    
    # --- DYNAMIC PROGRAMMING (DP) TABLES ---
    # M:  Score ending with a match/mismatch between seq1[i] and seq2[j].
    # Ix: Score ending with a gap in seq2 (deletion from seq1).
    # Iy: Score ending with a gap in seq1 (insertion into seq1).
    M = [[0] * (m + 1) for _ in range(n + 1)]
    Ix = [[0] * (m + 1) for _ in range(n + 1)]
    Iy = [[0] * (m + 1) for _ in range(n + 1)]
    
    # --- POINTER TABLES (for backtracking) ---
    # These tables store the path taken to achieve the optimal score at each cell.
    Ptr_M = [[None] * (m + 1) for _ in range(n + 1)]
    Ptr_Ix = [[None] * (m + 1) for _ in range(n + 1)]
    Ptr_Iy = [[None] * (m + 1) for _ in range(n + 1)]

    # --- INITIALIZATION ---
    # Initialize the first row and column. A match (M) is impossible if one sequence is empty.
    for i in range(1, n + 1):
        M[i][0], Iy[i][0] = -float('inf'), -float('inf')
        Ix[i][0] = gap_open + (i - 1) * gap_extend # Cost of a gap of length i in seq2.
    for j in range(1, m + 1):
        M[0][j], Ix[0][j] = -float('inf'), -float('inf')
        Iy[0][j] = gap_open + (j - 1) * gap_extend # Cost of a gap of length j in seq1.

    # --- FILL DP TABLES ---
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match_score if seq1[i-1] == seq2[j-1] else mismatch_score
            
            # Calculate score for M(i,j): coming from a previous match, insertion, or deletion.
            m_scores = { 'M': M[i-1][j-1], 'Ix': Ix[i-1][j-1], 'Iy': Iy[i-1][j-1] }
            best_prev_matrix = max(m_scores, key=m_scores.get)
            M[i][j] = score + m_scores[best_prev_matrix]
            Ptr_M[i][j] = best_prev_matrix

            # Calculate score for Ix(i,j): either opening a new gap or extending an old one.
            score_open_x = M[i-1][j] + gap_open
            score_extend_x = Ix[i-1][j] + gap_extend
            if score_open_x > score_extend_x:
                Ix[i][j] = score_open_x
                Ptr_Ix[i][j] = 'M' # Came from the Match matrix.
            else:
                Ix[i][j] = score_extend_x
                Ptr_Ix[i][j] = 'Ix' # Came from the Ix matrix (extending gap).

            # Calculate score for Iy(i,j): either opening a new gap or extending an old one.
            score_open_y = M[i][j-1] + gap_open
            score_extend_y = Iy[i][j-1] + gap_extend
            if score_open_y > score_extend_y:
                Iy[i][j] = score_open_y
                Ptr_Iy[i][j] = 'M' # Came from the Match matrix.
            else:
                Iy[i][j] = score_extend_y
                Ptr_Iy[i][j] = 'Iy' # Came from the Iy matrix (extending gap).

    # --- BACKTRACKING ---
    # Start from the bottom-right corner and trace back to the top-left using the pointers.
    align1, align2 = "", ""
    i, j = n, m
    
    # Determine which matrix (M, Ix, or Iy) had the highest score in the last cell.
    last_scores = {'M': M[i][j], 'Ix': Ix[i][j], 'Iy': Iy[i][j]}
    matrix = max(last_scores, key=last_scores.get)

    while i > 0 or j > 0:
        if matrix == 'M': # A match or mismatch step.
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            matrix = Ptr_M[i][j] # Get the previous matrix from the pointer.
            i -= 1
            j -= 1
        elif matrix == 'Ix': # A deletion step (gap in seq2).
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            matrix = Ptr_Ix[i][j]
            i -= 1
        elif matrix == 'Iy': # An insertion step (gap in seq1).
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            matrix = Ptr_Iy[i][j]
            j -= 1
            
    return align1, align2

def verify_and_print_alignment(original_seq, mutated_seq, context_chars=20):
    """
    Aligns two sequences and prints a formatted, easy-to-read view of the differences,
    providing context around the mutated region.
    """
    print("\nAlignment Verification (Original vs. Mutated):")
    
    # Align the sequences to find the optimal alignment including gaps.
    aligned_orig, aligned_mut = needleman_wunsch_affine(original_seq, mutated_seq)
    
    # Find all indices where the aligned sequences differ.
    diff_indices = [i for i, (c1, c2) in enumerate(zip(aligned_orig, aligned_mut)) if c1 != c2]
            
    if not diff_indices:
        print("Sequences are identical. No mutations were effectively applied.")
        return

    # Determine a window around the differences to display for context.
    start_index = max(0, diff_indices[0] - context_chars)
    end_index = min(len(aligned_orig), diff_indices[-1] + context_chars + 1)

    # Extract the snippets of the alignment to be printed.
    orig_snippet = aligned_orig[start_index:end_index]
    mut_snippet = aligned_mut[start_index:end_index]
    
    # Create a "match bar" to visually highlight matches (|) and mismatches ( ).
    match_bar = "".join("|" if c1 == c2 else " " for c1, c2 in zip(orig_snippet, mut_snippet))

    # Print the final, formatted output.
    print(f"Displaying alignment from index {start_index} to {end_index}:")
    print(f"{'Original:':<10} {orig_snippet}")
    print(f"{'':<10} {match_bar}")
    print(f"{'Mutated:':<10} {mut_snippet}")


# =============================================================================
#  4. EXAMPLE USAGE (COMMENTED OUT)
# =============================================================================

# To run a test, you can uncomment the following lines:
# original_sequence = "APNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEY"
# start_pos = 698

# --- Test Case: A complex deletion-insertion and a failing substitution ---
# mutations = ['L747_M753delinsS','A755D']
# print(f"Applying mutations: {mutations}")
# mutated_sequence = mutate_sequence(original_sequence, mutations, start_pos)

# print(f"\nOriginal sequence length: {len(original_sequence)}")
# print(f"Mutated sequence length:  {len(mutated_sequence)}")

# verify_and_print_alignment(original_sequence, mutated_sequence)
