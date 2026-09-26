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


class _ReferenceNumbering:
    """Map mutation position numbers to indices of the *reference* sequence.

    Positions follow HGVS semantics: every mutation in a list refers to the
    original sequence, never to the sequence left behind by an earlier indel.
    ``residue_numbers`` (one label per residue) supports structure numbering
    with gaps or insertion codes; labels that are not plain integers (``184A``)
    or that repeat cannot be addressed by a numeric mutation label.
    """

    def __init__(self, seq, start_pos=1, residue_numbers=None):
        self.seq = seq
        self.index = {}
        if residue_numbers is None:
            start_i = _coerce_start_pos(start_pos)
            self.index = {start_i + i: i for i in range(len(seq))}
            return
        labels = list(residue_numbers)
        if len(labels) != len(seq):
            raise MutationError(
                f"Residue numbering has {len(labels)} label(s) for a sequence of {len(seq)} residue(s)."
            )
        seen = {}
        for i, label in enumerate(labels):
            txt = str(label).strip() if label is not None else ""
            if not re.fullmatch(r"-?\d+", txt):
                continue
            seen.setdefault(int(txt), []).append(i)
        self.index = {pos: idx[0] for pos, idx in seen.items() if len(idx) == 1}
        self.ambiguous = {pos for pos, idx in seen.items() if len(idx) > 1}

    def at(self, pos, mutation):
        pos = int(pos)
        if pos in getattr(self, "ambiguous", ()):
            raise MutationError(f"Mutation {mutation} references position {pos}, which is not unique in the numbering.")
        if pos not in self.index:
            raise MutationError(
                f"Mutation {mutation} references position {pos}, "
                "but that position is outside the current sequence."
            )
        return self.index[pos]

    def check(self, idx, expected_aa, pos, mutation, label="position"):
        expected = str(expected_aa or "").strip().upper()
        if expected and self.seq[idx] != expected:
            raise MutationError(
                f"Mutation {mutation} mismatch at {label} {int(pos)}: "
                f"found '{self.seq[idx]}' but expected '{expected}'."
            )

    def span(self, start_pos, start_aa, end_pos, end_aa, mutation):
        if int(end_pos) < int(start_pos):
            raise MutationError(f"Mutation {mutation} has an inverted range ({int(start_pos)}>{int(end_pos)}).")
        try:
            i0 = self.at(start_pos, mutation)
            i1 = self.at(end_pos, mutation)
        except MutationError as exc:
            raise MutationError(f"Mutation {mutation} references missing/out-of-bounds position(s): {exc}") from exc
        if i1 < i0:
            raise MutationError(f"Mutation {mutation} has an inverted range in the sequence numbering.")
        self.check(i0, start_aa, start_pos, mutation, label="range start")
        self.check(i1, end_aa, end_pos, mutation, label="range end")
        return i0, i1


def _parse_mutation(mutation, numbering):
    """Return ``(i0, i1, replacement)``: replace reference indices ``i0..i1`` (inclusive) by
    ``replacement``. A pure insertion before index ``i`` is ``(i, i - 1, text)``."""
    snp = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation)
    if snp:
        old_aa, pos, new_aa = snp.groups()
        idx = numbering.at(pos, mutation)
        numbering.check(idx, old_aa, pos, mutation)
        return idx, idx, new_aa

    if 'DELINS' in mutation:
        m = re.match(r'^([A-Z]?)(\d+)(?:_([A-Z]?)(\d+))?DELINS([A-Z]+)$', mutation)
        if not m:
            raise MutationError(f"Could not parse delins mutation: '{mutation}'.")
        start_aa, start, end_aa, end, insertion = m.groups()
        if end is None:
            end, end_aa = start, start_aa
        i0, i1 = numbering.span(start, start_aa, end, end_aa, mutation)
        return i0, i1, insertion

    if 'DEL' in mutation:
        m = re.match(r'^([A-Z]?)(\d+)(?:_([A-Z]?)(\d+))?DEL$', mutation)
        if not m:
            raise MutationError(f"Could not parse deletion mutation: '{mutation}'.")
        start_aa, start, end_aa, end = m.groups()
        if end is None:
            end, end_aa = start, start_aa
        i0, i1 = numbering.span(start, start_aa, end, end_aa, mutation)
        return i0, i1, ""

    if 'INS' in mutation:
        m = re.match(r'^([A-Z]?)(\d+)_([A-Z]?)(\d+)INS([A-Z]+)$', mutation)
        if not m:
            raise MutationError(f"Could not parse insertion mutation: '{mutation}'.")
        left_aa, left, right_aa, right, insertion = m.groups()
        if int(right) != int(left) + 1:
            raise MutationError(
                f"Insertion {mutation} must reference adjacent flanking residues "
                f"({int(left)} and {int(left) + 1})."
            )
        i_left = numbering.at(left, mutation)
        i_right = numbering.at(right, mutation)
        if i_right != i_left + 1:
            raise MutationError(f"Insertion {mutation} flanking residues are not adjacent in the sequence.")
        numbering.check(i_left, left_aa, left, mutation, label="left flank")
        numbering.check(i_right, right_aa, right, mutation, label="right flank")
        return i_right, i_left, insertion

    if 'DUP' in mutation:
        m = re.match(r'^([A-Z]?)(\d+)(?:_([A-Z]?)(\d+))?DUP$', mutation)
        if not m:
            raise MutationError(f"Could not parse duplication mutation: '{mutation}'.")
        start_aa, start, end_aa, end = m.groups()
        if end is None:
            end, end_aa = start, start_aa
        i0, i1 = numbering.span(start, start_aa, end, end_aa, mutation)
        return i1 + 1, i1, numbering.seq[i0:i1 + 1]

    raise MutationError(f"Unsupported mutation format: '{mutation}'.")


def _check_no_overlap(edits):
    """Edits must touch disjoint parts of the reference; otherwise their combined effect is ambiguous."""
    for a in range(len(edits)):
        for b in range(a + 1, len(edits)):
            (s1, e1, _r1, m1), (s2, e2, _r2, m2) = edits[a], edits[b]
            ins1, ins2 = e1 < s1, e2 < s2
            if ins1 and ins2:
                clash = s1 == s2
            elif ins1:
                clash = s2 < s1 <= e2          # insertion point strictly inside the replaced range
            elif ins2:
                clash = s1 < s2 <= e1
            else:
                clash = s1 <= e2 and s2 <= e1
            if clash:
                raise MutationError(
                    f"Mutations {m1} and {m2} affect overlapping residues; combine them into one delins."
                )

# =============================================================================
#  1. HELPER FUNCTIONS FOR SEQUENCE MANIPULATION
# =============================================================================

def convert_to_dict(sequence, start_pos):
    """
    Converts a sequence string to a dictionary mapping position numbers to amino acids.

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

def mutate_sequence(sequence, mutations, start_pos=1, residue_numbers=None):
    """
    Applies a list of mutations to a protein sequence.

    Every position refers to the original (reference) sequence, as in HGVS:
    ``["K2_T3del", "A7G"]`` mutates the residue numbered 7 in the input even
    though the deletion shifts it. Mutations that touch overlapping residues
    are rejected instead of being applied in an order-dependent way.

    Args:
        sequence (str): The original amino acid sequence.
        mutations (list): A list of mutations in a simplified HGVS-like format.
                          Supported formats:
                          - Substitution: "A123G"
                          - Deletion: "A10del", "10_12del"
                          - Insertion: "A10_B11insGGG"
                          - Deletion-Insertion: "A10delinsGGG", "10_12delinsGGG"
                          - Duplication: "A10dup", "10_12dup"
        start_pos (int): The number of the first residue (e.g., 1 for 1-based indexing).
        residue_numbers (list, optional): One position label per residue (e.g. PDB numbering
                          with gaps or insertion codes); overrides ``start_pos``.

    Returns:
        str: The mutated sequence.
    """
    seq = _clean_protein_sequence(sequence)
    numbering = _ReferenceNumbering(seq, start_pos, residue_numbers)
    edits = []
    for mutation in _iter_mutation_labels(mutations):
        i0, i1, replacement = _parse_mutation(mutation, numbering)
        edits.append((i0, i1, replacement, mutation))
    _check_no_overlap(edits)

    out = []
    cursor = 0
    # insertions before index i come before a replacement that starts at i
    for i0, i1, replacement, _m in sorted(edits, key=lambda e: (e[0], e[1] >= e[0])):
        out.append(seq[cursor:i0])
        out.append(replacement)
        cursor = max(cursor, i1 + 1)
    out.append(seq[cursor:])
    return "".join(out)

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
        Ptr_Ix[i][0] = 'Ix' # A leading gap can only continue along the first column.
    for j in range(1, m + 1):
        M[0][j], Ix[0][j] = -float('inf'), -float('inf')
        Iy[0][j] = gap_open + (j - 1) * gap_extend # Cost of a gap of length j in seq1.
        Ptr_Iy[0][j] = 'Iy'

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
        # On the first row/column only one move remains, whatever the pointer says.
        if i == 0:
            matrix = 'Iy'
        elif j == 0:
            matrix = 'Ix'
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
        else:
            raise RuntimeError(f"Alignment backtracking lost its path at ({i}, {j}).")

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
