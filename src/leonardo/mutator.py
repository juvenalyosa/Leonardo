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


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "SEC": "U",
    "PYL": "O", "TER": "*",
}

# Position token: residue number (may be negative) plus an optional PDB insertion code (184A).
_POS = r"(-?\d+)([A-Z]?)"
_SNP_RE = re.compile(r"^([A-Z])" + _POS + r"([A-Z*])$")
_RANGE = r"([A-Z]?)" + _POS + r"(?:_([A-Z]?)" + _POS + r")?"
_DELINS_RE = re.compile("^" + _RANGE + r"DELINS([A-Z]+)$")
_DEL_RE = re.compile("^" + _RANGE + r"DEL$")
_DUP_RE = re.compile("^" + _RANGE + r"DUP$")
_INS_RE = re.compile(r"^([A-Z]?)" + _POS + r"_([A-Z]?)" + _POS + r"INS([A-Z]+)$")

# Three-letter forms, anchored on whole codes so an insertion code cannot merge with the
# next residue ("GLY184GLYS" is Gly 184G -> Lys, never Gly 184G -> Ser).
# Each entry: (regex, groups holding one code, group holding a run of codes or None).
_AA3 = "(?:" + "|".join(sorted(_THREE_TO_ONE)) + ")"
_POS3_RANGE = "(" + _AA3 + ")?" + _POS + "(?:_(" + _AA3 + ")?" + _POS + ")?"
_THREE_LETTER_FORMS = (
    (re.compile("^(" + _AA3 + ")" + _POS + "(" + _AA3 + ")$"), (1, 4), None),
    (re.compile("^" + _POS3_RANGE + "DELINS((?:" + _AA3 + ")+)$"), (1, 4), 7),
    (re.compile("^" + _POS3_RANGE + "(?:DEL|DUP)$"), (1, 4), None),
    (re.compile("^(" + _AA3 + ")?" + _POS + "_(" + _AA3 + ")?" + _POS + "INS((?:" + _AA3 + ")+)$"), (1, 4), 7),
)


def _three_to_one(txt):
    for rx, code_groups, run_group in _THREE_LETTER_FORMS:
        m = rx.match(txt)
        if not m:
            continue
        pieces, last = [], 0
        for g in sorted(set(code_groups) | ({run_group} if run_group else set())):
            if m.group(g) is None:
                continue
            start, end = m.span(g)
            codes = re.findall("...", m.group(g))
            pieces += [txt[last:start], "".join(_THREE_TO_ONE[c] for c in codes)]
            last = end
        return "".join(pieces) + txt[last:]
    return txt


def normalize_mutation_label(label):
    """Canonical one-letter form of a mutation label.

    Upper-cases, drops ``p.`` and HGVS parentheses, and converts three-letter
    labels (``p.Lys10Gly`` -> ``K10G``, ``Trp24Ter`` -> ``W24*``). Three-letter
    conversion only applies when the label starts with a three-letter code, so a
    one-letter insertion such as ``A10_G11insSER`` (S-E-R) is left alone.
    """
    txt = re.sub(r"\s+", "", str(label or "")).upper()
    if txt.startswith("P."):
        txt = txt[2:]
    if txt.startswith("(") and txt.endswith(")"):
        txt = txt[1:-1]
    if re.match(_AA3 + r"-?\d", txt):
        txt = _three_to_one(txt)
    return txt


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
        label = normalize_mutation_label(item)
        if label:
            labels.append(label)
    return labels


def _coerce_start_pos(start_pos):
    try:
        return int(start_pos)
    except Exception as exc:
        raise MutationError(f"Invalid sequence start position: {start_pos!r}") from exc


def position_key(number, icode=""):
    """Residue position key used for matching: ``"-3"``, ``"184"``, ``"184A"``."""
    return f"{int(number)}{str(icode or '').strip().upper()}"


def _label_to_key(label):
    txt = str(label).strip().upper() if label is not None else ""
    m = re.fullmatch(r"(-?\d+)([A-Z]?)", txt)
    return position_key(m.group(1), m.group(2)) if m else None


def parse_mutation_label(mutation):
    """Parse one mutation label into its parts, without a sequence.

    Returns a dict with ``kind`` (``sub``, ``stop``, ``del``, ``delins``, ``ins``,
    ``dup``), ``start``/``end`` position keys (``"184A"``, ``"-3"``), the expected
    residues ``start_aa``/``end_aa`` ('' when omitted) and ``insertion`` (new
    residues). Raises :class:`MutationError` when the label cannot be parsed.
    """
    txt = normalize_mutation_label(mutation)
    m = _SNP_RE.match(txt)
    if m:
        old_aa, num, icode, new_aa = m.groups()
        key = position_key(num, icode)
        return {"label": txt, "kind": "stop" if new_aa == "*" else "sub", "start": key, "end": key,
                "start_aa": old_aa, "end_aa": old_aa, "insertion": "" if new_aa == "*" else new_aa}
    for kind, rx in (("delins", _DELINS_RE), ("del", _DEL_RE), ("dup", _DUP_RE)):
        m = rx.match(txt)
        if m:
            g = m.groups()
            start = position_key(g[1], g[2])
            if g[4] is None:
                end, end_aa = start, g[0]
            else:
                end, end_aa = position_key(g[4], g[5]), g[3]
            return {"label": txt, "kind": kind, "start": start, "end": end, "start_aa": g[0],
                    "end_aa": end_aa, "insertion": g[6] if kind == "delins" else ""}
    m = _INS_RE.match(txt)
    if m:
        left_aa, lnum, licode, right_aa, rnum, ricode, insertion = m.groups()
        return {"label": txt, "kind": "ins", "start": position_key(lnum, licode),
                "end": position_key(rnum, ricode), "start_aa": left_aa, "end_aa": right_aa,
                "insertion": insertion}
    for word, name in (("DELINS", "delins"), ("DEL", "deletion"), ("INS", "insertion"), ("DUP", "duplication")):
        if word in txt:
            raise MutationError(f"Could not parse {name} mutation: '{txt}'.")
    raise MutationError(f"Unsupported mutation format: '{txt}'.")


class _ReferenceNumbering:
    """Map mutation positions to indices of the *reference* sequence.

    Positions follow HGVS semantics: every mutation in a list refers to the
    original sequence, never to the sequence left behind by an earlier indel.
    ``residue_numbers`` (one label per residue) supports structure numbering
    with gaps, negative numbers or insertion codes (``184A``); labels that repeat
    cannot be addressed.
    """

    def __init__(self, seq, start_pos=1, residue_numbers=None):
        self.seq = seq
        self.ambiguous = set()
        if residue_numbers is None:
            start_i = _coerce_start_pos(start_pos)
            self.index = {position_key(start_i + i): i for i in range(len(seq))}
            return
        labels = list(residue_numbers)
        if len(labels) != len(seq):
            raise MutationError(
                f"Residue numbering has {len(labels)} label(s) for a sequence of {len(seq)} residue(s)."
            )
        seen = {}
        for i, label in enumerate(labels):
            key = _label_to_key(label)
            if key is not None:
                seen.setdefault(key, []).append(i)
        self.index = {key: idx[0] for key, idx in seen.items() if len(idx) == 1}
        self.ambiguous = {key for key, idx in seen.items() if len(idx) > 1}

    def at(self, pos, mutation):
        if pos in self.ambiguous:
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
                f"Mutation {mutation} mismatch at {label} {pos}: "
                f"found '{self.seq[idx]}' but expected '{expected}'."
            )

    def span(self, start_pos, start_aa, end_pos, end_aa, mutation):
        try:
            i0 = self.at(start_pos, mutation)
            i1 = self.at(end_pos, mutation)
        except MutationError as exc:
            raise MutationError(f"Mutation {mutation} references missing/out-of-bounds position(s): {exc}") from exc
        if i1 < i0:
            raise MutationError(f"Mutation {mutation} has an inverted range ({start_pos}>{end_pos}).")
        self.check(i0, start_aa, start_pos, mutation, label="range start")
        self.check(i1, end_aa, end_pos, mutation, label="range end")
        return i0, i1


def _parse_mutation(mutation, numbering):
    """Return ``(i0, i1, replacement)``: replace reference indices ``i0..i1`` (inclusive) by
    ``replacement``. A pure insertion before index ``i`` is ``(i, i - 1, text)``."""
    p = parse_mutation_label(mutation)
    kind = p["kind"]
    if kind in ("sub", "stop"):
        idx = numbering.at(p["start"], mutation)
        numbering.check(idx, p["start_aa"], p["start"], mutation)
        if kind == "stop":
            # nonsense mutation: the protein ends before this residue
            return idx, len(numbering.seq) - 1, ""
        return idx, idx, p["insertion"]
    if kind == "ins":
        i_left = numbering.at(p["start"], mutation)
        i_right = numbering.at(p["end"], mutation)
        if i_right != i_left + 1:
            raise MutationError(
                f"Insertion {mutation} flanking residues {p['start']} and {p['end']} are not adjacent in the sequence."
            )
        numbering.check(i_left, p["start_aa"], p["start"], mutation, label="left flank")
        numbering.check(i_right, p["end_aa"], p["end"], mutation, label="right flank")
        return i_right, i_left, p["insertion"]
    i0, i1 = numbering.span(p["start"], p["start_aa"], p["end"], p["end_aa"], mutation)
    if kind == "dup":
        return i1 + 1, i1, numbering.seq[i0:i1 + 1]
    return i0, i1, p["insertion"]


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
                stops = sorted((s, m) for s, m in ((s1, m1), (s2, m2)) if m.endswith("*"))
                if stops:
                    stop = stops[0][1]                  # the earlier stop codon ends the protein
                    other = m2 if stop == m1 else m1
                    raise MutationError(f"Mutation {other} lies after the stop codon {stop}; the protein ends there.")
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
                          - Substitution: "A123G"; nonsense (stop): "W24*" / "W24Ter"
                          - Deletion: "A10del", "10_12del"
                          - Insertion: "A10_B11insGGG"
                          - Deletion-Insertion: "A10delinsGGG", "10_12delinsGGG"
                          - Duplication: "A10dup", "10_12dup"
                          Positions may be negative ("A-3G") or carry a PDB insertion
                          code ("G184AA" = Gly 184A -> Ala; needs ``residue_numbers``).
                          Three-letter labels are accepted ("p.Lys10Gly", "p.(Trp24Ter)").
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
