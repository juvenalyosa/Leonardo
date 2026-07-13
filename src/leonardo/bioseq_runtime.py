import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter


AA_STANDARD_TEXT = "ACDEFGHIKLMNPQRSTVWY"
AA_STANDARD = set(AA_STANDARD_TEXT)
# IUPAC protein sequence inputs often contain ambiguous residues (B/Z/J/X) and
# rare encoded residues (U/O).  Keep them explicit instead of collapsing to X so
# users can see the uncertainty and downstream reports can avoid overcalling
# exact motifs on ambiguous positions.
AA_AMBIGUOUS = set("XBZJ")
AA_RARE = set("UO")
AA_WITH_UNKNOWN = AA_STANDARD | AA_AMBIGUOUS | AA_RARE
AA_CHARGED = set("DEKRH")
AA_HYDROPHOBIC = set("AVILMFWYCT")
AA_POLAR = set("STNQY")
AA_AROMATIC = set("FWYH")
AA_DISORDER_PROMOTING = set("ARGQSPEDK")
AA_NONPROLINE_TEXT = "".join(a for a in AA_STANDARD_TEXT if a != "P")

KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
    "X": -0.5,
    # Ambiguous/rare residues use conservative approximations for local
    # heuristics.  Exact tools should be run externally for final annotation.
    "B": -3.5,  # Asp/Asn-like
    "Z": -3.5,  # Glu/Gln-like
    "J": 4.15,  # Ile/Leu-like
    "U": 2.5,  # Selenocysteine approximated as cysteine
    "O": -3.9,  # Pyrrolysine approximated as lysine-like
}


BIOSEQ_ANALYSIS_CATALOG = [
    {
        "id": "homologs",
        "title": "A) Search homologs",
        "tools": "BLAST, HMMER",
        "purpose": "Find structural templates and evolutionary relatives.",
    },
    {
        "id": "msa",
        "title": "B) Build MSA",
        "tools": "Clustal Omega, MAFFT",
        "purpose": "Estimate residue conservation and functionally constrained positions.",
    },
    {
        "id": "domains",
        "title": "C) Detect domains",
        "tools": "InterPro, Pfam/CDD",
        "purpose": "Choose which region(s) to model and simulate.",
    },
    {
        "id": "disorder",
        "title": "D) Detect disorder",
        "tools": "IUPred",
        "purpose": "Avoid unstable/fuzzy tails unless biologically required.",
    },
    {
        "id": "transmembrane",
        "title": "E) Detect transmembrane and signal peptide",
        "tools": "DeepTMHMM/TMHMM, SignalP",
        "purpose": "Decide whether membrane/secretory setup is needed.",
    },
    {
        "id": "motifs",
        "title": "F) Review functional motifs",
        "tools": "PROSITE, InterPro",
        "purpose": "Preserve catalytic/functional residues and motif neighborhoods.",
    },
]


BIOSEQ_QUESTION_MAP = [
    {
        "question": "How similar is my protein to known proteins?",
        "analysis": "BLAST, HMMER",
    },
    {
        "question": "Which residues are likely important?",
        "analysis": "MSA conservation, motif scan",
    },
    {
        "question": "Which region should be simulated?",
        "analysis": "Domain detection + disorder prediction",
    },
    {
        "question": "Is it soluble or membrane-associated?",
        "analysis": "TMHMM/DeepTMHMM + SignalP",
    },
    {
        "question": "Does it contain flexible/problematic regions?",
        "analysis": "IUPred + low-complexity/conservation context",
    },
    {
        "question": "Does it have conserved functional sites?",
        "analysis": "PROSITE + InterPro + conservation",
    },
    {
        "question": "Could a mutation be important?",
        "analysis": "Residue conservation + motif/domain context",
    },
    {
        "question": "Is there coevolution evidence for internal contacts?",
        "analysis": "Coevolution tools (EVcouplings family)",
    },
]


TOOL_URLS = {
    "blast": "https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE=Proteins",
    "hmmer": "https://www.ebi.ac.uk/Tools/hmmer/search/phmmer",
    "msa": "https://www.ebi.ac.uk/jdispatcher/msa/clustalo",
    "interpro": "https://www.ebi.ac.uk/interpro/search/sequence/",
    "iupred": "https://iupred2a.elte.hu/",
    "tmhmm": "https://services.healthtech.dtu.dk/services/DeepTMHMM-1.0/",
    "signalp": "https://services.healthtech.dtu.dk/services/SignalP-6.0/",
    "prosite": "https://prosite.expasy.org/scanprosite/",
    "coevolution": "https://evcouplings.org/",
}

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FEATURE_FIELDS = (
    "accession,id,protein_name,organism_name,"
    "ft_motif,ft_domain,ft_region,ft_binding,ft_act_site,ft_site,sequence"
)
UNIPROT_FUNCTIONAL_FEATURE_TYPES = {
    "Motif",
    "Domain",
    "Region",
    "Active site",
    "Binding site",
    "Site",
}


def _is_fasta_metadata_line(line):
    stripped = str(line or "").strip()
    return bool(stripped.startswith((">", ";", "#")))


def _strip_inline_sequence_comment(line):
    text = str(line or "")
    limit = len(text)
    for marker in ("#", ";"):
        idx = text.find(marker)
        if idx >= 0 and (idx == 0 or text[idx - 1].isspace()):
            limit = min(limit, idx)
    return text[:limit]


def _iter_sequence_body_lines(text):
    for raw in str(text or "").splitlines():
        if not str(raw).strip():
            continue
        if _is_fasta_metadata_line(raw):
            continue
        body = _strip_inline_sequence_comment(raw)
        if str(body).strip():
            yield body


def _clean_sequence_chars(text, keep_gaps=False):
    out = []
    for ch in str(text or "").upper():
        if ch.isalpha():
            if ch in AA_STANDARD:
                out.append(ch)
            elif ch in AA_WITH_UNKNOWN:
                out.append(ch)
            else:
                continue
        elif keep_gaps and ch in "-.":
            out.append("-")
    return "".join(out)


def clean_sequence(text):
    if not str(text or "").strip():
        return ""
    return _clean_sequence_chars("".join(_iter_sequence_body_lines(text)), keep_gaps=False)


def sequence_text_residue_offsets(text, keep_gaps=False):
    """
    Return character offsets for residues retained by the normalizer.

    The UI uses these offsets to highlight sequence regions.  FASTA headers and
    comment lines must not count as residues, otherwise a motif/coverage hit can
    be shifted when users paste raw FASTA text.
    """
    raw = str(text or "")
    offsets = []
    base = 0
    for line in raw.splitlines(keepends=True):
        body = str(line).rstrip("\r\n")
        stripped = body.strip()
        if not stripped or _is_fasta_metadata_line(stripped):
            base += len(line)
            continue
        limit = len(body)
        for marker in ("#", ";"):
            idx = body.find(marker)
            if idx >= 0 and (idx == 0 or body[idx - 1].isspace()):
                limit = min(limit, idx)
        for rel, ch in enumerate(body[:limit]):
            if _clean_sequence_chars(ch, keep_gaps=bool(keep_gaps)):
                offsets.append(int(base + rel))
        base += len(line)
    return offsets


def parse_fasta(text, keep_gaps=False):
    raw = str(text or "")
    if not raw.strip():
        return []
    entries = []
    header = ""
    seq_lines = []
    for line in raw.splitlines():
        ln = str(line).strip()
        if not ln:
            continue
        if ln.startswith(">"):
            if seq_lines:
                seq = _clean_sequence_chars("".join(seq_lines), keep_gaps=bool(keep_gaps))
                if seq:
                    entries.append({"header": header or f"sequence_{len(entries)+1}", "sequence": seq})
            header = ln[1:].strip() or f"sequence_{len(entries)+1}"
            seq_lines = []
            continue
        if ln.startswith((";", "#")):
            continue
        seq_lines.append(_strip_inline_sequence_comment(ln))
    if seq_lines:
        seq = _clean_sequence_chars("".join(seq_lines), keep_gaps=bool(keep_gaps))
        if seq:
            entries.append({"header": header or f"sequence_{len(entries)+1}", "sequence": seq})
    if entries:
        return entries
    seq = _clean_sequence_chars("".join(_iter_sequence_body_lines(raw)), keep_gaps=bool(keep_gaps))
    if seq:
        return [{"header": "sequence_1", "sequence": seq}]
    return []


def sequence_basic_metrics(sequence):
    seq = clean_sequence(sequence)
    if not seq:
        return {"ok": False, "length": 0}
    cnt = Counter(seq)
    n = float(len(seq))
    frac = lambda aa_set: float(sum(int(cnt.get(a, 0)) for a in aa_set)) / n
    net_q = float(cnt.get("K", 0) + cnt.get("R", 0) + 0.1 * cnt.get("H", 0) - cnt.get("D", 0) - cnt.get("E", 0))
    ambiguous_count = int(sum(int(cnt.get(a, 0)) for a in AA_AMBIGUOUS))
    rare_count = int(sum(int(cnt.get(a, 0)) for a in AA_RARE))
    return {
        "ok": True,
        "length": int(len(seq)),
        "counts": dict(cnt),
        "fraction_hydrophobic": float(frac(AA_HYDROPHOBIC)),
        "fraction_polar": float(frac(AA_POLAR)),
        "fraction_charged": float(frac(AA_CHARGED)),
        "fraction_aromatic": float(frac(AA_AROMATIC)),
        "fraction_disorder_promoting": float(frac(AA_DISORDER_PROMOTING)),
        "ambiguous_count": int(ambiguous_count),
        "rare_count": int(rare_count),
        "fraction_unknown_or_ambiguous": float(ambiguous_count / n),
        "fraction_rare": float(rare_count / n),
        "fraction_nonstandard": float((ambiguous_count + rare_count) / n),
        "estimated_net_charge_ph7": float(net_q),
    }


def _window_average(values, window):
    arr = list(values or [])
    w = int(max(1, window))
    if len(arr) < w:
        return []
    out = []
    run = sum(float(x) for x in arr[:w])
    out.append(float(run) / float(w))
    for i in range(w, len(arr)):
        run += float(arr[i]) - float(arr[i - w])
        out.append(float(run) / float(w))
    return out


def _hydropathy(aa):
    return float(KYTE_DOOLITTLE.get(str(aa or "X").upper(), KYTE_DOOLITTLE["X"]))


def _merge_boolean_mask(mask, min_len=1):
    out = []
    if not mask:
        return out
    i = 0
    n = len(mask)
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask[j + 1]):
            j += 1
        start = int(i + 1)
        end = int(j + 1)
        if (end - start + 1) >= int(max(1, min_len)):
            out.append((start, end))
        i = j + 1
    return out


def predict_tm_segments(sequence, window=19, threshold=1.6, min_len=14):
    seq = clean_sequence(sequence)
    if not seq:
        return {"ok": False, "segments": []}
    vals = [_hydropathy(aa) for aa in seq]
    win = int(max(7, window))
    if len(seq) < win:
        # Short fragments should degrade gracefully: only evaluate them when
        # they are at least long enough to satisfy the requested segment length.
        if len(seq) >= int(max(7, min_len)):
            win = int(len(seq))
        else:
            return {"ok": True, "segments": []}
    avgs = _window_average(vals, win)
    if not avgs:
        return {"ok": True, "segments": []}
    mask = [False] * len(seq)
    for i, score in enumerate(avgs):
        if float(score) < float(threshold):
            continue
        a = int(i)
        b = int(i + win - 1)
        for p in range(a, b + 1):
            if 0 <= p < len(mask):
                mask[p] = True
    segs = []
    for start, end in _merge_boolean_mask(mask, min_len=int(max(7, min_len))):
        s0 = int(start - 1)
        e0 = int(end - 1)
        best = -999.0
        for wi, score in enumerate(avgs):
            ws = int(wi)
            we = int(wi + win - 1)
            if we < s0 or ws > e0:
                continue
            if float(score) > float(best):
                best = float(score)
        segs.append({"start": int(start), "end": int(end), "hydrophobicity": float(best)})
    return {"ok": True, "segments": segs}


def predict_disorder_segments(sequence, window=21, threshold=0.52, min_len=18):
    seq = clean_sequence(sequence)
    if not seq:
        return {"ok": False, "segments": [], "fraction": 0.0}
    n = len(seq)
    w = int(max(9, window))
    if n < w:
        if n >= int(max(8, min_len)):
            w = int(n)
        else:
            return {"ok": True, "segments": [], "fraction": 0.0}
    mask = [False] * n
    for i in range(0, n - w + 1):
        chunk = seq[i : i + w]
        kds = [_hydropathy(a) for a in chunk]
        hydro = float(sum(kds) / float(len(kds)))
        frac_dis = float(sum(1 for a in chunk if a in AA_DISORDER_PROMOTING)) / float(len(chunk))
        frac_charge = float(sum(1 for a in chunk if a in AA_CHARGED)) / float(len(chunk))
        hydro_penalty = max(0.0, -hydro / 3.5)
        score = 0.55 * frac_dis + 0.25 * frac_charge + 0.20 * hydro_penalty
        if score < float(threshold):
            continue
        for p in range(i, i + w):
            mask[p] = True
    segs = [{"start": a, "end": b} for a, b in _merge_boolean_mask(mask, min_len=int(max(8, min_len)))]
    frac = float(sum(1 for v in mask if bool(v))) / float(max(1, n))
    return {"ok": True, "segments": segs, "fraction": float(frac)}


def predict_signal_peptide(sequence):
    seq = clean_sequence(sequence)
    if not seq:
        return {"ok": False, "predicted": False}
    nterm = seq[:35]
    if len(nterm) < 16:
        return {"ok": True, "predicted": False}
    pos_n = int(sum(1 for a in nterm[:5] if a in {"K", "R"}))
    vals = [_hydropathy(a) for a in nterm]
    avgs = _window_average(vals, 8)
    max_h = max(avgs) if avgs else -999.0
    max_i = int(avgs.index(max_h)) if avgs else 0
    h_start = int(max_i + 1)
    h_end = int(max_i + 8)
    predicted = bool(pos_n >= 1 and max_h >= 1.2 and h_start <= 15 and h_end >= 14)
    cleavage = int(min(33, max(15, h_end + 2))) if predicted else 0
    return {
        "ok": True,
        "predicted": bool(predicted),
        "n_region_positive": int(pos_n),
        "h_region_start": int(h_start),
        "h_region_end": int(h_end),
        "h_region_hydrophobicity": float(max_h),
        "cleavage_site_estimate": int(cleavage),
    }


def find_simple_motifs(sequence, max_hits=80):
    seq = clean_sequence(sequence)
    if not seq:
        return []
    motifs = [
        ("N-glycosylation motif", f"N[{AA_NONPROLINE_TEXT}][ST][{AA_NONPROLINE_TEXT}]", True),
        ("P-loop NTP-binding-like motif", f"[AG][{AA_STANDARD_TEXT}]{{4}}GK[ST]", True),
        ("Casein kinase II phospho-like motif", f"[ST][{AA_STANDARD_TEXT}]{{2}}[DE]", True),
        ("PKC phospho-like motif", f"[ST][{AA_STANDARD_TEXT}][RK]", True),
        ("RGD motif", r"RGD", True),
        ("Lys/Arg basic cluster", r"[KR]{4,}", False),
    ]
    out = []
    limit = int(max(1, max_hits))
    for name, pat, allow_overlap in motifs:
        pattern = f"(?=({pat}))" if bool(allow_overlap) else str(pat)
        for m in re.finditer(pattern, seq):
            match = str(m.group(1) if bool(allow_overlap) else m.group(0))
            start = int((m.start(1) if bool(allow_overlap) else m.start()) + 1)
            end = int(start + len(match) - 1)
            out.append(
                {
                    "motif": str(name),
                    "start": int(start),
                    "end": int(end),
                    "match": str(match),
                }
            )
            if len(out) >= limit:
                return out
    return out


def _format_residue_range(start, end, prefix="local"):
    try:
        a = int(start or 0)
        b = int(end or 0)
    except Exception:
        return f"{prefix} ?"
    if a <= 0 or b <= 0:
        return f"{prefix} ?"
    if b < a:
        a, b = b, a
    return f"{prefix} {a}" if a == b else f"{prefix} {a}-{b}"


def _sequence_segment(sequence, start, end, max_len=48):
    seq = clean_sequence(sequence)
    try:
        a = int(start or 0)
        b = int(end or 0)
    except Exception:
        return ""
    if a <= 0 or b <= 0 or not seq:
        return ""
    if b < a:
        a, b = b, a
    if a > len(seq):
        return ""
    frag = seq[a - 1 : min(b, len(seq))]
    limit = int(max(8, max_len))
    if len(frag) > limit:
        head = max(4, limit - 12)
        frag = f"{frag[:head]}...{frag[-6:]}"
    return str(frag)


def _numbered_residue_list(sequence, start, end, max_items=18):
    seq = clean_sequence(sequence)
    try:
        a = int(start or 0)
        b = int(end or 0)
    except Exception:
        return ""
    if a <= 0 or b <= 0 or not seq:
        return ""
    if b < a:
        a, b = b, a
    b = min(int(b), len(seq))
    if a > b:
        return ""
    items = [f"{seq[i - 1]}{i}" for i in range(a, b + 1)]
    limit = int(max(4, max_items))
    if len(items) > limit:
        keep_head = max(2, limit - 4)
        items = items[:keep_head] + ["..."] + items[-3:]
    return ", ".join(items)


def format_local_motif_summary_line(motif, sequence=""):
    row = dict(motif or {})
    name = str(row.get("motif", "Motif") or "Motif").strip()
    try:
        start = int(row.get("start", 0) or 0)
        end = int(row.get("end", 0) or 0)
    except Exception:
        start = 0
        end = 0
    match = str(row.get("match", "") or "").strip()
    if not match:
        match = _sequence_segment(sequence, start, end, max_len=80)
    residues = _numbered_residue_list(sequence or match, start if sequence else 1, end if sequence else len(match))
    loc = _format_residue_range(start, end, prefix="local")
    detail = f"{name} | {loc}"
    if match:
        detail += f" | amino acids: {match}"
    if residues:
        detail += f" | residues: {residues}"
    return detail


def format_uniprot_feature_summary_line(feature, sequence=""):
    row = dict(feature or {})
    ftype = str(row.get("type", "Feature") or "Feature").strip()
    desc = str(row.get("description", "") or "").strip()
    label = f"{ftype}: {desc}" if desc else ftype
    try:
        start = int(row.get("start", 0) or 0)
        end = int(row.get("end", 0) or 0)
    except Exception:
        start = 0
        end = 0
    try:
        qstart = int(row.get("query_start", 0) or 0)
        qend = int(row.get("query_end", 0) or 0)
    except Exception:
        qstart = 0
        qend = 0
    locs = [_format_residue_range(start, end, prefix="UniProt")]
    if qstart > 0 and qend > 0:
        locs.append(_format_residue_range(qstart, qend, prefix="local"))
    segment = _sequence_segment(sequence, qstart, qend) if qstart > 0 and qend > 0 else ""
    residues = _numbered_residue_list(sequence, qstart, qend) if qstart > 0 and qend > 0 else ""
    detail = f"{label} | {'; '.join(locs)}"
    if segment:
        detail += f" | amino acids: {segment}"
    if residues:
        detail += f" | residues: {residues}"
    return detail


def build_motif_summary_lines(sequence, local_motifs=None, uniprot_summary=None, max_items=80):
    """
    Build a concise, UI-ready motif summary.

    Keeping this formatter in the runtime makes the popup deterministic and
    testable while the Qt layer only decides where to display the lines.
    """
    seq = clean_sequence(sequence)
    motifs = list(local_motifs if local_motifs is not None else find_simple_motifs(seq))
    uniprot = dict(uniprot_summary or {})
    features = list(uniprot.get("features", []) or []) if bool(uniprot.get("ok", False)) else []
    lines = ["Motif and Functional-Site Summary", f"Sequence length: {len(seq)} aa", ""]
    lines.append(f"Local pattern motifs: {len(motifs)}")
    if motifs:
        for m in motifs[: int(max(1, max_items))]:
            lines.append("- " + format_local_motif_summary_line(m, seq))
    else:
        lines.append("- No local motif patterns detected.")
    lines.append("")
    lines.append(f"UniProt curated annotations: {len(features)}")
    if bool(uniprot.get("ok", False)):
        src = " ".join(
            x
            for x in (
                str(uniprot.get("accession", "") or "").strip(),
                str(uniprot.get("entry_id", "") or "").strip(),
            )
            if x
        )
        protein = str(uniprot.get("protein_name", "") or "").strip()
        organism = str(uniprot.get("organism", "") or "").strip()
        header_bits = [x for x in (src, protein, organism) if x]
        if header_bits:
            lines.append("Source: " + " | ".join(header_bits))
        if features:
            for f in features[: int(max(1, max_items))]:
                lines.append("- " + format_uniprot_feature_summary_line(f, seq))
        else:
            lines.append("- No UniProt motif/domain/site annotations overlapped this sequence.")
    else:
        err = str(uniprot.get("error", "UniProt lookup was not available.") or "UniProt lookup was not available.")
        lines.append("- " + err)
    return lines


def _terminal_disorder_lengths(disorder_segments, seq_len):
    n_term = 0
    c_term = 0
    for seg in list(disorder_segments or []):
        a = int(seg.get("start", 0) or 0)
        b = int(seg.get("end", 0) or 0)
        if a <= 2:
            n_term = max(n_term, int(max(0, b)))
        if b >= int(seq_len - 1):
            c_term = max(c_term, int(max(0, int(seq_len) - a + 1)))
    return int(n_term), int(c_term)


def build_simulation_readiness_report(sequence):
    seq = clean_sequence(sequence)
    if not seq:
        return {"ok": False, "error": "Sequence is empty."}
    metrics = sequence_basic_metrics(seq)
    tm = predict_tm_segments(seq)
    dis = predict_disorder_segments(seq)
    sig = predict_signal_peptide(seq)
    motifs = find_simple_motifs(seq)
    tm_segments = list(tm.get("segments", []) or [])
    dis_segments = list(dis.get("segments", []) or [])
    dis_frac = float(dis.get("fraction", 0.0) or 0.0)
    n_term_dis, c_term_dis = _terminal_disorder_lengths(dis_segments, len(seq))
    core_start = 1 + n_term_dis if n_term_dis >= 12 else 1
    core_end = len(seq) - c_term_dis if c_term_dis >= 12 else len(seq)
    if core_end < core_start:
        core_start = 1
        core_end = len(seq)
    recommendations = []
    if tm_segments:
        recommendations.append("Predicted transmembrane region(s): prepare membrane system for MD.")
    else:
        recommendations.append("No strong transmembrane signal: soluble setup is likely appropriate.")
    if bool(sig.get("predicted", False)):
        recommendations.append(
            "Signal peptide-like N-terminus detected; consider simulating mature chain without signal peptide."
        )
    if dis_frac >= 0.30:
        recommendations.append("High disorder fraction: trim long disordered tails or model them separately.")
    elif dis_frac >= 0.15:
        recommendations.append("Moderate disorder: inspect flexible termini and loops before production MD.")
    else:
        recommendations.append("Low-to-moderate disorder: full-length simulation is more likely to be stable.")
    if len(seq) > 1000:
        recommendations.append("Very long sequence: split by domains and simulate prioritized regions first.")
    recommendations.append("Run InterPro/Pfam domain annotation before final model selection.")
    return {
        "ok": True,
        "sequence": str(seq),
        "length": int(len(seq)),
        "metrics": dict(metrics),
        "tm_segments": tm_segments,
        "disorder_segments": dis_segments,
        "disorder_fraction": float(dis_frac),
        "signal_peptide": dict(sig),
        "motifs": motifs,
        "suggested_structured_core": {"start": int(core_start), "end": int(core_end)},
        "recommendations": list(recommendations),
    }


def compute_msa_conservation(fasta_text):
    entries = parse_fasta(fasta_text, keep_gaps=True)
    if len(entries) < 2:
        return {"ok": False, "error": "Provide at least two aligned sequences in FASTA format."}
    seqs = [str(e.get("sequence", "") or "") for e in entries]
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        return {"ok": False, "error": "All MSA sequences must have the same aligned length."}
    aln_len = int(next(iter(lengths)))
    if aln_len <= 0:
        return {"ok": False, "error": "Alignment is empty."}
    if not any(any(ch != "-" for ch in seq) for seq in seqs):
        return {"ok": False, "error": "Alignment contains no residue observations."}
    nseq = int(len(seqs))
    reference_positions = []
    ref_pos = 0
    for ch in seqs[0]:
        if ch == "-":
            reference_positions.append(None)
        else:
            ref_pos += 1
            reference_positions.append(int(ref_pos))
    cols = []
    for i in range(aln_len):
        col = [s[i] for s in seqs]
        nongap = [c for c in col if c != "-"]
        cov = float(len(nongap)) / float(max(1, nseq))
        ref_position = reference_positions[i] if i < len(reference_positions) else None
        if not nongap:
            cols.append(
                {
                    "position": int(i + 1),
                    "alignment_position": int(i + 1),
                    "reference_position": ref_position,
                    "consensus": "-",
                    "conservation": 0.0,
                    "entropy": 0.0,
                    "coverage": float(cov),
                    "gap_fraction": 1.0,
                    "n_observed": 0,
                    "counts": {},
                }
            )
            continue
        cnt = Counter(nongap)
        top_aa, top_n = cnt.most_common(1)[0]
        cons = float(top_n) / float(len(nongap))
        probs = [float(v) / float(len(nongap)) for v in cnt.values()]
        ent = -sum(float(p) * math.log(float(p), 2) for p in probs if p > 1.0e-12)
        cols.append(
            {
                "position": int(i + 1),
                "alignment_position": int(i + 1),
                "reference_position": ref_position,
                "consensus": str(top_aa),
                "conservation": float(cons),
                "entropy": float(ent),
                "coverage": float(cov),
                "gap_fraction": float(1.0 - cov),
                "n_observed": int(len(nongap)),
                "counts": dict(cnt),
            }
        )
    informative_cols = [c for c in cols if int(c.get("n_observed", 0) or 0) > 0]
    mean_cons = float(
        sum(float(c["conservation"]) for c in informative_cols) / float(max(1, len(informative_cols)))
    )
    top = sorted(
        [c for c in cols if float(c["coverage"]) >= 0.70 and int(c.get("n_observed", 0) or 0) >= 2],
        key=lambda x: (-float(x["conservation"]), float(x["entropy"]), int(x["position"])),
    )
    return {
        "ok": True,
        "n_sequences": int(nseq),
        "alignment_length": int(aln_len),
        "mean_conservation": float(mean_cons),
        "columns": cols,
        "top_positions": top[:50],
    }


def _uniprot_location_value(location, key):
    try:
        val = (location.get(key, {}) or {}).get("value", None)
        return int(val)
    except Exception:
        return None


def _uniprot_text_value(obj):
    if not isinstance(obj, dict):
        return ""
    try:
        return str(obj.get("value", "") or "").strip()
    except Exception:
        return ""


def _uniprot_protein_name(entry):
    desc = dict((entry or {}).get("proteinDescription", {}) or {})
    for block_key in ("recommendedName", "submissionNames"):
        block = desc.get(block_key, {})
        if isinstance(block, list):
            blocks = block
        else:
            blocks = [block]
        for block in blocks:
            if not isinstance(block, dict):
                continue
            name = _uniprot_text_value(dict(block.get("fullName", {}) or {}))
            if name:
                return name
    return ""


def _uniprot_organism_name(entry):
    org = dict((entry or {}).get("organism", {}) or {})
    return str(org.get("scientificName", "") or org.get("commonName", "") or "").strip()


def _uniprot_query_url(query, size=5):
    params = {
        "query": str(query or ""),
        "format": "json",
        "size": str(int(max(1, size))),
        "fields": UNIPROT_FEATURE_FIELDS,
    }
    return UNIPROT_SEARCH_URL + "?" + urllib.parse.urlencode(params)


def _uniprot_get_json(url, timeout=6.0, urlopen_fn=None):
    opener = urlopen_fn or urllib.request.urlopen
    req = urllib.request.Request(
        str(url),
        headers={
            "Accept": "application/json",
            "User-Agent": "Leonardo-ProtLigInteract/1.0",
        },
    )
    with opener(req, timeout=float(timeout)) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", "replace"))


def _uniprot_sequence_match_score(query_seq, entry):
    qseq = clean_sequence(query_seq)
    eseq = clean_sequence(((entry or {}).get("sequence", {}) or {}).get("value", ""))
    score = 0.0
    if "reviewed" in str((entry or {}).get("entryType", "") or "").lower():
        score += 20.0
    try:
        score += float((entry or {}).get("annotationScore", 0.0) or 0.0)
    except Exception:
        pass
    if qseq and eseq:
        if qseq == eseq:
            score += 1000.0
        elif qseq in eseq:
            score += 800.0 + float(len(qseq)) / float(max(1, len(eseq)))
        elif eseq in qseq:
            score += 700.0 + float(len(eseq)) / float(max(1, len(qseq)))
        else:
            score -= abs(float(len(qseq)) - float(len(eseq))) / float(max(1, max(len(qseq), len(eseq))))
    return float(score)


def _select_uniprot_entry(results, sequence=""):
    rows = [r for r in list(results or []) if isinstance(r, dict)]
    if not rows:
        return {}
    return dict(max(rows, key=lambda row: _uniprot_sequence_match_score(sequence, row)))


def _uniprot_query_offset(query_seq, entry_seq):
    qseq = clean_sequence(query_seq)
    eseq = clean_sequence(entry_seq)
    if not qseq or not eseq:
        return None
    idx = eseq.find(qseq)
    if idx >= 0:
        return int(idx)
    idx = qseq.find(eseq)
    if idx >= 0:
        return int(-idx)
    return None


def _uniprot_feature_to_row(feature, query_offset=None, query_length=0):
    if not isinstance(feature, dict):
        return {}
    ftype = str(feature.get("type", "") or "").strip()
    if ftype not in UNIPROT_FUNCTIONAL_FEATURE_TYPES:
        return {}
    loc = dict(feature.get("location", {}) or {})
    start = _uniprot_location_value(loc, "start")
    end = _uniprot_location_value(loc, "end")
    if start is None and end is None:
        return {}
    if start is None:
        start = int(end)
    if end is None:
        end = int(start)
    if int(end) < int(start):
        start, end = end, start
    desc = str(feature.get("description", "") or "").strip()
    ligand = dict(feature.get("ligand", {}) or {})
    ligand_name = str(ligand.get("name", "") or "").strip()
    if ligand_name and ligand_name not in desc:
        desc = f"{desc}; ligand={ligand_name}" if desc else f"ligand={ligand_name}"
    fid = str(feature.get("featureId", "") or "").strip()
    row = {
        "type": ftype,
        "start": int(start),
        "end": int(end),
        "description": desc,
        "feature_id": fid,
    }
    if query_offset is not None and int(query_length or 0) > 0:
        qs = int(start) - int(query_offset)
        qe = int(end) - int(query_offset)
        if qe >= 1 and qs <= int(query_length):
            row["query_start"] = int(max(1, qs))
            row["query_end"] = int(min(int(query_length), qe))
            row["overlaps_query"] = True
        else:
            row["overlaps_query"] = False
    return row


def _uniprot_features_from_entry(entry, sequence="", max_features=40):
    eseq = clean_sequence(((entry or {}).get("sequence", {}) or {}).get("value", ""))
    qseq = clean_sequence(sequence)
    offset = _uniprot_query_offset(qseq, eseq)
    rows = []
    for feature in list((entry or {}).get("features", []) or []):
        row = _uniprot_feature_to_row(
            feature,
            query_offset=offset,
            query_length=len(qseq),
        )
        if not row:
            continue
        # If the user pasted an observed fragment, keep annotations that overlap
        # that fragment.  For full-length/no-match cases keep all UniProt rows.
        if offset is not None and qseq and not bool(row.get("overlaps_query", False)):
            continue
        rows.append(row)
        if len(rows) >= int(max(1, max_features)):
            break
    return rows


def fetch_uniprot_feature_summary(
    sequence="",
    accession="",
    pdb_id="",
    chain_id="",
    timeout=6.0,
    max_results=5,
    max_features=40,
    urlopen_fn=None,
):
    """
    Fetch curated UniProtKB motif/domain/site annotations for the local report.

    The lookup prefers a user-provided UniProt accession.  Otherwise it uses a
    PDB cross-reference, then selects the best returned UniProt entry by sequence
    match so multi-chain PDB entries do not silently annotate the wrong protein.
    """
    seq = clean_sequence(sequence)
    acc = str(accession or "").strip().upper()
    pdb = str(pdb_id or "").strip().upper()
    chain = str(chain_id or "").strip()
    if acc:
        query = f"accession:{acc}"
        source = "accession"
    elif re.match(r"^[0-9A-Za-z]{4}$", pdb or ""):
        query = f"xref:pdb-{pdb}"
        source = "pdb"
    else:
        return {
            "ok": False,
            "error": "Provide a UniProt accession or inferable PDB ID for UniProt annotation.",
            "features": [],
        }
    url = _uniprot_query_url(query, size=int(max(1, max_results)))
    try:
        data = _uniprot_get_json(url, timeout=float(timeout), urlopen_fn=urlopen_fn)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"UniProt lookup failed: {exc}", "url": url, "features": []}
    results = list((data or {}).get("results", []) or [])
    if not results:
        return {"ok": False, "error": f"No UniProtKB entry found for {query}.", "url": url, "features": []}
    entry = _select_uniprot_entry(results, sequence=seq)
    entry_seq = clean_sequence((entry.get("sequence", {}) or {}).get("value", ""))
    features = _uniprot_features_from_entry(entry, sequence=seq, max_features=int(max(1, max_features)))
    accession_out = str(entry.get("primaryAccession", "") or "").strip()
    return {
        "ok": True,
        "source": source,
        "query": query,
        "url": url,
        "accession": accession_out,
        "entry_id": str(entry.get("uniProtkbId", "") or "").strip(),
        "protein_name": _uniprot_protein_name(entry),
        "organism": _uniprot_organism_name(entry),
        "chain": chain,
        "entry_sequence_length": int(len(entry_seq)),
        "query_sequence_length": int(len(seq)),
        "features": features,
        "feature_count": int(len(features)),
    }


def build_external_tool_urls(sequence):
    urls = dict(TOOL_URLS)
    return urls


def summarize_question_answers(sequence):
    seq = clean_sequence(sequence)
    if not seq:
        return []
    rep = build_simulation_readiness_report(seq)
    if not bool(rep.get("ok", False)):
        return []
    tm_n = int(len(rep.get("tm_segments", []) or []))
    dis_frac = float(rep.get("disorder_fraction", 0.0) or 0.0)
    core = dict(rep.get("suggested_structured_core", {}) or {})
    motifs = list(rep.get("motifs", []) or [])
    out = []
    for row in list(BIOSEQ_QUESTION_MAP):
        q = str(row.get("question", "") or "").strip()
        ans = ""
        ql = q.lower()
        if "soluble or membrane" in ql:
            ans = "Likely membrane-associated" if tm_n > 0 else "Likely soluble"
        elif "flexible" in ql:
            ans = f"Estimated disorder fraction: {dis_frac:.2f}"
        elif "which region should be simulated" in ql:
            ans = f"Suggested structured core: {int(core.get('start', 1))}-{int(core.get('end', len(seq)))}"
        elif "functional sites" in ql:
            ans = f"Simple motif scan found {len(motifs)} hit(s)."
        elif "mutation" in ql:
            ans = "Prioritize mutations at highly conserved/motif-domain residues."
        else:
            ans = "Use the recommended analysis tools listed."
        out.append({"question": q, "analysis": str(row.get("analysis", "") or ""), "answer_hint": ans})
    return out


__all__ = [
    "BIOSEQ_ANALYSIS_CATALOG",
    "BIOSEQ_QUESTION_MAP",
    "clean_sequence",
    "sequence_text_residue_offsets",
    "parse_fasta",
    "sequence_basic_metrics",
    "predict_tm_segments",
    "predict_disorder_segments",
    "predict_signal_peptide",
    "find_simple_motifs",
    "build_motif_summary_lines",
    "build_simulation_readiness_report",
    "compute_msa_conservation",
    "fetch_uniprot_feature_summary",
    "build_external_tool_urls",
    "summarize_question_answers",
]
