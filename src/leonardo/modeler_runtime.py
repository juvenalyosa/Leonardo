import csv
import datetime
import glob
import os
import re
import urllib.request
from math import isfinite

import numpy as np

from .mutator import MutationError, parse_mutation_label, position_key


def _modeler_float_or_nan(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def build_modeler_run_root(workspace_dir, flow_label="protein_modeling", now_fn=None, exists_fn=None):
    root = str(workspace_dir or "").strip()
    flow = str(flow_label or "protein_modeling").strip() or "protein_modeling"
    if not root:
        return ""
    exists = exists_fn if callable(exists_fn) else os.path.exists
    now = now_fn if callable(now_fn) else datetime.datetime.now

    ts = ""
    try:
        ts = str(now().strftime("run_%Y%m%d_%H%M%S_%f") or "").strip()
    except Exception:
        ts = ""
    if not ts:
        ts = "run_modeler"
    cand = os.path.join(root, flow, ts)
    if not bool(exists(cand)):
        return str(cand)
    idx = 2
    while True:
        alt = f"{cand}_{int(idx)}"
        if not bool(exists(alt)):
            return str(alt)
        idx += 1


def _resolve_modeler_path(path, run_root=""):
    txt = str(path or "").strip()
    if not txt:
        return ""
    if os.path.isabs(txt):
        return str(txt)
    root = str(run_root or "").strip()
    if not root:
        return str(txt)
    return str(os.path.join(root, txt))


def normalize_modeler_build_output(build_output, run_root="", sequence_name="modeljuv"):
    root = str(run_root or "").strip()
    seq_name = str(sequence_name or "modeljuv").strip() or "modeljuv"
    if isinstance(build_output, dict):
        payload = dict(build_output or {})
        best_model_path = _resolve_modeler_path(payload.get("best_model_path", ""), root)
        generated_model_paths = [
            _resolve_modeler_path(p, root)
            for p in list(payload.get("generated_model_paths", []) or [])
            if str(p or "").strip()
        ]
        score_records = []
        for rec in list(payload.get("score_records", []) or []):
            if not isinstance(rec, dict):
                continue
            rr = dict(rec or {})
            rr["path"] = _resolve_modeler_path(rr.get("path", ""), root)
            try:
                rr["dope_score"] = float(rr.get("dope_score", float("nan")))
            except Exception:
                rr["dope_score"] = float("nan")
            score_records.append(rr)
        selected_cluster_records = []
        for rec in list(payload.get("selected_cluster_records", []) or []):
            if not isinstance(rec, dict):
                continue
            rr = dict(rec or {})
            rr["source_path"] = _resolve_modeler_path(rr.get("source_path", ""), root)
            rr["cluster_path"] = _resolve_modeler_path(rr.get("cluster_path", ""), root)
            try:
                rr["dope_score"] = float(rr.get("dope_score", float("nan")))
            except Exception:
                rr["dope_score"] = float("nan")
            selected_cluster_records.append(rr)
        return {
            "best_model_path": str(best_model_path or ""),
            "generated_model_paths": [str(p) for p in list(generated_model_paths or []) if str(p or "").strip()],
            "score_records": list(score_records),
            "selected_cluster_records": list(selected_cluster_records),
            "requested_model_count": int(payload.get("requested_model_count", 0) or 0),
            "generated_model_count": int(payload.get("generated_model_count", len(generated_model_paths)) or len(generated_model_paths)),
            "evaluated_model_count": int(payload.get("evaluated_model_count", len(score_records)) or len(score_records)),
            "median_dope_score": payload.get("median_dope_score", None),
            "elapsed_time_sec": _modeler_float_or_nan(payload.get("elapsed_time_sec", float("nan"))),
            "final_pdb_dir": _resolve_modeler_path(payload.get("final_pdb_dir", ""), root),
        }

    best_model_path = _resolve_modeler_path(build_output, root)
    inferred_model_paths = []
    if root:
        patt = os.path.join(root, "pdbs", f"{seq_name}.B9999*.pdb")
        inferred_model_paths = [str(os.path.abspath(p)) for p in list(sorted(glob.glob(patt)) or [])]
    return {
        "best_model_path": str(best_model_path or ""),
        "generated_model_paths": list(inferred_model_paths),
        "score_records": [],
        "selected_cluster_records": [],
        "requested_model_count": 0,
        "generated_model_count": int(len(inferred_model_paths)),
        "evaluated_model_count": 0,
        "median_dope_score": None,
        "elapsed_time_sec": float("nan"),
        "final_pdb_dir": str(os.path.join(root, "pdbs")) if root else "",
    }


def validate_modeler_model_count(model_paths, requested_count):
    paths = []
    seen = set()
    for path in list(model_paths or []):
        txt = str(path or "").strip()
        if (not txt) or (txt in seen):
            continue
        seen.add(txt)
        paths.append(txt)
    try:
        requested = int(requested_count or 0)
    except Exception:
        requested = 0
    actual = int(len(paths))
    ok = bool(requested <= 0 or actual == requested)
    error = ""
    if not ok:
        error = (
            f"MODELLER generated {actual} original model(s), but {requested} were requested. "
            "The build is incomplete; inspect the run directory/logs before using the result."
        )
    return {
        "ok": bool(ok),
        "requested_count": int(requested),
        "actual_count": int(actual),
        "model_paths": list(paths),
        "error": str(error),
    }


def parse_modeler_template_code(template_code):
    raw = str(template_code or "").strip().replace(" ", "")
    m = re.match(r"^([0-9A-Za-z]{4})[:_\-]?([A-Za-z0-9])$", raw)
    if not m:
        return {
            "template_code": "",
            "pdb_id": "",
            "chain": "",
        }
    code = f"{m.group(1).lower()}{m.group(2)}"
    return {
        "template_code": str(code),
        "pdb_id": str(m.group(1)).lower(),
        "chain": str(m.group(2)).strip(),
    }


def _canonical_modeler_template_from_available(pdb_id, chain_id, available_template_codes=None):
    """
    Resolve a PDB+chain token to the exact code already known by the UI/search.

    Chain identifiers can be case-sensitive in mmCIF, so the helper prefers an
    exact chain match and only falls back to case-insensitive matching when that
    produces one unambiguous available template.
    """
    pdb_txt = str(pdb_id or "").strip().lower()
    chain_txt = str(chain_id or "").strip()
    if not pdb_txt or not chain_txt:
        return ""
    exact = []
    folded = []
    seen = set()
    for code in list(available_template_codes or []):
        info = dict(parse_modeler_template_code(code) or {})
        cand = str(info.get("template_code", "") or "").strip()
        if (not cand) or (cand in seen):
            continue
        seen.add(cand)
        if str(info.get("pdb_id", "") or "").lower() != pdb_txt:
            continue
        avail_chain = str(info.get("chain", "") or "").strip()
        if avail_chain == chain_txt:
            exact.append(cand)
        if avail_chain.lower() == chain_txt.lower():
            folded.append(cand)
    if len(exact) == 1:
        return str(exact[0])
    if len(folded) == 1:
        return str(folded[0])
    return ""


def normalize_modeler_template_input_token(token, available_template_codes=None):
    raw = str(token or "").strip()
    if not raw:
        return ""
    raw = raw.replace(" ", "")
    m_full = re.match(r"^([0-9A-Za-z]{4})[:_\-]?([A-Za-z0-9])$", raw)
    if m_full:
        pdb_id = str(m_full.group(1)).lower()
        chain_id = str(m_full.group(2))
        resolved = _canonical_modeler_template_from_available(
            pdb_id,
            chain_id,
            available_template_codes=available_template_codes,
        )
        if resolved:
            return str(resolved)
        return f"{pdb_id}{chain_id}"
    m_pdb = re.match(r"^([0-9A-Za-z]{4})$", raw)
    if not m_pdb:
        return ""
    pdb_id = str(m_pdb.group(1) or "").lower()
    matches = []
    seen = set()
    for code in list(available_template_codes or []):
        info = dict(parse_modeler_template_code(code) or {})
        cand = str(info.get("template_code", "") or "").strip()
        if (not cand) or cand in seen:
            continue
        seen.add(cand)
        if str(info.get("pdb_id", "") or "").lower() == pdb_id:
            matches.append(cand)
    if len(matches) == 1:
        return str(matches[0])
    return ""


def resolve_modeler_template_selection(
    table_template_codes=None,
    manual_text="",
    available_template_codes=None,
    fallback_template_codes=None,
):
    available = []
    seen_available = set()
    for code in list(available_template_codes or []):
        norm = normalize_modeler_template_input_token(code, None)
        if (not norm) or (norm in seen_available):
            continue
        seen_available.add(norm)
        available.append(norm)

    manual_tokens = [tok for tok in re.split(r"[\s,;]+", str(manual_text or "")) if str(tok or "").strip()]
    payload = {
        "selected": [],
        "source": "table",
        "has_manual_input": bool(manual_tokens),
        "manual_resolved": [],
        "manual_unresolved": [],
    }

    if manual_tokens:
        payload["source"] = "manual"
        seen = set()
        for token in manual_tokens:
            norm = normalize_modeler_template_input_token(token, available)
            if not norm:
                payload["manual_unresolved"].append(str(token))
                continue
            if norm in seen:
                continue
            seen.add(norm)
            payload["manual_resolved"].append(norm)
        payload["selected"] = list(payload["manual_resolved"])
        return payload

    seen = set()
    for code in list(table_template_codes or []):
        norm = normalize_modeler_template_input_token(code, available)
        if (not norm) or (norm in seen):
            continue
        seen.add(norm)
        payload["selected"].append(norm)
    if payload["selected"]:
        return payload

    payload["source"] = "fallback"
    for code in list(fallback_template_codes or []):
        norm = normalize_modeler_template_input_token(code, available)
        if (not norm) or (norm in seen):
            continue
        seen.add(norm)
        payload["selected"].append(norm)
    return payload


def select_modeler_search_result_variant(variant_results, requested_rows=30):
    try:
        requested = max(1, int(requested_rows or 0))
    except Exception:
        requested = 30
    best = None
    best_rank = (-1, -1, -1, -1, -1)
    for rec in list(variant_results or []):
        if not isinstance(rec, dict):
            continue
        tag = str(rec.get("tag", "") or "")
        resp = dict(rec.get("response", {}) or {})
        rset = list(rec.get("result_set", resp.get("result_set", []) or []) or [])
        count = int(len(rset))
        if count <= 0:
            continue
        paged = 1 if ("paginate" in tag or "pager" in tag) else 0
        sort = 1 if "sort" in tag else 0
        nonminimal = 0 if "|minimal" in tag else 1
        coverage = int(min(count, requested))
        rank = (coverage, count, paged, sort, nonminimal)
        if rank > best_rank:
            best_rank = rank
            best = {
                "tag": str(tag),
                "response": dict(resp),
                "result_set": list(rset),
                "count": int(count),
            }
    return dict(best or {})


def filter_modeler_template_components(components, template_code):
    info = dict(parse_modeler_template_code(template_code) or {})
    chain = str(info.get("chain", "") or "").strip()
    out = []
    seen = set()
    for comp in list(components or []):
        if not isinstance(comp, dict):
            continue
        if chain and str(comp.get("chain", "") or "").strip() != chain:
            continue
        key = (
            str(comp.get("chain", "") or "").strip(),
            str(comp.get("resi", "") or "").strip(),
            str(comp.get("resn", "") or "").strip().upper(),
            str(comp.get("segi", "") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(comp or {}))
    return list(out)


def resolve_modeler_template_transfer_flags(
    include_nonprotein=False,
    receptor_obj="",
    hetatm_source_code="",
    selected_component_counts=None,
    sequence_mode_template_fallback=False,
):
    counts_in = dict(selected_component_counts or {}) if isinstance(selected_component_counts, dict) else {}
    counts = {
        "cofactor": int(counts_in.get("cofactor", 0) or 0),
        "ion": int(counts_in.get("ion", 0) or 0),
        "water": int(counts_in.get("water", 0) or 0),
    }
    include_nonprotein_flag = bool(include_nonprotein)
    receptor = str(receptor_obj or "").strip()
    source = str(hetatm_source_code or "").strip()
    out = {
        "strategy": "disabled",
        "direct_template_hetatm": False,
        "direct_template_water": False,
        "counts": dict(counts),
    }
    if source:
        out["strategy"] = "template_source_post_merge" if include_nonprotein_flag else "template_source_disabled"
        return out
    if not include_nonprotein_flag:
        return out
    if receptor:
        out["strategy"] = "receptor_post_merge"
        return out
    if bool(sequence_mode_template_fallback):
        out["strategy"] = "modeller_direct_all"
        out["direct_template_hetatm"] = True
        out["direct_template_water"] = True
        return out
    out["strategy"] = "modeller_direct_selected"
    out["direct_template_hetatm"] = bool(counts["cofactor"] or counts["ion"])
    out["direct_template_water"] = bool(counts["water"])
    return out


def should_modeler_autoresolve_mutation_start(mode):
    # Mutation numbering must match the user-provided/receptor numbering exactly.
    # Silent start-position rewrites are unsafe because they can move a requested
    # mutation onto a different residue without the user noticing.
    return False


def _expand_modeler_mutation_position_range(start_pos, end_pos, max_span=512):
    try:
        start_i = int(start_pos)
        end_i = int(end_pos)
    except Exception:
        return []
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    try:
        max_span_i = int(max_span)
    except Exception:
        max_span_i = 512
    if max_span_i <= 0:
        max_span_i = 512
    span_len = int(end_i - start_i + 1)
    if span_len <= 0:
        return []
    if span_len > max_span_i:
        if start_i == end_i:
            return [int(start_i)]
        return [int(start_i), int(end_i)]
    return [int(v) for v in range(int(start_i), int(end_i) + 1)]


def _modeler_position_value(key):
    """``"184"``/``"-3"`` -> int, ``"184A"`` (insertion code) -> str, anything else -> None."""
    txt = str(key if key is not None else "").strip().upper()
    if re.fullmatch(r"-?\d+", txt):
        return int(txt)
    m = re.fullmatch(r"(-?\d+)([A-Z])", txt)
    return position_key(m.group(1), m.group(2)) if m else None


def extract_modeler_mutation_target_positions(mutation_labels, max_span=512):
    """Residue positions named by mutation labels, in label order: ints for plain numbers
    (negative allowed), ``"184A"``-style strings for insertion codes. A range with an
    insertion code at either end contributes only its end points."""
    out = []
    seen = set()
    for mut in list(mutation_labels or []):
        try:
            p = parse_mutation_label(mut)
        except MutationError:
            continue
        start = _modeler_position_value(p["start"])
        end = _modeler_position_value(p["end"])
        if p["kind"] in ("sub", "stop"):
            positions = [start]
        elif p["kind"] == "ins":
            positions = [start, end]
        elif isinstance(start, int) and isinstance(end, int):
            positions = _expand_modeler_mutation_position_range(start, end, max_span=max_span)
        else:
            positions = [start, end]
        for pos in positions:
            if pos is None or pos in seen:
                continue
            seen.add(pos)
            out.append(pos)
    return list(out)


def modeler_mutation_target_labels(mutation_labels, residue_labels):
    """Residue labels of a chain (``["183", "184", "184A", ...]``) touched by the mutations.

    Ranges are resolved on the chain order, so ``183_185del`` includes ``184A``.
    Labels that do not occur in the chain are skipped.
    """
    labels = [str(x).strip().upper() for x in list(residue_labels or [])]
    index = {}
    for i, lbl in enumerate(labels):
        key = _modeler_position_value(lbl)
        if key is not None:
            index.setdefault(str(key), i)
    out = []
    for mut in list(mutation_labels or []):
        try:
            p = parse_mutation_label(mut)
        except MutationError:
            continue
        i0 = index.get(p["start"])
        i1 = index.get(p["end"])
        if p["kind"] in ("sub", "stop"):
            idx = [i0]
        elif p["kind"] == "ins" or i0 is None or i1 is None or i1 < i0:
            idx = [i0, i1]
        else:
            idx = list(range(i0, i1 + 1))
        for i in idx:
            if i is not None and labels[i] not in out:
                out.append(labels[i])
    return out


def modeler_sequence_window_covers_positions(start_pos, sequence, positions):
    seq = str(sequence or "").strip().upper()
    pos_list = []
    seen = set()
    for pos_i in list(positions or []):
        pos = _modeler_position_value(pos_i)
        if pos is None or pos in seen:
            continue
        seen.add(pos)
        pos_list.append(pos)
    if not pos_list:
        return True
    if not seq:
        return False
    try:
        start_i = int(start_pos)
    except Exception:
        start_i = 1
    end_i = int(start_i + len(seq) - 1)
    # an insertion-code position ("184A") only exists in structure numbering
    return all(isinstance(pos, int) and start_i <= pos <= end_i for pos in pos_list)


def resolve_modeler_mutation_input_sequence(
    prepared_sequence,
    prepared_start,
    explicit_sequence="",
    explicit_start=1,
    mutation_positions=None,
    mode="gap",
    prepared_labels=None,
):
    prepared_seq = str(prepared_sequence or "").strip().upper()
    explicit_seq = str(explicit_sequence or "").strip().upper()
    try:
        prepared_start_i = int(prepared_start)
    except Exception:
        prepared_start_i = 1
    try:
        explicit_start_i = int(explicit_start)
    except Exception:
        explicit_start_i = 1
    positions = []
    seen = set()
    for pos_i in list(mutation_positions or []):
        pos = _modeler_position_value(pos_i)
        if pos is None or pos in seen:
            continue
        seen.add(pos)
        positions.append(pos)
    labels = list(prepared_labels or [])
    if labels and len(labels) == len(prepared_seq):
        # structure numbering (gaps, jumps, insertion codes): a position fits when it is labelled
        numbered = {_modeler_position_value(x) for x in labels if x is not None}
        prepared_fits = all(p in numbered for p in positions)
    else:
        prepared_fits = bool(
            modeler_sequence_window_covers_positions(
                prepared_start_i,
                prepared_seq,
                positions,
            )
        )
    explicit_fits = bool(
        modeler_sequence_window_covers_positions(
            explicit_start_i,
            explicit_seq,
            positions,
        )
    )
    out = {
        "sequence": str(prepared_seq),
        "start_pos": int(prepared_start_i),
        "used_explicit_sequence": False,
        "strategy": "prepared",
        "note": "",
        "prepared_fits": bool(prepared_fits),
        "explicit_fits": bool(explicit_fits),
    }
    if not explicit_seq:
        return out
    mode_txt = str(mode or "").strip().lower()
    if (mode_txt == "sequence") or (not prepared_seq):
        out["sequence"] = str(explicit_seq)
        out["start_pos"] = int(explicit_start_i)
        out["used_explicit_sequence"] = True
        out["strategy"] = "explicit_sequence_mode" if mode_txt == "sequence" else "explicit_fallback"
        if mode_txt == "sequence":
            out["note"] = "Using explicit sequence/start input for sequence-based modeling."
        else:
            out["note"] = "Using explicit sequence/start input because no receptor-derived base sequence was prepared."
        return out
    if (
        positions
        and (not prepared_fits)
        and bool(explicit_fits)
        and (
            str(explicit_seq) != str(prepared_seq)
            or int(explicit_start_i) != int(prepared_start_i)
        )
    ):
        out["sequence"] = str(explicit_seq)
        out["start_pos"] = int(explicit_start_i)
        out["used_explicit_sequence"] = True
        out["strategy"] = "explicit_numbering_override"
        out["note"] = (
            "Requested mutation numbering is outside the current gap-mode chain window; "
            "using the explicit sequence/start input instead."
        )
    return out


def _run_modeler_sequence_aligner(seq_a, seq_b, aligner_fn=None):
    fn = aligner_fn
    if not callable(fn):
        try:
            from .mutator import needleman_wunsch_affine as fn
        except Exception:
            fn = None
    left = str(seq_a or "").strip().upper()
    right = str(seq_b or "").strip().upper()
    if not callable(fn):
        return left, right
    try:
        out = fn(left, right)
    except Exception:
        return left, right
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        return str(out[0] or ""), str(out[1] or "")
    return left, right


def build_modeler_mutation_highlight_indices(original_sequence, mutated_sequence, aligner_fn=None):
    seq_orig = str(original_sequence or "").strip().upper()
    seq_mut = str(mutated_sequence or "").strip().upper()
    if (not seq_orig) or (not seq_mut):
        return []
    aligned_orig, aligned_mut = _run_modeler_sequence_aligner(seq_orig, seq_mut, aligner_fn=aligner_fn)
    if (not aligned_orig) or (not aligned_mut) or (len(aligned_orig) != len(aligned_mut)):
        return []
    out = []
    seen = set()
    idx_mut = 0
    mut_len = int(len(seq_mut))
    for aa_orig, aa_mut in zip(aligned_orig, aligned_mut):
        orig_has = str(aa_orig) != "-"
        mut_has = str(aa_mut) != "-"
        cur_mut_idx = int(idx_mut) if mut_has else None
        if (not (orig_has and mut_has and (str(aa_orig) == str(aa_mut)))):
            if mut_has and cur_mut_idx is not None and 0 <= int(cur_mut_idx) < mut_len:
                if int(cur_mut_idx) not in seen:
                    seen.add(int(cur_mut_idx))
                    out.append(int(cur_mut_idx))
            elif orig_has and (not mut_has):
                anchor_prev = int(idx_mut) - 1
                anchor_next = int(idx_mut)
                for anchor in (anchor_prev, anchor_next):
                    if 0 <= int(anchor) < mut_len and int(anchor) not in seen:
                        seen.add(int(anchor))
                        out.append(int(anchor))
        if mut_has:
            idx_mut += 1
    return list(out)


def map_modeler_highlight_indices_to_positions(
    highlight_indices,
    mutated_sequence,
    observed_sequence,
    observed_positions,
    aligner_fn=None,
):
    idx_targets = []
    seen_idx = set()
    for idx in list(highlight_indices or []):
        try:
            idx_i = int(idx)
        except Exception:
            continue
        if idx_i in seen_idx:
            continue
        seen_idx.add(idx_i)
        idx_targets.append(int(idx_i))
    seq_mut = str(mutated_sequence or "").strip().upper()
    seq_obs = str(observed_sequence or "").strip().upper()
    pos_list = list(observed_positions or [])
    if (not idx_targets) or (not seq_mut) or (not seq_obs) or (len(pos_list) != len(seq_obs)):
        return []
    target_set = set(int(i) for i in list(idx_targets or []))
    aligned_mut, aligned_obs = _run_modeler_sequence_aligner(seq_mut, seq_obs, aligner_fn=aligner_fn)
    if (not aligned_mut) or (not aligned_obs) or (len(aligned_mut) != len(aligned_obs)):
        return []
    out = []
    seen_pos = set()
    idx_mut = 0
    idx_obs = 0
    for aa_mut, aa_obs in zip(aligned_mut, aligned_obs):
        mut_has = str(aa_mut) != "-"
        obs_has = str(aa_obs) != "-"
        cur_mut_idx = int(idx_mut) if mut_has else None
        cur_obs_idx = int(idx_obs) if obs_has else None
        if (
            mut_has
            and obs_has
            and cur_mut_idx in target_set
            and 0 <= int(cur_obs_idx) < len(pos_list)
        ):
            pos_i = _modeler_position_value(pos_list[int(cur_obs_idx)])
            if pos_i is not None and pos_i not in seen_pos:
                seen_pos.add(pos_i)
                out.append(pos_i)
        if mut_has:
            idx_mut += 1
        if obs_has:
            idx_obs += 1
    return list(out)


def build_modeler_energy_distribution_payload(score_records, bins=0):
    clean = []
    for rec in list(score_records or []):
        if not isinstance(rec, dict):
            continue
        try:
            dope = float(rec.get("dope_score", float("nan")))
        except Exception:
            dope = float("nan")
        if not isfinite(dope):
            continue
        rr = dict(rec or {})
        rr["dope_score"] = float(dope)
        clean.append(rr)
    if not clean:
        return {
            "ok": False,
            "error": "No finite DOPE scores were available for plotting.",
        }

    energies = np.asarray([float(rec["dope_score"]) for rec in list(clean or [])], dtype=float).reshape(-1)
    order = np.argsort(energies)
    energies = energies[order]
    clean = [dict(clean[int(i)]) for i in list(order)]
    best_energy = float(np.min(energies))
    delta = energies - float(best_energy)
    n = int(delta.size)

    try:
        requested_bins = int(bins or 0)
    except Exception:
        requested_bins = 0
    if requested_bins <= 0:
        requested_bins = int(max(6, min(24, round(np.sqrt(max(1, n))) + 2)))

    spread = float(np.max(delta) - np.min(delta)) if n > 0 else 0.0
    if (not isfinite(spread)) or spread <= 1.0e-12:
        edges = np.asarray([0.0, 1.0], dtype=float)
        hist_density = np.asarray([1.0], dtype=float)
    else:
        hist_density, edges = np.histogram(delta, bins=int(requested_bins), density=True)
        hist_density = np.asarray(hist_density, dtype=float)
        edges = np.asarray(edges, dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_delta = float(np.mean(delta)) if n > 0 else float("nan")
    std_delta = float(np.std(delta, ddof=0)) if n > 0 else float("nan")

    return {
        "ok": True,
        "error": "",
        "score_records": list(clean),
        "energies": np.asarray(energies, dtype=float),
        "delta_energies": np.asarray(delta, dtype=float),
        "hist_edges": np.asarray(edges, dtype=float),
        "hist_density": np.asarray(hist_density, dtype=float),
        "hist_centers": np.asarray(centers, dtype=float),
        "best_energy": float(best_energy),
        "mean_delta": float(mean_delta),
        "std_delta": float(std_delta),
        "n_models": int(n),
    }


def fit_modeler_maxwell_boltzmann_overlay(delta_energies):
    arr = np.asarray(delta_energies, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {
            "ok": False,
            "error": "At least two finite energies are required for a Maxwell-Boltzmann fit.",
        }
    arr = arr - float(np.min(arr))
    mean_delta = float(np.mean(arr))
    if (not isfinite(mean_delta)) or mean_delta <= 1.0e-12:
        return {
            "ok": False,
            "error": "Maxwell-Boltzmann fit requires non-zero spread in shifted energies.",
        }

    beta = 1.5 / float(mean_delta)
    xmax = float(max(np.max(arr) * 1.05, 1.0e-6))
    x = np.linspace(0.0, xmax, 256, dtype=float)
    y = np.asarray((2.0 / np.sqrt(np.pi)) * (beta ** 1.5) * np.sqrt(x) * np.exp((-1.0 * beta) * x), dtype=float)
    return {
        "ok": True,
        "error": "",
        "beta": float(beta),
        "fit_x": np.asarray(x, dtype=float),
        "fit_y": np.asarray(y, dtype=float),
        "fit_label": "Maxwell-Boltzmann fit",
        "mean_delta": float(mean_delta),
    }


def build_modeler_boltzmann_plot_payload(score_records, bins=0):
    # Backward-compatible alias kept for existing callers/tests.
    return dict(build_modeler_energy_distribution_payload(score_records, bins=bins) or {})


def write_modeler_score_records_csv(csv_path, score_records):
    out_csv = str(csv_path or "").strip()
    rows = [dict(r or {}) for r in list(score_records or []) if isinstance(r, dict)]
    if (not out_csv) or (not rows):
        return ""
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        with open(out_csv, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["rank", "model_path", "dope_score"])
            wr.writeheader()
            for idx, rec in enumerate(list(rows or []), start=1):
                wr.writerow(
                    {
                        "rank": int(idx),
                        "model_path": str(rec.get("path", "") or ""),
                        "dope_score": f"{float(rec.get('dope_score', float('nan'))):.6f}",
                    }
                )
    except Exception:
        return ""
    return str(out_csv)


def fetch_hetatm_source_template_pdb(template_code, run_root):
    """
    Ensure a local coordinate file for *template_code* (e.g. '3ptbA') is present on disk,
    reusing the copy staged/downloaded by MODELLER when available.

    Returns the local path to a PDB or mmCIF file, or an empty string on failure.
    Does not require MODELLER or PyMOL — pure stdlib only.
    """
    code = str(template_code or "").strip()
    if len(code) < 5:
        return ""
    pdb_id = code[:-1].lower()
    job_dir = os.path.join(str(run_root or ""), "modeling_job")
    tpl_pdb = os.path.join(job_dir, f"{pdb_id}.pdb")
    tpl_cif = os.path.join(job_dir, f"{pdb_id}.cif")
    if os.path.isfile(tpl_pdb):
        return str(tpl_pdb)
    if os.path.isfile(tpl_cif):
        return str(tpl_cif)
    try:
        os.makedirs(job_dir, exist_ok=True)
        for url, out_path in (
            (f"https://files.rcsb.org/download/{pdb_id}.pdb", tpl_pdb),
            (f"https://files.rcsb.org/download/{pdb_id}.cif", tpl_cif),
        ):
            try:
                urllib.request.urlretrieve(url, out_path)
                if os.path.isfile(out_path):
                    return str(out_path)
            except Exception:
                continue
        return ""
    except Exception:
        return ""


def read_hetatm_lines_from_pdb(pdb_path):
    """
    Read all HETATM and associated CONECT lines from *pdb_path*.

    Returns a list of raw PDB line strings (newlines stripped).
    Pure file-reading — no PyMOL or MODELLER dependency.
    """
    lines = []
    try:
        with open(str(pdb_path or ""), "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.startswith("HETATM") or line.startswith("CONECT"):
                    lines.append(line)
    except Exception:
        return []
    return lines


# Peptide C(i-1)-N(i) is 1.33 A; anything beyond this is a chain break (missing residues).
MODELER_PEPTIDE_BOND_MAX_ANG = 2.0


def modeler_residue_label(resseq, icode=""):
    """PDB residue label including the insertion code, e.g. ``184`` or ``184A``."""
    return f"{int(resseq)}{str(icode or '').strip()}"


def modeler_peptide_bonded(prev_c_xyz, next_n_xyz, max_ang=MODELER_PEPTIDE_BOND_MAX_ANG):
    """True/False when both backbone atoms are known, None otherwise (e.g. CA-only models)."""
    if prev_c_xyz is None or next_n_xyz is None:
        return None
    try:
        d = float(np.linalg.norm(np.asarray(next_n_xyz, dtype=float) - np.asarray(prev_c_xyz, dtype=float)))
    except Exception:
        return None
    if not isfinite(d):
        return None
    return bool(d <= float(max_ang))


def build_modeler_chain_entry(chain_key, residues):
    """
    Summarize one protein chain for modeling from its observed residues in chain order.

    ``residues``: dicts with ``resseq`` (int), ``icode`` (str), ``resn``, ``aa1`` and
    ``bonded_to_prev`` (True/False/None). Residues with insertion codes (``184A``) are kept.
    A numbering jump between peptide-bonded residues (chymotrypsin numbering: 34 -> 37)
    is not a gap; only real chain breaks count as missing residues. When the bond cannot
    be checked (``None``) the numbering difference decides, as before.
    """
    rows = []
    seen = set()
    for rec in list(residues or []):
        try:
            resseq = int(rec.get("resseq"))
        except Exception:
            continue
        icode = str(rec.get("icode", "") or "").strip()
        key = (resseq, icode)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "resseq": resseq,
            "icode": icode,
            "resn": str(rec.get("resn", "") or "").strip().upper(),
            "aa1": str(rec.get("aa1", "X") or "X").strip().upper()[:1] or "X",
            "bonded_to_prev": rec.get("bonded_to_prev", None),
        })
    if not rows:
        return None

    gaps = []
    breaks = []
    jumps = []
    for prev, cur in zip(rows, rows[1:]):
        diff = int(cur["resseq"]) - int(prev["resseq"])
        bonded = cur["bonded_to_prev"]
        prev_lbl = modeler_residue_label(prev["resseq"], prev["icode"])
        cur_lbl = modeler_residue_label(cur["resseq"], cur["icode"])
        if bonded is True:
            if diff > 1:
                jumps.append((prev_lbl, cur_lbl))
            continue
        if bonded is None and diff <= 1:
            continue
        if diff > 1:
            gaps.append((int(prev["resseq"]) + 1, int(cur["resseq"]) - 1))
            breaks.append({"after": prev_lbl, "before": cur_lbl, "missing": int(diff - 1)})
        else:
            breaks.append({"after": prev_lbl, "before": cur_lbl, "missing": None})

    positions = [int(r["resseq"]) for r in rows]
    # keyed by residue label so 184 and 184A stay distinct
    pos_to_resn = {}
    pos_to_aa1 = {}
    for r in rows:
        lbl = modeler_residue_label(r["resseq"], r["icode"])
        pos_to_resn.setdefault(lbl, r["resn"])
        pos_to_aa1.setdefault(lbl, r["aa1"])
    return {
        "chain": chain_key,
        "start": int(rows[0]["resseq"]),
        "end": int(rows[-1]["resseq"]),
        "positions": positions,
        "residue_labels": [modeler_residue_label(r["resseq"], r["icode"]) for r in rows],
        "pos_to_resn": pos_to_resn,
        "pos_to_aa1": pos_to_aa1,
        "observed_seq": "".join(r["aa1"] for r in rows),
        "observed_len": int(len(rows)),
        "internal_gaps": gaps,
        "chain_breaks": breaks,
        "numbering_jumps": jumps,
        "missing_total": int(sum(b - a + 1 for a, b in gaps)),
    }


def _modeler_label_number(label):
    """Residue number of a PDB label (``"184A"`` -> 184, ``"-3"`` -> -3), None if not numbered."""
    m = re.fullmatch(r"(-?\d+)[A-Za-z]?", str(label or "").strip())
    return int(m.group(1)) if m else None


def modeler_gap_fill_sequence(
    chain_entry, full_sequence, aligner_fn=None, min_identity=0.9, min_coverage=0.9, fill_termini=False
):
    """
    Rebuild the chain between its first and last observed residue, filling missing
    residues from a full/reference sequence. With ``fill_termini`` the reference
    residues before the first and after the last observed residue are added too
    (N/C-terminal extension); they are numbered outward from the observed ends.

    The observed residues are placed on the reference by sequence alignment, not by
    residue number, so numbering jumps and insertion codes cannot shift residues.
    Returns the sequence plus one label per residue: the PDB label for observed
    residues, the interpolated number for filled residues when the numbering of the
    gap agrees with the number filled, else None.
    """
    entry = dict(chain_entry or {})
    obs = str(entry.get("observed_seq", "") or "").strip().upper()
    labels = list(entry.get("residue_labels", []) or [])
    if len(labels) != len(obs):
        labels = [str(p) for p in list(entry.get("positions", []) or [])]
    full = re.sub(r"\s+", "", str(full_sequence or "")).upper()
    out = {"ok": False, "sequence": obs, "residue_labels": list(labels), "filled": 0,
           "filled_n_term": 0, "filled_c_term": 0,
           "identity": float("nan"), "coverage": float("nan"), "error": ""}
    if not obs or not full or len(labels) != len(obs):
        out["error"] = "Observed chain or reference sequence is empty."
        return out
    a_obs, a_full = _run_modeler_sequence_aligner(obs, full, aligner_fn=aligner_fn)
    if len(a_obs) != len(a_full):
        out["error"] = "Sequence alignment failed."
        return out

    cols = []
    i_obs = 0
    for c_obs, c_full in zip(a_obs, a_full):
        o_idx = i_obs if c_obs != "-" else None
        if c_obs != "-":
            i_obs += 1
        cols.append((o_idx, c_obs, c_full))
    paired = [(o, co, cf) for o, co, cf in cols if o is not None and cf != "-"]
    coverage = len(paired) / float(len(obs))
    identity = (sum(1 for _o, co, cf in paired if co == cf) / float(len(paired))) if paired else 0.0
    out["identity"], out["coverage"] = float(identity), float(coverage)
    if coverage < float(min_coverage) or identity < float(min_identity):
        out["error"] = (
            f"The reference sequence does not match the observed chain (identity {100 * identity:.0f}%, "
            f"coverage {100 * coverage:.0f}%); check that it is the same protein."
        )
        return out

    first = next(k for k, c in enumerate(cols) if c[0] is not None)
    last = max(k for k, c in enumerate(cols) if c[0] is not None)
    n_term = c_term = 0
    if fill_termini:
        n_term = sum(1 for c in cols[:first] if c[2] != "-")
        c_term = sum(1 for c in cols[last + 1:] if c[2] != "-")
        first, last = 0, len(cols) - 1
    seq_chars = []
    seq_labels = []
    for o_idx, c_obs, c_full in cols[first:last + 1]:
        if o_idx is not None:
            seq_chars.append(c_obs)
            seq_labels.append(labels[o_idx])
        elif c_full != "-":
            seq_chars.append(c_full)
            seq_labels.append(None)
    filled = sum(1 for lbl in seq_labels if lbl is None)

    k = 0
    while k < len(seq_labels):
        if seq_labels[k] is not None:
            k += 1
            continue
        run_end = k
        while run_end < len(seq_labels) and seq_labels[run_end] is None:
            run_end += 1
        prev_lbl = str(seq_labels[k - 1]) if k > 0 else ""
        next_lbl = str(seq_labels[run_end]) if run_end < len(seq_labels) else ""
        prev_num = _modeler_label_number(prev_lbl)
        next_num = _modeler_label_number(next_lbl)
        if not prev_lbl and next_num is not None:          # N-terminal extension
            for n, j in enumerate(range(run_end - 1, k - 1, -1), start=1):
                seq_labels[j] = str(next_num - n)
        elif not next_lbl and prev_num is not None:        # C-terminal extension
            for n, j in enumerate(range(k, run_end), start=1):
                seq_labels[j] = str(prev_num + n)
        elif re.fullmatch(r"-?\d+", prev_lbl) and re.fullmatch(r"-?\d+", next_lbl):
            if int(next_lbl) - int(prev_lbl) - 1 == run_end - k:
                for n, j in enumerate(range(k, run_end), start=1):
                    seq_labels[j] = str(int(prev_lbl) + n)
        k = run_end

    out.update({"ok": True, "sequence": "".join(seq_chars), "residue_labels": seq_labels, "filled": int(filled),
                "filled_n_term": int(n_term), "filled_c_term": int(c_term)})
    return out
